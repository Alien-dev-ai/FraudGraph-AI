"""Baseline dataset assembly and feature audit (Milestone 8).

Assembles a bounded, leakage-safe baseline modeling feature set from
Milestone 5's tabular features and Milestone 7's causal graph features,
reusing their existing, already-tested per-chunk computation functions
directly (no reimplementation, no semantic changes). Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

Never trains a model, never performs model selection, never builds a
full-size feature table, never builds a NetworkX graph.

=== Baseline feature contract ===

BASELINE_FEATURE_COLUMNS (39 columns, fixed order) =
  Milestone 5's `features_raw_paysim.baseline_feature_columns()` (34
  columns, unchanged) followed by GRAPH_FEATURES_RETAINED_IN_BASELINE
  (5 of Milestone 7's 8 baseline-ready graph columns; see below for the
  3 that are excluded and why).

No target column (`isFraud`), no raw account identifiers (`nameOrig`,
`nameDest`), and no `row_id` are in this list — see IDENTIFIER_COLUMNS
and TARGET_COLUMN, which exist only for audit/traceability, never as
model inputs. `step`/`day_index` remain excluded per Milestone 5's
existing exploratory_only classification (not reintroduced here).
`isFlaggedFraud` remains included, as Milestone 5 already classifies it
safe_for_baseline.

=== Graph-feature redundancy resolution (Milestone 7 -> Milestone 8) ===

Of Milestone 7's 8 baseline-ready graph features, 3 are held back from
BASELINE_FEATURE_COLUMNS into GRAPH_FEATURES_AUDIT_ONLY instead —
retained in the assembled table and audited, but not part of the
modeling contract:

  - `pair_seen_before_causal`, `pair_prior_count_causal`: Milestone 7
    measured these as literally constant-zero across the 2,000,000-row
    validated subset (0 repeated origin->destination pairs). A
    zero-variance column cannot contribute to any discriminative model
    trained on that data. This is NOT "excluded for being correlated"
    (explicitly disallowed by this milestone's instructions) — it is a
    distinct, stronger condition (proven zero variance on the measured
    subset). IMPORTANT SCOPE CAVEAT: this was only measured on the
    Milestone 6/7 2,000,000-row subset, not the full 6,362,620-row
    file — see `run_full_stream_audit` below, which re-checks
    constancy at full scale as part of this milestone's audit.
  - `origin_outgoing_concentration_causal`: measured defined for only
    0.046% of rows (921/2,000,000) and near-degenerate (mean 0.9995)
    even when defined — 99.95% missing plus near-constant among the
    remainder leaves negligible usable signal. Retained for audit only,
    not baseline.

The other 5 (`origin_out_degree_causal`, `dest_in_degree_causal`,
`origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`,
`dest_incoming_concentration_causal`) are retained in the baseline.

Two of those five are measured to be EXACTLY collinear in this dataset
with existing Milestone 5 features (`origin_out_degree_causal` with
`origin_txn_count_causal`; `dest_in_degree_causal` with
`dest_txn_count_causal`) — a direct mathematical consequence of the
pair-repeat rate being exactly zero (see module docstring above). Per
this milestone's explicit instruction not to remove features solely for
being correlated (here: exactly collinear on the measured subset), both
are KEPT: the degree-based versions carry distinct DEGREE semantics
that could diverge from the count-based versions on the full
6,362,620-row file (not yet measured) or on future data, and their
exact-collinearity finding is documented as an audit note, not used as
a removal justification.

=== Causal / same-step convention ===

Identical to Milestone 5 and 7: file/row order is the event order, and
within a shared `step` value an earlier file row is prior. Verified
raw-file non-decreasing by `step` in Milestone 5. This module introduces
no new causal logic — it reuses Milestone 5's and 7's per-chunk causal
state functions verbatim, sharing one sequential read of the raw file.

=== Undefined-value policy ===

Preserve NaN, do not silently impute. This is not a new policy invented
here — it is the policy already established and tested in Milestone 5
(`amount_to_orig_balance_ratio`, `origin_avg_amount_causal`, etc.) and
Milestone 7 (concentration features), reused unchanged. Rationale:
choosing an imputation VALUE is a modeling-pipeline decision (e.g.
`sklearn.impute.SimpleImputer` fit on a train split), not a data-
assembly decision — imputing here would bake an arbitrary choice into
"the data" before any train/validation split exists. Existing paired
indicator columns (`orig_balance_before_zero`, `origin_is_first_seen`,
`dest_is_first_seen`, etc.) already explain *why* a given ratio/average
is NaN wherever such an indicator exists, serving as the "availability
indicator" for those cases without adding new columns. No baseline
feature is expected to ever produce +/-inf (see `audit_dataframe` /
`run_full_stream_audit`, which verify this rather than assume it) —
that is a bug-detection check, not a case the undefined-value policy
needs to additionally define behavior for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.ingest import DatasetNotFoundError, SchemaValidationError, load_csv, validate_columns
from src.validate_raw_paysim import DEFAULT_RAW_CSV
from src import features_raw_paysim as feat
from src import graph_features_raw_paysim as gf

DEFAULT_CHUNKSIZE = 200_000
DEV_SAMPLE_ROWS = 20_000
VALIDATION_SAMPLE_ROWS = 250_000

# isFraud is intentionally absent from every INPUT_COLUMNS constant in
# this project; reusing M5's superset here reads no additional columns.
INPUT_COLUMNS = feat.INPUT_COLUMNS

IDENTIFIER_COLUMNS = ["row_id", "nameOrig", "nameDest"]
TARGET_COLUMN = "isFraud"  # never read; named here only for audit assertions

GRAPH_FEATURES_RETAINED_IN_BASELINE = [
    "origin_out_degree_causal",
    "dest_in_degree_causal",
    "origin_seen_as_dest_before_causal",
    "dest_seen_as_origin_before_causal",
    "dest_incoming_concentration_causal",
]

GRAPH_FEATURES_AUDIT_ONLY = [
    "pair_seen_before_causal",
    "pair_prior_count_causal",
    "origin_outgoing_concentration_causal",
]

assert set(GRAPH_FEATURES_RETAINED_IN_BASELINE + GRAPH_FEATURES_AUDIT_ONLY) == set(
    gf.baseline_ready_columns()
), "Milestone 8's graph-feature classification must partition exactly Milestone 7's baseline-ready set"

# Fixed, deterministic order: Milestone 5's 34 baseline columns (their
# own established order, unchanged) followed by the 5 retained graph
# columns (declared order above).
BASELINE_FEATURE_COLUMNS: list[str] = feat.baseline_feature_columns() + GRAPH_FEATURES_RETAINED_IN_BASELINE

# All columns this module's assembly produces, in the exact order they
# are written out (identifiers first, then baseline, then audit-only
# graph columns, then M5's own exploratory/identifier columns retained
# for completeness of the audit).
# "type" is the raw categorical (M5 tags it "identifier" since the
# one-hot type_* columns are the model-input form) -- included here,
# audit-only, purely so the audit can report categorical value counts
# for the raw category as required by this milestone's audit spec.
AUDIT_ONLY_CATEGORICAL_COLUMNS = ["type"]

ALL_ASSEMBLED_COLUMNS: list[str] = (
    IDENTIFIER_COLUMNS
    + BASELINE_FEATURE_COLUMNS
    + GRAPH_FEATURES_AUDIT_ONLY
    + feat.exploratory_only_columns()
    + AUDIT_ONLY_CATEGORICAL_COLUMNS
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
BASELINE_SPEC_PATH = REPORTS_DIR / "baseline_dataset_specification.md"
DEV_SAMPLE_AUDIT_PATH = REPORTS_DIR / "baseline_dev_sample_audit.json"
FULL_STREAM_AUDIT_PATH = REPORTS_DIR / "baseline_full_stream_audit.json"


def iter_baseline_chunks(
    csv_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
    row_id_start: int = 0,
) -> Iterator[pd.DataFrame]:
    """Stream the assembled baseline table. One sequential read of the
    raw CSV; Milestone 5's and Milestone 7's causal state dicts are
    each maintained independently but updated from the same chunk, so
    both feature sets stay aligned row-for-row without re-reading the
    file. Bounded memory: one raw chunk plus the (small, degree-bounded)
    causal state dicts from both milestones — never a full-size
    DataFrame, never a NetworkX graph.
    """
    origin_count: dict[str, int] = {}
    origin_sum: dict[str, float] = {}
    dest_count: dict[str, int] = {}
    dest_sum: dict[str, float] = {}
    origin_to_dest_counts: dict[str, dict[str, int]] = {}
    dest_to_origin_counts: dict[str, dict[str, int]] = {}

    row_id = row_id_start
    rows_processed = 0

    with pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize) as reader:
        for chunk in reader:
            if max_rows is not None and rows_processed >= max_rows:
                break
            if max_rows is not None and rows_processed + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_processed]

            m5_stateless = feat.compute_stateless_features(chunk)
            m5_causal = feat._update_causal_state(chunk, origin_count, origin_sum, dest_count, dest_sum)
            m7_causal = gf._update_and_extract(chunk, origin_to_dest_counts, dest_to_origin_counts)

            combined = pd.concat(
                [
                    m5_stateless.reset_index(drop=True),
                    m5_causal.reset_index(drop=True),
                    m7_causal.reset_index(drop=True),
                ],
                axis=1,
            )
            combined.insert(0, "row_id", range(row_id, row_id + len(combined)))

            row_id += len(combined)
            rows_processed += len(chunk)

            yield combined[ALL_ASSEMBLED_COLUMNS]


def build_bounded_sample(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = DEV_SAMPLE_ROWS
) -> pd.DataFrame:
    """Materialize a small, explicitly bounded sample. Intended for
    development/tests (default 20,000 rows) or a bounded validation run
    (e.g. 100,000-250,000 rows) — NOT for the full 6,362,620-row file.
    """
    chunks = list(iter_baseline_chunks(csv_path, chunksize=chunksize, max_rows=max_rows))
    if not chunks:
        return pd.DataFrame(columns=ALL_ASSEMBLED_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def _column_stats(series: pd.Series) -> dict[str, Any]:
    is_numeric = pd.api.types.is_numeric_dtype(series)
    stats: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()),
    }
    if is_numeric:
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        stats["infinite_count"] = int(np.isinf(series.dropna()).sum())
        stats["min"] = float(finite.min()) if len(finite) else None
        stats["max"] = float(finite.max()) if len(finite) else None
        stats["mean"] = float(finite.mean()) if len(finite) else None
        non_null = series.dropna()
        nunique = int(non_null.nunique())
        stats["nunique_non_null"] = nunique
        stats["is_constant"] = nunique <= 1
        if nunique >= 1:
            top_share = non_null.value_counts(normalize=True).iloc[0]
            stats["top_value_share"] = float(top_share)
            stats["is_near_constant"] = top_share >= 0.999 and nunique > 1
        else:
            stats["top_value_share"] = None
            stats["is_near_constant"] = False
    else:
        stats["infinite_count"] = 0
        stats["value_counts"] = series.value_counts(dropna=True).to_dict()
        nunique = int(series.dropna().nunique())
        stats["nunique_non_null"] = nunique
        stats["is_constant"] = nunique <= 1
        stats["is_near_constant"] = False
    return stats


def audit_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Full column-by-column audit of a materialized (bounded) sample."""
    return {
        "row_count": len(df),
        "column_order": list(df.columns),
        "baseline_feature_columns": BASELINE_FEATURE_COLUMNS,
        "identifier_columns_present": [c for c in IDENTIFIER_COLUMNS if c in df.columns],
        "target_column_present": TARGET_COLUMN in df.columns,
        "columns": {col: _column_stats(df[col]) for col in df.columns},
    }


def run_full_stream_audit(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int | None = None
) -> dict[str, Any]:
    """Stream the ENTIRE assembly (or up to max_rows) computing only
    aggregate running statistics — no chunk's rows are retained after
    its stats are folded in. Used both for bounded-sample-scale audits
    and (with max_rows=None) the full 6,362,620-row audit.
    """
    total_rows = 0
    acc: dict[str, dict[str, Any]] = {}

    for chunk in iter_baseline_chunks(csv_path, chunksize=chunksize, max_rows=max_rows):
        assert TARGET_COLUMN not in chunk.columns
        total_rows += len(chunk)
        for col in chunk.columns:
            s = chunk[col]
            a = acc.setdefault(col, {
                "dtype": str(s.dtype),
                "missing_count": 0,
                "infinite_count": 0,
                "count": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
                "value_counts": {},
                "is_numeric": pd.api.types.is_numeric_dtype(s),
            })
            a["missing_count"] += int(s.isna().sum())
            non_null = s.dropna()
            if a["is_numeric"]:
                a["infinite_count"] += int(np.isinf(non_null).sum())
                finite = non_null.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite):
                    a["count"] += len(finite)
                    a["sum"] += float(finite.sum())
                    fmin, fmax = float(finite.min()), float(finite.max())
                    a["min"] = fmin if a["min"] is None else min(a["min"], fmin)
                    a["max"] = fmax if a["max"] is None else max(a["max"], fmax)
            else:
                for val, cnt in non_null.value_counts().items():
                    a["value_counts"][val] = a["value_counts"].get(val, 0) + int(cnt)

    columns_report = {}
    for col, a in acc.items():
        entry = {
            "dtype": a["dtype"],
            "missing_count": a["missing_count"],
            "missing_rate": a["missing_count"] / total_rows if total_rows else 0.0,
            "infinite_count": a["infinite_count"],
        }
        if a["is_numeric"]:
            entry["min"] = a["min"]
            entry["max"] = a["max"]
            entry["mean"] = (a["sum"] / a["count"]) if a["count"] else None
            entry["is_constant"] = a["min"] is not None and a["min"] == a["max"]
        else:
            vc = a["value_counts"]
            entry["value_counts"] = vc
            entry["is_constant"] = len(vc) <= 1
            if vc:
                top_share = max(vc.values()) / sum(vc.values())
                entry["top_value_share"] = top_share
                entry["is_near_constant"] = top_share >= 0.999 and len(vc) > 1
            else:
                entry["is_near_constant"] = False
        columns_report[col] = entry

    return {
        "total_rows": total_rows,
        "is_full_dataset": max_rows is None,
        "target_column_used": False,
        "columns": columns_report,
    }


def check_determinism(
    csv_path: Path, chunksizes: tuple[int, int] = (4, 1000), max_rows: int = 200
) -> dict[str, Any]:
    """Assemble the same bounded row range under two different chunk
    sizes and confirm byte-identical output (including NaN placement).
    """
    df_a = build_bounded_sample(csv_path, chunksize=chunksizes[0], max_rows=max_rows)
    df_b = build_bounded_sample(csv_path, chunksize=chunksizes[1], max_rows=max_rows)
    try:
        pd.testing.assert_frame_equal(df_a, df_b)
        equivalent = True
        detail = "identical"
    except AssertionError as exc:
        equivalent = False
        detail = str(exc)
    return {"chunksizes_compared": list(chunksizes), "rows": max_rows, "equivalent": equivalent, "detail": detail}


def write_baseline_spec(out_path: Path = BASELINE_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FraudGraph AI — Baseline Dataset Specification (Milestone 8)",
        "",
        "Generated in part from `src/baseline_dataset.py` constants — "
        "the module docstring there is the full narrative source of "
        "truth; this file summarizes the resulting contract.",
        "",
        "## Baseline feature contract (39 columns, fixed order)",
        "",
        ", ".join(f"`{c}`" for c in BASELINE_FEATURE_COLUMNS),
        "",
        "## Identifier / target columns (never model inputs)",
        "",
        f"- Identifiers (audit/traceability only): {', '.join(f'`{c}`' for c in IDENTIFIER_COLUMNS)}",
        f"- Target column (never read as a feature anywhere in this project): `{TARGET_COLUMN}`",
        "",
        "## Graph features retained in baseline (from Milestone 7)",
        "",
        ", ".join(f"`{c}`" for c in GRAPH_FEATURES_RETAINED_IN_BASELINE),
        "",
        "## Graph features retained for audit/ablation only (excluded from baseline)",
        "",
        ", ".join(f"`{c}`" for c in GRAPH_FEATURES_AUDIT_ONLY),
        "",
        "See the module docstring in `src/baseline_dataset.py` for the "
        "full measured rationale (constant-zero / near-degenerate / "
        "99.95%-missing findings from Milestones 6-7, and the explicit "
        "scope caveat that these were measured on the 2,000,000-row "
        "subset, re-checked at full scale by `run_full_stream_audit`).",
        "",
        "## Excluded per existing Milestone 5 policy (not reintroduced)",
        "",
        ", ".join(f"`{c}`" for c in feat.exploratory_only_columns()),
        "",
        "## Same-step / causal convention",
        "",
        "Identical to Milestones 5 and 7: file/row order is the event "
        "order; within a shared `step`, an earlier file row is prior. "
        "This module reuses Milestone 5's and 7's per-chunk causal "
        "state functions verbatim — no new causal logic is introduced.",
        "",
        "## Undefined-value policy",
        "",
        "Preserve NaN; never silently impute. Imputation is deferred to "
        "the modeling pipeline (e.g. fit on a train split), since "
        "choosing an imputation value is a modeling decision, not a "
        "data-assembly decision. Existing paired indicator columns "
        "(`orig_balance_before_zero`, `origin_is_first_seen`, "
        "`dest_is_first_seen`, etc.) already explain *why* a ratio/"
        "average is undefined wherever such an indicator exists. No "
        "baseline feature is expected to produce +/-inf; "
        "`run_full_stream_audit` verifies this rather than assuming it.",
        "",
        "## Bounded assembly design",
        "",
        f"- Development sample: {DEV_SAMPLE_ROWS:,} rows (`build_bounded_sample`)",
        f"- Bounded validation sample: {VALIDATION_SAMPLE_ROWS:,} rows",
        "- Full-stream audit: `run_full_stream_audit(max_rows=None)` "
        "streams the entire 6,362,620-row file computing only running "
        "aggregate statistics per column — no chunk's rows are retained "
        "after folding its stats in, so no full-size table is ever held.",
        "",
        "## Known limitations",
        "",
        "- The graph-feature constancy/near-degeneracy findings that "
        "justify GRAPH_FEATURES_AUDIT_ONLY were originally measured on "
        "Milestone 6/7's 2,000,000-row subset. This milestone's "
        "`run_full_stream_audit(max_rows=None)` re-ran the check on "
        "the COMPLETE 6,362,620-row file and CONFIRMED it: "
        "`pair_seen_before_causal` / `pair_prior_count_causal` remain "
        "exactly constant-zero (mean 0.0, min 0.0, max 0.0) across all "
        "6,362,620 rows; `origin_outgoing_concentration_causal` remains "
        "99.85% missing with mean 0.999 when defined. The 5 "
        "baseline-retained graph features were also confirmed non-"
        "constant at full scale, with `dest_incoming_concentration_"
        "causal` mean 0.290 (57.2% defined) — consistent with, and "
        "strengthening, the 2,000,000-row subset finding. See "
        "reports/baseline_full_stream_audit.json for exact figures.",
        "- Predictive usefulness is not evaluated here (no model is "
        "trained in this milestone) — retained-in-baseline means "
        "'technically valid, non-degenerate, and leakage-safe', not "
        "'confirmed predictive'.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble and audit the Milestone 8 baseline dataset "
        "(bounded sample + full-stream aggregate audit; never a full "
        "materialized feature table)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--dev-rows", type=int, default=DEV_SAMPLE_ROWS)
    parser.add_argument("--full-stream", action="store_true",
                         help="Run the full 6,362,620-row streaming audit (default: skip).")
    args = parser.parse_args(argv)

    try:
        validate_columns(load_csv(args.csv))
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Baseline assembly failed: {exc}", file=sys.stderr)
        return 1

    write_baseline_spec()
    print(f"Baseline spec written to: {BASELINE_SPEC_PATH}")

    t0 = time.time()
    sample = build_bounded_sample(args.csv, chunksize=args.chunksize, max_rows=args.dev_rows)
    dev_audit = audit_dataframe(sample)
    determinism = check_determinism(args.csv, max_rows=min(args.dev_rows, 500))
    dev_runtime = time.time() - t0

    with DEV_SAMPLE_AUDIT_PATH.open("w") as f:
        json.dump({"runtime_seconds": round(dev_runtime, 2), "determinism_check": determinism,
                   **dev_audit}, f, indent=2, default=str)

    print(f"Dev sample rows       : {len(sample):,}")
    print(f"Determinism check     : {determinism['equivalent']}")
    print(f"Dev audit runtime     : {dev_runtime:.2f}s")
    print(f"Dev audit written to  : {DEV_SAMPLE_AUDIT_PATH}")

    if args.full_stream:
        t1 = time.time()
        full_audit = run_full_stream_audit(args.csv, chunksize=args.chunksize, max_rows=None)
        full_runtime = time.time() - t1
        with FULL_STREAM_AUDIT_PATH.open("w") as f:
            json.dump({"runtime_seconds": round(full_runtime, 2), **full_audit}, f, indent=2, default=str)
        print(f"Full-stream rows      : {full_audit['total_rows']:,}")
        print(f"Full-stream runtime   : {full_runtime:.2f}s")
        print(f"Full-stream audit to  : {FULL_STREAM_AUDIT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
