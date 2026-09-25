"""Explainable tabular feature engineering for raw PaySim (Milestone 5).

Transforms raw PaySim rows into model-ready tabular features. Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

Never writes to, modifies, or samples the raw CSV. No graph is built here
(deferred to a later milestone) and no model is trained here.

LABEL SAFETY: `isFraud` is never read by this module. INPUT_COLUMNS
(below) deliberately omits it, so it is structurally impossible for any
feature computed here to be derived from the fraud label.

TEMPORAL LEAKAGE: Milestone 4's EDA confirmed the raw CSV is already
sorted by `step` (non-decreasing). This module relies on that fact to
compute identity/activity aggregates as CAUSAL, leave-one-out-and-more
("prior transactions only") running statistics in a single sequential
pass — strictly stronger than simple leave-one-out, since it also
excludes same-account transactions that occur *later* in the file/step
order, not just the current row. Ties within the same `step` value are
broken by file/row order, which PaySim does not itself resolve; this is
a documented, deterministic convention, not a claim about true
sub-hour ordering.

SIMULATION-BOUNDARY ARTIFACT: Milestone 4 found that the final simulated
day (step bucket 30) has legitimate transaction volume collapse to a
fraction of earlier days while fraud volume stays roughly flat, spiking
the apparent fraud rate to 100% for that bucket. Raw `step` and the
derived `day_index` therefore both risk teaching a model "late in the
simulation" rather than a real temporal fraud pattern, and are labeled
EXPLORATORY_ONLY below. `hour_of_day` (step modulo 24) is kept as the
recommended time feature since it is cyclical and does not encode
absolute position in the simulation.

BALANCE MISMATCH DIAGNOSTICS: `orig_balance_mismatch` /
`dest_balance_mismatch` flag rows where newBalance does not equal
oldBalance +/- amount. Milestone 3 established this is a well-documented
PaySim simulator characteristic (frequent even for legitimate
transactions), NOT an automatic fraud indicator — these columns describe
the simulator's balance bookkeeping, not a verdict on the transaction.

See reports/feature_specification.md for the full per-feature reference,
generated from FEATURE_METADATA below (the single source of truth).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.ingest import DatasetNotFoundError, SchemaValidationError, load_csv, validate_columns
from src.validate_raw_paysim import BALANCE_TOLERANCE, DEFAULT_RAW_CSV, VALID_TRANSACTION_TYPES

DEFAULT_CHUNKSIZE = 500_000
STEPS_PER_DAY = 24
SORTED_TYPES = sorted(VALID_TRANSACTION_TYPES)

# isFraud is intentionally absent — see module docstring "LABEL SAFETY".
INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

FEATURE_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "feature_specification.md"
)

# Single source of truth for both the generated documentation
# (reports/feature_specification.md) and the leakage-safety test
# assertions. `leakage_status` is one of:
#   "safe_for_baseline" - fine to use in the first predictive baseline
#   "exploratory_only"  - documented limitation, excluded from baseline
#   "identifier"         - not a feature; kept for traceability/joins
FEATURE_METADATA: dict[str, dict[str, str]] = {
    "row_id": {
        "category": "identifier",
        "leakage_status": "identifier",
        "description": "Synthetic sequential row index (0-based), not part "
        "of raw PaySim. Added only for traceability/joins across chunks.",
    },
    "step": {
        "category": "transaction",
        "leakage_status": "exploratory_only",
        "description": "Raw PaySim simulation hour (1-743), passed through "
        "unchanged. Encodes absolute position in the simulation, which "
        "correlates with the Milestone 4 simulation-boundary artifact "
        "(legitimate-volume collapse in the final day). Not recommended "
        "for the first baseline.",
    },
    "day_index": {
        "category": "transaction",
        "leakage_status": "exploratory_only",
        "description": "(step - 1) // 24, 0-indexed simulated day. This is "
        "exactly the bucketing that showed the 100% fraud-rate artifact "
        "on the final day in Milestone 4 (driven by legitimate-volume "
        "collapse, not a real signal). Not recommended for the first "
        "baseline without correcting for that artifact.",
    },
    "hour_of_day": {
        "category": "transaction",
        "leakage_status": "safe_for_baseline",
        "description": "(step - 1) % 24. Cyclical hour-of-day; does not "
        "encode absolute position in the simulation, so it does not "
        "inherit the day_index/step boundary-artifact risk.",
    },
    "type": {
        "category": "transaction",
        "leakage_status": "identifier",
        "description": "Raw transaction type string, kept for readability "
        "and joins. Most estimators need the one-hot type_* columns "
        "instead.",
    },
    **{
        f"type_{t}": {
            "category": "transaction",
            "leakage_status": "safe_for_baseline",
            "description": f"One-hot indicator: 1 if type == '{t}' else 0. "
            "All five type_* columns are always present in every chunk "
            "(fixed category list), so schema is identical regardless of "
            "which types appear in a given chunk.",
        }
        for t in SORTED_TYPES
    },
    "amount": {
        "category": "transaction",
        "leakage_status": "safe_for_baseline",
        "description": "Raw transaction amount, passed through unchanged.",
    },
    "log_amount": {
        "category": "transaction",
        "leakage_status": "safe_for_baseline",
        "description": "log1p(max(amount, 0)). log1p is safe at amount=0 "
        "(returns 0 rather than -inf); the max(...,0) guard is defensive "
        "since Milestone 3 already confirmed no negative amounts exist "
        "in the raw file.",
    },
    "isFlaggedFraud": {
        "category": "transaction",
        "leakage_status": "safe_for_baseline",
        "description": "Raw PaySim rule-based flag, passed through "
        "unchanged. Not the target (isFraud); Milestone 4 found it only "
        "catches 16 of 8,213 fraud cases, so treat it as a weak signal.",
    },
    "nameOrig": {
        "category": "identity",
        "leakage_status": "identifier",
        "description": "Raw origin account id. High-cardinality "
        "identifier, kept for traceability/joins (e.g. to a future graph "
        "milestone), not intended as a raw model input.",
    },
    "nameDest": {
        "category": "identity",
        "leakage_status": "identifier",
        "description": "Raw destination account id. Same caveat as "
        "nameOrig.",
    },
    "is_dest_merchant": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if nameDest starts with 'M' (merchant), else 0. "
        "PaySim does not simulate merchant balances (Milestone 3); used "
        "to interpret dest_balance_mismatch correctly.",
    },
    "orig_balance_before": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "= oldbalanceOrg, renamed for clarity.",
    },
    "orig_balance_after": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "= newbalanceOrig, renamed for clarity.",
    },
    "dest_balance_before": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "= oldbalanceDest, renamed for clarity.",
    },
    "dest_balance_after": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "= newbalanceDest, renamed for clarity.",
    },
    "orig_balance_delta": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "orig_balance_after - orig_balance_before. Signed: "
        "positive for CASH_IN (funds added), typically negative otherwise.",
    },
    "dest_balance_delta": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "dest_balance_after - dest_balance_before.",
    },
    "amount_to_orig_balance_ratio": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "amount / orig_balance_before when orig_balance_"
        "before > 0, else NaN (safe zero handling — never divides by "
        "zero or raises). Pair with orig_balance_before_zero to "
        "distinguish 'undefined' from 'small'.",
    },
    "amount_to_dest_balance_ratio": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "amount / dest_balance_before when dest_balance_"
        "before > 0, else NaN (safe zero handling).",
    },
    "orig_balance_before_zero": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if orig_balance_before == 0, else 0.",
    },
    "orig_balance_after_zero": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if orig_balance_after == 0, else 0.",
    },
    "dest_balance_before_zero": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if dest_balance_before == 0, else 0.",
    },
    "dest_balance_after_zero": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if dest_balance_after == 0, else 0.",
    },
    "orig_balance_mismatch": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if orig_balance_after != orig_balance_before "
        "+/- amount (direction-aware: CASH_IN adds, all other types "
        "subtract; tolerance 0.01), else 0. SIMULATOR-DERIVED DIAGNOSTIC "
        "ONLY (Milestone 3) — a large share of legitimate rows also "
        "mismatch; this is NOT an automatic fraud indicator.",
    },
    "dest_balance_mismatch": {
        "category": "balance",
        "leakage_status": "safe_for_baseline",
        "description": "1 if dest_balance_after != dest_balance_before + "
        "amount (tolerance 0.01), else 0. Defined as 0 (not applicable) "
        "for merchant destinations, since PaySim does not track merchant "
        "balances at all — use is_dest_merchant to distinguish this case. "
        "SIMULATOR-DERIVED DIAGNOSTIC ONLY, not an automatic fraud "
        "indicator (see orig_balance_mismatch).",
    },
    "origin_txn_count_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "Count of PRIOR transactions by this nameOrig, "
        "strictly before the current row in file/step order (0 for a "
        "first occurrence). Computed in a single sequential pass; never "
        "includes the current row or any later transaction.",
    },
    "origin_total_amount_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "Sum of amounts of this nameOrig's prior "
        "transactions only (0.0 if none).",
    },
    "origin_avg_amount_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "origin_total_amount_causal / origin_txn_count_"
        "causal when count > 0, else NaN (no prior history yet).",
    },
    "origin_is_first_seen": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "1 if this is nameOrig's first transaction in "
        "file/step order (origin_txn_count_causal == 0), else 0.",
    },
    "dest_txn_count_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "Count of PRIOR transactions to this nameDest, "
        "strictly before the current row (0 for a first occurrence).",
    },
    "dest_total_amount_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "Sum of amounts of this nameDest's prior "
        "transactions only (0.0 if none).",
    },
    "dest_avg_amount_causal": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "dest_total_amount_causal / dest_txn_count_causal "
        "when count > 0, else NaN.",
    },
    "dest_is_first_seen": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "1 if this is nameDest's first transaction in "
        "file/step order, else 0.",
    },
    "amount_vs_origin_avg_ratio": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "amount / origin_avg_amount_causal when defined, "
        "else NaN. How large this transaction is relative to this "
        "account's own prior average (cold-start-safe).",
    },
    "amount_vs_dest_avg_ratio": {
        "category": "identity",
        "leakage_status": "safe_for_baseline",
        "description": "amount / dest_avg_amount_causal when defined, "
        "else NaN.",
    },
}

# Deliberately NOT implemented as a materialized column: a naive
# full-dataset (non-causal) version of the identity aggregates above.
# It would require holding a second full set of per-account dicts
# alongside the causal running dicts (roughly doubling this module's
# peak memory) purely to populate a feature the instructions already
# say must be excluded from the first baseline. The causal aggregate's
# final value per account (reachable by taking that account's last
# occurrence and adding 1 to the count / that row's own amount to the
# sum) already recovers the full-dataset total if ever needed for
# offline exploration, so no information is lost by omitting it here.


def baseline_feature_columns() -> list[str]:
    """Columns tagged safe_for_baseline — recommended input for Milestone 6+."""
    return [c for c, meta in FEATURE_METADATA.items() if meta["leakage_status"] == "safe_for_baseline"]


def exploratory_only_columns() -> list[str]:
    return [c for c, meta in FEATURE_METADATA.items() if meta["leakage_status"] == "exploratory_only"]


def identifier_columns() -> list[str]:
    return [c for c, meta in FEATURE_METADATA.items() if meta["leakage_status"] == "identifier"]


def all_feature_columns() -> list[str]:
    return list(FEATURE_METADATA.keys())


def compute_stateless_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """Per-row features requiring no cross-row state. Pure function of the
    input chunk — safe to call independently on any chunk in any order.
    """
    out = pd.DataFrame(index=chunk.index)

    step = chunk["step"]
    out["step"] = step
    out["day_index"] = ((step - 1) // STEPS_PER_DAY).astype(int)
    out["hour_of_day"] = ((step - 1) % STEPS_PER_DAY).astype(int)

    out["type"] = chunk["type"]
    for t in SORTED_TYPES:
        out[f"type_{t}"] = (chunk["type"] == t).astype(int)

    amount = chunk["amount"]
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount.clip(lower=0))
    out["isFlaggedFraud"] = chunk["isFlaggedFraud"]

    out["nameOrig"] = chunk["nameOrig"]
    out["nameDest"] = chunk["nameDest"]

    is_merchant = chunk["nameDest"].str.startswith("M")
    out["is_dest_merchant"] = is_merchant.astype(int)

    orig_before = chunk["oldbalanceOrg"]
    orig_after = chunk["newbalanceOrig"]
    dest_before = chunk["oldbalanceDest"]
    dest_after = chunk["newbalanceDest"]

    out["orig_balance_before"] = orig_before
    out["orig_balance_after"] = orig_after
    out["dest_balance_before"] = dest_before
    out["dest_balance_after"] = dest_after
    out["orig_balance_delta"] = orig_after - orig_before
    out["dest_balance_delta"] = dest_after - dest_before

    out["amount_to_orig_balance_ratio"] = np.where(orig_before > 0, amount / orig_before, np.nan)
    out["amount_to_dest_balance_ratio"] = np.where(dest_before > 0, amount / dest_before, np.nan)

    out["orig_balance_before_zero"] = (orig_before == 0).astype(int)
    out["orig_balance_after_zero"] = (orig_after == 0).astype(int)
    out["dest_balance_before_zero"] = (dest_before == 0).astype(int)
    out["dest_balance_after_zero"] = (dest_after == 0).astype(int)

    orig_sign = np.where(chunk["type"] == "CASH_IN", 1, -1)
    expected_orig_after = orig_before + orig_sign * amount
    orig_mismatch = ~np.isclose(orig_after, expected_orig_after, atol=BALANCE_TOLERANCE)
    out["orig_balance_mismatch"] = orig_mismatch.astype(int)

    expected_dest_after = dest_before + amount
    dest_mismatch_raw = ~np.isclose(dest_after, expected_dest_after, atol=BALANCE_TOLERANCE)
    out["dest_balance_mismatch"] = np.where(is_merchant, 0, dest_mismatch_raw.astype(int))

    return out


def _update_causal_state(
    chunk: pd.DataFrame,
    origin_count: dict[str, int],
    origin_sum: dict[str, float],
    dest_count: dict[str, int],
    dest_sum: dict[str, float],
) -> pd.DataFrame:
    """Assign each row the running state as of BEFORE that row (prior
    transactions only), then update the running state with this chunk's
    rows. Requires the input to already be in causal (file/step) order —
    see module docstring "TEMPORAL LEAKAGE".
    """
    n = len(chunk)
    origin_txn_count = np.empty(n, dtype=np.int64)
    origin_total_amount = np.empty(n, dtype=np.float64)
    dest_txn_count = np.empty(n, dtype=np.int64)
    dest_total_amount = np.empty(n, dtype=np.float64)

    names_orig = chunk["nameOrig"].to_numpy()
    names_dest = chunk["nameDest"].to_numpy()
    amounts = chunk["amount"].to_numpy()

    for i in range(n):
        o = names_orig[i]
        d = names_dest[i]
        a = float(amounts[i])

        origin_txn_count[i] = origin_count.get(o, 0)
        origin_total_amount[i] = origin_sum.get(o, 0.0)
        dest_txn_count[i] = dest_count.get(d, 0)
        dest_total_amount[i] = dest_sum.get(d, 0.0)

        origin_count[o] = origin_count.get(o, 0) + 1
        origin_sum[o] = origin_sum.get(o, 0.0) + a
        dest_count[d] = dest_count.get(d, 0) + 1
        dest_sum[d] = dest_sum.get(d, 0.0) + a

    out = pd.DataFrame(index=chunk.index)
    out["origin_txn_count_causal"] = origin_txn_count
    out["origin_total_amount_causal"] = origin_total_amount
    out["origin_avg_amount_causal"] = np.where(
        origin_txn_count > 0, origin_total_amount / np.maximum(origin_txn_count, 1), np.nan
    )
    out["origin_is_first_seen"] = (origin_txn_count == 0).astype(int)

    out["dest_txn_count_causal"] = dest_txn_count
    out["dest_total_amount_causal"] = dest_total_amount
    out["dest_avg_amount_causal"] = np.where(
        dest_txn_count > 0, dest_total_amount / np.maximum(dest_txn_count, 1), np.nan
    )
    out["dest_is_first_seen"] = (dest_txn_count == 0).astype(int)

    out["amount_vs_origin_avg_ratio"] = np.where(
        origin_txn_count > 0, amounts / out["origin_avg_amount_causal"], np.nan
    )
    out["amount_vs_dest_avg_ratio"] = np.where(
        dest_txn_count > 0, amounts / out["dest_avg_amount_causal"], np.nan
    )
    return out


def iter_engineered_chunks(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, row_id_start: int = 0
) -> Iterator[pd.DataFrame]:
    """Stream engineered feature chunks. Bounded memory: at most one raw
    chunk plus four small running dicts (per-account count/sum, origin
    and destination) are held at a time — never a full-size DataFrame.

    isFraud is never read (see INPUT_COLUMNS) — this generator cannot
    produce a label-derived feature even by accident.
    """
    origin_count: dict[str, int] = {}
    origin_sum: dict[str, float] = {}
    dest_count: dict[str, int] = {}
    dest_sum: dict[str, float] = {}

    row_id = row_id_start
    for raw_chunk in pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize):
        stateless = compute_stateless_features(raw_chunk)
        causal = _update_causal_state(raw_chunk, origin_count, origin_sum, dest_count, dest_sum)

        combined = pd.concat([stateless, causal], axis=1)
        combined.insert(0, "row_id", range(row_id, row_id + len(combined)))
        row_id += len(combined)

        yield combined.reset_index(drop=True)


def build_features(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """Concatenate all engineered chunks into one DataFrame.

    Intended for small fixtures (tests) or deliberately small inputs
    only. Do NOT call this on the full ~6.36M-row raw file — that would
    materialize a full-size transformed dataset in memory, which
    Milestone 5 explicitly reserves for later, explicit approval. Use
    `iter_engineered_chunks` (or `validate_against_full_dataset` below)
    against the real file instead.
    """
    chunks = list(iter_engineered_chunks(csv_path, chunksize=chunksize))
    if not chunks:
        return pd.DataFrame(columns=all_feature_columns())
    return pd.concat(chunks, ignore_index=True)


def validate_against_full_dataset(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE
) -> dict[str, Any]:
    """Stream the real dataset through the engineering pipeline WITHOUT
    materializing or writing a full-size transformed table — only
    lightweight summary statistics are accumulated, verifying the
    pipeline runs end-to-end and produces a sane, leak-free, consistent
    schema at full scale.
    """
    total_rows = 0
    expected_columns = None
    nan_counts: dict[str, int] = {}
    mismatch_true_count = 0
    zero_amount_rows = 0
    zero_orig_balance_rows = 0

    for chunk in iter_engineered_chunks(csv_path, chunksize=chunksize):
        assert "isFraud" not in chunk.columns, "isFraud leaked into engineered features"

        if expected_columns is None:
            expected_columns = list(chunk.columns)
        else:
            assert list(chunk.columns) == expected_columns, "schema drift across chunks"

        total_rows += len(chunk)
        for col in chunk.columns:
            nan_counts[col] = nan_counts.get(col, 0) + int(chunk[col].isna().sum())

        mismatch_true_count += int(chunk["orig_balance_mismatch"].sum())
        zero_amount_rows += int((chunk["amount"] == 0).sum())
        zero_orig_balance_rows += int((chunk["orig_balance_before"] == 0).sum())

    return {
        "total_rows": total_rows,
        "column_count": len(expected_columns) if expected_columns else 0,
        "columns": expected_columns,
        "nan_counts_nonzero": {k: v for k, v in nan_counts.items() if v > 0},
        "orig_balance_mismatch_count": mismatch_true_count,
        "zero_amount_rows": zero_amount_rows,
        "zero_orig_balance_rows": zero_orig_balance_rows,
        "isFraud_present": False,
    }


def write_feature_spec(out_path: Path = FEATURE_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FraudGraph AI — Raw PaySim Feature Specification (Milestone 5)",
        "",
        "Generated from `src/features_raw_paysim.py::FEATURE_METADATA` "
        "(single source of truth — do not hand-edit this file).",
        "",
        "`isFraud` is never read by the feature-engineering module "
        "(see `INPUT_COLUMNS`), so no feature here can be derived from "
        "the fraud label — confirmed absent from both `INPUT_COLUMNS` "
        "and `FEATURE_METADATA`.",
        "",
        "## Chronological ordering assumption (Milestone 5/6 audit)",
        "",
        "- The raw CSV was directly verified to be non-decreasing by "
        "`step` (`step.is_monotonic_increasing == True`).",
        "- **PaySim provides no timestamp finer than `step` (1 simulated "
        "hour).** 6,361,877 of the file's 6,362,620 rows (99.99%) share "
        "their `step` value with at least one other row, so intra-step "
        "ordering is the norm, not an edge case.",
        "- Causal identity features (`*_causal`, `*_is_first_seen`, "
        "`amount_vs_*_avg_ratio`) treat **file/row order as the "
        "causality order**: a row is only ever influenced by rows that "
        "appear strictly earlier in the file. When two rows share the "
        "same `step`, the one appearing first in the file is treated as "
        "prior — a documented, deterministic convention, not a claim "
        "about true sub-hour timing (PaySim does not resolve this "
        "itself).",
        "- This means a same-step row that appears LATER in the file "
        "can never influence an EARLIER row's causal features, even "
        "though both rows carry the identical `step` value. Verified "
        "directly by `tests/test_features_raw_paysim.py::"
        "TestSameStepCausalOrdering`.",
        "",
        "Leakage status legend:",
        "- **safe_for_baseline** — recommended for the first predictive model",
        "- **exploratory_only** — documented limitation, excluded from the baseline",
        "- **identifier** — not a model feature; kept for traceability/joins",
        "",
    ]

    by_category: dict[str, list[str]] = {}
    for col, meta in FEATURE_METADATA.items():
        by_category.setdefault(meta["category"], []).append(col)

    for category in ["transaction", "balance", "identity", "identifier"]:
        cols = by_category.get(category)
        if not cols:
            continue
        lines.append(f"## {category.capitalize()} features")
        lines.append("")
        for col in cols:
            meta = FEATURE_METADATA[col]
            lines.append(f"### `{col}` — *{meta['leakage_status']}*")
            lines.append(meta["description"])
            lines.append("")

    lines.append("## Recommended baseline feature columns")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in baseline_feature_columns()))
    lines.append("")
    lines.append("## Excluded from baseline (exploratory only)")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in exploratory_only_columns()))
    lines.append("")
    lines.append("## Identifiers (not model features)")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in identifier_columns()))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the feature-engineering pipeline against the "
        "real raw PaySim CSV (streamed; does not materialize a full-size "
        "transformed dataset) and (re)write the feature specification doc."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--spec-out", type=Path, default=FEATURE_SPEC_PATH)
    args = parser.parse_args(argv)

    try:
        validate_columns(load_csv(args.csv))
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Feature engineering validation failed: {exc}", file=sys.stderr)
        return 1

    write_feature_spec(args.spec_out)
    print(f"Feature specification written to: {args.spec_out}")

    result = validate_against_full_dataset(args.csv, chunksize=args.chunksize)
    print(f"Total rows processed        : {result['total_rows']:,}")
    print(f"Columns produced             : {result['column_count']}")
    print(f"isFraud present in output    : {result['isFraud_present']}")
    print(f"orig_balance_mismatch count  : {result['orig_balance_mismatch_count']:,}")
    print(f"Zero-amount rows             : {result['zero_amount_rows']:,}")
    print(f"Zero orig-balance rows       : {result['zero_orig_balance_rows']:,}")
    print(f"Columns with any NaN         : {list(result['nan_counts_nonzero'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
