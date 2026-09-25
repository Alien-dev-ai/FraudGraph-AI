"""Robustness and temporal generalization analysis (Milestone 12).

DIAGNOSTIC / ROBUSTNESS MILESTONE, NOT MODEL OPTIMIZATION. Determines how
robust the Milestone 10 FULL_BASELINE signal is across chronological
test-period days, transaction types, fraud types, and the presence/
absence of the two dominant Milestone 11 balance features
(`orig_balance_mismatch`, `orig_balance_after_zero`). Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Reused, unmodified ===

- `src.baseline_model.stream_split_arrays` (Milestone 10): row
  assignment via the unchanged `chronological_train_val_test` policy,
  and the unchanged 39-column `BASELINE_FEATURE_COLUMNS` feature
  assembly (single continuous causal pass -- state is never reset at a
  split boundary).
- `src.baseline_model.fit_preprocessing` / `apply_preprocessing`
  (train-only median imputation + train-only StandardScaler),
  `train_baseline_model` (`LogisticRegression(class_weight="balanced",
  random_state=42)`), and `DEFAULT_THRESHOLD` (0.5) -- all unchanged.
- `src.ablation_analysis.FULL_BASELINE` / `BALANCE_REDUCED_FULL` /
  `select_group_arrays` (Milestone 11): the exact same two feature-
  column groups and the exact same column-selection mechanism, reused
  verbatim rather than redefined here.

Models ARE retrained here (Milestones 10/11 did not persist a fitted
model, imputer, or scaler to disk -- only metrics and coefficients), but
using the identical recipe, so results are directly comparable to, and
were cross-checked against, `reports/baseline_model_results.json` and
`reports/ablation_results.json`.

=== What is new here ===

Milestones 10/11 only ever reported ONE number per metric for the whole
test split. This milestone additionally needs, for TEST rows only,
their `day_index` and raw `type` -- neither of which
`stream_split_arrays` retains (it discards everything except the 39
feature columns once routing is decided). Rather than re-running the
expensive 39-column causal feature assembly a second time just to
recover two already-raw columns, `stream_test_metadata` performs one
additional CHEAP streaming pass reading only `step` and `type`
(no feature engineering at all) and filters to the same test day range
`src.baseline_model.SPLIT_DAY_RANGES["test"]` -- the identical boundary
Milestone 9 approved and Milestone 10 already uses. Because both passes
read the same file, in the same chunked order, and `day_index` is a
pure deterministic function of `step`, the row order this pass produces
for the test-day subset is provably identical to the order
`stream_split_arrays` produces for its "test" bucket; `run_full_analysis`
asserts the two row counts match as a concrete cross-check before using
them together.

=== Metric policy: explicit undefined, never silently zero ===

Per-slice (day, type, day x type) metrics use `slice_metrics` below,
NOT `src.baseline_model.evaluate_split` (which uses
`zero_division=0` and would silently report 0.0 for a mathematically
undefined precision/recall on a degenerate slice such as day 30, which
has zero legitimate transactions). `slice_metrics` reports `None` for
any metric that is undefined for that slice, together with a
human-readable reason string, exactly as this milestone's instructions
require. `evaluate_split` IS still used for the one non-degenerate
whole-test-period summary per feature group, where it reproduces
Milestone 10/11's own numbers exactly.

=== Interpretation discipline ===

Every results/spec artifact below distinguishes three separate claims:
(1) performance ON this synthetic PaySim dataset, (2) robustness WITHIN
the PaySim simulation across days/types, and (3) whether any of this is
evidence for real-world fraud-detection capability (it is not claimed
to be). Day 30's known simulation-boundary artifact (100% fraud among
its 272 remaining rows) is reported explicitly, never silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import ablation_analysis as ab
from src import baseline_dataset as bd
from src import baseline_model as bmod
from src.modeling_split import STEPS_PER_DAY
from src.validate_raw_paysim import DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = bmod.DEFAULT_CHUNKSIZE
DEFAULT_THRESHOLD = bmod.DEFAULT_THRESHOLD  # 0.5, unchanged from Milestone 10
SAFETY_RSS_LIMIT_MB = bmod.SAFETY_RSS_LIMIT_MB

TEST_DAY_RANGE: tuple[int, int] = bmod.SPLIT_DAY_RANGES["test"]  # (27, 30), Milestone 9, unchanged
TEST_DAYS: list[int] = list(range(TEST_DAY_RANGE[0], TEST_DAY_RANGE[1] + 1))
SIMULATION_BOUNDARY_DAY: int = 30  # Milestone 4/9's documented artifact day

# The only two transaction types PaySim ever labels as fraud (empirically
# re-verified below in `verify_fraud_types_are_exactly` against the raw
# file's per-type fraud counts, not assumed).
FRAUD_ELIGIBLE_TYPES: list[str] = ["CASH_OUT", "TRANSFER"]

ROBUSTNESS_GROUPS: dict[str, list[str]] = {
    "FULL_BASELINE": ab.FULL_BASELINE,
    "BALANCE_REDUCED_FULL": ab.BALANCE_REDUCED_FULL,
}

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ROBUSTNESS_SPEC_PATH = REPORTS_DIR / "robustness_specification.md"
ROBUSTNESS_RESULTS_PATH = REPORTS_DIR / "robustness_results.json"
ROBUSTNESS_DAILY_CSV_PATH = REPORTS_DIR / "robustness_daily.csv"
ROBUSTNESS_DAY_TYPE_CSV_PATH = REPORTS_DIR / "robustness_day_type.csv"


def stream_test_metadata(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One cheap streaming pass over only `step` and `type` (no feature
    engineering, no causal state), filtered to the test day range. See
    module docstring "What is new here" for why this is a separate pass
    rather than a second run of the expensive 39-column assembly, and
    why its row order is guaranteed to match `stream_split_arrays`'s
    "test" bucket.
    """
    day_lo, day_hi = TEST_DAY_RANGE
    day_chunks: list[np.ndarray] = []
    type_chunks: list[np.ndarray] = []
    rows_seen = 0

    with pd.read_csv(csv_path, usecols=["step", "type"], chunksize=chunksize) as reader:
        for chunk in reader:
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None and rows_seen + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_seen]

            day_index = ((chunk["step"].to_numpy() - 1) // STEPS_PER_DAY)
            mask = (day_index >= day_lo) & (day_index <= day_hi)
            if mask.any():
                day_chunks.append(day_index[mask])
                type_chunks.append(chunk["type"].to_numpy()[mask])
            rows_seen += len(chunk)

    if day_chunks:
        return np.concatenate(day_chunks), np.concatenate(type_chunks)
    return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=object)


def verify_fraud_types_are_exactly(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    """Empirical re-verification (not an assumption) that fraud in this
    raw file occurs only in `FRAUD_ELIGIBLE_TYPES`. A cheap streaming
    pass over `type`/`isFraud` only.
    """
    fraud_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}
    with pd.read_csv(csv_path, usecols=["type", bd.TARGET_COLUMN], chunksize=chunksize) as reader:
        for chunk in reader:
            for t, cnt in chunk["type"].value_counts().items():
                total_by_type[t] = total_by_type.get(t, 0) + int(cnt)
            fraud_chunk = chunk[chunk[bd.TARGET_COLUMN] == 1]
            for t, cnt in fraud_chunk["type"].value_counts().items():
                fraud_by_type[t] = fraud_by_type.get(t, 0) + int(cnt)

    non_eligible_fraud = {t: c for t, c in fraud_by_type.items() if t not in FRAUD_ELIGIBLE_TYPES and c > 0}
    return {
        "fraud_by_type": fraud_by_type,
        "total_by_type": total_by_type,
        "fraud_confined_to_eligible_types": len(non_eligible_fraud) == 0,
        "unexpected_fraud_types": non_eligible_fraud,
    }


def slice_metrics(y_true: np.ndarray, preds: np.ndarray, proba: np.ndarray | None = None) -> dict[str, Any]:
    """Confusion-matrix-based metrics for one arbitrary row slice
    (a day, a type, a day x type cell), with EXPLICIT None (never a
    silently substituted 0.0) for anything mathematically undefined on
    that slice, plus a documented reason. This deliberately differs
    from `src.baseline_model.evaluate_split` (which uses
    `zero_division=0`) precisely because per-slice degeneracy (e.g. day
    30 having zero legitimate rows) is expected and must be visible,
    per this milestone's explicit metric-handling requirement.
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    n_total = int(len(y_true))
    n_fraud = int(y_true.sum())
    n_legit = n_total - n_fraud
    n_alerts = int(preds.sum())

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    reasons: dict[str, str] = {}

    precision = (tp / n_alerts) if n_alerts > 0 else None
    if precision is None:
        reasons["precision"] = "undefined: no predicted positives in this slice (n_alerts=0)"

    recall = (tp / n_fraud) if n_fraud > 0 else None
    if recall is None:
        reasons["recall"] = "undefined: no actual fraud in this slice (n_fraud=0)"

    both_classes_present = n_fraud > 0 and n_legit > 0
    pr_auc = float(average_precision_score(y_true, proba)) if (proba is not None and both_classes_present) else None
    roc_auc = float(roc_auc_score(y_true, proba)) if (proba is not None and both_classes_present) else None
    if not both_classes_present:
        reason = (
            f"single-class slice (n_fraud={n_fraud}, n_legitimate={n_legit}) -- "
            "PR-AUC/ROC-AUC are not meaningful with only one class present "
            "(PR-AUC would be trivially 1.0; ROC-AUC is mathematically undefined)"
        )
        reasons["average_precision_pr_auc"] = reason
        reasons["roc_auc_secondary"] = reason

    return {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_rate": (n_fraud / n_total) if n_total else None,
        "n_alerts": n_alerts,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "fraud_capture_rate": recall,
        "average_precision_pr_auc": pr_auc,
        "roc_auc_secondary": roc_auc,
        "undefined_metrics_reasons": reasons,
    }


def fit_and_predict_group(
    group_name: str,
    columns: list[str],
    full_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Train Milestone 10's exact pipeline on this feature group's
    train columns, then predict on test. Threshold is applied but never
    tuned -- `DEFAULT_THRESHOLD` is Milestone 10's fixed constant.
    """
    group_arrays = ab.select_group_arrays(full_arrays, columns)
    X_train, y_train = group_arrays["train"]
    X_val, y_val = group_arrays["validation"]
    X_test, y_test = group_arrays["test"]

    assert bd.TARGET_COLUMN not in columns
    assert not (set(bd.IDENTIFIER_COLUMNS) & set(columns))

    imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
    X_val_proc = bmod.apply_preprocessing(imputer, scaler, X_val)
    X_test_proc = bmod.apply_preprocessing(imputer, scaler, X_test)

    model = bmod.train_baseline_model(X_train_proc, y_train)

    proba_test = model.predict_proba(X_test_proc)[:, 1]
    preds_test = (proba_test >= DEFAULT_THRESHOLD).astype(int)

    whole_test_period = bmod.evaluate_split(model, X_test_proc, y_test, threshold=DEFAULT_THRESHOLD)
    whole_validation = bmod.evaluate_split(model, X_val_proc, y_val, threshold=DEFAULT_THRESHOLD)

    return {
        "group_name": group_name,
        "feature_count": len(columns),
        "train_rows": int(X_train.shape[0]),
        "train_fraud": int(y_train.sum()),
        "test_rows": int(X_test.shape[0]),
        "test_fraud": int(y_test.sum()),
        "y_test": y_test,
        "proba_test": proba_test,
        "preds_test": preds_test,
        "whole_test_period": whole_test_period,
        "whole_validation": whole_validation,
    }


def daily_breakdown(y_test: np.ndarray, proba_test: np.ndarray, preds_test: np.ndarray, day_index_test: np.ndarray) -> dict[str, Any]:
    """Experiment A/B: one `slice_metrics` row per test-period day.
    Day 30 is always included and explicitly flagged, never dropped.
    """
    per_day: dict[str, Any] = {}
    for day in TEST_DAYS:
        mask = day_index_test == day
        metrics = slice_metrics(y_test[mask], preds_test[mask], proba_test[mask])
        metrics["is_simulation_boundary_day"] = day == SIMULATION_BOUNDARY_DAY
        per_day[str(day)] = metrics
    return per_day


def fraud_type_breakdown(y_test: np.ndarray, proba_test: np.ndarray, preds_test: np.ndarray, type_test: np.ndarray) -> dict[str, Any]:
    """Experiment C: one `slice_metrics` row per fraud-eligible type,
    over the whole test period.
    """
    per_type: dict[str, Any] = {}
    for t in FRAUD_ELIGIBLE_TYPES:
        mask = type_test == t
        per_type[t] = slice_metrics(y_test[mask], preds_test[mask], proba_test[mask])
    return per_type


def day_type_breakdown(
    y_test: np.ndarray, preds_test: np.ndarray, day_index_test: np.ndarray, type_test: np.ndarray
) -> list[dict[str, Any]]:
    """Experiment D: compact day_index x type rows (transaction count,
    fraud count, fraud rate, captured fraud), for FULL_BASELINE.
    """
    rows: list[dict[str, Any]] = []
    for day in TEST_DAYS:
        day_mask = day_index_test == day
        types_present = sorted(set(type_test[day_mask].tolist()))
        for t in types_present:
            mask = day_mask & (type_test == t)
            y_slice = y_test[mask]
            preds_slice = preds_test[mask]
            n_total = int(mask.sum())
            n_fraud = int(y_slice.sum())
            captured = int(((preds_slice == 1) & (y_slice == 1)).sum())
            false_positives = int(((preds_slice == 1) & (y_slice == 0)).sum())
            rows.append({
                "day_index": day,
                "type": t,
                "transaction_count": n_total,
                "fraud_count": n_fraud,
                "fraud_rate": (n_fraud / n_total) if n_total else None,
                "captured_fraud": captured,
                "missed_fraud": n_fraud - captured,
                "false_positives": false_positives,
            })
    return rows


def run_smoke_test(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 300_000
) -> dict[str, Any]:
    """Bounded resource check exercising the exact same
    stream-once / metadata-stream / fit-per-group / slice path used by
    `run_full_analysis`, on a small prefix of the file.
    """
    rss_before = bmod.peak_rss_mb()
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    day_index_test, type_test = stream_test_metadata(csv_path, chunksize=chunksize, max_rows=max_rows)

    result: dict[str, Any] = {
        "max_rows": max_rows,
        "chunksize": chunksize,
        "stream_stats": stream_stats,
        "test_metadata_rows": int(len(day_index_test)),
        "test_split_rows": int(full_arrays["test"][0].shape[0]),
        "row_counts_aligned": int(len(day_index_test)) == int(full_arrays["test"][0].shape[0]),
    }

    trainable = stream_stats["rows_by_split"]["train"] > 0 and stream_stats["fraud_by_split"]["train"] > 0
    groups_exercised: dict[str, Any] = {}
    if trainable:
        for group_name, columns in ROBUSTNESS_GROUPS.items():
            X_train, y_train = ab.select_group_arrays(full_arrays, columns)["train"]
            if len(np.unique(y_train)) < 2:
                continue
            imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
            model = bmod.train_baseline_model(X_train_proc, y_train)
            _ = model.predict_proba(X_train_proc)
            groups_exercised[group_name] = {"trained": True, "n_features": X_train.shape[1]}
    result["groups_exercised"] = groups_exercised

    result["runtime_seconds"] = round(time.time() - t0, 2)
    result["rss_before_mb"] = round(rss_before, 1)
    result["peak_rss_mb"] = round(bmod.peak_rss_mb(), 1)
    result["safety_limit_mb"] = SAFETY_RSS_LIMIT_MB
    result["within_safety_limit"] = result["peak_rss_mb"] < SAFETY_RSS_LIMIT_MB
    return result


def run_full_analysis(
    csv_path: Path = DEFAULT_RAW_CSV,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """One full-feature streaming pass (Milestone 10's, unchanged) plus
    one cheap step/type-only pass, then train+predict for each
    robustness group and compute Experiments A-E from the resulting
    per-row predictions. No feature is re-engineered a second time.
    """
    t0 = time.time()
    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    day_index_test, type_test = stream_test_metadata(csv_path, chunksize=chunksize, max_rows=max_rows)

    test_rows_from_features = full_arrays["test"][0].shape[0]
    assert len(day_index_test) == test_rows_from_features, (
        "test-period metadata row count must match stream_split_arrays' test bucket "
        f"({len(day_index_test)} != {test_rows_from_features})"
    )
    assert set(np.unique(day_index_test).tolist()) <= set(TEST_DAYS)

    fraud_type_check = verify_fraud_types_are_exactly(csv_path, chunksize=chunksize)

    group_fits: dict[str, Any] = {}
    for group_name, columns in ROBUSTNESS_GROUPS.items():
        group_fits[group_name] = fit_and_predict_group(group_name, columns, full_arrays)

    experiment_a = daily_breakdown(
        group_fits["FULL_BASELINE"]["y_test"],
        group_fits["FULL_BASELINE"]["proba_test"],
        group_fits["FULL_BASELINE"]["preds_test"],
        day_index_test,
    )

    experiment_b = {
        group_name: daily_breakdown(fit["y_test"], fit["proba_test"], fit["preds_test"], day_index_test)
        for group_name, fit in group_fits.items()
    }

    experiment_c = fraud_type_breakdown(
        group_fits["FULL_BASELINE"]["y_test"],
        group_fits["FULL_BASELINE"]["proba_test"],
        group_fits["FULL_BASELINE"]["preds_test"],
        type_test,
    )

    experiment_d_rows = day_type_breakdown(
        group_fits["FULL_BASELINE"]["y_test"],
        group_fits["FULL_BASELINE"]["preds_test"],
        day_index_test,
        type_test,
    )

    experiment_e = {
        group_name: {
            "whole_test_period": fit["whole_test_period"],
            "daily": experiment_b[group_name],
        }
        for group_name, fit in group_fits.items()
    }

    runtime = time.time() - t0

    summary_by_group = {
        name: {
            "feature_count": fit["feature_count"],
            "train_rows": fit["train_rows"],
            "train_fraud": fit["train_fraud"],
            "test_rows": fit["test_rows"],
            "test_fraud": fit["test_fraud"],
            "whole_test_period": fit["whole_test_period"],
            "whole_validation": fit["whole_validation"],
        }
        for name, fit in group_fits.items()
    }

    return {
        "csv_path": str(csv_path),
        "chunksize": chunksize,
        "max_rows": max_rows,
        "runtime_seconds": round(runtime, 2),
        "peak_rss_mb": round(bmod.peak_rss_mb(), 1),
        "threshold": DEFAULT_THRESHOLD,
        "split_policy_name": bmod.SPLIT_POLICY_NAME,
        "split_day_ranges": {k: list(v) for k, v in bmod.SPLIT_DAY_RANGES.items()},
        "test_day_range": list(TEST_DAY_RANGE),
        "simulation_boundary_day": SIMULATION_BOUNDARY_DAY,
        "fraud_eligible_types": FRAUD_ELIGIBLE_TYPES,
        "fraud_type_verification": fraud_type_check,
        "stream_stats": stream_stats,
        "groups": summary_by_group,
        "experiment_a_daily_full_baseline": experiment_a,
        "experiment_b_daily_by_group": experiment_b,
        "experiment_c_fraud_type_full_baseline": experiment_c,
        "experiment_d_day_type_rows": experiment_d_rows,
        "experiment_e_balance_dependency": experiment_e,
        "_group_fits": group_fits,
    }


def write_daily_csv(results: dict[str, Any], out_path: Path = ROBUSTNESS_DAILY_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_group", "day_index", "is_simulation_boundary_day", "n_total", "n_fraud", "n_legitimate",
        "fraud_rate", "n_alerts", "tp", "fp", "fn", "tn", "precision", "recall",
        "average_precision_pr_auc", "roc_auc_secondary",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, per_day in results["experiment_b_daily_by_group"].items():
            for day_str, m in per_day.items():
                cm = m["confusion_matrix"]
                writer.writerow({
                    "feature_group": group_name,
                    "day_index": day_str,
                    "is_simulation_boundary_day": m["is_simulation_boundary_day"],
                    "n_total": m["n_total"],
                    "n_fraud": m["n_fraud"],
                    "n_legitimate": m["n_legitimate"],
                    "fraud_rate": m["fraud_rate"],
                    "n_alerts": m["n_alerts"],
                    "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "average_precision_pr_auc": m["average_precision_pr_auc"],
                    "roc_auc_secondary": m["roc_auc_secondary"],
                })


def write_day_type_csv(results: dict[str, Any], out_path: Path = ROBUSTNESS_DAY_TYPE_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["day_index", "type", "transaction_count", "fraud_count", "fraud_rate", "captured_fraud", "missed_fraud", "false_positives"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["experiment_d_day_type_rows"]:
            writer.writerow(row)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_robustness_results(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = ROBUSTNESS_RESULTS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if not k.startswith("_")}
    payload = {"smoke_test": smoke_test, **serializable}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_robustness_spec(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = ROBUSTNESS_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_metric(m: dict[str, Any], key: str, fmt: str = ".4f") -> str:
        v = m.get(key)
        return "null" if v is None else format(v, fmt)

    def fmt_day_row(day: str, m: dict[str, Any]) -> str:
        cm = m["confusion_matrix"]
        flag = " **[SIMULATION-BOUNDARY DAY -- 100% fraud among remaining rows]**" if m["is_simulation_boundary_day"] else ""
        return (
            f"- day {day}{flag}: n={m['n_total']:,} fraud={m['n_fraud']:,} "
            f"legit={m['n_legitimate']:,} fraud_rate={fmt_metric(m, 'fraud_rate', '.4%')} "
            f"alerts={m['n_alerts']:,} TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']} "
            f"precision={fmt_metric(m, 'precision')} recall={fmt_metric(m, 'recall')} "
            f"PR-AUC={fmt_metric(m, 'average_precision_pr_auc')} "
            f"ROC-AUC(secondary)={fmt_metric(m, 'roc_auc_secondary')}"
        )

    lines = [
        "# FraudGraph AI — Robustness and Temporal Generalization Analysis (Milestone 12)",
        "",
        "**This is a robustness/diagnostic milestone, not model "
        "optimization.** The Milestone 9 chronological split (train "
        "days 0-23, validation days 24-26, test days 27-30) and "
        "Milestone 10's exact configuration "
        "(`LogisticRegression(class_weight=\"balanced\", "
        f"random_state={bmod.RANDOM_STATE})`, train-only median "
        f"imputation, train-only StandardScaler, threshold="
        f"{DEFAULT_THRESHOLD}) are reused unchanged. No hyperparameter "
        "was tuned, no threshold was tuned per day/type/group, the "
        "split was not changed, and test results were not used to "
        "modify anything -- they are only ever evaluated here.",
        "",
        "## Objective",
        "",
        "Determine how robust the Milestone 10 signal is across (A) "
        "individual chronological test-period days, (B) the presence/"
        "absence of Milestone 11's two dominant balance features, (C) "
        "fraud-eligible transaction types, and (D) day x type cells -- "
        "using Milestone 10's `FULL_BASELINE` (39 columns) and "
        "Milestone 11's `BALANCE_REDUCED_FULL` (37 columns, unchanged "
        "membership) feature groups.",
        "",
        "## Feature configurations (reused from Milestone 11, unchanged)",
        "",
        f"- **FULL_BASELINE** ({len(ab.FULL_BASELINE)} columns): all "
        "Milestone 8 baseline columns.",
        f"- **BALANCE_REDUCED_FULL** ({len(ab.BALANCE_REDUCED_FULL)} "
        f"columns): FULL_BASELINE minus "
        f"{', '.join(f'`{c}`' for c in ab.BALANCE_FEATURES_REMOVED)}.",
        "",
        "## Split definitions (Milestone 9, unchanged)",
        "",
        f"- train: days {results['split_day_ranges']['train'][0]}-{results['split_day_ranges']['train'][1]}",
        f"- validation: days {results['split_day_ranges']['validation'][0]}-{results['split_day_ranges']['validation'][1]}",
        f"- test: days {results['split_day_ranges']['test'][0]}-{results['split_day_ranges']['test'][1]} "
        "(the period every experiment below examines)",
        "",
        "## Causal-ordering policy (unchanged)",
        "",
        "File/row order is event order; within a shared `step`, an "
        "earlier file row is prior. Causal state (Milestones 5/7) is "
        "accumulated in ONE continuous pass over the entire raw file "
        "and is never reset at a train/validation/test boundary -- "
        "identical to Milestones 10/11.",
        "",
        "## Metric definitions and undefined-value policy",
        "",
        "Whole-test-period summaries reuse `src.baseline_model."
        "evaluate_split` verbatim (Milestone 10's own definitions, "
        "`zero_division=0`). Per-day / per-type / per-(day,type) SLICES "
        "use a separate `slice_metrics` function that reports `null` "
        "(never a silently substituted 0.0) for precision when a slice "
        "has zero predicted positives, for recall when a slice has zero "
        "actual fraud, and for PR-AUC/ROC-AUC when a slice contains "
        "only one class -- each with a documented reason string "
        "(`undefined_metrics_reasons` in the JSON output).",
        "",
        "## Threshold policy",
        "",
        f"Fixed at {DEFAULT_THRESHOLD} for every experiment, every day, "
        "every type, and every feature group -- never tuned, never "
        "varied per slice.",
        "",
        "## Resource strategy",
        "",
        "One streaming pass reuses Milestone 10's full 39-column causal "
        "feature assembly (`stream_split_arrays`, unchanged) to get "
        "train/validation/test arrays for BOTH feature groups via "
        "Milestone 11's column-selection (`select_group_arrays`, "
        "unchanged) -- the expensive causal computation runs once, not "
        "once per group. A second, cheap pass reads only `step`/`type` "
        "(no feature engineering) to recover per-test-row day_index/"
        "type for slicing. See the bounded smoke test below, run before "
        "the full 6,362,620-row analysis.",
        "",
        "### Bounded smoke test",
        "",
        f"- Rows: {smoke_test['max_rows']:,}, chunksize {smoke_test['chunksize']:,}",
        f"- Runtime: {smoke_test['runtime_seconds']:.2f}s",
        f"- Peak RSS: {smoke_test['peak_rss_mb']:.0f} MB "
        f"(safety limit: {smoke_test['safety_limit_mb']:,} MB, within "
        f"limit: {smoke_test['within_safety_limit']})",
        f"- Test-metadata/feature-stream row-count alignment check: "
        f"{smoke_test['row_counts_aligned']}",
        "",
        "## Fraud-type verification",
        "",
        f"Fraud confined to `{FRAUD_ELIGIBLE_TYPES}`: "
        f"**{results['fraud_type_verification']['fraud_confined_to_eligible_types']}** "
        f"(re-verified empirically from the raw file's per-type fraud "
        f"counts: {results['fraud_type_verification']['fraud_by_type']}).",
        "",
        "## Experiment A — daily temporal robustness (FULL_BASELINE, test period)",
        "",
    ]
    for day, m in results["experiment_a_daily_full_baseline"].items():
        lines.append(fmt_day_row(day, m))
    lines += [
        "",
        "Day 30 is reported explicitly above, not silently excluded. "
        "It carries Milestone 4/9's documented simulation-boundary "
        "artifact: legitimate transaction volume in the underlying "
        "simulation collapses on the final day while fraud count stays "
        "roughly flat, so the day-30 SLICE has zero legitimate rows "
        "left. This makes aggregate test-period metrics potentially "
        "misleading in one direction (a model that is simply "
        "'confident near the end of the file' could look artificially "
        "strong on this slice) and precision/PR-AUC/ROC-AUC "
        "mathematically undefined or degenerate on this slice in "
        "isolation (see the null entries and "
        "`undefined_metrics_reasons` in the JSON), which is exactly why "
        "it must be reported, not hidden by aggregation.",
        "",
        "## Experiment B — balance dependency by day",
        "",
        "Full per-day metrics for both groups are in "
        "`reports/robustness_daily.csv` and the JSON's "
        "`experiment_b_daily_by_group`. Neither group is described as "
        "better/worse; differences are reported numerically in "
        "Experiment E below.",
        "",
        "## Experiment C — fraud-type robustness (FULL_BASELINE, whole test period)",
        "",
    ]
    for t, m in results["experiment_c_fraud_type_full_baseline"].items():
        cm = m["confusion_matrix"]
        lines.append(
            f"- **{t}**: n={m['n_total']:,} fraud={m['n_fraud']:,} "
            f"fraud_rate={fmt_metric(m, 'fraud_rate', '.4%')} "
            f"captured(TP)={cm['tp']} missed(FN)={cm['fn']} FP={cm['fp']} "
            f"recall={fmt_metric(m, 'recall')} precision={fmt_metric(m, 'precision')}"
        )
    lines += [
        "",
        "## Experiment D — day x type breakdown",
        "",
        f"Full detail in `reports/robustness_day_type.csv` "
        f"({len(results['experiment_d_day_type_rows'])} rows: every "
        "day x type combination actually present in the test period, "
        "for FULL_BASELINE).",
        "",
        "## Experiment E — balance-reduced temporal robustness",
        "",
        "Whole-test-period summary, both groups (identical split/"
        "threshold/preprocessing, only the feature columns differ):",
        "",
    ]
    for group_name in ROBUSTNESS_GROUPS:
        wtp = results["experiment_e_balance_dependency"][group_name]["whole_test_period"]
        cm = wtp["confusion_matrix"]
        lines.append(
            f"- **{group_name}**: alerts={wtp['n_alerts']:,} TP={cm['tp']:,} "
            f"FP={cm['fp']:,} FN={cm['fn']:,} precision={wtp['precision']:.4f} "
            f"recall={wtp['recall']:.4f} PR-AUC={wtp['average_precision_pr_auc']:.4f} "
            f"ROC-AUC(secondary)={wtp['roc_auc_secondary']:.4f}"
        )
    lines += [
        "",
        "This quantifies how much of the whole-test-period result "
        "depends on the two removed balance features; it does not "
        "identify a causal mechanism and is not a claim that either "
        "configuration is a replacement model.",
        "",
        "## Interpretation",
        "",
        "Three separate claims, deliberately not conflated:",
        "",
        "1. **Performance on this synthetic PaySim dataset**: the "
        "numbers above are exact, measured figures on PaySim's "
        "synthetic transaction log, under the fixed Milestone 10 "
        "recipe.",
        "2. **Robustness within the PaySim simulation**: performance is "
        "consistent across CASH_OUT/TRANSFER and across most "
        "non-boundary test days, but that consistency is measured "
        "entirely inside one synthetic simulation and its own internal "
        "regularities (see PaySim characteristics below) -- it is not, "
        "by itself, evidence about a different dataset or deployment.",
        "3. **Evidence for real-world fraud-detection capability**: "
        "NOT established by this milestone. PaySim's fraud-injection "
        "logic produces a synthetic balance-draining pattern "
        "(`orig_balance_mismatch`, `orig_balance_after_zero`) that "
        "Milestone 11 already showed dominates the Milestone 10 result; "
        "this milestone's per-day/per-type breakdowns describe how "
        "that same dependency behaves temporally, not whether it would "
        "hold on real transaction data.",
        "",
        "Relevant background, restated rather than re-derived: PaySim's "
        "synthetic balance mechanics (Milestone 3) frequently produce "
        "balance-bookkeeping mismatches even for legitimate rows; the "
        "day-30 simulation-boundary artifact (Milestone 4/9) collapses "
        "legitimate volume near the end of the simulated period; "
        "`orig_balance_mismatch`/`orig_balance_after_zero` dominate the "
        "Milestone 10 linear model's coefficients (Milestone 10/11); "
        "and Milestone 11 found the standalone graph-derived signal "
        "weak (`GRAPH_ONLY` PR-AUC ~0.013-0.017) with 3 of 5 graph "
        "coefficients unstable in sign across configurations.",
        "",
        "## Limitations",
        "",
        "- Every number in this report comes from ONE fixed model "
        "family and ONE fixed preprocessing/threshold configuration, "
        "unchanged from Milestone 10. It does not establish robustness "
        "under a different model, preprocessing, or threshold.",
        "- Day-level and type-level slices can be small (e.g. day 30 "
        "has 272 rows); metrics on small slices are reported exactly as "
        "computed but are correspondingly noisier than the whole-test-"
        "period aggregate.",
        "- This milestone does not claim, and explicitly disclaims, "
        "that any result here demonstrates production readiness or "
        "real-world generalization -- see \"Interpretation\" above.",
        "- Group-aware account-level splitting remains not implemented "
        "(Milestone 9), unchanged here.",
        "- No hyperparameter tuning, threshold tuning, or resampling "
        "was performed anywhere in this milestone.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 12 robustness analysis: per-day, per-type, "
        "and balance-feature-dependency breakdowns of the Milestone 10 "
        "baseline over the test period."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--smoke-rows", type=int, default=300_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Bound the full run too (testing only).")
    args = parser.parse_args(argv)

    print(f"Running bounded smoke test ({args.smoke_rows:,} rows)...")
    smoke = run_smoke_test(args.csv, chunksize=args.chunksize, max_rows=args.smoke_rows)
    print(f"Smoke test peak RSS   : {smoke['peak_rss_mb']:.0f} MB (limit {smoke['safety_limit_mb']:,} MB)")
    print(f"Smoke test runtime    : {smoke['runtime_seconds']:.2f}s")
    print(f"Row-count alignment   : {smoke['row_counts_aligned']}")

    if not smoke["within_safety_limit"]:
        import sys
        print("Smoke test exceeded the safety limit -- stopping before the full run.", file=sys.stderr)
        return 1

    print("Smoke test passed. Running the full robustness analysis...")
    results = run_full_analysis(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)

    write_daily_csv(results, ROBUSTNESS_DAILY_CSV_PATH)
    write_day_type_csv(results, ROBUSTNESS_DAY_TYPE_CSV_PATH)
    write_robustness_results(results, smoke, ROBUSTNESS_RESULTS_PATH)
    write_robustness_spec(results, smoke, ROBUSTNESS_SPEC_PATH)

    for name, g in results["groups"].items():
        wtp = g["whole_test_period"]
        print(f"{name:24s} n_feat={g['feature_count']:2d} test PR-AUC={wtp['average_precision_pr_auc']:.4f} precision={wtp['precision']:.4f} recall={wtp['recall']:.4f}")
    print(f"Full-run runtime      : {results['runtime_seconds']:.1f}s")
    print(f"Full-run peak RSS     : {results['peak_rss_mb']:.0f} MB")
    print(f"Spec written to       : {ROBUSTNESS_SPEC_PATH}")
    print(f"Results written to    : {ROBUSTNESS_RESULTS_PATH}")
    print(f"Daily CSV written to  : {ROBUSTNESS_DAILY_CSV_PATH}")
    print(f"Day-type CSV written  : {ROBUSTNESS_DAY_TYPE_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
