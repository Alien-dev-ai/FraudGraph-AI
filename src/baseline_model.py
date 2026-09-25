"""Baseline classifier training and evaluation for raw PaySim (Milestone 10).

FIRST MODELING MILESTONE. The goal is a trustworthy, explainable BASELINE,
not optimized performance. This module deliberately does NOT: hyperparameter
tune, tune a decision threshold against any evaluation split, oversample,
undersample, apply SMOTE, train multiple competing model families, or use
the test split for any model or preprocessing decision. Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Reused, unmodified, approved artifacts ===

- `src.baseline_dataset.iter_baseline_chunks` (Milestone 8): the exact
  39-column `BASELINE_FEATURE_COLUMNS` contract, assembled via Milestone
  5's and 7's already-tested causal per-chunk functions. This module
  does not reimplement or alter any feature computation.
- `src.modeling_split.CANDIDATE_POLICIES["chronological_train_val_test"]`
  (Milestone 9): the approved day-index boundaries (train: days 0-23,
  validation: days 24-26, test: days 27-30). This module does not
  redefine or adjust these boundaries; if they were ever infeasible to
  honor, this module would stop and report rather than silently
  substitute a different split.

`isFraud` is read exactly once per row, in a second bounded streaming
pass over only that one column (mirroring the read pattern already used
by `src.modeling_split`'s own label-audit columns), and is used ONLY as
the target array `y` — it is never joined into, or made reachable from,
the feature matrix `X`.

=== Causal integrity across the split ===

The whole raw file is streamed in a SINGLE continuous forward pass to
compute causal features (Milestones 5 and 7's running per-account /
per-pair state). Rows are routed into train/validation/test by
`day_index` only AFTER their features have been computed from that one
continuous pass -- the running causal state is never reset at a split
boundary. This is intentional and correct: it mirrors what a real
deployed system's running history would look like (a live system does
not forget everything it has seen when a reporting period ends), and it
means no validation/test row's causal features are computed from a
"training-only" state that a production system would not have had
access to at prediction time either. See
`reports/modeling_split_specification.md` section E ("Compatibility with
causal graph features") for the same argument made at the design stage.

=== Resource-aware assembly (no full-size pandas DataFrame) ===

`stream_split_arrays` never concatenates the raw per-chunk pandas
DataFrames it reads. For each chunk it immediately:
  1. slices out only the 39 fixed BASELINE_FEATURE_COLUMNS as a float32
     numpy array (dropping identifiers, audit-only graph columns, and
     the raw chunk itself),
  2. routes each row's float32 feature vector and label into one of
     three per-split Python lists of small arrays,
and drops the chunk. Only at the very end are each split's list of
small arrays concatenated into one final array. Peak memory is
bounded by (a) one raw chunk at a time and (b) the three growing lists
of float32 slices, which together hold no more data than the final
three arrays themselves -- never a fourth, fully-assembled, all-columns
pandas copy. `run_smoke_test` measures actual peak RSS on a bounded
prefix of the file before the full 6,362,620-row run is attempted; see
`reports/baseline_model_results.json` for the measured figures and
`SAFETY_RSS_LIMIT_MB` for the guard this module checks against before
proceeding to the full run.

=== Preprocessing (train-only fit) ===

`SimpleImputer(strategy="median")` and `StandardScaler()` are fit ONLY
on the train split's feature matrix; validation and test are only ever
`.transform()`-ed, never used to compute imputation medians or scaling
parameters. This follows Milestone 8's documented undefined-value
policy verbatim: NaN is preserved through feature assembly and only
resolved here, in the modeling pipeline, using train-only statistics
(`src/baseline_dataset.py` module docstring, "Undefined-value policy").

=== Model ===

A single `sklearn.linear_model.LogisticRegression` with
`class_weight="balanced"` (handles severe class imbalance via
reweighting, not resampling) and a fixed `random_state`. No
hyperparameter search is performed; `MODEL_PARAMS` below is the
complete, fixed configuration. Logistic regression is chosen for
explainability (a signed coefficient per feature, directly readable
against the known feature contract) and because it is cheap enough to
train on the full ~6.2M-row training split on CPU.

=== Evaluation ===

`DEFAULT_THRESHOLD` (0.5) and `ALERT_BUDGET_RATE` (0.01, i.e. "review
the top 1% highest-scored transactions") are both fixed constants
decided here, in the module source, independent of any split's realized
label distribution -- neither is chosen by looking at validation or
test outcomes. Validation and test are evaluated separately and never
combined into one reported number. See `evaluate_split` for the full
metric set (PR-AUC primary; ROC-AUC explicitly secondary per Milestone
9 section A's imbalance warning).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src import baseline_dataset as bd
from src.modeling_split import CANDIDATE_POLICIES
from src.validate_raw_paysim import DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = 200_000
RANDOM_STATE = 42

# Fixed a priori -- never chosen by looking at validation/test outcomes.
DEFAULT_THRESHOLD = 0.5
ALERT_BUDGET_RATE = 0.01  # "review the top 1% highest-scored transactions"

# Guard checked after the bounded smoke test, before the full run.
SAFETY_RSS_LIMIT_MB = 8_000  # this machine has 16 GB physical RAM (measured)

SPLIT_POLICY_NAME = "chronological_train_val_test"
SPLIT_DAY_RANGES: dict[str, tuple[int, int]] = CANDIDATE_POLICIES[SPLIT_POLICY_NAME]
SPLIT_NAMES = list(SPLIT_DAY_RANGES.keys())

FEATURE_COLUMNS: list[str] = bd.BASELINE_FEATURE_COLUMNS

# Defensive, load-time re-assertion of guarantees this module depends on
# but does not itself enforce (Milestones 8/9 already test these).
assert bd.TARGET_COLUMN not in FEATURE_COLUMNS, "target column must never be a model input"
assert not (set(bd.IDENTIFIER_COLUMNS) & set(FEATURE_COLUMNS)), "identifiers must never be model inputs"
assert len(FEATURE_COLUMNS) == 39, "baseline feature contract changed size unexpectedly"

# Cross-checked against Milestone 9's audited label distribution
# (reports/modeling_split_specification.md section C) -- only asserted
# when running against the full, unmodified raw file (see main()).
EXPECTED_FULL_DATASET_ROWS = 6_362_620
EXPECTED_FRAUD_BY_SPLIT = {"train": 6_373, "validation": 792, "test": 1_048}
EXPECTED_ROWS_BY_SPLIT = {"train": 6_201_194, "validation": 80_316, "test": 81_110}

MODEL_PARAMS: dict[str, Any] = dict(
    # l1_ratio=0.0 is this sklearn version's non-deprecated spelling of
    # an L2 penalty (see LogisticRegression's own deprecation notice for
    # the `penalty` argument).
    l1_ratio=0.0,
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_SPEC_PATH = REPORTS_DIR / "baseline_model_specification.md"
MODEL_RESULTS_PATH = REPORTS_DIR / "baseline_model_results.json"


def peak_rss_mb() -> float:
    """Whole-process peak resident set size so far, in MB.

    `ru_maxrss` is a monotonic high-water mark since process start
    (never decreases), reported in bytes on macOS/BSD and kilobytes on
    Linux -- see the Python `resource` docs. Comparing two calls,
    before/after a phase, approximates that phase's own peak whenever
    the phase dominates total memory use, which holds here (no other
    large allocation happens before `stream_split_arrays`/training).
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def day_index_ranges() -> dict[str, tuple[int, int]]:
    return SPLIT_DAY_RANGES


def stream_split_arrays(
    csv_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """One continuous streaming pass producing train/validation/test
    (X, y) float32/int64 numpy arrays, per the module docstring's
    "Resource-aware assembly" and "Causal integrity across the split"
    sections. Never materializes a full-size, all-columns pandas
    DataFrame; never resets causal state at a split boundary.
    """
    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"X": [], "y": []} for name in SPLIT_NAMES
    }
    rows_by_split = {name: 0 for name in SPLIT_NAMES}
    fraud_by_split = {name: 0 for name in SPLIT_NAMES}
    rows_seen = 0
    rows_outside_policy = 0

    with pd.read_csv(csv_path, usecols=[bd.TARGET_COLUMN], chunksize=chunksize) as label_reader:
        for feat_chunk in bd.iter_baseline_chunks(csv_path, chunksize=chunksize, max_rows=max_rows):
            assert bd.TARGET_COLUMN not in feat_chunk.columns

            label_chunk_full = next(label_reader)[bd.TARGET_COLUMN].to_numpy()
            n = len(feat_chunk)
            y_chunk = label_chunk_full[:n]

            day_index = feat_chunk["day_index"].to_numpy()
            X_chunk = feat_chunk[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

            assigned = np.zeros(n, dtype=bool)
            for name, (d0, d1) in SPLIT_DAY_RANGES.items():
                mask = (day_index >= d0) & (day_index <= d1)
                if mask.any():
                    buffers[name]["X"].append(X_chunk[mask])
                    buffers[name]["y"].append(y_chunk[mask])
                    rows_by_split[name] += int(mask.sum())
                    fraud_by_split[name] += int(y_chunk[mask].sum())
                    assigned |= mask

            rows_outside_policy += int((~assigned).sum())
            rows_seen += n

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_features = len(FEATURE_COLUMNS)
    for name in SPLIT_NAMES:
        x_parts, y_parts = buffers[name]["X"], buffers[name]["y"]
        X = np.concatenate(x_parts, axis=0) if x_parts else np.empty((0, n_features), dtype=np.float32)
        y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.int64)
        arrays[name] = (X, y)

    stats = {
        "rows_seen": rows_seen,
        "rows_by_split": rows_by_split,
        "fraud_by_split": fraud_by_split,
        "rows_outside_policy": rows_outside_policy,
    }
    return arrays, stats


def _replace_infinite_with_nan(X: np.ndarray) -> np.ndarray:
    """Replace +/-inf with NaN, ahead of imputation.

    Milestone 8's full-scale audit (reports/baseline_full_stream_audit.json)
    found 2 of 6,362,620 rows with +inf in `amount_vs_dest_avg_ratio`
    (dividing by a causal average small enough to overflow) -- a real,
    if extremely rare, characteristic of the assembled feature table at
    full scale, not a bug in this module. This is a deterministic,
    non-learned elementwise transform (no statistic is fit from any
    split's data), so applying it identically to train/validation/test
    introduces no leakage -- it only lets the train-only-fit
    `SimpleImputer` treat "infinite" the same way it already treats
    "missing".
    """
    return np.where(np.isfinite(X), X, np.nan)


def fit_preprocessing(X_train: np.ndarray) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    """Fit imputer + scaler on TRAIN ONLY. NaN handling follows
    Milestone 8's documented policy: raw features preserve NaN, and the
    imputation VALUE (train median per column) is a modeling decision
    made here, not baked into the assembled dataset.
    """
    X_train = _replace_infinite_with_nan(X_train)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return imputer, scaler, X_scaled


def apply_preprocessing(imputer: SimpleImputer, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Transform-only -- never refits on validation/test data."""
    X = _replace_infinite_with_nan(X)
    return scaler.transform(imputer.transform(X))


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the single approved baseline model. No hyperparameter
    search: `MODEL_PARAMS` is the complete, fixed configuration.
    """
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_split(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    alert_budget_rate: float = ALERT_BUDGET_RATE,
) -> dict[str, Any]:
    """Evaluate one split in isolation. `threshold` and
    `alert_budget_rate` are both fixed inputs (module-level constants
    by default) -- this function never searches over them.
    """
    proba = model.predict_proba(X)[:, 1]

    assert proba.shape[0] == y.shape[0], "prediction count must equal evaluation row count"
    assert np.isfinite(proba).all(), "predicted probabilities contain NaN/inf"
    assert proba.min() >= 0.0 and proba.max() <= 1.0, "predicted probabilities outside [0, 1]"

    preds = (proba >= threshold).astype(int)

    n_total = int(len(y))
    n_fraud = int(y.sum())
    n_legit = n_total - n_fraud
    n_alerts = int(preds.sum())

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y, proba)) if n_fraud > 0 else None
    roc_auc = float(roc_auc_score(y, proba)) if 0 < n_fraud < n_total else None
    precision = float(precision_score(y, preds, zero_division=0))
    recall = float(recall_score(y, preds, zero_division=0))
    fp_rate = float(fp / n_legit) if n_legit else None

    # Fixed alert budget: top-K by score, K decided from the budget
    # rate constant and this split's own row count only -- never from
    # this split's labels.
    k = max(1, int(round(alert_budget_rate * n_total)))
    order = np.argsort(-proba, kind="stable")
    budget_preds = np.zeros(n_total, dtype=int)
    budget_preds[order[:k]] = 1
    budget_precision = float(precision_score(y, budget_preds, zero_division=0))
    budget_recall = float(recall_score(y, budget_preds, zero_division=0))

    return {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legitimate": n_legit,
        "fraud_prevalence": (n_fraud / n_total) if n_total else None,
        "threshold": threshold,
        "n_alerts": n_alerts,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": precision,
        "recall": recall,
        "fraud_capture_rate": recall,
        "false_positive_count": int(fp),
        "false_positive_rate": fp_rate,
        "average_precision_pr_auc": pr_auc,
        "roc_auc_secondary": roc_auc,
        "alert_budget_rate": alert_budget_rate,
        "alert_budget_count": k,
        "precision_at_alert_budget": budget_precision,
        "recall_at_alert_budget": budget_recall,
    }


def feature_importance(model: LogisticRegression) -> list[dict[str, Any]]:
    """Signed coefficients paired with the fixed feature-name contract,
    sorted by absolute magnitude -- the explainability artifact this
    milestone's model choice is meant to provide.
    """
    coefs = model.coef_[0]
    ranked = sorted(zip(FEATURE_COLUMNS, coefs), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": name, "coefficient": float(c)} for name, c in ranked]


def run_smoke_test(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 300_000
) -> dict[str, Any]:
    """Bounded resource check, run BEFORE the full 6,362,620-row
    pipeline. Exercises the exact same streaming/assembly/preprocessing/
    training code path on a small prefix of the file and reports peak
    RSS, so the full run is never attempted "blind".
    """
    rss_before = peak_rss_mb()
    t0 = time.time()

    arrays, stream_stats = stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    X_train, y_train = arrays["train"]

    result: dict[str, Any] = {
        "max_rows": max_rows,
        "chunksize": chunksize,
        "stream_stats": stream_stats,
        "runtime_seconds": None,
        "peak_rss_mb": None,
        "rss_before_mb": round(rss_before, 1),
        "train_rows_available": int(X_train.shape[0]),
    }

    if X_train.shape[0] > 0 and int(y_train.sum()) > 0 and int((1 - y_train).sum()) > 0:
        imputer, scaler, X_train_proc = fit_preprocessing(X_train)
        model = train_baseline_model(X_train_proc, y_train)
        _ = model.predict_proba(X_train_proc)  # exercise the full path end-to-end
        result["model_trained_on_smoke_sample"] = True
    else:
        # Not enough rows/classes in the bounded prefix to fit a model --
        # still a valid, informative smoke test of the streaming/assembly
        # memory behavior, just not of model fitting.
        result["model_trained_on_smoke_sample"] = False

    result["runtime_seconds"] = round(time.time() - t0, 2)
    result["peak_rss_mb"] = round(peak_rss_mb(), 1)
    result["safety_limit_mb"] = SAFETY_RSS_LIMIT_MB
    result["within_safety_limit"] = result["peak_rss_mb"] < SAFETY_RSS_LIMIT_MB
    return result


def run_pipeline(
    csv_path: Path = DEFAULT_RAW_CSV,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    alert_budget_rate: float = ALERT_BUDGET_RATE,
) -> dict[str, Any]:
    """End-to-end baseline pipeline: stream+assemble -> fit preprocessing
    on train only -> train the one approved model -> evaluate validation
    and test separately. Returns everything `main()` needs to write the
    spec/results artifacts, and everything the test suite needs to
    assert the Section F sanity checks.
    """
    t0 = time.time()
    arrays, stream_stats = stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["validation"]
    X_test, y_test = arrays["test"]

    imputer, scaler, X_train_proc = fit_preprocessing(X_train)
    X_val_proc = apply_preprocessing(imputer, scaler, X_val)
    X_test_proc = apply_preprocessing(imputer, scaler, X_test)

    model = train_baseline_model(X_train_proc, y_train)

    validation_metrics = evaluate_split(model, X_val_proc, y_val, threshold=threshold, alert_budget_rate=alert_budget_rate)
    test_metrics = evaluate_split(model, X_test_proc, y_test, threshold=threshold, alert_budget_rate=alert_budget_rate)

    runtime = time.time() - t0

    return {
        "csv_path": str(csv_path),
        "chunksize": chunksize,
        "max_rows": max_rows,
        "runtime_seconds": round(runtime, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "stream_stats": stream_stats,
        "split_policy_name": SPLIT_POLICY_NAME,
        "split_day_ranges": {k: list(v) for k, v in SPLIT_DAY_RANGES.items()},
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "preprocessing": {
            "imputer": {"class": "SimpleImputer", "strategy": "median", "fit_on": "train_only"},
            "scaler": {"class": "StandardScaler", "fit_on": "train_only"},
        },
        "model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "params": MODEL_PARAMS,
            "hyperparameter_tuning_performed": False,
            "threshold_tuning_performed": False,
            "resampling_performed": False,
        },
        "train_rows": int(X_train.shape[0]),
        "train_fraud": int(y_train.sum()),
        "validation": validation_metrics,
        "test": test_metrics,
        "feature_importance": feature_importance(model),
        "_model": model,
        "_imputer": imputer,
        "_scaler": scaler,
    }


def run_sanity_checks(results: dict[str, Any], check_against_full_dataset: bool) -> dict[str, Any]:
    """Section F baseline sanity checks. Raises AssertionError on any
    violation -- these are correctness invariants, not soft warnings.
    """
    stream_stats = results["stream_stats"]
    rows_by_split = stream_stats["rows_by_split"]
    fraud_by_split = stream_stats["fraud_by_split"]

    total_rows_assigned = sum(rows_by_split.values())
    assert total_rows_assigned + stream_stats["rows_outside_policy"] == stream_stats["rows_seen"], (
        "train/validation/test row counts (plus any rows outside the policy) must sum to rows_seen"
    )
    assert stream_stats["rows_outside_policy"] == 0, (
        "chronological_train_val_test is expected to cover every day_index in the dataset "
        f"(0-30); found {stream_stats['rows_outside_policy']} uncovered rows"
    )

    for split_name in ("validation", "test"):
        metrics = results[split_name]
        assert metrics["n_total"] == rows_by_split[split_name]
        assert metrics["n_fraud"] == fraud_by_split[split_name]
        assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == metrics["n_fraud"]
        assert metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["fp"] == metrics["n_legitimate"]
        assert sum(metrics["confusion_matrix"].values()) == metrics["n_total"]

    assert results["train_rows"] == rows_by_split["train"]
    assert results["train_fraud"] == fraud_by_split["train"]

    if check_against_full_dataset:
        assert stream_stats["rows_seen"] == EXPECTED_FULL_DATASET_ROWS
        assert rows_by_split == EXPECTED_ROWS_BY_SPLIT
        assert fraud_by_split == EXPECTED_FRAUD_BY_SPLIT

    return {"passed": True, "checked_against_full_dataset_audit": check_against_full_dataset}


def write_model_spec(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = MODEL_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val, test = results["validation"], results["test"]

    def fmt_metrics(m: dict[str, Any]) -> list[str]:
        cm = m["confusion_matrix"]
        return [
            f"- Rows: {m['n_total']:,} ({m['n_fraud']:,} fraud, {m['n_legitimate']:,} legitimate, "
            f"prevalence {m['fraud_prevalence']:.4%})",
            f"- Threshold: {m['threshold']} (fixed default, not tuned on this split)",
            f"- Alerts raised: {m['n_alerts']:,}",
            f"- Confusion matrix: TP={cm['tp']:,} FP={cm['fp']:,} FN={cm['fn']:,} TN={cm['tn']:,}",
            f"- Precision: {m['precision']:.4f}  |  Recall (fraud capture rate): {m['recall']:.4f}",
            f"- False positives: {m['false_positive_count']:,} "
            f"({m['false_positive_rate']:.4%} of legitimate rows)" if m["false_positive_rate"] is not None else "- False positives: n/a",
            f"- Average Precision / PR-AUC (primary): "
            f"{m['average_precision_pr_auc']:.4f}" if m["average_precision_pr_auc"] is not None else "- PR-AUC: n/a",
            f"- ROC-AUC (secondary diagnostic only): "
            f"{m['roc_auc_secondary']:.4f}" if m["roc_auc_secondary"] is not None else "- ROC-AUC: n/a",
            f"- At a fixed {m['alert_budget_rate']:.2%} alert budget "
            f"({m['alert_budget_count']:,} alerts): precision={m['precision_at_alert_budget']:.4f}, "
            f"recall={m['recall_at_alert_budget']:.4f}",
        ]

    top_features = results["feature_importance"][:15]

    lines = [
        "# FraudGraph AI — Baseline Model Specification (Milestone 10)",
        "",
        "BASELINE ONLY. No hyperparameter optimization, no decision-"
        "threshold tuning, no oversampling/undersampling/SMOTE, no "
        "competing model families, and the test split was never used "
        "for any model, preprocessing, threshold, or budget decision. "
        "This document establishes a trustworthy first baseline, not a "
        "performance claim.",
        "",
        "## 1. Dataset / split design",
        "",
        f"Split policy: `{results['split_policy_name']}` (Milestone 9, unchanged):",
        "",
    ]
    for name, (d0, d1) in results["split_day_ranges"].items():
        lines.append(f"- **{name}**: days {d0}-{d1}")
    lines += [
        "",
        "Rows are assigned to a split by `day_index` only after the "
        "single continuous causal-feature pass over the whole file has "
        "computed their features — the running causal state is never "
        "reset at a split boundary (see module docstring, \"Causal "
        "integrity across the split\").",
        "",
        "## 2. Feature contract",
        "",
        f"{results['feature_count']} columns, unchanged from Milestone 8's "
        "`BASELINE_FEATURE_COLUMNS`:",
        "",
        ", ".join(f"`{c}`" for c in results["feature_columns"]),
        "",
        "`isFraud` (target), and `row_id`/`nameOrig`/`nameDest` "
        "(identifiers), are never part of this list and are never read "
        "into the feature matrix `X` — see `src/baseline_model.py`'s "
        "load-time assertions.",
        "",
        "## 3. Preprocessing",
        "",
        "- **Infinite-value handling**: +/-inf is replaced with NaN "
        "before imputation (deterministic, non-learned, applied "
        "identically to every split -- no leakage). Milestone 8's "
        "full-scale audit found 2 of 6,362,620 rows with +inf in "
        "`amount_vs_dest_avg_ratio`; see "
        "`reports/baseline_full_stream_audit.json`.",
        "- **Imputation**: `sklearn.impute.SimpleImputer(strategy=\"median\")`, "
        "fit on the TRAIN split only. Per-column medians are computed "
        "from train rows only and applied unchanged to validation/test.",
        "- **Scaling**: `sklearn.preprocessing.StandardScaler()`, fit on "
        "the TRAIN split only (after imputation), applied unchanged to "
        "validation/test.",
        "- No new categorical encoding was introduced: the categorical "
        "`type` column is already represented in the baseline contract "
        "via Milestone 5's fixed one-hot `type_*` columns.",
        "- NaN is never silently dropped or zero-filled before this "
        "stage — Milestone 8's undefined-value policy preserves it "
        "through feature assembly; the imputation VALUE is this "
        "milestone's modeling decision, made from train data only.",
        "",
        "## 4. Model",
        "",
        f"`{results['model']['class']}`, parameters (complete, fixed — no "
        "search performed):",
        "",
        "```",
        json.dumps(results["model"]["params"], indent=2),
        "```",
        "",
        f"- `class_weight=\"balanced\"` handles the ~0.13% fraud "
        "prevalence via reweighting, not resampling.",
        f"- `random_state={RANDOM_STATE}` fixed for determinism.",
        "- Explainability: signed per-feature coefficients (top 15 by "
        "magnitude below).",
        "",
        "| Feature | Coefficient |",
        "|---|---|",
    ]
    for row in top_features:
        lines.append(f"| `{row['feature']}` | {row['coefficient']:+.4f} |")

    lines += [
        "",
        "## 5. Evaluation methodology",
        "",
        f"- Default classification threshold: **{DEFAULT_THRESHOLD}** — "
        "fixed in module source, never chosen from validation or test "
        "outcomes.",
        f"- Fixed alert budget: **{ALERT_BUDGET_RATE:.2%}** of a split's "
        "rows (\"review the top-scored 1%\") — the rate is a constant "
        "decided ahead of time; only the resulting count (a function of "
        "that split's own row count) varies by split.",
        "- Validation and test are evaluated separately and never "
        "pooled into one reported figure.",
        "- Primary metric: Average Precision / PR-AUC. ROC-AUC is "
        "reported only as a secondary diagnostic — Milestone 9 section A "
        "warns it is dominated by the large true-negative count under "
        "0.13% prevalence and can look strong even when precision at any "
        "usable alert budget is weak.",
        "",
        "## 6. Results",
        "",
        "### Validation",
        "",
        *fmt_metrics(val),
        "",
        "### Test",
        "",
        *fmt_metrics(test),
        "",
        f"Training rows: {results['train_rows']:,} "
        f"({results['train_fraud']:,} fraud). Full pipeline runtime: "
        f"{results['runtime_seconds']:.1f}s. Peak RSS during the full "
        f"run: {results['peak_rss_mb']:.0f} MB.",
        "",
        "### Bounded smoke test (run before the full 6,362,620-row pipeline)",
        "",
        f"- Rows: {smoke_test['max_rows']:,}, chunksize {smoke_test['chunksize']:,}",
        f"- Runtime: {smoke_test['runtime_seconds']:.2f}s",
        f"- Peak RSS: {smoke_test['peak_rss_mb']:.0f} MB "
        f"(safety limit: {smoke_test['safety_limit_mb']:,} MB, "
        f"within limit: {smoke_test['within_safety_limit']})",
        "",
        "## 7. Limitations",
        "",
        "- **Near-perfect validation/test PR-AUC (1.0000) is almost "
        "certainly a PaySim simulator artifact, not evidence of a "
        "production-ready detector.** The two largest-magnitude "
        "coefficients are `orig_balance_mismatch` and "
        "`orig_balance_after_zero` — PaySim's own fraud-injection logic "
        "is documented (Milestone 3) to characteristically drain the "
        "origin account's balance to zero, which these two features "
        "capture almost exactly. A real fraud population would not be "
        "expected to be this cleanly separable on balance bookkeeping "
        "alone. This score should be read as 'the baseline correctly "
        "learned PaySim's synthetic tell', not as a general fraud-"
        "detection capability claim — exactly the caution Milestone 9 "
        "section G asks for ('avoid claims that a high metric "
        "automatically means the detector is operationally useful').",
        "- This is a first baseline: one model family, no hyperparameter "
        "search, an untuned default 0.5 threshold, and an untuned 1% "
        "alert budget. None of the reported numbers should be read as "
        "'this model is production-ready.'",
        "- ROC-AUC alone is not interpretable under this prevalence "
        "(Milestone 9 section A/G); absolute alert and false-positive "
        "counts are reported alongside every rate for exactly this "
        "reason — a high rate metric does not by itself mean the "
        "detector is operationally useful at any real alert budget.",
        "- Group-aware (account-level) splitting was assessed and not "
        "implemented at the design stage (Milestone 9 section C) and "
        "remains not implemented here.",
        "- No model comparison, ablation, or graph-vs-tabular study was "
        "performed — explicitly out of scope for this milestone.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_model_results(results: dict[str, Any], smoke_test: dict[str, Any], sanity: dict[str, Any], out_path: Path = MODEL_RESULTS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in results.items() if not k.startswith("_")}
    payload = {
        "smoke_test": smoke_test,
        "sanity_checks": sanity,
        **serializable,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Milestone 10 baseline classifier "
        "(bounded smoke test first, then the full streamed pipeline; never "
        "hyperparameter-tuned or threshold-tuned)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--smoke-rows", type=int, default=300_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Bound the full run too (testing only).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    print(f"Running bounded smoke test ({args.smoke_rows:,} rows)...")
    smoke = run_smoke_test(args.csv, chunksize=args.chunksize, max_rows=args.smoke_rows)
    print(f"Smoke test peak RSS   : {smoke['peak_rss_mb']:.0f} MB (limit {smoke['safety_limit_mb']:,} MB)")
    print(f"Smoke test runtime    : {smoke['runtime_seconds']:.2f}s")

    if not smoke["within_safety_limit"]:
        print(
            "Smoke test peak RSS exceeded the safety limit -- stopping "
            "before the full run. A disk-backed/incremental strategy "
            "would be required; not attempting the full in-memory "
            "approach.",
            file=sys.stderr,
        )
        return 1

    print("Smoke test passed. Running full pipeline...")
    results = run_pipeline(args.csv, chunksize=args.chunksize, max_rows=args.max_rows, threshold=args.threshold)

    check_against_full = args.csv == DEFAULT_RAW_CSV and args.max_rows is None
    sanity = run_sanity_checks(results, check_against_full_dataset=check_against_full)

    write_model_spec(results, smoke, MODEL_SPEC_PATH)
    write_model_results(results, smoke, sanity, MODEL_RESULTS_PATH)

    print(f"Train rows           : {results['train_rows']:,} ({results['train_fraud']:,} fraud)")
    print(f"Validation PR-AUC     : {results['validation']['average_precision_pr_auc']}")
    print(f"Validation precision  : {results['validation']['precision']:.4f}")
    print(f"Validation recall     : {results['validation']['recall']:.4f}")
    print(f"Test PR-AUC           : {results['test']['average_precision_pr_auc']}")
    print(f"Test precision        : {results['test']['precision']:.4f}")
    print(f"Test recall           : {results['test']['recall']:.4f}")
    print(f"Full-run runtime      : {results['runtime_seconds']:.1f}s")
    print(f"Full-run peak RSS     : {results['peak_rss_mb']:.0f} MB")
    print(f"Sanity checks passed  : {sanity['passed']}")
    print(f"Spec written to       : {MODEL_SPEC_PATH}")
    print(f"Results written to    : {MODEL_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
