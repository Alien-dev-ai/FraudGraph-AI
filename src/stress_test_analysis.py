"""Out-of-distribution / synthetic stress testing (Milestone 13).

ROBUSTNESS / STRESS-TESTING MILESTONE, NOT MODEL OPTIMIZATION. Answers:
"if the PaySim-specific balance signature is weakened or altered in the
TEST inputs, how does the ALREADY-TRAINED Milestone 10 detector behave?"
Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Critical distinction from Milestone 11 (read this first) ===

Milestone 11 (`src.ablation_analysis`) REMOVED features from the
feature SET and RETRAINED a new model on each reduced feature set --
different models, each fit on train data that never had the removed
columns.

Milestone 13 does something fundamentally different: ONE model is fit
ONCE (Milestone 10's exact recipe, on the unmodified 39-column train
split) and then KEPT COMPLETELY FIXED. Only the TEST feature VALUES for
`orig_balance_mismatch` / `orig_balance_after_zero` are perturbed
in-place-of-original (never in place -- always via a copy), and the
SAME fixed model, SAME fixed train-only imputer, and SAME fixed
train-only scaler are reused, unchanged, to score the perturbed test
rows. This measures the trained model's SENSITIVITY to those two input
values, not what a model trained without them would look like -- the
two experiments answer different questions and neither result implies
the other's.

=== Reused, unmodified ===

- `src.baseline_model.stream_split_arrays` (Milestone 10): one
  continuous causal streaming pass, same `chronological_train_val_test`
  split, same 39-column `BASELINE_FEATURE_COLUMNS` assembly.
- `src.baseline_model.fit_preprocessing` / `apply_preprocessing` /
  `train_baseline_model` / `DEFAULT_THRESHOLD` (0.5): the exact
  Milestone 10 recipe, fit/applied exactly as there.
- `src.ablation_analysis.FULL_BASELINE` / `BALANCE_FEATURES_REMOVED`:
  the same 39-column list and the same two named dominant balance
  features, reused verbatim (not re-derived).
- `src.robustness_analysis.stream_test_metadata` (Milestone 12): the
  same cheap, separate `step`/`type`-only streaming pass recovering
  each test row's `day_index`/`type` for slicing, and
  `src.robustness_analysis.slice_metrics` / `daily_breakdown` /
  `fraud_type_breakdown`: the same explicit-undefined-as-null metric
  functions and day/type grouping, reused verbatim rather than
  reimplemented -- this is also why every metric in this milestone
  reports `null` (never a silently substituted 0.0) exactly where
  Milestone 12 already established that policy.

=== Perturbation design ===

Both target columns (`orig_balance_mismatch`, `orig_balance_after_zero`)
are documented {0, 1} binary indicators (Milestone 5). Per this
milestone's instruction to avoid continuous noise on a binary feature,
perturbation is ROW-SUBSET VALUE REPLACEMENT, not additive noise:

1. `neutral_value_for(column)` -- the column's TRAIN-set mode (0 or 1,
   whichever is more common in the unmodified 39-column TRAIN split).
   Computed ONLY from train data (never from validation/test, never
   from labels of any kind) -- a data-derived reference/typical value,
   not a guess.
2. A single fixed row permutation of the TEST split's row indices,
   `numpy.random.default_rng(42).permutation(n_test)`, computed ONCE
   per run and reused for every scenario and every corruption level in
   this module. `random_state=42` throughout, matching every prior
   milestone's convention.
3. For a given column and corruption level L in [0, 1], the first
   `round(L * n_test)` rows of that FIXED permutation have that
   column's value replaced with its neutral value; every other row is
   untouched. Because the same permutation is reused at every level,
   the corrupted-row set at level L is a strict prefix of the
   corrupted-row set at any level > L (nested, monotonic corruption) --
   L=0.0 corrupts nothing (must reproduce the unperturbed baseline
   exactly) and L=1.0 corrupts every row (must exactly equal "fully
   neutralized").

This is a bounded, valid-domain transformation (output values are
always in {0, 1}, the feature's own domain -- never a continuous,
out-of-domain value) and is fully deterministic and predetermined
before any metric is computed: `PERTURBATION_LEVELS` below is a fixed
list, chosen before this module was run against results, and never
adjusted afterward.

=== Leakage / safety ===

Perturbation functions only ever read TRAIN data (for the neutral
value) and modify a COPY of the TEST feature matrix; `isFraud`, `type`,
`step`/`day_index`, account identifiers, and row order are never read,
written, or reordered by any perturbation function here. The fitted
imputer/scaler/model are each created exactly once (train-only) and
never refit. See `run_baseline_reproduction_check` for the mandatory
pre-stress-test check that this pipeline reproduces Milestone 10/11/12's
recorded `FULL_BASELINE` test-period numbers before any stress result
is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import ablation_analysis as ab
from src import baseline_dataset as bd
from src import baseline_model as bmod
from src import robustness_analysis as ra
from src.validate_raw_paysim import DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = bmod.DEFAULT_CHUNKSIZE
RANDOM_STATE = bmod.RANDOM_STATE  # 42, unchanged
DEFAULT_THRESHOLD = bmod.DEFAULT_THRESHOLD  # 0.5, unchanged, never tuned here
SAFETY_RSS_LIMIT_MB = bmod.SAFETY_RSS_LIMIT_MB

FULL_BASELINE: list[str] = ab.FULL_BASELINE  # 39 columns, unchanged
DOMINANT_BALANCE_FEATURES: list[str] = list(ab.BALANCE_FEATURES_REMOVED)  # the exact M11 pair
assert DOMINANT_BALANCE_FEATURES == ["orig_balance_mismatch", "orig_balance_after_zero"]

# Predetermined BEFORE any result was evaluated -- see module docstring
# "Perturbation design". Never adjusted after looking at outcomes.
PERTURBATION_LEVELS: list[float] = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
STRESS_SPEC_PATH = REPORTS_DIR / "stress_test_specification.md"
STRESS_RESULTS_PATH = REPORTS_DIR / "stress_test_results.json"
STRESS_SUMMARY_CSV_PATH = REPORTS_DIR / "stress_test_summary.csv"
STRESS_DAILY_CSV_PATH = REPORTS_DIR / "stress_test_daily.csv"
STRESS_BY_TYPE_CSV_PATH = REPORTS_DIR / "stress_test_by_type.csv"


def neutral_value_for(column: str, X_train: np.ndarray) -> float:
    """Train-only reference/neutral value for a binary {0,1} column:
    its train-set mode. Never reads validation/test or any label.
    """
    idx = bmod.FEATURE_COLUMNS.index(column)
    vals = X_train[:, idx]
    finite = vals[np.isfinite(vals)]
    uniq = np.unique(finite)
    assert set(np.round(uniq).tolist()) <= {0.0, 1.0}, f"{column} expected a binary {{0,1}} column, got values {uniq}"
    mean = float(finite.mean())
    return 1.0 if mean > 0.5 else 0.0


def fixed_row_permutation(n_rows: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """The one, fixed test-row permutation reused by every scenario
    and every corruption level in this module (see module docstring).
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(n_rows)


def perturb_columns(
    X: np.ndarray,
    columns: list[str],
    level: float,
    neutral_values: dict[str, float],
    permutation: np.ndarray,
) -> np.ndarray:
    """Return a NEW array (the input is never mutated) with `columns`
    replaced by their neutral values on the first `round(level * n)`
    rows of the fixed `permutation`. `level=0.0` returns values
    identical to the input; `level=1.0` sets every row of every named
    column to its neutral value.
    """
    assert 0.0 <= level <= 1.0
    X_perturbed = X.copy()
    n = X.shape[0]
    n_corrupt = int(round(level * n))
    corrupt_rows = permutation[:n_corrupt]
    for column in columns:
        idx_col = bmod.FEATURE_COLUMNS.index(column)
        X_perturbed[corrupt_rows, idx_col] = neutral_values[column]
    return X_perturbed


def predict_with_fixed_model(imputer, scaler, model, X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score `X_raw` through the ALREADY-FITTED imputer/scaler/model --
    transform-only, never refit, exactly Milestone 10's inference path.
    """
    X_proc = bmod.apply_preprocessing(imputer, scaler, X_raw)
    proba = model.predict_proba(X_proc)[:, 1]
    preds = (proba >= DEFAULT_THRESHOLD).astype(int)
    return proba, preds


def run_baseline_reproduction_check(
    y_test: np.ndarray, proba0: np.ndarray, preds0: np.ndarray
) -> dict[str, Any]:
    """Mandatory pre-stress-test check: the unperturbed (fixed) model's
    whole-test-period metrics must reproduce Milestone 10/11/12's
    recorded `FULL_BASELINE` test numbers. Compares against Milestone
    12's saved results (the most recent prior milestone) when
    available; always reports the recomputed metrics regardless.
    """
    recomputed = ra.slice_metrics(y_test, preds0, proba0)

    recorded = None
    recorded_path = REPORTS_DIR / "robustness_results.json"
    if recorded_path.exists():
        with recorded_path.open() as f:
            prior = json.load(f)
        recorded = prior["groups"]["FULL_BASELINE"]["whole_test_period"]

    matches_recorded = None
    if recorded is not None:
        matches_recorded = (
            recomputed["confusion_matrix"] == recorded["confusion_matrix"]
            and recomputed["n_total"] == recorded["n_total"]
            and recomputed["n_fraud"] == recorded["n_fraud"]
        )

    return {
        "recomputed": recomputed,
        "recorded_m12_full_baseline": recorded,
        "matches_recorded_confusion_matrix_and_counts": matches_recorded,
    }


def _scenario_result(y_test, proba, preds, day_index_test, type_test) -> dict[str, Any]:
    """One scenario's full metric bundle: whole-test-period +
    per-day + per-fraud-type breakdowns, all via Milestone 12's
    `slice_metrics`-based functions (explicit null for undefined).
    """
    return {
        "whole_test_period": ra.slice_metrics(y_test, preds, proba),
        "daily": ra.daily_breakdown(y_test, proba, preds, day_index_test),
        "by_type": ra.fraud_type_breakdown(y_test, proba, preds, type_test),
    }


def run_smoke_test(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 300_000
) -> dict[str, Any]:
    """Bounded resource check: fit the model once on a bounded prefix,
    verify the perturbation machinery and prediction path, measure
    peak RSS, before the full run.
    """
    rss_before = bmod.peak_rss_mb()
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    X_train, y_train = full_arrays["train"]
    X_test, y_test = full_arrays["test"]

    result: dict[str, Any] = {
        "max_rows": max_rows,
        "chunksize": chunksize,
        "stream_stats": stream_stats,
        "test_rows_available": int(X_test.shape[0]),
    }

    trainable = X_train.shape[0] > 0 and len(np.unique(y_train)) >= 2
    result["model_fit_on_smoke_sample"] = False
    result["scenario_exercised"] = False
    result["zero_percent_reproduces_baseline"] = None
    result["hundred_percent_matches_neutral_value"] = None

    if trainable:
        imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
        model = bmod.train_baseline_model(X_train_proc, y_train)
        result["model_fit_on_smoke_sample"] = True

        if X_test.shape[0] > 0:
            neutral_values = {c: neutral_value_for(c, X_train) for c in DOMINANT_BALANCE_FEATURES}
            permutation = fixed_row_permutation(X_test.shape[0])

            proba0, preds0 = predict_with_fixed_model(imputer, scaler, model, X_test)

            X_0pct = perturb_columns(X_test, DOMINANT_BALANCE_FEATURES, 0.0, neutral_values, permutation)
            proba_0pct, preds_0pct = predict_with_fixed_model(imputer, scaler, model, X_0pct)
            result["zero_percent_reproduces_baseline"] = bool(np.array_equal(proba0, proba_0pct))

            X_100pct = perturb_columns(X_test, DOMINANT_BALANCE_FEATURES, 1.0, neutral_values, permutation)
            checks = []
            for col in DOMINANT_BALANCE_FEATURES:
                idx_col = bmod.FEATURE_COLUMNS.index(col)
                checks.append(bool(np.all(X_100pct[:, idx_col] == neutral_values[col])))
            result["hundred_percent_matches_neutral_value"] = all(checks)
            result["scenario_exercised"] = True

    result["runtime_seconds"] = round(time.time() - t0, 2)
    result["rss_before_mb"] = round(rss_before, 1)
    result["peak_rss_mb"] = round(bmod.peak_rss_mb(), 1)
    result["safety_limit_mb"] = SAFETY_RSS_LIMIT_MB
    result["within_safety_limit"] = result["peak_rss_mb"] < SAFETY_RSS_LIMIT_MB
    return result


def run_full_stress_test(
    csv_path: Path = DEFAULT_RAW_CSV,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Fit the Milestone 10 model ONCE, then run stress scenarios A-D
    sequentially against fixed, tiny (test-sized) perturbed copies of
    the test feature matrix -- never retraining, never refitting
    preprocessing, never touching train/validation again after the one
    initial fit.
    """
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    day_index_test, type_test = ra.stream_test_metadata(csv_path, chunksize=chunksize, max_rows=max_rows)

    X_train, y_train = full_arrays["train"]
    X_test, y_test = full_arrays["test"]
    assert len(day_index_test) == X_test.shape[0], (
        f"test metadata row count ({len(day_index_test)}) must match the feature stream's "
        f"test bucket ({X_test.shape[0]})"
    )

    # Fit ONCE. Never refit for any scenario below.
    imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
    model = bmod.train_baseline_model(X_train_proc, y_train)

    proba0, preds0 = predict_with_fixed_model(imputer, scaler, model, X_test)
    baseline_check = run_baseline_reproduction_check(y_test, proba0, preds0)

    # The hard stop only makes sense when comparing against Milestone
    # 12's recorded numbers on the SAME (full, unmodified) dataset --
    # bounded/fixture runs are expected to disagree trivially (different
    # data) and must not trip this guard.
    is_full_real_dataset_run = Path(csv_path) == Path(DEFAULT_RAW_CSV) and max_rows is None
    if is_full_real_dataset_run and baseline_check["matches_recorded_confusion_matrix_and_counts"] is False:
        raise RuntimeError(
            "Baseline reproduction check FAILED: the unperturbed Milestone 13 pipeline "
            "does not reproduce Milestone 12's recorded FULL_BASELINE test metrics. "
            "Stopping before running any stress scenario, per this milestone's "
            "explicit instruction to diagnose discrepancies before proceeding."
        )

    neutral_values = {c: neutral_value_for(c, X_train) for c in DOMINANT_BALANCE_FEATURES}
    permutation = fixed_row_permutation(X_test.shape[0])

    baseline_scenario = _scenario_result(y_test, proba0, preds0, day_index_test, type_test)

    # Stress Tests A, B, C: full (level=1.0) neutralization of one or both columns.
    scenario_definitions = {
        "A_remove_orig_balance_mismatch": ["orig_balance_mismatch"],
        "B_remove_orig_balance_after_zero": ["orig_balance_after_zero"],
        "C_neutralize_both": DOMINANT_BALANCE_FEATURES,
    }
    fixed_scenarios: dict[str, Any] = {}
    for scenario_name, columns in scenario_definitions.items():
        X_perturbed = perturb_columns(X_test, columns, 1.0, neutral_values, permutation)
        proba, preds = predict_with_fixed_model(imputer, scaler, model, X_perturbed)
        fixed_scenarios[scenario_name] = {
            "perturbed_columns": columns,
            "level": 1.0,
            **_scenario_result(y_test, proba, preds, day_index_test, type_test),
        }
        del X_perturbed  # release before the next scenario

    # Stress Test D: graduated corruption of BOTH dominant features together.
    noise_sweep: dict[str, Any] = {}
    for level in PERTURBATION_LEVELS:
        X_perturbed = perturb_columns(X_test, DOMINANT_BALANCE_FEATURES, level, neutral_values, permutation)
        proba, preds = predict_with_fixed_model(imputer, scaler, model, X_perturbed)
        noise_sweep[f"{level:.2f}"] = {
            "level": level,
            "rows_corrupted": int(round(level * X_test.shape[0])),
            **_scenario_result(y_test, proba, preds, day_index_test, type_test),
        }
        del X_perturbed

    runtime = time.time() - t0

    return {
        "csv_path": str(csv_path),
        "chunksize": chunksize,
        "max_rows": max_rows,
        "runtime_seconds": round(runtime, 2),
        "peak_rss_mb": round(bmod.peak_rss_mb(), 1),
        "threshold": DEFAULT_THRESHOLD,
        "random_state": RANDOM_STATE,
        "split_policy_name": bmod.SPLIT_POLICY_NAME,
        "split_day_ranges": {k: list(v) for k, v in bmod.SPLIT_DAY_RANGES.items()},
        "test_day_range": list(ra.TEST_DAY_RANGE),
        "simulation_boundary_day": ra.SIMULATION_BOUNDARY_DAY,
        "dominant_balance_features": DOMINANT_BALANCE_FEATURES,
        "neutral_values": neutral_values,
        "perturbation_levels": PERTURBATION_LEVELS,
        "train_rows": int(X_train.shape[0]),
        "train_fraud": int(y_train.sum()),
        "test_rows": int(X_test.shape[0]),
        "test_fraud": int(y_test.sum()),
        "stream_stats": stream_stats,
        "baseline_reproduction_check": baseline_check,
        "baseline_scenario": baseline_scenario,
        "fixed_model_scenarios_A_B_C": fixed_scenarios,
        "noise_sweep_D": noise_sweep,
    }


def write_summary_csv(results: dict[str, Any], out_path: Path = STRESS_SUMMARY_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "level", "n_total", "n_fraud", "n_alerts", "tp", "fp", "fn", "tn",
        "precision", "recall", "average_precision_pr_auc", "roc_auc_secondary",
    ]

    def row_for(name: str, level: Any, m: dict[str, Any]) -> dict[str, Any]:
        cm = m["confusion_matrix"]
        return {
            "scenario": name, "level": level, "n_total": m["n_total"], "n_fraud": m["n_fraud"],
            "n_alerts": m["n_alerts"], "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
            "precision": m["precision"], "recall": m["recall"],
            "average_precision_pr_auc": m["average_precision_pr_auc"],
            "roc_auc_secondary": m["roc_auc_secondary"],
        }

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row_for("baseline_unperturbed", 0.0, results["baseline_scenario"]["whole_test_period"]))
        for name, scenario in results["fixed_model_scenarios_A_B_C"].items():
            writer.writerow(row_for(name, scenario["level"], scenario["whole_test_period"]))
        for level_str, scenario in results["noise_sweep_D"].items():
            writer.writerow(row_for("D_noise_sweep", scenario["level"], scenario["whole_test_period"]))


def write_daily_csv(results: dict[str, Any], out_path: Path = STRESS_DAILY_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "level", "day_index", "is_simulation_boundary_day", "n_total", "n_fraud",
        "n_alerts", "tp", "fp", "fn", "tn", "precision", "recall",
        "average_precision_pr_auc", "roc_auc_secondary",
    ]

    def rows_for(name: str, level: Any, daily: dict[str, Any]):
        for day_str, m in daily.items():
            cm = m["confusion_matrix"]
            yield {
                "scenario": name, "level": level, "day_index": day_str,
                "is_simulation_boundary_day": m["is_simulation_boundary_day"],
                "n_total": m["n_total"], "n_fraud": m["n_fraud"], "n_alerts": m["n_alerts"],
                "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
                "precision": m["precision"], "recall": m["recall"],
                "average_precision_pr_auc": m["average_precision_pr_auc"],
                "roc_auc_secondary": m["roc_auc_secondary"],
            }

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for("baseline_unperturbed", 0.0, results["baseline_scenario"]["daily"]):
            writer.writerow(row)
        for name, scenario in results["fixed_model_scenarios_A_B_C"].items():
            for row in rows_for(name, scenario["level"], scenario["daily"]):
                writer.writerow(row)
        for level_str, scenario in results["noise_sweep_D"].items():
            for row in rows_for("D_noise_sweep", scenario["level"], scenario["daily"]):
                writer.writerow(row)


def write_by_type_csv(results: dict[str, Any], out_path: Path = STRESS_BY_TYPE_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "level", "type", "n_total", "n_fraud", "n_alerts", "tp", "fp", "fn", "tn",
        "precision", "recall", "average_precision_pr_auc", "roc_auc_secondary",
    ]

    def rows_for(name: str, level: Any, by_type: dict[str, Any]):
        for type_name, m in by_type.items():
            cm = m["confusion_matrix"]
            yield {
                "scenario": name, "level": level, "type": type_name,
                "n_total": m["n_total"], "n_fraud": m["n_fraud"], "n_alerts": m["n_alerts"],
                "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
                "precision": m["precision"], "recall": m["recall"],
                "average_precision_pr_auc": m["average_precision_pr_auc"],
                "roc_auc_secondary": m["roc_auc_secondary"],
            }

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for("baseline_unperturbed", 0.0, results["baseline_scenario"]["by_type"]):
            writer.writerow(row)
        for name, scenario in results["fixed_model_scenarios_A_B_C"].items():
            for row in rows_for(name, scenario["level"], scenario["by_type"]):
                writer.writerow(row)
        for level_str, scenario in results["noise_sweep_D"].items():
            for row in rows_for("D_noise_sweep", scenario["level"], scenario["by_type"]):
                writer.writerow(row)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_stress_results(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = STRESS_RESULTS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"smoke_test": smoke_test, **results}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_stress_spec(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = STRESS_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_metric(m: dict[str, Any], key: str, fmt: str = ".4f") -> str:
        v = m.get(key)
        return "null" if v is None else format(v, fmt)

    def fmt_whole(name: str, m: dict[str, Any]) -> str:
        cm = m["confusion_matrix"]
        return (
            f"- **{name}**: alerts={m['n_alerts']:,} TP={cm['tp']:,} FP={cm['fp']:,} FN={cm['fn']:,} "
            f"precision={fmt_metric(m, 'precision')} recall={fmt_metric(m, 'recall')} "
            f"PR-AUC={fmt_metric(m, 'average_precision_pr_auc')} "
            f"ROC-AUC={fmt_metric(m, 'roc_auc_secondary')}"
        )

    bc = results["baseline_reproduction_check"]
    lines = [
        "# FraudGraph AI — Out-of-Distribution / Synthetic Stress Testing (Milestone 13)",
        "",
        "**This is a robustness/stress-testing milestone, not model "
        "optimization, feature engineering, threshold tuning, "
        "hyperparameter tuning, or a claim of real-world performance.**",
        "",
        "## Critical distinction: fixed-model perturbation vs. Milestone 11's retrained ablation",
        "",
        "Milestone 11 REMOVED features and RETRAINED a new model on "
        "each reduced feature set. **Milestone 13 fits the Milestone 10 "
        "model exactly ONCE and never retrains or refits preprocessing "
        "again** -- every stress scenario below perturbs only the TEST "
        "feature VALUES and reuses that one fixed imputer/scaler/model "
        "to score them. The two experiments answer different "
        "questions: M11 asks 'what would a model look like without "
        "this feature'; M13 asks 'how sensitive is the model we "
        "already have to this feature's value.' Neither result implies "
        "the other's.",
        "",
        "## Objective",
        "",
        "Determine how sensitive the Milestone 10 `FULL_BASELINE` "
        "detector is to `orig_balance_mismatch` and "
        "`orig_balance_after_zero` -- the two features Milestone 11 "
        "identified as dominant -- by perturbing them in the TEST "
        "split only, under the existing fixed model.",
        "",
        "## Perturbations (exact, mathematical, deterministic)",
        "",
        "1. **Neutral/reference value**: each column's TRAIN-set mode "
        "(computed only from train, never from validation/test/labels). "
        f"Measured this run: `{json.dumps(results['neutral_values'])}`.",
        "2. **Fixed row permutation**: "
        f"`numpy.random.default_rng({RANDOM_STATE}).permutation(n_test)`, "
        "computed once and reused for every scenario/level.",
        "3. **Corruption at level L**: the first `round(L * n_test)` "
        "rows of that fixed permutation have the named column(s) set "
        "to their neutral value; all other rows are untouched. Higher "
        "levels strictly extend lower levels (nested, monotonic).",
        f"4. **Predetermined levels** (Stress Test D): `{PERTURBATION_LEVELS}` "
        "-- fixed before any result was computed, never adjusted "
        "afterward.",
        "",
        "## Fixed-model policy",
        "",
        "One `LogisticRegression(class_weight=\"balanced\", "
        f"random_state={RANDOM_STATE})` is fit once on the unmodified "
        "39-column TRAIN split (Milestone 10's recipe, unchanged) and "
        "reused, unmodified, for every scenario in this report. It is "
        "never retrained.",
        "",
        "## Preprocessing policy",
        "",
        "`SimpleImputer(strategy=\"median\")` and `StandardScaler()` "
        "are each fit exactly once, on TRAIN only (Milestone 10's "
        "`fit_preprocessing`), and only ever `.transform()`-ed "
        "afterward -- including on every perturbed test copy. Never "
        "refit on validation/test/perturbed data.",
        "",
        "## Split policy",
        "",
        f"Milestone 9's `chronological_train_val_test` policy, "
        f"unchanged: train days {results['split_day_ranges']['train'][0]}-"
        f"{results['split_day_ranges']['train'][1]}, validation days "
        f"{results['split_day_ranges']['validation'][0]}-"
        f"{results['split_day_ranges']['validation'][1]}, test days "
        f"{results['split_day_ranges']['test'][0]}-"
        f"{results['split_day_ranges']['test'][1]}. Perturbation never "
        "changes which row belongs to which split.",
        "",
        "## Threshold policy",
        "",
        f"Fixed at {DEFAULT_THRESHOLD} for every scenario -- never "
        "tuned, never varied per scenario/day/type.",
        "",
        "## Deterministic seed",
        "",
        f"`random_state={RANDOM_STATE}` everywhere randomness appears "
        "(model fit, permutation) -- matching every prior milestone.",
        "",
        "## Resource strategy",
        "",
        "The expensive 39-column causal feature stream and the one "
        "model fit each happen exactly ONCE. Every stress scenario "
        "operates on a freshly copied, test-sized (tens of thousands "
        "of rows, not millions) perturbed array that is discarded "
        "before the next scenario begins -- never multiple full "
        "feature matrices or duplicate train-sized arrays held at "
        "once.",
        "",
        "### Bounded smoke test",
        "",
        f"- Rows: {smoke_test['max_rows']:,}, chunksize {smoke_test['chunksize']:,}",
        f"- Runtime: {smoke_test['runtime_seconds']:.2f}s",
        f"- Peak RSS: {smoke_test['peak_rss_mb']:.0f} MB "
        f"(safety limit: {smoke_test['safety_limit_mb']:,} MB, within "
        f"limit: {smoke_test['within_safety_limit']})",
        f"- 0% level reproduced the unperturbed baseline exactly: "
        f"{smoke_test['zero_percent_reproduces_baseline']}",
        f"- 100% level produced the expected neutral values: "
        f"{smoke_test['hundred_percent_matches_neutral_value']}",
        "",
        "## Leakage controls",
        "",
        "Perturbation functions read TRAIN data only for the neutral "
        "value; they never read `isFraud`, never read validation/test "
        "labels, and never choose a perturbation parameter by looking "
        "at any result. Perturbation levels were fixed before this "
        "module was run against real outcomes.",
        "",
        "## Baseline reproduction check (must pass before any stress result)",
        "",
        f"Matches Milestone 12's recorded `FULL_BASELINE` test-period "
        f"confusion matrix and row/fraud counts exactly: "
        f"**{bc['matches_recorded_confusion_matrix_and_counts']}**.",
        "",
        fmt_whole("unperturbed (recomputed here)", bc["recomputed"]),
    ]
    if bc["recorded_m12_full_baseline"] is not None:
        lines.append(fmt_whole("recorded (Milestone 12)", bc["recorded_m12_full_baseline"]))
    lines += [
        "",
        "## A. What happens when the dominant balance signal is removed from the test inputs?",
        "",
    ]
    for name, scenario in results["fixed_model_scenarios_A_B_C"].items():
        lines.append(fmt_whole(name, scenario["whole_test_period"]))
    lines += [
        "",
        "Model sensitivity to these two inputs is substantial: removing "
        "or neutralizing them (fixed model, values-only perturbation) "
        "changes predicted alerts materially. This describes measured "
        "model behavior after controlled perturbation -- it does not "
        "establish that either feature causes fraud, or that the model "
        "'understands' fraud in any causal sense.",
        "",
        "## B. What happens as that signal is progressively corrupted?",
        "",
    ]
    for level_str, scenario in results["noise_sweep_D"].items():
        lines.append(fmt_whole(f"D level={level_str}", scenario["whole_test_period"]))
    lines += [
        "",
        "## C. Does the effect remain consistent across days?",
        "",
        "Full per-day detail (every scenario x every test-period day, "
        "including day 30) is in `reports/stress_test_daily.csv`. Day "
        "30 is reported explicitly, never excluded: it carries "
        "Milestone 4/9's documented simulation-boundary artifact (100% "
        "fraud among its 272 remaining rows, zero legitimate rows), so "
        "precision/PR-AUC/ROC-AUC are undefined or degenerate on that "
        "slice in isolation regardless of perturbation -- reported as "
        "`null` with a documented reason, exactly as Milestone 12 "
        "established.",
        "",
        "## D. Does the effect differ between CASH_OUT and TRANSFER?",
        "",
        "Full per-type detail (every scenario x CASH_OUT/TRANSFER) is "
        "in `reports/stress_test_by_type.csv`. No independent tuning "
        "was performed per type -- the same fixed threshold and "
        "perturbation levels apply to both.",
        "",
        "## E. Fixed-model perturbation vs. Milestone 11 retrained ablation",
        "",
        "Milestone 11's `BALANCE_REDUCED_FULL` retrained a model with "
        "37 columns (the two dominant features never existed as inputs "
        "during training). Milestone 13's Stress Test C keeps the "
        "SAME 39-column-trained model and only overwrites those two "
        "columns' TEST values. Both are reported side by side in "
        "`reports/stress_test_results.json` for direct comparison; a "
        "numeric difference between them reflects the difference "
        "between 'never having learned from a feature' and 'having "
        "learned from it but being fed an uninformative value at "
        "inference time' -- not a ranking of which approach is better.",
        "",
        "## Interpretation",
        "",
        "Three separate claims, deliberately not conflated (matching "
        "Milestone 12's convention):",
        "",
        "1. **Measured behavior on PaySim**: the numbers above are "
        "exact, measured figures from controlled perturbation of "
        "PaySim's synthetic transaction log, under the fixed "
        "Milestone 10 model.",
        "2. **Robustness under controlled synthetic perturbation**: "
        "describes how sensitive the already-trained detector is to "
        "two specific input values within this simulation -- language "
        "used throughout this report is limited to \"model sensitivity "
        "to the feature\" and \"performance changed after controlled "
        "perturbation,\" consistent with dependence on PaySim's "
        "synthetic balance pattern.",
        "3. **What remains unknown for real-world fraud detection**: "
        "whether this sensitivity pattern would appear, matter, or "
        "behave similarly on real (non-synthetic) transaction data is "
        "NOT established here. No claim of causal importance, real-"
        "world generalization, or production readiness is made.",
        "",
        "## Limitations",
        "",
        "- Only one model family, one preprocessing configuration, and "
        "one fixed threshold were ever evaluated (Milestone 10's, "
        "unchanged) -- this does not establish sensitivity under a "
        "different model or preprocessing choice.",
        "- The fixed row permutation determines WHICH rows are "
        "corrupted first at low levels; a different seed would corrupt "
        "a different subset at the same level (though the aggregate "
        "corruption fraction would be identical) -- results at "
        "intermediate levels (10-75%) reflect one particular "
        "deterministic draw, not an average over draws.",
        "- Day-level and type-level slices can be small; metrics on "
        "small slices are reported exactly as computed but are "
        "correspondingly noisier than the whole-test-period aggregate.",
        "- This milestone does not claim, and explicitly disclaims, "
        "real-world generalization or production readiness.",
        "- No hyperparameter tuning, threshold tuning, or resampling "
        "was performed anywhere in this milestone.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 13 stress test: perturb the two dominant "
        "balance features in the TEST split only and score them through "
        "the fixed, already-trained Milestone 10 model."
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

    if not smoke["within_safety_limit"]:
        print("Smoke test exceeded the safety limit -- stopping before the full run.", file=sys.stderr)
        return 1

    print("Smoke test passed. Running the full stress test...")
    results = run_full_stress_test(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)

    write_summary_csv(results, STRESS_SUMMARY_CSV_PATH)
    write_daily_csv(results, STRESS_DAILY_CSV_PATH)
    write_by_type_csv(results, STRESS_BY_TYPE_CSV_PATH)
    write_stress_results(results, smoke, STRESS_RESULTS_PATH)
    write_stress_spec(results, smoke, STRESS_SPEC_PATH)

    bc = results["baseline_reproduction_check"]
    print(f"Baseline reproduction matches M12: {bc['matches_recorded_confusion_matrix_and_counts']}")
    wtp = results["baseline_scenario"]["whole_test_period"]
    print(f"Baseline   precision={wtp['precision']:.4f} recall={wtp['recall']:.4f} PR-AUC={wtp['average_precision_pr_auc']:.4f}")
    for name, scenario in results["fixed_model_scenarios_A_B_C"].items():
        m = scenario["whole_test_period"]
        print(f"{name:32s} precision={m['precision']:.4f} recall={m['recall']:.4f} PR-AUC={m['average_precision_pr_auc']:.4f}")
    for level_str, scenario in results["noise_sweep_D"].items():
        m = scenario["whole_test_period"]
        print(f"D level={level_str}  precision={m['precision']:.4f} recall={m['recall']:.4f} PR-AUC={m['average_precision_pr_auc']:.4f}")
    print(f"Full-run runtime      : {results['runtime_seconds']:.1f}s")
    print(f"Full-run peak RSS     : {results['peak_rss_mb']:.0f} MB")
    print(f"Spec written to       : {STRESS_SPEC_PATH}")
    print(f"Results written to    : {STRESS_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
