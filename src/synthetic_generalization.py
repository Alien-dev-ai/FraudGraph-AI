"""Synthetic fraud-pattern generalization testing (Milestone 14).

CONTROLLED SYNTHETIC GENERALIZATION EXPERIMENT, NOT PRODUCTION VALIDATION.
Answers: "does the fixed Milestone 10 detector react to fraud-like
BEHAVIORAL patterns that do NOT carry PaySim's dominant balance
signature (`orig_balance_mismatch`, `orig_balance_after_zero`), or is it
primarily keyed on that exact synthetic signature?" Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Synthetic examples, not naturally-occurring fraud (read first) ===

Every synthetic example in this module is built from an EXISTING,
genuinely LEGITIMATE test-period transaction (`isFraud == 0`). Its
ORIGINAL PaySim label is 0 and is NEVER changed, anywhere, in any array
this module produces. A SEPARATE, explicit `synthetic_stress_label`
records what the transformation INTENDS to test (1 = "constructed to
look fraud-like via a specific mechanism", 0 = "constructed as a
distractor that should still look legitimate") -- this is a synthetic,
counterfactual evaluation label, never presented as, or mixed into,
`isFraud`. See `SCENARIO_DEFINITIONS` below for exactly which label
each scenario uses. This module never relabels a real fraud row, never
reads `isFraud` to pick which rows to transform (only to confirm a
source row is legitimate, i.e. `isFraud == 0`, which is a data
property, not a tuning parameter), and never looks at model predictions
before finalizing which rows/transformations to use.

=== Reused, unmodified ===

- `src.baseline_model.stream_split_arrays` (Milestone 10): one
  continuous causal pass, unchanged `chronological_train_val_test`
  split, unchanged 39-column `BASELINE_FEATURE_COLUMNS`.
- `src.baseline_model.fit_preprocessing` / `apply_preprocessing` /
  `train_baseline_model` / `DEFAULT_THRESHOLD` (0.5): Milestone 10's
  exact recipe, fit ONCE on train and never refit (Mode 1, mandatory
  per this milestone's instructions -- see "Model policy" below for why
  Mode 2 is not implemented).
- `src.robustness_analysis.stream_test_metadata` / `slice_metrics` /
  `daily_breakdown` / `fraud_type_breakdown` (Milestone 12): the same
  cheap step/type-only metadata stream and the same explicit-null-for-
  undefined metric/grouping functions, reused verbatim. Day 30's
  breakdown entry is produced by this exact same reuse -- see
  "Day-30 degeneracy" below for why it is always empty here.
- `src.stress_test_analysis.neutral_value_for` /
  `predict_with_fixed_model` / `run_baseline_reproduction_check`
  (Milestone 13): the exact train-only neutral-value computation, the
  exact fixed-model scoring path, and the exact pre-flight reproduction
  check, reused verbatim rather than reimplemented.
- `src.ablation_analysis.BALANCE_FEATURES_REMOVED` (Milestone 11): the
  same two named dominant features.
- `src.validate_raw_paysim.BALANCE_TOLERANCE`: the exact tolerance
  Milestone 5's own mismatch computation uses (see Recipe D below).

=== Model policy: Mode 1 only ===

Mode 1 (fixed model, synthetic feature values only, no retraining) is
implemented, per this milestone's mandatory requirement. Mode 2
(optional retrained variant) is NOT implemented: this milestone's own
instructions constrain Mode 2 to train on the identical, unmodified
Milestone 10 train set with an identical model configuration and
without adding any synthetic example to training -- which means a Mode
2 model would be fit on exactly the same data with exactly the same
deterministic algorithm as Mode 1's model, and would therefore be
bit-for-bit identical to it (Milestone 10/11/12/13 already
demonstrated this determinism repeatedly). Implementing Mode 2 would
add a second full model-fit's resource cost for a result already known
to be identical, so it is skipped as providing no additional
information, consistent with this milestone's resource constraints.

=== Transformation selection (done before implementation) ===

Candidates considered from the existing 39-column feature contract,
evaluated against "does perturbing this create an impossible or
meaningless state":

  SELECTED -- `origin_is_first_seen` / `origin_txn_count_causal` /
  `origin_total_amount_causal` / `origin_avg_amount_causal` /
  `amount_vs_origin_avg_ratio` (Milestone 5, identity/activity
  category, non-balance): jointly set to the EXACT combination
  Milestone 5 already documents for a genuine first-time-seen origin
  account (count=0, total=0.0, average/ratio = NaN "no prior history
  yet"). This is not an invented state -- it is the same state real
  first-time transactions already occupy elsewhere in the dataset --
  and it encodes a well-known, non-PaySim-specific fraud heuristic
  (a freshly created / "mule" account transacting for the first time),
  entirely independent of balance bookkeeping. Used as "Recipe
  fresh_account" below.

  SELECTED -- `orig_balance_mismatch` / `orig_balance_after_zero`
  (Milestone 5, balance category): each is a standalone {0,1}
  indicator column with no other column depending on its value, so
  overwriting either to its Milestone-13-style train-only neutral/
  reference value is self-consistent by construction. Used as
  "Recipe neutral_mismatch" / "Recipe neutral_after_zero".

  SELECTED -- `orig_balance_after` (Milestone 5, balance category):
  overwritten to a fixed, small, NONZERO fraction of the PRESERVED
  `orig_balance_before` (see `RESIDUAL_FRACTION` below) -- a different
  balance-draining pattern than PaySim's own typical "drains to exactly
  zero" fraud signature. Every column Milestone 5 defines as a
  function of `orig_balance_after` (`orig_balance_delta`,
  `orig_balance_after_zero`, `orig_balance_mismatch`) is recomputed
  from the new value using the IDENTICAL formula
  `src.features_raw_paysim.compute_stateless_features` uses, so the
  resulting row is always internally consistent -- never a state that
  formula could not itself have produced. Used as "Recipe
  altered_balance_drain".

  REJECTED -- `amount` / `type_*` one-hot columns: explicitly
  prohibited by this milestone's instructions (amount and transaction
  type must be preserved).
  REJECTED -- destination-side balance columns (`dest_balance_before`/
  `dest_balance_after` and their zero-flags): Milestone 3 documents
  that PaySim does not simulate merchant balances at all, so mutating
  these for a merchant destination would contradict an already-
  documented simulator invariant; kept out of scope to avoid
  constructing a state the simulator itself never produces.
  REJECTED -- `origin_before_zero`, i.e. flipping `orig_balance_before_
  zero` alone: would desynchronize the flag from the (preserved)
  `orig_balance_before` value it describes, an internally inconsistent
  state; not used.
  REJECTED -- the 5 retained graph-derived features (Milestone 7):
  Milestone 11 already found these carry weak, sign-unstable
  standalone signal with no clear behavioral-fraud story of their own;
  perturbing them would not constitute a well-justified "fraud-like"
  driver, so they are left at their natural observed values in every
  scenario here.

=== Scenarios (fixed BEFORE any result was examined) ===

  A (behavioral signal only, both dominant balance features neutral):
    fresh_account + neutral_mismatch + neutral_after_zero,
    synthetic_stress_label = 1.
  B (behavioral signal, normal/neutral after-zero, mismatch left as
    naturally observed on the source row):
    fresh_account + neutral_after_zero, synthetic_stress_label = 1.
  C (altered balance pattern only, no behavioral change):
    altered_balance_drain, synthetic_stress_label = 1.
  D (non-fraud distractor: behavioral signal only, balance features
    left exactly as observed -- i.e. ordinary, legitimate):
    fresh_account, synthetic_stress_label = 0.

=== Source-row selection (deterministic, predetermined) ===

Source rows are genuinely legitimate (`isFraud == 0`) TEST-period
CASH_OUT/TRANSFER rows (the only two fraud-eligible types, re-verified
in Milestones 12/13). For each (day, type) cell among days
{27, 28, 29} x {CASH_OUT, TRANSFER}, the first `SAMPLE_PER_CELL` (500,
fixed before any result was examined; the smallest available cell --
day 29 TRANSFER, 880 legitimate rows -- comfortably exceeds it) such
rows IN FILE ORDER are selected -- no random sampling is needed or
used, satisfying this milestone's preference for deterministic
selection "by row_id or file order" even more strictly than a seeded
draw would.

=== Day-30 degeneracy (never silently dropped) ===

Day 30 has ZERO legitimate CASH_OUT/TRANSFER test rows (Milestone 4/9's
simulation-boundary artifact: day 30 is 100% fraud). SOURCE_DAYS
therefore excludes 30 -- there is mechanically nothing to source a
synthetic example FROM on that day. This is not this module hiding
day 30: `src.robustness_analysis.daily_breakdown` still iterates over
all of `TEST_DAYS` (27-30), so every scenario's `daily` breakdown below
explicitly contains a day-30 entry with `n_total: 0` and every metric
`null` with a documented reason -- the absence is visible, not
silently omitted.

=== Leakage / safety ===

`isFraud` is read only to confirm `== 0` (a data property, never a
tuning input); model predictions are never consulted before finalizing
which rows or transformations to use; `SCENARIO_DEFINITIONS`,
`SAMPLE_PER_CELL`, and `RESIDUAL_FRACTION` are fixed module-level
constants, decided before this module was ever run against real
outputs. The fitted imputer/scaler/model are each created exactly once
and never refit on synthetic or test data.
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

from src import ablation_analysis as ab
from src import baseline_model as bmod
from src import robustness_analysis as ra
from src import stress_test_analysis as st
from src.validate_raw_paysim import BALANCE_TOLERANCE, DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = bmod.DEFAULT_CHUNKSIZE
RANDOM_STATE = bmod.RANDOM_STATE  # 42, unchanged; not actually needed for selection
# (deterministic file-order selection is used instead -- see module docstring),
# retained here only to document the project's standing convention.
DEFAULT_THRESHOLD = bmod.DEFAULT_THRESHOLD  # 0.5, unchanged, never tuned here
SAFETY_RSS_LIMIT_MB = bmod.SAFETY_RSS_LIMIT_MB

DOMINANT_BALANCE_FEATURES: list[str] = list(ab.BALANCE_FEATURES_REMOVED)
FEATURE_COLUMNS: list[str] = bmod.FEATURE_COLUMNS
IDX: dict[str, int] = {c: FEATURE_COLUMNS.index(c) for c in FEATURE_COLUMNS}

# Predetermined BEFORE any result was evaluated.
SAMPLE_PER_CELL: int = 500
RESIDUAL_FRACTION: float = 0.05  # Recipe C: leave a small, nonzero residual balance

SOURCE_TYPES: list[str] = list(ra.FRAUD_ELIGIBLE_TYPES)  # ["CASH_OUT", "TRANSFER"]
SOURCE_DAYS: list[int] = [d for d in ra.TEST_DAYS if d != ra.SIMULATION_BOUNDARY_DAY]  # [27, 28, 29]

SCENARIO_DEFINITIONS: dict[str, dict[str, Any]] = {
    "A_behavioral_without_balance_signature": {
        "recipes": ["fresh_account", "neutral_mismatch", "neutral_after_zero"],
        "synthetic_stress_label": 1,
        "description": (
            "Legitimate CASH_OUT/TRANSFER transaction made to look "
            "fraud-like purely via the fresh-account behavioral "
            "signature, with BOTH dominant balance features forced to "
            "their neutral/reference value -- isolates whether the "
            "model reacts to a non-balance behavioral anomaly alone."
        ),
    },
    "B_behavioral_with_normal_after_zero": {
        "recipes": ["fresh_account", "neutral_after_zero"],
        "synthetic_stress_label": 1,
        "description": (
            "Same fresh-account behavioral signature, but only "
            "`orig_balance_after_zero` is forced neutral; "
            "`orig_balance_mismatch` is left exactly as naturally "
            "observed on the source row (not forced either way)."
        ),
    },
    "C_altered_balance_pattern": {
        "recipes": ["altered_balance_drain"],
        "synthetic_stress_label": 1,
        "description": (
            "No behavioral change. `orig_balance_after` is overwritten "
            "to a small nonzero residual fraction of the preserved "
            "`orig_balance_before` -- a different balance-draining "
            "pattern than PaySim's own typical exact-zero signature -- "
            "with all dependent columns recomputed consistently."
        ),
    },
    "D_non_fraud_distractor": {
        "recipes": ["fresh_account"],
        "synthetic_stress_label": 0,
        "description": (
            "Fresh-account behavioral signature applied to an "
            "otherwise perfectly ordinary legitimate transaction; both "
            "dominant balance features are left exactly as naturally "
            "observed (i.e. legitimate). Tests whether the model "
            "overreacts to the behavioral pattern alone, in the "
            "ABSENCE of the dominant PaySim balance signature."
        ),
    },
}

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SYNTH_SPEC_PATH = REPORTS_DIR / "synthetic_generalization_specification.md"
SYNTH_RESULTS_PATH = REPORTS_DIR / "synthetic_generalization_results.json"
SYNTH_DAILY_CSV_PATH = REPORTS_DIR / "synthetic_generalization_daily.csv"
SYNTH_BY_TYPE_CSV_PATH = REPORTS_DIR / "synthetic_generalization_by_type.csv"


def apply_fresh_account_recipe(X: np.ndarray) -> np.ndarray:
    """Recipe `fresh_account`. Returns a NEW array (never mutates the
    input). See module docstring "Transformation selection".
    """
    X = X.copy()
    X[:, IDX["origin_is_first_seen"]] = 1.0
    X[:, IDX["origin_txn_count_causal"]] = 0.0
    X[:, IDX["origin_total_amount_causal"]] = 0.0
    X[:, IDX["origin_avg_amount_causal"]] = np.nan
    X[:, IDX["amount_vs_origin_avg_ratio"]] = np.nan
    return X


def apply_neutral_balance_feature(X: np.ndarray, column: str, neutral_value: float) -> np.ndarray:
    """Recipes `neutral_mismatch` / `neutral_after_zero`. Standalone
    {0,1} indicator overwrite -- self-consistent by construction, since
    no other column in the contract is defined as a function of either
    named column.
    """
    X = X.copy()
    X[:, IDX[column]] = neutral_value
    return X


def apply_altered_balance_drain(X: np.ndarray, residual_fraction: float = RESIDUAL_FRACTION) -> np.ndarray:
    """Recipe `altered_balance_drain`. `amount` and `orig_balance_before`
    are read but never modified. `orig_balance_after` is overwritten;
    every dependent column is recomputed with the SAME formula
    `src.features_raw_paysim.compute_stateless_features` uses, so the
    result is always a state that formula itself could produce.
    """
    X = X.copy()
    before = X[:, IDX["orig_balance_before"]]
    amount = X[:, IDX["amount"]]

    new_after = before * residual_fraction
    X[:, IDX["orig_balance_after"]] = new_after
    X[:, IDX["orig_balance_delta"]] = new_after - before
    X[:, IDX["orig_balance_after_zero"]] = (new_after == 0).astype(np.float32)

    is_cash_in = X[:, IDX["type_CASH_IN"]] == 1.0
    sign = np.where(is_cash_in, 1.0, -1.0)
    expected_after = before + sign * amount
    mismatch = (~np.isclose(new_after, expected_after, atol=BALANCE_TOLERANCE)).astype(np.float32)
    X[:, IDX["orig_balance_mismatch"]] = mismatch
    return X


_RECIPE_FUNCTIONS = {
    "fresh_account": lambda X, neutral_values: apply_fresh_account_recipe(X),
    "neutral_mismatch": lambda X, neutral_values: apply_neutral_balance_feature(
        X, "orig_balance_mismatch", neutral_values["orig_balance_mismatch"]
    ),
    "neutral_after_zero": lambda X, neutral_values: apply_neutral_balance_feature(
        X, "orig_balance_after_zero", neutral_values["orig_balance_after_zero"]
    ),
    "altered_balance_drain": lambda X, neutral_values: apply_altered_balance_drain(X),
}


def build_synthetic_features(X_source: np.ndarray, recipes: list[str], neutral_values: dict[str, float]) -> np.ndarray:
    X = X_source.copy()
    for recipe_name in recipes:
        X = _RECIPE_FUNCTIONS[recipe_name](X, neutral_values)
    return X


def select_source_row_indices(
    y_test: np.ndarray, day_index_test: np.ndarray, type_test: np.ndarray, day: int, type_name: str, n: int
) -> np.ndarray:
    """Deterministic file-order selection: the first `n` legitimate
    (`isFraud == 0`) rows of the given (day, type) cell. `isFraud` is
    read only to confirm legitimacy, never to pick a "good" example.
    """
    mask = (y_test == 0) & (day_index_test == day) & (type_test == type_name)
    idx = np.nonzero(mask)[0]
    return idx[:n]


def build_scenario_source_set(
    X_test: np.ndarray, y_test: np.ndarray, day_index_test: np.ndarray, type_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """The one shared source-row set (same for every scenario) --
    computed once and reused, since which ROWS are eligible does not
    depend on the scenario's recipe.
    """
    x_parts: list[np.ndarray] = []
    day_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    cell_counts: dict[str, int] = {}

    for day in SOURCE_DAYS:
        for type_name in SOURCE_TYPES:
            idx = select_source_row_indices(y_test, day_index_test, type_test, day, type_name, SAMPLE_PER_CELL)
            cell_counts[f"day{day}_{type_name}"] = int(len(idx))
            if len(idx) == 0:
                continue
            x_parts.append(X_test[idx])
            day_parts.append(day_index_test[idx])
            type_parts.append(type_test[idx])

    if x_parts:
        X_source = np.concatenate(x_parts, axis=0)
        day_source = np.concatenate(day_parts, axis=0)
        type_source = np.concatenate(type_parts, axis=0)
    else:
        X_source = np.empty((0, X_test.shape[1]), dtype=np.float32)
        day_source = np.empty((0,), dtype=day_index_test.dtype)
        type_source = np.empty((0,), dtype=type_test.dtype)

    return X_source, day_source, type_source, cell_counts


def predict_with_fixed_model_safe(imputer, scaler, model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`src.stress_test_analysis.predict_with_fixed_model`, guarded for
    the zero-row case (e.g. a (day, type) cell with no eligible
    legitimate source rows) -- `SimpleImputer`/`StandardScaler` raise
    on an empty array, so this returns empty prediction arrays instead
    of calling the fixed model at all. Never changes behavior for a
    non-empty input.
    """
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return st.predict_with_fixed_model(imputer, scaler, model, X)


def _scenario_metric_bundle(y_synth, proba_synth, preds_synth, day_synth, type_synth) -> dict[str, Any]:
    return {
        "whole_scenario": ra.slice_metrics(y_synth, preds_synth, proba_synth),
        "daily": ra.daily_breakdown(y_synth, proba_synth, preds_synth, day_synth),
        "by_type": ra.fraud_type_breakdown(y_synth, proba_synth, preds_synth, type_synth),
    }


def run_smoke_test(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 300_000
) -> dict[str, Any]:
    """Bounded resource check: fit the model once on a bounded prefix,
    verify the recipe/selection machinery, measure peak RSS.
    """
    rss_before = bmod.peak_rss_mb()
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    day_index_test, type_test = ra.stream_test_metadata(csv_path, chunksize=chunksize, max_rows=max_rows)
    X_train, y_train = full_arrays["train"]
    X_test, y_test = full_arrays["test"]

    result: dict[str, Any] = {
        "max_rows": max_rows,
        "chunksize": chunksize,
        "stream_stats": stream_stats,
        "test_rows_available": int(X_test.shape[0]),
        "model_fit_on_smoke_sample": False,
        "scenario_exercised": False,
        "source_rows_selected": 0,
        "original_values_unchanged": None,
    }

    trainable = X_train.shape[0] > 0 and len(np.unique(y_train)) >= 2
    if trainable:
        imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
        model = bmod.train_baseline_model(X_train_proc, y_train)
        result["model_fit_on_smoke_sample"] = True

        if X_test.shape[0] > 0 and len(day_index_test) == X_test.shape[0]:
            X_source, day_source, type_source, cell_counts = build_scenario_source_set(
                X_test, y_test, day_index_test, type_test
            )
            result["source_rows_selected"] = int(len(X_source))
            if len(X_source) > 0:
                X_source_before = X_source.copy()
                neutral_values = {c: st.neutral_value_for(c, X_train) for c in DOMINANT_BALANCE_FEATURES}
                X_synth = build_synthetic_features(X_source, SCENARIO_DEFINITIONS["A_behavioral_without_balance_signature"]["recipes"], neutral_values)
                _ = st.predict_with_fixed_model(imputer, scaler, model, X_synth)
                result["original_values_unchanged"] = bool(np.array_equal(X_source, X_source_before, equal_nan=True))
                result["scenario_exercised"] = True

    result["runtime_seconds"] = round(time.time() - t0, 2)
    result["rss_before_mb"] = round(rss_before, 1)
    result["peak_rss_mb"] = round(bmod.peak_rss_mb(), 1)
    result["safety_limit_mb"] = SAFETY_RSS_LIMIT_MB
    result["within_safety_limit"] = result["peak_rss_mb"] < SAFETY_RSS_LIMIT_MB
    return result


def run_full_synthetic_analysis(
    csv_path: Path = DEFAULT_RAW_CSV,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)
    day_index_test, type_test = ra.stream_test_metadata(csv_path, chunksize=chunksize, max_rows=max_rows)

    X_train, y_train = full_arrays["train"]
    X_test, y_test = full_arrays["test"]
    assert len(day_index_test) == X_test.shape[0], (
        f"test metadata row count ({len(day_index_test)}) must match the feature stream's "
        f"test bucket ({X_test.shape[0]})"
    )

    # Fit ONCE (Mode 1). Never refit.
    imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
    model = bmod.train_baseline_model(X_train_proc, y_train)

    proba0, preds0 = predict_with_fixed_model_safe(imputer, scaler, model, X_test)
    baseline_check = st.run_baseline_reproduction_check(y_test, proba0, preds0)

    is_full_real_dataset_run = Path(csv_path) == Path(DEFAULT_RAW_CSV) and max_rows is None
    if is_full_real_dataset_run and baseline_check["matches_recorded_confusion_matrix_and_counts"] is False:
        raise RuntimeError(
            "Baseline reproduction check FAILED: the unperturbed Milestone 14 pipeline "
            "does not reproduce Milestone 10/11/12/13's recorded FULL_BASELINE test "
            "metrics. Stopping before synthetic evaluation, per this milestone's "
            "explicit instruction to diagnose discrepancies before proceeding."
        )

    original_scenario = {
        "whole_test_period": ra.slice_metrics(y_test, preds0, proba0),
        "daily": ra.daily_breakdown(y_test, proba0, preds0, day_index_test),
        "by_type": ra.fraud_type_breakdown(y_test, proba0, preds0, type_test),
    }

    neutral_values = {c: st.neutral_value_for(c, X_train) for c in DOMINANT_BALANCE_FEATURES}

    # Source rows are the SAME across every scenario -- computed once.
    X_source, day_source, type_source, cell_counts = build_scenario_source_set(
        X_test, y_test, day_index_test, type_test
    )
    X_source_snapshot = X_source.copy()  # for the "original values unchanged" check below

    synthetic_scenarios: dict[str, Any] = {}
    for scenario_name, spec in SCENARIO_DEFINITIONS.items():
        X_synth = build_synthetic_features(X_source, spec["recipes"], neutral_values)
        proba_s, preds_s = predict_with_fixed_model_safe(imputer, scaler, model, X_synth)
        y_synth = np.full(len(X_synth), spec["synthetic_stress_label"], dtype=np.int64)

        synthetic_scenarios[scenario_name] = {
            "recipes": spec["recipes"],
            "synthetic_stress_label": spec["synthetic_stress_label"],
            "description": spec["description"],
            "n_examples": int(len(X_synth)),
            "cell_counts": cell_counts,
            **_scenario_metric_bundle(y_synth, proba_s, preds_s, day_source, type_source),
        }
        del X_synth  # release before the next scenario

    source_rows_unchanged = bool(np.array_equal(X_source, X_source_snapshot, equal_nan=True))

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
        "source_days": SOURCE_DAYS,
        "source_types": SOURCE_TYPES,
        "sample_per_cell": SAMPLE_PER_CELL,
        "residual_fraction": RESIDUAL_FRACTION,
        "dominant_balance_features": DOMINANT_BALANCE_FEATURES,
        "neutral_values": neutral_values,
        "source_cell_counts": cell_counts,
        "source_rows_total": int(len(X_source)),
        "source_rows_unchanged_after_all_scenarios": source_rows_unchanged,
        "day_30_synthetic_note": (
            "Day 30 has zero legitimate CASH_OUT/TRANSFER test rows (100% fraud -- "
            "Milestone 4/9 simulation-boundary artifact), so no synthetic example can "
            "be sourced from it. Every scenario's 'daily' breakdown below still "
            "contains an explicit day-30 entry (n_total=0, every metric null with a "
            "documented reason) via src.robustness_analysis.daily_breakdown's "
            "unconditional iteration over all four test-period days -- this is a "
            "mechanical consequence of source-row availability, not an omission."
        ),
        "train_rows": int(X_train.shape[0]),
        "train_fraud": int(y_train.sum()),
        "test_rows": int(X_test.shape[0]),
        "test_fraud": int(y_test.sum()),
        "stream_stats": stream_stats,
        "baseline_reproduction_check": baseline_check,
        "original_scenario": original_scenario,
        "synthetic_scenarios": synthetic_scenarios,
    }


def write_daily_csv(results: dict[str, Any], out_path: Path = SYNTH_DAILY_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "synthetic_stress_label", "day_index", "is_simulation_boundary_day",
        "n_total", "n_alerts", "tp", "fp", "fn", "tn", "precision", "recall",
        "average_precision_pr_auc", "roc_auc_secondary",
    ]

    def rows_for(name: str, label: Any, daily: dict[str, Any]):
        for day_str, m in daily.items():
            cm = m["confusion_matrix"]
            yield {
                "scenario": name, "synthetic_stress_label": label, "day_index": day_str,
                "is_simulation_boundary_day": m["is_simulation_boundary_day"],
                "n_total": m["n_total"], "n_alerts": m["n_alerts"],
                "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
                "precision": m["precision"], "recall": m["recall"],
                "average_precision_pr_auc": m["average_precision_pr_auc"],
                "roc_auc_secondary": m["roc_auc_secondary"],
            }

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for("original_test_period", "n/a (real isFraud)", results["original_scenario"]["daily"]):
            writer.writerow(row)
        for name, scenario in results["synthetic_scenarios"].items():
            for row in rows_for(name, scenario["synthetic_stress_label"], scenario["daily"]):
                writer.writerow(row)


def write_by_type_csv(results: dict[str, Any], out_path: Path = SYNTH_BY_TYPE_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "synthetic_stress_label", "type", "n_total", "n_alerts",
        "tp", "fp", "fn", "tn", "precision", "recall",
        "average_precision_pr_auc", "roc_auc_secondary",
    ]

    def rows_for(name: str, label: Any, by_type: dict[str, Any]):
        for type_name, m in by_type.items():
            cm = m["confusion_matrix"]
            yield {
                "scenario": name, "synthetic_stress_label": label, "type": type_name,
                "n_total": m["n_total"], "n_alerts": m["n_alerts"],
                "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"],
                "precision": m["precision"], "recall": m["recall"],
                "average_precision_pr_auc": m["average_precision_pr_auc"],
                "roc_auc_secondary": m["roc_auc_secondary"],
            }

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_for("original_test_period", "n/a (real isFraud)", results["original_scenario"]["by_type"]):
            writer.writerow(row)
        for name, scenario in results["synthetic_scenarios"].items():
            for row in rows_for(name, scenario["synthetic_stress_label"], scenario["by_type"]):
                writer.writerow(row)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_synth_results(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = SYNTH_RESULTS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"smoke_test": smoke_test, **results}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_synth_spec(results: dict[str, Any], smoke_test: dict[str, Any], out_path: Path = SYNTH_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_metric(m: dict[str, Any], key: str, fmt: str = ".4f") -> str:
        v = m.get(key)
        return "null" if v is None else format(v, fmt)

    def fmt_whole(name: str, m: dict[str, Any]) -> str:
        cm = m["confusion_matrix"]
        return (
            f"- **{name}**: n={m['n_total']:,} alerts={m['n_alerts']:,} TP={cm['tp']:,} "
            f"FP={cm['fp']:,} FN={cm['fn']:,} precision={fmt_metric(m, 'precision')} "
            f"recall={fmt_metric(m, 'recall')} PR-AUC={fmt_metric(m, 'average_precision_pr_auc')} "
            f"ROC-AUC={fmt_metric(m, 'roc_auc_secondary')}"
        )

    bc = results["baseline_reproduction_check"]
    lines = [
        "# FraudGraph AI — Synthetic Fraud-Pattern Generalization (Milestone 14)",
        "",
        "**This is a controlled synthetic generalization experiment, "
        "not production validation, not real-world generalization "
        "proof, and not model optimization.** Every synthetic example "
        "is built from a genuinely LEGITIMATE test-period transaction; "
        "its ORIGINAL PaySim label (`isFraud`) is always 0 and is "
        "never changed. A separate `synthetic_stress_label` (1 = "
        "\"constructed to look fraud-like\", 0 = \"constructed as a "
        "legitimate-looking distractor\") is the only label used to "
        "evaluate synthetic scenarios below -- it is never confused "
        "with, or mixed into, `isFraud`.",
        "",
        "## Objective",
        "",
        "Does the fixed Milestone 10 `FULL_BASELINE` detector react to "
        "fraud-like BEHAVIORAL patterns that do not carry PaySim's "
        "dominant balance signature, or is it primarily recognizing "
        "that exact synthetic signature? This is evidence about "
        "SENSITIVITY to the simulated fraud mechanism, not a "
        "real-world generalization test.",
        "",
        "## Model policy: Mode 1 only",
        "",
        "Mode 1 (fixed model, synthetic feature VALUES only, no "
        "retraining) is implemented, as mandated. Mode 2 (optional "
        "retrained variant) is deliberately NOT implemented -- this "
        "milestone's own constraints on Mode 2 (identical train set, "
        "identical model configuration, synthetic examples never added "
        "to training) mean a Mode 2 model would be bit-for-bit "
        "identical to Mode 1's, so it would add a second model-fit's "
        "resource cost for a result already known. See module "
        "docstring for the full reasoning.",
        "",
        "## Transformation selection (fixed before any result was examined)",
        "",
        "Recipes, each internally self-consistent and built only from "
        "existing 39-column features:",
        "",
        "- **fresh_account**: `origin_is_first_seen=1`, "
        "`origin_txn_count_causal=0`, `origin_total_amount_causal=0.0`, "
        "`origin_avg_amount_causal=NaN`, `amount_vs_origin_avg_ratio="
        "NaN` -- the exact state Milestone 5 already defines for a "
        "genuine first-time-seen origin account; encodes a "
        "non-balance, non-PaySim-specific fraud heuristic (freshly "
        "created / mule account).",
        "- **neutral_mismatch** / **neutral_after_zero**: overwrite "
        "`orig_balance_mismatch` / `orig_balance_after_zero` to their "
        "train-only reference value (Milestone 13's `neutral_value_for`, "
        f"measured this run: `{json.dumps(results['neutral_values'])}`).",
        "- **altered_balance_drain**: `orig_balance_after` set to "
        f"`{RESIDUAL_FRACTION:.0%}` of the preserved `orig_balance_"
        "before`; `orig_balance_delta`/`orig_balance_after_zero`/"
        "`orig_balance_mismatch` recomputed with Milestone 5's own "
        "formula so the row is always a state that formula could "
        "itself produce.",
        "",
        "Rejected candidates and why (amount/type columns; destination-"
        "side balance columns; flipping a zero-flag without its "
        "underlying value; the 5 graph-derived features) are documented "
        "in full in the module docstring's \"Transformation selection\" "
        "section.",
        "",
        "## Scenarios",
        "",
    ]
    for name, spec in SCENARIO_DEFINITIONS.items():
        lines.append(f"- **{name}** (synthetic_stress_label={spec['synthetic_stress_label']}, recipes={spec['recipes']}): {spec['description']}")
    lines += [
        "",
        "## Source-row selection",
        "",
        f"Genuinely legitimate (`isFraud == 0`) CASH_OUT/TRANSFER test "
        f"rows, days {SOURCE_DAYS} only (day 30 excluded -- see "
        "\"Day-30 degeneracy\" below). First "
        f"{SAMPLE_PER_CELL} such rows IN FILE ORDER per (day, type) "
        f"cell -- no random sampling used. Actual cell counts this "
        f"run: `{json.dumps(results['source_cell_counts'])}` "
        f"(total {results['source_rows_total']:,} source rows, shared "
        "identically across every scenario).",
        "",
        "## Day-30 degeneracy",
        "",
        results["day_30_synthetic_note"],
        "",
        "## Resource strategy",
        "",
        "The expensive 39-column causal feature stream and the one "
        "model fit each happen exactly once. Every scenario transforms "
        "the SAME small (thousands of rows, not millions), shared "
        "source-row copy and is discarded before the next scenario.",
        "",
        "### Bounded smoke test",
        "",
        f"- Rows: {smoke_test['max_rows']:,}, chunksize {smoke_test['chunksize']:,}",
        f"- Runtime: {smoke_test['runtime_seconds']:.2f}s",
        f"- Peak RSS: {smoke_test['peak_rss_mb']:.0f} MB "
        f"(safety limit: {smoke_test['safety_limit_mb']:,} MB, within "
        f"limit: {smoke_test['within_safety_limit']})",
        f"- Original source values unchanged after transformation: "
        f"{smoke_test['original_values_unchanged']}",
        "",
        "## Baseline reproduction check (must pass before synthetic evaluation)",
        "",
        f"Matches the recorded `FULL_BASELINE` test-period confusion "
        f"matrix/counts exactly: "
        f"**{bc['matches_recorded_confusion_matrix_and_counts']}**.",
        "",
        fmt_whole("unperturbed original test period", bc["recomputed"]),
        "",
        "## Original vs. synthetic results",
        "",
        "Original (real `isFraud`, untouched test rows) and synthetic "
        "(constructed `synthetic_stress_label`) results are reported "
        "separately below and never pooled into one metric.",
        "",
        fmt_whole("ORIGINAL test period (real isFraud)", results["original_scenario"]["whole_test_period"]),
        "",
    ]
    for name, scenario in results["synthetic_scenarios"].items():
        lines.append(
            fmt_whole(
                f"SYNTHETIC {name} (n={scenario['n_examples']:,}, label={scenario['synthetic_stress_label']})",
                scenario["whole_scenario"],
            )
        )
    lines += [
        "",
        "For scenarios A/B/C (`synthetic_stress_label=1`), `recall` is "
        "the fraction of constructed fraud-like examples the fixed "
        "model actually flagged (its \"synthetic fraud capture rate\"); "
        "`precision` is trivially 1.0 whenever any alert fires, "
        "because every row in that scenario shares the same intended "
        "label (a single-class evaluation set, exactly like Milestone "
        "12/13's day-30 slices) -- PR-AUC/ROC-AUC are therefore `null` "
        "for the same documented reason. For scenario D "
        "(`synthetic_stress_label=0`), `precision` is the meaningful "
        "number: it is 0.0 whenever the model fires ANY alert on a "
        "purely legitimate-with-a-behavioral-quirk row (a genuine "
        "\"synthetic false positive\"), and `recall`/PR-AUC/ROC-AUC "
        "are `null` (no positive class exists in that scenario by "
        "construction).",
        "",
        "## Daily and transaction-type detail",
        "",
        "Full breakdown (every scenario x every test-period day, "
        "including the explicit day-30 zero-row entry) is in "
        "`reports/synthetic_generalization_daily.csv`; every scenario "
        "x CASH_OUT/TRANSFER is in "
        "`reports/synthetic_generalization_by_type.csv`. No "
        "independent tuning was performed per day or per type.",
        "",
        "## Comparison",
        "",
        "1. **Original PaySim fraud**: see the ORIGINAL row above -- "
        "the model's real, measured performance on genuinely labeled "
        "test-period fraud.",
        "2. **Synthetic fraud with altered balance signature** "
        "(scenarios A/B/C): see each scenario's capture rate above -- "
        "how often the fixed model still fires when the familiar "
        "PaySim balance tell is absent or altered.",
        "3. **Synthetic legitimate distractors** (scenario D): see its "
        "precision above -- how often the fixed model fires on an "
        "ordinary legitimate transaction carrying only a non-balance "
        "behavioral quirk.",
        "",
        "These three are reported side by side, never ranked "
        "best/worst, to show whether the detector's behavior changes "
        "materially when the familiar PaySim signature is absent -- "
        "not to declare a winner.",
        "",
        "## Interpretation",
        "",
        "Precise language only: \"performance under controlled "
        "synthetic perturbation,\" \"behavior when the PaySim balance "
        "signature is altered,\" \"evidence about sensitivity to the "
        "simulated fraud mechanism.\" This report does NOT claim the "
        "model generalizes to real fraud, detects novel fraud, or is "
        "production-ready -- real-world generalization remains "
        "untested. Synthetic examples are not real fraud; they are "
        "counterfactual evaluation constructs whose usefulness depends "
        "entirely on the transformation assumptions above, and PaySim "
        "itself is a synthetic simulation, not real transaction data.",
        "",
        "## Limitations",
        "",
        "- Every synthetic example is a deterministic transformation "
        "of a genuinely legitimate row; none is drawn from, or "
        "validated against, any real fraud case outside PaySim.",
        "- The specific recipes chosen (fresh-account behavioral "
        "signature; a 5% residual balance drain) are one reasonable, "
        "documented choice among many possible non-balance/altered-"
        "balance transformations -- results are specific to these "
        "choices, not a general statement about all possible "
        "behavioral fraud patterns.",
        "- `SAMPLE_PER_CELL=500` bounds the day-29/TRANSFER cell "
        "(smallest available, 880 legitimate rows) comfortably, but "
        "results are still based on a few thousand constructed rows "
        "per scenario, not the full test period.",
        "- Day 30 has zero eligible source rows by construction; no "
        "synthetic evidence exists for that day (see \"Day-30 "
        "degeneracy\").",
        "- No hyperparameter tuning, threshold tuning, feature "
        "engineering, or model-family comparison was performed "
        "anywhere in this milestone.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 14 synthetic fraud-pattern generalization: "
        "score deterministic behavioral/balance-pattern transformations of "
        "legitimate test rows through the fixed Milestone 10 model."
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

    print("Smoke test passed. Running the full synthetic-generalization analysis...")
    results = run_full_synthetic_analysis(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)

    write_daily_csv(results, SYNTH_DAILY_CSV_PATH)
    write_by_type_csv(results, SYNTH_BY_TYPE_CSV_PATH)
    write_synth_results(results, smoke, SYNTH_RESULTS_PATH)
    write_synth_spec(results, smoke, SYNTH_SPEC_PATH)

    bc = results["baseline_reproduction_check"]
    print(f"Baseline reproduction check passed: {bc['matches_recorded_confusion_matrix_and_counts']}")
    orig = results["original_scenario"]["whole_test_period"]
    print(f"Original test period   precision={orig['precision']:.4f} recall={orig['recall']:.4f}")
    def _fmt(v: float | None) -> str:
        return "null" if v is None else f"{v:.4f}"

    for name, scenario in results["synthetic_scenarios"].items():
        m = scenario["whole_scenario"]
        print(
            f"{name:42s} n={scenario['n_examples']:5d} alerts={m['n_alerts']:5d} "
            f"precision={_fmt(m['precision'])} recall={_fmt(m['recall'])}"
        )
    print(f"Full-run runtime      : {results['runtime_seconds']:.1f}s")
    print(f"Full-run peak RSS     : {results['peak_rss_mb']:.0f} MB")
    print(f"Spec written to       : {SYNTH_SPEC_PATH}")
    print(f"Results written to    : {SYNTH_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
