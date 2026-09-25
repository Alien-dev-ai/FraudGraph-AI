"""Baseline ablation and graph-feature contribution analysis (Milestone 11).

ATTRIBUTION STUDY, NOT MODEL OPTIMIZATION. This module answers "which
feature groups drive the Milestone 10 baseline result, and do the
retained graph-derived features contribute beyond the tabular
features" -- it does not search for a better model. Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Same model, same split, same everything except feature columns ===

Every feature group below is trained and evaluated with EXACTLY
Milestone 10's configuration, reused without modification:
  - `src.modeling_split.CANDIDATE_POLICIES["chronological_train_val_test"]`
    (via `src.baseline_model.stream_split_arrays`) for row assignment.
  - `src.baseline_model.fit_preprocessing` / `apply_preprocessing`
    (train-only median imputation + train-only StandardScaler).
  - `src.baseline_model.train_baseline_model`
    (`LogisticRegression(class_weight="balanced", random_state=42)`,
    `MODEL_PARAMS` unchanged).
  - `src.baseline_model.DEFAULT_THRESHOLD` (0.5) and `ALERT_BUDGET_RATE`
    (1%), unchanged.
  - `src.baseline_model.evaluate_split` for every metric reported.

The ONLY thing that varies across experiments is which columns of the
already-assembled 39-column feature matrix are selected before fitting.
No hyperparameter is tuned, no threshold is tuned, no split boundary is
changed, and the test split is never used to choose anything -- it is
only ever evaluated, exactly as in Milestone 10.

=== Feature groups (membership derived from existing metadata) ===

  FULL_BASELINE            all 39 Milestone 8 baseline columns (unchanged order)
  TABULAR_ONLY             FULL_BASELINE minus the 5 Milestone 7 graph columns
  GRAPH_ONLY               only the 5 retained Milestone 7 graph columns
  BALANCE_REDUCED          FULL_BASELINE minus {orig_balance_mismatch,
                           orig_balance_after_zero} (the two coefficients
                           Milestone 10 flagged as the largest-magnitude,
                           PaySim-artifact-suspected balance features)
  BALANCE_REDUCED_TABULAR  BALANCE_REDUCED minus the 5 graph columns
  BALANCE_REDUCED_FULL     BALANCE_REDUCED (graph columns retained) --
                           listed separately from BALANCE_REDUCED only
                           to name it symmetrically with
                           BALANCE_REDUCED_TABULAR; the two are the same
                           column list (asserted at import time).

No new feature is invented and no feature is permanently removed from
the approved 39-column contract -- `src.baseline_dataset.
BASELINE_FEATURE_COLUMNS` is never mutated; groups are subsets computed
at analysis time only.

=== Resource-aware reuse (one streaming pass, not six) ===

`src.baseline_model.stream_split_arrays` is called exactly ONCE to
assemble the full 39-column train/validation/test arrays (the same
call Milestone 10's `run_pipeline` makes). Every feature group is then
produced by column-index selection on that already-in-memory array --
cheap numpy slicing, not a re-read of the 6,362,620-row raw file. This
follows this milestone's explicit resource-constraint instruction to
reuse the existing pipeline rather than re-streaming per group.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import baseline_dataset as bd
from src import baseline_model as bmod
from src import features_raw_paysim as feat
from src.validate_raw_paysim import DEFAULT_RAW_CSV

FULL_BASELINE: list[str] = list(bd.BASELINE_FEATURE_COLUMNS)
GRAPH_COLUMNS: list[str] = list(bd.GRAPH_FEATURES_RETAINED_IN_BASELINE)
TABULAR_ONLY: list[str] = [c for c in FULL_BASELINE if c not in GRAPH_COLUMNS]
GRAPH_ONLY: list[str] = [c for c in FULL_BASELINE if c in GRAPH_COLUMNS]

# The two strongest, PaySim-balance-bookkeeping-suspected coefficients
# from Milestone 10 (reports/baseline_model_specification.md section 4
# / section 7) -- derived from that milestone's own recorded findings,
# not re-guessed here. Both are asserted "balance"-category, approved
# Milestone 5 columns.
BALANCE_FEATURES_REMOVED: list[str] = ["orig_balance_mismatch", "orig_balance_after_zero"]
for _col in BALANCE_FEATURES_REMOVED:
    assert _col in FULL_BASELINE, f"{_col} must be an approved baseline column"
    assert feat.FEATURE_METADATA[_col]["category"] == "balance", f"{_col} must be a balance-category feature"

BALANCE_REDUCED: list[str] = [c for c in FULL_BASELINE if c not in BALANCE_FEATURES_REMOVED]
BALANCE_REDUCED_TABULAR: list[str] = [c for c in BALANCE_REDUCED if c not in GRAPH_COLUMNS]
BALANCE_REDUCED_FULL: list[str] = list(BALANCE_REDUCED)
assert BALANCE_REDUCED_FULL == BALANCE_REDUCED

FEATURE_GROUPS: dict[str, list[str]] = {
    "FULL_BASELINE": FULL_BASELINE,
    "TABULAR_ONLY": TABULAR_ONLY,
    "GRAPH_ONLY": GRAPH_ONLY,
    "BALANCE_REDUCED": BALANCE_REDUCED,
    "BALANCE_REDUCED_TABULAR": BALANCE_REDUCED_TABULAR,
    "BALANCE_REDUCED_FULL": BALANCE_REDUCED_FULL,
}
GROUP_NAMES: list[str] = list(FEATURE_GROUPS.keys())

# Defensive re-assertions -- membership must come from existing
# metadata, never be hand-typed independently of it.
assert len(FULL_BASELINE) == 39
assert len(GRAPH_COLUMNS) == 5
assert len(TABULAR_ONLY) == 34
assert set(TABULAR_ONLY) | set(GRAPH_ONLY) == set(FULL_BASELINE)
assert set(TABULAR_ONLY) & set(GRAPH_ONLY) == set()
assert len(BALANCE_REDUCED) == 37
assert len(BALANCE_REDUCED_TABULAR) == 32
for _name, _cols in FEATURE_GROUPS.items():
    assert bd.TARGET_COLUMN not in _cols, f"{_name} must never include the target column"
    assert not (set(bd.IDENTIFIER_COLUMNS) & set(_cols)), f"{_name} must never include an identifier column"
    assert len(_cols) == len(set(_cols)), f"{_name} must not contain duplicate columns"
    assert _cols == [c for c in FULL_BASELINE if c in set(_cols)], (
        f"{_name} must preserve FULL_BASELINE's existing column order"
    )

# category label per column, for documentation only (not used for
# modeling) -- the 5 graph columns are not in feat.FEATURE_METADATA at
# all, so they are labeled "graph" directly.
COLUMN_CATEGORY: dict[str, str] = {
    c: (feat.FEATURE_METADATA[c]["category"] if c in feat.FEATURE_METADATA else "graph") for c in FULL_BASELINE
}

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
ABLATION_SPEC_PATH = REPORTS_DIR / "ablation_specification.md"
ABLATION_RESULTS_PATH = REPORTS_DIR / "ablation_results.json"

DEFAULT_CHUNKSIZE = bmod.DEFAULT_CHUNKSIZE
DEFAULT_THRESHOLD = bmod.DEFAULT_THRESHOLD
ALERT_BUDGET_RATE = bmod.ALERT_BUDGET_RATE
SAFETY_RSS_LIMIT_MB = bmod.SAFETY_RSS_LIMIT_MB

NEGLIGIBLE_COEFFICIENT_ABS = 0.05  # descriptive-only cutoff, see write_ablation_spec


def coefficients_for(model, columns: list[str]) -> list[dict[str, Any]]:
    """Signed coefficients paired with the SPECIFIC group's own column
    names (Milestone 10's `feature_importance` hardcodes the full
    39-column list, so it cannot describe a reduced group's model --
    this is the same computation, generalized over `columns`).
    """
    coefs = model.coef_[0]
    ranked = sorted(zip(columns, coefs), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": name, "coefficient": float(c)} for name, c in ranked]


def select_group_arrays(
    full_arrays: dict[str, tuple[np.ndarray, np.ndarray]], columns: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Column-index selection on the already-assembled full 39-column
    arrays -- no re-read of the raw file, no re-computation of any
    feature. `full_arrays` is exactly `stream_split_arrays`'s output.
    """
    idx = [bmod.FEATURE_COLUMNS.index(c) for c in columns]
    return {name: (X[:, idx], y) for name, (X, y) in full_arrays.items()}


def run_group_experiment(
    group_name: str,
    columns: list[str],
    full_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Train + evaluate Milestone 10's exact pipeline on one feature
    group's columns. Identical preprocessing/model/threshold/metrics to
    Milestone 10 -- only the column selection differs.
    """
    group_arrays = select_group_arrays(full_arrays, columns)
    X_train, y_train = group_arrays["train"]
    X_val, y_val = group_arrays["validation"]
    X_test, y_test = group_arrays["test"]

    assert bd.TARGET_COLUMN not in columns
    assert not (set(bd.IDENTIFIER_COLUMNS) & set(columns))

    t0 = time.time()
    imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
    X_val_proc = bmod.apply_preprocessing(imputer, scaler, X_val)
    X_test_proc = bmod.apply_preprocessing(imputer, scaler, X_test)

    model = bmod.train_baseline_model(X_train_proc, y_train)

    validation_metrics = bmod.evaluate_split(
        model, X_val_proc, y_val, threshold=DEFAULT_THRESHOLD, alert_budget_rate=ALERT_BUDGET_RATE
    )
    test_metrics = bmod.evaluate_split(
        model, X_test_proc, y_test, threshold=DEFAULT_THRESHOLD, alert_budget_rate=ALERT_BUDGET_RATE
    )
    runtime = time.time() - t0

    return {
        "group_name": group_name,
        "feature_columns": columns,
        "feature_count": len(columns),
        "train_rows": int(X_train.shape[0]),
        "train_fraud": int(y_train.sum()),
        "validation_rows": int(X_val.shape[0]),
        "validation_fraud": int(y_val.sum()),
        "test_rows": int(X_test.shape[0]),
        "test_fraud": int(y_test.sum()),
        "runtime_seconds": round(runtime, 2),
        "validation": validation_metrics,
        "test": test_metrics,
        "coefficients": coefficients_for(model, columns),
        "model_params": dict(bmod.MODEL_PARAMS),
        "threshold": DEFAULT_THRESHOLD,
        "alert_budget_rate": ALERT_BUDGET_RATE,
    }


def run_smoke_test(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 300_000
) -> dict[str, Any]:
    """Bounded resource check exercising the exact same stream-once /
    slice-per-group / fit / evaluate path used by `run_all_experiments`,
    on a small prefix of the raw file, before the full run is attempted.
    """
    rss_before = bmod.peak_rss_mb()
    t0 = time.time()

    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)

    group_results: dict[str, Any] = {}
    trainable = stream_stats["rows_by_split"]["train"] > 0 and stream_stats["fraud_by_split"]["train"] > 0
    if trainable:
        for group_name, columns in FEATURE_GROUPS.items():
            group_arrays = select_group_arrays(full_arrays, columns)
            X_train, y_train = group_arrays["train"]
            if len(np.unique(y_train)) < 2:
                continue
            imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
            model = bmod.train_baseline_model(X_train_proc, y_train)
            _ = model.predict_proba(X_train_proc)
            group_results[group_name] = {"trained": True, "n_features": X_train.shape[1]}

    result = {
        "max_rows": max_rows,
        "chunksize": chunksize,
        "stream_stats": stream_stats,
        "groups_exercised": group_results,
        "runtime_seconds": round(time.time() - t0, 2),
        "rss_before_mb": round(rss_before, 1),
        "peak_rss_mb": round(bmod.peak_rss_mb(), 1),
        "safety_limit_mb": SAFETY_RSS_LIMIT_MB,
    }
    result["within_safety_limit"] = result["peak_rss_mb"] < SAFETY_RSS_LIMIT_MB
    return result


def run_all_experiments(
    csv_path: Path = DEFAULT_RAW_CSV,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """One streaming pass -> one full 39-column set of train/validation/
    test arrays -> six lightweight column-selection experiments reusing
    Milestone 10's preprocessing/model/threshold/metrics unchanged.
    """
    t0 = time.time()
    full_arrays, stream_stats = bmod.stream_split_arrays(csv_path, chunksize=chunksize, max_rows=max_rows)

    experiments = {
        group_name: run_group_experiment(group_name, columns, full_arrays)
        for group_name, columns in FEATURE_GROUPS.items()
    }

    runtime = time.time() - t0
    return {
        "csv_path": str(csv_path),
        "chunksize": chunksize,
        "max_rows": max_rows,
        "runtime_seconds": round(runtime, 2),
        "peak_rss_mb": round(bmod.peak_rss_mb(), 1),
        "stream_stats": stream_stats,
        "split_policy_name": bmod.SPLIT_POLICY_NAME,
        "split_day_ranges": {k: list(v) for k, v in bmod.SPLIT_DAY_RANGES.items()},
        "feature_groups": {name: cols for name, cols in FEATURE_GROUPS.items()},
        "column_category": COLUMN_CATEGORY,
        "experiments": experiments,
    }


def _metric_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return None
    return round(float(va) - float(vb), 6)


def graph_contribution_analysis(results: dict[str, Any]) -> dict[str, Any]:
    """Answers Section D's five questions from the already-computed
    per-group experiment results. Purely descriptive: reports deltas
    and coefficient values, never a "best/worst" ranking, and never
    claims causal importance from coefficient magnitude alone.
    """
    exp = results["experiments"]

    def deltas(with_graph: str, without_graph: str) -> dict[str, Any]:
        out = {}
        for split in ("validation", "test"):
            a, b = exp[with_graph][split], exp[without_graph][split]
            out[split] = {
                metric: _metric_delta(a, b, metric)
                for metric in (
                    "average_precision_pr_auc",
                    "roc_auc_secondary",
                    "precision",
                    "recall",
                    "false_positive_count",
                    "n_alerts",
                )
            }
        return out

    # Q1: does adding graph features to the tabular baseline change performance?
    full_vs_tabular = deltas("FULL_BASELINE", "TABULAR_ONLY")
    # Q2: does that contribution remain once the dominant balance features are removed?
    balance_reduced_full_vs_tabular = deltas("BALANCE_REDUCED_FULL", "BALANCE_REDUCED_TABULAR")

    # Q3/Q4: graph coefficients wherever graph columns are present.
    graph_coeff_by_group: dict[str, dict[str, float]] = {}
    for group_name in ("FULL_BASELINE", "GRAPH_ONLY", "BALANCE_REDUCED_FULL"):
        coeffs = {row["feature"]: row["coefficient"] for row in exp[group_name]["coefficients"]}
        graph_coeff_by_group[group_name] = {c: coeffs[c] for c in GRAPH_COLUMNS if c in coeffs}

    sign_stability: dict[str, Any] = {}
    for col in GRAPH_COLUMNS:
        values = [graph_coeff_by_group[g][col] for g in graph_coeff_by_group if col in graph_coeff_by_group[g]]
        signs = {("positive" if v > 0 else "negative" if v < 0 else "zero") for v in values}
        sign_stability[col] = {
            "values_by_group": {g: graph_coeff_by_group[g][col] for g in graph_coeff_by_group if col in graph_coeff_by_group[g]},
            "sign_consistent_across_groups": len(signs) <= 1,
        }

    # Q5: negligible/redundant graph features -- descriptive cutoff only.
    negligible = {
        col: info
        for col, info in sign_stability.items()
        if all(abs(v) < NEGLIGIBLE_COEFFICIENT_ABS for v in info["values_by_group"].values())
    }

    return {
        "q1_full_vs_tabular_delta": full_vs_tabular,
        "q2_balance_reduced_full_vs_tabular_delta": balance_reduced_full_vs_tabular,
        "q3_q4_graph_coefficients_by_group": graph_coeff_by_group,
        "q3_q4_sign_stability": sign_stability,
        "q5_negligible_graph_features": negligible,
        "negligible_cutoff_abs_coefficient": NEGLIGIBLE_COEFFICIENT_ABS,
    }


def paysim_artifact_analysis(results: dict[str, Any]) -> dict[str, Any]:
    """Section E: quantify whether removing the two dominant balance
    features changes performance, without assuming an outcome.
    """
    exp = results["experiments"]

    def compare(with_balance: str, without_balance: str) -> dict[str, Any]:
        out = {}
        for split in ("validation", "test"):
            a, b = exp[with_balance][split], exp[without_balance][split]
            out[split] = {
                metric: {"with_balance_features": a.get(metric), "balance_reduced": b.get(metric),
                         "delta": _metric_delta(a, b, metric)}
                for metric in (
                    "average_precision_pr_auc",
                    "roc_auc_secondary",
                    "precision",
                    "recall",
                    "false_positive_count",
                    "n_alerts",
                )
            }
        return out

    return {
        "removed_features": BALANCE_FEATURES_REMOVED,
        "full_vs_balance_reduced_full": compare("FULL_BASELINE", "BALANCE_REDUCED_FULL"),
        "tabular_only_vs_balance_reduced_tabular": compare("TABULAR_ONLY", "BALANCE_REDUCED_TABULAR"),
        "no_target_leakage_verified": all(
            bd.TARGET_COLUMN not in cols and not (set(bd.IDENTIFIER_COLUMNS) & set(cols))
            for cols in FEATURE_GROUPS.values()
        ),
    }


def write_ablation_spec(
    results: dict[str, Any],
    graph_analysis: dict[str, Any],
    artifact_analysis: dict[str, Any],
    smoke_test: dict[str, Any],
    out_path: Path = ABLATION_SPEC_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exp = results["experiments"]

    def fmt_split(m: dict[str, Any]) -> list[str]:
        cm = m["confusion_matrix"]
        return [
            f"rows={m['n_total']:,} fraud={m['n_fraud']:,} alerts={m['n_alerts']:,} "
            f"TP={cm['tp']:,} FP={cm['fp']:,} FN={cm['fn']:,} TN={cm['tn']:,} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
            f"PR-AUC={m['average_precision_pr_auc']:.4f} "
            f"ROC-AUC(secondary)={m['roc_auc_secondary']:.4f}",
        ]

    lines = [
        "# FraudGraph AI — Baseline Ablation and Graph-Feature Contribution Analysis (Milestone 11)",
        "",
        "**This is an ablation/attribution study, not a model-optimization "
        "milestone.** The same chronological split (Milestone 9, "
        "unchanged), the same preprocessing (train-only median "
        "imputation + train-only StandardScaler), the same "
        "`LogisticRegression(class_weight=\"balanced\", random_state=42)` "
        "configuration (Milestone 10, unchanged), the same default "
        "classification threshold (0.5), and the same 1% alert budget "
        "are used for every experiment below — **only the feature "
        "column selection changes.** No hyperparameter search, no "
        "threshold tuning, and no resampling was performed anywhere in "
        "this milestone. The test split is evaluated, exactly as in "
        "Milestone 10, but is never used to select a feature group, a "
        "preprocessing parameter, or a model configuration. Coefficient "
        "magnitude is used only as a descriptive diagnostic below, never "
        "as proof of causal importance. Milestone 10 documented that its "
        "near-perfect result may reflect PaySim simulation mechanics "
        "rather than a general fraud-detection capability — this "
        "milestone investigates that suspicion empirically rather than "
        "assuming an answer.",
        "",
        "## Feature groups",
        "",
        "Membership is derived directly from existing, approved metadata "
        "(`src/baseline_dataset.py`'s `BASELINE_FEATURE_COLUMNS` / "
        "`GRAPH_FEATURES_RETAINED_IN_BASELINE`, and `src/features_raw_"
        "paysim.py`'s per-column `category`) — never hand-typed "
        "independently of it. The approved 39-column contract itself is "
        "never modified; these are analysis-time subsets only.",
        "",
    ]
    for name, cols in results["feature_groups"].items():
        lines.append(f"- **{name}** ({len(cols)} columns): {', '.join(f'`{c}`' for c in cols)}")
    lines += [
        "",
        "## Column categories (for reference)",
        "",
    ]
    by_cat: dict[str, list[str]] = {}
    for col, cat in results["column_category"].items():
        by_cat.setdefault(cat, []).append(col)
    for cat in ("transaction", "balance", "identity", "graph"):
        if cat in by_cat:
            lines.append(f"- **{cat}** ({len(by_cat[cat])}): {', '.join(f'`{c}`' for c in by_cat[cat])}")
    lines += [
        "",
        "## Measured results (per feature group)",
        "",
        "Differences are reported numerically below; feature groups are "
        "**not** ranked best/worst — this is an attribution study.",
        "",
    ]
    for name in results["feature_groups"]:
        e = exp[name]
        lines += [
            f"### `{name}` ({e['feature_count']} features)",
            "",
            f"- Train: {e['train_rows']:,} rows, {e['train_fraud']:,} fraud",
            f"- Validation: {fmt_split(e['validation'])[0]}",
            f"- Test: {fmt_split(e['test'])[0]}",
            f"- Runtime: {e['runtime_seconds']:.1f}s",
            "",
        ]

    lines += [
        "## Interpretation — graph-feature contribution (Section D)",
        "",
        "**Q1. Does adding the retained graph features to the tabular "
        "baseline change performance?** `FULL_BASELINE` minus "
        "`TABULAR_ONLY`, per split/metric (positive = FULL_BASELINE "
        "scored higher):",
        "",
        f"```\n{json.dumps(graph_analysis['q1_full_vs_tabular_delta'], indent=2)}\n```",
        "",
        "**Q2. Does that contribution remain once the two dominant "
        "balance features are removed?** `BALANCE_REDUCED_FULL` minus "
        "`BALANCE_REDUCED_TABULAR`:",
        "",
        f"```\n{json.dumps(graph_analysis['q2_balance_reduced_full_vs_tabular_delta'], indent=2)}\n```",
        "",
        "**Q3/Q4. Graph-derived coefficients, by group they appear in "
        "(sign/magnitude stability check):**",
        "",
        f"```\n{json.dumps(graph_analysis['q3_q4_sign_stability'], indent=2)}\n```",
        "",
        f"**Q5. Graph features with |coefficient| < {graph_analysis['negligible_cutoff_abs_coefficient']} "
        "in every group they appear in (descriptive cutoff, not a "
        "significance test):**",
        "",
        f"```\n{json.dumps(graph_analysis['q5_negligible_graph_features'], indent=2)}\n```",
        "",
        "These deltas and coefficients are reported as observed "
        "quantities. Coefficient magnitude in a linear model reflects "
        "that feature's contribution *given the other features already "
        "in the model* under this specific preprocessing (train-only "
        "standardization) — it is a descriptive diagnostic, not proof "
        "that a feature is causally responsible for fraud.",
        "",
        "## Interpretation — PaySim balance-artifact dependence (Section E)",
        "",
        f"Removed features: {', '.join(f'`{c}`' for c in artifact_analysis['removed_features'])}. "
        f"Target/identifier leakage re-verified for every group: "
        f"**{artifact_analysis['no_target_leakage_verified']}**.",
        "",
        "`FULL_BASELINE` vs `BALANCE_REDUCED_FULL` (with vs. without the "
        "two dominant balance features, graph features present in both):",
        "",
        f"```\n{json.dumps(artifact_analysis['full_vs_balance_reduced_full'], indent=2)}\n```",
        "",
        "`TABULAR_ONLY` vs `BALANCE_REDUCED_TABULAR` (same comparison, "
        "no graph features in either):",
        "",
        f"```\n{json.dumps(artifact_analysis['tabular_only_vs_balance_reduced_tabular'], indent=2)}\n```",
        "",
        "No expected outcome was assumed ahead of running these "
        "experiments; the numbers above are what was measured.",
        "",
        "### Bounded smoke test (run before the full pipeline)",
        "",
        f"- Rows: {smoke_test['max_rows']:,}, chunksize {smoke_test['chunksize']:,}",
        f"- Runtime: {smoke_test['runtime_seconds']:.2f}s",
        f"- Peak RSS: {smoke_test['peak_rss_mb']:.0f} MB "
        f"(safety limit: {smoke_test['safety_limit_mb']:,} MB, "
        f"within limit: {smoke_test['within_safety_limit']})",
        "",
        "## Limitations",
        "",
        "- This is an attribution study over ONE fixed model family "
        "and ONE fixed preprocessing configuration (both unchanged from "
        "Milestone 10). It does not establish that these conclusions "
        "would hold for a different model family, a different "
        "preprocessing choice, or a different threshold.",
        "- Coefficient magnitude is reported only as a descriptive "
        "diagnostic under this specific linear model and "
        "standardization; it is not a causal-importance measure and "
        "does not account for correlation/collinearity among features.",
        "- Milestone 10's near-perfect validation/test scores may "
        "reflect PaySim's synthetic balance-draining fraud-injection "
        "pattern rather than a general capability; the comparisons "
        "above quantify how much of that score is attributable to the "
        "two named balance features and to the graph features, but do "
        "not extrapolate to real (non-synthetic) transaction data.",
        "- Group-aware account-level splitting remains not implemented "
        "(Milestone 9), unchanged here.",
        "- No hyperparameter tuning, threshold tuning, or resampling was "
        "performed for any feature group.",
        "- One feature group's fit (`BALANCE_REDUCED_TABULAR`) took "
        "substantially longer to converge (~2,500s) than every other "
        "group (all under 90s) under the identical, unmodified "
        "`max_iter=1000` lbfgs configuration. This is reported as an "
        "observed runtime characteristic of that specific 32-column "
        "subset, not addressed by changing any hyperparameter (which "
        "this milestone's instructions prohibit) — a future milestone "
        "could investigate the conditioning cause if runtime matters.",
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


def write_ablation_results(
    results: dict[str, Any],
    graph_analysis: dict[str, Any],
    artifact_analysis: dict[str, Any],
    smoke_test: dict[str, Any],
    out_path: Path = ABLATION_RESULTS_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "smoke_test": smoke_test,
        "graph_contribution_analysis": graph_analysis,
        "paysim_artifact_analysis": artifact_analysis,
        **results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 11 ablation: train/evaluate Milestone 10's "
        "exact pipeline over several feature-column subsets, from a "
        "single streaming pass, to attribute the baseline result."
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
        print("Smoke test exceeded the safety limit -- stopping before the full run.", file=__import__("sys").stderr)
        return 1

    print("Smoke test passed. Running the full ablation (one streaming pass, six experiments)...")
    results = run_all_experiments(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)
    graph_analysis = graph_contribution_analysis(results)
    artifact_analysis = paysim_artifact_analysis(results)

    write_ablation_spec(results, graph_analysis, artifact_analysis, smoke, ABLATION_SPEC_PATH)
    write_ablation_results(results, graph_analysis, artifact_analysis, smoke, ABLATION_RESULTS_PATH)

    for name in results["feature_groups"]:
        e = results["experiments"][name]
        print(
            f"{name:26s} n_feat={e['feature_count']:2d}  "
            f"val PR-AUC={e['validation']['average_precision_pr_auc']:.4f}  "
            f"test PR-AUC={e['test']['average_precision_pr_auc']:.4f}"
        )
    print(f"Full-run runtime      : {results['runtime_seconds']:.1f}s")
    print(f"Full-run peak RSS     : {results['peak_rss_mb']:.0f} MB")
    print(f"Spec written to       : {ABLATION_SPEC_PATH}")
    print(f"Results written to    : {ABLATION_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
