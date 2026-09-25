"""Modeling dataset strategy and split design (Milestone 9).

DESIGN AND VALIDATION ONLY. No model is trained, no scaler/imputer/
encoder is fit, no threshold is tuned, no resampling is performed. Reads
only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Label-audit scope note (isFraud) ===

This module reads `isFraud` ONLY to quantify fraud counts/rates per
temporal region for split-design purposes (`SPLIT_AUDIT_COLUMNS` below:
`step`, `type`, `isFraud`). This is a label-distribution AUDIT, not
feature engineering — `isFraud` is never joined to, or exposed as, a
model input column. `src/baseline_dataset.py`'s `BASELINE_FEATURE_COLUMNS`
(imported here only for the target/identifier-exclusion tests) already
guarantees `isFraud` is absent from every feature-producing module in
this project; this module does not change that guarantee, it only adds
a clearly separate label/audit computation alongside it.

=== Causal / same-step convention (reused unchanged) ===

Identical to Milestones 5, 7, and 8: file/row order is the event order;
within a shared PaySim `step`, an earlier file row is prior. Verified
non-decreasing by `step` in Milestone 5 (re-confirmed in `verify_step_
ordering` below). A day bucket is `day_index = (step - 1) // 24`
(Milestone 5's convention, reused unchanged — `step` itself remains
excluded from the modeling feature contract per that existing policy;
it is used here only to DEFINE region boundaries for splitting, not
reintroduced as a feature).

=== Key finding driving this milestone's split design ===

Milestone 4 flagged the final day (day 30: 272 rows, 100% fraud) as a
"simulation-boundary artifact." Re-verified and EXTENDED here: fraud
COUNT is remarkably stable across all 31 days (~210-320/day), while
LEGITIMATE volume swings from ~1,070 (day 2) to ~575,000 (day 0) rows/
day, independent of the fraud count. This means several MID-simulation
days (2, 4, 18, 25, 26 among them), not just the final day, show
inflated fraud RATES purely from low legitimate volume — a broader
version of the Milestone 4 finding, not a boundary-only phenomenon. See
`compute_per_day_stats` output and `reports/modeling_split_summary.json`
for the exact per-day numbers this conclusion is based on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingest import DatasetNotFoundError, SchemaValidationError, load_csv, validate_columns
from src.validate_raw_paysim import DEFAULT_RAW_CSV
from src.baseline_dataset import BASELINE_FEATURE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN

DEFAULT_CHUNKSIZE = 200_000
STEPS_PER_DAY = 24  # matches src.features_raw_paysim's day_index convention

# LABEL-AUDIT ONLY -- see module docstring. Never used to build a
# feature; only to quantify fraud counts/rates per temporal region.
SPLIT_AUDIT_COLUMNS = ["step", "type", TARGET_COLUMN]

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
SPLIT_SPEC_PATH = REPORTS_DIR / "modeling_split_specification.md"
SPLIT_SUMMARY_PATH = REPORTS_DIR / "modeling_split_summary.json"


def verify_step_ordering(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    """Bounded streaming re-verification that the raw file is
    non-decreasing by step, and quantification of same-step ties.
    Holds only a handful of running scalars -- never retains rows.
    """
    total_rows = 0
    same_step_as_prev = 0
    is_non_decreasing = True
    prev_last_step: int | None = None
    min_step: int | None = None
    max_step: int | None = None

    with pd.read_csv(csv_path, usecols=["step"], chunksize=chunksize) as reader:
        for chunk in reader:
            steps = chunk["step"].to_numpy()
            if not (steps[1:] >= steps[:-1]).all():
                is_non_decreasing = False
            if prev_last_step is not None and steps[0] < prev_last_step:
                is_non_decreasing = False
            same_step_as_prev += int((steps[1:] == steps[:-1]).sum())
            if prev_last_step is not None and steps[0] == prev_last_step:
                same_step_as_prev += 1

            cur_min, cur_max = int(steps.min()), int(steps.max())
            min_step = cur_min if min_step is None else min(min_step, cur_min)
            max_step = cur_max if max_step is None else max(max_step, cur_max)
            prev_last_step = int(steps[-1])
            total_rows += len(chunk)

    return {
        "total_rows": total_rows,
        "is_non_decreasing_by_step": is_non_decreasing,
        "min_step": min_step,
        "max_step": max_step,
        "unique_step_count": (max_step - min_step + 1) if (min_step is not None) else 0,
        "rows_sharing_step_with_prior_row": same_step_as_prev,
        "rows_sharing_step_with_prior_row_pct": (
            same_step_as_prev / total_rows if total_rows else 0.0
        ),
    }


def compute_per_day_stats(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE
) -> dict[int, dict[str, Any]]:
    """One bounded streaming pass over the raw file computing, per
    day_index bucket: total_count, fraud_count, type_counts, min_step,
    max_step. Accumulates only a ~31-entry dict (day_index -> stats),
    never retains rows.
    """
    per_day: dict[int, dict[str, Any]] = {}

    with pd.read_csv(csv_path, usecols=SPLIT_AUDIT_COLUMNS, chunksize=chunksize) as reader:
        for chunk in reader:
            day = ((chunk["step"] - 1) // STEPS_PER_DAY).astype(int)
            for d, sub in chunk.groupby(day):
                entry = per_day.setdefault(int(d), {
                    "total_count": 0, "fraud_count": 0, "type_counts": {},
                    "min_step": None, "max_step": None,
                })
                entry["total_count"] += len(sub)
                entry["fraud_count"] += int(sub[TARGET_COLUMN].sum())
                for t, cnt in sub["type"].value_counts().items():
                    entry["type_counts"][t] = entry["type_counts"].get(t, 0) + int(cnt)
                smin, smax = int(sub["step"].min()), int(sub["step"].max())
                entry["min_step"] = smin if entry["min_step"] is None else min(entry["min_step"], smin)
                entry["max_step"] = smax if entry["max_step"] is None else max(entry["max_step"], smax)

    for entry in per_day.values():
        entry["fraud_rate"] = entry["fraud_count"] / entry["total_count"] if entry["total_count"] else None

    return per_day


def summarize_region(
    per_day_stats: dict[int, dict[str, Any]], day_start: int, day_end: int
) -> dict[str, Any]:
    """Aggregate the (small, precomputed) per-day table over an
    inclusive day range [day_start, day_end] into one region summary.
    Pure aggregation over ~31 entries -- no file access, no row
    retention.
    """
    days_in_range = [d for d in per_day_stats if day_start <= d <= day_end]
    total_rows = sum(per_day_stats[d]["total_count"] for d in days_in_range)
    fraud_rows = sum(per_day_stats[d]["fraud_count"] for d in days_in_range)
    type_counts: dict[str, int] = {}
    for d in days_in_range:
        for t, c in per_day_stats[d]["type_counts"].items():
            type_counts[t] = type_counts.get(t, 0) + c
    steps = [per_day_stats[d]["min_step"] for d in days_in_range] + [
        per_day_stats[d]["max_step"] for d in days_in_range
    ]

    return {
        "day_range": [day_start, day_end],
        "days_covered": sorted(days_in_range),
        "total_rows": total_rows,
        "fraud_rows": fraud_rows,
        "fraud_rate": fraud_rows / total_rows if total_rows else None,
        "type_counts": type_counts,
        "step_range": [min(steps), max(steps)] if steps else None,
        "has_zero_fraud": fraud_rows == 0,
        "is_fraud_only": total_rows > 0 and fraud_rows == total_rows,
    }


# Candidate policies, expressed as inclusive day_index ranges. Boundaries
# chosen from compute_per_day_stats' actual output (see module docstring
# and reports/modeling_split_summary.json), not picked before looking at
# the data.
CANDIDATE_POLICIES: dict[str, dict[str, tuple[int, int]]] = {
    "chronological_holdout_80_20": {
        "train": (0, 23),
        "test": (24, 30),
    },
    "chronological_train_val_test": {
        "train": (0, 23),
        "validation": (24, 26),
        "test": (27, 30),
    },
    "naive_tail_only_test": {
        # Deliberately included as a NEGATIVE example: a short tail
        # window ending exactly at day 30 in isolation.
        "train": (0, 29),
        "test": (30, 30),
    },
    "random_stratified_diagnostic": {
        # Not a temporal region at all -- placeholder entry documented
        # as a non-chronological comparison; see write_split_spec.
    },
}


def evaluate_policy(
    per_day_stats: dict[int, dict[str, Any]], policy_name: str
) -> dict[str, Any]:
    ranges = CANDIDATE_POLICIES[policy_name]
    if not ranges:  # random_stratified_diagnostic has no day ranges
        return {"policy": policy_name, "regions": {}, "note": "not a temporal region-based policy"}
    return {
        "policy": policy_name,
        "regions": {name: summarize_region(per_day_stats, d0, d1) for name, (d0, d1) in ranges.items()},
    }


def evaluate_all_policies(per_day_stats: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {name: evaluate_policy(per_day_stats, name) for name in CANDIDATE_POLICIES}


def check_no_overlap_and_full_accounting(
    per_day_stats: dict[int, dict[str, Any]], policy_name: str
) -> dict[str, Any]:
    """For a day-range-based policy, verify regions don't overlap and
    that the boundaries are internally consistent (does not require
    every day 0..30 to be covered -- a policy may deliberately hold out
    days, which is itself reported).
    """
    ranges = CANDIDATE_POLICIES[policy_name]
    if not ranges:
        return {"policy": policy_name, "applicable": False}

    seen_days: set[int] = set()
    overlap_days: set[int] = set()
    for _, (d0, d1) in ranges.items():
        region_days = set(range(d0, d1 + 1))
        overlap_days |= (seen_days & region_days)
        seen_days |= region_days

    all_days = set(per_day_stats.keys())
    covered_total_rows = sum(
        per_day_stats[d]["total_count"] for d in seen_days if d in per_day_stats
    )
    grand_total_rows = sum(v["total_count"] for v in per_day_stats.values())

    return {
        "policy": policy_name,
        "applicable": True,
        "no_overlap": len(overlap_days) == 0,
        "overlap_days": sorted(overlap_days),
        "days_covered_by_policy": sorted(seen_days),
        "days_in_dataset_not_covered": sorted(all_days - seen_days),
        "rows_covered_by_policy": covered_total_rows,
        "rows_in_dataset": grand_total_rows,
        "rows_accounted_pct": covered_total_rows / grand_total_rows if grand_total_rows else 0.0,
    }


def assess_group_split_feasibility(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    """Bounded streaming check of whether account-level (origin or
    destination) group-splitting is likely to be meaningful, without
    implementing an actual group split. Reuses only cheap running
    counters (account -> count), matching the same technique already
    validated in Milestones 4/6/7 at full scale.
    """
    origin_counts: dict[str, int] = {}
    dest_counts: dict[str, int] = {}
    total_rows = 0

    with pd.read_csv(csv_path, usecols=["nameOrig", "nameDest"], chunksize=chunksize) as reader:
        for chunk in reader:
            for name, cnt in chunk["nameOrig"].value_counts().items():
                origin_counts[name] = origin_counts.get(name, 0) + int(cnt)
            for name, cnt in chunk["nameDest"].value_counts().items():
                dest_counts[name] = dest_counts.get(name, 0) + int(cnt)
            total_rows += len(chunk)

    origin_repeat_accounts = sum(1 for c in origin_counts.values() if c > 1)
    dest_repeat_accounts = sum(1 for c in dest_counts.values() if c > 1)

    return {
        "total_rows": total_rows,
        "unique_origin_accounts": len(origin_counts),
        "unique_destination_accounts": len(dest_counts),
        "origin_accounts_with_multiple_transactions": origin_repeat_accounts,
        "origin_repeat_rate": origin_repeat_accounts / len(origin_counts) if origin_counts else 0.0,
        "destination_accounts_with_multiple_transactions": dest_repeat_accounts,
        "destination_repeat_rate": dest_repeat_accounts / len(dest_counts) if dest_counts else 0.0,
    }


def write_split_spec(
    per_day_stats: dict[int, dict[str, Any]],
    policy_evaluations: dict[str, Any],
    group_feasibility: dict[str, Any],
    step_verification: dict[str, Any],
    out_path: Path = SPLIT_SPEC_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_region(name: str, r: dict[str, Any]) -> list[str]:
        if not r:
            return [f"- **{name}**: not a temporal region"]
        if r["total_rows"] == 0 or r["step_range"] is None:
            return [
                f"- **{name}** (days {r['day_range'][0]}-{r['day_range'][1]}): "
                "no rows in this range for the input analyzed"
            ]
        flag = ""
        if r["has_zero_fraud"]:
            flag = " — **ZERO FRAUD, unusable for evaluation**"
        elif r["is_fraud_only"]:
            flag = " — **100% FRAUD, degenerate label distribution, unusable in isolation**"
        return [
            f"- **{name}** (days {r['day_range'][0]}-{r['day_range'][1]}, "
            f"steps {r['step_range'][0]}-{r['step_range'][1]}): "
            f"{r['total_rows']:,} rows, {r['fraud_rows']:,} fraud "
            f"({r['fraud_rate']:.4%}){flag}",
        ]

    lines = [
        "# FraudGraph AI — Modeling Split Specification (Milestone 9)",
        "",
        "DESIGN AND VALIDATION ONLY. No model was trained; no scaler/"
        "imputer/encoder was fit; no threshold was tuned; no resampling "
        "was performed. `isFraud` is read ONLY for label-distribution "
        "audit in this document (never as a feature) — see "
        "`src/modeling_split.py` docstring.",
        "",
        "## A. Evaluation objective",
        "",
        "Fraud is a severe minority class (8,213 of 6,362,620 raw "
        "transactions, 0.129% — Milestones 4/5). Accuracy is unsuitable: "
        "a trivial always-legitimate classifier scores 99.87% accuracy "
        "while catching zero fraud. Candidate metrics for the eventual "
        "modeling milestone (none computed here):",
        "",
        "- **Average precision / PR-AUC** — summarizes the precision-"
        "recall tradeoff across all thresholds; the right primary "
        "summary metric under severe imbalance, since it is not "
        "inflated by the large true-negative count the way ROC-AUC can "
        "be.",
        "- **Precision at a fixed alert budget** (e.g. top-K flagged "
        "transactions per day) — the operational question a fraud team "
        "actually faces: 'if we can only review N alerts, what fraction "
        "are real fraud?'",
        "- **Recall at a fixed alert budget** — 'of all fraud, how much "
        "do we catch within that same review budget?' — the direct "
        "complement, since budget is usually the true constraint (review "
        "staff/time), not a probability threshold.",
        "- **Recall at a fixed precision target** — for compliance/cost "
        "framings where a maximum tolerable false-positive rate is set "
        "first (e.g. 'no more than 1 false alarm per 10 real fraud "
        "cases') and recall is then read off at that operating point.",
        "- **Confusion matrix counts** (TP/FP/FN/TN) — the raw counts "
        "underlying every ratio above; always reported alongside rates "
        "so a rate is never presented without its denominator.",
        "- **Fraud capture rate** — recall, framed operationally as "
        "'percent of total fraud dollar/count volume caught'.",
        "- **False-positive burden** — the absolute count of legitimate "
        "transactions flagged, since with ~6.35M legitimate "
        "transactions even a 0.1% false-positive rate is thousands of "
        "alerts — a rate alone hides the operational review cost.",
        "- **ROC-AUC** — retained only as a SECONDARY metric. Under "
        "0.13% prevalence, ROC-AUC is dominated by the huge true-"
        "negative count and can look excellent even when precision at "
        "any usable alert budget is poor; never the sole metric.",
        "",
        "## B. Temporal structure",
        "",
        f"Verified via bounded streaming pass (not reused from an old "
        f"report): non-decreasing by step = "
        f"**{step_verification['is_non_decreasing_by_step']}**, over "
        f"{step_verification['total_rows']:,} rows, steps "
        f"{step_verification['min_step']}-{step_verification['max_step']} "
        f"({step_verification['unique_step_count']} unique step values). "
        f"{step_verification['rows_sharing_step_with_prior_row']:,} rows "
        f"({step_verification['rows_sharing_step_with_prior_row_pct']:.4%}) "
        f"share their step value with the immediately preceding row — "
        f"same-step file-order causality (Milestone 5's convention: an "
        f"earlier file row is prior when steps tie) governs the vast "
        f"majority of the dataset, not an edge case.",
        "",
        "**Key finding (extends Milestone 4's 'simulation-boundary "
        "artifact')**: fraud COUNT is stable across all 31 days "
        "(~210-320/day: min "
        f"{min(v['fraud_count'] for v in per_day_stats.values())}, max "
        f"{max(v['fraud_count'] for v in per_day_stats.values())}), while "
        "LEGITIMATE volume swings from "
        f"{min(v['total_count'] for v in per_day_stats.values()):,} to "
        f"{max(v['total_count'] for v in per_day_stats.values()):,} "
        "rows/day independent of fraud count. This means several "
        "MID-simulation days, not only the final day, show inflated "
        "fraud rates purely from low legitimate volume:",
        "",
    ]
    for d in sorted(per_day_stats):
        v = per_day_stats[d]
        if v["fraud_rate"] and v["fraud_rate"] > 0.01:
            lines.append(
                f"  - day {d}: {v['total_count']:,} total, {v['fraud_count']:,} "
                f"fraud, rate {v['fraud_rate']:.4%}"
            )
    lines += [
        "",
        "`step`/`day_index` are used here only to DEFINE region "
        "boundaries — neither is reintroduced as a model feature; "
        "Milestone 5's exploratory_only classification for both stands "
        "unchanged.",
        "",
        "## C. Candidate split policies",
        "",
    ]

    for policy_name, evaluation in policy_evaluations.items():
        lines.append(f"### `{policy_name}`")
        if not evaluation["regions"]:
            lines.append("Not a temporal region-based policy — see discussion below.")
            lines.append("")
            continue
        for region_name, region in evaluation["regions"].items():
            lines += fmt_region(region_name, region)
        lines.append("")

    lines += [
        "**Group-aware splitting** — assessed, not implemented:",
        f"- {group_feasibility['unique_origin_accounts']:,} unique origin "
        f"accounts; only "
        f"{group_feasibility['origin_accounts_with_multiple_transactions']:,} "
        f"({group_feasibility['origin_repeat_rate']:.4%}) ever transact "
        "more than once (Milestones 4/6 confirmed this at full scale). "
        "Group-splitting by origin account would be nearly "
        "indistinguishable from a random row split, since almost every "
        "origin contributes exactly one row to begin with — it buys "
        "essentially no leakage protection beyond what chronological "
        "ordering already gives, for real implementation cost (ensuring "
        "no origin straddles a boundary). Not implemented.",
        f"- {group_feasibility['unique_destination_accounts']:,} unique "
        "destination accounts; "
        f"{group_feasibility['destination_accounts_with_multiple_transactions']:,} "
        f"({group_feasibility['destination_repeat_rate']:.4%}) recur. "
        "This is more meaningful than the origin side, but destination "
        "reuse (merchants especially) is expected and often *desirable* "
        "to keep visible in both train and test (a merchant's normal "
        "pattern needs to appear in training for the model to recognize "
        "deviations from it at test time) — group-splitting it away "
        "would remove exactly the signal Milestone 7's causal degree/"
        "concentration features are designed to use. Not implemented.",
        "",
        "**Random stratified split** — discussed as a diagnostic "
        "comparison only, not recommended for production evaluation: it "
        "would let the model train on rows that occur chronologically "
        "AFTER some test rows, which is unrealistic for a system that "
        "will only ever predict on the future relative to its training "
        "data, and would mix days with wildly different legitimate-"
        "volume/fraud-rate regimes (see section B) between train and "
        "test in a way no real deployment could replicate. It would "
        "likely look more optimistic than a chronological split for "
        "exactly that reason — this is the standard argument against "
        "random splits for time-ordered data, and this dataset's own "
        "day-to-day volume swings make it concrete here rather than "
        "hypothetical.",
        "",
        "## D. Split feasibility quantification",
        "",
        "See per-policy region breakdowns in section C above, and "
        "`reports/modeling_split_summary.json` for the complete "
        "per-day table and full evaluate_all_policies() output "
        "(row counts, fraud counts/rates, type distributions, step "
        "ranges, zero-fraud/fraud-only flags) underlying every figure "
        "in this document.",
        "",
        "## E. Recommended policy",
        "",
        "**`chronological_train_val_test`** (train: days 0-23, "
        "validation: days 24-26, test: days 27-30) is recommended as "
        "the primary policy:",
        "",
        "- **Temporal realism**: strictly earlier-to-later ordering "
        "across all three regions; no region uses information from a "
        "chronologically later region.",
        "- **Fraud availability**: every region has a substantial, "
        "non-zero, non-degenerate fraud count (see section C) — unlike "
        "`naive_tail_only_test`, whose single-day test region (day 30 "
        "alone) is 100% fraud and cannot evaluate false-positive "
        "behavior at all (there are no true negatives to be wrong "
        "about).",
        "- **Leakage safety**: causal features (Milestones 5 and 7) are "
        "already computed strictly from prior rows regardless of split, "
        "but the split itself adds a second, independent layer of "
        "protection — no validation/test-region transaction's outcome "
        "can influence which training rows are chosen, since training "
        "is a strict earlier-day prefix.",
        "- **Stability of evaluation**: the test region (days 27-30, "
        f"{policy_evaluations['chronological_train_val_test']['regions']['test']['total_rows']:,} "
        "rows) absorbs day 30's artifact (272 rows) into a much larger "
        "4-day window rather than evaluating on it in isolation, while "
        "still being the chronologically final, most realistic-for-"
        "deployment window.",
        "- **Reproducibility**: day-index boundaries are exact integers "
        "derived from `step`, deterministic and re-derivable from the "
        "raw file with no randomness involved.",
        "- **Compatibility with causal graph features**: Milestone 7's "
        "features only ever reference strictly prior rows in file/step "
        "order — a chronological split is the only policy that keeps "
        "\"prior\" during evaluation meaning the same thing it will mean "
        "in production (in a random split, a test row's causal features "
        "would have been computed against a training-set future that a "
        "real production system would not yet have observed at "
        "prediction time).",
        "",
        "`chronological_holdout_80_20` (train/test only, no separate "
        "validation region) is documented as an acceptable SIMPLER "
        "alternative when a validation region isn't needed (e.g. no "
        "hyperparameter search planned) — same train/test day boundaries "
        "as the recommended policy's train/test macro-split, just "
        "without carving out days 24-26 for validation.",
        "",
        "`naive_tail_only_test` is documented and evaluated specifically "
        "as the counter-example this milestone's instructions warned "
        "against: a convenient chronological cut that happens to land "
        "on an artifact-dominated (100% fraud) region. It is NOT "
        "recommended.",
        "",
        "## F. Known limitations",
        "",
        "- Day boundaries are chosen from this dataset's own observed "
        "per-day statistics (section B), not a generic rule — they "
        "would need to be re-derived, not blindly reused, for any other "
        "dataset or a future full re-ingestion.",
        "- This document establishes split FEASIBILITY and POLICY only. "
        "No model has been trained against any of these splits; nothing "
        "here should be read as a performance claim.",
        "- Group-aware splitting was assessed analytically (section C) "
        "but not implemented; if a future milestone's evaluation shows "
        "destination-account leakage is a real concern in practice, "
        "this would need revisiting with an actual implementation.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Design and validate the Milestone 9 modeling split "
        "policy (bounded aggregate analysis only; no model training)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--spec-out", type=Path, default=SPLIT_SPEC_PATH)
    parser.add_argument("--summary-out", type=Path, default=SPLIT_SUMMARY_PATH)
    args = parser.parse_args(argv)

    try:
        validate_columns(load_csv(args.csv))
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Split design failed: {exc}")
        return 1

    t0 = time.time()
    step_verification = verify_step_ordering(args.csv, chunksize=args.chunksize)
    per_day_stats = compute_per_day_stats(args.csv, chunksize=args.chunksize)
    policy_evaluations = evaluate_all_policies(per_day_stats)
    overlap_checks = {
        name: check_no_overlap_and_full_accounting(per_day_stats, name)
        for name in CANDIDATE_POLICIES
    }
    group_feasibility = assess_group_split_feasibility(args.csv, chunksize=args.chunksize)
    runtime = time.time() - t0

    write_split_spec(per_day_stats, policy_evaluations, group_feasibility, step_verification, args.spec_out)

    summary = {
        "runtime_seconds": round(runtime, 2),
        "target_column_used_for": "label audit only, never as a feature",
        "step_verification": step_verification,
        "per_day_stats": {str(k): v for k, v in sorted(per_day_stats.items())},
        "policy_evaluations": policy_evaluations,
        "overlap_checks": overlap_checks,
        "group_split_feasibility": group_feasibility,
        "baseline_feature_columns_count": len(BASELINE_FEATURE_COLUMNS),
        "identifier_columns": IDENTIFIER_COLUMNS,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Spec written to    : {args.spec_out}")
    print(f"Summary written to : {args.summary_out}")
    print(f"Non-decreasing by step: {step_verification['is_non_decreasing_by_step']}")
    print(f"Total rows          : {step_verification['total_rows']:,}")
    print(f"Runtime             : {runtime:.2f}s")
    for name, check in overlap_checks.items():
        if check["applicable"]:
            print(f"  [{name}] no_overlap={check['no_overlap']} rows_accounted={check['rows_accounted_pct']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
