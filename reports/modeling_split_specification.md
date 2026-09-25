# FraudGraph AI — Modeling Split Specification (Milestone 9)

DESIGN AND VALIDATION ONLY. No model was trained; no scaler/imputer/encoder was fit; no threshold was tuned; no resampling was performed. `isFraud` is read ONLY for label-distribution audit in this document (never as a feature) — see `src/modeling_split.py` docstring.

## A. Evaluation objective

Fraud is a severe minority class (8,213 of 6,362,620 raw transactions, 0.129% — Milestones 4/5). Accuracy is unsuitable: a trivial always-legitimate classifier scores 99.87% accuracy while catching zero fraud. Candidate metrics for the eventual modeling milestone (none computed here):

- **Average precision / PR-AUC** — summarizes the precision-recall tradeoff across all thresholds; the right primary summary metric under severe imbalance, since it is not inflated by the large true-negative count the way ROC-AUC can be.
- **Precision at a fixed alert budget** (e.g. top-K flagged transactions per day) — the operational question a fraud team actually faces: 'if we can only review N alerts, what fraction are real fraud?'
- **Recall at a fixed alert budget** — 'of all fraud, how much do we catch within that same review budget?' — the direct complement, since budget is usually the true constraint (review staff/time), not a probability threshold.
- **Recall at a fixed precision target** — for compliance/cost framings where a maximum tolerable false-positive rate is set first (e.g. 'no more than 1 false alarm per 10 real fraud cases') and recall is then read off at that operating point.
- **Confusion matrix counts** (TP/FP/FN/TN) — the raw counts underlying every ratio above; always reported alongside rates so a rate is never presented without its denominator.
- **Fraud capture rate** — recall, framed operationally as 'percent of total fraud dollar/count volume caught'.
- **False-positive burden** — the absolute count of legitimate transactions flagged, since with ~6.35M legitimate transactions even a 0.1% false-positive rate is thousands of alerts — a rate alone hides the operational review cost.
- **ROC-AUC** — retained only as a SECONDARY metric. Under 0.13% prevalence, ROC-AUC is dominated by the huge true-negative count and can look excellent even when precision at any usable alert budget is poor; never the sole metric.

## B. Temporal structure

Verified via bounded streaming pass (not reused from an old report): non-decreasing by step = **True**, over 6,362,620 rows, steps 1-743 (743 unique step values). 6,361,877 rows (99.9883%) share their step value with the immediately preceding row — same-step file-order causality (Milestone 5's convention: an earlier file row is prior when steps tie) governs the vast majority of the dataset, not an edge case.

**Key finding (extends Milestone 4's 'simulation-boundary artifact')**: fraud COUNT is stable across all 31 days (~210-320/day: min 216, max 320), while LEGITIMATE volume swings from 272 to 574,255 rows/day independent of fraud count. This means several MID-simulation days, not only the final day, show inflated fraud rates purely from low legitimate volume:

  - day 2: 1,070 total, 310 fraud, rate 28.9720%
  - day 4: 9,789 total, 252 fraud, rate 2.5743%
  - day 17: 20,999 total, 268 fraud, rate 1.2763%
  - day 18: 11,300 total, 256 fraud, rate 2.2655%
  - day 19: 19,727 total, 236 fraud, rate 1.1963%
  - day 20: 24,593 total, 272 fraud, rate 1.1060%
  - day 25: 13,885 total, 272 fraud, rate 1.9589%
  - day 26: 8,578 total, 280 fraud, rate 3.2642%
  - day 27: 14,661 total, 248 fraud, rate 1.6916%
  - day 29: 11,287 total, 268 fraud, rate 2.3744%
  - day 30: 272 total, 272 fraud, rate 100.0000%

`step`/`day_index` are used here only to DEFINE region boundaries — neither is reintroduced as a model feature; Milestone 5's exploratory_only classification for both stands unchanged.

## C. Candidate split policies

### `chronological_holdout_80_20`
- **train** (days 0-23, steps 1-576): 6,201,194 rows, 6,373 fraud (0.1028%)
- **test** (days 24-30, steps 577-743): 161,426 rows, 1,840 fraud (1.1398%)

### `chronological_train_val_test`
- **train** (days 0-23, steps 1-576): 6,201,194 rows, 6,373 fraud (0.1028%)
- **validation** (days 24-26, steps 577-648): 80,316 rows, 792 fraud (0.9861%)
- **test** (days 27-30, steps 649-743): 81,110 rows, 1,048 fraud (1.2921%)

### `naive_tail_only_test`
- **train** (days 0-29, steps 1-720): 6,362,348 rows, 7,941 fraud (0.1248%)
- **test** (days 30-30, steps 721-743): 272 rows, 272 fraud (100.0000%) — **100% FRAUD, degenerate label distribution, unusable in isolation**

### `random_stratified_diagnostic`
Not a temporal region-based policy — see discussion below.

**Group-aware splitting** — assessed, not implemented:
- 6,353,307 unique origin accounts; only 9,298 (0.1463%) ever transact more than once (Milestones 4/6 confirmed this at full scale). Group-splitting by origin account would be nearly indistinguishable from a random row split, since almost every origin contributes exactly one row to begin with — it buys essentially no leakage protection beyond what chronological ordering already gives, for real implementation cost (ensuring no origin straddles a boundary). Not implemented.
- 2,722,362 unique destination accounts; 459,658 (16.8845%) recur. This is more meaningful than the origin side, but destination reuse (merchants especially) is expected and often *desirable* to keep visible in both train and test (a merchant's normal pattern needs to appear in training for the model to recognize deviations from it at test time) — group-splitting it away would remove exactly the signal Milestone 7's causal degree/concentration features are designed to use. Not implemented.

**Random stratified split** — discussed as a diagnostic comparison only, not recommended for production evaluation: it would let the model train on rows that occur chronologically AFTER some test rows, which is unrealistic for a system that will only ever predict on the future relative to its training data, and would mix days with wildly different legitimate-volume/fraud-rate regimes (see section B) between train and test in a way no real deployment could replicate. It would likely look more optimistic than a chronological split for exactly that reason — this is the standard argument against random splits for time-ordered data, and this dataset's own day-to-day volume swings make it concrete here rather than hypothetical.

## D. Split feasibility quantification

See per-policy region breakdowns in section C above, and `reports/modeling_split_summary.json` for the complete per-day table and full evaluate_all_policies() output (row counts, fraud counts/rates, type distributions, step ranges, zero-fraud/fraud-only flags) underlying every figure in this document.

## E. Recommended policy

**`chronological_train_val_test`** (train: days 0-23, validation: days 24-26, test: days 27-30) is recommended as the primary policy:

- **Temporal realism**: strictly earlier-to-later ordering across all three regions; no region uses information from a chronologically later region.
- **Fraud availability**: every region has a substantial, non-zero, non-degenerate fraud count (see section C) — unlike `naive_tail_only_test`, whose single-day test region (day 30 alone) is 100% fraud and cannot evaluate false-positive behavior at all (there are no true negatives to be wrong about).
- **Leakage safety**: causal features (Milestones 5 and 7) are already computed strictly from prior rows regardless of split, but the split itself adds a second, independent layer of protection — no validation/test-region transaction's outcome can influence which training rows are chosen, since training is a strict earlier-day prefix.
- **Stability of evaluation**: the test region (days 27-30, 81,110 rows) absorbs day 30's artifact (272 rows) into a much larger 4-day window rather than evaluating on it in isolation, while still being the chronologically final, most realistic-for-deployment window.
- **Reproducibility**: day-index boundaries are exact integers derived from `step`, deterministic and re-derivable from the raw file with no randomness involved.
- **Compatibility with causal graph features**: Milestone 7's features only ever reference strictly prior rows in file/step order — a chronological split is the only policy that keeps "prior" during evaluation meaning the same thing it will mean in production (in a random split, a test row's causal features would have been computed against a training-set future that a real production system would not yet have observed at prediction time).

`chronological_holdout_80_20` (train/test only, no separate validation region) is documented as an acceptable SIMPLER alternative when a validation region isn't needed (e.g. no hyperparameter search planned) — same train/test day boundaries as the recommended policy's train/test macro-split, just without carving out days 24-26 for validation.

`naive_tail_only_test` is documented and evaluated specifically as the counter-example this milestone's instructions warned against: a convenient chronological cut that happens to land on an artifact-dominated (100% fraud) region. It is NOT recommended.

## F. Known limitations

- Day boundaries are chosen from this dataset's own observed per-day statistics (section B), not a generic rule — they would need to be re-derived, not blindly reused, for any other dataset or a future full re-ingestion.
- This document establishes split FEASIBILITY and POLICY only. No model has been trained against any of these splits; nothing here should be read as a performance claim.
- Group-aware splitting was assessed analytically (section C) but not implemented; if a future milestone's evaluation shows destination-account leakage is a real concern in practice, this would need revisiting with an actual implementation.

