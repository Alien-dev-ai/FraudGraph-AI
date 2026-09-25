# FraudGraph AI — Robustness and Temporal Generalization Analysis (Milestone 12)

**This is a robustness/diagnostic milestone, not model optimization.** The Milestone 9 chronological split (train days 0-23, validation days 24-26, test days 27-30) and Milestone 10's exact configuration (`LogisticRegression(class_weight="balanced", random_state=42)`, train-only median imputation, train-only StandardScaler, threshold=0.5) are reused unchanged. No hyperparameter was tuned, no threshold was tuned per day/type/group, the split was not changed, and test results were not used to modify anything -- they are only ever evaluated here.

## Objective

Determine how robust the Milestone 10 signal is across (A) individual chronological test-period days, (B) the presence/absence of Milestone 11's two dominant balance features, (C) fraud-eligible transaction types, and (D) day x type cells -- using Milestone 10's `FULL_BASELINE` (39 columns) and Milestone 11's `BALANCE_REDUCED_FULL` (37 columns, unchanged membership) feature groups.

## Feature configurations (reused from Milestone 11, unchanged)

- **FULL_BASELINE** (39 columns): all Milestone 8 baseline columns.
- **BALANCE_REDUCED_FULL** (37 columns): FULL_BASELINE minus `orig_balance_mismatch`, `orig_balance_after_zero`.

## Split definitions (Milestone 9, unchanged)

- train: days 0-23
- validation: days 24-26
- test: days 27-30 (the period every experiment below examines)

## Causal-ordering policy (unchanged)

File/row order is event order; within a shared `step`, an earlier file row is prior. Causal state (Milestones 5/7) is accumulated in ONE continuous pass over the entire raw file and is never reset at a train/validation/test boundary -- identical to Milestones 10/11.

## Metric definitions and undefined-value policy

Whole-test-period summaries reuse `src.baseline_model.evaluate_split` verbatim (Milestone 10's own definitions, `zero_division=0`). Per-day / per-type / per-(day,type) SLICES use a separate `slice_metrics` function that reports `null` (never a silently substituted 0.0) for precision when a slice has zero predicted positives, for recall when a slice has zero actual fraud, and for PR-AUC/ROC-AUC when a slice contains only one class -- each with a documented reason string (`undefined_metrics_reasons` in the JSON output).

## Threshold policy

Fixed at 0.5 for every experiment, every day, every type, and every feature group -- never tuned, never varied per slice.

## Resource strategy

One streaming pass reuses Milestone 10's full 39-column causal feature assembly (`stream_split_arrays`, unchanged) to get train/validation/test arrays for BOTH feature groups via Milestone 11's column-selection (`select_group_arrays`, unchanged) -- the expensive causal computation runs once, not once per group. A second, cheap pass reads only `step`/`type` (no feature engineering) to recover per-test-row day_index/type for slicing. See the bounded smoke test below, run before the full 6,362,620-row analysis.

### Bounded smoke test

- Rows: 300,000, chunksize 200,000
- Runtime: 9.32s
- Peak RSS: 629 MB (safety limit: 8,000 MB, within limit: True)
- Test-metadata/feature-stream row-count alignment check: True

## Fraud-type verification

Fraud confined to `['CASH_OUT', 'TRANSFER']`: **True** (re-verified empirically from the raw file's per-type fraud counts: {'CASH_OUT': 4116, 'TRANSFER': 4097}).

## Experiment A — daily temporal robustness (FULL_BASELINE, test period)

- day 27: n=14,661 fraud=248 legit=14,413 fraud_rate=1.6916% alerts=248 TP=248 FP=0 FN=0 TN=14413 precision=1.0000 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- day 28: n=54,890 fraud=260 legit=54,630 fraud_rate=0.4737% alerts=261 TP=260 FP=1 FN=0 TN=54629 precision=0.9962 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- day 29: n=11,287 fraud=268 legit=11,019 fraud_rate=2.3744% alerts=268 TP=268 FP=0 FN=0 TN=11019 precision=1.0000 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- day 30 **[SIMULATION-BOUNDARY DAY -- 100% fraud among remaining rows]**: n=272 fraud=272 legit=0 fraud_rate=100.0000% alerts=272 TP=272 FP=0 FN=0 TN=0 precision=1.0000 recall=1.0000 PR-AUC=null ROC-AUC(secondary)=null

Day 30 is reported explicitly above, not silently excluded. It carries Milestone 4/9's documented simulation-boundary artifact: legitimate transaction volume in the underlying simulation collapses on the final day while fraud count stays roughly flat, so the day-30 SLICE has zero legitimate rows left. This makes aggregate test-period metrics potentially misleading in one direction (a model that is simply 'confident near the end of the file' could look artificially strong on this slice) and precision/PR-AUC/ROC-AUC mathematically undefined or degenerate on this slice in isolation (see the null entries and `undefined_metrics_reasons` in the JSON), which is exactly why it must be reported, not hidden by aggregation.

## Experiment B — balance dependency by day

Full per-day metrics for both groups are in `reports/robustness_daily.csv` and the JSON's `experiment_b_daily_by_group`. Neither group is described as better/worse; differences are reported numerically in Experiment E below.

## Experiment C — fraud-type robustness (FULL_BASELINE, whole test period)

- **CASH_OUT**: n=26,377 fraud=524 fraud_rate=1.9866% captured(TP)=524 missed(FN)=0 FP=1 recall=1.0000 precision=0.9981
- **TRANSFER**: n=7,993 fraud=524 fraud_rate=6.5557% captured(TP)=524 missed(FN)=0 FP=0 recall=1.0000 precision=1.0000

## Experiment D — day x type breakdown

Full detail in `reports/robustness_day_type.csv` (17 rows: every day x type combination actually present in the test period, for FULL_BASELINE).

## Experiment E — balance-reduced temporal robustness

Whole-test-period summary, both groups (identical split/threshold/preprocessing, only the feature columns differ):

- **FULL_BASELINE**: alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- **BALANCE_REDUCED_FULL**: alerts=4,908 TP=1,032 FP=3,876 FN=16 precision=0.2103 recall=0.9847 PR-AUC=0.8889 ROC-AUC(secondary)=0.9965

This quantifies how much of the whole-test-period result depends on the two removed balance features; it does not identify a causal mechanism and is not a claim that either configuration is a replacement model.

## Interpretation

Three separate claims, deliberately not conflated:

1. **Performance on this synthetic PaySim dataset**: the numbers above are exact, measured figures on PaySim's synthetic transaction log, under the fixed Milestone 10 recipe.
2. **Robustness within the PaySim simulation**: performance is consistent across CASH_OUT/TRANSFER and across most non-boundary test days, but that consistency is measured entirely inside one synthetic simulation and its own internal regularities (see PaySim characteristics below) -- it is not, by itself, evidence about a different dataset or deployment.
3. **Evidence for real-world fraud-detection capability**: NOT established by this milestone. PaySim's fraud-injection logic produces a synthetic balance-draining pattern (`orig_balance_mismatch`, `orig_balance_after_zero`) that Milestone 11 already showed dominates the Milestone 10 result; this milestone's per-day/per-type breakdowns describe how that same dependency behaves temporally, not whether it would hold on real transaction data.

Relevant background, restated rather than re-derived: PaySim's synthetic balance mechanics (Milestone 3) frequently produce balance-bookkeeping mismatches even for legitimate rows; the day-30 simulation-boundary artifact (Milestone 4/9) collapses legitimate volume near the end of the simulated period; `orig_balance_mismatch`/`orig_balance_after_zero` dominate the Milestone 10 linear model's coefficients (Milestone 10/11); and Milestone 11 found the standalone graph-derived signal weak (`GRAPH_ONLY` PR-AUC ~0.013-0.017) with 3 of 5 graph coefficients unstable in sign across configurations.

## Limitations

- Every number in this report comes from ONE fixed model family and ONE fixed preprocessing/threshold configuration, unchanged from Milestone 10. It does not establish robustness under a different model, preprocessing, or threshold.
- Day-level and type-level slices can be small (e.g. day 30 has 272 rows); metrics on small slices are reported exactly as computed but are correspondingly noisier than the whole-test-period aggregate.
- This milestone does not claim, and explicitly disclaims, that any result here demonstrates production readiness or real-world generalization -- see "Interpretation" above.
- Group-aware account-level splitting remains not implemented (Milestone 9), unchanged here.
- No hyperparameter tuning, threshold tuning, or resampling was performed anywhere in this milestone.

