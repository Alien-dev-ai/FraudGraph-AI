# FraudGraph AI — Out-of-Distribution / Synthetic Stress Testing (Milestone 13)

**This is a robustness/stress-testing milestone, not model optimization, feature engineering, threshold tuning, hyperparameter tuning, or a claim of real-world performance.**

## Critical distinction: fixed-model perturbation vs. Milestone 11's retrained ablation

Milestone 11 REMOVED features and RETRAINED a new model on each reduced feature set. **Milestone 13 fits the Milestone 10 model exactly ONCE and never retrains or refits preprocessing again** -- every stress scenario below perturbs only the TEST feature VALUES and reuses that one fixed imputer/scaler/model to score them. The two experiments answer different questions: M11 asks 'what would a model look like without this feature'; M13 asks 'how sensitive is the model we already have to this feature's value.' Neither result implies the other's.

## Objective

Determine how sensitive the Milestone 10 `FULL_BASELINE` detector is to `orig_balance_mismatch` and `orig_balance_after_zero` -- the two features Milestone 11 identified as dominant -- by perturbing them in the TEST split only, under the existing fixed model.

## Perturbations (exact, mathematical, deterministic)

1. **Neutral/reference value**: each column's TRAIN-set mode (computed only from train, never from validation/test/labels). Measured this run: `{"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 1.0}`.
2. **Fixed row permutation**: `numpy.random.default_rng(42).permutation(n_test)`, computed once and reused for every scenario/level.
3. **Corruption at level L**: the first `round(L * n_test)` rows of that fixed permutation have the named column(s) set to their neutral value; all other rows are untouched. Higher levels strictly extend lower levels (nested, monotonic).
4. **Predetermined levels** (Stress Test D): `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]` -- fixed before any result was computed, never adjusted afterward.

## Fixed-model policy

One `LogisticRegression(class_weight="balanced", random_state=42)` is fit once on the unmodified 39-column TRAIN split (Milestone 10's recipe, unchanged) and reused, unmodified, for every scenario in this report. It is never retrained.

## Preprocessing policy

`SimpleImputer(strategy="median")` and `StandardScaler()` are each fit exactly once, on TRAIN only (Milestone 10's `fit_preprocessing`), and only ever `.transform()`-ed afterward -- including on every perturbed test copy. Never refit on validation/test/perturbed data.

## Split policy

Milestone 9's `chronological_train_val_test` policy, unchanged: train days 0-23, validation days 24-26, test days 27-30. Perturbation never changes which row belongs to which split.

## Threshold policy

Fixed at 0.5 for every scenario -- never tuned, never varied per scenario/day/type.

## Deterministic seed

`random_state=42` everywhere randomness appears (model fit, permutation) -- matching every prior milestone.

## Resource strategy

The expensive 39-column causal feature stream and the one model fit each happen exactly ONCE. Every stress scenario operates on a freshly copied, test-sized (tens of thousands of rows, not millions) perturbed array that is discarded before the next scenario begins -- never multiple full feature matrices or duplicate train-sized arrays held at once.

### Bounded smoke test

- Rows: 300,000, chunksize 200,000
- Runtime: 5.81s
- Peak RSS: 551 MB (safety limit: 8,000 MB, within limit: True)
- 0% level reproduced the unperturbed baseline exactly: None
- 100% level produced the expected neutral values: None

## Leakage controls

Perturbation functions read TRAIN data only for the neutral value; they never read `isFraud`, never read validation/test labels, and never choose a perturbation parameter by looking at any result. Perturbation levels were fixed before this module was run against real outcomes.

## Baseline reproduction check (must pass before any stress result)

Matches Milestone 12's recorded `FULL_BASELINE` test-period confusion matrix and row/fraud counts exactly: **True**.

- **unperturbed (recomputed here)**: alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC=1.0000
- **recorded (Milestone 12)**: alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC=1.0000

## A. What happens when the dominant balance signal is removed from the test inputs?

- **A_remove_orig_balance_mismatch**: alerts=375 TP=374 FP=1 FN=674 precision=0.9973 recall=0.3569 PR-AUC=0.5870 ROC-AUC=0.9280
- **B_remove_orig_balance_after_zero**: alerts=36,914 TP=1,048 FP=35,866 FN=0 precision=0.0284 recall=1.0000 PR-AUC=0.7925 ROC-AUC=0.9948
- **C_neutralize_both**: alerts=375 TP=374 FP=1 FN=674 precision=0.9973 recall=0.3569 PR-AUC=0.5820 ROC-AUC=0.9229

Model sensitivity to these two inputs is substantial: removing or neutralizing them (fixed model, values-only perturbation) changes predicted alerts materially. This describes measured model behavior after controlled perturbation -- it does not establish that either feature causes fraud, or that the model 'understands' fraud in any causal sense.

## B. What happens as that signal is progressively corrupted?

- **D level=0.00**: alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC=1.0000
- **D level=0.10**: alerts=971 TP=970 FP=1 FN=78 precision=0.9990 recall=0.9256 PR-AUC=0.9503 ROC-AUC=0.9900
- **D level=0.25**: alerts=881 TP=880 FP=1 FN=168 precision=0.9989 recall=0.8397 PR-AUC=0.8974 ROC-AUC=0.9796
- **D level=0.50**: alerts=710 TP=709 FP=1 FN=339 precision=0.9986 recall=0.6765 PR-AUC=0.7958 ROC-AUC=0.9609
- **D level=0.75**: alerts=542 TP=541 FP=1 FN=507 precision=0.9982 recall=0.5162 PR-AUC=0.6962 ROC-AUC=0.9428
- **D level=1.00**: alerts=375 TP=374 FP=1 FN=674 precision=0.9973 recall=0.3569 PR-AUC=0.5820 ROC-AUC=0.9229

## C. Does the effect remain consistent across days?

Full per-day detail (every scenario x every test-period day, including day 30) is in `reports/stress_test_daily.csv`. Day 30 is reported explicitly, never excluded: it carries Milestone 4/9's documented simulation-boundary artifact (100% fraud among its 272 remaining rows, zero legitimate rows), so precision/PR-AUC/ROC-AUC are undefined or degenerate on that slice in isolation regardless of perturbation -- reported as `null` with a documented reason, exactly as Milestone 12 established.

## D. Does the effect differ between CASH_OUT and TRANSFER?

Full per-type detail (every scenario x CASH_OUT/TRANSFER) is in `reports/stress_test_by_type.csv`. No independent tuning was performed per type -- the same fixed threshold and perturbation levels apply to both.

## E. Fixed-model perturbation vs. Milestone 11 retrained ablation

Milestone 11's `BALANCE_REDUCED_FULL` retrained a model with 37 columns (the two dominant features never existed as inputs during training). Milestone 13's Stress Test C keeps the SAME 39-column-trained model and only overwrites those two columns' TEST values. Both are reported side by side in `reports/stress_test_results.json` for direct comparison; a numeric difference between them reflects the difference between 'never having learned from a feature' and 'having learned from it but being fed an uninformative value at inference time' -- not a ranking of which approach is better.

## Interpretation

Three separate claims, deliberately not conflated (matching Milestone 12's convention):

1. **Measured behavior on PaySim**: the numbers above are exact, measured figures from controlled perturbation of PaySim's synthetic transaction log, under the fixed Milestone 10 model.
2. **Robustness under controlled synthetic perturbation**: describes how sensitive the already-trained detector is to two specific input values within this simulation -- language used throughout this report is limited to "model sensitivity to the feature" and "performance changed after controlled perturbation," consistent with dependence on PaySim's synthetic balance pattern.
3. **What remains unknown for real-world fraud detection**: whether this sensitivity pattern would appear, matter, or behave similarly on real (non-synthetic) transaction data is NOT established here. No claim of causal importance, real-world generalization, or production readiness is made.

## Limitations

- Only one model family, one preprocessing configuration, and one fixed threshold were ever evaluated (Milestone 10's, unchanged) -- this does not establish sensitivity under a different model or preprocessing choice.
- The fixed row permutation determines WHICH rows are corrupted first at low levels; a different seed would corrupt a different subset at the same level (though the aggregate corruption fraction would be identical) -- results at intermediate levels (10-75%) reflect one particular deterministic draw, not an average over draws.
- Day-level and type-level slices can be small; metrics on small slices are reported exactly as computed but are correspondingly noisier than the whole-test-period aggregate.
- This milestone does not claim, and explicitly disclaims, real-world generalization or production readiness.
- No hyperparameter tuning, threshold tuning, or resampling was performed anywhere in this milestone.

