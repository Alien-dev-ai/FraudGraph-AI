# FraudGraph AI — Synthetic Fraud-Pattern Generalization (Milestone 14)

**This is a controlled synthetic generalization experiment, not production validation, not real-world generalization proof, and not model optimization.** Every synthetic example is built from a genuinely LEGITIMATE test-period transaction; its ORIGINAL PaySim label (`isFraud`) is always 0 and is never changed. A separate `synthetic_stress_label` (1 = "constructed to look fraud-like", 0 = "constructed as a legitimate-looking distractor") is the only label used to evaluate synthetic scenarios below -- it is never confused with, or mixed into, `isFraud`.

## Objective

Does the fixed Milestone 10 `FULL_BASELINE` detector react to fraud-like BEHAVIORAL patterns that do not carry PaySim's dominant balance signature, or is it primarily recognizing that exact synthetic signature? This is evidence about SENSITIVITY to the simulated fraud mechanism, not a real-world generalization test.

## Model policy: Mode 1 only

Mode 1 (fixed model, synthetic feature VALUES only, no retraining) is implemented, as mandated. Mode 2 (optional retrained variant) is deliberately NOT implemented -- this milestone's own constraints on Mode 2 (identical train set, identical model configuration, synthetic examples never added to training) mean a Mode 2 model would be bit-for-bit identical to Mode 1's, so it would add a second model-fit's resource cost for a result already known. See module docstring for the full reasoning.

## Transformation selection (fixed before any result was examined)

Recipes, each internally self-consistent and built only from existing 39-column features:

- **fresh_account**: `origin_is_first_seen=1`, `origin_txn_count_causal=0`, `origin_total_amount_causal=0.0`, `origin_avg_amount_causal=NaN`, `amount_vs_origin_avg_ratio=NaN` -- the exact state Milestone 5 already defines for a genuine first-time-seen origin account; encodes a non-balance, non-PaySim-specific fraud heuristic (freshly created / mule account).
- **neutral_mismatch** / **neutral_after_zero**: overwrite `orig_balance_mismatch` / `orig_balance_after_zero` to their train-only reference value (Milestone 13's `neutral_value_for`, measured this run: `{"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 1.0}`).
- **altered_balance_drain**: `orig_balance_after` set to `5%` of the preserved `orig_balance_before`; `orig_balance_delta`/`orig_balance_after_zero`/`orig_balance_mismatch` recomputed with Milestone 5's own formula so the row is always a state that formula could itself produce.

Rejected candidates and why (amount/type columns; destination-side balance columns; flipping a zero-flag without its underlying value; the 5 graph-derived features) are documented in full in the module docstring's "Transformation selection" section.

## Scenarios

- **A_behavioral_without_balance_signature** (synthetic_stress_label=1, recipes=['fresh_account', 'neutral_mismatch', 'neutral_after_zero']): Legitimate CASH_OUT/TRANSFER transaction made to look fraud-like purely via the fresh-account behavioral signature, with BOTH dominant balance features forced to their neutral/reference value -- isolates whether the model reacts to a non-balance behavioral anomaly alone.
- **B_behavioral_with_normal_after_zero** (synthetic_stress_label=1, recipes=['fresh_account', 'neutral_after_zero']): Same fresh-account behavioral signature, but only `orig_balance_after_zero` is forced neutral; `orig_balance_mismatch` is left exactly as naturally observed on the source row (not forced either way).
- **C_altered_balance_pattern** (synthetic_stress_label=1, recipes=['altered_balance_drain']): No behavioral change. `orig_balance_after` is overwritten to a small nonzero residual fraction of the preserved `orig_balance_before` -- a different balance-draining pattern than PaySim's own typical exact-zero signature -- with all dependent columns recomputed consistently.
- **D_non_fraud_distractor** (synthetic_stress_label=0, recipes=['fresh_account']): Fresh-account behavioral signature applied to an otherwise perfectly ordinary legitimate transaction; both dominant balance features are left exactly as naturally observed (i.e. legitimate). Tests whether the model overreacts to the behavioral pattern alone, in the ABSENCE of the dominant PaySim balance signature.

## Source-row selection

Genuinely legitimate (`isFraud == 0`) CASH_OUT/TRANSFER test rows, days [27, 28, 29] only (day 30 excluded -- see "Day-30 degeneracy" below). First 500 such rows IN FILE ORDER per (day, type) cell -- no random sampling used. Actual cell counts this run: `{"day27_CASH_OUT": 500, "day27_TRANSFER": 500, "day28_CASH_OUT": 500, "day28_TRANSFER": 500, "day29_CASH_OUT": 500, "day29_TRANSFER": 500}` (total 3,000 source rows, shared identically across every scenario).

## Day-30 degeneracy

Day 30 has zero legitimate CASH_OUT/TRANSFER test rows (100% fraud -- Milestone 4/9 simulation-boundary artifact), so no synthetic example can be sourced from it. Every scenario's 'daily' breakdown below still contains an explicit day-30 entry (n_total=0, every metric null with a documented reason) via src.robustness_analysis.daily_breakdown's unconditional iteration over all four test-period days -- this is a mechanical consequence of source-row availability, not an omission.

## Resource strategy

The expensive 39-column causal feature stream and the one model fit each happen exactly once. Every scenario transforms the SAME small (thousands of rows, not millions), shared source-row copy and is discarded before the next scenario.

### Bounded smoke test

- Rows: 300,000, chunksize 200,000
- Runtime: 6.11s
- Peak RSS: 546 MB (safety limit: 8,000 MB, within limit: True)
- Original source values unchanged after transformation: None

## Baseline reproduction check (must pass before synthetic evaluation)

Matches the recorded `FULL_BASELINE` test-period confusion matrix/counts exactly: **True**.

- **unperturbed original test period**: n=81,110 alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC=1.0000

## Original vs. synthetic results

Original (real `isFraud`, untouched test rows) and synthetic (constructed `synthetic_stress_label`) results are reported separately below and never pooled into one metric.

- **ORIGINAL test period (real isFraud)**: n=81,110 alerts=1,049 TP=1,048 FP=1 FN=0 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC=1.0000

- **SYNTHETIC A_behavioral_without_balance_signature (n=3,000, label=1)**: n=3,000 alerts=0 TP=0 FP=0 FN=3,000 precision=null recall=0.0000 PR-AUC=null ROC-AUC=null
- **SYNTHETIC B_behavioral_with_normal_after_zero (n=3,000, label=1)**: n=3,000 alerts=221 TP=221 FP=0 FN=2,779 precision=1.0000 recall=0.0737 PR-AUC=null ROC-AUC=null
- **SYNTHETIC C_altered_balance_pattern (n=3,000, label=1)**: n=3,000 alerts=0 TP=0 FP=0 FN=3,000 precision=null recall=0.0000 PR-AUC=null ROC-AUC=null
- **SYNTHETIC D_non_fraud_distractor (n=3,000, label=0)**: n=3,000 alerts=0 TP=0 FP=0 FN=0 precision=null recall=null PR-AUC=null ROC-AUC=null

For scenarios A/B/C (`synthetic_stress_label=1`), `recall` is the fraction of constructed fraud-like examples the fixed model actually flagged (its "synthetic fraud capture rate"); `precision` is trivially 1.0 whenever any alert fires, because every row in that scenario shares the same intended label (a single-class evaluation set, exactly like Milestone 12/13's day-30 slices) -- PR-AUC/ROC-AUC are therefore `null` for the same documented reason. For scenario D (`synthetic_stress_label=0`), `precision` is the meaningful number: it is 0.0 whenever the model fires ANY alert on a purely legitimate-with-a-behavioral-quirk row (a genuine "synthetic false positive"), and `recall`/PR-AUC/ROC-AUC are `null` (no positive class exists in that scenario by construction).

## Daily and transaction-type detail

Full breakdown (every scenario x every test-period day, including the explicit day-30 zero-row entry) is in `reports/synthetic_generalization_daily.csv`; every scenario x CASH_OUT/TRANSFER is in `reports/synthetic_generalization_by_type.csv`. No independent tuning was performed per day or per type.

## Comparison

1. **Original PaySim fraud**: see the ORIGINAL row above -- the model's real, measured performance on genuinely labeled test-period fraud.
2. **Synthetic fraud with altered balance signature** (scenarios A/B/C): see each scenario's capture rate above -- how often the fixed model still fires when the familiar PaySim balance tell is absent or altered.
3. **Synthetic legitimate distractors** (scenario D): see its precision above -- how often the fixed model fires on an ordinary legitimate transaction carrying only a non-balance behavioral quirk.

These three are reported side by side, never ranked best/worst, to show whether the detector's behavior changes materially when the familiar PaySim signature is absent -- not to declare a winner.

## Interpretation

Precise language only: "performance under controlled synthetic perturbation," "behavior when the PaySim balance signature is altered," "evidence about sensitivity to the simulated fraud mechanism." This report does NOT claim the model generalizes to real fraud, detects novel fraud, or is production-ready -- real-world generalization remains untested. Synthetic examples are not real fraud; they are counterfactual evaluation constructs whose usefulness depends entirely on the transformation assumptions above, and PaySim itself is a synthetic simulation, not real transaction data.

## Limitations

- Every synthetic example is a deterministic transformation of a genuinely legitimate row; none is drawn from, or validated against, any real fraud case outside PaySim.
- The specific recipes chosen (fresh-account behavioral signature; a 5% residual balance drain) are one reasonable, documented choice among many possible non-balance/altered-balance transformations -- results are specific to these choices, not a general statement about all possible behavioral fraud patterns.
- `SAMPLE_PER_CELL=500` bounds the day-29/TRANSFER cell (smallest available, 880 legitimate rows) comfortably, but results are still based on a few thousand constructed rows per scenario, not the full test period.
- Day 30 has zero eligible source rows by construction; no synthetic evidence exists for that day (see "Day-30 degeneracy").
- No hyperparameter tuning, threshold tuning, feature engineering, or model-family comparison was performed anywhere in this milestone.

