# FraudGraph AI — Baseline Model Specification (Milestone 10)

BASELINE ONLY. No hyperparameter optimization, no decision-threshold tuning, no oversampling/undersampling/SMOTE, no competing model families, and the test split was never used for any model, preprocessing, threshold, or budget decision. This document establishes a trustworthy first baseline, not a performance claim.

## 1. Dataset / split design

Split policy: `chronological_train_val_test` (Milestone 9, unchanged):

- **train**: days 0-23
- **validation**: days 24-26
- **test**: days 27-30

Rows are assigned to a split by `day_index` only after the single continuous causal-feature pass over the whole file has computed their features — the running causal state is never reset at a split boundary (see module docstring, "Causal integrity across the split").

## 2. Feature contract

39 columns, unchanged from Milestone 8's `BASELINE_FEATURE_COLUMNS`:

`hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`, `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`

`isFraud` (target), and `row_id`/`nameOrig`/`nameDest` (identifiers), are never part of this list and are never read into the feature matrix `X` — see `src/baseline_model.py`'s load-time assertions.

## 3. Preprocessing

- **Infinite-value handling**: +/-inf is replaced with NaN before imputation (deterministic, non-learned, applied identically to every split -- no leakage). Milestone 8's full-scale audit found 2 of 6,362,620 rows with +inf in `amount_vs_dest_avg_ratio`; see `reports/baseline_full_stream_audit.json`.
- **Imputation**: `sklearn.impute.SimpleImputer(strategy="median")`, fit on the TRAIN split only. Per-column medians are computed from train rows only and applied unchanged to validation/test.
- **Scaling**: `sklearn.preprocessing.StandardScaler()`, fit on the TRAIN split only (after imputation), applied unchanged to validation/test.
- No new categorical encoding was introduced: the categorical `type` column is already represented in the baseline contract via Milestone 5's fixed one-hot `type_*` columns.
- NaN is never silently dropped or zero-filled before this stage — Milestone 8's undefined-value policy preserves it through feature assembly; the imputation VALUE is this milestone's modeling decision, made from train data only.

## 4. Model

`sklearn.linear_model.LogisticRegression`, parameters (complete, fixed — no search performed):

```
{
  "l1_ratio": 0.0,
  "C": 1.0,
  "class_weight": "balanced",
  "max_iter": 1000,
  "solver": "lbfgs",
  "random_state": 42
}
```

- `class_weight="balanced"` handles the ~0.13% fraud prevalence via reweighting, not resampling.
- `random_state=42` fixed for determinism.
- Explainability: signed per-feature coefficients (top 15 by magnitude below).

| Feature | Coefficient |
|---|---|
| `orig_balance_mismatch` | -9.4483 |
| `orig_balance_after_zero` | +9.0124 |
| `orig_balance_before_zero` | +2.3742 |
| `type_CASH_OUT` | +1.9740 |
| `dest_balance_before_zero` | +1.7297 |
| `type_PAYMENT` | -1.5738 |
| `is_dest_merchant` | -1.5738 |
| `type_CASH_IN` | -1.2593 |
| `type_TRANSFER` | +1.1138 |
| `isFlaggedFraud` | +0.8984 |
| `dest_total_amount_causal` | +0.8526 |
| `orig_balance_delta` | -0.7573 |
| `dest_avg_amount_causal` | -0.7272 |
| `log_amount` | +0.7157 |
| `type_DEBIT` | -0.4354 |

## 5. Evaluation methodology

- Default classification threshold: **0.5** — fixed in module source, never chosen from validation or test outcomes.
- Fixed alert budget: **1.00%** of a split's rows ("review the top-scored 1%") — the rate is a constant decided ahead of time; only the resulting count (a function of that split's own row count) varies by split.
- Validation and test are evaluated separately and never pooled into one reported figure.
- Primary metric: Average Precision / PR-AUC. ROC-AUC is reported only as a secondary diagnostic — Milestone 9 section A warns it is dominated by the large true-negative count under 0.13% prevalence and can look strong even when precision at any usable alert budget is weak.

## 6. Results

### Validation

- Rows: 80,316 (792 fraud, 79,524 legitimate, prevalence 0.9861%)
- Threshold: 0.5 (fixed default, not tuned on this split)
- Alerts raised: 794
- Confusion matrix: TP=792 FP=2 FN=0 TN=79,522
- Precision: 0.9975  |  Recall (fraud capture rate): 1.0000
- False positives: 2 (0.0025% of legitimate rows)
- Average Precision / PR-AUC (primary): 1.0000
- ROC-AUC (secondary diagnostic only): 1.0000
- At a fixed 1.00% alert budget (803 alerts): precision=0.9863, recall=1.0000

### Test

- Rows: 81,110 (1,048 fraud, 80,062 legitimate, prevalence 1.2921%)
- Threshold: 0.5 (fixed default, not tuned on this split)
- Alerts raised: 1,049
- Confusion matrix: TP=1,048 FP=1 FN=0 TN=80,061
- Precision: 0.9990  |  Recall (fraud capture rate): 1.0000
- False positives: 1 (0.0012% of legitimate rows)
- Average Precision / PR-AUC (primary): 1.0000
- ROC-AUC (secondary diagnostic only): 1.0000
- At a fixed 1.00% alert budget (811 alerts): precision=1.0000, recall=0.7739

Training rows: 6,201,194 (6,373 fraud). Full pipeline runtime: 143.5s. Peak RSS during the full run: 7004 MB.

### Bounded smoke test (run before the full 6,362,620-row pipeline)

- Rows: 300,000, chunksize 200,000
- Runtime: 5.77s
- Peak RSS: 554 MB (safety limit: 8,000 MB, within limit: True)

## 7. Limitations

- **Near-perfect validation/test PR-AUC (1.0000) is almost certainly a PaySim simulator artifact, not evidence of a production-ready detector.** The two largest-magnitude coefficients are `orig_balance_mismatch` and `orig_balance_after_zero` — PaySim's own fraud-injection logic is documented (Milestone 3) to characteristically drain the origin account's balance to zero, which these two features capture almost exactly. A real fraud population would not be expected to be this cleanly separable on balance bookkeeping alone. This score should be read as 'the baseline correctly learned PaySim's synthetic tell', not as a general fraud-detection capability claim — exactly the caution Milestone 9 section G asks for ('avoid claims that a high metric automatically means the detector is operationally useful').
- This is a first baseline: one model family, no hyperparameter search, an untuned default 0.5 threshold, and an untuned 1% alert budget. None of the reported numbers should be read as 'this model is production-ready.'
- ROC-AUC alone is not interpretable under this prevalence (Milestone 9 section A/G); absolute alert and false-positive counts are reported alongside every rate for exactly this reason — a high rate metric does not by itself mean the detector is operationally useful at any real alert budget.
- Group-aware (account-level) splitting was assessed and not implemented at the design stage (Milestone 9 section C) and remains not implemented here.
- No model comparison, ablation, or graph-vs-tabular study was performed — explicitly out of scope for this milestone.

