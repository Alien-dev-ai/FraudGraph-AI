# FraudGraph AI — Baseline Ablation and Graph-Feature Contribution Analysis (Milestone 11)

**This is an ablation/attribution study, not a model-optimization milestone.** The same chronological split (Milestone 9, unchanged), the same preprocessing (train-only median imputation + train-only StandardScaler), the same `LogisticRegression(class_weight="balanced", random_state=42)` configuration (Milestone 10, unchanged), the same default classification threshold (0.5), and the same 1% alert budget are used for every experiment below — **only the feature column selection changes.** No hyperparameter search, no threshold tuning, and no resampling was performed anywhere in this milestone. The test split is evaluated, exactly as in Milestone 10, but is never used to select a feature group, a preprocessing parameter, or a model configuration. Coefficient magnitude is used only as a descriptive diagnostic below, never as proof of causal importance. Milestone 10 documented that its near-perfect result may reflect PaySim simulation mechanics rather than a general fraud-detection capability — this milestone investigates that suspicion empirically rather than assuming an answer.

## Feature groups

Membership is derived directly from existing, approved metadata (`src/baseline_dataset.py`'s `BASELINE_FEATURE_COLUMNS` / `GRAPH_FEATURES_RETAINED_IN_BASELINE`, and `src/features_raw_paysim.py`'s per-column `category`) — never hand-typed independently of it. The approved 39-column contract itself is never modified; these are analysis-time subsets only.

- **FULL_BASELINE** (39 columns): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`, `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`
- **TABULAR_ONLY** (34 columns): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`
- **GRAPH_ONLY** (5 columns): `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`
- **BALANCE_REDUCED** (37 columns): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`, `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`
- **BALANCE_REDUCED_TABULAR** (32 columns): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`
- **BALANCE_REDUCED_FULL** (37 columns): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`, `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`

## Column categories (for reference)

- **transaction** (9): `hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`
- **balance** (15): `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`
- **identity** (10): `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`
- **graph** (5): `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`

## Measured results (per feature group)

Differences are reported numerically below; feature groups are **not** ranked best/worst — this is an attribution study.

### `FULL_BASELINE` (39 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=794 TP=792 FP=2 FN=0 TN=79,522 precision=0.9975 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- Test: rows=81,110 fraud=1,048 alerts=1,049 TP=1,048 FP=1 FN=0 TN=80,061 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- Runtime: 73.6s

### `TABULAR_ONLY` (34 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=793 TP=792 FP=1 FN=0 TN=79,523 precision=0.9987 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- Test: rows=81,110 fraud=1,048 alerts=1,049 TP=1,048 FP=1 FN=0 TN=80,061 precision=0.9990 recall=1.0000 PR-AUC=1.0000 ROC-AUC(secondary)=1.0000
- Runtime: 64.8s

### `GRAPH_ONLY` (5 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=39,770 TP=542 FP=39,228 FN=250 TN=40,296 precision=0.0136 recall=0.6843 PR-AUC=0.0133 ROC-AUC(secondary)=0.6080
- Test: rows=81,110 fraud=1,048 alerts=40,280 TP=693 FP=39,587 FN=355 TN=40,475 precision=0.0172 recall=0.6613 PR-AUC=0.0171 ROC-AUC(secondary)=0.6009
- Runtime: 9.4s

### `BALANCE_REDUCED` (37 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=4,778 TP=781 FP=3,997 FN=11 TN=75,527 precision=0.1635 recall=0.9861 PR-AUC=0.8870 ROC-AUC(secondary)=0.9968
- Test: rows=81,110 fraud=1,048 alerts=4,908 TP=1,032 FP=3,876 FN=16 TN=76,186 precision=0.2103 recall=0.9847 PR-AUC=0.8889 ROC-AUC(secondary)=0.9965
- Runtime: 76.4s

### `BALANCE_REDUCED_TABULAR` (32 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=4,814 TP=781 FP=4,033 FN=11 TN=75,491 precision=0.1622 recall=0.9861 PR-AUC=0.8885 ROC-AUC(secondary)=0.9968
- Test: rows=81,110 fraud=1,048 alerts=4,944 TP=1,034 FP=3,910 FN=14 TN=76,152 precision=0.2091 recall=0.9866 PR-AUC=0.8902 ROC-AUC(secondary)=0.9966
- Runtime: 2522.1s

### `BALANCE_REDUCED_FULL` (37 features)

- Train: 6,201,194 rows, 6,373 fraud
- Validation: rows=80,316 fraud=792 alerts=4,778 TP=781 FP=3,997 FN=11 TN=75,527 precision=0.1635 recall=0.9861 PR-AUC=0.8870 ROC-AUC(secondary)=0.9968
- Test: rows=81,110 fraud=1,048 alerts=4,908 TP=1,032 FP=3,876 FN=16 TN=76,186 precision=0.2103 recall=0.9847 PR-AUC=0.8889 ROC-AUC(secondary)=0.9965
- Runtime: 88.6s

## Interpretation — graph-feature contribution (Section D)

**Q1. Does adding the retained graph features to the tabular baseline change performance?** `FULL_BASELINE` minus `TABULAR_ONLY`, per split/metric (positive = FULL_BASELINE scored higher):

```
{
  "validation": {
    "average_precision_pr_auc": 0.0,
    "roc_auc_secondary": 0.0,
    "precision": -0.001258,
    "recall": 0.0,
    "false_positive_count": 1.0,
    "n_alerts": 1.0
  },
  "test": {
    "average_precision_pr_auc": 0.0,
    "roc_auc_secondary": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "false_positive_count": 0.0,
    "n_alerts": 0.0
  }
}
```

**Q2. Does that contribution remain once the two dominant balance features are removed?** `BALANCE_REDUCED_FULL` minus `BALANCE_REDUCED_TABULAR`:

```
{
  "validation": {
    "average_precision_pr_auc": -0.001551,
    "roc_auc_secondary": -3e-05,
    "precision": 0.001222,
    "recall": 0.0,
    "false_positive_count": -36.0,
    "n_alerts": -36.0
  },
  "test": {
    "average_precision_pr_auc": -0.001253,
    "roc_auc_secondary": -2.9e-05,
    "precision": 0.001127,
    "recall": -0.001908,
    "false_positive_count": -34.0,
    "n_alerts": -36.0
  }
}
```

**Q3/Q4. Graph-derived coefficients, by group they appear in (sign/magnitude stability check):**

```
{
  "origin_out_degree_causal": {
    "values_by_group": {
      "FULL_BASELINE": -0.03892727568745613,
      "GRAPH_ONLY": 0.018804961815476418,
      "BALANCE_REDUCED_FULL": 0.0015580655308440328
    },
    "sign_consistent_across_groups": false
  },
  "dest_in_degree_causal": {
    "values_by_group": {
      "FULL_BASELINE": -0.3357146084308624,
      "GRAPH_ONLY": -0.46438363194465637,
      "BALANCE_REDUCED_FULL": -0.09008075296878815
    },
    "sign_consistent_across_groups": true
  },
  "origin_seen_as_dest_before_causal": {
    "values_by_group": {
      "FULL_BASELINE": -0.05954362079501152,
      "GRAPH_ONLY": -0.05716777965426445,
      "BALANCE_REDUCED_FULL": -0.15909387171268463
    },
    "sign_consistent_across_groups": true
  },
  "dest_seen_as_origin_before_causal": {
    "values_by_group": {
      "FULL_BASELINE": -0.1465592086315155,
      "GRAPH_ONLY": 0.006178300827741623,
      "BALANCE_REDUCED_FULL": -0.11939394474029541
    },
    "sign_consistent_across_groups": false
  },
  "dest_incoming_concentration_causal": {
    "values_by_group": {
      "FULL_BASELINE": 0.3089789152145386,
      "GRAPH_ONLY": -0.0977654755115509,
      "BALANCE_REDUCED_FULL": -0.023501351475715637
    },
    "sign_consistent_across_groups": false
  }
}
```

**Q5. Graph features with |coefficient| < 0.05 in every group they appear in (descriptive cutoff, not a significance test):**

```
{
  "origin_out_degree_causal": {
    "values_by_group": {
      "FULL_BASELINE": -0.03892727568745613,
      "GRAPH_ONLY": 0.018804961815476418,
      "BALANCE_REDUCED_FULL": 0.0015580655308440328
    },
    "sign_consistent_across_groups": false
  }
}
```

These deltas and coefficients are reported as observed quantities. Coefficient magnitude in a linear model reflects that feature's contribution *given the other features already in the model* under this specific preprocessing (train-only standardization) — it is a descriptive diagnostic, not proof that a feature is causally responsible for fraud.

## Interpretation — PaySim balance-artifact dependence (Section E)

Removed features: `orig_balance_mismatch`, `orig_balance_after_zero`. Target/identifier leakage re-verified for every group: **True**.

`FULL_BASELINE` vs `BALANCE_REDUCED_FULL` (with vs. without the two dominant balance features, graph features present in both):

```
{
  "validation": {
    "average_precision_pr_auc": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.8869878333199144,
      "delta": 0.113012
    },
    "roc_auc_secondary": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9967665088336206,
      "delta": 0.003233
    },
    "precision": {
      "with_balance_features": 0.9974811083123426,
      "balance_reduced": 0.16345751360401842,
      "delta": 0.834024
    },
    "recall": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9861111111111112,
      "delta": 0.013889
    },
    "false_positive_count": {
      "with_balance_features": 2,
      "balance_reduced": 3997,
      "delta": -3995.0
    },
    "n_alerts": {
      "with_balance_features": 794,
      "balance_reduced": 4778,
      "delta": -3984.0
    }
  },
  "test": {
    "average_precision_pr_auc": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.8889211349650832,
      "delta": 0.111079
    },
    "roc_auc_secondary": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9965253967774212,
      "delta": 0.003475
    },
    "precision": {
      "with_balance_features": 0.9990467111534795,
      "balance_reduced": 0.21026894865525672,
      "delta": 0.788778
    },
    "recall": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9847328244274809,
      "delta": 0.015267
    },
    "false_positive_count": {
      "with_balance_features": 1,
      "balance_reduced": 3876,
      "delta": -3875.0
    },
    "n_alerts": {
      "with_balance_features": 1049,
      "balance_reduced": 4908,
      "delta": -3859.0
    }
  }
}
```

`TABULAR_ONLY` vs `BALANCE_REDUCED_TABULAR` (same comparison, no graph features in either):

```
{
  "validation": {
    "average_precision_pr_auc": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.8885386587089319,
      "delta": 0.111461
    },
    "roc_auc_secondary": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9967963740315483,
      "delta": 0.003204
    },
    "precision": {
      "with_balance_features": 0.9987389659520807,
      "balance_reduced": 0.16223514748649773,
      "delta": 0.836504
    },
    "recall": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9861111111111112,
      "delta": 0.013889
    },
    "false_positive_count": {
      "with_balance_features": 1,
      "balance_reduced": 4033,
      "delta": -4032.0
    },
    "n_alerts": {
      "with_balance_features": 793,
      "balance_reduced": 4814,
      "delta": -4021.0
    }
  },
  "test": {
    "average_precision_pr_auc": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.8901740695476695,
      "delta": 0.109826
    },
    "roc_auc_secondary": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9965543819475021,
      "delta": 0.003446
    },
    "precision": {
      "with_balance_features": 0.9990467111534795,
      "balance_reduced": 0.20914239482200647,
      "delta": 0.789904
    },
    "recall": {
      "with_balance_features": 1.0,
      "balance_reduced": 0.9866412213740458,
      "delta": 0.013359
    },
    "false_positive_count": {
      "with_balance_features": 1,
      "balance_reduced": 3910,
      "delta": -3909.0
    },
    "n_alerts": {
      "with_balance_features": 1049,
      "balance_reduced": 4944,
      "delta": -3895.0
    }
  }
}
```

No expected outcome was assumed ahead of running these experiments; the numbers above are what was measured.

### Bounded smoke test (run before the full pipeline)

- Rows: 300,000, chunksize 200,000
- Runtime: 19.50s
- Peak RSS: 632 MB (safety limit: 8,000 MB, within limit: True)

## Limitations

- This is an attribution study over ONE fixed model family and ONE fixed preprocessing configuration (both unchanged from Milestone 10). It does not establish that these conclusions would hold for a different model family, a different preprocessing choice, or a different threshold.
- Coefficient magnitude is reported only as a descriptive diagnostic under this specific linear model and standardization; it is not a causal-importance measure and does not account for correlation/collinearity among features.
- Milestone 10's near-perfect validation/test scores may reflect PaySim's synthetic balance-draining fraud-injection pattern rather than a general capability; the comparisons above quantify how much of that score is attributable to the two named balance features and to the graph features, but do not extrapolate to real (non-synthetic) transaction data.
- Group-aware account-level splitting remains not implemented (Milestone 9), unchanged here.
- No hyperparameter tuning, threshold tuning, or resampling was performed for any feature group.
- One feature group's fit (`BALANCE_REDUCED_TABULAR`) took substantially longer to converge (~2,500s) than every other group (all under 90s) under the identical, unmodified `max_iter=1000` lbfgs configuration. This is reported as an observed runtime characteristic of that specific 32-column subset, not addressed by changing any hyperparameter (which this milestone's instructions prohibit) — a future milestone could investigate the conditioning cause if runtime matters.

