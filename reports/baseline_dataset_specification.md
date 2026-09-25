# FraudGraph AI — Baseline Dataset Specification (Milestone 8)

Generated in part from `src/baseline_dataset.py` constants — the module docstring there is the full narrative source of truth; this file summarizes the resulting contract.

## Baseline feature contract (39 columns, fixed order)

`hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`, `origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`

## Identifier / target columns (never model inputs)

- Identifiers (audit/traceability only): `row_id`, `nameOrig`, `nameDest`
- Target column (never read as a feature anywhere in this project): `isFraud`

## Graph features retained in baseline (from Milestone 7)

`origin_out_degree_causal`, `dest_in_degree_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `dest_incoming_concentration_causal`

## Graph features retained for audit/ablation only (excluded from baseline)

`pair_seen_before_causal`, `pair_prior_count_causal`, `origin_outgoing_concentration_causal`

See the module docstring in `src/baseline_dataset.py` for the full measured rationale (constant-zero / near-degenerate / 99.95%-missing findings from Milestones 6-7, and the explicit scope caveat that these were measured on the 2,000,000-row subset, re-checked at full scale by `run_full_stream_audit`).

## Excluded per existing Milestone 5 policy (not reintroduced)

`step`, `day_index`

## Same-step / causal convention

Identical to Milestones 5 and 7: file/row order is the event order; within a shared `step`, an earlier file row is prior. This module reuses Milestone 5's and 7's per-chunk causal state functions verbatim — no new causal logic is introduced.

## Undefined-value policy

Preserve NaN; never silently impute. Imputation is deferred to the modeling pipeline (e.g. fit on a train split), since choosing an imputation value is a modeling decision, not a data-assembly decision. Existing paired indicator columns (`orig_balance_before_zero`, `origin_is_first_seen`, `dest_is_first_seen`, etc.) already explain *why* a ratio/average is undefined wherever such an indicator exists. No baseline feature is expected to produce +/-inf; `run_full_stream_audit` verifies this rather than assuming it.

## Bounded assembly design

- Development sample: 20,000 rows (`build_bounded_sample`)
- Bounded validation sample: 250,000 rows
- Full-stream audit: `run_full_stream_audit(max_rows=None)` streams the entire 6,362,620-row file computing only running aggregate statistics per column — no chunk's rows are retained after folding its stats in, so no full-size table is ever held.

## Known limitations

- The graph-feature constancy/near-degeneracy findings that justify GRAPH_FEATURES_AUDIT_ONLY were originally measured on Milestone 6/7's 2,000,000-row subset. This milestone's `run_full_stream_audit(max_rows=None)` re-ran the check on the COMPLETE 6,362,620-row file and CONFIRMED it: `pair_seen_before_causal` / `pair_prior_count_causal` remain exactly constant-zero (mean 0.0, min 0.0, max 0.0) across all 6,362,620 rows; `origin_outgoing_concentration_causal` remains 99.85% missing with mean 0.999 when defined. The 5 baseline-retained graph features were also confirmed non-constant at full scale, with `dest_incoming_concentration_causal` mean 0.290 (57.2% defined) — consistent with, and strengthening, the 2,000,000-row subset finding. See reports/baseline_full_stream_audit.json for exact figures.
- Predictive usefulness is not evaluated here (no model is trained in this milestone) — retained-in-baseline means 'technically valid, non-degenerate, and leakage-safe', not 'confirmed predictive'.

