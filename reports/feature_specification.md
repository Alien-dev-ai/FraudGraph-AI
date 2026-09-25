# FraudGraph AI — Raw PaySim Feature Specification (Milestone 5)

Generated from `src/features_raw_paysim.py::FEATURE_METADATA` (single source of truth — do not hand-edit this file).

`isFraud` is never read by the feature-engineering module (see `INPUT_COLUMNS`), so no feature here can be derived from the fraud label — confirmed absent from both `INPUT_COLUMNS` and `FEATURE_METADATA`.

## Chronological ordering assumption (Milestone 5/6 audit)

- The raw CSV was directly verified to be non-decreasing by `step` (`step.is_monotonic_increasing == True`).
- **PaySim provides no timestamp finer than `step` (1 simulated hour).** 6,361,877 of the file's 6,362,620 rows (99.99%) share their `step` value with at least one other row, so intra-step ordering is the norm, not an edge case.
- Causal identity features (`*_causal`, `*_is_first_seen`, `amount_vs_*_avg_ratio`) treat **file/row order as the causality order**: a row is only ever influenced by rows that appear strictly earlier in the file. When two rows share the same `step`, the one appearing first in the file is treated as prior — a documented, deterministic convention, not a claim about true sub-hour timing (PaySim does not resolve this itself).
- This means a same-step row that appears LATER in the file can never influence an EARLIER row's causal features, even though both rows carry the identical `step` value. Verified directly by `tests/test_features_raw_paysim.py::TestSameStepCausalOrdering`.

Leakage status legend:
- **safe_for_baseline** — recommended for the first predictive model
- **exploratory_only** — documented limitation, excluded from the baseline
- **identifier** — not a model feature; kept for traceability/joins

## Transaction features

### `step` — *exploratory_only*
Raw PaySim simulation hour (1-743), passed through unchanged. Encodes absolute position in the simulation, which correlates with the Milestone 4 simulation-boundary artifact (legitimate-volume collapse in the final day). Not recommended for the first baseline.

### `day_index` — *exploratory_only*
(step - 1) // 24, 0-indexed simulated day. This is exactly the bucketing that showed the 100% fraud-rate artifact on the final day in Milestone 4 (driven by legitimate-volume collapse, not a real signal). Not recommended for the first baseline without correcting for that artifact.

### `hour_of_day` — *safe_for_baseline*
(step - 1) % 24. Cyclical hour-of-day; does not encode absolute position in the simulation, so it does not inherit the day_index/step boundary-artifact risk.

### `type` — *identifier*
Raw transaction type string, kept for readability and joins. Most estimators need the one-hot type_* columns instead.

### `type_CASH_IN` — *safe_for_baseline*
One-hot indicator: 1 if type == 'CASH_IN' else 0. All five type_* columns are always present in every chunk (fixed category list), so schema is identical regardless of which types appear in a given chunk.

### `type_CASH_OUT` — *safe_for_baseline*
One-hot indicator: 1 if type == 'CASH_OUT' else 0. All five type_* columns are always present in every chunk (fixed category list), so schema is identical regardless of which types appear in a given chunk.

### `type_DEBIT` — *safe_for_baseline*
One-hot indicator: 1 if type == 'DEBIT' else 0. All five type_* columns are always present in every chunk (fixed category list), so schema is identical regardless of which types appear in a given chunk.

### `type_PAYMENT` — *safe_for_baseline*
One-hot indicator: 1 if type == 'PAYMENT' else 0. All five type_* columns are always present in every chunk (fixed category list), so schema is identical regardless of which types appear in a given chunk.

### `type_TRANSFER` — *safe_for_baseline*
One-hot indicator: 1 if type == 'TRANSFER' else 0. All five type_* columns are always present in every chunk (fixed category list), so schema is identical regardless of which types appear in a given chunk.

### `amount` — *safe_for_baseline*
Raw transaction amount, passed through unchanged.

### `log_amount` — *safe_for_baseline*
log1p(max(amount, 0)). log1p is safe at amount=0 (returns 0 rather than -inf); the max(...,0) guard is defensive since Milestone 3 already confirmed no negative amounts exist in the raw file.

### `isFlaggedFraud` — *safe_for_baseline*
Raw PaySim rule-based flag, passed through unchanged. Not the target (isFraud); Milestone 4 found it only catches 16 of 8,213 fraud cases, so treat it as a weak signal.

## Balance features

### `is_dest_merchant` — *safe_for_baseline*
1 if nameDest starts with 'M' (merchant), else 0. PaySim does not simulate merchant balances (Milestone 3); used to interpret dest_balance_mismatch correctly.

### `orig_balance_before` — *safe_for_baseline*
= oldbalanceOrg, renamed for clarity.

### `orig_balance_after` — *safe_for_baseline*
= newbalanceOrig, renamed for clarity.

### `dest_balance_before` — *safe_for_baseline*
= oldbalanceDest, renamed for clarity.

### `dest_balance_after` — *safe_for_baseline*
= newbalanceDest, renamed for clarity.

### `orig_balance_delta` — *safe_for_baseline*
orig_balance_after - orig_balance_before. Signed: positive for CASH_IN (funds added), typically negative otherwise.

### `dest_balance_delta` — *safe_for_baseline*
dest_balance_after - dest_balance_before.

### `amount_to_orig_balance_ratio` — *safe_for_baseline*
amount / orig_balance_before when orig_balance_before > 0, else NaN (safe zero handling — never divides by zero or raises). Pair with orig_balance_before_zero to distinguish 'undefined' from 'small'.

### `amount_to_dest_balance_ratio` — *safe_for_baseline*
amount / dest_balance_before when dest_balance_before > 0, else NaN (safe zero handling).

### `orig_balance_before_zero` — *safe_for_baseline*
1 if orig_balance_before == 0, else 0.

### `orig_balance_after_zero` — *safe_for_baseline*
1 if orig_balance_after == 0, else 0.

### `dest_balance_before_zero` — *safe_for_baseline*
1 if dest_balance_before == 0, else 0.

### `dest_balance_after_zero` — *safe_for_baseline*
1 if dest_balance_after == 0, else 0.

### `orig_balance_mismatch` — *safe_for_baseline*
1 if orig_balance_after != orig_balance_before +/- amount (direction-aware: CASH_IN adds, all other types subtract; tolerance 0.01), else 0. SIMULATOR-DERIVED DIAGNOSTIC ONLY (Milestone 3) — a large share of legitimate rows also mismatch; this is NOT an automatic fraud indicator.

### `dest_balance_mismatch` — *safe_for_baseline*
1 if dest_balance_after != dest_balance_before + amount (tolerance 0.01), else 0. Defined as 0 (not applicable) for merchant destinations, since PaySim does not track merchant balances at all — use is_dest_merchant to distinguish this case. SIMULATOR-DERIVED DIAGNOSTIC ONLY, not an automatic fraud indicator (see orig_balance_mismatch).

## Identity features

### `nameOrig` — *identifier*
Raw origin account id. High-cardinality identifier, kept for traceability/joins (e.g. to a future graph milestone), not intended as a raw model input.

### `nameDest` — *identifier*
Raw destination account id. Same caveat as nameOrig.

### `origin_txn_count_causal` — *safe_for_baseline*
Count of PRIOR transactions by this nameOrig, strictly before the current row in file/step order (0 for a first occurrence). Computed in a single sequential pass; never includes the current row or any later transaction.

### `origin_total_amount_causal` — *safe_for_baseline*
Sum of amounts of this nameOrig's prior transactions only (0.0 if none).

### `origin_avg_amount_causal` — *safe_for_baseline*
origin_total_amount_causal / origin_txn_count_causal when count > 0, else NaN (no prior history yet).

### `origin_is_first_seen` — *safe_for_baseline*
1 if this is nameOrig's first transaction in file/step order (origin_txn_count_causal == 0), else 0.

### `dest_txn_count_causal` — *safe_for_baseline*
Count of PRIOR transactions to this nameDest, strictly before the current row (0 for a first occurrence).

### `dest_total_amount_causal` — *safe_for_baseline*
Sum of amounts of this nameDest's prior transactions only (0.0 if none).

### `dest_avg_amount_causal` — *safe_for_baseline*
dest_total_amount_causal / dest_txn_count_causal when count > 0, else NaN.

### `dest_is_first_seen` — *safe_for_baseline*
1 if this is nameDest's first transaction in file/step order, else 0.

### `amount_vs_origin_avg_ratio` — *safe_for_baseline*
amount / origin_avg_amount_causal when defined, else NaN. How large this transaction is relative to this account's own prior average (cold-start-safe).

### `amount_vs_dest_avg_ratio` — *safe_for_baseline*
amount / dest_avg_amount_causal when defined, else NaN.

## Identifier features

### `row_id` — *identifier*
Synthetic sequential row index (0-based), not part of raw PaySim. Added only for traceability/joins across chunks.

## Recommended baseline feature columns

`hour_of_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`, `amount`, `log_amount`, `isFlaggedFraud`, `is_dest_merchant`, `orig_balance_before`, `orig_balance_after`, `dest_balance_before`, `dest_balance_after`, `orig_balance_delta`, `dest_balance_delta`, `amount_to_orig_balance_ratio`, `amount_to_dest_balance_ratio`, `orig_balance_before_zero`, `orig_balance_after_zero`, `dest_balance_before_zero`, `dest_balance_after_zero`, `orig_balance_mismatch`, `dest_balance_mismatch`, `origin_txn_count_causal`, `origin_total_amount_causal`, `origin_avg_amount_causal`, `origin_is_first_seen`, `dest_txn_count_causal`, `dest_total_amount_causal`, `dest_avg_amount_causal`, `dest_is_first_seen`, `amount_vs_origin_avg_ratio`, `amount_vs_dest_avg_ratio`

## Excluded from baseline (exploratory only)

`step`, `day_index`

## Identifiers (not model features)

`row_id`, `type`, `nameOrig`, `nameDest`

