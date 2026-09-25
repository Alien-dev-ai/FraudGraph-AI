# FraudGraph AI — Graph-Derived Feature Specification (Milestone 7)

Generated from `src/graph_features_raw_paysim.py::FEATURE_METADATA` (single source of truth — do not hand-edit this file).

`isFraud` is never read by this module (see `INPUT_COLUMNS`).

## Overlap analysis vs. Milestone 5

Milestone 5's causal identity features (`origin_txn_count_causal`, `destination_txn_count_causal` [named `dest_txn_count_causal`], `origin_total_amount_causal`, `destination_total_amount_causal` [`dest_total_amount_causal`], `origin_avg_amount_causal`, `destination_avg_amount_causal` [`dest_avg_amount_causal`]) are per-account totals, independent of *who* the counterparty was. None of them are reimplemented here. This module adds relational information instead: unique-counterparty counts (degree), pairwise relationship history, dual-role detection, and per-account concentration across counterparties.

**Measured overlap finding**: because Milestone 6 found essentially zero repeated (nameOrig, nameDest) pairs in the 2,000,000-row validated subset, `origin_out_degree_causal` and `dest_in_degree_causal` closely track Milestone 5's `origin_txn_count_causal` / `dest_txn_count_causal` *in this specific dataset instance* — they would diverge meaningfully in a dataset with more repeat-pair structure. This is reported transparently rather than presenting these as more novel than they empirically are here.

## Deferred: two-hop / shared-counterparty features

Investigated using Milestone 6's already-built 2,000,000-row graph (no larger graph was built). With mean out/in-degree around 0.70 (Milestone 6), most nodes have 0-1 direct neighbors, so two-hop neighborhoods are trivial for the majority of accounts, and the 'shared counterparty' signal already reduces to `dest_in_degree_causal` for this dataset's structure. A causal streaming two-hop feature was not implemented — see `explore_two_hop_neighborhoods` in the module for the exploratory characterization actually run.

## Features

### `row_id` — baseline_eligible: *identifier*
- **Definition**: Synthetic sequential row index (0-based).
- **Graph interpretation**: Not a graph feature; join key only.
- **Causal status**: n/a
- **Memory/computation**: O(1) per row.
- **Known limitations**: None.

### `origin_out_degree_causal` — baseline_eligible: *yes*
- **Definition**: Number of DISTINCT destination accounts this nameOrig has sent to, strictly before the current row.
- **Graph interpretation**: Causal out-degree of the origin node in the account-level transaction graph (Milestone 6's node/edge model), as of just before this transaction.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) lookup (len of a dict) against the streaming origin_to_dest_counts structure.
- **Known limitations**: Measured to closely track Milestone 5's origin_txn_count_causal in THIS dataset (differs only when an origin repeats a destination, which Milestone 6 found occurs in 0 of 2,000,000 validated rows) — see reports/graph_feature_specification.md 'Overlap analysis' for the measured divergence rate.

### `dest_in_degree_causal` — baseline_eligible: *yes*
- **Definition**: Number of DISTINCT origin accounts that have sent to this nameDest, strictly before the current row.
- **Graph interpretation**: Causal in-degree of the destination node — this transaction's 'fan-in so far' signal.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) lookup against dest_to_origin_counts.
- **Known limitations**: Same measured near-equivalence to Milestone 5's dest_txn_count_causal as origin_out_degree_causal above, for the same reason (near-zero repeated pairs).

### `pair_seen_before_causal` — baseline_eligible: *yes*
- **Definition**: 1 if this exact (nameOrig, nameDest) pair has occurred in a strictly prior row, else 0.
- **Graph interpretation**: Edge-history indicator — has this specific relationship (this graph edge) been used before.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) dict-of-dict lookup.
- **Known limitations**: Measured to be 1 for essentially 0 rows in the 2,000,000-row validated subset (Milestone 6: repeated_pair_count = 0), so this feature is a constant-0 column across the entire validated subset. Retained because it is cheap, well-defined, and would activate correctly if a future dataset variant had genuine repeat-pair structure — the test suite proves it works via a hand-crafted fixture with real repeated pairs, not just the (trivially passing) real dataset.

### `pair_prior_count_causal` — baseline_eligible: *yes*
- **Definition**: Number of times this exact (nameOrig, nameDest) pair has occurred in strictly prior rows (0 if never).
- **Graph interpretation**: Edge weight (transaction count) as of just before this transaction — the causal analogue of Milestone 6's static transaction_count edge attribute.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) dict-of-dict lookup.
- **Known limitations**: Same near-always-0 caveat as pair_seen_before_causal in this dataset instance.

### `origin_seen_as_dest_before_causal` — baseline_eligible: *yes*
- **Definition**: 1 if this nameOrig has appeared as a DESTINATION in any strictly prior row (i.e. has previously received money), else 0.
- **Graph interpretation**: Dual-role / layering signal: this account is now sending money but has previously been a money recipient too.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) membership check against dest_to_origin_counts' key set.
- **Known limitations**: Not cross-checked against Milestone 5 (no equivalent feature exists there) — genuinely new relational information not derivable from per-account totals alone. Measured on the 2,000,000-row validated subset: rate = 0.00505% (101 of 2,000,000 rows) — rare but genuinely nonzero, unlike pair_seen_before_causal, so this feature is NOT a constant column in practice.

### `dest_seen_as_origin_before_causal` — baseline_eligible: *yes*
- **Definition**: 1 if this nameDest has appeared as an ORIGIN in any strictly prior row (i.e. has previously sent money), else 0.
- **Graph interpretation**: Dual-role / layering signal from the destination side: money is now flowing into an account that has previously sent money out.
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(1) membership check against origin_to_dest_counts' key set.
- **Known limitations**: Same non-constant nature as origin_seen_as_dest_before_causal. Measured on the 2,000,000-row validated subset: rate = 0.02805% (561 of 2,000,000 rows) — about 5.5x more common than the origin-side version, consistent with destinations (in-degree mean 0.70+) being more likely to have prior activity than origins.

### `origin_outgoing_concentration_causal` — baseline_eligible: *yes*
- **Definition**: max(prior transaction counts to any single destination) / (total prior transaction count), for this nameOrig. NaN if no prior transactions.
- **Graph interpretation**: How concentrated this origin's prior outgoing activity is on a single counterparty (1.0 = all prior transactions went to one destination; lower = spread across several).
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(degree) — sum/max over the origin's current counterparty-count dict; degree is small (Milestone 6: out-degree max 3 in the 2M subset), so this stays cheap.
- **Known limitations**: Trivially 1.0 whenever origin_out_degree_causal <= 1 (the overwhelming majority of rows, since origins rarely repeat at all — Milestone 4: only 0.15% of origin accounts transact more than once). Confirmed on the 2,000,000-row subset: only defined for 921 rows (0.046%), with mean 0.9995 when defined — essentially always 1.0 in practice, so this feature carries very little standalone signal in this dataset. Amount-weighted concentration was considered and deferred as redundant given how rarely an origin has more than one prior counterparty to weigh.

### `dest_incoming_concentration_causal` — baseline_eligible: *yes*
- **Definition**: max(prior transaction counts from any single origin) / (total prior transaction count), for this nameDest. NaN if no prior transactions.
- **Graph interpretation**: How concentrated this destination's prior incoming activity is from a single counterparty (1.0 = all prior incoming transactions came from one origin).
- **Causal status**: causal (prior rows only, file/row order)
- **Memory/computation**: O(degree) — degree can be larger here (Milestone 6: in-degree max 101 in the 2M subset) but still small relative to available memory.
- **Known limitations**: More often meaningfully < 1.0 than the origin-side version, since destinations (especially merchants) commonly receive from multiple distinct origins (Milestone 4: mean 2.34 transactions per destination account). Confirmed on the 2,000,000-row subset: defined for 1,149,411 rows (57.5%), with mean 0.294 when defined — genuinely informative and NOT degenerate, unlike the origin-side version.

## Baseline-ready columns

`origin_out_degree_causal`, `dest_in_degree_causal`, `pair_seen_before_causal`, `pair_prior_count_causal`, `origin_seen_as_dest_before_causal`, `dest_seen_as_origin_before_causal`, `origin_outgoing_concentration_causal`, `dest_incoming_concentration_causal`

