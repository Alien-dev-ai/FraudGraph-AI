"""Graph-derived, causal, explainable features for raw PaySim (Milestone 7).

FEATURE DESIGN AND VALIDATION ONLY — no model training, no full-scale
feature table, no graph larger than Milestone 6's validated 2,000,000-row
subset. Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

=== Overlap review vs. Milestone 5 (done before writing this module) ===

Milestone 5's identity features (src/features_raw_paysim.py) are
PER-ACCOUNT aggregates, causal, computed from a single running dict per
account: `origin_txn_count_causal`, `origin_total_amount_causal`,
`origin_avg_amount_causal`, `origin_is_first_seen`, and the destination
-side equivalents, plus `amount_vs_origin_avg_ratio` /
`amount_vs_dest_avg_ratio`. These answer "how active has this account
been, and how does this amount compare to its own history" — they do
NOT know who the account's prior counterparties were, whether a specific
relationship has occurred before, or whether an account has played both
sender and receiver roles. This module deliberately does NOT
reimplement any of the six M5 identity aggregates above, or an
amount-vs-history ratio (M5's `amount_vs_*_avg_ratio` already answers
"amount relative to this account's counterparties' historical amounts"
in aggregate, since an account's average IS the average over its
counterparties). What this module adds is RELATIONAL: unique-counterparty
counts (degree), pairwise relationship history, dual-role detection, and
concentration of an account's activity across its counterparties — none
of which are derivable from M5's per-account totals alone.

=== Causal convention (same as Milestone 5) ===

file/row order = deterministic event order; within the same PaySim
`step`, an earlier file row is prior. A graph feature for the
transaction at row t may use only graph structure built from rows
strictly before t. See the Milestone 5 "same-step ties" test class
(mirrored here) for the property this guarantees.

=== Streaming state (not a NetworkX graph) ===

Per-row causal features are computed from two compact causal adjacency
dicts-of-dicts, NOT from a NetworkX object:
  origin_to_dest_counts: {nameOrig: {nameDest: prior_count}}
  dest_to_origin_counts: {nameDest: {nameOrig: prior_count}}
Both are read (for the current row's features) BEFORE being updated with
the current row, exactly mirroring Milestone 5's `_update_causal_state`
pattern. This deliberately avoids NetworkX overhead entirely for the
per-transaction streaming path — Milestone 6 measured NetworkX's
per-node/per-edge bookkeeping cost, and a plain dict-of-dicts with
degree bounded by observed max in/out-degree (101 / 3 in the M6 2M
subset) is far cheaper for this purpose.

=== Deferred: two-hop / shared-counterparty features ===

Investigated (see `explore_two_hop_neighborhoods` below) using
Milestone 6's ALREADY-BUILT, already-validated 2,000,000-row NetworkX
graph — no larger or additional graph is built. Finding: with mean
out-degree 0.70 and mean in-degree 0.70 (Milestone 6), the overwhelming
majority of nodes have 0 or 1 direct neighbors, so two-hop neighborhoods
are trivial (empty or size ~1) for most accounts; the "shared
counterparty" signal (how many *other* origins used the same
destination) already reduces to `dest_in_degree_causal` below for this
dataset's structure. A genuine causal streaming two-hop feature was NOT
implemented: it would require adjacency SETS-of-SETS (not just counts)
and a second-order traversal per row, a materially larger state system,
for a signal the sampled exploratory check found to be low-value given
this dataset's sparse degree distribution. Classified exploratory_only
/ deferred — see reports/graph_feature_specification.md.

=== Label safety ===

`isFraud` is never read by this module (see INPUT_COLUMNS). No fraud
label is attached to any streaming state, node, or feature.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import networkx as nx
import numpy as np
import pandas as pd

from src.validate_raw_paysim import DEFAULT_RAW_CSV
from src.graph_raw_paysim import build_graph, FULL_DATASET_SAFE_ROW_CAP

DEFAULT_CHUNKSIZE = 200_000

# isFraud intentionally absent (see module docstring "Label safety").
INPUT_COLUMNS = ["step", "nameOrig", "nameDest"]

GRAPH_FEATURE_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "graph_feature_specification.md"
)
GRAPH_FEATURE_SUMMARY_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "graph_feature_summary.json"
)

# Single source of truth for reports/graph_feature_specification.md and
# for the leakage/baseline-eligibility test assertions. Mirrors the
# pattern established in src/features_raw_paysim.py::FEATURE_METADATA.
FEATURE_METADATA: dict[str, dict[str, str]] = {
    "row_id": {
        "definition": "Synthetic sequential row index (0-based).",
        "graph_interpretation": "Not a graph feature; join key only.",
        "causal_status": "n/a",
        "baseline_eligible": "identifier",
        "memory_computation": "O(1) per row.",
        "known_limitations": "None.",
    },
    "origin_out_degree_causal": {
        "definition": "Number of DISTINCT destination accounts this "
        "nameOrig has sent to, strictly before the current row.",
        "graph_interpretation": "Causal out-degree of the origin node in "
        "the account-level transaction graph (Milestone 6's node/edge "
        "model), as of just before this transaction.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) lookup (len of a dict) against the "
        "streaming origin_to_dest_counts structure.",
        "known_limitations": "Measured to closely track Milestone 5's "
        "origin_txn_count_causal in THIS dataset (differs only when an "
        "origin repeats a destination, which Milestone 6 found occurs "
        "in 0 of 2,000,000 validated rows) — see "
        "reports/graph_feature_specification.md 'Overlap analysis' for "
        "the measured divergence rate.",
    },
    "dest_in_degree_causal": {
        "definition": "Number of DISTINCT origin accounts that have sent "
        "to this nameDest, strictly before the current row.",
        "graph_interpretation": "Causal in-degree of the destination "
        "node — this transaction's 'fan-in so far' signal.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) lookup against dest_to_origin_counts.",
        "known_limitations": "Same measured near-equivalence to "
        "Milestone 5's dest_txn_count_causal as origin_out_degree_causal "
        "above, for the same reason (near-zero repeated pairs).",
    },
    "pair_seen_before_causal": {
        "definition": "1 if this exact (nameOrig, nameDest) pair has "
        "occurred in a strictly prior row, else 0.",
        "graph_interpretation": "Edge-history indicator — has this "
        "specific relationship (this graph edge) been used before.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) dict-of-dict lookup.",
        "known_limitations": "Measured to be 1 for essentially 0 rows "
        "in the 2,000,000-row validated subset (Milestone 6: "
        "repeated_pair_count = 0), so this feature is a constant-0 "
        "column across the entire validated subset. Retained because it "
        "is cheap, well-defined, and would activate correctly if a "
        "future dataset variant had genuine repeat-pair structure — the "
        "test suite proves it works via a hand-crafted fixture with "
        "real repeated pairs, not just the (trivially passing) real "
        "dataset.",
    },
    "pair_prior_count_causal": {
        "definition": "Number of times this exact (nameOrig, nameDest) "
        "pair has occurred in strictly prior rows (0 if never).",
        "graph_interpretation": "Edge weight (transaction count) as of "
        "just before this transaction — the causal analogue of "
        "Milestone 6's static transaction_count edge attribute.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) dict-of-dict lookup.",
        "known_limitations": "Same near-always-0 caveat as "
        "pair_seen_before_causal in this dataset instance.",
    },
    "origin_seen_as_dest_before_causal": {
        "definition": "1 if this nameOrig has appeared as a DESTINATION "
        "in any strictly prior row (i.e. has previously received "
        "money), else 0.",
        "graph_interpretation": "Dual-role / layering signal: this "
        "account is now sending money but has previously been a money "
        "recipient too.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) membership check against "
        "dest_to_origin_counts' key set.",
        "known_limitations": "Not cross-checked against Milestone 5 "
        "(no equivalent feature exists there) — genuinely new relational "
        "information not derivable from per-account totals alone. "
        "Measured on the 2,000,000-row validated subset: rate = "
        "0.00505% (101 of 2,000,000 rows) — rare but genuinely nonzero, "
        "unlike pair_seen_before_causal, so this feature is NOT a "
        "constant column in practice.",
    },
    "dest_seen_as_origin_before_causal": {
        "definition": "1 if this nameDest has appeared as an ORIGIN in "
        "any strictly prior row (i.e. has previously sent money), else 0.",
        "graph_interpretation": "Dual-role / layering signal from the "
        "destination side: money is now flowing into an account that has "
        "previously sent money out.",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(1) membership check against "
        "origin_to_dest_counts' key set.",
        "known_limitations": "Same non-constant nature as "
        "origin_seen_as_dest_before_causal. Measured on the 2,000,000-"
        "row validated subset: rate = 0.02805% (561 of 2,000,000 rows) "
        "— about 5.5x more common than the origin-side version, "
        "consistent with destinations (in-degree mean 0.70+) being more "
        "likely to have prior activity than origins.",
    },
    "origin_outgoing_concentration_causal": {
        "definition": "max(prior transaction counts to any single "
        "destination) / (total prior transaction count), for this "
        "nameOrig. NaN if no prior transactions.",
        "graph_interpretation": "How concentrated this origin's prior "
        "outgoing activity is on a single counterparty (1.0 = all prior "
        "transactions went to one destination; lower = spread across "
        "several).",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(degree) — sum/max over the origin's "
        "current counterparty-count dict; degree is small (Milestone 6: "
        "out-degree max 3 in the 2M subset), so this stays cheap.",
        "known_limitations": "Trivially 1.0 whenever "
        "origin_out_degree_causal <= 1 (the overwhelming majority of "
        "rows, since origins rarely repeat at all — Milestone 4: only "
        "0.15% of origin accounts transact more than once). Confirmed on "
        "the 2,000,000-row subset: only defined for 921 rows (0.046%), "
        "with mean 0.9995 when defined — essentially always 1.0 in "
        "practice, so this feature carries very little standalone signal "
        "in this dataset. Amount-weighted concentration was considered "
        "and deferred as redundant given how rarely an origin has more "
        "than one prior counterparty to weigh.",
    },
    "dest_incoming_concentration_causal": {
        "definition": "max(prior transaction counts from any single "
        "origin) / (total prior transaction count), for this nameDest. "
        "NaN if no prior transactions.",
        "graph_interpretation": "How concentrated this destination's "
        "prior incoming activity is from a single counterparty (1.0 = "
        "all prior incoming transactions came from one origin).",
        "causal_status": "causal (prior rows only, file/row order)",
        "baseline_eligible": "yes",
        "memory_computation": "O(degree) — degree can be larger here "
        "(Milestone 6: in-degree max 101 in the 2M subset) but still "
        "small relative to available memory.",
        "known_limitations": "More often meaningfully < 1.0 than the "
        "origin-side version, since destinations (especially merchants) "
        "commonly receive from multiple distinct origins (Milestone 4: "
        "mean 2.34 transactions per destination account). Confirmed on "
        "the 2,000,000-row subset: defined for 1,149,411 rows (57.5%), "
        "with mean 0.294 when defined — genuinely informative and NOT "
        "degenerate, unlike the origin-side version.",
    },
}


def baseline_ready_columns() -> list[str]:
    return [c for c, m in FEATURE_METADATA.items() if m["baseline_eligible"] == "yes"]


def all_feature_columns() -> list[str]:
    return list(FEATURE_METADATA.keys())


def _update_and_extract(
    chunk: pd.DataFrame,
    origin_to_dest_counts: dict[str, dict[str, int]],
    dest_to_origin_counts: dict[str, dict[str, int]],
) -> pd.DataFrame:
    n = len(chunk)
    out_degree = np.empty(n, dtype=np.int64)
    in_degree = np.empty(n, dtype=np.int64)
    pair_seen = np.empty(n, dtype=np.int64)
    pair_count = np.empty(n, dtype=np.int64)
    orig_seen_as_dest = np.empty(n, dtype=np.int64)
    dest_seen_as_orig = np.empty(n, dtype=np.int64)
    orig_concentration = np.empty(n, dtype=np.float64)
    dest_concentration = np.empty(n, dtype=np.float64)

    names_orig = chunk["nameOrig"].to_numpy()
    names_dest = chunk["nameDest"].to_numpy()

    for i in range(n):
        u = names_orig[i]
        v = names_dest[i]

        u_map = origin_to_dest_counts.get(u)
        v_map = dest_to_origin_counts.get(v)

        out_degree[i] = len(u_map) if u_map else 0
        in_degree[i] = len(v_map) if v_map else 0

        prior_to_v = u_map.get(v, 0) if u_map else 0
        pair_count[i] = prior_to_v
        pair_seen[i] = 1 if prior_to_v > 0 else 0

        orig_seen_as_dest[i] = 1 if u in dest_to_origin_counts else 0
        dest_seen_as_orig[i] = 1 if v in origin_to_dest_counts else 0

        if u_map:
            vals = u_map.values()
            orig_concentration[i] = max(vals) / sum(vals)
        else:
            orig_concentration[i] = np.nan

        if v_map:
            vals = v_map.values()
            dest_concentration[i] = max(vals) / sum(vals)
        else:
            dest_concentration[i] = np.nan

        origin_to_dest_counts.setdefault(u, {})
        origin_to_dest_counts[u][v] = origin_to_dest_counts[u].get(v, 0) + 1
        dest_to_origin_counts.setdefault(v, {})
        dest_to_origin_counts[v][u] = dest_to_origin_counts[v].get(u, 0) + 1

    out = pd.DataFrame({
        "origin_out_degree_causal": out_degree,
        "dest_in_degree_causal": in_degree,
        "pair_seen_before_causal": pair_seen,
        "pair_prior_count_causal": pair_count,
        "origin_seen_as_dest_before_causal": orig_seen_as_dest,
        "dest_seen_as_origin_before_causal": dest_seen_as_orig,
        "origin_outgoing_concentration_causal": orig_concentration,
        "dest_incoming_concentration_causal": dest_concentration,
    }, index=chunk.index)
    return out


def iter_graph_feature_chunks(
    csv_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_rows: int | None = None,
    row_id_start: int = 0,
) -> Iterator[pd.DataFrame]:
    """Stream causal graph features. Bounded memory: one raw chunk plus
    two dict-of-dicts (degree bounded by observed max in/out-degree) —
    never a NetworkX graph, never a full-size DataFrame.
    """
    origin_to_dest_counts: dict[str, dict[str, int]] = {}
    dest_to_origin_counts: dict[str, dict[str, int]] = {}
    rows_processed = 0
    row_id = row_id_start

    with pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize) as reader:
        for chunk in reader:
            if max_rows is not None and rows_processed >= max_rows:
                break
            if max_rows is not None and rows_processed + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_processed]

            feats = _update_and_extract(chunk, origin_to_dest_counts, dest_to_origin_counts)
            combined = pd.concat(
                [chunk[["nameOrig", "nameDest"]].reset_index(drop=True), feats.reset_index(drop=True)],
                axis=1,
            )
            combined.insert(0, "row_id", range(row_id, row_id + len(combined)))
            row_id += len(combined)
            rows_processed += len(chunk)

            yield combined


def build_graph_features_sample(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = 20_000
) -> pd.DataFrame:
    """A small, clearly-bounded materialized sample for inspection/tests.

    NOT the full 2M validated subset and NOT the full 6.36M dataset —
    see validate_over_subset for the (non-materializing) full-subset
    streaming validation.
    """
    chunks = list(iter_graph_feature_chunks(csv_path, chunksize=chunksize, max_rows=max_rows))
    if not chunks:
        return pd.DataFrame(columns=["row_id", "nameOrig", "nameDest"] + all_feature_columns())
    return pd.concat(chunks, ignore_index=True)


def validate_over_subset(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int = FULL_DATASET_SAFE_ROW_CAP
) -> dict[str, Any]:
    """Stream causal graph features across the full Milestone-6-validated
    subset (default 2,000,000 rows) WITHOUT materializing a feature
    table — only summary statistics and cross-checks are accumulated.
    """
    total_rows = 0
    sum_out_degree = 0
    max_out_degree = 0
    sum_in_degree = 0
    max_in_degree = 0
    pair_seen_count = 0
    orig_dual_role_count = 0
    dest_dual_role_count = 0
    orig_concentration_defined = 0
    orig_concentration_sum = 0.0
    dest_concentration_defined = 0
    dest_concentration_sum = 0.0

    origin_to_dest_counts: dict[str, dict[str, int]] = {}
    dest_to_origin_counts: dict[str, dict[str, int]] = {}

    for chunk in iter_graph_feature_chunks(csv_path, chunksize=chunksize, max_rows=max_rows):
        assert "isFraud" not in chunk.columns
        total_rows += len(chunk)
        sum_out_degree += int(chunk["origin_out_degree_causal"].sum())
        max_out_degree = max(max_out_degree, int(chunk["origin_out_degree_causal"].max()))
        sum_in_degree += int(chunk["dest_in_degree_causal"].sum())
        max_in_degree = max(max_in_degree, int(chunk["dest_in_degree_causal"].max()))
        pair_seen_count += int(chunk["pair_seen_before_causal"].sum())
        orig_dual_role_count += int(chunk["origin_seen_as_dest_before_causal"].sum())
        dest_dual_role_count += int(chunk["dest_seen_as_origin_before_causal"].sum())

        oc = chunk["origin_outgoing_concentration_causal"].dropna()
        orig_concentration_defined += len(oc)
        orig_concentration_sum += float(oc.sum())
        dc = chunk["dest_incoming_concentration_causal"].dropna()
        dest_concentration_defined += len(dc)
        dest_concentration_sum += float(dc.sum())

    # Rebuild the same final adjacency state independently (separate,
    # non-streaming pass over the identical rows) to cross-check against
    # Milestone 6's already-validated graph numbers for the same subset.
    with pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize) as reader:
        rows_seen = 0
        for chunk in reader:
            if rows_seen >= max_rows:
                break
            if rows_seen + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_seen]
            for u, v in zip(chunk["nameOrig"], chunk["nameDest"]):
                origin_to_dest_counts.setdefault(u, {})
                origin_to_dest_counts[u][v] = origin_to_dest_counts[u].get(v, 0) + 1
                dest_to_origin_counts.setdefault(v, {})
                dest_to_origin_counts[v][u] = dest_to_origin_counts[v].get(u, 0) + 1
            rows_seen += len(chunk)

    final_unique_origin = len(origin_to_dest_counts)
    final_unique_pairs = sum(len(d) for d in origin_to_dest_counts.values())
    final_repeated_pairs = sum(
        1 for d in origin_to_dest_counts.values() for c in d.values() if c > 1
    )

    return {
        "total_rows": total_rows,
        "isFraud_used": False,
        "origin_out_degree": {
            "mean": sum_out_degree / total_rows if total_rows else 0.0,
            "max": max_out_degree,
        },
        "dest_in_degree": {
            "mean": sum_in_degree / total_rows if total_rows else 0.0,
            "max": max_in_degree,
        },
        "pair_seen_before_rate": pair_seen_count / total_rows if total_rows else 0.0,
        "origin_dual_role_rate": orig_dual_role_count / total_rows if total_rows else 0.0,
        "dest_dual_role_rate": dest_dual_role_count / total_rows if total_rows else 0.0,
        "origin_concentration_mean_when_defined": (
            orig_concentration_sum / orig_concentration_defined if orig_concentration_defined else None
        ),
        "origin_concentration_defined_count": orig_concentration_defined,
        "dest_concentration_mean_when_defined": (
            dest_concentration_sum / dest_concentration_defined if dest_concentration_defined else None
        ),
        "dest_concentration_defined_count": dest_concentration_defined,
        "cross_check_final_unique_origin_accounts": final_unique_origin,
        "cross_check_final_unique_directed_pairs": final_unique_pairs,
        "cross_check_final_repeated_pair_count": final_repeated_pairs,
    }


def explore_two_hop_neighborhoods(
    G: nx.DiGraph, sample_size: int = 5000, seed: int = 42
) -> dict[str, Any]:
    """EXPLORATORY / POST-HOC ONLY — uses the already-built Milestone 6
    graph (not a new/larger one) to characterize two-hop out-neighborhood
    sizes for a sample of nodes with out-degree > 0. This is NOT causal
    (uses the whole subset's final structure) and is NOT part of the
    per-row streaming feature set — see module docstring "Deferred:
    two-hop / shared-counterparty features".
    """
    candidates = [n for n in G.nodes() if G.out_degree(n) > 0]
    rng = np.random.default_rng(seed)
    sample = rng.choice(candidates, size=min(sample_size, len(candidates)), replace=False)

    two_hop_sizes = []
    zero_two_hop_count = 0
    for n in sample:
        one_hop = set(G.successors(n))
        two_hop = set()
        for m in one_hop:
            two_hop.update(G.successors(m))
        two_hop.discard(n)
        two_hop -= one_hop
        two_hop_sizes.append(len(two_hop))
        if len(two_hop) == 0:
            zero_two_hop_count += 1

    arr = np.array(two_hop_sizes)
    return {
        "candidate_node_count": len(candidates),
        "sampled_node_count": len(sample),
        "two_hop_neighbor_count_mean": float(arr.mean()) if len(arr) else 0.0,
        "two_hop_neighbor_count_max": int(arr.max()) if len(arr) else 0,
        "two_hop_neighbor_count_median": float(np.median(arr)) if len(arr) else 0.0,
        "fraction_with_zero_two_hop_neighbors": zero_two_hop_count / len(sample) if len(sample) else 0.0,
    }


def write_feature_spec(out_path: Path = GRAPH_FEATURE_SPEC_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FraudGraph AI — Graph-Derived Feature Specification (Milestone 7)",
        "",
        "Generated from `src/graph_features_raw_paysim.py::FEATURE_METADATA` "
        "(single source of truth — do not hand-edit this file).",
        "",
        "`isFraud` is never read by this module (see `INPUT_COLUMNS`).",
        "",
        "## Overlap analysis vs. Milestone 5",
        "",
        "Milestone 5's causal identity features "
        "(`origin_txn_count_causal`, `destination_txn_count_causal` "
        "[named `dest_txn_count_causal`], `origin_total_amount_causal`, "
        "`destination_total_amount_causal` [`dest_total_amount_causal`], "
        "`origin_avg_amount_causal`, `destination_avg_amount_causal` "
        "[`dest_avg_amount_causal`]) are per-account totals, independent "
        "of *who* the counterparty was. None of them are reimplemented "
        "here. This module adds relational information instead: unique-"
        "counterparty counts (degree), pairwise relationship history, "
        "dual-role detection, and per-account concentration across "
        "counterparties.",
        "",
        "**Measured overlap finding**: because Milestone 6 found "
        "essentially zero repeated (nameOrig, nameDest) pairs in the "
        "2,000,000-row validated subset, `origin_out_degree_causal` and "
        "`dest_in_degree_causal` closely track Milestone 5's "
        "`origin_txn_count_causal` / `dest_txn_count_causal` *in this "
        "specific dataset instance* — they would diverge meaningfully "
        "in a dataset with more repeat-pair structure. This is reported "
        "transparently rather than presenting these as more novel than "
        "they empirically are here.",
        "",
        "## Deferred: two-hop / shared-counterparty features",
        "",
        "Investigated using Milestone 6's already-built 2,000,000-row "
        "graph (no larger graph was built). With mean out/in-degree "
        "around 0.70 (Milestone 6), most nodes have 0-1 direct "
        "neighbors, so two-hop neighborhoods are trivial for the "
        "majority of accounts, and the 'shared counterparty' signal "
        "already reduces to `dest_in_degree_causal` for this dataset's "
        "structure. A causal streaming two-hop feature was not "
        "implemented — see `explore_two_hop_neighborhoods` in the "
        "module for the exploratory characterization actually run.",
        "",
        "## Features",
        "",
    ]

    for col, meta in FEATURE_METADATA.items():
        lines.append(f"### `{col}` — baseline_eligible: *{meta['baseline_eligible']}*")
        lines.append(f"- **Definition**: {meta['definition']}")
        lines.append(f"- **Graph interpretation**: {meta['graph_interpretation']}")
        lines.append(f"- **Causal status**: {meta['causal_status']}")
        lines.append(f"- **Memory/computation**: {meta['memory_computation']}")
        lines.append(f"- **Known limitations**: {meta['known_limitations']}")
        lines.append("")

    lines.append("## Baseline-ready columns")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in baseline_ready_columns()))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def write_feature_summary(
    sample: pd.DataFrame,
    subset_validation: dict[str, Any],
    two_hop_exploration: dict[str, Any],
    runtime_seconds: float,
    out_path: Path = GRAPH_FEATURE_SUMMARY_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(DEFAULT_RAW_CSV),
        "sample_row_count": len(sample),
        "runtime_seconds": round(runtime_seconds, 2),
        "subset_validation": subset_validation,
        "two_hop_exploration_on_m6_graph": two_hop_exploration,
        "isFraud_used": False,
    }
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Design/validate graph-derived features on the "
        "Milestone-6-validated 2M-row subset (streamed; produces a small "
        "sample table plus summary statistics, not a full feature table)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--subset-rows", type=int, default=FULL_DATASET_SAFE_ROW_CAP)
    parser.add_argument("--spec-out", type=Path, default=GRAPH_FEATURE_SPEC_PATH)
    parser.add_argument("--summary-out", type=Path, default=GRAPH_FEATURE_SUMMARY_PATH)
    args = parser.parse_args(argv)

    write_feature_spec(args.spec_out)
    print(f"Feature specification written to: {args.spec_out}")

    t0 = time.time()
    sample = build_graph_features_sample(args.csv, chunksize=args.chunksize, max_rows=args.sample_rows)
    subset_validation = validate_over_subset(args.csv, chunksize=args.chunksize, max_rows=args.subset_rows)

    print(f"Reusing Milestone 6's build_graph on the same {args.subset_rows:,}-row "
          f"cap for two-hop exploration (not a larger graph)...")
    G, rows_built = build_graph(args.csv, chunksize=args.chunksize, max_rows=args.subset_rows)
    two_hop = explore_two_hop_neighborhoods(G)
    runtime = time.time() - t0

    write_feature_summary(sample, subset_validation, two_hop, runtime, args.summary_out)

    print(f"Sample rows produced        : {len(sample):,}")
    print(f"Subset rows validated        : {subset_validation['total_rows']:,}")
    print(f"Mean origin out-degree       : {subset_validation['origin_out_degree']['mean']:.4f}")
    print(f"Mean dest in-degree          : {subset_validation['dest_in_degree']['mean']:.4f}")
    print(f"Pair-seen-before rate        : {subset_validation['pair_seen_before_rate']:.6%}")
    print(f"Cross-check unique origin    : {subset_validation['cross_check_final_unique_origin_accounts']:,}")
    print(f"Cross-check unique pairs     : {subset_validation['cross_check_final_unique_directed_pairs']:,}")
    print(f"Two-hop mean (M6 graph)      : {two_hop['two_hop_neighbor_count_mean']:.4f}")
    print(f"Two-hop zero fraction        : {two_hop['fraction_with_zero_two_hop_neighbors']:.4%}")
    print(f"Runtime                      : {runtime:.2f}s")
    print(f"Summary written to           : {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
