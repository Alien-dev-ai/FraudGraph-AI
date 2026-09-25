"""Account-level transaction graph construction for raw PaySim (Milestone 6).

GRAPH CONSTRUCTION AND VALIDATION ONLY. No model training, no graph-derived
ML feature table, no full-size serialized graph artifact. Reads only:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

LABEL SAFETY: `isFraud` is never read by this module (see INPUT_COLUMNS).
Graph topology and edge/node attributes are entirely label-independent.

=== Graph model decision (documented per Milestone 6 requirement) ===

Node definition: one node per unique account identifier (every distinct
`nameOrig` or `nameDest` value). NOT one node per transaction.

Edge direction: nameOrig -> nameDest, matching PaySim's sender/receiver
semantics (funds conceptually flow from origin to destination).

Representation: an AGGREGATED DiGraph (one edge per unique (nameOrig,
nameDest) pair, carrying transaction statistics), not a MultiDiGraph with
one edge per transaction. This was an empirically-measured decision, not
an assumption:

  - Repeated origin->destination pairs are measured to be extremely rare
    in this dataset: across the first 4,000,000 rows probed during this
    milestone's design phase, ZERO repeated pairs occurred (edge count
    exactly equaled row count at every checkpoint). This is consistent
    with Milestone 4, which found only 9,298 of 6,353,307 origin
    accounts (0.15%) ever transact more than once. So a MultiDiGraph's
    "one edge per transaction" would produce almost exactly as many
    edges as an aggregated DiGraph anyway — aggregation buys essentially
    no edge-count reduction here.
  - What it DOES buy: a MultiDiGraph adds an extra per-pair key layer
    (`G._succ[u][v][key] = attrs`) on top of DiGraph's `G._succ[u][v] =
    attrs`, for a benefit (representing genuine parallel edges) that is
    almost never exercised in this dataset. An aggregated DiGraph avoids
    that layer while still correctly preserving repeat-pair information
    via `transaction_count` (verified >1 exactly when a pair truly
    repeats) — no information is lost for the rare pairs that do repeat.
  - Edge-attribute schema was also measured, not assumed: a version with
    a nested `type_counts` dict (one small dict holding 1 type entry per
    edge, for the near-universal case of a single-transaction edge) used
    LESS memory than a version with 5 flat `count_<TYPE>` integer fields
    per edge (2.67GB vs 3.33GB peak footprint at 2,000,000 rows) — flat
    fields pushed each edge's attribute dict past a CPython dict-resize
    threshold. The nested-dict schema was kept for this reason.

=== Memory strategy and a measured, reported limitation ===

The full raw file is 6,362,620 rows. Before attempting a full build, this
milestone measured actual (not estimated) memory cost at 1M/2M/4M-row
checkpoints (peak RSS 1.48GB / 2.85GB / 5.11GB respectively) and
extrapolated the full build to ~7.7-8.1GB peak RSS against ~7.5GB
observed free RAM on this machine at measurement time — too tight a
margin to safely attempt live, per this milestone's explicit instruction
to STOP and report rather than invent a workaround.

Per user decision, this milestone therefore builds and validates the
graph against a documented, capped real-data subset
(FULL_DATASET_SAFE_ROW_CAP rows — chosen to match the measured-safe
2,000,000-row checkpoint) rather than the full 6,362,620-row file. The
full-scale build is deferred; see reports/graph_specification.md
"Known limitations" for the exact numbers and next-step options.

=== Merchant handling ===

Merchant destination accounts (nameDest starting with 'M') are retained
as ordinary graph nodes — never filtered out. Milestones 3-5 established
that PaySim does not simulate merchant balances (oldbalanceDest =
newbalanceDest = 0 for essentially all such rows); this is a data
characteristic of the *raw columns*, not a reason to exclude merchant
accounts from the graph. `account_type` ('merchant' vs 'customer') is
recorded as a node attribute, derived purely from the id prefix
convention (Milestone 3), independent of any balance or label field.

=== Self-loops ===

Measured directly against the FULL raw file (a cheap vectorized pandas
comparison, unlike graph construction): zero rows have nameOrig ==
nameDest (see SELF_LOOP_ROW_COUNT_FULL_DATASET below). None were
removed — there was nothing to remove — and the construction code does
not special-case or filter them; if a future dataset variant contained
self-loops, nx.DiGraph supports them natively and they would be
retained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from src.ingest import DatasetNotFoundError, SchemaValidationError, load_csv, validate_columns
from src.validate_raw_paysim import DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = 200_000

# Measured directly on the full raw file during this milestone's design
# phase (a single vectorized pandas comparison over nameOrig/nameDest —
# cheap, unlike graph construction, so this was run at full scale rather
# than extrapolated). See module docstring "Self-loops".
SELF_LOOP_ROW_COUNT_FULL_DATASET = 0
FULL_DATASET_ROW_COUNT = 6_362_620

# Chosen to match the largest checkpoint directly measured safe during
# this milestone's memory probing (2,000,000 rows -> 2.85GB peak RSS,
# against ~7.5GB free at measurement time). See module docstring
# "Memory strategy". Full-scale construction is deferred pending a
# revised memory strategy or explicit approval to accept the risk.
FULL_DATASET_SAFE_ROW_CAP = 2_000_000

# isFraud is intentionally absent — graph topology/attributes must be
# label-independent (see module docstring "LABEL SAFETY").
INPUT_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFlaggedFraud"]

GRAPH_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "graph_specification.md"
)
GRAPH_STATS_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "graph_summary.json"
)


def _account_type(account_id: str) -> str:
    return "merchant" if account_id.startswith("M") else "customer"


def build_graph(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int | None = None
) -> tuple[nx.DiGraph, int]:
    """Build the aggregated account-level DiGraph.

    Streams the CSV in chunks (usecols=INPUT_COLUMNS; isFraud is never
    read). Only one raw chunk plus the graph itself are held in memory —
    no separate accumulation dict is built alongside the graph (data is
    written directly into the graph's own edge/node attribute dicts, so
    memory is not doubled).

    `max_rows`, if given, stops after that many rows have been processed
    (used for the documented safe-subset real-data run; see
    FULL_DATASET_SAFE_ROW_CAP). Returns (graph, rows_processed).
    """
    G = nx.DiGraph()
    rows_processed = 0

    with pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize) as reader:
        for chunk in reader:
            if max_rows is not None and rows_processed >= max_rows:
                break
            if max_rows is not None and rows_processed + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_processed]

            for row in chunk.itertuples(index=False):
                u, v = row.nameOrig, row.nameDest

                if u not in G:
                    G.add_node(u, account_type=_account_type(u))
                if v not in G:
                    G.add_node(v, account_type=_account_type(v))

                if G.has_edge(u, v):
                    e = G[u][v]
                    e["transaction_count"] += 1
                    e["total_amount"] += row.amount
                    if row.amount < e["min_amount"]:
                        e["min_amount"] = row.amount
                    if row.amount > e["max_amount"]:
                        e["max_amount"] = row.amount
                    e["last_step"] = row.step
                    e["type_counts"][row.type] = e["type_counts"].get(row.type, 0) + 1
                    e["flagged_fraud_count"] += int(row.isFlaggedFraud)
                else:
                    G.add_edge(
                        u, v,
                        transaction_count=1,
                        total_amount=float(row.amount),
                        min_amount=float(row.amount),
                        max_amount=float(row.amount),
                        first_step=int(row.step),
                        last_step=int(row.step),
                        type_counts={row.type: 1},
                        flagged_fraud_count=int(row.isFlaggedFraud),
                    )

            rows_processed += len(chunk)

    return G, rows_processed


def _degree_summary(degrees: Iterable[int]) -> dict[str, float]:
    arr = np.array(list(degrees), dtype=np.int64)
    if arr.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def compute_graph_statistics(G: nx.DiGraph) -> dict[str, Any]:
    origins = {u for u, _ in G.edges()}
    dests = {v for _, v in G.edges()}
    both_sides = origins & dests

    repeated_pairs = sum(1 for _, _, d in G.edges(data=True) if d["transaction_count"] > 1)
    self_loops = nx.number_of_selfloops(G)

    sccs = list(nx.strongly_connected_components(G))
    wccs = list(nx.weakly_connected_components(G))
    scc_sizes = sorted((len(c) for c in sccs), reverse=True)
    wcc_sizes = sorted((len(c) for c in wccs), reverse=True)

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "unique_origin_accounts": len(origins),
        "unique_destination_accounts": len(dests),
        "accounts_on_both_sides": len(both_sides),
        "isolated_node_count": sum(1 for _ in nx.isolates(G)),
        "self_loop_count": self_loops,
        "repeated_pair_count": repeated_pairs,
        "unique_directed_pair_count": G.number_of_edges(),
        "density": nx.density(G),
        "in_degree_summary": _degree_summary(d for _, d in G.in_degree()),
        "out_degree_summary": _degree_summary(d for _, d in G.out_degree()),
        "strongly_connected_components": {
            "count": len(sccs),
            "largest_size": scc_sizes[0] if scc_sizes else 0,
            "size_1_count": sum(1 for s in scc_sizes if s == 1),
            "size_gt_1_count": sum(1 for s in scc_sizes if s > 1),
        },
        "weakly_connected_components": {
            "count": len(wccs),
            "largest_size": wcc_sizes[0] if wcc_sizes else 0,
            "size_1_count": sum(1 for s in wcc_sizes if s == 1),
            "size_gt_1_count": sum(1 for s in wcc_sizes if s > 1),
        },
        "merchant_node_count": sum(
            1 for _, d in G.nodes(data=True) if d.get("account_type") == "merchant"
        ),
        "customer_node_count": sum(
            1 for _, d in G.nodes(data=True) if d.get("account_type") == "customer"
        ),
    }


def independent_raw_accumulation(
    csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE, max_rows: int | None = None
) -> dict[str, Any]:
    """Recompute the cross-checkable quantities directly from the raw CSV
    via a SEPARATE code path (plain sets/counters, no graph object at
    all), so validation is a genuine independent cross-check rather than
    reading the graph's own numbers back at itself.
    """
    unique_origin: set[str] = set()
    unique_dest: set[str] = set()
    unique_pairs: set[tuple[str, str]] = set()
    self_loop_count = 0
    total_rows = 0

    with pd.read_csv(csv_path, usecols=INPUT_COLUMNS, chunksize=chunksize) as reader:
        for chunk in reader:
            if max_rows is not None and total_rows >= max_rows:
                break
            if max_rows is not None and total_rows + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - total_rows]

            unique_origin.update(chunk["nameOrig"].unique())
            unique_dest.update(chunk["nameDest"].unique())
            unique_pairs.update(zip(chunk["nameOrig"], chunk["nameDest"]))
            self_loop_count += int((chunk["nameOrig"] == chunk["nameDest"]).sum())
            total_rows += len(chunk)

    return {
        "total_rows": total_rows,
        "unique_origin_accounts": len(unique_origin),
        "unique_destination_accounts": len(unique_dest),
        "unique_directed_pairs": len(unique_pairs),
        "self_loop_count": self_loop_count,
    }


def validate_graph_against_raw(
    G: nx.DiGraph, csv_path: Path, rows_processed: int, chunksize: int = DEFAULT_CHUNKSIZE
) -> dict[str, Any]:
    """Cross-check the built graph against an independently-computed pass
    over the same (possibly capped) raw rows.
    """
    baseline = independent_raw_accumulation(csv_path, chunksize=chunksize, max_rows=rows_processed)

    graph_origin_count = len({u for u, _ in G.edges()})
    graph_dest_count = len({v for _, v in G.edges()})
    graph_pair_count = G.number_of_edges()
    graph_self_loops = nx.number_of_selfloops(G)
    sum_transaction_count = sum(d["transaction_count"] for _, _, d in G.edges(data=True))

    checks = {
        "row_count_matches": rows_processed == baseline["total_rows"],
        "unique_origin_matches": graph_origin_count == baseline["unique_origin_accounts"],
        "unique_destination_matches": graph_dest_count == baseline["unique_destination_accounts"],
        "unique_pair_count_matches": graph_pair_count == baseline["unique_directed_pairs"],
        "self_loop_count_matches": graph_self_loops == baseline["self_loop_count"],
        "transaction_count_sum_matches_row_count": sum_transaction_count == rows_processed,
    }

    return {
        "rows_validated": rows_processed,
        "baseline": baseline,
        "graph_derived": {
            "unique_origin_accounts": graph_origin_count,
            "unique_destination_accounts": graph_dest_count,
            "unique_directed_pairs": graph_pair_count,
            "self_loop_count": graph_self_loops,
            "sum_edge_transaction_count": sum_transaction_count,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def write_graph_summary(
    stats: dict[str, Any],
    validation: dict[str, Any],
    rows_processed: int,
    is_full_dataset: bool,
    runtime_seconds: float,
    out_path: Path = GRAPH_STATS_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(DEFAULT_RAW_CSV),
        "rows_processed": rows_processed,
        "is_full_dataset": is_full_dataset,
        "full_dataset_row_count": FULL_DATASET_ROW_COUNT,
        "runtime_seconds": round(runtime_seconds, 2),
        "graph_statistics": stats,
        "validation": validation,
        "isFraud_used": False,
    }
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the account-level PaySim transaction "
        "graph on a memory-safe row cap (see FULL_DATASET_SAFE_ROW_CAP)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-rows", type=int, default=FULL_DATASET_SAFE_ROW_CAP)
    parser.add_argument("--summary-out", type=Path, default=GRAPH_STATS_PATH)
    args = parser.parse_args(argv)

    try:
        validate_columns(load_csv(args.csv))
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Graph construction failed: {exc}", file=sys.stderr)
        return 1

    t0 = time.time()
    G, rows_processed = build_graph(args.csv, chunksize=args.chunksize, max_rows=args.max_rows)
    stats = compute_graph_statistics(G)
    validation = validate_graph_against_raw(G, args.csv, rows_processed, chunksize=args.chunksize)
    runtime = time.time() - t0

    is_full = rows_processed >= FULL_DATASET_ROW_COUNT
    write_graph_summary(stats, validation, rows_processed, is_full, runtime, args.summary_out)

    print(f"Rows processed        : {rows_processed:,} of {FULL_DATASET_ROW_COUNT:,} "
          f"({'FULL DATASET' if is_full else 'CAPPED SUBSET'})")
    print(f"Nodes                 : {stats['node_count']:,}")
    print(f"Edges                 : {stats['edge_count']:,}")
    print(f"Self-loops            : {stats['self_loop_count']:,}")
    print(f"Repeated pairs        : {stats['repeated_pair_count']:,}")
    print(f"All validation checks passed: {validation['all_checks_passed']}")
    print(f"Runtime               : {runtime:.2f}s")
    print(f"Summary written to    : {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
