# FraudGraph AI — Account-Level Transaction Graph Specification (Milestone 6)

Scope: **graph construction and validation only**. No graph-derived ML
features, no model training, no full-size serialized graph artifact.
Source: `src/graph_raw_paysim.py`. Reads only
`data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv`.

## Graph definition

- **Node**: one node per unique account identifier (every distinct
  `nameOrig` or `nameDest` value across the processed rows). **Not** one
  node per transaction.
- **Edge**: `nameOrig -> nameDest`, directed, matching PaySim's
  sender/receiver semantics (funds conceptually flow origin -> destination).
- **Representation**: `networkx.DiGraph` (not `MultiDiGraph`) with one
  aggregated edge per unique `(nameOrig, nameDest)` pair, carrying
  transaction statistics as edge attributes.

## Node semantics

Each node carries one attribute:
- `account_type`: `"merchant"` if the account id starts with `"M"`, else
  `"customer"` — derived purely from the Milestone 3 id-prefix
  convention, never from balance fields or `isFraud`.

No other node attributes are set in this milestone (no fraud labels, no
graph-derived centrality/features — see "Deferred to Milestone 7").

## Edge semantics

Each edge carries:
| Attribute | Meaning |
|---|---|
| `transaction_count` | number of raw rows aggregated into this edge |
| `total_amount` | sum of `amount` across those rows |
| `min_amount` / `max_amount` | smallest/largest `amount` among those rows |
| `first_step` / `last_step` | earliest/latest raw `step` among those rows (file/row order — see Milestone 5's chronological-ordering documentation for what "earliest" means when steps tie) |
| `type_counts` | dict of `{transaction_type: count}` among those rows |
| `flagged_fraud_count` | sum of raw `isFlaggedFraud` (NOT `isFraud`) among those rows |

`isFraud` is never read by `src/graph_raw_paysim.py` (see `INPUT_COLUMNS`
in the module) — topology and every edge/node attribute above are
entirely label-independent, by construction, not by convention.

## Aggregation decision (MultiDiGraph vs. aggregated DiGraph)

**Chosen: aggregated DiGraph.** This was an empirically measured
decision, not an assumption made up front:

1. **Repeated pairs are measured to be extremely rare in this dataset.**
   Across a 1M/2M/4M-row probe run during this milestone's design phase,
   edge count exactly equaled row count at every checkpoint (zero
   repeated pairs through 4,000,000 rows). On the 2,000,000-row real
   validation run reported below, `repeated_pair_count = 0` and
   `edge_count = row_count = 2,000,000` exactly, even though 921 origin
   accounts *did* repeat (each repeat just happened to target a
   different destination). This is consistent with Milestone 4: only
   9,298 of 6,353,307 origin accounts (0.15%) ever transact more than
   once across the *entire* dataset.
2. Given that, a `MultiDiGraph`'s "one edge object per transaction"
   would produce almost exactly as many edges as the aggregated
   approach — the choice buys almost nothing in this dataset's actual
   structure, while adding NetworkX's extra per-pair multi-edge key
   layer (`G._succ[u][v][key] = attrs` vs. `G._succ[u][v] = attrs`) on
   every single edge for a capability (parallel edges) that's almost
   never exercised here.
3. **No information is lost for the rare pairs that do repeat**:
   `transaction_count > 1` on an edge is exactly the marker for a
   genuinely repeated pair, and `type_counts`/`min_amount`/`max_amount`/
   `first_step`/`last_step` still capture the distribution across those
   repeats — just not each individual raw row.
4. The edge-attribute **schema** itself was also measured, not assumed:
   a version storing 5 flat `count_<TYPE>` integer fields per edge used
   *more* memory than the nested `type_counts` dict version (3.33GB vs.
   2.67GB peak footprint at 2,000,000 rows) — flat fields pushed each
   edge's attribute dict past a CPython dict-resize threshold. The
   nested-dict schema was kept for measured efficiency, not aesthetics.

If a future dataset variant had genuinely high repeat-pair rates, this
decision should be revisited — the measurement above is specific to
PaySim's simulated account-generation behavior.

## Causal / time interpretation

`first_step` / `last_step` reflect file/row order, matching Milestone 5's
established convention: the raw CSV is verified non-decreasing by `step`,
and ties within a step are broken by file/row order (a documented,
deterministic convention — PaySim provides no finer timestamp). No new
causal claims are introduced by the graph; it inherits Milestone 5's
documented limitation as-is.

## Merchant handling

Merchant destination accounts (`nameDest` starting with `"M"`) are
**retained as ordinary graph nodes** — never filtered out. Milestones
3-5 established that PaySim does not simulate merchant balances
(`oldbalanceDest = newbalanceDest = 0` for effectively all such rows);
this is a characteristic of the *raw balance columns*, not a reason to
exclude merchant accounts from the graph. On the 2,000,000-row real
validation run, 668,858 of 2,849,485 nodes (23.5%) are tagged
`account_type = "merchant"` and fully participate in the graph topology
(in-degree up to 101 in this subset).

## Self-loop handling

Measured **directly on the full 6,362,620-row raw file** (a single
vectorized pandas comparison of `nameOrig == nameDest` — cheap, unlike
graph construction, so this one check was run at full scale rather than
extrapolated): **zero rows have `nameOrig == nameDest`.** Recorded as
`SELF_LOOP_ROW_COUNT_FULL_DATASET = 0` in the module. Nothing was
removed — there was nothing to remove. The construction code does not
special-case or filter self-loops; `nx.DiGraph` supports them natively,
and the test suite (`TestSelfLoops` in
`tests/test_graph_raw_paysim.py`) proves retention behavior against a
fixture that deliberately includes one, so the code path is verified
even though the real dataset never exercises it.

## Repeated-edge handling

See "Aggregation decision" above. `repeated_pair_count` (edges with
`transaction_count > 1`) is reported as an explicit graph statistic; on
the real 2,000,000-row validation run it was 0, matching the direct
row-vs-edge-count comparison.

## Memory considerations and a measured, reported limitation

Before attempting any real-data build, this milestone measured actual
(not estimated) memory cost via a throwaway probe script at three
checkpoints:

| Rows | Nodes | Edges | Peak RSS | Runtime |
|---|---|---|---|---|
| 1,000,000 | 1,422,617 | 1,000,000 | 1.48 GB | 17.0s |
| 2,000,000 | 2,849,485 | 2,000,000 | 2.85 GB | 36.0s |
| 4,000,000 | 5,706,278 | 4,000,000 | 5.11 GB | 61.3s |

Extrapolating to the full 6,362,620 rows: **~7.7-8.1 GB peak RSS**,
~95-100s runtime, against ~7.5 GB observed free RAM on this machine at
measurement time (fluctuating 5-7.5 GB across this project's sessions,
on a shared 16 GB machine already running other applications) — too
tight a margin to safely attempt live.

**Per explicit instruction to stop and report rather than invent a
workaround, and per the user's explicit decision, this milestone builds
and validates the graph against a capped subset
(`FULL_DATASET_SAFE_ROW_CAP = 2,000,000` rows) instead of the full
6,362,620-row file.** The capped real-data run (reported in the Final
Report) completed at **4.89 GB peak RSS** — higher than the isolated
2M-row build probe (2.85GB) because the full pipeline also holds an
independent validation pass's sets and computes strongly/weakly
connected components concurrently with the live graph. This is itself
useful evidence: the *full-pipeline* (build + validate + stats) cost is
meaningfully higher than the *build-only* cost, so the full-dataset
extrapolation above likely understates true full-pipeline memory need.

**The full 6,362,620-row graph has not been constructed.** See "Known
limitations" below for next-step options.

## Known limitations

1. **Full-dataset graph not built.** Only a 2,000,000-row (31.4%) prefix
   of the raw file has been processed into a graph this milestone,
   because the measured/extrapolated memory cost of the full dataset
   was too close to observed available RAM to attempt safely. The node
   count, degree distributions, and component statistics reported here
   describe this subset, not the complete transaction graph, and should
   not be treated as final production statistics.
2. **Per-row Python loop.** `build_graph` iterates rows with
   `itertuples` rather than a vectorized pandas/NetworkX bulk-load path,
   because per-row conditional aggregation (first-seen vs. repeat pair)
   doesn't vectorize cleanly with NetworkX's API. This is the dominant
   runtime cost (roughly 60,000-70,000 rows/sec observed) and would need
   revisiting for a full-dataset build.
3. **No on-disk/incremental graph store.** The graph is held entirely
   in process memory; there is no checkpointing or streaming persistence
   strategy that would let a full build proceed with bounded memory
   (e.g., writing completed edges to disk and dropping them from RAM).
   That would be a reasonable next step if full-scale construction is
   required before more RAM is available.
4. **`min_amount`/`max_amount`/`type_counts` are exact but only over the
   processed subset** — extending the row cap later requires rebuilding
   from scratch (no incremental "resume from row N" path exists yet).

## Deferred to Milestone 7 (not built here)

- Graph-derived features (degree, centrality, community/clustering
  signals, neighborhood aggregates) for use in modeling.
- Attaching `isFraud` to nodes/edges for any purpose (this milestone
  does not use it at all, including for validation — validation used
  only `step`, `type`, `amount`, `nameOrig`, `nameDest`,
  `isFlaggedFraud`).
- A strategy (and explicit approval) for full-scale (6,362,620-row)
  graph construction — options include: accepting the measured memory
  risk and running it directly, a streaming/checkpointed construction
  that bounds peak memory, or running on a machine with more available
  RAM.
- Any graph serialization/persistence artifact under `data/processed/`.
- Model training of any kind.

## Files

- `src/graph_raw_paysim.py` — construction, statistics, independent
  validation, CLI
- `tests/test_graph_raw_paysim.py`, `tests/fixtures/graph_mini.csv` —
  hand-verified deterministic tests
- `reports/graph_summary.json` — machine-readable output of the capped
  real-data run
- `reports/graph_specification.md` — this document
