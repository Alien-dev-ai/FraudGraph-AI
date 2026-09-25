"""Tests for src.graph_raw_paysim, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v

Fixture (tests/fixtures/graph_mini.csv), hand-verified expected graph:
  Nodes (6): C1, C2, C3, C4, C5, M1
  Edges (4): C1->M1 (x3, PAYMENT x2 + TRANSFER x1), C2->C3 (x2),
             C4->C4 (self-loop, x1), C5->M1 (x1)
  isFraud column is present in the fixture with nonzero/asymmetric
  values specifically to prove the graph module ignores it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pandas as pd

from src import graph_raw_paysim as gr  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GRAPH_CSV = FIXTURES_DIR / "graph_mini.csv"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"

SMALL_CHUNKSIZE = 2  # forces 4 chunks over graph_mini.csv's 7 rows


class TestNoLabelLeakage(unittest.TestCase):
    def test_isfraud_not_in_input_columns(self):
        self.assertNotIn("isFraud", gr.INPUT_COLUMNS)

    def test_source_fixture_has_isfraud_but_graph_ignores_it(self):
        raw = pd.read_csv(GRAPH_CSV)
        self.assertIn("isFraud", raw.columns)
        self.assertTrue((raw["isFraud"] != 0).any())  # fixture has real 1s
        G, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        for _, _, data in G.edges(data=True):
            self.assertNotIn("isFraud", data)
        for _, data in G.nodes(data=True):
            self.assertNotIn("isFraud", data)


class TestNodeCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G, cls.rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_node_count_and_identities(self):
        self.assertEqual(self.G.number_of_nodes(), 6)
        self.assertEqual(set(self.G.nodes()), {"C1", "C2", "C3", "C4", "C5", "M1"})

    def test_no_transaction_level_nodes(self):
        # 7 rows must not produce 7+ nodes -- nodes are accounts, not txns.
        self.assertLess(self.G.number_of_nodes(), self.rows)

    def test_merchant_node_retained_with_account_type(self):
        self.assertIn("M1", self.G.nodes())
        self.assertEqual(self.G.nodes["M1"]["account_type"], "merchant")

    def test_customer_nodes_tagged_correctly(self):
        for n in ["C1", "C2", "C3", "C4", "C5"]:
            self.assertEqual(self.G.nodes[n]["account_type"], "customer")


class TestDirectedEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_edge_direction_is_orig_to_dest(self):
        self.assertTrue(self.G.has_edge("C1", "M1"))
        self.assertFalse(self.G.has_edge("M1", "C1"))

    def test_edge_count_equals_unique_pairs(self):
        self.assertEqual(self.G.number_of_edges(), 4)


class TestRepeatedPairAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_c1_to_m1_aggregated_correctly(self):
        e = self.G["C1"]["M1"]
        self.assertEqual(e["transaction_count"], 3)
        self.assertAlmostEqual(e["total_amount"], 180.0)
        self.assertAlmostEqual(e["min_amount"], 30.0)
        self.assertAlmostEqual(e["max_amount"], 100.0)
        self.assertEqual(e["first_step"], 1)
        self.assertEqual(e["last_step"], 4)

    def test_c1_to_m1_type_counts(self):
        e = self.G["C1"]["M1"]
        self.assertEqual(e["type_counts"], {"PAYMENT": 2, "TRANSFER": 1})

    def test_c1_to_m1_flagged_fraud_count(self):
        # only row 3 (step 3) has isFlaggedFraud=1
        e = self.G["C1"]["M1"]
        self.assertEqual(e["flagged_fraud_count"], 1)

    def test_c2_to_c3_aggregated_correctly(self):
        e = self.G["C2"]["C3"]
        self.assertEqual(e["transaction_count"], 2)
        self.assertAlmostEqual(e["total_amount"], 260.0)
        self.assertEqual(e["type_counts"], {"CASH_OUT": 1, "PAYMENT": 1})

    def test_singleton_edge_min_equals_max(self):
        e = self.G["C5"]["M1"]
        self.assertEqual(e["transaction_count"], 1)
        self.assertEqual(e["min_amount"], e["max_amount"])
        self.assertEqual(e["type_counts"], {"CASH_IN": 1})


class TestSelfLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_self_loop_retained(self):
        self.assertTrue(self.G.has_edge("C4", "C4"))
        self.assertEqual(nx.number_of_selfloops(self.G), 1)

    def test_self_loop_edge_attributes_correct(self):
        e = self.G["C4"]["C4"]
        self.assertEqual(e["transaction_count"], 1)
        self.assertAlmostEqual(e["total_amount"], 10.0)

    def test_full_dataset_has_zero_self_loops_documented(self):
        # This is a recorded measurement (module docstring / constant),
        # not something re-derived from a fixture -- confirm it's set.
        self.assertEqual(gr.SELF_LOOP_ROW_COUNT_FULL_DATASET, 0)


class TestChunkBoundaryEquivalence(unittest.TestCase):
    def test_output_identical_regardless_of_chunksize(self):
        g_small, rows_small = gr.build_graph(GRAPH_CSV, chunksize=1)
        g_large, rows_large = gr.build_graph(GRAPH_CSV, chunksize=100)
        self.assertEqual(rows_small, rows_large)
        self.assertEqual(set(g_small.nodes()), set(g_large.nodes()))
        self.assertEqual(set(g_small.edges()), set(g_large.edges()))
        for u, v in g_small.edges():
            self.assertEqual(g_small[u][v], g_large[u][v])

    def test_repeated_build_is_deterministic(self):
        g1, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        g2, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        self.assertEqual(sorted(g1.nodes()), sorted(g2.nodes()))
        self.assertEqual(sorted(g1.edges()), sorted(g2.edges()))
        for u, v in g1.edges():
            self.assertEqual(g1[u][v], g2[u][v])


class TestMaxRowsCap(unittest.TestCase):
    def test_max_rows_caps_processing(self):
        G, rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=4)
        self.assertEqual(rows, 4)
        # only rows 1-4 processed: C1->M1 (x3), C2->C3 (x1, row2)
        self.assertEqual(G["C1"]["M1"]["transaction_count"], 3)
        self.assertEqual(G["C2"]["C3"]["transaction_count"], 1)
        self.assertFalse(G.has_edge("C4", "C4"))  # row 5 excluded
        self.assertNotIn("C5", G.nodes())  # row 6 excluded


class TestGraphStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G, _ = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        cls.stats = gr.compute_graph_statistics(cls.G)

    def test_node_and_edge_counts(self):
        self.assertEqual(self.stats["node_count"], 6)
        self.assertEqual(self.stats["edge_count"], 4)

    def test_unique_origin_and_destination_counts(self):
        self.assertEqual(self.stats["unique_origin_accounts"], 4)  # C1,C2,C4,C5
        self.assertEqual(self.stats["unique_destination_accounts"], 3)  # M1,C3,C4

    def test_accounts_on_both_sides(self):
        self.assertEqual(self.stats["accounts_on_both_sides"], 1)  # C4 only

    def test_isolated_nodes_is_zero(self):
        self.assertEqual(self.stats["isolated_node_count"], 0)

    def test_self_loop_and_repeated_pair_counts(self):
        self.assertEqual(self.stats["self_loop_count"], 1)
        self.assertEqual(self.stats["repeated_pair_count"], 2)  # C1->M1, C2->C3
        self.assertEqual(self.stats["unique_directed_pair_count"], 4)

    def test_density(self):
        expected = 4 / (6 * 5)
        self.assertAlmostEqual(self.stats["density"], expected)

    def test_degree_summaries(self):
        self.assertEqual(self.stats["in_degree_summary"]["max"], 2)  # M1
        self.assertEqual(self.stats["out_degree_summary"]["max"], 1)

    def test_strongly_connected_components(self):
        scc = self.stats["strongly_connected_components"]
        self.assertEqual(scc["count"], 6)  # no genuine multi-node cycles
        self.assertEqual(scc["largest_size"], 1)
        self.assertEqual(scc["size_gt_1_count"], 0)

    def test_weakly_connected_components(self):
        wcc = self.stats["weakly_connected_components"]
        self.assertEqual(wcc["count"], 3)  # {C1,M1,C5}, {C2,C3}, {C4}
        self.assertEqual(wcc["largest_size"], 3)

    def test_merchant_and_customer_node_counts(self):
        self.assertEqual(self.stats["merchant_node_count"], 1)
        self.assertEqual(self.stats["customer_node_count"], 5)


class TestIndependentValidation(unittest.TestCase):
    def test_independent_accumulation_matches_graph(self):
        G, rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        validation = gr.validate_graph_against_raw(G, GRAPH_CSV, rows, chunksize=SMALL_CHUNKSIZE)
        self.assertTrue(validation["all_checks_passed"])
        for check_name, passed in validation["checks"].items():
            self.assertTrue(passed, msg=f"{check_name} failed")

    def test_transaction_count_sum_equals_row_count(self):
        G, rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        total = sum(d["transaction_count"] for _, _, d in G.edges(data=True))
        self.assertEqual(total, rows)
        self.assertEqual(total, 7)

    def test_validation_detects_a_genuine_mismatch(self):
        # Sanity check that the validator isn't vacuously true: build the
        # graph on only 4 rows but validate against a raw pass of all 7 --
        # this MUST be reported as a mismatch.
        G, rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=4)
        validation = gr.validate_graph_against_raw(G, GRAPH_CSV, rows_processed=7, chunksize=SMALL_CHUNKSIZE)
        self.assertFalse(validation["all_checks_passed"])
        self.assertFalse(validation["checks"]["transaction_count_sum_matches_row_count"])

    def test_reuses_eda_mini_fixture_too(self):
        G, rows = gr.build_graph(EDA_CSV, chunksize=4)
        self.assertEqual(rows, 13)
        validation = gr.validate_graph_against_raw(G, EDA_CSV, rows, chunksize=4)
        self.assertTrue(validation["all_checks_passed"])


class TestSummaryWriter(unittest.TestCase):
    def test_writes_valid_json_with_expected_fields(self):
        G, rows = gr.build_graph(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE)
        stats = gr.compute_graph_statistics(G)
        validation = gr.validate_graph_against_raw(G, GRAPH_CSV, rows, chunksize=SMALL_CHUNKSIZE)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "summary.json"
            gr.write_graph_summary(stats, validation, rows, is_full_dataset=False,
                                    runtime_seconds=1.23, out_path=out_path)
            loaded = json.loads(out_path.read_text())
            self.assertEqual(loaded["rows_processed"], 7)
            self.assertFalse(loaded["is_full_dataset"])
            self.assertFalse(loaded["isFraud_used"])
            self.assertIn("graph_statistics", loaded)
            self.assertIn("validation", loaded)


class TestMissingAndMalformedInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            gr.build_graph(FIXTURES_DIR / "does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
