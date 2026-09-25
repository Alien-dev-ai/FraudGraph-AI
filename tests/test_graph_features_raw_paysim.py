"""Tests for src.graph_features_raw_paysim, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v

Primary fixture (tests/fixtures/graph_mini.csv, reused from Milestone 6),
hand-verified row-by-row expected causal features:
  row0 C1->M1 (step1): all zero/NaN (first ever transaction)
  row1 C2->C3 (step2): all zero/NaN (first for C2/C3)
  row2 C1->M1 (step3, 2nd C1->M1): degree 1/1, pair_seen=1, count=1, concentration 1.0/1.0
  row3 C1->M1 (step4, 3rd C1->M1): degree 1/1, pair_seen=1, count=2, concentration 1.0/1.0
  row4 C4->C4 (step5, self-loop): all zero/NaN
  row5 C5->M1 (step6): dest_in_degree=1 (M1 seen once via C1), dest_concentration=1.0
  row6 C2->C3 (step7, 2nd C2->C3): degree 1/1, pair_seen=1, count=1, concentration 1.0/1.0
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import graph_features_raw_paysim as gf  # noqa: E402
from src import features_raw_paysim as feat  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GRAPH_CSV = FIXTURES_DIR / "graph_mini.csv"
DUAL_ROLE_CSV = FIXTURES_DIR / "dual_role.csv"
SAME_STEP_CSV = FIXTURES_DIR / "same_step_ties.csv"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"

SMALL_CHUNKSIZE = 2  # forces 4 chunks over graph_mini.csv's 7 rows


class TestNoLabelLeakage(unittest.TestCase):
    def test_isfraud_not_in_input_columns(self):
        self.assertNotIn("isFraud", gf.INPUT_COLUMNS)

    def test_isfraud_not_in_feature_metadata(self):
        self.assertNotIn("isFraud", gf.FEATURE_METADATA)

    def test_source_fixture_has_isfraud_but_output_ignores_it(self):
        raw = pd.read_csv(GRAPH_CSV)
        self.assertTrue((raw["isFraud"] != 0).any())  # fixture has real 1s
        df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
        self.assertNotIn("isFraud", df.columns)

    def test_altering_isfraud_values_does_not_change_output(self):
        # Build a modified copy of the fixture with different isFraud
        # values and confirm the graph features are byte-identical.
        raw = pd.read_csv(GRAPH_CSV)
        modified = raw.copy()
        modified["isFraud"] = 1 - modified["isFraud"]  # flip every label
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod_path = Path(tmp_dir) / "modified.csv"
            modified.to_csv(mod_path, index=False)
            df_original = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
            df_modified = gf.build_graph_features_sample(mod_path, chunksize=SMALL_CHUNKSIZE, max_rows=100)
            pd.testing.assert_frame_equal(
                df_original[gf.all_feature_columns()], df_modified[gf.all_feature_columns()]
            )


class TestDegreeCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)

    def test_first_transaction_has_zero_degree(self):
        row0 = self.df.iloc[0]
        self.assertEqual(row0["origin_out_degree_causal"], 0)
        self.assertEqual(row0["dest_in_degree_causal"], 0)

    def test_repeated_pair_out_degree_stays_at_one(self):
        # C1->M1 occurs 3 times; out-degree counts UNIQUE destinations,
        # so it must stay 1 across all three occurrences, not grow to 3.
        c1_m1_rows = self.df[(self.df["nameOrig"] == "C1") & (self.df["nameDest"] == "M1")]
        self.assertEqual(len(c1_m1_rows), 3)
        self.assertTrue((c1_m1_rows.iloc[1:]["origin_out_degree_causal"] == 1).all())
        self.assertTrue((c1_m1_rows.iloc[1:]["dest_in_degree_causal"] == 1).all())

    def test_dest_in_degree_reflects_multiple_origins(self):
        # M1 receives from C1 (x3) and C5 (x1) -> when C5 sends, M1's
        # in-degree (unique senders) is 1 (only C1 so far).
        row5 = self.df[(self.df["nameOrig"] == "C5") & (self.df["nameDest"] == "M1")].iloc[0]
        self.assertEqual(row5["dest_in_degree_causal"], 1)


class TestRepeatedCounterpartyBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)

    def test_pair_prior_count_increments_correctly(self):
        c1_m1_rows = self.df[(self.df["nameOrig"] == "C1") & (self.df["nameDest"] == "M1")].reset_index(drop=True)
        self.assertEqual(list(c1_m1_rows["pair_prior_count_causal"]), [0, 1, 2])

    def test_concentration_is_one_when_single_counterparty_repeats(self):
        c1_m1_rows = self.df[(self.df["nameOrig"] == "C1") & (self.df["nameDest"] == "M1")].reset_index(drop=True)
        self.assertTrue(math.isnan(c1_m1_rows.loc[0, "origin_outgoing_concentration_causal"]))
        self.assertEqual(c1_m1_rows.loc[1, "origin_outgoing_concentration_causal"], 1.0)
        self.assertEqual(c1_m1_rows.loc[2, "origin_outgoing_concentration_causal"], 1.0)


class TestFirstSeenRelationshipBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)

    def test_first_occurrence_pair_not_seen_before(self):
        row0 = self.df.iloc[0]
        self.assertEqual(row0["pair_seen_before_causal"], 0)

    def test_second_occurrence_pair_seen_before(self):
        c1_m1_rows = self.df[(self.df["nameOrig"] == "C1") & (self.df["nameDest"] == "M1")].reset_index(drop=True)
        self.assertEqual(list(c1_m1_rows["pair_seen_before_causal"]), [0, 1, 1])

    def test_c2_c3_pair_seen_before_on_repeat(self):
        c2_c3_rows = self.df[(self.df["nameOrig"] == "C2") & (self.df["nameDest"] == "C3")].reset_index(drop=True)
        self.assertEqual(list(c2_c3_rows["pair_seen_before_causal"]), [0, 1])


class TestDualRoleDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = gf.build_graph_features_sample(DUAL_ROLE_CSV, chunksize=1, max_rows=100)

    def test_no_dual_role_on_first_transaction(self):
        row0 = self.df.iloc[0]  # C1 -> C2
        self.assertEqual(row0["origin_seen_as_dest_before_causal"], 0)
        self.assertEqual(row0["dest_seen_as_origin_before_causal"], 0)

    def test_origin_seen_as_dest_before_triggers(self):
        # row1: C2 -> C3. C2 was a DESTINATION in row0, so acting as
        # origin now, it should be flagged as having received before.
        row1 = self.df.iloc[1]
        self.assertEqual(row1["nameOrig"], "C2")
        self.assertEqual(row1["origin_seen_as_dest_before_causal"], 1)

    def test_dest_seen_as_origin_before_triggers(self):
        # row2: C4 -> C1. C1 was an ORIGIN in row0, so being a
        # destination now, it should be flagged as having sent before.
        row2 = self.df.iloc[2]
        self.assertEqual(row2["nameDest"], "C1")
        self.assertEqual(row2["dest_seen_as_origin_before_causal"], 1)

    def test_graph_mini_fixture_never_triggers_dual_role(self):
        # sanity check: graph_mini.csv has no genuine dual-role accounts
        # (the C4 self-loop doesn't count -- same row, not prior).
        df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
        self.assertTrue((df["origin_seen_as_dest_before_causal"] == 0).all())
        self.assertTrue((df["dest_seen_as_origin_before_causal"] == 0).all())


class TestSelfLoopHandling(unittest.TestCase):
    def test_self_loop_row_has_zero_degree_and_no_dual_role(self):
        df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
        row = df[(df["nameOrig"] == "C4") & (df["nameDest"] == "C4")].iloc[0]
        self.assertEqual(row["origin_out_degree_causal"], 0)
        self.assertEqual(row["dest_in_degree_causal"], 0)
        self.assertEqual(row["pair_seen_before_causal"], 0)
        self.assertEqual(row["origin_seen_as_dest_before_causal"], 0)
        self.assertEqual(row["dest_seen_as_origin_before_causal"], 0)


class TestSameStepCausalOrdering(unittest.TestCase):
    """Mirrors Milestone 5's same-step audit test for the tabular
    pipeline, applied here to graph features. Fixture: 3 rows at step=5
    (C1->M1, C1->M2, C2->M1) plus a step=6 control row."""

    @classmethod
    def setUpClass(cls):
        cls.df_tiny_chunks = gf.build_graph_features_sample(SAME_STEP_CSV, chunksize=1, max_rows=100)
        cls.df_one_chunk = gf.build_graph_features_sample(SAME_STEP_CSV, chunksize=10, max_rows=100)

    def test_input_fixture_has_same_step_ties(self):
        raw = pd.read_csv(SAME_STEP_CSV)
        self.assertEqual(raw["step"].iloc[0], raw["step"].iloc[1])
        self.assertEqual(raw["step"].iloc[0], raw["step"].iloc[2])

    def test_earlier_row_unaffected_by_later_same_step_row(self):
        # row0 (C1->M1, first in file) must NOT see row2 (C2->M1, later,
        # same step) -- dest_in_degree(M1) must be 0, not 1.
        row0 = self.df_tiny_chunks.iloc[0]
        self.assertEqual(row0["nameOrig"], "C1")
        self.assertEqual(row0["nameDest"], "M1")
        self.assertEqual(row0["dest_in_degree_causal"], 0)

    def test_later_row_correctly_sees_earlier_same_step_row(self):
        # row2 (C2->M1, third in file, same step as row0) DOES see
        # row0's C1->M1 as prior -- dest_in_degree(M1) must be 1.
        row2 = self.df_tiny_chunks.iloc[2]
        self.assertEqual(row2["nameOrig"], "C2")
        self.assertEqual(row2["nameDest"], "M1")
        self.assertEqual(row2["dest_in_degree_causal"], 1)

    def test_result_identical_regardless_of_chunk_boundaries(self):
        pd.testing.assert_frame_equal(self.df_tiny_chunks, self.df_one_chunk)


class TestChunkBoundaryEquivalenceAndDeterminism(unittest.TestCase):
    def test_output_identical_regardless_of_chunksize(self):
        df_small = gf.build_graph_features_sample(GRAPH_CSV, chunksize=1, max_rows=100)
        df_large = gf.build_graph_features_sample(GRAPH_CSV, chunksize=100, max_rows=100)
        pd.testing.assert_frame_equal(df_small, df_large)

    def test_repeated_runs_produce_identical_output(self):
        df1 = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
        df2 = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=100)
        pd.testing.assert_frame_equal(df1, df2)


class TestZeroAndEmptyNeighborhoodHandling(unittest.TestCase):
    def test_first_ever_row_has_nan_concentration_not_error(self):
        df = gf.build_graph_features_sample(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=1)
        row0 = df.iloc[0]
        self.assertTrue(math.isnan(row0["origin_outgoing_concentration_causal"]))
        self.assertTrue(math.isnan(row0["dest_incoming_concentration_causal"]))

    def test_empty_input_produces_empty_frame_with_correct_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_csv = Path(tmp_dir) / "empty.csv"
            empty_csv.write_text(
                "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
                "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
            )
            df = gf.build_graph_features_sample(empty_csv, chunksize=10, max_rows=100)
            self.assertEqual(len(df), 0)


class TestCrossValidationAgainstMilestone5(unittest.TestCase):
    """When no repeated pairs exist for an account (true for every
    account in eda_mini.csv), origin_out_degree_causal must exactly
    equal Milestone 5's origin_txn_count_causal -- both count 'prior
    transactions by this account' and coincide whenever there are no
    repeated destinations, which Milestone 6 found to be true almost
    everywhere in the real dataset.
    """

    def test_degree_matches_m5_txn_count_when_no_repeated_pairs(self):
        m5_df = feat.build_features(EDA_CSV, chunksize=4)
        m7_df = gf.build_graph_features_sample(EDA_CSV, chunksize=4, max_rows=100)
        self.assertEqual(len(m5_df), len(m7_df))
        for i in range(len(m5_df)):
            self.assertEqual(
                m5_df.loc[i, "origin_txn_count_causal"],
                m7_df.loc[i, "origin_out_degree_causal"],
                msg=f"row {i} diverges -- eda_mini.csv has no repeated pairs, so these must match",
            )
            self.assertEqual(
                m5_df.loc[i, "dest_txn_count_causal"],
                m7_df.loc[i, "dest_in_degree_causal"],
            )


class TestValidateOverSubset(unittest.TestCase):
    def test_runs_cleanly_and_cross_checks_pass(self):
        result = gf.validate_over_subset(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=7)
        self.assertEqual(result["total_rows"], 7)
        self.assertFalse(result["isFraud_used"])
        self.assertEqual(result["cross_check_final_unique_origin_accounts"], 4)  # C1,C2,C4,C5
        self.assertEqual(result["cross_check_final_unique_directed_pairs"], 4)
        self.assertEqual(result["cross_check_final_repeated_pair_count"], 2)  # C1->M1, C2->C3

    def test_pair_seen_before_rate_matches_hand_count(self):
        result = gf.validate_over_subset(GRAPH_CSV, chunksize=SMALL_CHUNKSIZE, max_rows=7)
        # 2 rows (the 2nd C1->M1 and 2nd C2->C3) have pair_seen_before=1
        # out of 7 total -- wait: 3rd C1->M1 also has it =1, so 3 rows total
        self.assertAlmostEqual(result["pair_seen_before_rate"], 3 / 7)


class TestFeatureMetadataAndSpec(unittest.TestCase):
    def test_every_metadata_entry_has_required_fields(self):
        required = {"definition", "graph_interpretation", "causal_status",
                    "baseline_eligible", "memory_computation", "known_limitations"}
        for col, meta in gf.FEATURE_METADATA.items():
            self.assertTrue(required.issubset(meta.keys()), msg=col)

    def test_baseline_ready_columns_exclude_row_id(self):
        self.assertNotIn("row_id", gf.baseline_ready_columns())

    def test_writes_markdown_with_overlap_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "spec.md"
            gf.write_feature_spec(out_path)
            text = out_path.read_text()
            self.assertIn("Overlap analysis", text)
            self.assertIn("Deferred: two-hop", text)
            for col in gf.all_feature_columns():
                self.assertIn(f"`{col}`", text)


class TestMissingInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            list(gf.iter_graph_feature_chunks(FIXTURES_DIR / "does_not_exist.csv"))


if __name__ == "__main__":
    unittest.main()
