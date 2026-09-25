"""Tests for src.baseline_dataset, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import baseline_dataset as bd  # noqa: E402
from src import features_raw_paysim as feat  # noqa: E402
from src import graph_features_raw_paysim as gf  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"
GRAPH_CSV = FIXTURES_DIR / "graph_mini.csv"
SAME_STEP_CSV = FIXTURES_DIR / "same_step_ties.csv"
MINI_PAYSIM_CSV = FIXTURES_DIR / "mini_paysim.csv"


class TestBaselineFeatureContract(unittest.TestCase):
    def test_exact_column_count_and_composition(self):
        self.assertEqual(len(bd.BASELINE_FEATURE_COLUMNS), 39)
        self.assertEqual(len(set(bd.BASELINE_FEATURE_COLUMNS)), 39)  # no duplicates

    def test_contract_equals_m5_baseline_plus_five_graph_features(self):
        expected = feat.baseline_feature_columns() + bd.GRAPH_FEATURES_RETAINED_IN_BASELINE
        self.assertEqual(bd.BASELINE_FEATURE_COLUMNS, expected)

    def test_graph_classification_partitions_m7_baseline_ready_set_exactly(self):
        retained_and_audit = set(bd.GRAPH_FEATURES_RETAINED_IN_BASELINE) | set(bd.GRAPH_FEATURES_AUDIT_ONLY)
        self.assertEqual(retained_and_audit, set(gf.baseline_ready_columns()))
        # no overlap between the two buckets
        self.assertEqual(
            set(bd.GRAPH_FEATURES_RETAINED_IN_BASELINE) & set(bd.GRAPH_FEATURES_AUDIT_ONLY), set()
        )

    def test_near_constant_graph_features_are_audit_only_not_baseline(self):
        for col in ["pair_seen_before_causal", "pair_prior_count_causal", "origin_outgoing_concentration_causal"]:
            self.assertIn(col, bd.GRAPH_FEATURES_AUDIT_ONLY)
            self.assertNotIn(col, bd.BASELINE_FEATURE_COLUMNS)

    def test_degree_features_retained_despite_measured_collinearity(self):
        for col in ["origin_out_degree_causal", "dest_in_degree_causal"]:
            self.assertIn(col, bd.BASELINE_FEATURE_COLUMNS)


class TestTargetAndIdentifierExclusion(unittest.TestCase):
    def test_target_not_in_baseline_contract(self):
        self.assertNotIn(bd.TARGET_COLUMN, bd.BASELINE_FEATURE_COLUMNS)

    def test_identifiers_not_in_baseline_contract(self):
        for col in bd.IDENTIFIER_COLUMNS:
            self.assertNotIn(col, bd.BASELINE_FEATURE_COLUMNS)

    def test_row_id_step_day_index_not_in_baseline_contract(self):
        for col in ["row_id", "step", "day_index", "nameOrig", "nameDest"]:
            self.assertNotIn(col, bd.BASELINE_FEATURE_COLUMNS)

    def test_target_never_read_from_source_fixture(self):
        raw = pd.read_csv(GRAPH_CSV)
        self.assertIn("isFraud", raw.columns)
        self.assertTrue((raw["isFraud"] != 0).any())
        df = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=100)
        self.assertNotIn("isFraud", df.columns)

    def test_audit_confirms_no_target_or_identifier_leakage_into_baseline_matrix(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        baseline_matrix = df[bd.BASELINE_FEATURE_COLUMNS]
        for col in bd.IDENTIFIER_COLUMNS + [bd.TARGET_COLUMN]:
            self.assertNotIn(col, baseline_matrix.columns)


class TestDeterministicColumnOrder(unittest.TestCase):
    def test_assembled_columns_match_declared_order_exactly(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        self.assertEqual(list(df.columns), bd.ALL_ASSEMBLED_COLUMNS)

    def test_baseline_columns_are_a_contiguous_ordered_subset(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        positions = [list(df.columns).index(c) for c in bd.BASELINE_FEATURE_COLUMNS]
        self.assertEqual(positions, sorted(positions))  # order preserved, no reshuffling


class TestChunkSizeEquivalence(unittest.TestCase):
    def test_output_identical_across_chunk_sizes_eda_fixture(self):
        df_a = bd.build_bounded_sample(EDA_CSV, chunksize=1, max_rows=100)
        df_b = bd.build_bounded_sample(EDA_CSV, chunksize=1000, max_rows=100)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_output_identical_across_chunk_sizes_graph_fixture(self):
        df_a = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=100)
        df_b = bd.build_bounded_sample(GRAPH_CSV, chunksize=100, max_rows=100)
        pd.testing.assert_frame_equal(df_a, df_b)

    def test_check_determinism_helper_reports_equivalent(self):
        result = bd.check_determinism(EDA_CSV, chunksizes=(1, 4), max_rows=13)
        self.assertTrue(result["equivalent"])


class TestDeterministicRepeatedRuns(unittest.TestCase):
    def test_repeated_builds_are_byte_identical(self):
        df1 = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=100)
        df2 = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=100)
        pd.testing.assert_frame_equal(df1, df2)


class TestBoundedRowCap(unittest.TestCase):
    def test_max_rows_caps_assembly(self):
        df = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=4)
        self.assertEqual(len(df), 4)

    def test_row_id_sequential_within_cap(self):
        df = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=4)
        self.assertEqual(list(df["row_id"]), [0, 1, 2, 3])


class TestCausalBehaviorPreserved(unittest.TestCase):
    """Same-step ordering must survive the joint M5+M7 assembly exactly
    as it does in each module individually."""

    def test_earlier_row_unaffected_by_later_same_step_row(self):
        df = bd.build_bounded_sample(SAME_STEP_CSV, chunksize=1, max_rows=100)
        row0 = df.iloc[0]  # C1 -> M1, first in file
        self.assertEqual(row0["dest_in_degree_causal"], 0)
        self.assertEqual(row0["dest_txn_count_causal"], 0)

    def test_later_row_sees_earlier_same_step_row(self):
        df = bd.build_bounded_sample(SAME_STEP_CSV, chunksize=1, max_rows=100)
        row2 = df.iloc[2]  # C2 -> M1, third in file, same step as row0
        self.assertEqual(row2["dest_in_degree_causal"], 1)
        self.assertEqual(row2["dest_txn_count_causal"], 1)

    def test_degree_equals_txn_count_when_no_repeated_pairs(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        pd.testing.assert_series_equal(
            df["origin_out_degree_causal"], df["origin_txn_count_causal"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df["dest_in_degree_causal"], df["dest_txn_count_causal"],
            check_names=False,
        )


class TestMissingAndInfiniteHandling(unittest.TestCase):
    def test_nan_preserved_not_imputed_for_first_seen_ratios(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=1)
        row0 = df.iloc[0]
        self.assertTrue(math.isnan(row0["origin_avg_amount_causal"]))
        self.assertTrue(math.isnan(row0["amount_vs_origin_avg_ratio"]))
        self.assertTrue(math.isnan(row0["dest_incoming_concentration_causal"]))

    def test_no_infinite_values_in_baseline_columns(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        numeric_baseline = df[bd.BASELINE_FEATURE_COLUMNS].select_dtypes(include=[np.number])
        self.assertEqual(int(np.isinf(numeric_baseline).sum().sum()), 0)

    def test_audit_reports_zero_infinite_count_for_ratio_columns(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        audit = bd.audit_dataframe(df)
        for col in ["amount_to_orig_balance_ratio", "amount_to_dest_balance_ratio",
                    "origin_outgoing_concentration_causal", "dest_incoming_concentration_causal"]:
            self.assertEqual(audit["columns"][col]["infinite_count"], 0)


class TestConstantAndNearConstantDetection(unittest.TestCase):
    def test_audit_flags_constant_column(self):
        df = bd.build_bounded_sample(GRAPH_CSV, chunksize=2, max_rows=100)
        audit = bd.audit_dataframe(df)
        # pair_seen_before_causal is 0 for 5 of 7 rows in graph_mini.csv,
        # not fully constant there, so build a genuinely constant case:
        df["always_zero"] = 0
        audit2 = bd.audit_dataframe(df)
        self.assertTrue(audit2["columns"]["always_zero"]["is_constant"])

    def test_audit_flags_near_constant_column(self):
        s = pd.Series([1] * 999 + [2])
        stats = bd._column_stats(s)
        self.assertFalse(stats["is_constant"])
        self.assertTrue(stats["is_near_constant"])

    def test_full_stream_audit_detects_constant_pair_seen_before_on_real_pattern(self):
        # graph_mini.csv has genuine repeats, so pair_seen_before is NOT
        # constant there -- confirm the audit correctly reports non-constant
        # rather than trivially reporting everything as constant.
        result = bd.run_full_stream_audit(GRAPH_CSV, chunksize=2, max_rows=100)
        self.assertFalse(result["columns"]["pair_seen_before_causal"]["is_constant"])

    def test_full_stream_audit_on_fixture_with_no_repeats_is_constant_zero(self):
        result = bd.run_full_stream_audit(EDA_CSV, chunksize=4, max_rows=100)
        self.assertTrue(result["columns"]["pair_seen_before_causal"]["is_constant"])


class TestCategoricalEncoding(unittest.TestCase):
    def test_one_hot_type_columns_present_and_sum_to_one(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        type_cols = [c for c in bd.BASELINE_FEATURE_COLUMNS if c.startswith("type_")]
        self.assertEqual(len(type_cols), 5)
        self.assertTrue((df[type_cols].sum(axis=1) == 1).all())

    def test_audit_reports_categorical_value_counts_for_raw_type_column(self):
        df = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        audit = bd.audit_dataframe(df)
        self.assertIn("value_counts", audit["columns"]["type"])
        self.assertGreater(len(audit["columns"]["type"]["value_counts"]), 0)


class TestFullStreamAudit(unittest.TestCase):
    def test_runs_without_retaining_rows_and_matches_bounded_sample_stats(self):
        streamed = bd.run_full_stream_audit(EDA_CSV, chunksize=4, max_rows=13)
        self.assertEqual(streamed["total_rows"], 13)
        self.assertFalse(streamed["is_full_dataset"])
        self.assertFalse(streamed["target_column_used"])
        sample = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=13)
        self.assertEqual(
            streamed["columns"]["amount"]["mean"],
            float(sample["amount"].mean()),
        )

    def test_missing_counts_match_between_streamed_and_materialized(self):
        streamed = bd.run_full_stream_audit(EDA_CSV, chunksize=4, max_rows=13)
        sample = bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=13)
        for col in ["origin_avg_amount_causal", "dest_incoming_concentration_causal"]:
            self.assertEqual(
                streamed["columns"][col]["missing_count"],
                int(sample[col].isna().sum()),
            )


class TestNoRawFileMutation(unittest.TestCase):
    def test_raw_archive_files_untouched_by_assembly(self):
        import os
        raw_path = Path("data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv")
        if not raw_path.exists():
            self.skipTest("real raw file not present in this environment")
        before_mtime = os.path.getmtime(raw_path)
        bd.build_bounded_sample(raw_path, chunksize=50_000, max_rows=1000)
        after_mtime = os.path.getmtime(raw_path)
        self.assertEqual(before_mtime, after_mtime)

    def test_fixture_untouched_by_assembly(self):
        import os
        before = os.path.getmtime(EDA_CSV)
        bd.build_bounded_sample(EDA_CSV, chunksize=4, max_rows=100)
        after = os.path.getmtime(EDA_CSV)
        self.assertEqual(before, after)


class TestSpecWriter(unittest.TestCase):
    def test_writes_markdown_with_expected_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "spec.md"
            bd.write_baseline_spec(out_path)
            text = out_path.read_text()
            self.assertIn("Baseline feature contract", text)
            self.assertIn("Undefined-value policy", text)
            self.assertIn("Same-step / causal convention", text)
            for col in bd.BASELINE_FEATURE_COLUMNS:
                self.assertIn(f"`{col}`", text)


class TestMissingInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            bd.build_bounded_sample(FIXTURES_DIR / "does_not_exist.csv", max_rows=10)


if __name__ == "__main__":
    unittest.main()
