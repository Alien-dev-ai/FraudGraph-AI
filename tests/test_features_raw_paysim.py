"""Tests for src.features_raw_paysim, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import features_raw_paysim as feat  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"  # already in step-sorted (causal) order
MINI_PAYSIM_CSV = FIXTURES_DIR / "mini_paysim.csv"
QUALITY_EDGE_CSV = FIXTURES_DIR / "quality_edge_cases.csv"
SAME_STEP_CSV = FIXTURES_DIR / "same_step_ties.csv"

SMALL_CHUNKSIZE = 4  # forces 4 chunks over eda_mini.csv's 13 rows


def _by_row_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("row_id")


class TestNoLabelLeakage(unittest.TestCase):
    def test_isfraud_not_in_input_columns(self):
        self.assertNotIn("isFraud", feat.INPUT_COLUMNS)

    def test_isfraud_not_in_feature_metadata(self):
        self.assertNotIn("isFraud", feat.FEATURE_METADATA)
        self.assertNotIn("isFraud", feat.all_feature_columns())
        self.assertNotIn("isFraud", feat.baseline_feature_columns())

    def test_isfraud_not_in_engineered_output(self):
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        self.assertNotIn("isFraud", df.columns)

    def test_source_fixture_has_isfraud_but_output_does_not(self):
        # eda_mini.csv DOES have an isFraud column in the raw file; this
        # confirms exclusion is active, not just coincidentally absent.
        raw = pd.read_csv(EDA_CSV)
        self.assertIn("isFraud", raw.columns)
        engineered = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        self.assertNotIn("isFraud", engineered.columns)


class TestRowAlignmentAndSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(EDA_CSV)
        cls.df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_row_count_matches_input(self):
        self.assertEqual(len(self.df), len(self.raw))

    def test_all_declared_columns_present(self):
        for col in feat.all_feature_columns():
            self.assertIn(col, self.df.columns, msg=f"missing column {col}")

    def test_row_id_is_sequential(self):
        self.assertEqual(list(self.df["row_id"]), list(range(len(self.df))))

    def test_amount_and_step_align_with_raw_in_file_order(self):
        self.assertEqual(list(self.df["amount"]), list(self.raw["amount"]))
        self.assertEqual(list(self.df["step"]), list(self.raw["step"]))
        self.assertEqual(list(self.df["nameOrig"]), list(self.raw["nameOrig"]))

    def test_one_hot_type_columns_sum_to_one_per_row(self):
        type_cols = [f"type_{t}" for t in feat.SORTED_TYPES]
        row_sums = self.df[type_cols].sum(axis=1)
        self.assertTrue((row_sums == 1).all())


class TestDeterminism(unittest.TestCase):
    def test_repeated_runs_produce_identical_output(self):
        df1 = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        df2 = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        pd.testing.assert_frame_equal(df1, df2)

    def test_output_identical_regardless_of_chunksize(self):
        df_small_chunks = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        df_one_chunk = feat.build_features(EDA_CSV, chunksize=1000)
        pd.testing.assert_frame_equal(df_small_chunks, df_one_chunk)


class TestCausalIdentityFeaturesAcrossChunks(unittest.TestCase):
    """eda_mini.csv has exactly two repeat origin accounts (C1, C2) and two
    repeat destination accounts (M1, C10); with chunksize=4 each repeat
    pair straddles a chunk boundary, directly exercising cross-chunk
    running-state carry-over.
    """

    @classmethod
    def setUpClass(cls):
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        cls.by_name_orig = df.set_index("nameOrig", drop=False)
        cls.by_name_dest_and_row = df

    def test_origin_c1_second_occurrence_sees_first_as_prior(self):
        # C1: row0 amount=100 (step1), row7 amount=80 (step8) -- different chunks
        rows = self.by_name_orig.loc[["C1"]]
        self.assertEqual(len(rows), 2)
        first, second = rows.iloc[0], rows.iloc[1]
        self.assertEqual(first["origin_is_first_seen"], 1)
        self.assertEqual(first["origin_txn_count_causal"], 0)
        self.assertEqual(second["origin_is_first_seen"], 0)
        self.assertEqual(second["origin_txn_count_causal"], 1)
        self.assertAlmostEqual(second["origin_total_amount_causal"], 100.0)
        self.assertAlmostEqual(second["origin_avg_amount_causal"], 100.0)
        self.assertAlmostEqual(second["amount_vs_origin_avg_ratio"], 80.0 / 100.0)

    def test_origin_c2_second_occurrence_crosses_two_chunk_boundaries(self):
        # C2: row1 amount=200 (step2, chunk 0), row10 amount=500 (step32, chunk 2)
        rows = self.by_name_orig.loc[["C2"]]
        first, second = rows.iloc[0], rows.iloc[1]
        self.assertEqual(first["origin_txn_count_causal"], 0)
        self.assertEqual(second["origin_txn_count_causal"], 1)
        self.assertAlmostEqual(second["origin_total_amount_causal"], 200.0)
        self.assertAlmostEqual(second["amount_vs_origin_avg_ratio"], 500.0 / 200.0)

    def test_dest_m1_second_occurrence_sees_first_as_prior(self):
        df = self.by_name_dest_and_row
        rows = df[df["nameDest"] == "M1"]
        self.assertEqual(len(rows), 2)
        first, second = rows.iloc[0], rows.iloc[1]
        self.assertEqual(first["dest_is_first_seen"], 1)
        self.assertEqual(second["dest_is_first_seen"], 0)
        self.assertEqual(second["dest_txn_count_causal"], 1)
        self.assertAlmostEqual(second["dest_total_amount_causal"], 100.0)
        self.assertAlmostEqual(second["amount_vs_dest_avg_ratio"], 120.0 / 100.0)

    def test_dest_c10_second_occurrence_sees_first_as_prior(self):
        df = self.by_name_dest_and_row
        rows = df[df["nameDest"] == "C10"]
        first, second = rows.iloc[0], rows.iloc[1]
        self.assertEqual(second["dest_txn_count_causal"], 1)
        self.assertAlmostEqual(second["dest_total_amount_causal"], 200.0)
        self.assertAlmostEqual(second["amount_vs_dest_avg_ratio"], 250.0 / 200.0)

    def test_unique_accounts_are_always_first_seen(self):
        df = self.by_name_dest_and_row
        c3_row = df[df["nameOrig"] == "C3"].iloc[0]
        self.assertEqual(c3_row["origin_is_first_seen"], 1)
        self.assertEqual(c3_row["origin_txn_count_causal"], 0)
        self.assertTrue(math.isnan(c3_row["origin_avg_amount_causal"]))
        self.assertTrue(math.isnan(c3_row["amount_vs_origin_avg_ratio"]))


class TestSameStepCausalOrdering(unittest.TestCase):
    """Milestone 6 audit: proves that when two rows share the same `step`
    value, the row appearing LATER in the file can never influence the
    causal features of the row appearing EARLIER in the file — causality
    follows file/row order, not step value alone (PaySim has no finer
    timestamp than step; see feature_specification.md).

    Fixture (tests/fixtures/same_step_ties.csv), all at step=5 except
    the last row:
      row0: C1 -> M1, amount=100  (first  C1 row, first  M1 row)
      row1: C1 -> M2, amount=50   (second C1 row, same step as row0)
      row2: C2 -> M1, amount=200  (second M1 row, same step as row0)
      row3: C4 -> M3, amount=10   (step=6, unrelated control row)
    """

    @classmethod
    def setUpClass(cls):
        cls.df = feat.build_features(SAME_STEP_CSV, chunksize=1)  # 4 chunks, 1 row each
        cls.df_single_chunk = feat.build_features(SAME_STEP_CSV, chunksize=10)

    def test_input_fixture_has_same_step_ties(self):
        raw = pd.read_csv(SAME_STEP_CSV)
        self.assertEqual(raw["step"].iloc[0], raw["step"].iloc[1])
        self.assertEqual(raw["step"].iloc[0], raw["step"].iloc[2])

    def test_earlier_row_unaffected_by_later_same_step_row_origin(self):
        # row0 (C1, first in file) must NOT see row1 (C1, later, same step).
        row0 = self.df.iloc[0]
        self.assertEqual(row0["nameOrig"], "C1")
        self.assertEqual(row0["origin_is_first_seen"], 1)
        self.assertEqual(row0["origin_txn_count_causal"], 0)
        self.assertEqual(row0["origin_total_amount_causal"], 0.0)
        self.assertTrue(math.isnan(row0["origin_avg_amount_causal"]))

    def test_later_row_correctly_sees_earlier_same_step_row_origin(self):
        # row1 (C1, second in file, same step as row0) DOES see row0 as prior.
        row1 = self.df.iloc[1]
        self.assertEqual(row1["nameOrig"], "C1")
        self.assertEqual(row1["origin_is_first_seen"], 0)
        self.assertEqual(row1["origin_txn_count_causal"], 1)
        self.assertAlmostEqual(row1["origin_total_amount_causal"], 100.0)
        self.assertAlmostEqual(row1["amount_vs_origin_avg_ratio"], 50.0 / 100.0)

    def test_earlier_row_unaffected_by_later_same_step_row_dest(self):
        # row0 (dest M1, first in file) must NOT see row2 (dest M1, later, same step).
        row0 = self.df.iloc[0]
        self.assertEqual(row0["nameDest"], "M1")
        self.assertEqual(row0["dest_is_first_seen"], 1)
        self.assertEqual(row0["dest_txn_count_causal"], 0)
        self.assertEqual(row0["dest_total_amount_causal"], 0.0)

    def test_later_row_correctly_sees_earlier_same_step_row_dest(self):
        # row2 (dest M1, third in file, same step as row0) DOES see row0 as prior.
        row2 = self.df.iloc[2]
        self.assertEqual(row2["nameDest"], "M1")
        self.assertEqual(row2["dest_is_first_seen"], 0)
        self.assertEqual(row2["dest_txn_count_causal"], 1)
        self.assertAlmostEqual(row2["dest_total_amount_causal"], 100.0)

    def test_result_identical_regardless_of_chunk_boundaries(self):
        # chunksize=1 (every same-step tie crosses a chunk boundary) must
        # give byte-identical causal values to chunksize=10 (single chunk,
        # ties resolved within one chunk) -- proves chunking never changes
        # the causal ordering, only where the state happens to be held.
        pd.testing.assert_frame_equal(self.df, self.df_single_chunk)


class TestSafeZeroHandling(unittest.TestCase):
    def test_zero_orig_balance_gives_nan_ratio_not_error(self):
        # row 13 in eda_mini.csv: C19 has oldbalanceOrg = 0
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        row = df[df["nameOrig"] == "C19"].iloc[0]
        self.assertEqual(row["orig_balance_before"], 0)
        self.assertTrue(math.isnan(row["amount_to_orig_balance_ratio"]))
        self.assertEqual(row["orig_balance_before_zero"], 1)

    def test_zero_dest_balance_gives_nan_ratio_not_error(self):
        # row 13 in eda_mini.csv: C20 (dest) has oldbalanceDest = 0
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        row = df[df["nameDest"] == "C20"].iloc[0]
        self.assertEqual(row["dest_balance_before"], 0)
        self.assertTrue(math.isnan(row["amount_to_dest_balance_ratio"]))
        self.assertEqual(row["dest_balance_before_zero"], 1)

    def test_zero_amount_produces_zero_log_amount_not_error(self):
        chunk = pd.DataFrame({
            "step": [1],
            "type": ["PAYMENT"],
            "amount": [0.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [0.0],
            "newbalanceOrig": [0.0],
            "nameDest": ["M1"],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
            "isFlaggedFraud": [0],
        })
        out = feat.compute_stateless_features(chunk)
        self.assertEqual(out.loc[0, "log_amount"], 0.0)
        self.assertFalse(np.isinf(out.loc[0, "log_amount"]))
        self.assertTrue(math.isnan(out.loc[0, "amount_to_orig_balance_ratio"]))
        self.assertTrue(math.isnan(out.loc[0, "amount_to_dest_balance_ratio"]))

    def test_merchant_destination_mismatch_is_not_applicable_zero(self):
        # row 0 in eda_mini.csv: dest M1, oldbalanceDest=newbalanceDest=0,
        # amount=100 -- naive formula would flag this, but merchant dest
        # mismatch is defined as 0 (not applicable) by design.
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        row = df.iloc[0]
        self.assertEqual(row["is_dest_merchant"], 1)
        self.assertEqual(row["dest_balance_mismatch"], 0)


class TestBalanceMismatchDirectionAware(unittest.TestCase):
    def test_cash_in_uses_addition_not_subtraction(self):
        # row 3 in eda_mini.csv (0-indexed): CASH_IN, C4, old=200, amount=150,
        # new=350 = 200 + 150 -> consistent under CASH_IN's addition rule.
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        row = df[df["nameOrig"] == "C4"].iloc[0]
        self.assertEqual(row["type"], "CASH_IN")
        self.assertEqual(row["orig_balance_mismatch"], 0)


class TestDayAndHourFeatures(unittest.TestCase):
    def test_day_index_and_hour_of_day_formula(self):
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        row_step1 = df[df["step"] == 1].iloc[0]
        row_step30 = df[df["step"] == 30].iloc[0]
        self.assertEqual(row_step1["day_index"], 0)
        self.assertEqual(row_step1["hour_of_day"], 0)
        self.assertEqual(row_step30["day_index"], 1)
        self.assertEqual(row_step30["hour_of_day"], 5)


class TestFeatureMetadataConsistency(unittest.TestCase):
    def test_every_output_column_has_metadata(self):
        df = feat.build_features(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        for col in df.columns:
            self.assertIn(col, feat.FEATURE_METADATA, msg=f"undocumented column {col}")

    def test_leakage_status_values_are_valid(self):
        valid = {"safe_for_baseline", "exploratory_only", "identifier"}
        for col, meta in feat.FEATURE_METADATA.items():
            self.assertIn(meta["leakage_status"], valid, msg=col)

    def test_step_and_day_index_are_exploratory_only(self):
        self.assertEqual(feat.FEATURE_METADATA["step"]["leakage_status"], "exploratory_only")
        self.assertEqual(feat.FEATURE_METADATA["day_index"]["leakage_status"], "exploratory_only")

    def test_hour_of_day_is_safe_for_baseline(self):
        self.assertEqual(feat.FEATURE_METADATA["hour_of_day"]["leakage_status"], "safe_for_baseline")

    def test_baseline_columns_exclude_step_and_day_index(self):
        baseline = feat.baseline_feature_columns()
        self.assertNotIn("step", baseline)
        self.assertNotIn("day_index", baseline)
        self.assertIn("hour_of_day", baseline)

    def test_identifiers_excluded_from_baseline(self):
        baseline = feat.baseline_feature_columns()
        self.assertNotIn("nameOrig", baseline)
        self.assertNotIn("nameDest", baseline)
        self.assertNotIn("row_id", baseline)


class TestValidateAgainstFullDataset(unittest.TestCase):
    """Exercises validate_against_full_dataset on fixtures (stands in for
    the real dataset in tests; the real file is validated separately by
    running the module's CLI, not by unit tests)."""

    def test_runs_cleanly_and_reports_no_isfraud(self):
        result = feat.validate_against_full_dataset(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        self.assertEqual(result["total_rows"], 13)
        self.assertFalse(result["isFraud_present"])
        self.assertEqual(result["column_count"], len(feat.all_feature_columns()))

    def test_reuses_mini_paysim_fixture_too(self):
        result = feat.validate_against_full_dataset(MINI_PAYSIM_CSV, chunksize=3)
        self.assertEqual(result["total_rows"], 10)

    def test_reuses_quality_edge_cases_fixture(self):
        result = feat.validate_against_full_dataset(QUALITY_EDGE_CSV, chunksize=2)
        self.assertEqual(result["total_rows"], 6)


class TestMissingAndMalformedInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            list(feat.iter_engineered_chunks(FIXTURES_DIR / "does_not_exist.csv"))


class TestFeatureSpecWriter(unittest.TestCase):
    def test_writes_markdown_with_all_columns_documented(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "spec.md"
            feat.write_feature_spec(out_path)
            text = out_path.read_text()
            for col in feat.all_feature_columns():
                self.assertIn(f"`{col}`", text)
            self.assertIn("safe_for_baseline", text)
            self.assertIn("exploratory_only", text)


if __name__ == "__main__":
    unittest.main()
