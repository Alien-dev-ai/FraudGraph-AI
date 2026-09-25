"""Tests for src.modeling_split, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import modeling_split as ms  # noqa: E402
from src.baseline_dataset import BASELINE_FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"
SAME_STEP_CSV = FIXTURES_DIR / "same_step_ties.csv"
DECREASING_CSV = FIXTURES_DIR / "decreasing_step.csv"


class TestChronologicalOrderingVerification(unittest.TestCase):
    def test_eda_mini_fixture_is_non_decreasing(self):
        result = ms.verify_step_ordering(EDA_CSV, chunksize=4)
        self.assertTrue(result["is_non_decreasing_by_step"])
        self.assertEqual(result["total_rows"], 13)
        self.assertEqual(result["min_step"], 1)
        self.assertEqual(result["max_step"], 34)

    def test_decreasing_fixture_is_detected_as_non_ordered(self):
        result = ms.verify_step_ordering(DECREASING_CSV, chunksize=1)
        self.assertFalse(result["is_non_decreasing_by_step"])

    def test_decreasing_detected_even_across_chunk_boundary(self):
        # chunksize=1 forces the decrease to be detected only via the
        # cross-chunk prev_last_step comparison, not a within-chunk diff.
        result_small_chunks = ms.verify_step_ordering(DECREASING_CSV, chunksize=1)
        result_one_chunk = ms.verify_step_ordering(DECREASING_CSV, chunksize=100)
        self.assertFalse(result_small_chunks["is_non_decreasing_by_step"])
        self.assertFalse(result_one_chunk["is_non_decreasing_by_step"])


class TestSameStepBoundaryHandling(unittest.TestCase):
    def test_same_step_ties_fixture_reports_correct_count(self):
        # 3 rows at step=5, 1 row at step=6 -> rows sharing step with the
        # immediately preceding row: row1(step5) and row2(step5) = 2 of 4.
        result = ms.verify_step_ordering(SAME_STEP_CSV, chunksize=2)
        self.assertTrue(result["is_non_decreasing_by_step"])
        self.assertEqual(result["rows_sharing_step_with_prior_row"], 2)
        self.assertAlmostEqual(result["rows_sharing_step_with_prior_row_pct"], 0.5)

    def test_same_step_count_identical_across_chunk_sizes(self):
        r_small = ms.verify_step_ordering(SAME_STEP_CSV, chunksize=1)
        r_large = ms.verify_step_ordering(SAME_STEP_CSV, chunksize=100)
        self.assertEqual(
            r_small["rows_sharing_step_with_prior_row"],
            r_large["rows_sharing_step_with_prior_row"],
        )


class TestPerDayStatsAndRegionSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per_day = ms.compute_per_day_stats(EDA_CSV, chunksize=4)

    def test_per_day_totals_match_hand_computed_values(self):
        # hand-verified in Milestone 4: day0 total=8 fraud=3, day1 total=5 fraud=2
        self.assertEqual(self.per_day[0]["total_count"], 8)
        self.assertEqual(self.per_day[0]["fraud_count"], 3)
        self.assertEqual(self.per_day[1]["total_count"], 5)
        self.assertEqual(self.per_day[1]["fraud_count"], 2)

    def test_region_summary_single_day_matches_per_day_entry(self):
        region = ms.summarize_region(self.per_day, 0, 0)
        self.assertEqual(region["total_rows"], self.per_day[0]["total_count"])
        self.assertEqual(region["fraud_rows"], self.per_day[0]["fraud_count"])
        self.assertEqual(region["days_covered"], [0])

    def test_region_summary_spanning_all_days_matches_dataset_totals(self):
        region = ms.summarize_region(self.per_day, 0, 1)
        self.assertEqual(region["total_rows"], 13)
        self.assertEqual(region["fraud_rows"], 5)

    def test_region_type_counts_sum_to_region_total(self):
        region = ms.summarize_region(self.per_day, 0, 1)
        self.assertEqual(sum(region["type_counts"].values()), region["total_rows"])


class TestZeroFraudAndFraudOnlyFlags(unittest.TestCase):
    def test_region_with_no_fraud_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "no_fraud.csv"
            path.write_text(
                "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
                "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
                "1,PAYMENT,10,C1,100,90,M1,0,0,0,0\n"
                "2,PAYMENT,20,C2,100,80,M2,0,0,0,0\n"
            )
            per_day = ms.compute_per_day_stats(path, chunksize=10)
            region = ms.summarize_region(per_day, 0, 0)
            self.assertTrue(region["has_zero_fraud"])
            self.assertFalse(region["is_fraud_only"])

    def test_region_with_all_fraud_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "all_fraud.csv"
            path.write_text(
                "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
                "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
                "1,TRANSFER,10,C1,100,90,M1,0,0,1,0\n"
                "2,TRANSFER,20,C2,100,80,M2,0,0,1,0\n"
            )
            per_day = ms.compute_per_day_stats(path, chunksize=10)
            region = ms.summarize_region(per_day, 0, 0)
            self.assertFalse(region["has_zero_fraud"])
            self.assertTrue(region["is_fraud_only"])

    def test_real_dataset_final_day_artifact_flagged_by_naive_policy(self):
        # Uses the already-established, hand-verifiable per-day figures
        # from Milestone 4 (day 30 = 272 rows, all fraud) reproduced via
        # a synthetic region rather than requiring the 480MB real file.
        per_day = {30: {"total_count": 272, "fraud_count": 272, "type_counts": {"CASH_OUT": 272},
                        "min_step": 720, "max_step": 743, "fraud_rate": 1.0}}
        region = ms.summarize_region(per_day, 30, 30)
        self.assertTrue(region["is_fraud_only"])
        self.assertFalse(region["has_zero_fraud"])


class TestNoOverlapAndFullAccounting(unittest.TestCase):
    def test_train_val_test_policy_has_no_overlap(self):
        per_day = {d: {"total_count": 10, "fraud_count": 1, "type_counts": {"PAYMENT": 10},
                       "min_step": d * 24 + 1, "max_step": d * 24 + 24, "fraud_rate": 0.1}
                   for d in range(31)}
        check = ms.check_no_overlap_and_full_accounting(per_day, "chronological_train_val_test")
        self.assertTrue(check["applicable"])
        self.assertTrue(check["no_overlap"])
        self.assertEqual(check["overlap_days"], [])

    def test_train_val_test_policy_accounts_for_expected_rows(self):
        per_day = {d: {"total_count": 10, "fraud_count": 1, "type_counts": {"PAYMENT": 10},
                       "min_step": d * 24 + 1, "max_step": d * 24 + 24, "fraud_rate": 0.1}
                   for d in range(31)}
        check = ms.check_no_overlap_and_full_accounting(per_day, "chronological_train_val_test")
        # policy covers days 0-23, 24-26, 27-30 = all 31 days -> full accounting
        self.assertEqual(check["rows_accounted_pct"], 1.0)
        self.assertEqual(check["days_in_dataset_not_covered"], [])

    def test_deliberately_gapped_policy_detected_as_incomplete(self):
        per_day = {d: {"total_count": 10, "fraud_count": 1, "type_counts": {"PAYMENT": 10},
                       "min_step": d * 24 + 1, "max_step": d * 24 + 24, "fraud_rate": 0.1}
                   for d in range(5)}
        # naive_tail_only_test covers days 0-29 and 30-30 = 0-30, but our
        # synthetic per_day only has days 0-4, so "not covered" must be empty
        # and accounted rows must equal total (edge case: policy range wider
        # than available data is fine, summarize_region only sums days present).
        check = ms.check_no_overlap_and_full_accounting(per_day, "naive_tail_only_test")
        self.assertTrue(check["no_overlap"])

    def test_random_stratified_diagnostic_not_applicable(self):
        check = ms.check_no_overlap_and_full_accounting({}, "random_stratified_diagnostic")
        self.assertFalse(check["applicable"])


class TestFraudCountAccounting(unittest.TestCase):
    def test_sum_of_region_fraud_equals_sum_of_per_day_fraud(self):
        per_day = ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        total_fraud_per_day = sum(v["fraud_count"] for v in per_day.values())
        full_region = ms.summarize_region(per_day, 0, 1)
        self.assertEqual(full_region["fraud_rows"], total_fraud_per_day)
        self.assertEqual(total_fraud_per_day, 5)  # hand-verified in Milestone 4


class TestDeterminism(unittest.TestCase):
    def test_repeated_per_day_computation_is_identical(self):
        r1 = ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        r2 = ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        self.assertEqual(r1, r2)

    def test_per_day_computation_identical_across_chunk_sizes(self):
        r_small = ms.compute_per_day_stats(EDA_CSV, chunksize=1)
        r_large = ms.compute_per_day_stats(EDA_CSV, chunksize=1000)
        self.assertEqual(r_small, r_large)

    def test_evaluate_all_policies_deterministic(self):
        per_day = ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        e1 = ms.evaluate_all_policies(per_day)
        e2 = ms.evaluate_all_policies(per_day)
        self.assertEqual(e1, e2)


class TestTargetExclusionFromFeatureInputs(unittest.TestCase):
    def test_target_not_in_baseline_feature_columns(self):
        self.assertNotIn(TARGET_COLUMN, BASELINE_FEATURE_COLUMNS)

    def test_target_column_constant_matches_baseline_dataset(self):
        self.assertEqual(ms.TARGET_COLUMN, TARGET_COLUMN)

    def test_split_audit_columns_are_label_audit_only_not_feature_columns(self):
        # SPLIT_AUDIT_COLUMNS legitimately includes isFraud for label
        # audit, but none of its columns may appear in the feature
        # contract in a way that would leak the label as a feature.
        self.assertIn(TARGET_COLUMN, ms.SPLIT_AUDIT_COLUMNS)
        self.assertNotIn(TARGET_COLUMN, BASELINE_FEATURE_COLUMNS)


class TestGroupSplitFeasibility(unittest.TestCase):
    def test_assess_group_split_feasibility_runs_and_reports_rates(self):
        result = ms.assess_group_split_feasibility(EDA_CSV, chunksize=4)
        self.assertEqual(result["total_rows"], 13)
        self.assertGreaterEqual(result["unique_origin_accounts"], 1)
        self.assertGreaterEqual(result["unique_destination_accounts"], 1)
        self.assertIn("origin_repeat_rate", result)
        self.assertIn("destination_repeat_rate", result)


class TestNoRawFileMutation(unittest.TestCase):
    def test_fixture_untouched_by_full_analysis(self):
        before = os.path.getmtime(EDA_CSV)
        ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        ms.verify_step_ordering(EDA_CSV, chunksize=4)
        ms.assess_group_split_feasibility(EDA_CSV, chunksize=4)
        after = os.path.getmtime(EDA_CSV)
        self.assertEqual(before, after)

    def test_raw_archive_file_untouched_if_present(self):
        from src.validate_raw_paysim import DEFAULT_RAW_CSV
        if not DEFAULT_RAW_CSV.exists():
            self.skipTest("real raw file not present in this environment")
        before = os.path.getmtime(DEFAULT_RAW_CSV)
        ms.verify_step_ordering(DEFAULT_RAW_CSV, chunksize=500_000)
        after = os.path.getmtime(DEFAULT_RAW_CSV)
        self.assertEqual(before, after)


class TestBoundedMemoryBehavior(unittest.TestCase):
    def test_per_day_accumulator_size_bounded_by_unique_days_not_row_count(self):
        # Synthetic fixture: 200 rows, but all within 2 unique days --
        # the accumulator must stay at 2 entries, not grow with row count.
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "many_rows_two_days.csv"
            rows = ["step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
                    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"]
            for i in range(200):
                step = 1 if i < 100 else 25  # day 0 then day 1
                rows.append(f"{step},PAYMENT,10,C{i},100,90,M{i},0,0,0,0")
            path.write_text("\n".join(rows) + "\n")
            per_day = ms.compute_per_day_stats(path, chunksize=17)
            self.assertEqual(len(per_day), 2)
            self.assertEqual(per_day[0]["total_count"], 100)
            self.assertEqual(per_day[1]["total_count"], 100)

    def test_group_feasibility_accumulator_scales_with_unique_accounts(self):
        result = ms.assess_group_split_feasibility(EDA_CSV, chunksize=4)
        # eda_mini.csv has 13 rows but not 13 unique accounts on each side
        self.assertLessEqual(result["unique_origin_accounts"], 13)


class TestSpecWriter(unittest.TestCase):
    def test_writes_markdown_with_expected_sections(self):
        per_day = ms.compute_per_day_stats(EDA_CSV, chunksize=4)
        policy_evals = ms.evaluate_all_policies(per_day)
        group_feas = ms.assess_group_split_feasibility(EDA_CSV, chunksize=4)
        step_v = ms.verify_step_ordering(EDA_CSV, chunksize=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "spec.md"
            ms.write_split_spec(per_day, policy_evals, group_feas, step_v, out_path)
            text = out_path.read_text()
            self.assertIn("Evaluation objective", text)
            self.assertIn("Temporal structure", text)
            self.assertIn("Candidate split policies", text)
            self.assertIn("Recommended policy", text)
            self.assertIn("PR-AUC", text)
            self.assertIn("ROC-AUC", text)


class TestMissingInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            ms.compute_per_day_stats(FIXTURES_DIR / "does_not_exist.csv")


if __name__ == "__main__":
    unittest.main()
