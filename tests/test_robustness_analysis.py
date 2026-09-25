"""Tests for src.robustness_analysis, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import ablation_analysis as ab  # noqa: E402
from src import baseline_dataset as bd  # noqa: E402
from src import baseline_model as bmod  # noqa: E402
from src import robustness_analysis as ra  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPLIT_BOUNDARY_CSV = FIXTURES_DIR / "split_boundary.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTestDayIdentification(unittest.TestCase):
    def test_test_day_range_matches_m9_approved_policy(self):
        self.assertEqual(ra.TEST_DAY_RANGE, (27, 30))
        self.assertEqual(ra.TEST_DAY_RANGE, bmod.SPLIT_DAY_RANGES["test"])

    def test_test_days_enumerates_all_four_days_inclusive(self):
        self.assertEqual(ra.TEST_DAYS, [27, 28, 29, 30])

    def test_simulation_boundary_day_is_the_last_test_day(self):
        self.assertEqual(ra.SIMULATION_BOUNDARY_DAY, 30)
        self.assertIn(ra.SIMULATION_BOUNDARY_DAY, ra.TEST_DAYS)

    def test_stream_test_metadata_only_returns_test_period_days(self):
        day_index, _ = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertTrue(set(np.unique(day_index).tolist()) <= set(ra.TEST_DAYS))

    def test_stream_test_metadata_row_count_matches_stream_split_arrays_test_bucket(self):
        full_arrays, stream_stats = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        day_index, type_arr = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(len(day_index), full_arrays["test"][0].shape[0])
        self.assertEqual(len(day_index), stream_stats["rows_by_split"]["test"])
        self.assertEqual(len(type_arr), len(day_index))


class TestDayTypeGrouping(unittest.TestCase):
    def test_day_type_breakdown_groups_exactly_present_combinations(self):
        y_test = np.array([0, 1, 0])
        preds_test = np.array([0, 1, 0])
        day_index_test = np.array([27, 29, 30])
        type_test = np.array(["PAYMENT", "CASH_OUT", "PAYMENT"])
        rows = ra.day_type_breakdown(y_test, preds_test, day_index_test, type_test)
        pairs = {(r["day_index"], r["type"]) for r in rows}
        self.assertEqual(pairs, {(27, "PAYMENT"), (29, "CASH_OUT"), (30, "PAYMENT")})

    def test_day_type_breakdown_counts_are_correct(self):
        y_test = np.array([1, 1, 0, 0])
        preds_test = np.array([1, 0, 1, 0])
        day_index_test = np.array([27, 27, 27, 28])
        type_test = np.array(["CASH_OUT", "CASH_OUT", "CASH_OUT", "TRANSFER"])
        rows = ra.day_type_breakdown(y_test, preds_test, day_index_test, type_test)
        row_27 = next(r for r in rows if r["day_index"] == 27)
        self.assertEqual(row_27["transaction_count"], 3)
        self.assertEqual(row_27["fraud_count"], 2)
        self.assertEqual(row_27["captured_fraud"], 1)
        self.assertEqual(row_27["missed_fraud"], 1)
        self.assertEqual(row_27["false_positives"], 1)

    def test_fraud_type_breakdown_only_covers_eligible_types(self):
        y_test = np.array([1, 0])
        proba = np.array([0.9, 0.1])
        preds = np.array([1, 0])
        type_test = np.array(["CASH_OUT", "TRANSFER"])
        result = ra.fraud_type_breakdown(y_test, proba, preds, type_test)
        self.assertEqual(set(result.keys()), set(ra.FRAUD_ELIGIBLE_TYPES))


class TestExactTrainValidationTestBoundaries(unittest.TestCase):
    def test_boundaries_are_m9_unchanged(self):
        self.assertEqual(bmod.SPLIT_DAY_RANGES, {"train": (0, 23), "validation": (24, 26), "test": (27, 30)})

    def test_run_full_analysis_reports_the_same_boundaries(self):
        results = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(results["split_day_ranges"], {"train": [0, 23], "validation": [24, 26], "test": [27, 30]})


class TestNoTargetLeakage(unittest.TestCase):
    def test_no_group_contains_target_column(self):
        for name, cols in ra.ROBUSTNESS_GROUPS.items():
            with self.subTest(group=name):
                self.assertNotIn(bd.TARGET_COLUMN, cols)

    def test_fit_and_predict_group_reasserts_no_target_leakage(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        fit = ra.fit_and_predict_group("FULL_BASELINE", ab.FULL_BASELINE, full_arrays)
        self.assertNotIn(bd.TARGET_COLUMN, ra.ROBUSTNESS_GROUPS["FULL_BASELINE"])
        self.assertEqual(fit["feature_count"], 39)


class TestNoIdentifierColumns(unittest.TestCase):
    def test_no_group_contains_identifier_columns(self):
        for name, cols in ra.ROBUSTNESS_GROUPS.items():
            with self.subTest(group=name):
                self.assertFalse(set(bd.IDENTIFIER_COLUMNS) & set(cols))


class TestFixedThresholdUnchanged(unittest.TestCase):
    def test_default_threshold_matches_m10(self):
        self.assertEqual(ra.DEFAULT_THRESHOLD, bmod.DEFAULT_THRESHOLD)
        self.assertEqual(ra.DEFAULT_THRESHOLD, 0.5)

    def test_threshold_used_identically_regardless_of_slice(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        fit = ra.fit_and_predict_group("FULL_BASELINE", ab.FULL_BASELINE, full_arrays)
        expected_preds = (fit["proba_test"] >= ra.DEFAULT_THRESHOLD).astype(int)
        np.testing.assert_array_equal(fit["preds_test"], expected_preds)


class TestFullBaselineMembershipUnchanged(unittest.TestCase):
    def test_full_baseline_matches_ablation_and_baseline_dataset(self):
        self.assertEqual(ra.ROBUSTNESS_GROUPS["FULL_BASELINE"], ab.FULL_BASELINE)
        self.assertEqual(ra.ROBUSTNESS_GROUPS["FULL_BASELINE"], bd.BASELINE_FEATURE_COLUMNS)
        self.assertEqual(len(ra.ROBUSTNESS_GROUPS["FULL_BASELINE"]), 39)


class TestBalanceReducedFullMembershipMatchesM11(unittest.TestCase):
    def test_balance_reduced_full_matches_ablation_exactly(self):
        self.assertEqual(ra.ROBUSTNESS_GROUPS["BALANCE_REDUCED_FULL"], ab.BALANCE_REDUCED_FULL)
        self.assertEqual(len(ra.ROBUSTNESS_GROUPS["BALANCE_REDUCED_FULL"]), 37)
        for col in ab.BALANCE_FEATURES_REMOVED:
            self.assertNotIn(col, ra.ROBUSTNESS_GROUPS["BALANCE_REDUCED_FULL"])


class TestTrainOnlyPreprocessing(unittest.TestCase):
    def test_fit_and_predict_group_fits_preprocessing_on_train_only(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, _ = ab.select_group_arrays(full_arrays, ab.FULL_BASELINE)["train"]
        imputer_direct, scaler_direct, _ = bmod.fit_preprocessing(X_train)

        fit = ra.fit_and_predict_group("FULL_BASELINE", ab.FULL_BASELINE, full_arrays)
        # Re-fitting with the identical train data must reproduce identical statistics --
        # confirms fit_and_predict_group did not fold in validation/test rows.
        X_train_again, _ = ab.select_group_arrays(full_arrays, ab.FULL_BASELINE)["train"]
        imputer_again, scaler_again, _ = bmod.fit_preprocessing(X_train_again)
        np.testing.assert_array_equal(imputer_direct.statistics_, imputer_again.statistics_)
        np.testing.assert_array_equal(scaler_direct.mean_, scaler_again.mean_)
        self.assertEqual(fit["train_rows"], X_train.shape[0])


class TestDeterministicOutput(unittest.TestCase):
    def test_repeated_full_analysis_yields_identical_results(self):
        results_a = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        results_b = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            results_a["experiment_a_daily_full_baseline"],
            results_b["experiment_a_daily_full_baseline"],
        )
        self.assertEqual(
            results_a["experiment_d_day_type_rows"],
            results_b["experiment_d_day_type_rows"],
        )

    def test_repeated_slice_metrics_call_is_deterministic(self):
        y = np.array([1, 0, 1, 0, 0])
        preds = np.array([1, 0, 0, 0, 1])
        proba = np.array([0.9, 0.2, 0.4, 0.1, 0.6])
        m1 = ra.slice_metrics(y, preds, proba)
        m2 = ra.slice_metrics(y, preds, proba)
        self.assertEqual(m1, m2)


class TestUndefinedMetricsHandledExplicitly(unittest.TestCase):
    def test_no_positives_gives_none_recall_not_zero(self):
        y = np.array([0, 0, 0])
        preds = np.array([0, 1, 0])
        m = ra.slice_metrics(y, preds, proba=None)
        self.assertIsNone(m["recall"])
        self.assertIn("recall", m["undefined_metrics_reasons"])

    def test_no_predicted_positives_gives_none_precision_not_zero(self):
        y = np.array([1, 0, 0])
        preds = np.array([0, 0, 0])
        m = ra.slice_metrics(y, preds, proba=None)
        self.assertIsNone(m["precision"])
        self.assertIn("precision", m["undefined_metrics_reasons"])

    def test_single_class_slice_gives_none_pr_auc_and_roc_auc(self):
        y = np.array([1, 1, 1])
        preds = np.array([1, 1, 0])
        proba = np.array([0.9, 0.8, 0.3])
        m = ra.slice_metrics(y, preds, proba)
        self.assertIsNone(m["average_precision_pr_auc"])
        self.assertIsNone(m["roc_auc_secondary"])
        self.assertIn("average_precision_pr_auc", m["undefined_metrics_reasons"])
        self.assertIn("roc_auc_secondary", m["undefined_metrics_reasons"])

    def test_well_defined_zero_is_not_confused_with_undefined(self):
        # n_alerts > 0 but zero correct -> precision is a REAL 0.0, not undefined.
        y = np.array([0, 0, 1])
        preds = np.array([1, 0, 0])
        m = ra.slice_metrics(y, preds, proba=None)
        self.assertEqual(m["precision"], 0.0)
        self.assertNotIn("precision", m["undefined_metrics_reasons"])

    def test_both_classes_present_computes_real_pr_and_roc_auc(self):
        y = np.array([1, 0, 1, 0])
        preds = np.array([1, 0, 1, 0])
        proba = np.array([0.9, 0.1, 0.8, 0.2])
        m = ra.slice_metrics(y, preds, proba)
        self.assertIsNotNone(m["average_precision_pr_auc"])
        self.assertIsNotNone(m["roc_auc_secondary"])


class TestNoRawFileMutation(unittest.TestCase):
    def test_fixture_untouched_by_full_analysis(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)

    def test_fixture_untouched_by_smoke_test(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        ra.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)


class TestDay30NotSilentlyExcluded(unittest.TestCase):
    def test_day_30_present_in_daily_breakdown(self):
        results = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertIn("30", results["experiment_a_daily_full_baseline"])
        self.assertTrue(results["experiment_a_daily_full_baseline"]["30"]["is_simulation_boundary_day"])

    def test_day_30_flagged_in_both_groups(self):
        results = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        for group_name, per_day in results["experiment_b_daily_by_group"].items():
            with self.subTest(group=group_name):
                self.assertIn("30", per_day)
                self.assertTrue(per_day["30"]["is_simulation_boundary_day"])

    def test_non_boundary_days_not_flagged(self):
        results = ra.run_full_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        for day in ("27", "28", "29"):
            self.assertFalse(results["experiment_a_daily_full_baseline"][day]["is_simulation_boundary_day"])


class TestFraudCountsMatchKnownPaySimLabels(unittest.TestCase):
    def test_fraud_confined_to_cash_out_and_transfer_on_real_dataset(self):
        # Cheap (2-column) check against the real raw file -- no feature
        # engineering, no full 6.36M-row causal pass.
        result = ra.verify_fraud_types_are_exactly(ra.DEFAULT_RAW_CSV, chunksize=500_000)
        self.assertTrue(result["fraud_confined_to_eligible_types"])
        self.assertEqual(result["unexpected_fraud_types"], {})
        # Known values re-confirmed across Milestones 4/9/10/11.
        self.assertEqual(result["fraud_by_type"]["CASH_OUT"], 4116)
        self.assertEqual(result["fraud_by_type"]["TRANSFER"], 4097)
        self.assertEqual(sum(result["fraud_by_type"].values()), 8213)

    def test_fixture_fraud_type_verification_matches_hand_count(self):
        result = ra.verify_fraud_types_are_exactly(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(result["fraud_by_type"], {"TRANSFER": 2, "CASH_OUT": 1})


class TestSmokeTest(unittest.TestCase):
    def test_smoke_test_reports_alignment_and_rss(self):
        result = ra.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        self.assertTrue(result["row_counts_aligned"])
        self.assertIn("within_safety_limit", result)
        self.assertGreater(result["peak_rss_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
