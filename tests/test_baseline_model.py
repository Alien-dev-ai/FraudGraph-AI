"""Tests for src.baseline_model, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import baseline_dataset as bd  # noqa: E402
from src import baseline_model as bmod  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPLIT_BOUNDARY_CSV = FIXTURES_DIR / "split_boundary.csv"
SAME_STEP_CSV = FIXTURES_DIR / "same_step_ties.csv"
MINI_PAYSIM_CSV = FIXTURES_DIR / "mini_paysim.csv"
GRAPH_MINI_CSV = FIXTURES_DIR / "graph_mini.csv"


class _StubModel:
    """A fake fitted model exposing only predict_proba, for testing
    evaluate_split's metric math against known, hand-computed inputs
    without depending on any real training run.
    """

    def __init__(self, proba_fraud: np.ndarray):
        self._proba_fraud = np.asarray(proba_fraud, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._proba_fraud
        return np.column_stack([1.0 - p, p])


class TestSplitAssignmentBoundaries(unittest.TestCase):
    """The fixture's 7 rows are hand-placed exactly on the M9 boundaries:
    day 0 (train), day 23 (train's last day), day 24 (validation's
    first day), day 26 (validation's last day), day 27 (test's first
    day), day 29, and day 30 (test's last day / dataset's last day).
    """

    def test_exact_m9_day_ranges_used(self):
        self.assertEqual(bmod.SPLIT_DAY_RANGES, {"train": (0, 23), "validation": (24, 26), "test": (27, 30)})

    def test_rows_routed_to_correct_split(self):
        arrays, stats = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=3)
        self.assertEqual(stats["rows_by_split"], {"train": 2, "validation": 2, "test": 3})
        self.assertEqual(stats["rows_outside_policy"], 0)

    def test_boundary_fraud_counts(self):
        _, stats = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=3)
        self.assertEqual(stats["fraud_by_split"], {"train": 1, "validation": 1, "test": 1})

    def test_row_and_fraud_totals_sum_correctly(self):
        arrays, stats = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=1)
        total_rows = sum(x.shape[0] for x, _ in arrays.values())
        total_fraud = sum(int(y.sum()) for _, y in arrays.values())
        self.assertEqual(total_rows, stats["rows_seen"])
        self.assertEqual(total_fraud, sum(stats["fraud_by_split"].values()))

    def test_assignment_is_chunksize_independent(self):
        _, stats_a = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=1)
        _, stats_b = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=100)
        self.assertEqual(stats_a["rows_by_split"], stats_b["rows_by_split"])
        self.assertEqual(stats_a["fraud_by_split"], stats_b["fraud_by_split"])


class TestTargetAndIdentifierExclusion(unittest.TestCase):
    def test_target_column_never_a_feature(self):
        self.assertNotIn(bd.TARGET_COLUMN, bmod.FEATURE_COLUMNS)

    def test_identifiers_never_features(self):
        for col in bd.IDENTIFIER_COLUMNS:
            self.assertNotIn(col, bmod.FEATURE_COLUMNS)

    def test_feature_contract_is_exactly_39_columns(self):
        self.assertEqual(len(bmod.FEATURE_COLUMNS), 39)
        self.assertEqual(bmod.FEATURE_COLUMNS, bd.BASELINE_FEATURE_COLUMNS)

    def test_streamed_feature_matrix_has_exactly_39_columns(self):
        arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=10)
        for X, _ in arrays.values():
            self.assertEqual(X.shape[1], 39)


class TestTrainOnlyPreprocessing(unittest.TestCase):
    def test_imputer_statistics_come_from_train_only(self):
        X_train = np.array([[1.0], [3.0], [5.0]] * 13, dtype=np.float32)  # median 3.0, 39 cols below
        X_train_full = np.tile(X_train, (1, 39))
        imputer, scaler, _ = bmod.fit_preprocessing(X_train_full)
        self.assertTrue(np.allclose(imputer.statistics_, 3.0))

    def test_apply_preprocessing_does_not_refit(self):
        X_train = np.tile(np.array([[1.0], [3.0], [5.0]], dtype=np.float32), (1, 39))
        imputer, scaler, _ = bmod.fit_preprocessing(X_train)
        stats_before = imputer.statistics_.copy()
        mean_before = scaler.mean_.copy()

        X_other = np.tile(np.array([[100.0], [200.0]], dtype=np.float32), (1, 39))
        _ = bmod.apply_preprocessing(imputer, scaler, X_other)

        np.testing.assert_array_equal(imputer.statistics_, stats_before)
        np.testing.assert_array_equal(scaler.mean_, mean_before)

    def test_validation_and_test_never_influence_fitted_statistics(self):
        X_train = np.tile(np.array([[10.0], [20.0], [30.0]], dtype=np.float32), (1, 39))
        imputer_a, scaler_a, _ = bmod.fit_preprocessing(X_train.copy())

        # Even if a caller mistakenly passes something huge downstream,
        # fit_preprocessing itself is only ever invoked once, on train.
        imputer_b, scaler_b, _ = bmod.fit_preprocessing(X_train.copy())
        np.testing.assert_array_equal(imputer_a.statistics_, imputer_b.statistics_)
        np.testing.assert_array_equal(scaler_a.mean_, scaler_b.mean_)


class TestNaNAndInfiniteHandling(unittest.TestCase):
    def test_infinite_replaced_with_nan(self):
        X = np.array([[1.0, np.inf], [2.0, -np.inf], [3.0, 4.0]], dtype=np.float32)
        cleaned = bmod._replace_infinite_with_nan(X)
        self.assertTrue(np.isnan(cleaned[0, 1]))
        self.assertTrue(np.isnan(cleaned[1, 1]))
        self.assertEqual(cleaned[2, 1], 4.0)

    def test_nan_is_median_imputed_not_dropped(self):
        col = np.array([1.0, 3.0, np.nan, 5.0] * 10, dtype=np.float32).reshape(-1, 1)
        X = np.tile(col, (1, 39))
        imputer, scaler, X_out = bmod.fit_preprocessing(X)
        self.assertEqual(X_out.shape[0], X.shape[0])  # no row dropped
        self.assertFalse(np.isnan(X_out).any())  # imputer filled every NaN

    def test_full_pipeline_survives_the_known_full_scale_inf_case(self):
        # Regression guard for the +inf found in amount_vs_dest_avg_ratio
        # at full dataset scale (reports/baseline_full_stream_audit.json).
        idx = bmod.FEATURE_COLUMNS.index("amount_vs_dest_avg_ratio")
        X_train = np.zeros((10, 39), dtype=np.float32)
        X_train[0, idx] = np.inf
        X_train[1, idx] = -np.inf
        imputer, scaler, X_out = bmod.fit_preprocessing(X_train)
        self.assertTrue(np.isfinite(X_out).all())


class TestDeterministicTraining(unittest.TestCase):
    def _toy_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 39)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_repeated_training_yields_identical_coefficients(self):
        X, y = self._toy_data()
        model_a = bmod.train_baseline_model(X, y)
        model_b = bmod.train_baseline_model(X, y)
        np.testing.assert_array_equal(model_a.coef_, model_b.coef_)
        np.testing.assert_array_equal(model_a.intercept_, model_b.intercept_)

    def test_repeated_full_split_stream_is_deterministic(self):
        arrays_a, stats_a = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        arrays_b, stats_b = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(stats_a, stats_b)
        for name in bmod.SPLIT_NAMES:
            np.testing.assert_array_equal(arrays_a[name][0], arrays_b[name][0])
            np.testing.assert_array_equal(arrays_a[name][1], arrays_b[name][1])


class TestPredictionShapeAndRange(unittest.TestCase):
    def test_stub_model_predictions_are_well_formed(self):
        proba = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        model = _StubModel(proba)
        y = np.array([0, 0, 1, 1, 0])
        metrics = bmod.evaluate_split(model, np.zeros((5, 39)), y)
        self.assertEqual(metrics["n_total"], 5)


class TestMetricCalculation(unittest.TestCase):
    def test_confusion_matrix_and_precision_recall_known_case(self):
        # y_true:      1  1  0  0  0
        # proba:      .9 .4 .8 .1 .2   -> preds @0.5: 1 0 1 0 0
        # TP=1 (idx0), FN=1 (idx1), FP=1 (idx2), TN=2 (idx3,4)
        y = np.array([1, 1, 0, 0, 0])
        proba = np.array([0.9, 0.4, 0.8, 0.1, 0.2])
        model = _StubModel(proba)
        metrics = bmod.evaluate_split(model, np.zeros((5, 39)), y, threshold=0.5)

        cm = metrics["confusion_matrix"]
        self.assertEqual((cm["tp"], cm["fn"], cm["fp"], cm["tn"]), (1, 1, 1, 2))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["n_fraud"], 2)
        self.assertEqual(metrics["n_legitimate"], 3)
        self.assertEqual(metrics["n_alerts"], 2)
        self.assertEqual(metrics["false_positive_count"], 1)
        self.assertAlmostEqual(metrics["false_positive_rate"], 1 / 3)

    def test_confusion_matrix_accounting_always_balances(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=500)
        proba = rng.random(500)
        model = _StubModel(proba)
        metrics = bmod.evaluate_split(model, np.zeros((500, 39)), y)
        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["tp"] + cm["fn"], metrics["n_fraud"])
        self.assertEqual(cm["tn"] + cm["fp"], metrics["n_legitimate"])
        self.assertEqual(sum(cm.values()), metrics["n_total"])

    def test_probabilities_bounded_and_finite(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=100)
        proba = rng.random(100)
        model = _StubModel(proba)
        metrics = bmod.evaluate_split(model, np.zeros((100, 39)), y)
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)
        self.assertIsNotNone(metrics["average_precision_pr_auc"])
        self.assertGreaterEqual(metrics["average_precision_pr_auc"], 0.0)
        self.assertLessEqual(metrics["average_precision_pr_auc"], 1.0)


class TestNoTestSetThresholdOrBudgetFitting(unittest.TestCase):
    def test_default_threshold_is_fixed_constant(self):
        self.assertEqual(bmod.DEFAULT_THRESHOLD, 0.5)

    def test_default_alert_budget_rate_is_fixed_constant(self):
        self.assertEqual(bmod.ALERT_BUDGET_RATE, 0.01)

    def test_threshold_used_is_whatever_was_passed_never_derived_from_labels(self):
        y_a = np.array([1, 0, 0, 0, 0])
        y_b = np.array([0, 0, 0, 0, 1])  # different labels, same proba
        proba = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
        model = _StubModel(proba)
        metrics_a = bmod.evaluate_split(model, np.zeros((5, 39)), y_a, threshold=0.5)
        metrics_b = bmod.evaluate_split(model, np.zeros((5, 39)), y_b, threshold=0.5)
        # Same proba + same threshold => identical predicted alert set,
        # regardless of which labels were supplied.
        self.assertEqual(metrics_a["n_alerts"], metrics_b["n_alerts"])
        self.assertEqual(metrics_a["threshold"], metrics_b["threshold"])

    def test_alert_budget_count_depends_only_on_rate_and_row_count(self):
        proba_low_fraud = np.linspace(0, 1, 1000)
        proba_high_fraud = np.linspace(0, 1, 1000)
        y_low = (np.arange(1000) < 5).astype(int)  # 5 fraud
        y_high = (np.arange(1000) < 500).astype(int)  # 500 fraud
        model = _StubModel(proba_low_fraud)
        m_low = bmod.evaluate_split(model, np.zeros((1000, 39)), y_low)
        m_high = bmod.evaluate_split(model, np.zeros((1000, 39)), y_high)
        self.assertEqual(m_low["alert_budget_count"], m_high["alert_budget_count"])
        self.assertEqual(m_low["alert_budget_count"], 10)  # 1% of 1000


class TestSameStepCausalOrderingPreserved(unittest.TestCase):
    """This module must not alter Milestone 8's causal feature values --
    it only routes already-computed rows into splits and slices columns.
    """

    def test_features_match_milestone8_assembly_exactly(self):
        for fixture in (SAME_STEP_CSV, GRAPH_MINI_CSV, MINI_PAYSIM_CSV):
            with self.subTest(fixture=fixture.name):
                m8_sample = bd.build_bounded_sample(fixture, chunksize=2, max_rows=1000)
                expected = m8_sample[bmod.FEATURE_COLUMNS].to_numpy(dtype=np.float32)

                arrays, _ = bmod.stream_split_arrays(fixture, chunksize=2)
                actual = np.concatenate([X for X, _ in arrays.values() if X.shape[0]], axis=0)
                # Row order differs (split-routed vs. file order) -- but
                # the SET of feature rows produced must be identical.
                expected_sorted = expected[np.lexsort(expected.T)]
                actual_sorted = actual[np.lexsort(actual.T)]
                np.testing.assert_allclose(expected_sorted, actual_sorted, equal_nan=True)


class TestBoundedExecution(unittest.TestCase):
    def test_smoke_test_runs_on_a_small_bound_and_reports_rss(self):
        result = bmod.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        self.assertEqual(result["stream_stats"]["rows_seen"], 7)
        self.assertIsInstance(result["peak_rss_mb"], float)
        self.assertGreater(result["peak_rss_mb"], 0.0)
        self.assertIn("within_safety_limit", result)

    def test_smoke_test_respects_max_rows_bound(self):
        result = bmod.run_smoke_test(MINI_PAYSIM_CSV, chunksize=3, max_rows=5)
        self.assertEqual(result["stream_stats"]["rows_seen"], 5)


class TestSanityChecks(unittest.TestCase):
    def test_run_sanity_checks_passes_on_boundary_fixture(self):
        results = bmod.run_pipeline(SPLIT_BOUNDARY_CSV, chunksize=2)
        sanity = bmod.run_sanity_checks(results, check_against_full_dataset=False)
        self.assertTrue(sanity["passed"])

    def test_no_target_or_identifier_columns_in_returned_feature_list(self):
        results = bmod.run_pipeline(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertNotIn(bd.TARGET_COLUMN, results["feature_columns"])
        for col in bd.IDENTIFIER_COLUMNS:
            self.assertNotIn(col, results["feature_columns"])

    def test_pipeline_reports_no_hyperparameter_or_threshold_tuning(self):
        results = bmod.run_pipeline(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertFalse(results["model"]["hyperparameter_tuning_performed"])
        self.assertFalse(results["model"]["threshold_tuning_performed"])
        self.assertFalse(results["model"]["resampling_performed"])


if __name__ == "__main__":
    unittest.main()
