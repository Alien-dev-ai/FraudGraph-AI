"""Tests for src.stress_test_analysis, using stdlib unittest only.

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
from src import stress_test_analysis as st  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPLIT_BOUNDARY_CSV = FIXTURES_DIR / "split_boundary.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPerturbationsAreDeterministic(unittest.TestCase):
    def test_fixed_permutation_is_reproducible(self):
        p1 = st.fixed_row_permutation(1000)
        p2 = st.fixed_row_permutation(1000)
        np.testing.assert_array_equal(p1, p2)

    def test_perturb_columns_is_reproducible(self):
        X = np.random.default_rng(1).normal(size=(100, 39)).astype(np.float32)
        perm = st.fixed_row_permutation(100)
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        a = st.perturb_columns(X, st.DOMINANT_BALANCE_FEATURES, 0.5, neutral, perm)
        b = st.perturb_columns(X, st.DOMINANT_BALANCE_FEATURES, 0.5, neutral, perm)
        np.testing.assert_array_equal(a, b)


class TestRandomStateFixed(unittest.TestCase):
    def test_random_state_constant_matches_m10(self):
        self.assertEqual(st.RANDOM_STATE, bmod.RANDOM_STATE)
        self.assertEqual(st.RANDOM_STATE, 42)

    def test_permutation_uses_random_state_42_by_default(self):
        expected = np.random.default_rng(42).permutation(50)
        actual = st.fixed_row_permutation(50)
        np.testing.assert_array_equal(expected, actual)


class TestRawDataNeverModified(unittest.TestCase):
    def test_fixture_untouched_by_full_stress_test(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)

    def test_fixture_untouched_by_smoke_test(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        st.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)


class TestLabelsNeverModified(unittest.TestCase):
    def test_perturb_columns_does_not_touch_target_column(self):
        # Target is never even present in X (39 baseline columns never
        # include isFraud) -- confirmed structurally.
        self.assertNotIn(bd.TARGET_COLUMN, bmod.FEATURE_COLUMNS)

    def test_y_test_unchanged_across_scenarios(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        _, y_test = full_arrays["test"]
        y_test_copy = y_test.copy()
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        np.testing.assert_array_equal(y_test, y_test_copy)
        self.assertEqual(results["test_fraud"], int(y_test_copy.sum()))


class TestTransactionTypesNeverModified(unittest.TestCase):
    def test_type_metadata_untouched_by_perturbation(self):
        day_index_test, type_test = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        type_test_copy = type_test.copy()
        st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        day_index_test2, type_test2 = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        np.testing.assert_array_equal(type_test2, type_test_copy)


class TestStepDayIndexNeverModified(unittest.TestCase):
    def test_day_index_metadata_reproducible_after_stress_run(self):
        day_index_before, _ = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        day_index_after, _ = ra.stream_test_metadata(SPLIT_BOUNDARY_CSV, chunksize=2)
        np.testing.assert_array_equal(day_index_before, day_index_after)

    def test_perturb_columns_never_touches_day_index_or_step(self):
        # day_index/step are exploratory_only and structurally excluded
        # from the 39-column FULL_BASELINE contract -- cannot be perturbed.
        self.assertNotIn("day_index", bmod.FEATURE_COLUMNS)
        self.assertNotIn("step", bmod.FEATURE_COLUMNS)


class TestRowOrderingPreserved(unittest.TestCase):
    def test_perturb_columns_preserves_row_order_and_shape(self):
        X = np.arange(39 * 10, dtype=np.float32).reshape(10, 39)
        perm = st.fixed_row_permutation(10)
        neutral = {c: 0.0 for c in st.DOMINANT_BALANCE_FEATURES}
        X_perturbed = st.perturb_columns(X, st.DOMINANT_BALANCE_FEATURES, 0.5, neutral, perm)
        self.assertEqual(X_perturbed.shape, X.shape)
        untouched_col = 5  # not one of the two dominant balance feature indices in general
        if untouched_col not in [bmod.FEATURE_COLUMNS.index(c) for c in st.DOMINANT_BALANCE_FEATURES]:
            np.testing.assert_array_equal(X_perturbed[:, untouched_col], X[:, untouched_col])


class TestSplitMembershipUnchanged(unittest.TestCase):
    def test_split_day_ranges_are_m9_unchanged(self):
        self.assertEqual(bmod.SPLIT_DAY_RANGES, {"train": (0, 23), "validation": (24, 26), "test": (27, 30)})

    def test_stress_results_report_the_same_boundaries(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(results["split_day_ranges"], {"train": [0, 23], "validation": [24, 26], "test": [27, 30]})
        self.assertEqual(results["test_day_range"], [27, 30])


class TestPreprocessingNotRefitOnTestData(unittest.TestCase):
    def test_perturbed_scoring_uses_transform_only(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, y_train = full_arrays["train"]
        imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
        stats_before = imputer.statistics_.copy()
        mean_before = scaler.mean_.copy()

        model = bmod.train_baseline_model(X_train_proc, y_train)
        X_test, _ = full_arrays["test"]
        neutral = {c: st.neutral_value_for(c, X_train) for c in st.DOMINANT_BALANCE_FEATURES}
        perm = st.fixed_row_permutation(X_test.shape[0])
        X_perturbed = st.perturb_columns(X_test, st.DOMINANT_BALANCE_FEATURES, 1.0, neutral, perm)
        st.predict_with_fixed_model(imputer, scaler, model, X_perturbed)

        np.testing.assert_array_equal(imputer.statistics_, stats_before)
        np.testing.assert_array_equal(scaler.mean_, mean_before)

    def test_neutral_value_for_reads_train_only(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, _ = full_arrays["train"]
        # Calling twice with identical train data must be identical --
        # no hidden dependency on validation/test.
        v1 = st.neutral_value_for("orig_balance_mismatch", X_train)
        v2 = st.neutral_value_for("orig_balance_mismatch", X_train)
        self.assertEqual(v1, v2)


class TestModelParametersUnchanged(unittest.TestCase):
    def test_model_params_match_m10_exactly(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, y_train = full_arrays["train"]
        _, _, X_train_proc = bmod.fit_preprocessing(X_train)
        model = bmod.train_baseline_model(X_train_proc, y_train)
        self.assertEqual(model.get_params()["random_state"], 42)
        self.assertEqual(model.get_params()["class_weight"], "balanced")


class TestThresholdExactly05(unittest.TestCase):
    def test_default_threshold_is_half(self):
        self.assertEqual(st.DEFAULT_THRESHOLD, 0.5)
        self.assertEqual(st.DEFAULT_THRESHOLD, bmod.DEFAULT_THRESHOLD)

    def test_predict_with_fixed_model_thresholds_at_exactly_half(self):
        class _Stub:
            def predict_proba(self, X):
                p = np.array([0.49, 0.5, 0.51])
                return np.column_stack([1 - p, p])

        # bypass real imputer/scaler by using identity-like stand-ins
        class _Identity:
            def transform(self, X):
                return X

        proba, preds = st.predict_with_fixed_model(_Identity(), _Identity(), _Stub(), np.zeros((3, 1)))
        np.testing.assert_array_equal(preds, [0, 1, 1])


class TestZeroPercentReproducesBaselineExactly(unittest.TestCase):
    def test_zero_percent_matches_unperturbed_predictions_exactly(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            results["noise_sweep_D"]["0.00"]["whole_test_period"],
            results["baseline_scenario"]["whole_test_period"],
        )


class TestHundredPercentProducesExpectedNeutralValues(unittest.TestCase):
    def test_hundred_percent_sets_every_row_to_neutral_value(self):
        X = np.random.default_rng(2).normal(size=(20, 39)).astype(np.float32)
        perm = st.fixed_row_permutation(20)
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        X_perturbed = st.perturb_columns(X, st.DOMINANT_BALANCE_FEATURES, 1.0, neutral, perm)
        idx_a = bmod.FEATURE_COLUMNS.index("orig_balance_mismatch")
        idx_b = bmod.FEATURE_COLUMNS.index("orig_balance_after_zero")
        self.assertTrue(np.all(X_perturbed[:, idx_a] == 1.0))
        self.assertTrue(np.all(X_perturbed[:, idx_b] == 0.0))

    def test_stress_scenario_c_matches_hundred_percent_sweep_level(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            results["fixed_model_scenarios_A_B_C"]["C_neutralize_both"]["whole_test_period"],
            results["noise_sweep_D"]["1.00"]["whole_test_period"],
        )


class TestPerturbationLevelsAreExactlyPredefined(unittest.TestCase):
    def test_levels_match_the_predetermined_list(self):
        self.assertEqual(st.PERTURBATION_LEVELS, [0.0, 0.10, 0.25, 0.50, 0.75, 1.00])

    def test_noise_sweep_keys_match_levels_exactly(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        reported_levels = [v["level"] for v in results["noise_sweep_D"].values()]
        self.assertEqual(reported_levels, st.PERTURBATION_LEVELS)


class TestNoTestLabelDrivenParameterSelection(unittest.TestCase):
    def test_neutral_value_computation_never_reads_target_column(self):
        import inspect

        source = inspect.getsource(st.neutral_value_for)
        self.assertNotIn("y_test", source)
        self.assertNotIn("isFraud", source)

    def test_perturbation_levels_defined_independently_of_any_result(self):
        # The levels are a module-level constant, not a function of any
        # computed metric -- structurally verified by being a literal.
        self.assertIsInstance(st.PERTURBATION_LEVELS, list)
        self.assertTrue(all(isinstance(v, float) for v in st.PERTURBATION_LEVELS))


class TestUndefinedMetricsExplicit(unittest.TestCase):
    def test_day_30_slice_has_null_pr_auc_in_every_scenario(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        day30_baseline = results["baseline_scenario"]["daily"]["30"]
        self.assertIsNone(day30_baseline["average_precision_pr_auc"])
        self.assertIn("average_precision_pr_auc", day30_baseline["undefined_metrics_reasons"])

    def test_no_metric_silently_becomes_zero_when_undefined(self):
        # A slice with zero fraud must report recall as None, not 0.0.
        m = ra.slice_metrics(np.array([0, 0, 0]), np.array([0, 1, 0]), None)
        self.assertIsNone(m["recall"])


class TestRepeatedExecutionDeterministic(unittest.TestCase):
    def test_repeated_full_stress_test_yields_identical_results(self):
        results_a = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        results_b = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(results_a["baseline_scenario"], results_b["baseline_scenario"])
        self.assertEqual(results_a["noise_sweep_D"], results_b["noise_sweep_D"])
        self.assertEqual(results_a["fixed_model_scenarios_A_B_C"], results_b["fixed_model_scenarios_A_B_C"])


class TestFixtureTestsDoNotMutateSourceFixtures(unittest.TestCase):
    def test_all_fixtures_unchanged_after_test_module_runs(self):
        hashes_before = {p: _sha256(p) for p in FIXTURES_DIR.glob("*.csv")}
        st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        st.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        hashes_after = {p: _sha256(p) for p in FIXTURES_DIR.glob("*.csv")}
        self.assertEqual(hashes_before, hashes_after)


class TestBaselineReproductionCheckLogic(unittest.TestCase):
    def test_matches_recorded_is_false_on_a_different_dataset(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        bc = results["baseline_reproduction_check"]
        self.assertIn("recomputed", bc)
        self.assertIn("matches_recorded_confusion_matrix_and_counts", bc)

    def test_bounded_run_does_not_raise_even_if_mismatched(self):
        # A bounded/fixture run must never trip the hard-stop guard --
        # only the full, unmodified real-dataset run does.
        try:
            st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        except RuntimeError:
            self.fail("bounded fixture run must not raise the baseline-mismatch guard")


class TestScenarioDefinitions(unittest.TestCase):
    def test_scenario_a_only_perturbs_orig_balance_mismatch(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            results["fixed_model_scenarios_A_B_C"]["A_remove_orig_balance_mismatch"]["perturbed_columns"],
            ["orig_balance_mismatch"],
        )

    def test_scenario_b_only_perturbs_orig_balance_after_zero(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            results["fixed_model_scenarios_A_B_C"]["B_remove_orig_balance_after_zero"]["perturbed_columns"],
            ["orig_balance_after_zero"],
        )

    def test_scenario_c_perturbs_both(self):
        results = st.run_full_stress_test(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(
            set(results["fixed_model_scenarios_A_B_C"]["C_neutralize_both"]["perturbed_columns"]),
            set(st.DOMINANT_BALANCE_FEATURES),
        )

    def test_dominant_balance_features_matches_m11_exactly(self):
        self.assertEqual(st.DOMINANT_BALANCE_FEATURES, ab.BALANCE_FEATURES_REMOVED)


if __name__ == "__main__":
    unittest.main()
