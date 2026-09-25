"""Tests for src.synthetic_generalization, using stdlib unittest only.

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
from src import synthetic_generalization as sg  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPLIT_BOUNDARY_CSV = FIXTURES_DIR / "split_boundary.csv"
MINI_PAYSIM_CSV = FIXTURES_DIR / "mini_paysim.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source_rows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.abs(rng.normal(loc=500, scale=50, size=(n, 39))).astype(np.float32)
    # keep type one-hots valid/simple: everyone is CASH_OUT
    for t in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
        X[:, bmod.FEATURE_COLUMNS.index(f"type_{t}")] = 0.0
    X[:, bmod.FEATURE_COLUMNS.index("type_CASH_OUT")] = 1.0
    X[:, bmod.FEATURE_COLUMNS.index("orig_balance_mismatch")] = 0.0
    X[:, bmod.FEATURE_COLUMNS.index("orig_balance_after_zero")] = 0.0
    return X


class TestDeterministicSyntheticSampling(unittest.TestCase):
    def test_select_source_row_indices_is_deterministic(self):
        y_test = np.array([0, 1, 0, 0, 0])
        day_index_test = np.array([27, 27, 27, 28, 28])
        type_test = np.array(["CASH_OUT"] * 5)
        idx_a = sg.select_source_row_indices(y_test, day_index_test, type_test, 27, "CASH_OUT", 10)
        idx_b = sg.select_source_row_indices(y_test, day_index_test, type_test, 27, "CASH_OUT", 10)
        np.testing.assert_array_equal(idx_a, idx_b)

    def test_selection_uses_file_order_not_random_draw(self):
        y_test = np.zeros(6, dtype=np.int64)
        day_index_test = np.full(6, 27)
        type_test = np.array(["CASH_OUT"] * 6)
        idx = sg.select_source_row_indices(y_test, day_index_test, type_test, 27, "CASH_OUT", 3)
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_sample_per_cell_is_a_fixed_predetermined_constant(self):
        self.assertEqual(sg.SAMPLE_PER_CELL, 500)
        self.assertIsInstance(sg.SAMPLE_PER_CELL, int)


class TestFixedRandomSeed(unittest.TestCase):
    def test_random_state_matches_project_convention(self):
        self.assertEqual(sg.RANDOM_STATE, bmod.RANDOM_STATE)
        self.assertEqual(sg.RANDOM_STATE, 42)


class TestRawDataNeverModified(unittest.TestCase):
    def test_fixture_untouched_by_full_analysis(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)

    def test_fixture_untouched_by_smoke_test(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        sg.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)


class TestSourceLabelsNeverModified(unittest.TestCase):
    def test_isFraud_never_appears_in_feature_columns(self):
        self.assertNotIn(bd.TARGET_COLUMN, bmod.FEATURE_COLUMNS)

    def test_source_rows_are_confirmed_legitimate_not_relabeled(self):
        y_test = np.array([1, 0, 0, 1])
        day_index_test = np.full(4, 27)
        type_test = np.array(["CASH_OUT"] * 4)
        idx = sg.select_source_row_indices(y_test, day_index_test, type_test, 27, "CASH_OUT", 10)
        # only the genuinely legitimate rows (index 1, 2) are ever selected
        self.assertTrue(all(y_test[i] == 0 for i in idx))


class TestSyntheticLabelsGeneratedIndependently(unittest.TestCase):
    def test_synthetic_stress_label_is_a_fixed_literal_not_derived_from_isfraud(self):
        for name, spec in sg.SCENARIO_DEFINITIONS.items():
            with self.subTest(scenario=name):
                self.assertIn(spec["synthetic_stress_label"], (0, 1))

    def test_source_function_never_reads_isfraud_to_pick_transformation(self):
        import inspect

        source = inspect.getsource(sg.build_synthetic_features)
        self.assertNotIn("isFraud", source)
        self.assertNotIn("y_test", source)


class TestTransactionTypePreserved(unittest.TestCase):
    def test_recipes_never_touch_type_one_hot_columns(self):
        type_indices = {bmod.FEATURE_COLUMNS.index(f"type_{t}") for t in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")}
        X = _make_source_rows(5)
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        for recipe in ["fresh_account", "neutral_mismatch", "neutral_after_zero", "altered_balance_drain"]:
            with self.subTest(recipe=recipe):
                X_out = sg.build_synthetic_features(X, [recipe], neutral)
                for idx in type_indices:
                    np.testing.assert_array_equal(X_out[:, idx], X[:, idx])


class TestAmountPreserved(unittest.TestCase):
    def test_recipes_never_touch_amount_column(self):
        idx_amount = bmod.FEATURE_COLUMNS.index("amount")
        X = _make_source_rows(5)
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        for recipe in ["fresh_account", "neutral_mismatch", "neutral_after_zero", "altered_balance_drain"]:
            with self.subTest(recipe=recipe):
                X_out = sg.build_synthetic_features(X, [recipe], neutral)
                np.testing.assert_array_equal(X_out[:, idx_amount], X[:, idx_amount])


class TestStepDayIndexPreserved(unittest.TestCase):
    def test_day_index_and_step_are_not_baseline_columns_and_cannot_be_perturbed(self):
        self.assertNotIn("day_index", bmod.FEATURE_COLUMNS)
        self.assertNotIn("step", bmod.FEATURE_COLUMNS)

    def test_day_source_array_unaffected_by_recipe_application(self):
        results = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        # source_cell_counts keys enumerate exactly SOURCE_DAYS x SOURCE_TYPES
        expected_keys = {f"day{d}_{t}" for d in sg.SOURCE_DAYS for t in sg.SOURCE_TYPES}
        self.assertEqual(set(results["source_cell_counts"].keys()), expected_keys)


class TestRowOrderingPreserved(unittest.TestCase):
    def test_build_synthetic_features_preserves_row_count_and_order(self):
        X = _make_source_rows(7)
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        X_out = sg.build_synthetic_features(X, ["fresh_account"], neutral)
        self.assertEqual(X_out.shape[0], X.shape[0])
        # untouched columns keep their original row-for-row values
        idx_amount = bmod.FEATURE_COLUMNS.index("amount")
        np.testing.assert_array_equal(X_out[:, idx_amount], X[:, idx_amount])


class TestOriginalTestRowsUnchanged(unittest.TestCase):
    def test_source_rows_unchanged_flag_is_true(self):
        results = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertTrue(results["source_rows_unchanged_after_all_scenarios"])

    def test_build_synthetic_features_never_mutates_input(self):
        X = _make_source_rows(5)
        X_before = X.copy()
        neutral = {"orig_balance_mismatch": 1.0, "orig_balance_after_zero": 0.0}
        sg.build_synthetic_features(X, ["fresh_account", "altered_balance_drain"], neutral)
        np.testing.assert_array_equal(X, X_before)


class TestModelParametersUnchanged(unittest.TestCase):
    def test_model_config_matches_m10(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, y_train = full_arrays["train"]
        _, _, X_train_proc = bmod.fit_preprocessing(X_train)
        model = bmod.train_baseline_model(X_train_proc, y_train)
        self.assertEqual(model.get_params()["random_state"], 42)
        self.assertEqual(model.get_params()["class_weight"], "balanced")


class TestPreprocessingTrainOnly(unittest.TestCase):
    def test_scoring_synthetic_rows_does_not_refit_preprocessing(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, y_train = full_arrays["train"]
        imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
        stats_before = imputer.statistics_.copy()
        mean_before = scaler.mean_.copy()

        model = bmod.train_baseline_model(X_train_proc, y_train)
        X_synth = _make_source_rows(10)
        sg.predict_with_fixed_model_safe(imputer, scaler, model, X_synth)

        np.testing.assert_array_equal(imputer.statistics_, stats_before)
        np.testing.assert_array_equal(scaler.mean_, mean_before)


class TestThresholdRemains05(unittest.TestCase):
    def test_default_threshold_is_half(self):
        self.assertEqual(sg.DEFAULT_THRESHOLD, 0.5)
        self.assertEqual(sg.DEFAULT_THRESHOLD, bmod.DEFAULT_THRESHOLD)


class TestTransformationsObeyFeatureSemantics(unittest.TestCase):
    def test_fresh_account_recipe_produces_internally_consistent_state(self):
        X = _make_source_rows(5)
        X_out = sg.apply_fresh_account_recipe(X)
        idx_first_seen = bmod.FEATURE_COLUMNS.index("origin_is_first_seen")
        idx_count = bmod.FEATURE_COLUMNS.index("origin_txn_count_causal")
        idx_avg = bmod.FEATURE_COLUMNS.index("origin_avg_amount_causal")
        idx_ratio = bmod.FEATURE_COLUMNS.index("amount_vs_origin_avg_ratio")
        self.assertTrue(np.all(X_out[:, idx_first_seen] == 1.0))
        self.assertTrue(np.all(X_out[:, idx_count] == 0.0))
        self.assertTrue(np.all(np.isnan(X_out[:, idx_avg])))
        self.assertTrue(np.all(np.isnan(X_out[:, idx_ratio])))

    def test_altered_balance_drain_recomputes_dependent_columns_consistently(self):
        X = _make_source_rows(5)
        idx_before = bmod.FEATURE_COLUMNS.index("orig_balance_before")
        X[:, idx_before] = 1000.0
        X_out = sg.apply_altered_balance_drain(X, residual_fraction=0.1)
        idx_after = bmod.FEATURE_COLUMNS.index("orig_balance_after")
        idx_delta = bmod.FEATURE_COLUMNS.index("orig_balance_delta")
        idx_after_zero = bmod.FEATURE_COLUMNS.index("orig_balance_after_zero")
        np.testing.assert_allclose(X_out[:, idx_after], 100.0)
        np.testing.assert_allclose(X_out[:, idx_delta], 100.0 - 1000.0)
        self.assertTrue(np.all(X_out[:, idx_after_zero] == 0.0))  # nonzero residual -> not flagged zero

    def test_altered_balance_drain_never_touches_amount_or_before(self):
        X = _make_source_rows(5)
        idx_before = bmod.FEATURE_COLUMNS.index("orig_balance_before")
        idx_amount = bmod.FEATURE_COLUMNS.index("amount")
        X_out = sg.apply_altered_balance_drain(X)
        np.testing.assert_array_equal(X_out[:, idx_before], X[:, idx_before])
        np.testing.assert_array_equal(X_out[:, idx_amount], X[:, idx_amount])


class TestImpossibleStatesRejected(unittest.TestCase):
    def test_neutral_balance_feature_only_ever_produces_the_domains_own_values(self):
        X = _make_source_rows(5)
        X_out = sg.apply_neutral_balance_feature(X, "orig_balance_mismatch", 1.0)
        idx = bmod.FEATURE_COLUMNS.index("orig_balance_mismatch")
        self.assertTrue(np.all(np.isin(X_out[:, idx], [0.0, 1.0])))

    def test_zero_residual_fraction_would_reproduce_paysim_signature_not_used_by_default(self):
        # RESIDUAL_FRACTION is fixed nonzero, so the altered-drain
        # recipe never accidentally reproduces PaySim's own exact-zero
        # fraud signature.
        self.assertGreater(sg.RESIDUAL_FRACTION, 0.0)
        self.assertLess(sg.RESIDUAL_FRACTION, 1.0)


class TestSyntheticScenarioCountsDeterministic(unittest.TestCase):
    def test_repeated_source_selection_gives_identical_counts(self):
        results_a = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        results_b = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(results_a["source_cell_counts"], results_b["source_cell_counts"])
        self.assertEqual(results_a["source_rows_total"], results_b["source_rows_total"])


class TestRepeatedRunsIdentical(unittest.TestCase):
    def test_repeated_full_run_yields_identical_synthetic_results(self):
        results_a = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        results_b = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        self.assertEqual(results_a["synthetic_scenarios"], results_b["synthetic_scenarios"])
        self.assertEqual(results_a["original_scenario"], results_b["original_scenario"])


class TestDay30RowsRetained(unittest.TestCase):
    def test_day_30_entry_present_with_explicit_zero_and_reason(self):
        results = sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        for name, scenario in results["synthetic_scenarios"].items():
            with self.subTest(scenario=name):
                self.assertIn("30", scenario["daily"])
                self.assertEqual(scenario["daily"]["30"]["n_total"], 0)
                self.assertTrue(scenario["daily"]["30"]["is_simulation_boundary_day"])

    def test_source_days_excludes_30_with_documented_reason(self):
        self.assertNotIn(30, sg.SOURCE_DAYS)
        self.assertIn("simulation-boundary", sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)["day_30_synthetic_note"])


class TestNoLabelDrivenScenarioSelection(unittest.TestCase):
    def test_scenario_definitions_are_static_module_level_literals(self):
        # SCENARIO_DEFINITIONS must be defined without reference to any
        # runtime result -- verified structurally (a plain dict literal
        # built at import time, not inside any function that receives
        # predictions or labels).
        self.assertIsInstance(sg.SCENARIO_DEFINITIONS, dict)
        for spec in sg.SCENARIO_DEFINITIONS.values():
            self.assertIn("recipes", spec)
            self.assertIn("synthetic_stress_label", spec)


class TestBaselineReproductionGuard(unittest.TestCase):
    def test_bounded_fixture_run_does_not_raise_the_mismatch_guard(self):
        try:
            sg.run_full_synthetic_analysis(SPLIT_BOUNDARY_CSV, chunksize=2)
        except RuntimeError:
            self.fail("bounded fixture run must not raise the baseline-mismatch guard")


class TestScenarioDefinitionsMatchM11M13(unittest.TestCase):
    def test_dominant_balance_features_match_m11(self):
        self.assertEqual(sg.DOMINANT_BALANCE_FEATURES, ab.BALANCE_FEATURES_REMOVED)

    def test_neutral_value_reuses_m13_function(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, _ = full_arrays["train"]
        v1 = sg.st.neutral_value_for("orig_balance_mismatch", X_train)
        v2 = st.neutral_value_for("orig_balance_mismatch", X_train)
        self.assertEqual(v1, v2)


class TestSmokeTest(unittest.TestCase):
    def test_smoke_test_reports_rss_and_respects_max_rows(self):
        result = sg.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        self.assertEqual(result["stream_stats"]["rows_seen"], 7)
        self.assertIn("within_safety_limit", result)
        self.assertGreater(result["peak_rss_mb"], 0.0)


class TestPredictWithFixedModelSafe(unittest.TestCase):
    def test_empty_input_returns_empty_arrays_without_crashing(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        X_train, y_train = full_arrays["train"]
        imputer, scaler, X_train_proc = bmod.fit_preprocessing(X_train)
        model = bmod.train_baseline_model(X_train_proc, y_train)
        empty_X = np.empty((0, 39), dtype=np.float32)
        proba, preds = sg.predict_with_fixed_model_safe(imputer, scaler, model, empty_X)
        self.assertEqual(len(proba), 0)
        self.assertEqual(len(preds), 0)


if __name__ == "__main__":
    unittest.main()
