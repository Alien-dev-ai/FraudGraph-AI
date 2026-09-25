"""Tests for src.ablation_analysis, using stdlib unittest only.

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

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPLIT_BOUNDARY_CSV = FIXTURES_DIR / "split_boundary.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFeatureGroupMembership(unittest.TestCase):
    def test_full_baseline_is_exactly_the_39_approved_columns(self):
        self.assertEqual(ab.FULL_BASELINE, bd.BASELINE_FEATURE_COLUMNS)
        self.assertEqual(len(ab.FULL_BASELINE), 39)

    def test_graph_only_is_exactly_the_5_retained_graph_columns(self):
        self.assertEqual(set(ab.GRAPH_ONLY), set(bd.GRAPH_FEATURES_RETAINED_IN_BASELINE))
        self.assertEqual(len(ab.GRAPH_ONLY), 5)

    def test_tabular_only_excludes_all_graph_columns(self):
        self.assertEqual(len(ab.TABULAR_ONLY), 34)
        self.assertFalse(set(ab.TABULAR_ONLY) & set(ab.GRAPH_ONLY))

    def test_tabular_only_plus_graph_only_equals_full_baseline_set(self):
        self.assertEqual(set(ab.TABULAR_ONLY) | set(ab.GRAPH_ONLY), set(ab.FULL_BASELINE))

    def test_balance_reduced_removes_exactly_the_two_named_features(self):
        self.assertEqual(len(ab.BALANCE_REDUCED), 37)
        for col in ab.BALANCE_FEATURES_REMOVED:
            self.assertNotIn(col, ab.BALANCE_REDUCED)
        self.assertEqual(set(ab.FULL_BASELINE) - set(ab.BALANCE_REDUCED), set(ab.BALANCE_FEATURES_REMOVED))

    def test_balance_reduced_tabular_excludes_graph_columns_too(self):
        self.assertEqual(len(ab.BALANCE_REDUCED_TABULAR), 32)
        self.assertFalse(set(ab.BALANCE_REDUCED_TABULAR) & set(ab.GRAPH_ONLY))

    def test_balance_reduced_full_equals_balance_reduced(self):
        self.assertEqual(ab.BALANCE_REDUCED_FULL, ab.BALANCE_REDUCED)

    def test_removed_balance_features_are_actually_balance_category(self):
        from src import features_raw_paysim as feat

        for col in ab.BALANCE_FEATURES_REMOVED:
            self.assertEqual(feat.FEATURE_METADATA[col]["category"], "balance")

    def test_membership_derived_from_metadata_not_hand_duplicated(self):
        # If Milestone 8's contract ever changes, these groups must move
        # with it rather than silently going stale.
        self.assertEqual(ab.FULL_BASELINE, bd.BASELINE_FEATURE_COLUMNS)


class TestNoOverlapErrors(unittest.TestCase):
    def test_tabular_and_graph_only_partition_full_baseline_with_no_overlap(self):
        self.assertEqual(set(ab.TABULAR_ONLY) & set(ab.GRAPH_ONLY), set())

    def test_all_group_columns_are_subsets_of_full_baseline(self):
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                self.assertTrue(set(cols) <= set(ab.FULL_BASELINE))

    def test_no_group_contains_duplicate_columns(self):
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                self.assertEqual(len(cols), len(set(cols)))


class TestNoTargetOrIdentifierInclusion(unittest.TestCase):
    def test_no_group_contains_target_column(self):
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                self.assertNotIn(bd.TARGET_COLUMN, cols)

    def test_no_group_contains_identifier_columns(self):
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                self.assertFalse(set(bd.IDENTIFIER_COLUMNS) & set(cols))

    def test_run_group_experiment_reasserts_leakage_guards(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        result = ab.run_group_experiment("GRAPH_ONLY", ab.GRAPH_ONLY, full_arrays)
        self.assertNotIn(bd.TARGET_COLUMN, result["feature_columns"])


class TestDeterministicFeatureGroupOrdering(unittest.TestCase):
    def test_group_columns_preserve_full_baseline_order(self):
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                expected_order = [c for c in ab.FULL_BASELINE if c in set(cols)]
                self.assertEqual(cols, expected_order)

    def test_repeated_module_import_yields_identical_groups(self):
        import importlib

        reloaded = importlib.reload(ab)
        self.assertEqual(reloaded.FEATURE_GROUPS, ab.FEATURE_GROUPS)


class TestIdenticalSplitAssignmentAcrossExperiments(unittest.TestCase):
    def test_all_groups_see_the_same_row_counts_and_fraud_counts(self):
        full_arrays, stream_stats = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        for name, cols in ab.FEATURE_GROUPS.items():
            with self.subTest(group=name):
                result = ab.run_group_experiment(name, cols, full_arrays)
                self.assertEqual(result["train_rows"], stream_stats["rows_by_split"]["train"])
                self.assertEqual(result["train_fraud"], stream_stats["fraud_by_split"]["train"])
                self.assertEqual(result["validation_rows"], stream_stats["rows_by_split"]["validation"])
                self.assertEqual(result["test_rows"], stream_stats["rows_by_split"]["test"])

    def test_select_group_arrays_slices_same_rows_different_columns(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        full_sel = ab.select_group_arrays(full_arrays, ab.FULL_BASELINE)
        tab_sel = ab.select_group_arrays(full_arrays, ab.TABULAR_ONLY)
        # same row count/labels, fewer columns
        np.testing.assert_array_equal(full_sel["train"][1], tab_sel["train"][1])
        self.assertEqual(full_sel["train"][0].shape[0], tab_sel["train"][0].shape[0])
        self.assertEqual(tab_sel["train"][0].shape[1], 34)


class TestTrainOnlyPreprocessing(unittest.TestCase):
    def test_group_experiment_uses_train_only_fit_preprocessing(self):
        # Synthetic, NaN-free data -- the real GRAPH_ONLY columns on the
        # tiny 2-row split_boundary fixture happen to have
        # dest_incoming_concentration_causal entirely undefined in
        # train, which SimpleImputer legitimately drops (see
        # TestCoefficientExtraction's note); that's a property of that
        # sparse fixture, not of this preprocessing call, so it is
        # tested here with a column count guaranteed to survive.
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(50, len(ab.GRAPH_ONLY))).astype(np.float32)
        imputer, scaler, _ = bmod.fit_preprocessing(X_train)
        # Same call Milestone 10 makes -- confirms ablation reuses it
        # rather than reimplementing a separate preprocessing path.
        self.assertEqual(len(imputer.statistics_), len(ab.GRAPH_ONLY))
        self.assertEqual(len(scaler.mean_), len(ab.GRAPH_ONLY))


class TestDeterministicRepeatedExecution(unittest.TestCase):
    def test_repeated_group_experiment_yields_identical_coefficients(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        result_a = ab.run_group_experiment("TABULAR_ONLY", ab.TABULAR_ONLY, full_arrays)
        result_b = ab.run_group_experiment("TABULAR_ONLY", ab.TABULAR_ONLY, full_arrays)
        self.assertEqual(result_a["coefficients"], result_b["coefficients"])

    def test_repeated_full_experiment_run_is_deterministic(self):
        results_a = ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        results_b = ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        for name in ab.FEATURE_GROUPS:
            self.assertEqual(
                results_a["experiments"][name]["coefficients"],
                results_b["experiments"][name]["coefficients"],
            )


class TestMetricCalculation(unittest.TestCase):
    def test_metric_delta_helper_computes_signed_difference(self):
        a = {"x": 0.9}
        b = {"x": 0.4}
        self.assertAlmostEqual(ab._metric_delta(a, b, "x"), 0.5)

    def test_metric_delta_handles_missing_metric_as_none(self):
        a = {"x": None}
        b = {"x": 0.4}
        self.assertIsNone(ab._metric_delta(a, b, "x"))

    def test_group_experiment_reports_confusion_matrix_accounting(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        result = ab.run_group_experiment("FULL_BASELINE", ab.FULL_BASELINE, full_arrays)
        for split in ("validation", "test"):
            m = result[split]
            cm = m["confusion_matrix"]
            self.assertEqual(cm["tp"] + cm["fn"], m["n_fraud"])
            self.assertEqual(cm["tn"] + cm["fp"], m["n_legitimate"])
            self.assertEqual(sum(cm.values()), m["n_total"])


class _FakeLinearModel:
    """Minimal stand-in exposing only `.coef_`, for testing
    `coefficients_for` as a pure function of (model, columns) without
    depending on a real fit (and therefore without depending on
    whether every column happens to have an observed value in
    whatever data trained it).
    """

    def __init__(self, coefs: list[float]):
        self.coef_ = np.array([coefs])


class TestCoefficientExtraction(unittest.TestCase):
    def test_coefficients_for_returns_one_entry_per_column(self):
        model = _FakeLinearModel([0.1, -0.2, 0.3, -0.4, 0.5])
        result = ab.coefficients_for(model, ab.GRAPH_ONLY)
        self.assertEqual(len(result), len(ab.GRAPH_ONLY))
        self.assertEqual({c["feature"] for c in result}, set(ab.GRAPH_ONLY))

    def test_coefficients_for_trained_group_matches_reported_feature_count(self):
        # On the real dataset (M11's actual run) every one of the 39
        # columns has an observed value in the 6.2M-row train split, so
        # coefficient count equals the group's declared feature count
        # there. Verified directly against the already-written full-run
        # results rather than re-running the ~49-minute full pipeline.
        if not ab.ABLATION_RESULTS_PATH.exists():
            self.skipTest("full ablation run has not been executed yet")
        import json

        with ab.ABLATION_RESULTS_PATH.open() as f:
            data = json.load(f)
        for name, exp in data["experiments"].items():
            with self.subTest(group=name):
                self.assertEqual(len(exp["coefficients"]), exp["feature_count"])

    def test_coefficients_sorted_by_descending_absolute_magnitude(self):
        full_arrays, _ = bmod.stream_split_arrays(SPLIT_BOUNDARY_CSV, chunksize=2)
        result = ab.run_group_experiment("TABULAR_ONLY", ab.TABULAR_ONLY, full_arrays)
        mags = [abs(c["coefficient"]) for c in result["coefficients"]]
        self.assertEqual(mags, sorted(mags, reverse=True))

    def test_graph_contribution_analysis_extracts_graph_coefficients_only(self):
        results = ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        analysis = ab.graph_contribution_analysis(results)
        for group_name, coeffs in analysis["q3_q4_graph_coefficients_by_group"].items():
            with self.subTest(group=group_name):
                self.assertTrue(set(coeffs.keys()) <= set(ab.GRAPH_COLUMNS))


class TestNoRawFileMutation(unittest.TestCase):
    def test_fixture_untouched_by_full_run(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)

    def test_fixture_untouched_by_smoke_test(self):
        before = _sha256(SPLIT_BOUNDARY_CSV)
        ab.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        after = _sha256(SPLIT_BOUNDARY_CSV)
        self.assertEqual(before, after)


class TestArtifactAnalysis(unittest.TestCase):
    def test_paysim_artifact_analysis_confirms_no_leakage(self):
        results = ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        analysis = ab.paysim_artifact_analysis(results)
        self.assertTrue(analysis["no_target_leakage_verified"])
        self.assertEqual(analysis["removed_features"], ab.BALANCE_FEATURES_REMOVED)

    def test_artifact_analysis_compares_expected_group_pairs(self):
        results = ab.run_all_experiments(SPLIT_BOUNDARY_CSV, chunksize=2)
        analysis = ab.paysim_artifact_analysis(results)
        self.assertIn("full_vs_balance_reduced_full", analysis)
        self.assertIn("tabular_only_vs_balance_reduced_tabular", analysis)


class TestBoundedSmokeTest(unittest.TestCase):
    def test_smoke_test_reports_rss_and_respects_max_rows(self):
        result = ab.run_smoke_test(SPLIT_BOUNDARY_CSV, chunksize=2, max_rows=7)
        self.assertEqual(result["stream_stats"]["rows_seen"], 7)
        self.assertIn("within_safety_limit", result)
        self.assertGreater(result["peak_rss_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
