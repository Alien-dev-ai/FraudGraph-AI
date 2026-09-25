"""Tests for src.eda_raw_paysim, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import eda_raw_paysim as eda  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EDA_CSV = FIXTURES_DIR / "eda_mini.csv"
MINI_PAYSIM_CSV = FIXTURES_DIR / "mini_paysim.csv"

# Force multiple chunks (13 rows / chunksize=4 -> 4 chunks) to exercise
# cross-chunk accumulation correctness, not just a single-chunk pass.
SMALL_CHUNKSIZE = 4


class TestChunkedPassCounts(unittest.TestCase):
    """Hand-verified expected counts for tests/fixtures/eda_mini.csv."""

    @classmethod
    def setUpClass(cls):
        cls.report = eda.build_report(EDA_CSV, chunksize=SMALL_CHUNKSIZE)

    def test_total_transactions(self):
        self.assertEqual(self.report["total_transactions"], 13)

    def test_counts_by_type(self):
        self.assertEqual(self.report["counts_by_type"], {
            "PAYMENT": 3, "CASH_OUT": 4, "TRANSFER": 3, "CASH_IN": 1, "DEBIT": 2,
        })

    def test_total_fraud_and_rate(self):
        self.assertEqual(self.report["total_fraud_transactions"], 5)
        self.assertAlmostEqual(self.report["overall_fraud_rate"], 5 / 13)

    def test_fraud_by_type(self):
        fbt = self.report["fraud_by_type"]
        self.assertEqual(fbt["PAYMENT"]["fraud_count"], 0)
        self.assertEqual(fbt["CASH_OUT"]["fraud_count"], 2)
        self.assertEqual(fbt["CASH_OUT"]["count"], 4)
        self.assertAlmostEqual(fbt["CASH_OUT"]["fraud_rate"], 0.5)
        self.assertEqual(fbt["TRANSFER"]["fraud_count"], 3)
        self.assertAlmostEqual(fbt["TRANSFER"]["fraud_rate"], 1.0)
        self.assertEqual(fbt["CASH_IN"]["fraud_count"], 0)
        self.assertEqual(fbt["DEBIT"]["fraud_count"], 0)

    def test_fraud_by_day_buckets(self):
        fbd = self.report["fraud_by_day"]
        # steps 1-8 -> day_bucket 0; steps 30/31/32/33/34 -> day_bucket 1
        self.assertEqual(fbd["0"]["total_count"], 8)
        self.assertEqual(fbd["0"]["fraud_count"], 3)
        self.assertEqual(fbd["1"]["total_count"], 5)
        self.assertEqual(fbd["1"]["fraud_count"], 2)

    def test_account_activity(self):
        acc = self.report["account_activity"]
        self.assertEqual(acc["unique_origin_accounts"], 11)
        self.assertEqual(acc["unique_destination_accounts"], 11)
        self.assertEqual(acc["origin_accounts_with_multiple_transactions"], 2)  # C1, C2
        self.assertEqual(acc["destination_accounts_with_multiple_transactions"], 2)  # M1, C10

    def test_zero_balance_patterns(self):
        zbp = self.report["zero_balance_patterns"]
        self.assertEqual(zbp["origin_both_zero_count"], 1)  # row 13 (C19)
        self.assertEqual(zbp["origin_both_zero_fraud_count"], 1)
        self.assertEqual(zbp["nonmerchant_destination_both_zero_count"], 1)  # row 3 (C11)
        self.assertEqual(zbp["nonmerchant_destination_both_zero_fraud_count"], 1)
        self.assertEqual(zbp["merchant_destination_rows"], 3)  # M1, M2, M1

    def test_flagged_vs_actual_fraud_crosstab(self):
        crosstab = self.report["flagged_vs_actual_fraud"]
        self.assertEqual(crosstab, {"0_0": 8, "0_1": 3, "1_0": 0, "1_1": 2})

    def test_chunking_does_not_change_totals_vs_single_chunk(self):
        single_chunk_report = eda.build_report(EDA_CSV, chunksize=1000)
        self.assertEqual(
            self.report["counts_by_type"], single_chunk_report["counts_by_type"]
        )
        self.assertEqual(
            self.report["total_fraud_transactions"],
            single_chunk_report["total_fraud_transactions"],
        )
        self.assertEqual(self.report["fraud_by_day"], single_chunk_report["fraud_by_day"])


class TestCorrelationsAgainstDirectComputation(unittest.TestCase):
    """Cross-validate the chunked accumulator's correlations against a
    direct, non-chunked numpy computation on the same fixture."""

    @classmethod
    def setUpClass(cls):
        cls.report = eda.build_report(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        cls.df = pd.read_csv(EDA_CSV)

    def test_amount_vs_oldbalanceorg_correlation_matches_numpy(self):
        expected = np.corrcoef(self.df["amount"], self.df["oldbalanceOrg"])[0, 1]
        actual = self.report["amount_balance_correlations"]["amount_vs_oldbalanceOrg"]
        self.assertAlmostEqual(actual, expected, places=8)

    def test_oldbalanceorg_vs_newbalanceorig_correlation_matches_numpy(self):
        expected = np.corrcoef(self.df["oldbalanceOrg"], self.df["newbalanceOrig"])[0, 1]
        actual = self.report["amount_balance_correlations"]["oldbalanceOrg_vs_newbalanceOrig"]
        self.assertAlmostEqual(actual, expected, places=8)


class TestQuantilePassAgainstDirectComputation(unittest.TestCase):
    """Cross-validate the light quantile pass against direct pandas groupby."""

    @classmethod
    def setUpClass(cls):
        cls.report = eda.build_report(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        cls.df = pd.read_csv(EDA_CSV)

    def test_amount_mean_by_type_matches_direct_groupby(self):
        expected = self.df.groupby("type")["amount"].mean()
        for t, stats in self.report["amount_stats_by_type"].items():
            self.assertAlmostEqual(stats["mean"], expected.loc[t], places=6)

    def test_amount_median_by_fraud_label_matches_direct_groupby(self):
        expected = self.df.groupby("isFraud")["amount"].median()
        for f, stats in self.report["amount_stats_by_fraud_label"].items():
            self.assertAlmostEqual(stats["p50"], expected.loc[int(f)], places=6)

    def test_fraud_amounts_sum_and_count(self):
        stats = self.report["amount_stats_by_fraud_label"]["1"]
        fraud_rows = self.df[self.df["isFraud"] == 1]
        self.assertEqual(stats["count"], len(fraud_rows))
        self.assertAlmostEqual(stats["mean"], fraud_rows["amount"].mean(), places=6)


class TestClassificationNotes(unittest.TestCase):
    def test_three_categories_present_and_nonempty(self):
        report = eda.build_report(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        notes = report["classification_notes"]
        self.assertIn("descriptive_observations", notes)
        self.assertIn("known_paysim_simulator_artifacts", notes)
        self.assertIn("findings_requiring_modeling_to_verify", notes)
        for key in notes:
            self.assertGreater(len(notes[key]), 0)


class TestReportWriters(unittest.TestCase):
    def test_writers_produce_expected_files(self):
        report = eda.build_report(EDA_CSV, chunksize=SMALL_CHUNKSIZE)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            json_path = tmp / "summary.json"
            type_csv_path = tmp / "by_type.csv"
            day_csv_path = tmp / "by_day.csv"
            text_path = tmp / "summary.txt"

            eda.write_json_report(report, json_path)
            eda.write_type_csv(report, type_csv_path)
            eda.write_day_csv(report, day_csv_path)
            eda.write_text_summary(report, text_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(type_csv_path.exists())
            self.assertTrue(day_csv_path.exists())
            self.assertTrue(text_path.exists())

            loaded = json.loads(json_path.read_text())
            self.assertEqual(loaded["total_transactions"], 13)

            type_csv_text = type_csv_path.read_text()
            self.assertIn("TRANSFER", type_csv_text)

            text_summary = text_path.read_text()
            self.assertIn("Descriptive observations", text_summary)
            self.assertIn("Known PaySim simulator artifacts", text_summary)
            self.assertIn("Findings requiring later modeling", text_summary)
            self.assertIn("No image plots", text_summary)


class TestMissingAndMalformedInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(eda.DatasetNotFoundError):
            eda.build_report(FIXTURES_DIR / "does_not_exist.csv")

    def test_missing_required_columns_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "bad.csv"
            bad_csv.write_text("step,type,amount\n1,PAYMENT,10.0\n")
            with self.assertRaises(eda.SchemaValidationError):
                eda.build_report(bad_csv)


class TestReusesMiniPaysimFixture(unittest.TestCase):
    """Sanity check against the Milestone 1 fixture too, not just eda_mini.csv."""

    def test_runs_cleanly_on_mini_paysim_fixture(self):
        report = eda.build_report(MINI_PAYSIM_CSV, chunksize=3)
        self.assertEqual(report["total_transactions"], 10)
        self.assertEqual(
            sum(report["counts_by_type"].values()), report["total_transactions"]
        )


if __name__ == "__main__":
    unittest.main()
