"""Tests for src.ingest, using stdlib unittest only (no extra dependencies).

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ingest  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MINI_CSV = FIXTURES_DIR / "mini_paysim.csv"


class TestSuccessfulLoad(unittest.TestCase):
    def test_loads_miniature_csv_and_builds_report(self):
        report = ingest.ingest(csv_path=MINI_CSV)
        self.assertEqual(report.row_count, 10)
        self.assertEqual(report.column_count, 11)
        self.assertEqual(set(report.columns), set(ingest.REQUIRED_COLUMNS))


class TestColumnValidation(unittest.TestCase):
    def test_raises_on_missing_required_columns(self):
        import pandas as pd

        df = pd.DataFrame({"step": [1], "type": ["PAYMENT"], "amount": [10.0]})
        with self.assertRaises(ingest.SchemaValidationError):
            ingest.validate_columns(df)

    def test_passes_when_all_required_columns_present(self):
        df = ingest.load_csv(MINI_CSV)
        ingest.validate_columns(df)  # should not raise


class TestMissingFileHandling(unittest.TestCase):
    def test_raises_when_csv_path_does_not_exist(self):
        missing_path = FIXTURES_DIR / "does_not_exist.csv"
        with self.assertRaises(ingest.DatasetNotFoundError):
            ingest.load_csv(missing_path)

    def test_raises_when_raw_dir_has_no_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_dir = Path(tmp_dir)
            with self.assertRaises(ingest.DatasetNotFoundError):
                ingest.find_dataset(raw_dir=empty_dir)

    def test_raises_when_raw_dir_missing(self):
        nonexistent_dir = FIXTURES_DIR / "no_such_directory"
        with self.assertRaises(ingest.DatasetNotFoundError):
            ingest.find_dataset(raw_dir=nonexistent_dir)


class TestMissingColumnHandling(unittest.TestCase):
    def test_ingest_raises_clear_error_for_malformed_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "bad.csv"
            bad_csv.write_text("step,type,amount\n1,PAYMENT,10.0\n")
            with self.assertRaises(ingest.SchemaValidationError) as ctx:
                ingest.ingest(csv_path=bad_csv)
            self.assertIn("nameOrig", str(ctx.exception))


class TestFraudLabelDistribution(unittest.TestCase):
    def test_reports_fraud_label_counts_and_rate(self):
        report = ingest.ingest(csv_path=MINI_CSV)
        self.assertEqual(sum(report.fraud_label_counts.values()), 10)
        self.assertEqual(report.fraud_label_counts.get(1), 2)
        self.assertEqual(report.fraud_label_counts.get(0), 8)
        self.assertAlmostEqual(report.fraud_label_rate, 0.2)

    def test_reports_missing_values(self):
        report = ingest.ingest(csv_path=MINI_CSV)
        self.assertEqual(report.missing_values.get("oldbalanceOrg"), 1)


if __name__ == "__main__":
    unittest.main()
