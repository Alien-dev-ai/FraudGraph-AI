"""Tests for src.validate_raw_paysim, using stdlib unittest only.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import validate_raw_paysim as vq  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MINI_CSV = FIXTURES_DIR / "mini_paysim.csv"
EDGE_CSV = FIXTURES_DIR / "quality_edge_cases.csv"


def _checks_by_name(report):
    return {c["name"]: c for c in report["checks"]}


class TestCleanFixture(unittest.TestCase):
    """mini_paysim.csv is well-formed except for one deliberate missing value."""

    @classmethod
    def setUpClass(cls):
        df = vq.load_and_validate(MINI_CSV)
        cls.report = vq.run_validation(df, MINI_CSV)
        cls.checks = _checks_by_name(cls.report)

    def test_row_and_column_counts(self):
        self.assertEqual(self.report["row_count"], 10)
        self.assertEqual(self.report["column_count"], 11)

    def test_schema_passes(self):
        self.assertEqual(self.checks["schema_and_dtypes"]["status"], "pass")

    def test_no_duplicates(self):
        self.assertEqual(self.checks["duplicate_rows"]["status"], "pass")
        self.assertEqual(self.checks["duplicate_rows"]["duplicate_row_count"], 0)

    def test_all_transaction_types_valid(self):
        self.assertEqual(self.checks["transaction_type_domain"]["status"], "pass")

    def test_no_negative_values(self):
        self.assertEqual(self.checks["nonnegative_amount_and_balance_fields"]["status"], "pass")

    def test_binary_flags_valid(self):
        self.assertEqual(self.checks["binary_flag_domain"]["status"], "pass")

    def test_missing_value_detected(self):
        check = self.checks["missing_values"]
        self.assertEqual(check["status"], "issue")
        self.assertEqual(check["missing_value_counts"]["oldbalanceOrg"], 1)

    def test_balance_consistency_is_expected_behavior_not_issue(self):
        self.assertEqual(self.checks["balance_consistency"]["status"], "expected_behavior")

    def test_duplicate_transaction_id_not_applicable(self):
        self.assertEqual(self.checks["duplicate_transaction_id"]["status"], "not_applicable")


class TestEdgeCaseFixture(unittest.TestCase):
    """quality_edge_cases.csv deliberately triggers one instance of each issue check."""

    @classmethod
    def setUpClass(cls):
        df = vq.load_and_validate(EDGE_CSV)
        cls.report = vq.run_validation(df, EDGE_CSV)
        cls.checks = _checks_by_name(cls.report)

    def test_duplicate_row_detected(self):
        check = self.checks["duplicate_rows"]
        self.assertEqual(check["status"], "issue")
        self.assertEqual(check["duplicate_row_count"], 1)

    def test_invalid_transaction_type_detected(self):
        check = self.checks["transaction_type_domain"]
        self.assertEqual(check["status"], "issue")
        self.assertIn("UNKNOWN_TYPE", check["invalid_types"])

    def test_negative_amount_detected(self):
        check = self.checks["nonnegative_amount_and_balance_fields"]
        self.assertEqual(check["status"], "issue")
        self.assertEqual(check["negative_value_counts"]["amount"], 1)

    def test_invalid_binary_flag_detected(self):
        check = self.checks["binary_flag_domain"]
        self.assertEqual(check["status"], "issue")
        self.assertEqual(check["invalid_details"]["isFraud"]["invalid_count"], 1)
        self.assertIn(2, check["invalid_details"]["isFraud"]["invalid_values"])

    def test_unexpected_account_prefixes_detected(self):
        check = self.checks["account_id_prefix_convention"]
        self.assertEqual(check["status"], "issue")
        self.assertEqual(check["invalid_origin_prefix_count"], 1)
        self.assertIn("X", check["unexpected_destination_prefixes"])

    def test_genuine_issues_found_flag_is_true(self):
        self.assertTrue(self.report["genuine_issues_found"])

    def test_merchant_balance_still_classified_expected_behavior(self):
        # Rows 1-2 have a merchant destination (M1) with zero balances either side;
        # this must stay "expected_behavior", not be conflated with the genuine issues.
        check = self.checks["balance_consistency"]
        self.assertEqual(check["status"], "expected_behavior")
        self.assertEqual(check["merchant_destination_accounts"], 2)
        self.assertEqual(check["merchant_destination_zero_balance_pct"], 100.0)


class TestReportWriting(unittest.TestCase):
    def test_write_report_produces_valid_json(self):
        df = vq.load_and_validate(MINI_CSV)
        report = vq.run_validation(df, MINI_CSV)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "nested" / "report.json"
            vq.write_report(report, out_path)
            self.assertTrue(out_path.exists())
            loaded = json.loads(out_path.read_text())
            self.assertEqual(loaded["row_count"], report["row_count"])
            self.assertIn("checks", loaded)
            self.assertIn("status_summary", loaded)


class TestMissingAndMalformedInput(unittest.TestCase):
    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(vq.DatasetNotFoundError):
            vq.load_and_validate(FIXTURES_DIR / "does_not_exist.csv")

    def test_missing_required_columns_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "bad.csv"
            bad_csv.write_text("step,type,amount\n1,PAYMENT,10.0\n")
            with self.assertRaises(vq.SchemaValidationError):
                vq.load_and_validate(bad_csv)


if __name__ == "__main__":
    unittest.main()
