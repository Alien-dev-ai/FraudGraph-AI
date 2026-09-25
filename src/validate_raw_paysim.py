"""Raw PaySim normalization and data-quality validation (Milestone 3).

The raw PaySim CSV already conforms to the canonical schema validated in
Milestone 1/2 (src.ingest.REQUIRED_COLUMNS) — no column renaming or type
coercion is required to "normalize" it. What this module does is confirm
that the raw file actually matches that canonical schema at the value
level (dtypes, categorical domains, binary flags, balance semantics) and
produce a compact, machine-readable data-quality report.

Checks performed:
  - expected columns present with expected dtype kinds
  - row count
  - duplicate rows (exact full-row duplicates)
  - transaction-type domain (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER)
  - nonnegative amount/balance fields
  - binary domain for isFraud / isFlaggedFraud
  - missing values per column
  - balance consistency (origin and destination balance arithmetic)
  - account-id prefix conventions (customer "C" / merchant "M")

Each check is tagged with a status of "pass", "issue", "expected_behavior",
or "not_applicable" so genuine data-quality problems are not conflated with
known, documented PaySim simulator characteristics.

Out of scope here: feature engineering, graph construction, model training,
dashboard work, and writing out a transformed copy of the dataset. This
module only reads the raw CSV; it never writes to data/raw/ or data/raw's
source file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest import (
    REQUIRED_COLUMNS,
    DatasetNotFoundError,
    SchemaValidationError,
    load_csv,
    validate_columns,
)

VALID_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

BINARY_COLUMNS = ["isFraud", "isFlaggedFraud"]

NONNEGATIVE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

EXPECTED_DTYPE_KINDS: dict[str, str] = {
    "step": "integer",
    "type": "string",
    "amount": "float",
    "nameOrig": "string",
    "oldbalanceOrg": "float",
    "newbalanceOrig": "float",
    "nameDest": "string",
    "oldbalanceDest": "float",
    "newbalanceDest": "float",
    "isFraud": "integer",
    "isFlaggedFraud": "integer",
}

BALANCE_TOLERANCE = 0.01  # PaySim amounts are cent-precision floats

DEFAULT_RAW_CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "archive"
    / "paysim_orig"
    / "synthetic_financial_datasets_fraud_detection.csv"
)

DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parent.parent / "reports" / "raw_paysim_quality_report.json"
)


def _dtype_kind(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "string"
    return str(series.dtype)


def check_schema(df: pd.DataFrame) -> dict[str, Any]:
    mismatches = {}
    for col, expected_kind in EXPECTED_DTYPE_KINDS.items():
        if col not in df.columns:
            continue  # already covered by validate_columns() upstream
        actual_kind = _dtype_kind(df[col])
        if actual_kind != expected_kind:
            mismatches[col] = {"expected": expected_kind, "actual": actual_kind}

    status = "pass" if not mismatches else "issue"
    return {
        "name": "schema_and_dtypes",
        "status": status,
        "expected_columns": REQUIRED_COLUMNS,
        "actual_columns": list(df.columns),
        "dtype_mismatches": mismatches,
        "detail": "All expected columns present with expected dtype kinds."
        if status == "pass"
        else f"{len(mismatches)} column(s) have unexpected dtype kinds.",
    }


def check_row_count(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": "row_count",
        "status": "pass",
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "detail": "Informational only; not a pass/fail check.",
    }


def check_duplicate_rows(df: pd.DataFrame) -> dict[str, Any]:
    dup_count = int(df.duplicated().sum())
    pct = (dup_count / len(df) * 100) if len(df) else 0.0
    status = "pass" if dup_count == 0 else "issue"
    return {
        "name": "duplicate_rows",
        "status": status,
        "duplicate_row_count": dup_count,
        "duplicate_row_pct": round(pct, 6),
        "detail": "No exact full-row duplicates found."
        if status == "pass"
        else "Exact full-row duplicates found — genuine data-quality concern, "
        "not an expected PaySim characteristic.",
    }


def check_transaction_types(df: pd.DataFrame) -> dict[str, Any]:
    observed = df["type"].value_counts(dropna=False)
    observed_counts = {str(k): int(v) for k, v in observed.items()}
    invalid_types = sorted(set(observed_counts) - VALID_TRANSACTION_TYPES)
    status = "pass" if not invalid_types else "issue"
    return {
        "name": "transaction_type_domain",
        "status": status,
        "valid_types": sorted(VALID_TRANSACTION_TYPES),
        "observed_counts": observed_counts,
        "invalid_types": invalid_types,
        "detail": "All transaction types fall within the known PaySim domain."
        if status == "pass"
        else f"Unexpected transaction type value(s) found: {invalid_types}.",
    }


def check_nonnegative_fields(df: pd.DataFrame) -> dict[str, Any]:
    negative_counts = {}
    for col in NONNEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        negative_counts[col] = int((df[col] < 0).sum())
    total_negative = sum(negative_counts.values())
    status = "pass" if total_negative == 0 else "issue"
    return {
        "name": "nonnegative_amount_and_balance_fields",
        "status": status,
        "negative_value_counts": negative_counts,
        "detail": "No negative amount or balance values found."
        if status == "pass"
        else "Negative values found in amount/balance fields — not valid for "
        "monetary fields, genuine data-quality concern.",
    }


def check_binary_flags(df: pd.DataFrame) -> dict[str, Any]:
    invalid_details = {}
    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].isin([0, 1])
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            invalid_details[col] = {
                "invalid_count": invalid_count,
                "invalid_values": sorted(set(df.loc[invalid_mask, col].unique().tolist())),
            }
    status = "pass" if not invalid_details else "issue"
    return {
        "name": "binary_flag_domain",
        "status": status,
        "columns_checked": BINARY_COLUMNS,
        "invalid_details": invalid_details,
        "detail": "isFraud and isFlaggedFraud contain only 0/1 values."
        if status == "pass"
        else "Non-binary values found in isFraud/isFlaggedFraud — genuine "
        "data-quality concern.",
    }


def check_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    total_missing = sum(missing.values())
    status = "pass" if total_missing == 0 else "issue"
    return {
        "name": "missing_values",
        "status": status,
        "missing_value_counts": missing,
        "detail": "No missing values in any column."
        if status == "pass"
        else "Missing values found — genuine data-quality concern requiring "
        "an explicit handling decision before modeling.",
    }


def check_balance_consistency(df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "oldbalanceOrg", "newbalanceOrig", "amount", "type",
    }
    if not required.issubset(df.columns):
        return {
            "name": "balance_consistency",
            "status": "not_applicable",
            "detail": "Required balance/account/type columns not present.",
        }

    is_merchant_dest = df["nameDest"].str.startswith("M")
    merchant_count = int(is_merchant_dest.sum())
    merchant_zero_balance = int(
        ((df.loc[is_merchant_dest, "oldbalanceDest"] == 0)
         & (df.loc[is_merchant_dest, "newbalanceDest"] == 0)).sum()
    ) if merchant_count else 0
    merchant_zero_balance_pct = (
        merchant_zero_balance / merchant_count * 100 if merchant_count else None
    )

    # CASH_IN adds funds to the origin account; every other type (CASH_OUT,
    # DEBIT, PAYMENT, TRANSFER) removes funds from it. Using a single
    # subtraction formula for all types would misclassify every CASH_IN row.
    orig_sign = np.where(df["type"] == "CASH_IN", 1, -1)
    expected_new_orig = df["oldbalanceOrg"] + orig_sign * df["amount"]
    orig_mismatch = ~np.isclose(
        df["newbalanceOrig"], expected_new_orig, atol=BALANCE_TOLERANCE
    )
    orig_mismatch_count = int(orig_mismatch.sum())
    orig_mismatch_by_fraud = (
        df.loc[orig_mismatch, "isFraud"].value_counts().to_dict()
        if "isFraud" in df.columns
        else {}
    )
    orig_mismatch_by_fraud = {str(k): int(v) for k, v in orig_mismatch_by_fraud.items()}

    non_merchant_dest = ~is_merchant_dest
    expected_new_dest = df["oldbalanceDest"] + df["amount"]
    dest_mismatch = non_merchant_dest & ~np.isclose(
        df["newbalanceDest"], expected_new_dest, atol=BALANCE_TOLERANCE
    )
    dest_mismatch_count = int(dest_mismatch.sum())

    return {
        "name": "balance_consistency",
        "status": "expected_behavior",
        "merchant_destination_accounts": merchant_count,
        "merchant_destination_zero_balance_pct": (
            round(merchant_zero_balance_pct, 4) if merchant_zero_balance_pct is not None else None
        ),
        "origin_balance_equation_mismatch_count": orig_mismatch_count,
        "origin_balance_equation_mismatch_pct": round(
            orig_mismatch_count / len(df) * 100, 4
        ) if len(df) else 0.0,
        "origin_balance_equation_mismatch_by_fraud_label": orig_mismatch_by_fraud,
        "destination_balance_equation_mismatch_count_nonmerchant": dest_mismatch_count,
        "destination_balance_equation_mismatch_pct_nonmerchant": round(
            dest_mismatch_count / max(int(non_merchant_dest.sum()), 1) * 100, 4
        ),
        "detail": (
            "Merchant destination accounts (nameDest starting with 'M') have "
            "balances of 0 by design — PaySim does not simulate merchant "
            "balances. The origin-side check is direction-aware (CASH_IN "
            "adds funds, all other types remove them). Even so, a large "
            "share of rows still show newBalance != oldBalance +/- amount "
            "(including the well-documented 'both balances reported as "
            "zero' pattern, and non-merchant destination balances that "
            "visibly do not track oldbalanceDest + amount, e.g. for "
            "CASH_IN). This lack of strict balance conservation is a "
            "widely documented PaySim simulator characteristic, not "
            "evidence of a corrupted raw file. Reported here as diagnostic "
            "statistics only — no derived columns are persisted."
        ),
    }


def check_account_prefixes(df: pd.DataFrame) -> dict[str, Any]:
    if not {"nameOrig", "nameDest"}.issubset(df.columns):
        return {
            "name": "account_id_prefix_convention",
            "status": "not_applicable",
            "detail": "nameOrig/nameDest columns not present.",
        }

    invalid_orig_prefix = int((~df["nameOrig"].str.startswith("C")).sum())

    dest_prefixes = df["nameDest"].str[0].value_counts()
    dest_prefix_counts = {str(k): int(v) for k, v in dest_prefixes.items()}
    unexpected_dest_prefixes = sorted(set(dest_prefix_counts) - {"C", "M"})

    status = "pass" if invalid_orig_prefix == 0 and not unexpected_dest_prefixes else "issue"
    return {
        "name": "account_id_prefix_convention",
        "status": status,
        "invalid_origin_prefix_count": invalid_orig_prefix,
        "destination_prefix_counts": dest_prefix_counts,
        "unexpected_destination_prefixes": unexpected_dest_prefixes,
        "detail": (
            "All origin accounts start with 'C' and all destination accounts "
            "start with 'C' or 'M', as expected by PaySim convention."
            if status == "pass"
            else "Account IDs found outside the expected 'C' (customer) / 'M' "
            "(merchant) prefix convention — genuine data-quality concern."
        ),
    }


def check_duplicate_transaction_id(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "name": "duplicate_transaction_id",
        "status": "not_applicable",
        "detail": (
            "Raw PaySim has no explicit transaction-id column, only "
            "step/account/amount fields. A true duplicate-transaction-id "
            "check cannot be meaningfully applied; exact full-row "
            "duplication is covered separately by 'duplicate_rows'."
        ),
    }


def run_validation(df: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    checks = [
        check_schema(df),
        check_row_count(df),
        check_duplicate_rows(df),
        check_transaction_types(df),
        check_nonnegative_fields(df),
        check_binary_flags(df),
        check_missing_values(df),
        check_balance_consistency(df),
        check_account_prefixes(df),
        check_duplicate_transaction_id(df),
    ]

    status_counts = {"pass": 0, "issue": 0, "expected_behavior": 0, "not_applicable": 0}
    for c in checks:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "status_summary": status_counts,
        "genuine_issues_found": status_counts["issue"] > 0,
        "checks": checks,
    }


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV via the existing ingestion loader (read-only)."""
    df = load_csv(csv_path)
    validate_columns(df)
    return df


def write_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate data quality of the raw PaySim CSV and write a JSON report."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    try:
        df = load_and_validate(args.csv)
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    report = run_validation(df, args.csv)
    write_report(report, args.out)

    print(f"Row count           : {report['row_count']:,}")
    print(f"Column count         : {report['column_count']}")
    print(f"Status summary       : {report['status_summary']}")
    print(f"Genuine issues found : {report['genuine_issues_found']}")
    for c in report["checks"]:
        print(f"  [{c['status']:>17}] {c['name']}: {c['detail']}")
    print(f"Report written to    : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
