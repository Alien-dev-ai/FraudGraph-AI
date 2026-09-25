"""PaySim ingestion skeleton for FraudGraph AI (Milestone 1).

Loads a PaySim-style transaction CSV, validates its schema, and reports
row/column counts, dtypes, missing values, and the fraud-label distribution.

Deliberately out of scope here: feature engineering, model training,
graph construction, and dataset sampling.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

FRAUD_LABEL_COLUMN = "isFraud"

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no candidate PaySim CSV can be located."""


class SchemaValidationError(ValueError):
    """Raised when a loaded CSV is missing required PaySim columns."""


@dataclass
class IngestionReport:
    source_path: Path
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    fraud_label_counts: dict[int, int] = field(default_factory=dict)
    fraud_label_rate: float | None = None

    def to_text(self) -> str:
        lines = [
            f"Source file        : {self.source_path}",
            f"Row count           : {self.row_count:,}",
            f"Column count         : {self.column_count}",
            f"Columns              : {', '.join(self.columns)}",
            "Dtypes:",
        ]
        for col, dtype in self.dtypes.items():
            lines.append(f"  - {col}: {dtype}")
        lines.append("Missing values:")
        any_missing = False
        for col, count in self.missing_values.items():
            if count > 0:
                any_missing = True
                lines.append(f"  - {col}: {count:,}")
        if not any_missing:
            lines.append("  - none")
        lines.append("Fraud label distribution (isFraud):")
        for label, count in sorted(self.fraud_label_counts.items()):
            lines.append(f"  - {label}: {count:,}")
        if self.fraud_label_rate is not None:
            lines.append(f"Fraud rate           : {self.fraud_label_rate:.6%}")
        return "\n".join(lines)


def find_dataset(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Locate a single PaySim-style CSV inside raw_dir.

    Raises DatasetNotFoundError if raw_dir is missing, empty of CSVs,
    or contains more than one CSV (ambiguous — caller should specify
    --csv explicitly in that case).
    """
    if not raw_dir.exists():
        raise DatasetNotFoundError(
            f"Raw data directory does not exist: {raw_dir}. "
            f"Create it and place a PaySim CSV inside, or pass --csv explicitly."
        )

    candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        raise DatasetNotFoundError(
            f"No CSV files found in {raw_dir}. "
            f"Download PaySim and place the CSV there, or pass --csv explicitly."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise DatasetNotFoundError(
            f"Multiple CSV files found in {raw_dir} ({names}). "
            f"Pass --csv to specify which one to ingest."
        )
    return candidates[0]


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise DatasetNotFoundError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise DatasetNotFoundError(f"Path is not a file: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise SchemaValidationError(f"CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise SchemaValidationError(f"CSV file could not be parsed: {csv_path} ({exc})") from exc

    if df.shape[0] == 0:
        raise SchemaValidationError(f"CSV file has no data rows: {csv_path}")

    return df


def validate_columns(df: pd.DataFrame, required: list[str] = REQUIRED_COLUMNS) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            "CSV is missing required PaySim columns: "
            f"{', '.join(missing)}. Found columns: {', '.join(df.columns)}"
        )


def build_report(df: pd.DataFrame, source_path: Path) -> IngestionReport:
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    fraud_label_counts: dict[int, int] = {}
    fraud_label_rate = None
    if FRAUD_LABEL_COLUMN in df.columns:
        counts = df[FRAUD_LABEL_COLUMN].value_counts(dropna=False)
        fraud_label_counts = {int(k): int(v) for k, v in counts.items()}
        total = int(counts.sum())
        fraud_count = int(counts.get(1, 0))
        fraud_label_rate = fraud_count / total if total else None

    return IngestionReport(
        source_path=source_path,
        row_count=df.shape[0],
        column_count=df.shape[1],
        columns=list(df.columns),
        dtypes=dtypes,
        missing_values=missing_values,
        fraud_label_counts=fraud_label_counts,
        fraud_label_rate=fraud_label_rate,
    )


def ingest(csv_path: Path | None = None, raw_dir: Path = DEFAULT_RAW_DIR) -> IngestionReport:
    """Locate, load, and validate a PaySim CSV, returning an IngestionReport."""
    resolved_path = csv_path if csv_path is not None else find_dataset(raw_dir)
    df = load_csv(resolved_path)
    validate_columns(df)
    return build_report(df, resolved_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest and validate a PaySim-style CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a specific CSV file. Defaults to searching data/raw/.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to search for a CSV when --csv is not given.",
    )
    args = parser.parse_args(argv)

    try:
        report = ingest(csv_path=args.csv, raw_dir=args.raw_dir)
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
