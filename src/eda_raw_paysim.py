"""Exploratory data analysis for the raw PaySim dataset (Milestone 4).

Reads only the confirmed raw file:
  data/raw/archive/paysim_orig/synthetic_financial_datasets_fraud_detection.csv

Never writes to, modifies, or deletes any dataset file. Produces compact,
machine-readable outputs under reports/.

Design notes on memory:
  - The bulk numeric/count aggregations (fraud-by-day, zero-balance
    patterns, isFlaggedFraud vs isFraud, account activity, amount/balance
    correlations) are computed with a single CHUNKED pass over the file
    (`run_chunked_pass`), using additive accumulators so no full-size
    DataFrame is ever held in memory — only one chunk at a time.
  - Exact amount-distribution quantiles (median/p90/p95/p99) cannot be
    computed additively across chunks without an approximation algorithm,
    and no such library is installed. Since only 3 of the 11 raw columns
    are actually needed for that analysis (`type`, `isFraud`, `amount`),
    `run_light_quantile_pass` loads just those three columns in a single
    non-chunked read — a small fraction of the full file's memory
    footprint — rather than adding algorithmic complexity for a dataset
    size that does not require it.

Because fraud is a ~0.13% minority class, every rate reported here is
accompanied by its underlying raw counts, and no interpretation should
treat overall accuracy as a meaningful summary statistic.

No feature engineering, graph construction, model training, sampling of
the raw dataset, or new package installation happens in this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest import DatasetNotFoundError, SchemaValidationError, load_csv, validate_columns
from src.validate_raw_paysim import DEFAULT_RAW_CSV

DEFAULT_CHUNKSIZE = 500_000
STEPS_PER_DAY = 24

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "eda_raw_paysim_summary.json"
TYPE_CSV_PATH = REPORTS_DIR / "eda_by_type.csv"
DAY_CSV_PATH = REPORTS_DIR / "eda_fraud_by_day.csv"
TEXT_SUMMARY_PATH = REPORTS_DIR / "eda_summary.txt"

CHUNK_COLUMNS = [
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

QUANTILE_COLUMNS = ["type", "isFraud", "amount"]

CORRELATION_PAIRS = [
    ("amount", "oldbalanceOrg"),
    ("amount", "newbalanceOrig"),
    ("amount", "oldbalanceDest"),
    ("amount", "newbalanceDest"),
    ("oldbalanceOrg", "newbalanceOrig"),
    ("oldbalanceDest", "newbalanceDest"),
]

QUANTILE_LEVELS = [0.5, 0.9, 0.95, 0.99]


def _empty_type_acc() -> dict[str, Any]:
    return {"count": 0, "fraud_count": 0}


def _empty_corr_acc() -> dict[str, float]:
    return {"n": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_xy": 0.0, "sum_x2": 0.0, "sum_y2": 0.0}


def _update_corr_acc(acc: dict[str, float], x: pd.Series, y: pd.Series) -> None:
    acc["n"] += int(len(x))
    acc["sum_x"] += float(x.sum())
    acc["sum_y"] += float(y.sum())
    acc["sum_xy"] += float((x * y).sum())
    acc["sum_x2"] += float((x * x).sum())
    acc["sum_y2"] += float((y * y).sum())


def _finalize_corr(acc: dict[str, float]) -> float | None:
    n = acc["n"]
    if n < 2:
        return None
    num = n * acc["sum_xy"] - acc["sum_x"] * acc["sum_y"]
    den_x = n * acc["sum_x2"] - acc["sum_x"] ** 2
    den_y = n * acc["sum_y2"] - acc["sum_y"] ** 2
    den = den_x * den_y
    if den <= 0:
        return None
    return float(num / np.sqrt(den))


def run_chunked_pass(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    """Single chunked pass computing every additive EDA metric.

    No full-size DataFrame is held at once — only one chunk of
    `chunksize` rows is in memory at a time, plus small running
    accumulators (dicts/Counters) that grow with cardinality, not with
    row count.
    """
    total_rows = 0
    counts_by_type: dict[str, dict[str, int]] = {}
    day_counts: dict[int, dict[str, int]] = {}
    origin_counts: dict[str, int] = {}
    dest_counts: dict[str, int] = {}

    origin_both_zero = 0
    origin_both_zero_fraud = 0
    dest_both_zero_nonmerchant = 0
    dest_both_zero_nonmerchant_fraud = 0
    merchant_dest_rows = 0
    merchant_dest_both_zero = 0

    flag_crosstab = {"0_0": 0, "0_1": 0, "1_0": 0, "1_1": 0}

    corr_acc = {pair: _empty_corr_acc() for pair in CORRELATION_PAIRS}

    for chunk in pd.read_csv(csv_path, usecols=CHUNK_COLUMNS, chunksize=chunksize):
        total_rows += len(chunk)

        for t, sub in chunk.groupby("type"):
            entry = counts_by_type.setdefault(t, _empty_type_acc())
            entry["count"] += len(sub)
            entry["fraud_count"] += int(sub["isFraud"].sum())

        day = ((chunk["step"] - 1) // STEPS_PER_DAY).astype(int)
        for d, sub in chunk.groupby(day):
            entry = day_counts.setdefault(int(d), _empty_type_acc())
            entry["count"] += len(sub)
            entry["fraud_count"] += int(sub["isFraud"].sum())

        for name, cnt in chunk["nameOrig"].value_counts().items():
            origin_counts[name] = origin_counts.get(name, 0) + int(cnt)
        for name, cnt in chunk["nameDest"].value_counts().items():
            dest_counts[name] = dest_counts.get(name, 0) + int(cnt)

        orig_zero_mask = (chunk["oldbalanceOrg"] == 0) & (chunk["newbalanceOrig"] == 0)
        origin_both_zero += int(orig_zero_mask.sum())
        origin_both_zero_fraud += int((orig_zero_mask & (chunk["isFraud"] == 1)).sum())

        is_merchant_dest = chunk["nameDest"].str.startswith("M")
        merchant_dest_rows += int(is_merchant_dest.sum())
        dest_zero_mask = (chunk["oldbalanceDest"] == 0) & (chunk["newbalanceDest"] == 0)
        merchant_dest_both_zero += int((is_merchant_dest & dest_zero_mask).sum())

        nonmerchant_zero_mask = (~is_merchant_dest) & dest_zero_mask
        dest_both_zero_nonmerchant += int(nonmerchant_zero_mask.sum())
        dest_both_zero_nonmerchant_fraud += int(
            (nonmerchant_zero_mask & (chunk["isFraud"] == 1)).sum()
        )

        for flag_val in (0, 1):
            for fraud_val in (0, 1):
                mask = (chunk["isFlaggedFraud"] == flag_val) & (chunk["isFraud"] == fraud_val)
                flag_crosstab[f"{flag_val}_{fraud_val}"] += int(mask.sum())

        for x_col, y_col in CORRELATION_PAIRS:
            _update_corr_acc(corr_acc[(x_col, y_col)], chunk[x_col], chunk[y_col])

    origin_values = np.array(list(origin_counts.values())) if origin_counts else np.array([0])
    dest_values = np.array(list(dest_counts.values())) if dest_counts else np.array([0])

    return {
        "total_transactions": total_rows,
        "counts_by_type": counts_by_type,
        "fraud_by_day": {
            str(d): v for d, v in sorted(day_counts.items())
        },
        "account_activity": {
            "unique_origin_accounts": len(origin_counts),
            "unique_destination_accounts": len(dest_counts),
            "origin_accounts_with_multiple_transactions": int((origin_values > 1).sum()),
            "destination_accounts_with_multiple_transactions": int((dest_values > 1).sum()),
            "origin_transactions_per_account": {
                "mean": float(origin_values.mean()),
                "median": float(np.median(origin_values)),
                "max": int(origin_values.max()),
            },
            "destination_transactions_per_account": {
                "mean": float(dest_values.mean()),
                "median": float(np.median(dest_values)),
                "max": int(dest_values.max()),
            },
        },
        "zero_balance_patterns": {
            "origin_both_zero_count": origin_both_zero,
            "origin_both_zero_fraud_count": origin_both_zero_fraud,
            "merchant_destination_rows": merchant_dest_rows,
            "merchant_destination_both_zero_count": merchant_dest_both_zero,
            "nonmerchant_destination_both_zero_count": dest_both_zero_nonmerchant,
            "nonmerchant_destination_both_zero_fraud_count": dest_both_zero_nonmerchant_fraud,
        },
        "flagged_vs_actual_fraud": flag_crosstab,
        "amount_balance_correlations": {
            f"{x}_vs_{y}": _finalize_corr(acc) for (x, y), acc in corr_acc.items()
        },
    }


def _describe_amount(series: pd.Series) -> dict[str, Any]:
    if len(series) == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p50": None, "p90": None, "p95": None, "p99": None, "max": None,
        }
    quantiles = series.quantile(QUANTILE_LEVELS)
    std = series.std()
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(std) if pd.notna(std) else None,
        "min": float(series.min()),
        "p50": float(quantiles.loc[0.5]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(series.max()),
    }


def run_light_quantile_pass(csv_path: Path) -> dict[str, Any]:
    """Exact amount-distribution stats using a minimal 3-column read.

    Loading only ['type', 'isFraud', 'amount'] (vs. all 11 raw columns)
    keeps this pass's memory footprint a fraction of a full-row load,
    while still allowing exact (non-approximated) quantiles.
    """
    df = pd.read_csv(csv_path, usecols=QUANTILE_COLUMNS)

    by_type = {
        str(t): _describe_amount(sub["amount"]) for t, sub in df.groupby("type")
    }
    by_fraud = {
        str(f): _describe_amount(sub["amount"]) for f, sub in df.groupby("isFraud")
    }
    by_type_and_fraud: dict[str, dict[str, Any]] = {}
    for (t, f), sub in df.groupby(["type", "isFraud"]):
        by_type_and_fraud.setdefault(str(t), {})[str(f)] = _describe_amount(sub["amount"])

    return {
        "amount_stats_overall": _describe_amount(df["amount"]),
        "amount_stats_by_type": by_type,
        "amount_stats_by_fraud_label": by_fraud,
        "amount_stats_by_type_and_fraud": by_type_and_fraud,
    }


def build_report(csv_path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> dict[str, Any]:
    # Validate the file matches the expected raw schema before analyzing it
    # (reuses ingest's loader on a throwaway first read; the real analysis
    # passes below re-read with usecols to avoid holding this copy).
    validate_columns(load_csv(csv_path))

    chunked = run_chunked_pass(csv_path, chunksize=chunksize)
    quantiles = run_light_quantile_pass(csv_path)

    counts_by_type = chunked["counts_by_type"]
    fraud_by_type_rate = {
        t: {
            "count": v["count"],
            "fraud_count": v["fraud_count"],
            "fraud_rate": (v["fraud_count"] / v["count"]) if v["count"] else None,
        }
        for t, v in counts_by_type.items()
    }

    fraud_by_day_rate = {
        d: {
            "total_count": v["count"],
            "fraud_count": v["fraud_count"],
            "fraud_rate": (v["fraud_count"] / v["count"]) if v["count"] else None,
        }
        for d, v in chunked["fraud_by_day"].items()
    }

    total = chunked["total_transactions"]
    total_fraud = sum(v["fraud_count"] for v in counts_by_type.values())

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(csv_path),
        "chunksize_used": chunksize,
        "total_transactions": total,
        "total_fraud_transactions": total_fraud,
        "overall_fraud_rate": (total_fraud / total) if total else None,
        "counts_by_type": {t: v["count"] for t, v in counts_by_type.items()},
        "fraud_by_type": fraud_by_type_rate,
        "fraud_by_day": fraud_by_day_rate,
        "account_activity": chunked["account_activity"],
        "zero_balance_patterns": chunked["zero_balance_patterns"],
        "flagged_vs_actual_fraud": chunked["flagged_vs_actual_fraud"],
        "amount_balance_correlations": chunked["amount_balance_correlations"],
        **quantiles,
    }
    report["classification_notes"] = _build_classification_notes(report)
    return report


def _build_classification_notes(report: dict[str, Any]) -> dict[str, list[str]]:
    descriptive: list[str] = [
        f"Total transactions: {report['total_transactions']:,}; "
        f"total fraud: {report['total_fraud_transactions']:,} "
        f"({report['overall_fraud_rate']:.6%}). Fraud is a severe minority "
        "class — raw counts are reported alongside every rate, and "
        "accuracy is not a meaningful metric on its own for this data.",
        "Fraud rate varies sharply by transaction type (see fraud_by_type) "
        "— this is a descriptive pattern in the raw labels, observed "
        "directly from the data.",
        "Account activity counts (unique origin/destination accounts, "
        "accounts with repeat transactions) are descriptive summaries of "
        "the raw account-id fields.",
    ]

    known_artifacts: list[str] = [
        "isFlaggedFraud triggers on only a tiny subset of actual fraud "
        "cases (see flagged_vs_actual_fraud) — PaySim's isFlaggedFraud is "
        "documented as a deliberately weak, simplistic control (a fixed "
        "amount threshold), not a competitive fraud signal.",
        "Merchant destination accounts (nameDest starting with 'M') "
        "report oldbalanceDest = newbalanceDest = 0 for effectively all "
        "such rows — PaySim does not simulate merchant balances by "
        "design (confirmed in Milestone 3).",
        "A large share of origin/destination balance pairs do not satisfy "
        "simple arithmetic consistency (newBalance = oldBalance +/- "
        "amount), including many rows where both balances are reported "
        "as exactly zero — this is a well-documented PaySim simulator "
        "characteristic (Milestone 3 balance_consistency check), not a "
        "data-quality defect introduced by this pipeline.",
    ]

    requires_modeling: list[str] = [
        "Correlations between amount and balance fields (see "
        "amount_balance_correlations) are purely descriptive linear "
        "associations; whether these fields (or nonlinear combinations of "
        "them) are actually predictive of fraud requires a trained model "
        "to verify, not EDA alone.",
        "Elevated fraud rates for TRANSFER/CASH_OUT versus other types "
        "are a strong candidate signal, but whether transaction type adds "
        "predictive value beyond amount/balance features (vs. being "
        "redundant with them) requires model evaluation.",
        "Any time-based pattern in fraud_by_day is based on a small "
        "number of simulated days and should not be treated as a "
        "seasonality finding without further modeling/validation.",
    ]

    return {
        "descriptive_observations": descriptive,
        "known_paysim_simulator_artifacts": known_artifacts,
        "findings_requiring_modeling_to_verify": requires_modeling,
    }


def write_json_report(report: dict[str, Any], out_path: Path = JSON_REPORT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def write_type_csv(report: dict[str, Any], out_path: Path = TYPE_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "type", "count", "fraud_count", "fraud_rate",
        "amount_mean", "amount_p50", "amount_p95", "amount_p99",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, fraud_info in sorted(report["fraud_by_type"].items()):
            stats = report["amount_stats_by_type"].get(t, {})
            writer.writerow({
                "type": t,
                "count": fraud_info["count"],
                "fraud_count": fraud_info["fraud_count"],
                "fraud_rate": fraud_info["fraud_rate"],
                "amount_mean": stats.get("mean"),
                "amount_p50": stats.get("p50"),
                "amount_p95": stats.get("p95"),
                "amount_p99": stats.get("p99"),
            })


def write_day_csv(report: dict[str, Any], out_path: Path = DAY_CSV_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["day_bucket", "total_count", "fraud_count", "fraud_rate"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d, info in sorted(report["fraud_by_day"].items(), key=lambda kv: int(kv[0])):
            writer.writerow({
                "day_bucket": d,
                "total_count": info["total_count"],
                "fraud_count": info["fraud_count"],
                "fraud_rate": info["fraud_rate"],
            })


def _ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    if max_value <= 0:
        return ""
    filled = int(round((value / max_value) * width))
    return "#" * filled


def write_text_summary(report: dict[str, Any], out_path: Path = TEXT_SUMMARY_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("FraudGraph AI — Raw PaySim EDA Summary (Milestone 4)")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Source: {report['source_file']}")
    lines.append("")
    lines.append(
        f"Total transactions: {report['total_transactions']:,} | "
        f"Total fraud: {report['total_fraud_transactions']:,} | "
        f"Overall fraud rate: {report['overall_fraud_rate']:.6%}"
    )
    lines.append("")
    lines.append(
        "NOTE: No image plots (matplotlib/seaborn) were generated — no "
        "plotting library is installed in this project's venv, and "
        "installing new packages is out of scope for this milestone. "
        "ASCII bar charts are provided below as a lightweight substitute."
    )
    lines.append("")

    lines.append("Transaction counts by type:")
    counts = report["counts_by_type"]
    max_count = max(counts.values()) if counts else 1
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {t:<10} {c:>10,}  {_ascii_bar(c, max_count)}")
    lines.append("")

    lines.append("Fraud rate by type:")
    fbt = report["fraud_by_type"]
    max_rate = max((v["fraud_rate"] or 0) for v in fbt.values()) if fbt else 1
    for t, v in sorted(fbt.items(), key=lambda kv: -(kv[1]["fraud_rate"] or 0)):
        rate = v["fraud_rate"] or 0
        lines.append(
            f"  {t:<10} {rate:>10.4%}  ({v['fraud_count']:,}/{v['count']:,})  "
            f"{_ascii_bar(rate, max_rate)}"
        )
    lines.append("")

    lines.append("--- Descriptive observations ---")
    for n in report["classification_notes"]["descriptive_observations"]:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("--- Known PaySim simulator artifacts (not data defects) ---")
    for n in report["classification_notes"]["known_paysim_simulator_artifacts"]:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("--- Findings requiring later modeling to verify ---")
    for n in report["classification_notes"]["findings_requiring_modeling_to_verify"]:
        lines.append(f"  - {n}")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis on the raw PaySim CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--json-out", type=Path, default=JSON_REPORT_PATH)
    parser.add_argument("--type-csv-out", type=Path, default=TYPE_CSV_PATH)
    parser.add_argument("--day-csv-out", type=Path, default=DAY_CSV_PATH)
    parser.add_argument("--text-out", type=Path, default=TEXT_SUMMARY_PATH)
    args = parser.parse_args(argv)

    try:
        report = build_report(args.csv, chunksize=args.chunksize)
    except (DatasetNotFoundError, SchemaValidationError) as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        return 1

    write_json_report(report, args.json_out)
    write_type_csv(report, args.type_csv_out)
    write_day_csv(report, args.day_csv_out)
    write_text_summary(report, args.text_out)

    print(f"Total transactions   : {report['total_transactions']:,}")
    print(f"Total fraud          : {report['total_fraud_transactions']:,}")
    print(f"Overall fraud rate   : {report['overall_fraud_rate']:.6%}")
    print(f"JSON report          : {args.json_out}")
    print(f"By-type CSV          : {args.type_csv_out}")
    print(f"By-day CSV           : {args.day_csv_out}")
    print(f"Text summary         : {args.text_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
