"""
trade_logger.py

Utilities for logging executed trades and (later) generating weekly reviews.

Current responsibilities:
    - Append executed (or planned) trades to execution_log.csv

Planned extensions:
    - Generate weekly_trade_review.csv summaries
    - Aggregate P&L, win rate, avg R multiple, etc.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Mapping, Any

import pandas as pd


# Canonical column order for execution_log.csv
EXECUTION_LOG_COLUMNS = [
    "trade_id",
    "ticker",
    "direction",
    "entry_date",
    "exit_date",
    "holding_days",
    "entry_price",
    "exit_price",
    "shares",
    "position_size",
    "stop_price",
    "target_price",
    "pnl_dollars",
    "pnl_pct",
    "exit_reason",
    "r_multiple",
    "signal_score",
    "rsi_14",
    "ret_1w",
    "ret_3w",
    "atr_14",
    "atr_pct",
    "spy_trend",
    "vix_bucket",
    "breadth_pct_above_50ma",
    "notes",
]


def _normalize_trade_record(trade: Mapping[str, Any]) -> dict:
    """
    Normalize a single trade dict into a consistent format suitable for logging.

    Expected (but not all strictly required) keys:
        - trade_id (str, optional)
        - ticker (str)
        - direction (str: 'LONG' or 'SHORT')
        - entry_date (datetime or str)
        - exit_date (datetime or str, optional)
        - holding_days (int, optional)
        - entry_price (float)
        - exit_price (float, optional)
        - shares (int)
        - position_size (float, optional)
        - stop_price (float)
        - target_price (float, optional)
        - pnl_dollars (float, optional)
        - pnl_pct (float, optional)
        - exit_reason (str, optional)
        - r_multiple (float, optional)
        - signal_score (float, optional)
        - rsi_14 (float, optional)
        - ret_1w (float, optional)
        - ret_3w (float, optional)
        - atr_14 (float, optional)
        - atr_pct (float, optional)
        - spy_trend (str, optional)
        - vix_bucket (str, optional)
        - breadth_pct_above_50ma (float, optional)
        - notes (str, optional)
    """
    record = dict(trade)  # shallow copy

    # Support legacy 'date' key as entry_date
    if "entry_date" not in record and "date" in record:
        record["entry_date"] = record.pop("date")

    # Date normalization for entry_date
    date_val = record.get("entry_date", datetime.today())
    if isinstance(date_val, datetime):
        record["entry_date"] = date_val.strftime("%Y-%m-%d")
    else:
        record["entry_date"] = str(date_val)

    # Date normalization for exit_date
    exit_date_val = record.get("exit_date", "")
    if isinstance(exit_date_val, datetime):
        record["exit_date"] = exit_date_val.strftime("%Y-%m-%d")
    else:
        record["exit_date"] = str(exit_date_val) if exit_date_val else ""

    # Basic string fields
    record["ticker"] = str(record.get("ticker", "")).upper()
    record["direction"] = str(record.get("direction", "LONG")).upper()

    # Cast float fields if present
    for key in [
        "entry_price",
        "exit_price",
        "position_size",
        "stop_price",
        "target_price",
        "pnl_dollars",
        "pnl_pct",
        "r_multiple",
        "signal_score",
        "rsi_14",
        "ret_1w",
        "ret_3w",
        "atr_14",
        "atr_pct",
        "breadth_pct_above_50ma",
    ]:
        if key in record and record[key] is not None and record[key] != "":
            try:
                record[key] = float(record[key])
            except (TypeError, ValueError):
                record[key] = ""

    # Cast int fields if present
    for key in ["shares", "holding_days"]:
        if key in record and record[key] is not None and record[key] != "":
            try:
                record[key] = int(record[key])
            except (TypeError, ValueError):
                record[key] = ""

    # Ensure all columns exist, defaulting to empty string
    for col in EXECUTION_LOG_COLUMNS:
        record.setdefault(col, "")

    # Return only the canonical columns in order
    return {col: record[col] for col in EXECUTION_LOG_COLUMNS}


def append_execution_log(
    log_path: str,
    trades: Iterable[Mapping[str, Any]],
) -> None:
    """
    Append a list of trade dicts to execution_log.csv.

    Parameters
    ----------
    log_path : str
        Path to the CSV file, e.g. 'Results/execution_log.csv'.
    trades : Iterable[Mapping[str, Any]]
        Each trade is a dict with keys described in _normalize_trade_record().

    Behavior
    --------
    - Normalizes each trade
    - Appends to existing CSV if it exists, otherwise creates a new one
    - Ensures directory exists
    - Writes rows in canonical EXECUTION_LOG_COLUMNS order
    """
    trades = list(trades)
    if not trades:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    normalized_records = [_normalize_trade_record(t) for t in trades]
    new_df = pd.DataFrame(normalized_records, columns=EXECUTION_LOG_COLUMNS)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, dtype=str)
        # Align columns: add any missing columns with empty string
        for col in EXECUTION_LOG_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[EXECUTION_LOG_COLUMNS]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(log_path, index=False)


if __name__ == "__main__":
    # Simple manual test / example
    example_trade = {
        "date": datetime.today(),
        "ticker": "AAPL",
        "direction": "LONG",
        "entry_price": 190.50,
        "stop_price": 184.00,
        "target_price": 203.00,
        "shares": 10,
        "signal_score": 0.85,
        "rsi_14": 55.2,
        "ret_1w": 0.03,
        "ret_3w": 0.07,
        "notes": "TEST TRADE - safe to delete",
    }

    append_execution_log("Results/execution_log.csv", [example_trade])
    print("Appended example trade to Results/execution_log.csv")
