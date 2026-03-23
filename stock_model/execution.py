

"""
execution.py

Helpers for going from model *candidates* to actual *executed trades*.

This module is intentionally simple and manual-first:
    - You run main.py to generate a daily candidates CSV.
    - You review the candidates, decide which ones you actually want to trade.
    - Then you can use the helpers here to:
        * Load the latest candidates file
        * Build trade records from the tickers you chose
        * Append them to execution_log.csv via trade_logger.append_execution_log

Nothing in here talks to a broker. It's just a thin layer between
"model output" and "what I actually did".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Any

import os
import re

import pandas as pd

from trade_logger import append_execution_log


# ------------------- Data structures -------------------


@dataclass
class ExecutedTradeInput:
    """
    Represents a single trade you actually took based on the model's candidates.

    Attributes
    ----------
    ticker : str
        The symbol you traded (must exist in the candidates CSV).
    entry_price : float | None
        Actual entry price. If None, we'll use the model's 'entry_price'
        from the candidates file.
    shares : int | None
        Actual number of shares. If None, we'll use the model's 'shares'
        from the candidates file.
    notes : str
        Any free-form notes about execution (slippage, partial fills, etc.).
    """
    ticker: str
    entry_price: float | None = None
    shares: int | None = None
    notes: str = ""


# ------------------- File helpers -------------------


def _extract_date_from_filename(path: Path) -> datetime | None:
    """
    Given a candidates_YYYY-MM-DD.csv path, extract the date.
    Returns None if no date pattern is found.
    """
    m = re.search(r"candidates_(\d{4}-\d{2}-\d{2})\.csv$", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None


def load_latest_candidates(candidates_dir: str | Path) -> tuple[Path | None, pd.DataFrame]:
    """
    Load the most recent candidates CSV from the given directory.

    Returns
    -------
    (path, df)
        path: Path to the latest candidates file, or None if none found
        df:   Loaded DataFrame (empty if none found)
    """
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.exists():
        return None, pd.DataFrame()

    files = list(candidates_dir.glob("candidates_*.csv"))
    if not files:
        return None, pd.DataFrame()

    dated_files = []
    for f in files:
        dt = _extract_date_from_filename(f)
        if dt is not None:
            dated_files.append((dt, f))

    if not dated_files:
        return None, pd.DataFrame()

    dated_files.sort(key=lambda x: x[0])
    latest_path = dated_files[-1][1]

    df = pd.read_csv(latest_path)
    return latest_path, df


# ------------------- Execution helpers -------------------


def build_trade_records_from_candidates(
    candidates_df: pd.DataFrame,
    executed_trades: Iterable[ExecutedTradeInput],
) -> list[dict]:
    """
    Construct trade dicts ready for logging from a candidates DataFrame
    and a list of ExecutedTradeInput objects.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        DataFrame produced by main.py (sized candidates).
    executed_trades : Iterable[ExecutedTradeInput]
        What you actually decided to trade.

    Returns
    -------
    list[dict]
        Trade records suitable for trade_logger.append_execution_log().
    """
    if candidates_df.empty:
        return []

    df = candidates_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # Index candidates by ticker for quick lookup
    idx = {t: i for i, t in enumerate(df["ticker"].tolist())}

    records: list[dict] = []
    today = datetime.today()

    for et in executed_trades:
        ticker = et.ticker.upper()
        if ticker not in idx:
            # Skip or raise; here we just skip and let you know via notes
            continue

        row = df.iloc[idx[ticker]]

        entry_price = float(et.entry_price) if et.entry_price is not None else float(row["entry_price"])
        shares = int(et.shares) if et.shares is not None else int(row["shares"])

        record = {
            "date": today,
            "ticker": ticker,
            "direction": row.get("direction", "LONG"),
            "entry_price": entry_price,
            "stop_price": float(row["stop_price"]),
            "target_price": float(row["target_price"]),
            "shares": shares,
            "signal_score": float(row.get("signal_score", 0.0)),
            "rsi_14": float(row.get("rsi_14", 0.0)),
            "ret_1w": float(row.get("ret_1w", 0.0)),
            "ret_3w": float(row.get("ret_3w", 0.0)),
            "notes": et.notes or "",
        }

        records.append(record)

    return records


def log_executed_trades(
    execution_log_path: str | Path,
    candidates_df: pd.DataFrame,
    executed_trades: Iterable[ExecutedTradeInput],
) -> None:
    """
    High-level helper to generate trade records from candidates + your actual
    executions and append them to execution_log.csv.
    """
    records = build_trade_records_from_candidates(candidates_df, executed_trades)
    if not records:
        return

    execution_log_path = str(execution_log_path)
    os.makedirs(os.path.dirname(execution_log_path), exist_ok=True)
    append_execution_log(execution_log_path, records)


if __name__ == "__main__":
    # Example usage (manual test)
    project_root = Path(__file__).resolve().parents[1]
    candidates_dir = project_root / "Results" / "candidates"
    exec_log_path = project_root / "Results" / "execution_log.csv"

    path, df_candidates = load_latest_candidates(candidates_dir)
    if path is None:
        print("No candidates files found.")
    else:
        print(f"Loaded latest candidates from: {path}")
        # Example: pretend we executed the first two tickers from the file
        example_trades = []
        for _, row in df_candidates.head(2).iterrows():
            example_trades.append(
                ExecutedTradeInput(
                    ticker=row["ticker"],
                    entry_price=None,  # use model entry
                    shares=None,       # use model shares
                    notes="SAMPLE EXECUTION FROM execution.py __main__",
                )
            )
        log_executed_trades(exec_log_path, df_candidates, example_trades)
        print(f"Logged {len(example_trades)} example trades to {exec_log_path}")