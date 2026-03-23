

"""
backtest.py

Lightweight backtesting and performance analysis utilities for the
Stock Swing Bet Model.

This module is designed to be *simple and incremental*:

    - Phase 1 (implemented here):
        Take a DataFrame of completed trades (e.g., from execution_log.csv
        once you add exit/close information) and compute:
            * per-trade P&L and return %
            * cumulative equity curve
            * high-level summary stats

    - Phase 2 (future work):
        Add a full signal-based historical backtest that:
            * walks through historical price data day by day
            * applies your signal rules
            * simulates entries, exits, and risk sizing

Right now this gives you a way to evaluate your actual trades
and any manually-created "what if" trade sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


# ------------------- Data structures -------------------


@dataclass
class BacktestConfig:
    """
    Configuration for evaluating completed trades.

    Attributes
    ----------
    initial_equity : float
        Starting account equity for the backtest.
    commission_per_share : float
        Flat commission per share (if you want to model costs).
        Set to 0.0 for commission-free.
    """
    initial_equity: float = 10_000.0
    commission_per_share: float = 0.0


@dataclass
class BacktestResult:
    """
    Container for backtest outputs.
    """
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: Dict[str, Any]


# ------------------- Core helpers -------------------


def _parse_date(value: Any) -> pd.Timestamp:
    """
    Parse a date-like value into a pandas Timestamp.
    """
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.to_datetime(value)
    return pd.to_datetime(str(value))


def compute_trade_metrics(
    trades: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame of trades with at least:
        - entry_date
        - exit_date
        - entry_price
        - exit_price
        - shares
        - direction ('LONG' or 'SHORT')
    compute:
        - gross_pnl
        - net_pnl (after commissions, if any)
        - return_pct

    Returns a new DataFrame with added columns.
    """
    if config is None:
        config = BacktestConfig()

    df = trades.copy()

    # Basic sanitation / typing
    required_cols = [
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "shares",
        "direction",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"compute_trade_metrics: missing required columns: {missing}")

    df["entry_date"] = df["entry_date"].apply(_parse_date)
    df["exit_date"] = df["exit_date"].apply(_parse_date)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").astype("Int64")
    df["direction"] = df["direction"].astype(str).str.upper()

    # Long/short P&L
    # For LONG:  (exit - entry) * shares
    # For SHORT: (entry - exit) * shares
    long_mask = df["direction"] == "LONG"
    short_mask = df["direction"] == "SHORT"

    df["gross_pnl"] = 0.0
    df.loc[long_mask, "gross_pnl"] = (
        (df.loc[long_mask, "exit_price"] - df.loc[long_mask, "entry_price"])
        * df.loc[long_mask, "shares"].astype(float)
    )
    df.loc[short_mask, "gross_pnl"] = (
        (df.loc[short_mask, "entry_price"] - df.loc[short_mask, "exit_price"])
        * df.loc[short_mask, "shares"].astype(float)
    )

    # Commissions
    if config.commission_per_share > 0:
        df["commission"] = (
            df["shares"].astype(float)
            * config.commission_per_share
            * 2.0  # entry + exit
        )
    else:
        df["commission"] = 0.0

    df["net_pnl"] = df["gross_pnl"] - df["commission"]

    # Return % relative to notional at entry
    notional = df["entry_price"] * df["shares"].astype(float)
    df["return_pct"] = df["net_pnl"] / notional.replace(0, pd.NA)

    return df


def build_equity_curve(
    trades_with_metrics: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Build an equity curve from a trades DataFrame that already has
    'net_pnl' and 'exit_date' columns.

    Returns a DataFrame with:
        - date
        - equity
        - daily_pnl
    """
    if config is None:
        config = BacktestConfig()

    df = trades_with_metrics.copy()
    if "net_pnl" not in df.columns or "exit_date" not in df.columns:
        raise ValueError("build_equity_curve: trades must have 'net_pnl' and 'exit_date' columns.")

    df["exit_date"] = df["exit_date"].apply(_parse_date)
    df = df.sort_values("exit_date")

    daily = df.groupby("exit_date")["net_pnl"].sum().reset_index()
    daily.rename(columns={"exit_date": "date", "net_pnl": "daily_pnl"}, inplace=True)

    daily["equity"] = config.initial_equity + daily["daily_pnl"].cumsum()

    return daily[["date", "equity", "daily_pnl"]]


def summarize_backtest(
    trades_with_metrics: pd.DataFrame,
    equity_curve: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, Any]:
    """
    Compute a few high-level summary statistics for the backtest.
    """
    if config is None:
        config = BacktestConfig()

    df = trades_with_metrics.copy()
    ec = equity_curve.copy()

    total_trades = len(df)
    winning_trades = int((df["net_pnl"] > 0).sum())
    losing_trades = int((df["net_pnl"] < 0).sum())

    total_net_pnl = float(df["net_pnl"].sum())
    final_equity = float(ec["equity"].iloc[-1]) if not ec.empty else config.initial_equity
    total_return_pct = (final_equity / config.initial_equity - 1.0) if config.initial_equity > 0 else 0.0

    avg_win = float(df.loc[df["net_pnl"] > 0, "net_pnl"].mean()) if winning_trades > 0 else 0.0
    avg_loss = float(df.loc[df["net_pnl"] < 0, "net_pnl"].mean()) if losing_trades > 0 else 0.0

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    summary: Dict[str, Any] = {
        "initial_equity": config.initial_equity,
        "final_equity": final_equity,
        "total_net_pnl": total_net_pnl,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    return summary


def run_backtest_from_trades(
    trades: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """
    High-level helper:
        1) Compute trade-level metrics
        2) Build equity curve
        3) Summarize

    This assumes `trades` already has entry/exit info (no signal logic here).
    """
    if config is None:
        config = BacktestConfig()

    trades_metrics = compute_trade_metrics(trades, config=config)
    equity_curve = build_equity_curve(trades_metrics, config=config)
    summary = summarize_backtest(trades_metrics, equity_curve, config=config)

    return BacktestResult(
        trades=trades_metrics,
        equity_curve=equity_curve,
        summary=summary,
    )


# ------------------- Convenience I/O helpers -------------------


def load_trades_from_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV of trades for backtesting.

    This is intentionally very generic. As long as the CSV has at least:
        - entry_date
        - exit_date
        - entry_price
        - exit_price
        - shares
        - direction
    you can use run_backtest_from_trades().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")
    return pd.read_csv(path)


def save_equity_curve_to_csv(
    equity_curve: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save the equity curve DataFrame to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    equity_curve.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage (requires a user-provided trades CSV with appropriate columns):
    example_trades_path = Path("Results") / "example_trades_for_backtest.csv"
    if example_trades_path.exists():
        trades_df = load_trades_from_csv(example_trades_path)
        cfg = BacktestConfig(initial_equity=10_000.0, commission_per_share=0.0)
        result = run_backtest_from_trades(trades_df, config=cfg)
        print("Backtest summary:")
        for k, v in result.summary.items():
            print(f"  {k}: {v}")
    else:
        print(
            "No example_trades_for_backtest.csv found in Results/. "
            "Create one with columns [entry_date, exit_date, entry_price, exit_price, shares, direction] "
            "to test backtest.py."
        )