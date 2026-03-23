"""
regime_analysis.py

Regime breakdown analysis for the Stock Swing Bet Model backtest results.

Reads:
  - Results/backtest/backtest_trades.csv
  - Results/backtest/backtest_weekly_summary.csv
  - Data/backtest_cache/ohlcv_5yr.csv  (for SPY — downloads if missing)

Writes to Results/backtest/ only:
  - regime_breakdown.csv
  - yearly_breakdown.csv
  - regime_analysis_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RESULTS_DIR   = PROJECT_ROOT / "Results" / "backtest"
TRADES_CSV    = RESULTS_DIR / "backtest_trades.csv"
WEEKLY_CSV    = RESULTS_DIR / "backtest_weekly_summary.csv"
OHLCV_CACHE   = PROJECT_ROOT / "Data" / "backtest_cache" / "ohlcv_5yr.csv"

START_DATE = "2021-01-01"
END_DATE   = "2026-03-12"


# ─── SPY data ─────────────────────────────────────────────────────────────────

def load_spy() -> pd.DataFrame:
    """
    Load SPY daily OHLCV.  First check ohlcv_5yr.csv; if absent, download.
    Returns DataFrame with columns [date, open, high, low, close, volume].
    """
    # Try cache first
    if OHLCV_CACHE.exists():
        df = pd.read_csv(OHLCV_CACHE, parse_dates=["date"])
        spy = df[df["ticker"] == "SPY"].copy()
        if not spy.empty:
            return spy[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    # Not in cache — download just SPY
    print("SPY not found in cache; downloading from yfinance...")
    raw = yf.download(
        tickers="SPY",
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("Could not download SPY data.")

    raw = raw.reset_index()

    # Handle MultiIndex (yfinance >=0.2.37 returns (price_type, ticker) MultiIndex
    # even for single tickers in some versions)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [
            "_".join(str(v) for v in col if v and str(v) != "SPY").strip("_") or str(col[0])
            for col in raw.columns
        ]

    raw.columns = [str(c).strip() for c in raw.columns]
    raw.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Adj Close": "adj_close",
        "Volume": "volume",
    }, inplace=True)

    spy = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy.sort_values("date", inplace=True)
    print(f"  Downloaded {len(spy)} SPY rows.")
    return spy


# ─── Regime labeling ─────────────────────────────────────────────────────────

def build_regime_labels(spy: pd.DataFrame, week_starts: pd.Series) -> pd.DataFrame:
    """
    For each week_start date, compute SPY regime using data UP TO the prior
    trading day (no lookahead bias).

    Regime is one of:
      BULL_TREND     — above both MAs, weekly vol std < 0.02
      BULL_VOLATILE  — above both MAs, weekly vol std >= 0.02
      BEAR_TREND     — below either MA, weekly vol std < 0.02
      BEAR_VOLATILE  — below either MA, weekly vol std >= 0.02

    Returns DataFrame indexed by week_start with a 'regime' column.
    """
    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy.sort_values("date", inplace=True)
    spy.set_index("date", inplace=True)

    # Daily indicators (computed on full history; we'll slice as-of prior Friday)
    spy["sma_50"]  = spy["close"].rolling(50,  min_periods=50).mean()
    spy["sma_100"] = spy["close"].rolling(100, min_periods=100).mean()
    spy["daily_ret"] = spy["close"].pct_change()

    # Weekly returns on Friday closes
    spy_weekly = spy["close"].resample("W-FRI").last().pct_change()
    spy_vol4w  = spy_weekly.rolling(4, min_periods=2).std()

    # Map weekly vol back to daily (forward-fill from each Friday)
    spy["weekly_vol"] = spy_vol4w.reindex(spy.index, method="ffill")

    date_set   = set(spy.index)
    all_dates  = sorted(date_set)

    rows = []
    week_starts_ts = pd.to_datetime(week_starts).unique()

    for monday in week_starts_ts:
        # Find last trading day strictly before monday (prior Friday)
        prior = monday - pd.Timedelta(days=1)
        for _ in range(7):
            if prior in date_set:
                break
            prior -= pd.Timedelta(days=1)
        else:
            rows.append({"week_start": monday, "regime": "UNKNOWN"})
            continue

        row = spy.loc[prior]
        sma50  = row["sma_50"]
        sma100 = row["sma_100"]
        vol    = row["weekly_vol"]
        close  = row["close"]

        # Default if indicators not yet computable
        if pd.isna(sma50) or pd.isna(sma100):
            rows.append({"week_start": monday, "regime": "UNKNOWN"})
            continue

        above_both = (close > sma50) and (close > sma100)
        high_vol   = (not pd.isna(vol)) and (vol >= 0.02)

        if above_both and not high_vol:
            regime = "BULL_TREND"
        elif above_both and high_vol:
            regime = "BULL_VOLATILE"
        elif not above_both and not high_vol:
            regime = "BEAR_TREND"
        else:
            regime = "BEAR_VOLATILE"

        rows.append({"week_start": monday, "regime": regime})

    return pd.DataFrame(rows).set_index("week_start")


# ─── Stats helpers ────────────────────────────────────────────────────────────

def compute_trade_stats(df: pd.DataFrame) -> dict:
    """Compute the standard trade stats dict for a subset of trades."""
    n = len(df)
    if n == 0:
        return {
            "total_trades": 0, "win_rate": np.nan, "avg_r": np.nan,
            "profit_factor": np.nan, "avg_pnl_per_trade": np.nan,
            "total_pnl": 0.0, "target_hit_pct": np.nan,
            "stop_hit_pct": np.nan, "time_exit_pct": np.nan,
        }

    wins   = df[df["pnl_dollars"] > 0]
    losses = df[df["pnl_dollars"] <= 0]
    gross_w = wins["pnl_dollars"].sum()
    gross_l = losses["pnl_dollars"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else np.nan

    ec = df["exit_reason"].value_counts()
    return {
        "total_trades":      n,
        "win_rate":          round(len(wins) / n, 4),
        "avg_r":             round(float(df["r_multiple"].mean()), 4),
        "profit_factor":     round(float(pf), 4) if not np.isnan(pf) else np.nan,
        "avg_pnl_per_trade": round(float(df["pnl_dollars"].mean()), 2),
        "total_pnl":         round(float(df["pnl_dollars"].sum()), 2),
        "target_hit_pct":    round(ec.get("target", 0) / n, 4),
        "stop_hit_pct":      round(ec.get("stop",   0) / n, 4),
        "time_exit_pct":     round(ec.get("time",   0) / n, 4),
    }


# ─── Main analysis ──────────────────────────────────��─────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    trades = pd.read_csv(TRADES_CSV, parse_dates=["week_start", "entry_date", "exit_date"])
    weekly = pd.read_csv(WEEKLY_CSV, parse_dates=["week_start"])
    spy    = load_spy()

    print(f"Loaded {len(trades)} trades, {len(weekly)} weeks.")

    # Compute holding days
    trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days

    # ── Step 1: Build regime labels ───────────────────────────────────────────
    print("Building SPY regime labels...")
    regime_map = build_regime_labels(spy, trades["week_start"])

    # ── Step 2: Tag each trade ────────────────────────────────────────────────
    trades["regime"] = trades["week_start"].map(regime_map["regime"])
    trades["year"]   = trades["entry_date"].dt.year

    unknown_pct = (trades["regime"] == "UNKNOWN").mean() * 100
    if unknown_pct > 5:
        print(f"  Warning: {unknown_pct:.1f}% of trades have UNKNOWN regime (SPY MA not yet computable).")

    # ── Step 3: Regime breakdown ──────────────────────────────────────────────
    regime_order = ["BULL_TREND", "BULL_VOLATILE", "BEAR_TREND", "BEAR_VOLATILE", "UNKNOWN"]
    regime_rows  = []
    for regime in regime_order:
        sub = trades[trades["regime"] == regime]
        if sub.empty:
            continue
        stats = compute_trade_stats(sub)
        stats["regime"] = regime
        regime_rows.append(stats)

    regime_df = pd.DataFrame(regime_rows)[[
        "regime", "total_trades", "win_rate", "avg_r", "profit_factor",
        "avg_pnl_per_trade", "total_pnl", "target_hit_pct", "stop_hit_pct", "time_exit_pct",
    ]]
    regime_df.to_csv(RESULTS_DIR / "regime_breakdown.csv", index=False)

    # ── Step 4: Yearly breakdown ──────────────────────────────────────────────
    # Equity start/end per year from weekly summary
    weekly["year"] = weekly["week_start"].dt.year

    yearly_equity: dict[int, tuple[float, float]] = {}
    for yr, grp in weekly.groupby("year"):
        grp_s = grp.sort_values("week_start")
        # equity_start = equity at end of last week of prior year
        prior = weekly[weekly["year"] < yr]
        eq_start = float(prior.sort_values("week_start")["equity"].iloc[-1]) \
                   if not prior.empty else 10_000.0
        eq_end   = float(grp_s["equity"].iloc[-1])
        yearly_equity[int(yr)] = (eq_start, eq_end)

    yearly_rows = []
    for yr in sorted(trades["year"].unique()):
        sub   = trades[trades["year"] == yr]
        stats = compute_trade_stats(sub)
        eq_start, eq_end = yearly_equity.get(yr, (np.nan, np.nan))
        annual_ret = ((eq_end / eq_start) - 1.0) * 100.0 \
                     if (not np.isnan(eq_start) and eq_start > 0) else np.nan
        yearly_rows.append({
            "year":             yr,
            "total_trades":     stats["total_trades"],
            "win_rate":         stats["win_rate"],
            "avg_r":            stats["avg_r"],
            "profit_factor":    stats["profit_factor"],
            "total_pnl":        stats["total_pnl"],
            "equity_start":     round(eq_start, 2),
            "equity_end":       round(eq_end,   2),
            "annual_return_pct": round(annual_ret, 4) if not np.isnan(annual_ret) else np.nan,
        })

    yearly_df = pd.DataFrame(yearly_rows)[[
        "year", "total_trades", "win_rate", "avg_r", "profit_factor",
        "total_pnl", "equity_start", "equity_end", "annual_return_pct",
    ]]
    yearly_df.to_csv(RESULTS_DIR / "yearly_breakdown.csv", index=False)

    # ── Step 5: Exit reason analysis ──────────────────────────────────────────
    target_trades = trades[trades["exit_reason"] == "target"]
    stop_trades   = trades[trades["exit_reason"] == "stop"]
    time_trades   = trades[trades["exit_reason"] == "time"]

    time_wins   = time_trades[time_trades["pnl_dollars"] > 0]
    time_losses = time_trades[time_trades["pnl_dollars"] <= 0]
    time_win_pct  = len(time_wins)   / len(time_trades) * 100 if len(time_trades) > 0 else 0.0
    time_loss_pct = len(time_losses) / len(time_trades) * 100 if len(time_trades) > 0 else 0.0

    time_win_avg_r  = float(time_wins["r_multiple"].mean())   if not time_wins.empty   else np.nan
    time_loss_avg_r = float(time_losses["r_multiple"].mean()) if not time_losses.empty else np.nan

    # Implication sentence
    if not np.isnan(time_win_avg_r) and not np.isnan(time_loss_avg_r):
        avg_time_r = float(time_trades["r_multiple"].mean())
        if avg_time_r < -0.05:
            implication = "Holding longer could help — time exits are losing on average."
        elif avg_time_r > 0.05:
            implication = "Time exits are protecting capital — forced exit is net positive."
        else:
            implication = "Time exits are roughly break-even; holding longer unlikely to change results materially."
    else:
        implication = "Insufficient data to determine."

    # ── Build summary text ────────────────────────────────────────────────────
    def _pct(v: float) -> str:
        return f"{v*100:.1f}%" if not np.isnan(v) else "N/A"

    def _f(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if not np.isnan(v) else "N/A"

    # Regime table
    regime_header = (
        f"{'Regime':<16} {'Trades':>7} {'Win%':>7} {'Avg R':>7} "
        f"{'PF':>6} {'Avg PnL':>9} {'Tot PnL':>9} "
        f"{'Tgt%':>6} {'Stp%':>6} {'Time%':>6}"
    )
    regime_sep = "-" * len(regime_header)
    regime_lines = [regime_header, regime_sep]
    for _, row in regime_df.iterrows():
        regime_lines.append(
            f"{row['regime']:<16} {int(row['total_trades']):>7} "
            f"{_pct(row['win_rate']):>7} {_f(row['avg_r']):>7} "
            f"{_f(row['profit_factor']):>6} {_f(row['avg_pnl_per_trade']):>9} "
            f"{_f(row['total_pnl']):>9} "
            f"{_pct(row['target_hit_pct']):>6} {_pct(row['stop_hit_pct']):>6} "
            f"{_pct(row['time_exit_pct']):>6}"
        )

    # Yearly table
    yr_header = (
        f"{'Year':>5} {'Trades':>7} {'Win%':>7} {'Avg R':>7} "
        f"{'PF':>6} {'Tot PnL':>9} {'Eq Start':>10} {'Eq End':>10} {'Return%':>8}"
    )
    yr_sep   = "-" * len(yr_header)
    yr_lines = [yr_header, yr_sep]
    for _, row in yearly_df.iterrows():
        yr_lines.append(
            f"{int(row['year']):>5} {int(row['total_trades']):>7} "
            f"{_pct(row['win_rate']):>7} {_f(row['avg_r']):>7} "
            f"{_f(row['profit_factor']):>6} {_f(row['total_pnl']):>9} "
            f"${row['equity_start']:>9,.2f} ${row['equity_end']:>9,.2f} "
            f"{_f(row['annual_return_pct'], 1):>7}%"
        )

    # Best / worst regime (by profit factor, excluding UNKNOWN)
    valid_regime = regime_df[regime_df["regime"] != "UNKNOWN"].dropna(subset=["profit_factor"])
    if not valid_regime.empty:
        best_r  = valid_regime.loc[valid_regime["profit_factor"].idxmax()]
        worst_r = valid_regime.loc[valid_regime["profit_factor"].idxmin()]
        best_regime_line  = (
            f"Best regime:    {best_r['regime']}  "
            f"Win rate: {_pct(best_r['win_rate'])}  "
            f"Profit factor: {_f(best_r['profit_factor'])}"
        )
        worst_regime_line = (
            f"Worst regime:   {worst_r['regime']}  "
            f"Win rate: {_pct(worst_r['win_rate'])}  "
            f"Profit factor: {_f(worst_r['profit_factor'])}"
        )
    else:
        best_regime_line  = "Best regime:    N/A"
        worst_regime_line = "Worst regime:   N/A"

    # Best / worst year
    valid_yearly = yearly_df.dropna(subset=["annual_return_pct"])
    if not valid_yearly.empty:
        best_y  = valid_yearly.loc[valid_yearly["annual_return_pct"].idxmax()]
        worst_y = valid_yearly.loc[valid_yearly["annual_return_pct"].idxmin()]
        best_yr_line  = (
            f"Best year:      {int(best_y['year'])}  "
            f"Return: {_f(best_y['annual_return_pct'], 1)}%  "
            f"Trades: {int(best_y['total_trades'])}"
        )
        worst_yr_line = (
            f"Worst year:     {int(worst_y['year'])}  "
            f"Return: {_f(worst_y['annual_return_pct'], 1)}%  "
            f"Trades: {int(worst_y['total_trades'])}"
        )
    else:
        best_yr_line  = "Best year:      N/A"
        worst_yr_line = "Worst year:     N/A"

    summary_lines = [
        "=== REGIME ANALYSIS SUMMARY ===",
        "",
        "--- By Market Regime ---",
        *regime_lines,
        "",
        "--- By Calendar Year ---",
        *yr_lines,
        "",
        "--- Key Findings ---",
        best_regime_line,
        worst_regime_line,
        best_yr_line,
        worst_yr_line,
        "",
        "--- Time Exit Analysis ---",
        f"Time exits that were profitable:  {time_win_pct:.1f}%  (avg R: {_f(time_win_avg_r)})",
        f"Time exits that were losses:      {time_loss_pct:.1f}%  (avg R: {_f(time_loss_avg_r)})",
        f"Implication: {implication}",
    ]

    summary = "\n".join(summary_lines)

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "regime_analysis_summary.txt"
    summary_path.write_text(summary)

    print(f"\nSaved to {RESULTS_DIR}/")
    print(f"  regime_breakdown.csv        ({len(regime_df)} regimes)")
    print(f"  yearly_breakdown.csv         ({len(yearly_df)} years)")
    print(f"  regime_analysis_summary.txt")
    print()
    print(summary)


if __name__ == "__main__":
    main()
