"""
backtest_mean_reversion.py

Standalone walk-forward weekly backtest for a mean reversion strategy
on the S&P 500 universe.

Strategy:
  - Long:  price is X% below MA20 (oversold extension — fade the drop)
  - Short: price is X% above MA20 (overbought extension — fade the rally)
  - Thresholds tested: 3%, 5%, 8%
  - Stop:   entry ± 1.5 × ATR14
  - Target: MA20 (reversion to mean)
  - Max 5 trades/week (5 most extended from MA20)
  - $35 risk per trade, $10,000 starting equity
  - 0.1% slippage per side
  - Mandatory Friday close (time stop)

Outputs to Results/backtest/mean_reversion/ — never modifies any other file.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Paths ────────────────────────────────────────────────────────────────────

# File lives at stock_model/backtests/mean_reversion/backtest_mean_reversion.py
# parents[2] => stock_model/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "backtest_cache"
CACHE_FILE   = CACHE_DIR / "ohlcv_5yr.csv"
RESULTS_DIR  = PROJECT_ROOT / "Results" / "backtest" / "mean_reversion"
MOM_RESULTS  = PROJECT_ROOT / "Results" / "backtest" / "momentum"


# ─── Backtest parameters ──────────────────────────────────────────────────────

START_DATE     = "2021-02-08"
END_DATE       = "2026-03-11"
RISK_PER_TRADE = 35.0
ACCOUNT_START  = 10_000.0
SLIPPAGE       = 0.001        # 0.1% per side
ATR_MULT       = 1.5
MAX_TRADES     = 5
THRESHOLDS     = [0.03, 0.05, 0.08]   # 3%, 5%, 8%

MIN_PRICE   = 5.0
MIN_AVG_VOL = 500_000

# Momentum V1 baseline (from Results/backtest/momentum/backtest_summary.txt)
MOM_TOTAL_TRADES  = 1185
MOM_WIN_RATE      = 48.3
MOM_PROFIT_FACTOR = 1.04
MOM_AVG_R         = 0.01
MOM_MAX_DD        = -9.9
MOM_TOTAL_RETURN  = 5.4


# ─── Indicator helpers ────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr       = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    """Load 5yr OHLCV from cache. Does not re-download."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_FILE}.\n"
            "Run stock_model/backtests/backtest_engine.py first to build the cache."
        )
    print(f"Loading cached data from {CACHE_FILE.name} ...")
    df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers.")
    return df


def load_spy(df_all: pd.DataFrame) -> pd.DataFrame:
    """Extract SPY from the OHLCV cache. Downloads only if absent."""
    spy = df_all[df_all["ticker"] == "SPY"].copy()
    if not spy.empty:
        return spy[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    print("SPY not found in cache; downloading from yfinance...")
    raw = yf.download(
        tickers="SPY", start="2021-01-01", end="2026-03-12",
        interval="1d", auto_adjust=False, progress=False,
    )
    if raw.empty:
        raise RuntimeError("Could not download SPY data.")
    raw = raw.reset_index()
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
    print(f"  Downloaded {len(spy)} SPY rows.")
    return spy.sort_values("date")


# ─── Indicator computation ────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling indicators for every ticker across the full date range.
    Adds: ma_20, atr_14, avg_vol_20
    """
    print("Computing indicators for all tickers...")
    frames: list[pd.DataFrame] = []

    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        g["ma_20"]      = g["close"].rolling(20, min_periods=20).mean()
        g["atr_14"]     = compute_atr(g)
        g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["date", "ticker"], inplace=True)
    print(f"  Done. {len(result):,} rows with indicators.")
    return result


# ─── SPY regime labels ────────────────────────────────────────────────────────

def build_spy_regime_map(spy: pd.DataFrame) -> dict:
    """
    Build {monday_timestamp: regime_str} for every Monday in the date range.
    Uses data strictly before Monday (no lookahead bias).

    Regimes: BULL_TREND, BULL_VOLATILE, BEAR_TREND, BEAR_VOLATILE, UNKNOWN
    """
    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy.sort_values("date", inplace=True)
    spy.set_index("date", inplace=True)

    spy["sma_50"]  = spy["close"].rolling(50,  min_periods=50).mean()
    spy["sma_100"] = spy["close"].rolling(100, min_periods=100).mean()

    spy_weekly        = spy["close"].resample("W-FRI").last().pct_change()
    spy_vol4w         = spy_weekly.rolling(4, min_periods=2).std()
    spy["weekly_vol"] = spy_vol4w.reindex(spy.index, method="ffill")

    date_set = set(spy.index)
    mondays  = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    result: dict = {}

    for monday in mondays:
        prior = monday - pd.Timedelta(days=1)
        found = False
        for _ in range(7):
            if prior in date_set:
                found = True
                break
            prior -= pd.Timedelta(days=1)

        if not found:
            result[monday] = "UNKNOWN"
            continue

        row    = spy.loc[prior]
        sma50  = row["sma_50"]
        sma100 = row["sma_100"]
        vol    = row["weekly_vol"]
        close  = row["close"]

        if pd.isna(sma50) or pd.isna(sma100):
            result[monday] = "UNKNOWN"
            continue

        above_both = (close > sma50) and (close > sma100)
        high_vol   = (not pd.isna(vol)) and (vol >= 0.02)

        if above_both and not high_vol:
            result[monday] = "BULL_TREND"
        elif above_both and high_vol:
            result[monday] = "BULL_VOLATILE"
        elif not above_both and not high_vol:
            result[monday] = "BEAR_TREND"
        else:
            result[monday] = "BEAR_VOLATILE"

    return result


# ─── Max drawdown ─────────────────────────────────────────────────────────────

def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    dd   = (equity_series - peak) / peak
    return float(dd.min() * 100.0)


# ─── Walk-forward backtest ────────────────────────────────────────────────────

def run_backtest_threshold(
    df: pd.DataFrame,
    threshold: float,
    regime_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward weekly backtest for a single MA20 extension threshold.

    Signal computed as of prior Friday (no lookahead).
    Entry: Monday open ± slippage.
    Exit:  stop / target (MA20) / Friday close.

    Returns (trades_df, weekly_summary_df).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build fast lookup: index by (date, ticker)
    df.set_index(["date", "ticker"], inplace=True)
    df.sort_index(inplace=True)

    date_set      = set(df.index.get_level_values("date").unique())
    mondays       = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    mondays       = [m for m in mondays if m <= pd.Timestamp(END_DATE)]
    thr_label     = int(round(threshold * 100))

    all_trades:   list[dict] = []
    weekly_rows:  list[dict] = []
    trades_so_far = 0
    total_weeks   = len(mondays)

    for week_num, monday in enumerate(mondays, start=1):

        if week_num % 50 == 0:
            print(f"  Week {week_num}/{total_weeks}... (trades so far: {trades_so_far})")

        regime = regime_map.get(monday, "UNKNOWN")

        # ── Find last trading day before Monday (prior Friday) ────────────────
        prior_day = monday - timedelta(days=3)   # start at Friday
        found_prior = False
        for _ in range(7):
            if prior_day in date_set:
                found_prior = True
                break
            prior_day -= timedelta(days=1)

        if not found_prior:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        # ── Signal snapshot as of prior Friday ───────────────────────────────
        try:
            signal_rows = df.loc[prior_day]
        except KeyError:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T

        signal_rows = signal_rows.copy()

        # Drop rows missing required indicators or failing price/volume filter
        valid = (
            signal_rows["ma_20"].notna()
            & signal_rows["atr_14"].notna()
            & signal_rows["avg_vol_20"].notna()
            & (signal_rows["close"] >= MIN_PRICE)
            & (signal_rows["avg_vol_20"] >= MIN_AVG_VOL)
            & (signal_rows["atr_14"] > 0)
            & (signal_rows["ma_20"] > 0)
        )
        signal_rows = signal_rows[valid]

        if signal_rows.empty:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        # ── Compute extension from MA20 ───────────────────────────────────────
        signal_rows["extension"] = (
            (signal_rows["close"] - signal_rows["ma_20"]) / signal_rows["ma_20"]
        )

        # Long: price is ≥ threshold% BELOW MA20  (extension <= -threshold)
        # Short: price is ≥ threshold% ABOVE MA20 (extension >= +threshold)
        long_cands  = signal_rows[signal_rows["extension"] <= -threshold].copy()
        short_cands = signal_rows[signal_rows["extension"] >=  threshold].copy()

        long_cands["abs_ext"]  = long_cands["extension"].abs()
        short_cands["abs_ext"] = short_cands["extension"].abs()
        long_cands["direction"]  = "long"
        short_cands["direction"] = "short"

        candidates = pd.concat([long_cands, short_cands])
        if candidates.empty:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        # Cap at MAX_TRADES — take 5 most extended from MA20
        top5 = candidates.nlargest(MAX_TRADES, "abs_ext")

        # ── Monday open: check data availability ──────────────────────────────
        if monday not in date_set:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        try:
            monday_data = df.loc[monday]
        except KeyError:
            _append_empty_week(weekly_rows, monday, regime)
            continue

        if isinstance(monday_data, pd.Series):
            monday_data = monday_data.to_frame().T

        # ── Collect Tue–Fri OHLCV for exit simulation ─────────────────────────
        week_ohlcv: dict = {}
        for d_offset in range(1, 5):
            d = monday + timedelta(days=d_offset)
            if d in date_set:
                try:
                    day_data = df.loc[d]
                    week_ohlcv[d] = (
                        day_data if not isinstance(day_data, pd.Series)
                        else day_data.to_frame().T
                    )
                except KeyError:
                    pass

        # ── Simulate each selected trade ──────────────────────────────────────
        week_pnl     = 0.0
        week_r       = 0.0
        week_winners = 0
        week_losers  = 0
        week_trades  = 0

        for ticker in top5.index:
            if ticker not in monday_data.index:
                continue

            mon_open = monday_data.loc[ticker, "open"]
            if pd.isna(mon_open) or mon_open <= 0:
                continue

            sig          = top5.loc[ticker]
            direction    = sig["direction"]
            atr_14       = float(sig["atr_14"])
            target_price = float(sig["ma_20"])   # revert to prior-Friday MA20

            if pd.isna(atr_14) or atr_14 <= 0:
                continue
            if pd.isna(target_price) or target_price <= 0:
                continue

            if direction == "long":
                entry_price = mon_open * (1.0 + SLIPPAGE)
                stop_price  = entry_price - ATR_MULT * atr_14
                risk_ps     = entry_price - stop_price
                # Target must be above entry (stock must still have room to revert up)
                if target_price <= entry_price:
                    continue
            else:  # short
                entry_price = mon_open * (1.0 - SLIPPAGE)
                stop_price  = entry_price + ATR_MULT * atr_14
                risk_ps     = stop_price - entry_price
                # Target must be below entry (stock must still have room to revert down)
                if target_price >= entry_price:
                    continue

            if risk_ps <= 0:
                continue

            shares = max(1, math.floor(RISK_PER_TRADE / risk_ps))

            # ── Walk Tue → Fri: check stop / target / time exit ───────────────
            exit_price  = None
            exit_date   = None
            exit_reason = None

            sorted_days = sorted(week_ohlcv.keys())
            for idx, day in enumerate(sorted_days):
                is_last = (idx == len(sorted_days) - 1)
                day_df  = week_ohlcv[day]

                if ticker not in day_df.index:
                    if is_last:
                        break
                    continue

                row       = day_df.loc[ticker]
                day_low   = float(row.get("low",   np.nan))
                day_high  = float(row.get("high",  np.nan))
                day_close = float(row.get("close", np.nan))

                if direction == "long":
                    if not np.isnan(day_low) and day_low <= stop_price:
                        exit_price  = stop_price * (1.0 - SLIPPAGE)
                        exit_date   = day
                        exit_reason = "stop"
                        break
                    elif not np.isnan(day_high) and day_high >= target_price:
                        exit_price  = target_price * (1.0 - SLIPPAGE)
                        exit_date   = day
                        exit_reason = "target"
                        break
                    elif is_last:
                        if not np.isnan(day_close):
                            exit_price  = day_close * (1.0 - SLIPPAGE)
                            exit_date   = day
                            exit_reason = "time"
                        break
                else:  # short
                    if not np.isnan(day_high) and day_high >= stop_price:
                        exit_price  = stop_price * (1.0 + SLIPPAGE)
                        exit_date   = day
                        exit_reason = "stop"
                        break
                    elif not np.isnan(day_low) and day_low <= target_price:
                        exit_price  = target_price * (1.0 + SLIPPAGE)
                        exit_date   = day
                        exit_reason = "target"
                        break
                    elif is_last:
                        if not np.isnan(day_close):
                            exit_price  = day_close * (1.0 + SLIPPAGE)
                            exit_date   = day
                            exit_reason = "time"
                        break

            if exit_price is None:
                continue

            if direction == "long":
                pnl        = (exit_price - entry_price) * shares
                r_multiple = (exit_price - entry_price) / risk_ps
            else:
                pnl        = (entry_price - exit_price) * shares
                r_multiple = (entry_price - exit_price) / risk_ps

            all_trades.append({
                "date":          monday.date(),
                "ticker":        ticker,
                "direction":     direction,
                "threshold_pct": thr_label,
                "entry_price":   round(entry_price,  4),
                "stop_price":    round(stop_price,   4),
                "target_price":  round(target_price, 4),
                "exit_price":    round(exit_price,   4),
                "exit_type":     exit_reason,
                "R_multiple":    round(r_multiple,   4),
                "pnl":           round(pnl,          2),
                "regime":        regime,
            })

            week_pnl      += pnl
            week_r        += r_multiple
            week_trades   += 1
            trades_so_far += 1
            if pnl > 0:
                week_winners += 1
            else:
                week_losers += 1

        weekly_rows.append({
            "week_start": monday.date(),
            "trades":     week_trades,
            "wins":       week_winners,
            "losses":     week_losers,
            "win_rate":   round(week_winners / week_trades, 4) if week_trades > 0 else np.nan,
            "avg_R":      round(week_r / week_trades, 4) if week_trades > 0 else np.nan,
            "weekly_pnl": round(week_pnl, 2),
            "regime":     regime,
        })

    return pd.DataFrame(all_trades), pd.DataFrame(weekly_rows)


def _append_empty_week(rows: list[dict], monday: pd.Timestamp, regime: str) -> None:
    rows.append({
        "week_start": monday.date(), "trades": 0, "wins": 0, "losses": 0,
        "win_rate": np.nan, "avg_R": np.nan, "weekly_pnl": 0.0, "regime": regime,
    })


# ─── Stats helpers ────────────────────────────────────────────────────────────

def compute_stats(trades: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """Compute standard stats dict for a set of trades."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "avg_r": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "total_return": 0.0,
            "target_pct": 0.0, "stop_pct": 0.0, "time_pct": 0.0, "total_pnl": 0.0,
        }

    wins    = int((trades["pnl"] > 0).sum())
    gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_l = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else float("inf")

    equity_curve = ACCOUNT_START + weekly["weekly_pnl"].cumsum()
    max_dd       = _max_drawdown(equity_curve)
    total_pnl    = float(trades["pnl"].sum())
    total_ret    = (total_pnl / ACCOUNT_START) * 100.0

    ec = trades["exit_type"].value_counts()
    return {
        "total_trades":  n,
        "win_rate":      round(wins / n * 100.0, 1),
        "avg_r":         round(float(trades["R_multiple"].mean()), 4),
        "profit_factor": round(float(pf), 2) if not math.isinf(pf) else 999.0,
        "max_dd":        round(max_dd, 1),
        "total_return":  round(total_ret, 1),
        "target_pct":    round(ec.get("target", 0) / n * 100.0, 1),
        "stop_pct":      round(ec.get("stop",   0) / n * 100.0, 1),
        "time_pct":      round(ec.get("time",   0) / n * 100.0, 1),
        "total_pnl":     round(total_pnl, 2),
    }


# ─── Summary builder ─────────────────────────────────────────────────────────

def build_summary(
    results: dict,           # {thr_int: (trades_df, weekly_df, stats_dict)}
    mom_weekly: pd.DataFrame,
) -> str:
    thresholds = [3, 5, 8]

    col_w = 16   # width of each threshold column

    lines = ["=== MEAN REVERSION BACKTEST SUMMARY ===", ""]

    # ── Results by threshold ──────────────────────────────────────────────────
    lines.append("--- Results by Threshold ---")
    hdr = f"{'Metric':<22}" + "".join(f"{'%d%% Extension' % t:>{col_w}}" for t in thresholds)
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    def _v(stats: dict, key: str, fmt: str) -> str:
        v = stats[key]
        if fmt == "d":
            return f"{v:>{col_w}d}"
        if fmt.endswith("%"):
            d = int(fmt[:-1]) if fmt[:-1].isdigit() else 1
            return f"{v:>{col_w - 1}.{d}f}%"
        # e.g. "4f" or "2f"
        d = int(fmt[:-1]) if fmt[:-1].isdigit() else 2
        return f"{v:>{col_w}.{d}f}"

    metric_rows = [
        ("Total trades",    "total_trades",  "d"),
        ("Win rate",        "win_rate",      "1%"),
        ("Avg R per trade", "avg_r",         "4f"),
        ("Profit factor",   "profit_factor", "2f"),
        ("Max drawdown",    "max_dd",        "1%"),
        ("Total return",    "total_return",  "1%"),
        ("Target hit %",    "target_pct",    "1%"),
        ("Stop hit %",      "stop_pct",      "1%"),
        ("Time exit %",     "time_pct",      "1%"),
    ]
    for label, key, fmt in metric_rows:
        row = f"{label:<22}" + "".join(_v(results[t][2], key, fmt) for t in thresholds)
        lines.append(row)

    lines.append("")

    # ── Best threshold (by profit factor) ────────────────────────────────────
    best_t = max(thresholds, key=lambda t: results[t][2]["profit_factor"])
    best_trades, best_weekly, best_stats = results[best_t]

    lines.append(f"--- Regime Breakdown (best threshold: {best_t}%) ---")
    rh = f"{'Regime':<16} {'Trades':>7} {'Win%':>7} {'Avg R':>7} {'PF':>7} {'Tot PnL':>10}"
    lines.append(rh)
    lines.append("-" * len(rh))

    regime_order = ["BULL_TREND", "BULL_VOLATILE", "BEAR_TREND", "BEAR_VOLATILE", "UNKNOWN"]
    for reg in regime_order:
        sub = best_trades[best_trades["regime"] == reg]
        if sub.empty:
            continue
        n     = len(sub)
        wins  = int((sub["pnl"] > 0).sum())
        avg_r = float(sub["R_multiple"].mean())
        gw    = sub.loc[sub["pnl"] > 0, "pnl"].sum()
        gl    = sub.loc[sub["pnl"] < 0, "pnl"].abs().sum()
        pf    = (gw / gl) if gl > 0 else float("inf")
        tot   = float(sub["pnl"].sum())
        wr    = wins / n * 100.0
        pf_s  = f"{pf:7.2f}" if not math.isinf(pf) else "   inf"
        lines.append(
            f"{reg:<16} {n:>7d} {wr:>6.1f}% {avg_r:>7.4f} {pf_s} ${tot:>9.2f}"
        )

    lines.append("")

    # ── Head to head ─────────────────────────────────────────────────────────
    lines.append("--- Head to Head: Best MR Threshold vs Momentum Baseline ---")
    h2h_hdr = f"{'Metric':<22} {'Momentum V1':>16} {'Mean Reversion (%d%%)' % best_t:>22}"
    lines.append(h2h_hdr)
    lines.append("-" * len(h2h_hdr))

    h2h_rows = [
        ("Total trades",   MOM_TOTAL_TRADES,  best_stats["total_trades"],  "d"),
        ("Win rate",       MOM_WIN_RATE,       best_stats["win_rate"],      "1%"),
        ("Profit factor",  MOM_PROFIT_FACTOR,  best_stats["profit_factor"], "2f"),
        ("Avg R",          MOM_AVG_R,          best_stats["avg_r"],         "4f"),
        ("Max drawdown",   MOM_MAX_DD,         best_stats["max_dd"],        "1%"),
        ("Total return",   MOM_TOTAL_RETURN,   best_stats["total_return"],  "1%"),
    ]

    def _h(v, fmt: str, width: int) -> str:
        if fmt == "d":
            return f"{int(v):>{width}d}"
        if fmt.endswith("%"):
            d = int(fmt[:-1]) if fmt[:-1].isdigit() else 1
            return f"{float(v):>{width - 1}.{d}f}%"
        d = int(fmt[:-1]) if fmt[:-1].isdigit() else 2
        return f"{float(v):>{width}.{d}f}"

    for label, mom_val, mr_val, fmt in h2h_rows:
        lines.append(f"{label:<22} {_h(mom_val, fmt, 16)} {_h(mr_val, fmt, 22)}")

    lines.append("")

    # ── Weekly return correlation ─────────────────────────────────────────────
    lines.append("--- Weekly Return Correlation ---")
    mom_pnl = mom_weekly[["week_start", "total_pnl"]].copy()
    mom_pnl["week_start"] = pd.to_datetime(mom_pnl["week_start"])
    mr_pnl  = best_weekly[["week_start", "weekly_pnl"]].copy()
    mr_pnl["week_start"] = pd.to_datetime(mr_pnl["week_start"])

    merged = mom_pnl.merge(mr_pnl, on="week_start", how="inner")
    if len(merged) >= 2:
        corr = merged["total_pnl"].corr(merged["weekly_pnl"])
        corr_s = f"{corr:.2f}"
        if abs(corr) >= 0.5:
            interp = "correlated"
        elif abs(corr) >= 0.2:
            interp = "weakly correlated"
        else:
            interp = "uncorrelated"
    else:
        corr_s = "N/A"
        interp = "insufficient data"

    lines.append(
        f"Correlation of weekly P&L between momentum and mean reversion: {corr_s}"
    )
    lines.append(f"Interpretation: {interp}")
    lines.append("")

    # ── Recommendation ────────────────────────────────────────────────────────
    lines.append("--- Recommendation ---")
    bpf  = best_stats["profit_factor"]
    bret = best_stats["total_return"]

    if bpf > MOM_PROFIT_FACTOR and bret > MOM_TOTAL_RETURN:
        rec = (
            f"The {best_t}% mean reversion threshold outperforms the momentum baseline on both "
            f"profit factor ({bpf:.2f} vs {MOM_PROFIT_FACTOR:.2f}) and total return "
            f"({bret:.1f}% vs {MOM_TOTAL_RETURN:.1f}%). "
        )
        if interp in ("uncorrelated", "weakly correlated"):
            rec += (
                f"Weekly returns are {interp}, so combining both strategies as a portfolio "
                f"overlay could reduce drawdown through diversification. "
                f"Consider allocating a small sleeve to mean reversion alongside momentum."
            )
        else:
            rec += (
                f"However, weekly returns are {interp}, limiting the diversification benefit "
                f"of running both simultaneously."
            )
    elif bpf >= 1.0:
        rec = (
            f"The {best_t}% threshold is the best tested, achieving a positive profit factor "
            f"({bpf:.2f}) and {bret:.1f}% total return — below the momentum baseline "
            f"({MOM_PROFIT_FACTOR:.2f} PF, {MOM_TOTAL_RETURN:.1f}% return). "
        )
        if interp in ("uncorrelated", "weakly correlated"):
            rec += (
                f"Given {interp} weekly returns, running both strategies simultaneously "
                f"could smooth the equity curve even if mean reversion underperforms alone."
            )
        else:
            rec += (
                f"Strategies are {interp}; blending is unlikely to provide significant "
                f"diversification benefit."
            )
    else:
        rec = (
            f"Mean reversion at the {best_t}% threshold (best tested) has a profit factor "
            f"below 1.0 ({bpf:.2f}), indicating the S&P 500 universe does not reliably "
            f"exhibit short-term mean reversion at weekly cadence under these parameters. "
            f"Consider wider thresholds, longer hold periods, or stricter universe filters "
            f"(e.g., require RSI < 30 for longs, RSI > 70 for shorts) before allocating capital."
        )

    lines.append(rec)

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load OHLCV from cache
    df_raw = load_ohlcv()
    spy    = load_spy(df_raw)

    # Step 2: Compute indicators
    df_ind = compute_all_indicators(df_raw)

    # Step 3: Build SPY regime map
    print("Building SPY regime labels...")
    regime_map = build_spy_regime_map(spy)

    # Step 4: Load momentum weekly summary for correlation
    mom_weekly_path = MOM_RESULTS / "backtest_weekly_summary.csv"
    if mom_weekly_path.exists():
        mom_weekly = pd.read_csv(mom_weekly_path, parse_dates=["week_start"])
    else:
        print(f"  Warning: momentum weekly summary not found at {mom_weekly_path}")
        mom_weekly = pd.DataFrame(columns=["week_start", "total_pnl"])

    # Step 5: Run backtest for each threshold
    results: dict = {}
    for threshold in THRESHOLDS:
        thr_label = int(round(threshold * 100))
        print(f"\nRunning mean reversion backtest — {thr_label}% threshold...")
        trades_df, weekly_df = run_backtest_threshold(df_ind, threshold, regime_map)
        stats = compute_stats(trades_df, weekly_df)
        results[thr_label] = (trades_df, weekly_df, stats)

        trades_path = RESULTS_DIR / f"mr_trades_{thr_label}.csv"
        weekly_path = RESULTS_DIR / f"mr_weekly_summary_{thr_label}.csv"
        trades_df.to_csv(trades_path, index=False)
        weekly_df.to_csv(weekly_path, index=False)
        print(f"  Saved {len(trades_df):>5} trades → {trades_path.name}")
        print(f"  Saved {len(weekly_df):>5} weeks  → {weekly_path.name}")

    # Step 6: Build and save summary
    print("\nBuilding summary...")
    summary = build_summary(results, mom_weekly)
    summary_path = RESULTS_DIR / "mr_summary.txt"
    summary_path.write_text(summary)
    print(f"  Saved → {summary_path.name}")
    print(f"\nAll outputs saved to {RESULTS_DIR}/\n")
    print(summary)


if __name__ == "__main__":
    main()
