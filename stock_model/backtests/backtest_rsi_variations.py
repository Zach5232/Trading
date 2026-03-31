"""
backtest_rsi_variations.py

Standalone RSI-filter sensitivity analysis against the V1 momentum baseline.

Tests six RSI-bound variations while keeping every other parameter identical
to backtest_engine.py V1:
  - Same scoring formula (40% rank_1w + 40% rank_3w + 20% rank_vol_surge)
  - Same ATR stop (1.5×), same 2R target, same Friday forced exit
  - Same $35 risk per trade, same 0.1% slippage, same $10,000 start equity
  - Same S&P 500 universe, same 5-year date range

Variation   Upper Cap   Lower Bound   Description
V1          RSI < 70    none          Current live model (baseline)
RSI-A       RSI < 60    none          Tighter overbought cap
RSI-B       RSI < 65    none          Moderate tighter cap
RSI-C       RSI < 75    none          Slightly looser cap
RSI-D       RSI < 70    RSI > 40      Add momentum floor
RSI-E       RSI < 70    RSI > 50      Stricter momentum floor
RSI-F       RSI < 65    RSI > 45      Combined tighter cap + floor

All input from local cache — no downloads unless SPY absent.
Outputs to Results/backtest/rsi_variations/ only.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Paths ────────────────────────────────────────────────────────────────────

# File lives at stock_model/backtests/backtest_rsi_variations.py
# parents[1] => stock_model/  (mirrors backtest_engine.py exactly)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "backtest_cache"
CACHE_FILE   = CACHE_DIR / "ohlcv_5yr.csv"
RESULTS_DIR  = PROJECT_ROOT / "Results" / "backtest" / "rsi_variations"


# ─── Backtest parameters — identical to backtest_engine.py ───────────────────

START_DATE     = "2021-01-01"
END_DATE       = "2026-03-12"
RISK_PER_TRADE = 35.0
ACCOUNT_START  = 10_000.0
SLIPPAGE       = 0.001        # 0.1% per side
ATR_MULT       = 1.5
REWARD_RISK    = 2.0
MAX_TRADES     = 5
MIN_ELIGIBLE   = 3

MIN_PRICE    = 5.0
MIN_AVG_VOL  = 500_000
ATR_PCT_LOW  = 0.005
ATR_PCT_HIGH = 0.06

# RSI bounds that never change (hard lower floor for all variations)
RSI_HARD_FLOOR = 30


# ─── Variation definitions ────────────────────────────────────────────────────

# Each entry: (label, rsi_low, rsi_high, description)
VARIATIONS: list[tuple[str, int, int, str]] = [
    ("V1",    30, 70, "Baseline — current live model"),
    ("RSI-A", 30, 60, "Tighter overbought cap"),
    ("RSI-B", 30, 65, "Moderate tighter cap"),
    ("RSI-C", 30, 75, "Slightly looser cap"),
    ("RSI-D", 40, 70, "Add momentum floor (RSI > 40)"),
    ("RSI-E", 50, 70, "Stricter momentum floor (RSI > 50)"),
    ("RSI-F", 45, 65, "Combined tighter cap + floor"),
]


# ─── Indicator helpers ────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr       = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    dd   = (equity_series - peak) / peak
    return float(dd.min() * 100.0)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    """Load 5yr OHLCV from cache. Does not re-download."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_FILE}.\n"
            "Run backtest_engine.py first to build the cache."
        )
    print(f"Loading cached OHLCV from {CACHE_FILE.name}...")
    df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers.")
    return df


def load_spy_data(df_all: pd.DataFrame) -> pd.DataFrame:
    """Load SPY daily data from the OHLCV cache, or download if absent."""
    spy = df_all[df_all["ticker"] == "SPY"].copy()
    if not spy.empty:
        return spy[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    print("SPY not found in cache; downloading from yfinance...")
    raw = yf.download(
        tickers="SPY", start=START_DATE, end=END_DATE,
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
    Mirrors backtest_engine.py compute_all_indicators exactly.
    """
    print("Computing indicators for all tickers...")
    frames: list[pd.DataFrame] = []

    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        g["ret_1w"]     = g["close"].pct_change(5,  fill_method=None)
        g["ret_3w"]     = g["close"].pct_change(15, fill_method=None)
        g["rsi_14"]     = compute_rsi(g["close"])
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
    Mirrors backtest_variations.py build_spy_regime_map exactly.

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


# ─── Walk-forward backtest (parameterized by RSI bounds) ─────────────────────

def run_variation(
    df: pd.DataFrame,
    label: str,
    rsi_low: int,
    rsi_high: int,
    regime_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward weekly backtest for one RSI-filter variation.
    Mirrors backtest_engine.py run_backtest exactly — only rsi_low/rsi_high differ.

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

    all_trades:   list[dict] = []
    weekly_rows:  list[dict] = []
    equity        = ACCOUNT_START
    trades_so_far = 0
    total_weeks   = len(mondays)

    for week_num, monday in enumerate(mondays, start=1):

        if week_num % 50 == 0:
            print(f"  Week {week_num}/{total_weeks}... (trades so far: {trades_so_far})")

        regime = regime_map.get(monday, "UNKNOWN")

        # ── Find last trading day before Monday (prior Friday) ────────────────
        prior_day   = monday - timedelta(days=3)   # start at Friday
        found_prior = False
        for _ in range(7):
            if prior_day in date_set:
                found_prior = True
                break
            prior_day -= timedelta(days=1)

        if not found_prior:
            _append_empty_week(weekly_rows, monday, equity, regime)
            continue

        # ── Signal snapshot as of prior Friday ───────────────────────────────
        try:
            signal_rows = df.loc[prior_day]
        except KeyError:
            _append_empty_week(weekly_rows, monday, equity, regime)
            continue

        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T

        signal_rows = signal_rows.copy()
        signal_rows["atr_pct"] = (
            signal_rows["atr_14"] / signal_rows["close"].replace(0, np.nan)
        )

        # ── Apply filters — only RSI bounds differ between variations ─────────
        cond = (
            (signal_rows["close"]      >= MIN_PRICE)
            & (signal_rows["avg_vol_20"] >= MIN_AVG_VOL)
            & (signal_rows["rsi_14"]     >= rsi_low)
            & (signal_rows["rsi_14"]     <= rsi_high)
            & (signal_rows["atr_pct"]    >= ATR_PCT_LOW)
            & (signal_rows["atr_pct"]    <= ATR_PCT_HIGH)
            & signal_rows["ret_1w"].notna()
            & signal_rows["ret_3w"].notna()
            & signal_rows["atr_14"].notna()
        )
        eligible = signal_rows[cond]

        if len(eligible) < MIN_ELIGIBLE:
            _append_empty_week(weekly_rows, monday, equity, regime)
            continue

        # ── Percentile rank scoring — identical to V1 ─────────────────────────
        eligible = eligible.copy()
        eligible["rank_1w"]       = eligible["ret_1w"].rank(pct=True)
        eligible["rank_3w"]       = eligible["ret_3w"].rank(pct=True)
        vol_ratio                  = eligible["volume"] / eligible["avg_vol_20"].replace(0, np.nan)
        eligible["rank_vol_surge"] = vol_ratio.fillna(0.0).rank(pct=True)
        eligible["signal_score"]   = (
            0.40 * eligible["rank_1w"]
            + 0.40 * eligible["rank_3w"]
            + 0.20 * eligible["rank_vol_surge"]
        )

        top5 = eligible[eligible["signal_score"] > 0].nlargest(MAX_TRADES, "signal_score")
        if top5.empty:
            _append_empty_week(weekly_rows, monday, equity, regime)
            continue

        # ── Monday open: check data availability ──────────────────────────────
        if monday not in date_set:
            _append_empty_week(weekly_rows, monday, equity, regime)
            continue

        try:
            monday_data = df.loc[monday]
        except KeyError:
            _append_empty_week(weekly_rows, monday, equity, regime)
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
        equity_start = equity

        for ticker in top5.index:
            if ticker not in monday_data.index:
                continue

            mon_open = monday_data.loc[ticker, "open"]
            if pd.isna(mon_open) or mon_open <= 0:
                continue

            atr_14 = float(top5.loc[ticker, "atr_14"])
            if pd.isna(atr_14) or atr_14 <= 0:
                continue

            entry_price  = mon_open * (1.0 + SLIPPAGE)
            stop_price   = entry_price - ATR_MULT * atr_14
            risk_ps      = entry_price - stop_price
            if risk_ps <= 0:
                continue

            shares       = max(1, math.floor(RISK_PER_TRADE / risk_ps))
            target_price = entry_price + REWARD_RISK * risk_ps

            # ── Walk Tue → Fri: stop / target / time exit ─────────────────────
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

            if exit_price is None:
                continue

            pnl        = (exit_price - entry_price) * shares
            r_multiple = (exit_price - entry_price) / risk_ps
            sig        = top5.loc[ticker]
            rsi_val    = float(sig["rsi_14"]) if pd.notna(sig.get("rsi_14")) else np.nan

            all_trades.append({
                "date":          monday.date(),
                "ticker":        ticker,
                "rsi_at_entry":  round(rsi_val, 4) if not np.isnan(rsi_val) else None,
                "upper_cap":     rsi_high,
                "lower_bound":   rsi_low,
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

        equity += week_pnl
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


def _append_empty_week(
    rows: list[dict], monday: pd.Timestamp, equity: float, regime: str
) -> None:
    rows.append({
        "week_start": monday.date(), "trades": 0, "wins": 0, "losses": 0,
        "win_rate": np.nan, "avg_R": np.nan, "weekly_pnl": 0.0, "regime": regime,
    })


# ─── Stats helpers ────────────────────────────────────────────────────────────

def compute_stats(trades: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """Compute standard stats dict for a completed variation run."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "avg_r": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "total_return": 0.0,
            "avg_rsi": np.nan, "total_pnl": 0.0,
        }

    wins    = int((trades["pnl"] > 0).sum())
    gross_w = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_l = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else float("inf")

    equity_curve = ACCOUNT_START + weekly["weekly_pnl"].cumsum()
    max_dd       = _max_drawdown(equity_curve)
    total_pnl    = float(trades["pnl"].sum())
    total_ret    = (total_pnl / ACCOUNT_START) * 100.0

    avg_rsi = float(trades["rsi_at_entry"].mean()) if "rsi_at_entry" in trades.columns else np.nan

    return {
        "total_trades":  n,
        "win_rate":      round(wins / n * 100.0, 1),
        "avg_r":         round(float(trades["R_multiple"].mean()), 4),
        "profit_factor": round(float(pf), 2) if not math.isinf(pf) else 999.0,
        "max_dd":        round(max_dd, 1),
        "total_return":  round(total_ret, 1),
        "avg_rsi":       round(avg_rsi, 1) if not np.isnan(avg_rsi) else np.nan,
        "total_pnl":     round(total_pnl, 2),
    }


# ─── Summary builder ─────────────────────────────────────────────────────────

def build_summary(
    variation_results: list[tuple[str, int, int, str, dict, pd.DataFrame]],
) -> str:
    """
    Build the full rsi_summary.txt content.

    variation_results: list of (label, rsi_low, rsi_high, desc, stats, trades_df)
    """
    lines = ["=== RSI FILTER VARIATIONS SUMMARY ===", ""]

    # ── Head to head table ────────────────────────────────────────────────────
    labels = [r[0] for r in variation_results]
    col_w  = 9   # width per variation column

    lines.append("--- Head to Head ---")

    hdr = f"{'Metric':<20}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    def _fmt(v, fmt: str) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'N/A':>{col_w}}"
        if fmt == "d":
            return f"{int(v):>{col_w}d}"
        if fmt.endswith("%"):
            d = int(fmt[:-1]) if fmt[:-1].isdigit() else 1
            return f"{float(v):>{col_w - 1}.{d}f}%"
        d = int(fmt[:-1]) if fmt[:-1].isdigit() else 2
        return f"{float(v):>{col_w}.{d}f}"

    metric_rows = [
        ("Total trades",    "total_trades",  "d"),
        ("Win rate",        "win_rate",       "1%"),
        ("Profit factor",   "profit_factor",  "2f"),
        ("Avg R",           "avg_r",          "4f"),
        ("Max drawdown",    "max_dd",         "1%"),
        ("Total return",    "total_return",   "1%"),
        ("Avg RSI at entry","avg_rsi",         "1f"),
    ]
    for label, key, fmt in metric_rows:
        row = f"{label:<20}" + "".join(_fmt(r[4][key], fmt) for r in variation_results)
        lines.append(row)

    lines.append("")

    # ── Best variation (by profit factor) ─────────────────────────────────────
    best = max(variation_results, key=lambda r: r[4]["profit_factor"])
    best_label, _, _, _, best_stats, best_trades = best

    lines.append(f"--- Regime Breakdown for best variation: {best_label} ---")
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
        pf_s  = f"{pf:7.2f}" if not math.isinf(pf) else "    inf"
        lines.append(
            f"{reg:<16} {n:>7d} {wr:>6.1f}% {avg_r:>7.4f} {pf_s} ${tot:>9.2f}"
        )

    lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    lines.append("--- Key Findings ---")

    best_pf  = max(variation_results, key=lambda r: r[4]["profit_factor"])
    best_wr  = max(variation_results, key=lambda r: r[4]["win_rate"])
    best_mdd = max(variation_results, key=lambda r: r[4]["max_dd"])  # least negative

    lines.append(
        f"Best variation by profit factor:  {best_pf[0]}  "
        f"(PF {best_pf[4]['profit_factor']:.2f}, "
        f"{best_pf[4]['total_trades']} trades, "
        f"{best_pf[4]['win_rate']:.1f}% win rate)"
    )
    lines.append(
        f"Best variation by win rate:       {best_wr[0]}  "
        f"({best_wr[4]['win_rate']:.1f}%, "
        f"PF {best_wr[4]['profit_factor']:.2f})"
    )
    lines.append(
        f"Best variation by max drawdown:   {best_mdd[0]}  "
        f"(MDD {best_mdd[4]['max_dd']:.1f}%)"
    )

    # Trade count impact: fewest vs most trades
    by_count  = sorted(variation_results, key=lambda r: r[4]["total_trades"])
    fewest    = by_count[0]
    most      = by_count[-1]
    diff      = most[4]["total_trades"] - fewest[4]["total_trades"]
    diff_pct  = diff / most[4]["total_trades"] * 100.0 if most[4]["total_trades"] > 0 else 0.0
    lines.append(
        f"Trade count — tightest filter ({fewest[0]}): {fewest[4]['total_trades']}  "
        f"vs loosest ({most[0]}): {most[4]['total_trades']}  "
        f"({diff} trade difference, {diff_pct:.1f}% reduction)"
    )

    lines.append("")

    # ── Recommendation ────────────────────────────────────────────────────────
    lines.append("--- Recommendation ---")

    # Get V1 stats for reference
    v1 = next(r for r in variation_results if r[0] == "V1")
    v1_stats = v1[4]

    # Check how spread out the profit factors are across all variations
    pf_values = [r[4]["profit_factor"] for r in variation_results]
    pf_spread = max(pf_values) - min(pf_values)
    wr_values = [r[4]["win_rate"] for r in variation_results]
    wr_spread = max(wr_values) - min(wr_values)

    # Assess RSI sensitivity
    if pf_spread < 0.05 and wr_spread < 2.0:
        sensitivity = "insensitive"
        sensitivity_detail = (
            f"Profit factor range is only {pf_spread:.2f} across all variations "
            f"and win rate spread is {wr_spread:.1f}pp — the model is effectively "
            f"RSI-insensitive within the 60–75 upper cap range tested."
        )
    elif pf_spread < 0.10:
        sensitivity = "weakly sensitive"
        sensitivity_detail = (
            f"Profit factor range is {pf_spread:.2f} and win rate spread is "
            f"{wr_spread:.1f}pp — the model shows weak RSI sensitivity."
        )
    else:
        sensitivity = "sensitive"
        sensitivity_detail = (
            f"Profit factor range is {pf_spread:.2f} and win rate spread is "
            f"{wr_spread:.1f}pp — RSI filter tuning has a material impact."
        )

    # Assess best variation vs V1
    best_beats_v1 = (
        best_pf[4]["profit_factor"] > v1_stats["profit_factor"]
        and best_pf[0] != "V1"
    )

    if best_beats_v1:
        rec = (
            f"The model is {sensitivity} to RSI filter changes. {sensitivity_detail} "
            f"{best_pf[0]} ({best_pf[3]}) beats V1 on profit factor "
            f"({best_pf[4]['profit_factor']:.2f} vs {v1_stats['profit_factor']:.2f}) "
            f"with {best_pf[4]['total_trades']} trades "
            f"({best_pf[4]['total_trades'] - v1_stats['total_trades']:+d} vs V1). "
        )
        if best_pf[4]["max_dd"] >= v1_stats["max_dd"]:
            rec += (
                f"Drawdown also improves ({best_pf[4]['max_dd']:.1f}% vs "
                f"{v1_stats['max_dd']:.1f}%). Consider adopting {best_pf[0]} "
                f"as the new filter with a forward test before going live."
            )
        else:
            rec += (
                f"However, drawdown is slightly worse ({best_pf[4]['max_dd']:.1f}% vs "
                f"{v1_stats['max_dd']:.1f}%). The improvement is marginal — "
                f"V1 filter remains defensible."
            )
    else:
        rec = (
            f"The model is {sensitivity} to RSI filter changes. {sensitivity_detail} "
            f"No variation beats V1 on profit factor. "
            f"The current RSI 30–70 filter is well-positioned: tightening it "
            f"reduces trade count without improving edge, and loosening it "
            f"adds noise. No RSI filter change is recommended at this time."
        )

    lines.append(rec)

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load OHLCV from cache
    df_raw = load_ohlcv()
    spy    = load_spy_data(df_raw)

    # Step 2: Compute indicators (once — shared across all variations)
    df_ind = compute_all_indicators(df_raw)

    # Step 3: Build SPY regime map (once — shared across all variations)
    print("Building SPY regime labels...")
    regime_map = build_spy_regime_map(spy)

    # Step 4: Run each variation
    variation_results: list[tuple] = []   # (label, rsi_low, rsi_high, desc, stats, trades_df)

    for label, rsi_low, rsi_high, desc in VARIATIONS:
        print(f"\nRunning {label} ({desc})  [RSI {rsi_low}–{rsi_high}]...")
        trades_df, weekly_df = run_variation(df_ind, label, rsi_low, rsi_high, regime_map)
        stats = compute_stats(trades_df, weekly_df)

        # Save per-variation CSVs
        trades_path = RESULTS_DIR / f"rsi_variation_trades_{label.replace('-', '_').lower()}.csv"
        weekly_path = RESULTS_DIR / f"rsi_variation_weekly_{label.replace('-', '_').lower()}.csv"
        trades_df.to_csv(trades_path, index=False)
        weekly_df.to_csv(weekly_path, index=False)
        print(f"  Saved {len(trades_df):>5} trades → {trades_path.name}")
        print(f"  Saved {len(weekly_df):>5} weeks  → {weekly_path.name}")

        variation_results.append((label, rsi_low, rsi_high, desc, stats, trades_df))

    # Step 5: Build and save summary
    print("\nBuilding summary...")
    summary = build_summary(variation_results)
    summary_path = RESULTS_DIR / "rsi_summary.txt"
    summary_path.write_text(summary)
    print(f"  Saved → {summary_path.name}")
    print(f"\nAll outputs saved to {RESULTS_DIR}/\n")
    print(summary)


if __name__ == "__main__":
    main()
