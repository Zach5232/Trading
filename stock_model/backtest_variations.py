"""
backtest_variations.py

Two targeted variations against the Version 1 baseline backtest.

Variation A — Regime Filter:
  Skip weeks when SPY is in BULL_TREND or BULL_VOLATILE. Trade only
  BEAR_VOLATILE, BEAR_TREND, and UNKNOWN weeks.

Variation B — 2-Week Hold For Winners:
  Same entry as V1. On week-1 Friday, if unrealized gain > 3%, carry
  the position into week 2 with stop moved to breakeven. Max hold = 2 weeks.

All input data read from local cache — no new downloads (except SPY if absent).
All output written to Results/backtest/ only.
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE   = PROJECT_ROOT / "Data" / "backtest_cache" / "ohlcv_5yr.csv"
RESULTS_DIR  = PROJECT_ROOT / "Results" / "backtest"

START_DATE = "2021-01-01"
END_DATE   = "2026-03-12"


# ─── Backtest parameters (same as V1) ─────────────────────────────────────────

RISK_PER_TRADE = 35.0
ACCOUNT_START  = 10_000.0
SLIPPAGE       = 0.001
ATR_MULT       = 1.5
REWARD_RISK    = 2.0
MAX_TRADES     = 5
MIN_ELIGIBLE   = 3

MIN_PRICE    = 5.0
MIN_AVG_VOL  = 500_000
RSI_LOW      = 30
RSI_HIGH     = 70
ATR_PCT_LOW  = 0.005
ATR_PCT_HIGH = 0.06

# V1 baseline stats for comparison display
V1_WIN_RATE       = 48.3
V1_PROFIT_FACTOR  = 1.04
V1_AVG_R          = 0.01
V1_MAX_DD         = -9.9
V1_TOTAL_RETURN   = 5.4
V1_TOTAL_TRADES   = 1185
V1_WEEKS_TRADED   = 237
V1_TOTAL_WEEKS    = 271
V1_FINAL_EQUITY   = 10_535.38


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    """Load 5yr OHLCV from cache (must already exist from backtest_engine.py)."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_FILE}. Run backtest_engine.py first."
        )
    print(f"Loading cached OHLCV from {CACHE_FILE.name}...")
    df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers.")
    return df


def load_spy_data() -> pd.DataFrame:
    """Load SPY daily data from cache, or download if not present."""
    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        spy = df[df["ticker"] == "SPY"].copy()
        if not spy.empty:
            return spy[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    print("SPY not in cache; downloading...")
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
    return spy.sort_values("date")


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling indicators per ticker across full date range."""
    print("Computing indicators...")
    frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        g["ret_1w"]     = g["close"].pct_change(5,  fill_method=None)
        g["ret_3w"]     = g["close"].pct_change(15, fill_method=None)
        g["rsi_14"]     = _compute_rsi(g["close"])
        g["atr_14"]     = _compute_atr(g)
        g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()
        frames.append(g)
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["date", "ticker"], inplace=True)
    print(f"  Done. {len(result):,} rows.")
    return result


# ─── SPY regime labels ────────────────────────────────────────────────────────

def build_spy_regime_map(spy: pd.DataFrame) -> dict:
    """
    Build {monday_timestamp: regime_str} for every Monday in the date range.
    Uses data strictly before Monday (no lookahead).

    Regimes: BULL_TREND, BULL_VOLATILE, BEAR_TREND, BEAR_VOLATILE, UNKNOWN
    """
    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy.sort_values("date", inplace=True)
    spy.set_index("date", inplace=True)

    spy["sma_50"]    = spy["close"].rolling(50,  min_periods=50).mean()
    spy["sma_100"]   = spy["close"].rolling(100, min_periods=100).mean()

    spy_weekly       = spy["close"].resample("W-FRI").last().pct_change()
    spy_vol4w        = spy_weekly.rolling(4, min_periods=2).std()
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

        row     = spy.loc[prior]
        sma50   = row["sma_50"]
        sma100  = row["sma_100"]
        vol     = row["weekly_vol"]
        close   = row["close"]

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


# ─── Shared signal helpers ────────────────────────────────────────────────────

def _get_prior_day(monday: pd.Timestamp, date_set: set) -> pd.Timestamp | None:
    """Return the last trading day strictly before monday, or None."""
    prior = monday - timedelta(days=3)  # start at Friday
    for _ in range(7):
        if prior in date_set:
            return prior
        prior -= timedelta(days=1)
    return None


def _get_top5(df_indexed: pd.DataFrame, prior_day: pd.Timestamp):
    """
    Compute signal snapshot as of prior_day and return top-5 eligible tickers.
    Returns None if not enough eligible tickers.
    """
    try:
        signal_rows = df_indexed.loc[prior_day]
    except KeyError:
        return None

    if isinstance(signal_rows, pd.Series):
        signal_rows = signal_rows.to_frame().T

    signal_rows = signal_rows.copy()
    signal_rows["atr_pct"] = signal_rows["atr_14"] / signal_rows["close"].replace(0, np.nan)

    cond = (
        (signal_rows["close"]      >= MIN_PRICE)
        & (signal_rows["avg_vol_20"] >= MIN_AVG_VOL)
        & (signal_rows["rsi_14"]     >= RSI_LOW)
        & (signal_rows["rsi_14"]     <= RSI_HIGH)
        & (signal_rows["atr_pct"]    >= ATR_PCT_LOW)
        & (signal_rows["atr_pct"]    <= ATR_PCT_HIGH)
        & signal_rows["ret_1w"].notna()
        & signal_rows["ret_3w"].notna()
        & signal_rows["atr_14"].notna()
    )
    eligible = signal_rows[cond]

    if len(eligible) < MIN_ELIGIBLE:
        return None

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
    return top5 if not top5.empty else None


def _collect_week_ohlcv(df_indexed: pd.DataFrame, monday: pd.Timestamp,
                         date_set: set, include_monday: bool = False) -> dict:
    """Return {date: day_df} for available trading days in the week."""
    start_offset = 0 if include_monday else 1
    days = [monday + timedelta(days=d) for d in range(start_offset, 5)]
    result = {}
    for d in days:
        if d in date_set:
            try:
                day_data = df_indexed.loc[d]
                result[d] = day_data if not isinstance(day_data, pd.Series) \
                                     else day_data.to_frame().T
            except KeyError:
                pass
    return result


def _simulate_exit(ticker: str, week_ohlcv: dict,
                   stop_price: float, target_price: float
                   ) -> tuple[float | None, pd.Timestamp | None, str | None]:
    """
    Walk through available days in week_ohlcv and return (exit_price, exit_date, exit_reason).
    Returns (None, None, None) if ticker data missing on all days.
    """
    sorted_days = sorted(week_ohlcv.keys())
    for idx, day in enumerate(sorted_days):
        is_last = (idx == len(sorted_days) - 1)
        day_df  = week_ohlcv[day]

        if ticker not in day_df.index:
            if is_last:
                return None, None, None
            continue

        row       = day_df.loc[ticker]
        day_low   = float(row.get("low",   np.nan))
        day_high  = float(row.get("high",  np.nan))
        day_close = float(row.get("close", np.nan))

        if not np.isnan(day_low) and day_low <= stop_price:
            return stop_price * (1.0 - SLIPPAGE), day, "stop"
        elif not np.isnan(day_high) and day_high >= target_price:
            return target_price * (1.0 - SLIPPAGE), day, "target"
        elif is_last:
            if not np.isnan(day_close):
                return day_close * (1.0 - SLIPPAGE), day, "time"
            return None, None, None

    return None, None, None


def _get_friday_close(ticker: str, week_ohlcv: dict) -> float | None:
    """Return the last available close for ticker in week_ohlcv, or None."""
    for day in sorted(week_ohlcv.keys(), reverse=True):
        day_df = week_ohlcv[day]
        if ticker in day_df.index:
            val = float(day_df.loc[ticker].get("close", np.nan))
            return val if not np.isnan(val) else None
    return None


def _build_trade_record(week_start, ticker, entry_date, exit_date, entry_price,
                         exit_price, stop_price, target_price, shares, risk_ps,
                         exit_reason, sig) -> dict:
    pnl        = (exit_price - entry_price) * shares
    r_multiple = (exit_price - entry_price) / risk_ps
    return {
        "week_start":     week_start if not hasattr(week_start, "date") else week_start.date(),
        "ticker":         ticker,
        "entry_date":     entry_date if not hasattr(entry_date, "date") else entry_date.date(),
        "exit_date":      exit_date  if not hasattr(exit_date,  "date") else exit_date.date(),
        "entry_price":    round(entry_price,  4),
        "exit_price":     round(exit_price,   4),
        "stop_price":     round(stop_price,   4),
        "target_price":   round(target_price, 4),
        "shares":         shares,
        "risk_per_share": round(risk_ps,      4),
        "pnl_dollars":    round(pnl,          2),
        "r_multiple":     round(r_multiple,   4),
        "exit_reason":    exit_reason,
        "ret_1w":         round(float(sig["ret_1w"]),       6) if pd.notna(sig.get("ret_1w"))  else None,
        "ret_3w":         round(float(sig["ret_3w"]),       6) if pd.notna(sig.get("ret_3w"))  else None,
        "rsi_14":         round(float(sig["rsi_14"]),       4) if pd.notna(sig.get("rsi_14"))  else None,
        "atr_14":         round(float(sig["atr_14"]),       4) if pd.notna(sig.get("atr_14"))  else None,
        "signal_score":   round(float(sig["signal_score"]), 6),
    }


def _empty_week_row(monday, equity: float) -> dict:
    return {
        "week_start": monday.date(), "trades": 0, "winners": 0, "losers": 0,
        "win_rate": np.nan, "total_pnl": 0.0, "total_r": 0.0,
        "equity": round(equity, 2), "weekly_return_pct": 0.0,
    }


def _weekly_row(monday, trades: int, winners: int, losers: int,
                week_pnl: float, week_r: float, equity: float,
                equity_start: float) -> dict:
    wr  = round(winners / trades, 4) if trades > 0 else np.nan
    ret = round(((equity / equity_start) - 1.0) * 100.0, 4) if equity_start > 0 else 0.0
    return {
        "week_start": monday.date(), "trades": trades,
        "winners": winners, "losers": losers, "win_rate": wr,
        "total_pnl": round(week_pnl, 2), "total_r": round(week_r, 4),
        "equity": round(equity, 2), "weekly_return_pct": ret,
    }


# ─── Variation A ─────────────────────────────────────────────────────────────

def run_variation_a(df_ind: pd.DataFrame, regime_map: dict
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Variation A: skip weeks where SPY regime is BULL_TREND or BULL_VOLATILE.
    Trade only BEAR_VOLATILE, BEAR_TREND, UNKNOWN weeks.
    """
    SKIP_REGIMES = {"BULL_TREND", "BULL_VOLATILE"}

    df = df_ind.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date", "ticker"], inplace=True)
    df.sort_index(inplace=True)

    date_set  = set(df.index.get_level_values("date").unique())
    mondays   = [m for m in pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
                 if m <= pd.Timestamp(END_DATE)]
    total_weeks = len(mondays)

    all_trades:  list[dict] = []
    weekly_rows: list[dict] = []
    equity        = ACCOUNT_START
    trades_so_far = 0

    for week_num, monday in enumerate(mondays, start=1):
        if week_num % 50 == 0:
            print(f"  [Var A] Week {week_num}/{total_weeks}... (trades: {trades_so_far})")

        regime = regime_map.get(monday, "UNKNOWN")

        # ── Skip if bull regime ───────────────────────────────────────────────
        if regime in SKIP_REGIMES:
            weekly_rows.append(_empty_week_row(monday, equity))
            continue

        prior_day = _get_prior_day(monday, date_set)
        if prior_day is None:
            weekly_rows.append(_empty_week_row(monday, equity))
            continue

        top5 = _get_top5(df, prior_day)
        if top5 is None:
            weekly_rows.append(_empty_week_row(monday, equity))
            continue

        if monday not in date_set:
            weekly_rows.append(_empty_week_row(monday, equity))
            continue

        try:
            monday_data = df.loc[monday]
        except KeyError:
            weekly_rows.append(_empty_week_row(monday, equity))
            continue

        if isinstance(monday_data, pd.Series):
            monday_data = monday_data.to_frame().T

        week_ohlcv   = _collect_week_ohlcv(df, monday, date_set)
        week_pnl     = 0.0
        week_r       = 0.0
        week_trades  = 0
        week_winners = 0
        week_losers  = 0
        equity_start = equity

        for ticker in top5.index:
            if ticker not in monday_data.index:
                continue
            mon_open = monday_data.loc[ticker, "open"]
            if pd.isna(mon_open) or mon_open <= 0:
                continue

            entry_price  = mon_open * (1.0 + SLIPPAGE)
            atr_14       = float(top5.loc[ticker, "atr_14"])
            if pd.isna(atr_14) or atr_14 <= 0:
                continue

            stop_price   = entry_price - ATR_MULT * atr_14
            risk_ps      = entry_price - stop_price
            if risk_ps <= 0:
                continue

            shares       = max(1, math.floor(RISK_PER_TRADE / risk_ps))
            target_price = entry_price + REWARD_RISK * risk_ps

            exit_price, exit_date, exit_reason = _simulate_exit(
                ticker, week_ohlcv, stop_price, target_price
            )
            if exit_price is None:
                continue

            rec = _build_trade_record(
                monday, ticker, monday, exit_date,
                entry_price, exit_price, stop_price, target_price,
                shares, risk_ps, exit_reason, top5.loc[ticker]
            )
            rec["variation"] = "A"
            all_trades.append(rec)

            pnl = rec["pnl_dollars"]
            week_pnl     += pnl
            week_r       += rec["r_multiple"]
            week_trades  += 1
            trades_so_far += 1
            if pnl > 0:
                week_winners += 1
            else:
                week_losers  += 1

        equity += week_pnl
        weekly_rows.append(_weekly_row(
            monday, week_trades, week_winners, week_losers,
            week_pnl, week_r, equity, equity_start
        ))

    trades_df = pd.DataFrame(all_trades)
    weekly_df = pd.DataFrame(weekly_rows)
    if "variation" not in trades_df.columns and not trades_df.empty:
        trades_df["variation"] = "A"
    return trades_df, weekly_df


# ─── Variation B ─────────────────────────────────────────────────────────────

def run_variation_b(df_ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Variation B: 2-week hold for winners.

    Week-1 Friday: if trade still open AND close > entry * 1.03,
    carry into week 2 with stop moved to entry (breakeven).
    Otherwise exit normally (time exit).

    Week-2: check Mon-Fri with updated stop. Forced exit on last day.
    """
    WIN_CARRY_THRESHOLD = 0.03   # > 3% gain to carry

    df = df_ind.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date", "ticker"], inplace=True)
    df.sort_index(inplace=True)

    date_set  = set(df.index.get_level_values("date").unique())
    mondays   = [m for m in pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
                 if m <= pd.Timestamp(END_DATE)]
    total_weeks = len(mondays)

    all_trades:  list[dict] = []
    weekly_rows: list[dict] = []
    equity        = ACCOUNT_START
    trades_so_far = 0

    # Carry list: trades that won't resolve until next week
    # Each item: dict with full trade context
    carry_list: list[dict] = []

    for week_num, monday in enumerate(mondays, start=1):
        if week_num % 50 == 0:
            print(f"  [Var B] Week {week_num}/{total_weeks}... (trades: {trades_so_far})")

        week_ohlcv_full = _collect_week_ohlcv(df, monday, date_set, include_monday=True)
        week_pnl     = 0.0
        week_r       = 0.0
        week_trades  = 0
        week_winners = 0
        week_losers  = 0
        equity_start = equity

        # ── Resolve carried positions from prior week ─────────────────────────
        new_carries: list[dict] = []
        for carry in carry_list:
            ticker        = carry["ticker"]
            entry_price   = carry["entry_price"]
            stop_price    = carry["stop_price"]   # already moved to breakeven
            target_price  = carry["target_price"]
            orig_stop     = carry["orig_stop"]
            shares        = carry["shares"]
            risk_ps       = carry["risk_ps"]

            # Walk full week (Mon–Fri) for the carried position
            exit_price, exit_date, exit_reason = _simulate_exit(
                ticker, week_ohlcv_full, stop_price, target_price
            )

            if exit_price is None:
                # Missing data for entire week — skip (don't carry a 3rd week)
                continue

            rec = _build_trade_record(
                carry["week_start"], ticker, carry["entry_date"], exit_date,
                entry_price, exit_price, orig_stop, target_price,
                shares, risk_ps, exit_reason, carry["sig"]
            )
            rec["variation"] = "B"
            all_trades.append(rec)

            pnl = rec["pnl_dollars"]
            week_pnl     += pnl
            week_r       += rec["r_multiple"]
            week_trades  += 1
            trades_so_far += 1
            if pnl > 0:
                week_winners += 1
            else:
                week_losers  += 1

        carry_list = []   # always clear — max 1 week of carry

        # ── New entries for this week ─────────────────────────────────────────
        prior_day = _get_prior_day(monday, date_set)
        if prior_day is None:
            equity += week_pnl
            weekly_rows.append(_weekly_row(
                monday, week_trades, week_winners, week_losers,
                week_pnl, week_r, equity, equity_start
            ) if week_trades > 0 else _empty_week_row(monday, equity))
            continue

        top5 = _get_top5(df, prior_day)
        if top5 is None:
            equity += week_pnl
            weekly_rows.append(_weekly_row(
                monday, week_trades, week_winners, week_losers,
                week_pnl, week_r, equity, equity_start
            ) if week_trades > 0 else _empty_week_row(monday, equity))
            continue

        if monday not in date_set:
            equity += week_pnl
            weekly_rows.append(_weekly_row(
                monday, week_trades, week_winners, week_losers,
                week_pnl, week_r, equity, equity_start
            ) if week_trades > 0 else _empty_week_row(monday, equity))
            continue

        try:
            monday_data = df.loc[monday]
        except KeyError:
            equity += week_pnl
            weekly_rows.append(_weekly_row(
                monday, week_trades, week_winners, week_losers,
                week_pnl, week_r, equity, equity_start
            ) if week_trades > 0 else _empty_week_row(monday, equity))
            continue

        if isinstance(monday_data, pd.Series):
            monday_data = monday_data.to_frame().T

        # Tue–Fri only for new entries (same as V1)
        week1_ohlcv = _collect_week_ohlcv(df, monday, date_set, include_monday=False)

        for ticker in top5.index:
            if ticker not in monday_data.index:
                continue
            mon_open = monday_data.loc[ticker, "open"]
            if pd.isna(mon_open) or mon_open <= 0:
                continue

            entry_price  = mon_open * (1.0 + SLIPPAGE)
            atr_14       = float(top5.loc[ticker, "atr_14"])
            if pd.isna(atr_14) or atr_14 <= 0:
                continue

            orig_stop    = entry_price - ATR_MULT * atr_14
            risk_ps      = entry_price - orig_stop
            if risk_ps <= 0:
                continue

            shares       = max(1, math.floor(RISK_PER_TRADE / risk_ps))
            target_price = entry_price + REWARD_RISK * risk_ps

            # Try to exit in week 1 (Tue–Fri)
            # Need to walk manually to catch the carryover case on the last day
            sorted_days = sorted(week1_ohlcv.keys())
            exit_price  = None
            exit_date   = None
            exit_reason = None
            carried     = False

            for idx, day in enumerate(sorted_days):
                is_last = (idx == len(sorted_days) - 1)
                day_df  = week1_ohlcv[day]

                if ticker not in day_df.index:
                    if is_last:
                        break
                    continue

                row       = day_df.loc[ticker]
                day_low   = float(row.get("low",   np.nan))
                day_high  = float(row.get("high",  np.nan))
                day_close = float(row.get("close", np.nan))

                if not np.isnan(day_low) and day_low <= orig_stop:
                    exit_price  = orig_stop * (1.0 - SLIPPAGE)
                    exit_date   = day
                    exit_reason = "stop"
                    break
                elif not np.isnan(day_high) and day_high >= target_price:
                    exit_price  = target_price * (1.0 - SLIPPAGE)
                    exit_date   = day
                    exit_reason = "target"
                    break
                elif is_last:
                    # Friday of week 1: check carry condition
                    if (not np.isnan(day_close)
                            and day_close > entry_price * (1.0 + WIN_CARRY_THRESHOLD)):
                        # Winner — carry into week 2 with breakeven stop
                        carry_list.append({
                            "ticker":      ticker,
                            "entry_price": entry_price,
                            "stop_price":  entry_price,  # breakeven
                            "orig_stop":   orig_stop,
                            "target_price": target_price,
                            "shares":      shares,
                            "risk_ps":     risk_ps,
                            "entry_date":  monday,
                            "week_start":  monday,
                            "sig":         top5.loc[ticker],
                        })
                        carried = True
                    else:
                        # Normal time exit
                        if not np.isnan(day_close):
                            exit_price  = day_close * (1.0 - SLIPPAGE)
                            exit_date   = day
                            exit_reason = "time"
                    break

            if carried:
                continue   # PnL deferred to next week

            if exit_price is None:
                continue

            rec = _build_trade_record(
                monday, ticker, monday, exit_date,
                entry_price, exit_price, orig_stop, target_price,
                shares, risk_ps, exit_reason, top5.loc[ticker]
            )
            rec["variation"] = "B"
            all_trades.append(rec)

            pnl = rec["pnl_dollars"]
            week_pnl     += pnl
            week_r       += rec["r_multiple"]
            week_trades  += 1
            trades_so_far += 1
            if pnl > 0:
                week_winners += 1
            else:
                week_losers  += 1

        equity += week_pnl
        weekly_rows.append(_weekly_row(
            monday, week_trades, week_winners, week_losers,
            week_pnl, week_r, equity, equity_start
        ) if week_trades > 0 else _empty_week_row(monday, equity))

    # Any remaining carries (last week) — drop; no week 2 to exit into
    if carry_list:
        print(f"  [Var B] {len(carry_list)} carries at end of date range — dropped.")

    trades_df = pd.DataFrame(all_trades)
    weekly_df = pd.DataFrame(weekly_rows)
    if "variation" not in trades_df.columns and not trades_df.empty:
        trades_df["variation"] = "B"
    return trades_df, weekly_df


# ─── Summary builders ────────────────────────────────────────────────────────

def _stats(trades_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict:
    """Compute key stats dict from trades and weekly DataFrames."""
    if trades_df.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "avg_r": 0.0,
            "profit_factor": 0.0, "max_dd": 0.0, "total_return": 0.0,
            "final_equity": ACCOUNT_START, "weeks_traded": 0,
            "total_weeks": len(weekly_df),
            "date_start": "N/A", "date_end": "N/A",
            "target_n": 0, "stop_n": 0, "time_n": 0,
        }

    total   = len(trades_df)
    wins    = int((trades_df["pnl_dollars"] > 0).sum())
    gross_w = trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gross_l = trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else float("inf")

    max_dd     = _max_drawdown(weekly_df["equity"])
    final_eq   = float(weekly_df["equity"].iloc[-1]) if not weekly_df.empty else ACCOUNT_START
    total_ret  = (final_eq / ACCOUNT_START - 1.0) * 100.0

    ec = trades_df["exit_reason"].value_counts()
    return {
        "total_trades":  total,
        "win_rate":      wins / total * 100.0,
        "avg_r":         float(trades_df["r_multiple"].mean()),
        "profit_factor": float(pf),
        "max_dd":        max_dd,
        "total_return":  total_ret,
        "final_equity":  final_eq,
        "weeks_traded":  int((weekly_df["trades"] > 0).sum()),
        "total_weeks":   len(weekly_df),
        "date_start":    str(trades_df["entry_date"].min()),
        "date_end":      str(trades_df["exit_date"].max()),
        "target_n":      int(ec.get("target", 0)),
        "stop_n":        int(ec.get("stop",   0)),
        "time_n":        int(ec.get("time",   0)),
    }


def _var_summary_block(label: str, description: str, s: dict) -> str:
    total = s["total_trades"]
    t_pct = s["target_n"] / total * 100 if total > 0 else 0.0
    st_pct = s["stop_n"]  / total * 100 if total > 0 else 0.0
    ti_pct = s["time_n"]  / total * 100 if total > 0 else 0.0
    pf_str = f"{s['profit_factor']:.2f}" if not np.isinf(s["profit_factor"]) else "inf"

    lines = [
        f"=== VARIATION {label} SUMMARY ({description}) ===",
        f"Date range:         {s['date_start']} to {s['date_end']}",
        f"Total weeks:        {s['total_weeks']}",
        f"Weeks with trades:  {s['weeks_traded']}  (vs V1: {V1_WEEKS_TRADED})",
        f"Total trades:       {total}  (vs V1: {V1_TOTAL_TRADES})",
        f"Win rate:           {s['win_rate']:.1f}%  (vs V1: {V1_WIN_RATE}%)",
        f"Avg R per trade:    {s['avg_r']:.2f}  (vs V1: {V1_AVG_R})",
        f"Profit factor:      {pf_str}  (vs V1: {V1_PROFIT_FACTOR})",
        f"Max drawdown:       {s['max_dd']:.1f}%  (vs V1: {V1_MAX_DD}%)",
        f"Final equity:       ${s['final_equity']:,.2f}  (vs V1: ${V1_FINAL_EQUITY:,.0f})",
        f"Total return:       {s['total_return']:.1f}%  (vs V1: {V1_TOTAL_RETURN}%)",
        "",
        "Exit breakdown:",
        f"  Target hit:       {s['target_n']}  ({t_pct:.1f}%)",
        f"  Stop hit:         {s['stop_n']}  ({st_pct:.1f}%)",
        f"  Time exit:        {s['time_n']}  ({ti_pct:.1f}%)",
    ]
    return "\n".join(lines)


def build_comparison_text(s_a: dict, s_b: dict) -> str:
    """Build the full variation_comparison.txt content."""

    v1_block = "\n".join([
        "=== VERSION 1 BASELINE ===",
        f"Total trades:       {V1_TOTAL_TRADES}",
        f"Win rate:           {V1_WIN_RATE}%",
        f"Avg R per trade:    {V1_AVG_R}",
        f"Profit factor:      {V1_PROFIT_FACTOR}",
        f"Max drawdown:       {V1_MAX_DD}%",
        f"Total return:       {V1_TOTAL_RETURN}%",
        f"Final equity:       ${V1_FINAL_EQUITY:,.2f}",
    ])

    block_a = _var_summary_block("A", "Regime Filter — skip BULL weeks", s_a)
    block_b = _var_summary_block("B", "2-Week Hold For Winners", s_b)

    pf_a = f"{s_a['profit_factor']:.2f}" if not np.isinf(s_a["profit_factor"]) else "inf"
    pf_b = f"{s_b['profit_factor']:.2f}" if not np.isinf(s_b["profit_factor"]) else "inf"

    table = "\n".join([
        f"{'Metric':<20} {'V1 Baseline':>14}  {'Var A (Regime)':>16}  {'Var B (2Wk Hold)':>16}",
        "-" * 72,
        f"{'Win Rate':<20} {V1_WIN_RATE:>13.1f}%  {s_a['win_rate']:>15.1f}%  {s_b['win_rate']:>15.1f}%",
        f"{'Profit Factor':<20} {V1_PROFIT_FACTOR:>14.2f}  {pf_a:>16}  {pf_b:>16}",
        f"{'Avg R':<20} {V1_AVG_R:>14.2f}  {s_a['avg_r']:>16.2f}  {s_b['avg_r']:>16.2f}",
        f"{'Max Drawdown':<20} {V1_MAX_DD:>13.1f}%  {s_a['max_dd']:>15.1f}%  {s_b['max_dd']:>15.1f}%",
        f"{'Total Return':<20} {V1_TOTAL_RETURN:>13.1f}%  {s_a['total_return']:>15.1f}%  {s_b['total_return']:>15.1f}%",
        f"{'Total Trades':<20} {V1_TOTAL_TRADES:>14}  {s_a['total_trades']:>16}  {s_b['total_trades']:>16}",
        f"{'Weeks Traded':<20} {V1_WEEKS_TRADED:>14}  {s_a['weeks_traded']:>16}  {s_b['weeks_traded']:>16}",
    ])

    # Build recommendation based on numbers
    versions = {
        "V1 Baseline": {
            "pf": V1_PROFIT_FACTOR, "dd": abs(V1_MAX_DD),
            "ret": V1_TOTAL_RETURN, "trades": V1_TOTAL_TRADES
        },
        "Variation A": {
            "pf": s_a["profit_factor"] if not np.isinf(s_a["profit_factor"]) else 99.0,
            "dd": abs(s_a["max_dd"]), "ret": s_a["total_return"],
            "trades": s_a["total_trades"]
        },
        "Variation B": {
            "pf": s_b["profit_factor"] if not np.isinf(s_b["profit_factor"]) else 99.0,
            "dd": abs(s_b["max_dd"]), "ret": s_b["total_return"],
            "trades": s_b["total_trades"]
        },
    }
    # Score: profit_factor / drawdown (higher is better risk-adjusted)
    scores = {k: v["pf"] / max(v["dd"], 0.1) for k, v in versions.items()}
    best_name  = max(scores, key=scores.get)
    worst_name = min(scores, key=scores.get)
    best  = versions[best_name]
    worst = versions[worst_name]
    a_pf  = versions["Variation A"]["pf"]
    b_pf  = versions["Variation B"]["pf"]
    a_dd  = versions["Variation A"]["dd"]
    b_dd  = versions["Variation B"]["dd"]

    # Build nuanced recommendation
    close_threshold = 0.05   # within 5% profit-factor is "close"
    rec_lines = []
    if best_name == "Variation A":
        rec_lines.append(
            f"Variation A (Regime Filter) produces the best risk-adjusted results. "
            f"By skipping BULL_TREND and BULL_VOLATILE weeks, it reduces total trades "
            f"to {s_a['total_trades']} while achieving a profit factor of {pf_a} "
            f"vs V1's {V1_PROFIT_FACTOR}, with a max drawdown of "
            f"{s_a['max_dd']:.1f}% vs V1's {V1_MAX_DD}%. "
        )
        if abs(a_pf - b_pf) < close_threshold * max(a_pf, b_pf):
            rec_lines.append(
                f"Variation B is competitive (PF {pf_b}), but Variation A's smaller "
                f"drawdown ({s_a['max_dd']:.1f}% vs {s_b['max_dd']:.1f}%) makes it "
                f"preferable for capital preservation."
            )
        else:
            rec_lines.append(
                f"Variation B trails with a profit factor of {pf_b} and drawdown of "
                f"{s_b['max_dd']:.1f}%. The regime filter is the stronger improvement."
            )
    elif best_name == "Variation B":
        rec_lines.append(
            f"Variation B (2-Week Hold) produces the best risk-adjusted results. "
            f"Extending winning trades into week 2 with a breakeven stop lifts the "
            f"profit factor to {pf_b} vs V1's {V1_PROFIT_FACTOR}, with a max drawdown "
            f"of {s_b['max_dd']:.1f}%. "
        )
        if abs(a_pf - b_pf) < close_threshold * max(a_pf, b_pf):
            rec_lines.append(
                f"Variation A is close (PF {pf_a}), but Variation B's higher return "
                f"({s_b['total_return']:.1f}% vs {s_a['total_return']:.1f}%) gives it "
                f"the edge. Both beat the baseline."
            )
        else:
            rec_lines.append(
                f"Variation A (PF {pf_a}) and V1 (PF {V1_PROFIT_FACTOR}) both lag. "
                f"The 2-week extension is the clearest single-rule improvement tested."
            )
    else:
        # V1 is best
        rec_lines.append(
            f"The V1 baseline retains the best risk-adjusted profile (PF {V1_PROFIT_FACTOR}, "
            f"drawdown {V1_MAX_DD}%). Neither variation demonstrates a clear improvement. "
        )
        if a_dd < b_dd:
            rec_lines.append(
                f"Variation A reduces trade frequency and drawdown ({s_a['max_dd']:.1f}%) "
                f"but at the cost of fewer trades ({s_a['total_trades']}). "
                f"Variation B's 2-week hold (PF {pf_b}) does not improve enough over V1 "
                f"to justify the added holding-period risk."
            )
        else:
            rec_lines.append(
                f"If forced to choose a variation, Variation A (PF {pf_a}) is marginally "
                f"preferable due to its smaller drawdown ({s_a['max_dd']:.1f}%)."
            )

    recommendation = " ".join(rec_lines)

    sections = [
        "=== BACKTEST VARIATION COMPARISON ===",
        "",
        v1_block,
        "",
        block_a,
        "",
        block_b,
        "",
        "--- HEAD TO HEAD ---",
        table,
        "",
        "--- RECOMMENDATION ---",
        recommendation,
    ]
    return "\n".join(sections)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data (shared between both variations) ────────────────────────────
    df_raw = load_ohlcv()
    df_ind = compute_all_indicators(df_raw)

    spy        = load_spy_data()
    regime_map = build_spy_regime_map(spy)
    regime_counts = {}
    for v in regime_map.values():
        regime_counts[v] = regime_counts.get(v, 0) + 1
    print(f"  Regime distribution: {dict(sorted(regime_counts.items()))}")

    # ── Run Variation A ───────────────────────────────────────────────────────
    print("\nRunning Variation A (Regime Filter)...")
    trades_a, weekly_a = run_variation_a(df_ind, regime_map)
    trades_a.to_csv(RESULTS_DIR / "variation_a_trades.csv",  index=False)
    weekly_a.to_csv(RESULTS_DIR / "variation_a_weekly.csv",  index=False)
    print(f"  Var A: {len(trades_a)} trades, {int((weekly_a['trades'] > 0).sum())} active weeks")

    # ── Run Variation B ───────────────────────────────────────────────────────
    print("\nRunning Variation B (2-Week Hold For Winners)...")
    trades_b, weekly_b = run_variation_b(df_ind)
    trades_b.to_csv(RESULTS_DIR / "variation_b_trades.csv",  index=False)
    weekly_b.to_csv(RESULTS_DIR / "variation_b_weekly.csv",  index=False)
    print(f"  Var B: {len(trades_b)} trades, {int((weekly_b['trades'] > 0).sum())} active weeks")

    # ── Build and save comparison ─────────────────────────────────────────────
    s_a = _stats(trades_a, weekly_a)
    s_b = _stats(trades_b, weekly_b)

    comparison = build_comparison_text(s_a, s_b)
    comp_path  = RESULTS_DIR / "variation_comparison.txt"
    comp_path.write_text(comparison)

    print(f"\nAll outputs saved to {RESULTS_DIR}/")
    print(f"  variation_a_trades.csv   ({len(trades_a)} rows)")
    print(f"  variation_a_weekly.csv   ({len(weekly_a)} rows)")
    print(f"  variation_b_trades.csv   ({len(trades_b)} rows)")
    print(f"  variation_b_weekly.csv   ({len(weekly_b)} rows)")
    print(f"  variation_comparison.txt")
    print()
    print(comparison)


if __name__ == "__main__":
    main()
