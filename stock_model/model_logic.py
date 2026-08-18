

"""
model_logic.py

Signal computation and candidate selection for the Stock Swing Bet Model.

Responsibilities:
    - Take long-format OHLCV data with columns:
      ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    - Compute indicators per ticker:
        * 1-week (5 trading days) momentum
        * 3-week (15 trading days) momentum
        * 14-day RSI
        * 14-day ATR
        * 20-day average volume
    - Apply basic filters (price, volume, ATR%). RSI is computed and
      reported but is NOT a hard filter (removed in V2).
    - Build a combined signal_score
    - Select top trade candidates for the day
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- V2 hard-filter thresholds (shared with main.py) ----------
# These are the single source of truth for filter thresholds used both by
# add_signals()'s eligibility mask and by main.py's diagnostic candidate-pool
# rejection-reason breakdown, so the two never drift apart.
MIN_PRICE = 20.0              # F3 — replaces V1's $5 minimum. No upper bound.
MIN_AVG_VOL = 1_000_000
ATR_STOP_MULTIPLE = 0.75      # V2 stop = entry - 0.75 * ATR (was 1.5x in V1)
MAX_STOP_DIST_PCT = 0.08      # hard cap on stop distance as a fraction of entry
MIN_RET_3W = 0.10             # F1 — 3-week return floor
MIN_VOL_SURGE = 1.0           # F2 — volume vs 20d average floor
MIN_DIST_52W = -0.30          # F5 — must be within 30% of the 52-week high
EXCLUDED_SECTOR = "Materials"  # F4


# ---------- Indicator helpers ----------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder RSI implementation.
    """
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=series.index).rolling(
        window=period, min_periods=period
    ).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(
        window=period, min_periods=period
    ).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) over 'period' days.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


# ---------- Core signal engine ----------

def add_signals(
    price_data: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    For each ticker, compute:
      - 1-week (5 trading days) momentum: ret_1w
      - 3-week (15 trading days) momentum: ret_3w
      - 14-day RSI: rsi_14
      - 14-day ATR: atr_14
      - 20-day average volume: avg_vol_20
      - 252-trading-day high (52-week high): high_252

    Then compute per-ticker latest row including:
      - price (latest close)
      - atr_pct (ATR as % of price)
      - sector (from sector_map, 'Other' if unmapped)
      - vol_surge (volume vs 20-day average)
      - dist_52w (distance from the 52-week high, as a decimal)
      - is_eligible flag from the V2 hard filters (F1-F5 plus the V1 filters)
      - signal_score
      - direction (currently 'LONG' or 'NONE')

    Parameters
    ----------
    sector_map : dict[str, str] | None
        {ticker: sector} mapping used for the Materials-sector exclusion (F4).
        Tickers not present map to 'Other'.

    Returns a DataFrame with one row per ticker (latest date).
    """
    df = price_data.copy()
    sector_map = sector_map or {}

    all_frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()

        # Momentum
        g["ret_1w"] = g["close"].pct_change(5, fill_method=None)
        g["ret_3w"] = g["close"].pct_change(15, fill_method=None)

        # RSI
        g["rsi_14"] = compute_rsi(g["close"], period=14)

        # ATR
        g["atr_14"] = compute_atr(g, period=14)

        # Liquidity
        if "avg_vol_20" not in g.columns:
            g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()

        # 52-week high (min_periods=1 so this degrades gracefully to a
        # shorter-window high when fewer than 252 days of history exist).
        g["high_252"] = g["high"].rolling(252, min_periods=1).max()

        all_frames.append(g)

    df_signals = pd.concat(all_frames)
    df_signals.sort_values(["ticker", "date"], inplace=True)

    # Take the latest row per ticker to form today's signal snapshot
    latest = (
        df_signals.groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Derived metrics
    latest["price"] = latest["close"]
    latest["atr_pct"] = latest["atr_14"] / latest["close"]
    latest["sector"] = latest["ticker"].map(sector_map).fillna("Other")
    latest["vol_surge"] = latest["volume"] / latest["avg_vol_20"].replace(0, np.nan)
    latest["dist_52w"] = (latest["close"] - latest["high_252"]) / latest["high_252"]

    # Stop distance = ATR_STOP_MULTIPLE × ATR / entry.
    latest["stop_dist_pct"] = (ATR_STOP_MULTIPLE * latest["atr_14"]) / latest["price"]

    # NOTE: rsi_14 is intentionally NOT part of the eligibility condition.
    # V2 backtesting showed the RSI>70 "overbought" cap excluded good
    # momentum trades, so it was removed as a hard filter. RSI is still
    # computed above and reported in both the pool and candidates CSVs —
    # it's diagnostic/informational only now, never exclusionary.
    cond = (
        (latest["price"] >= MIN_PRICE)                                   # F3
        & (latest["avg_vol_20"] >= MIN_AVG_VOL)
        & (latest["stop_dist_pct"] <= MAX_STOP_DIST_PCT)
        & (latest["ret_3w"].fillna(-np.inf) >= MIN_RET_3W)                # F1
        & (latest["vol_surge"].fillna(0.0) >= MIN_VOL_SURGE)              # F2
        & (latest["sector"] != EXCLUDED_SECTOR)                          # F4
        & (latest["dist_52w"].fillna(-np.inf) >= MIN_DIST_52W)            # F5
    )

    latest["is_eligible"] = cond

    # --- Percentile rank scoring (across the FULL universe) ---
    # Every ticker gets a real percentile-rank score — not just tickers that
    # pass the hard filters — so the candidate-pool diagnostic can show a
    # genuine, non-zero score for rejected tickers too (for comparison /
    # near-miss analysis). Hard-filter pass/fail is tracked independently
    # via is_eligible/direction below, so a non-zero score on an ineligible
    # ticker does NOT make it tradeable.
    latest["rank_1w"] = latest["ret_1w"].fillna(0.0).rank(pct=True)
    latest["rank_3w"] = latest["ret_3w"].fillna(0.0).rank(pct=True)
    latest["rank_vol_surge"] = latest["vol_surge"].fillna(0.0).rank(pct=True)

    # Weighted composite score (same weights as before)
    latest["score_raw"] = (
        0.40 * latest["rank_1w"]
        + 0.40 * latest["rank_3w"]
        + 0.20 * latest["rank_vol_surge"]
    )
    latest["signal_score"] = latest["score_raw"]

    # Long-only: a ticker is only a real trade candidate if it clears every
    # hard filter — independent of how it scores relative to the rest of
    # the universe, since a mediocre name in a weak week can still "win"
    # the percentile rank without meeting the model's actual entry bar.
    latest["direction"] = np.where(latest["is_eligible"], "LONG", "NONE")

    return latest


def select_top_candidates(
    signals_df: pd.DataFrame,
    max_trades: int = 5,
) -> pd.DataFrame:
    """
    Selects the top N trade candidates based on signal_score.
    Long-only for now.

    Returns a DataFrame sorted by signal_score (descending).
    """
    df = signals_df.copy()

    # Keep only eligible, long-direction names
    df = df[(df["direction"] == "LONG") & (df["is_eligible"])]

    if df.empty:
        return df.reset_index(drop=True)

    df = df.sort_values("signal_score", ascending=False)
    return df.head(max_trades).reset_index(drop=True)


def classify_current_regime(price_data: pd.DataFrame) -> str:
    """
    Classify the current market regime using SPY price data.

    Uses:
      - SPY 50-day simple moving average
      - SPY 100-day simple moving average
      - Rolling 4-week (20-day) SPY return std dev as volatility proxy

    Returns one of:
      "BULL_TREND"     — SPY above both MAs, low volatility
      "BULL_VOLATILE"  — SPY above both MAs, high volatility
      "BEAR_TREND"     — SPY below either MA, low volatility
      "BEAR_VOLATILE"  — SPY below either MA, high volatility
      "UNKNOWN"        — insufficient data to classify

    Parameters
    ----------
    price_data : pd.DataFrame
        Long-format OHLCV DataFrame with columns [ticker, date, close].
        Must include SPY.
    """
    df = price_data.copy()

    # Extract SPY only
    spy = df[df["ticker"] == "SPY"].sort_values("date").copy()

    if len(spy) < 105:
        return "UNKNOWN"

    spy["ma_50"] = spy["close"].rolling(50, min_periods=50).mean()
    spy["ma_100"] = spy["close"].rolling(100, min_periods=100).mean()

    # 20-day rolling std dev of daily returns as volatility proxy
    spy["daily_ret"] = spy["close"].pct_change()
    spy["vol_20"] = spy["daily_ret"].rolling(20, min_periods=20).std()

    latest = spy.iloc[-1]

    # If MAs not yet computable
    if pd.isna(latest["ma_50"]) or pd.isna(latest["ma_100"]):
        return "UNKNOWN"

    spy_price = latest["close"]
    above_both_mas = (spy_price > latest["ma_50"]) and (spy_price > latest["ma_100"])
    high_vol = (latest["vol_20"] >= 0.012)  # ~equivalent to weekly std dev 0.02

    if above_both_mas and not high_vol:
        return "BULL_TREND"
    elif above_both_mas and high_vol:
        return "BULL_VOLATILE"
    elif not above_both_mas and high_vol:
        return "BEAR_VOLATILE"
    else:
        return "BEAR_TREND"
