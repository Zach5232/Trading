

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
    - Apply basic filters (price, volume, ATR%, RSI caps)
    - Build a combined signal_score
    - Select top trade candidates for the day
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

def add_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker, compute:
      - 1-week (5 trading days) momentum: ret_1w
      - 3-week (15 trading days) momentum: ret_3w
      - 14-day RSI: rsi_14
      - 14-day ATR: atr_14
      - 20-day average volume: avg_vol_20

    Then compute per-ticker latest row including:
      - price (latest close)
      - atr_pct (ATR as % of price)
      - is_eligible flag from filters
      - signal_score
      - direction (currently 'LONG' or 'NONE')

    Returns a DataFrame with one row per ticker (latest date).
    """
    df = price_data.copy()

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

    # ------------------ Filters (tunable) ------------------
    min_price = 5.0
    max_price = 300.0
    min_avg_vol = 1_000_000
    # Stop distance = 1.5 × ATR / entry. Cap at 3% so no candidate
    # can ever produce a stop wider than the model rules allow.
    # Equivalent to: ATR% must be <= 0.02 (2% of price).
    max_stop_dist_pct = 0.03   # 1.5 × ATR / entry <= 3%
    rsi_lower_cap = 30         # avoid too crushed for momentum longs
    rsi_upper_cap = 70         # avoid too extended / blow-off
    # -------------------------------------------------------

    # Compute the actual stop distance a candidate would receive
    latest["stop_dist_pct"] = (1.5 * latest["atr_14"]) / latest["price"]

    cond = (
        (latest["price"] >= min_price)
        & (latest["price"] <= max_price)
        & (latest["avg_vol_20"] >= min_avg_vol)
        & (latest["stop_dist_pct"] <= max_stop_dist_pct)
        & (latest["rsi_14"] >= rsi_lower_cap)
        & (latest["rsi_14"] <= rsi_upper_cap)
    )

    latest["is_eligible"] = cond

    # --- Percentile rank scoring (across eligible universe only) ---
    # Step 1: compute ranks only on eligible rows
    eligible_mask = latest["is_eligible"]

    latest["rank_1w"] = 0.0
    latest["rank_3w"] = 0.0
    latest["rank_vol_surge"] = 0.0

    if eligible_mask.sum() > 1:
        eligible_idx = latest[eligible_mask].index

        latest.loc[eligible_idx, "rank_1w"] = (
            latest.loc[eligible_idx, "ret_1w"]
            .fillna(0.0)
            .rank(pct=True)
        )
        latest.loc[eligible_idx, "rank_3w"] = (
            latest.loc[eligible_idx, "ret_3w"]
            .fillna(0.0)
            .rank(pct=True)
        )
        # Volume surge = today's volume vs 20-day avg
        vol_ratio = latest.loc[eligible_idx, "volume"] / latest.loc[eligible_idx, "avg_vol_20"].replace(0, float("nan"))
        latest.loc[eligible_idx, "rank_vol_surge"] = vol_ratio.fillna(0.0).rank(pct=True)

    # Step 2: weighted composite score
    latest["score_raw"] = (
        0.40 * latest["rank_1w"]
        + 0.40 * latest["rank_3w"]
        + 0.20 * latest["rank_vol_surge"]
    )

    # Step 3: zero out ineligible rows
    latest["signal_score"] = latest["score_raw"].where(eligible_mask, 0.0)

    # For now, we are long-only: positive scores -> LONG, else NONE
    latest["direction"] = np.where(latest["signal_score"] > 0, "LONG", "NONE")

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
