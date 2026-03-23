"""
crypto/data_loader.py
=====================
Pull daily OHLCV for BTC-USD and ETH-USD and attach indicators.

Indicators computed here (no external TA library):
  - MA20    : 20-day simple moving average of close
  - ATR14   : 14-day Average True Range (Wilder / rolling mean of TR)
  - above_ma20 : bool, close > MA20

Output: long-format DataFrame sorted by [ticker, date].
"""

import warnings
import time
from datetime import datetime, timezone, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["BTC-USD", "ETH-USD"]
DEFAULT_START   = "2015-01-01"
MA_PERIOD       = 20
ATR_PERIOD      = 14

# Coinbase Advanced public market candles endpoint (no auth required)
_COINBASE_CANDLES_URL = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
)
_COINBASE_GRANULARITY = "ONE_DAY"
_COINBASE_TIMEOUT     = 10  # seconds


# ── Indicator helpers (ported from model_logic.py pattern) ─────────────────

def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = rolling mean of TR over `period` bars.
    Expects columns: High, Low, Close.  (capitalised — used by yfinance loader)
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _compute_atr_lower(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Same as _compute_atr but expects lowercase columns (close, high, low)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _compute_ma(series: pd.Series, period: int = MA_PERIOD) -> pd.Series:
    return series.rolling(period).mean()


# ── Main loader ────────────────────────────────────────────────────────────

def load_crypto_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance for each ticker, compute indicators,
    and return a single long-format DataFrame.

    Pass end=date.today().isoformat() to guarantee a fresh pull and prevent
    yfinance from serving stale cached data.

    Columns returned:
        ticker, date, open, high, low, close, volume,
        ma20, atr14, above_ma20
    """
    frames = []

    for ticker in tickers:
        end_label = end or "today"
        print(f"  Fetching {ticker} from {start} → {end_label}...")
        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )

        if raw.empty:
            print(f"  WARNING: No data returned for {ticker}. Skipping.")
            continue

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        ohlcv.index = pd.to_datetime(ohlcv.index)
        ohlcv.sort_index(inplace=True)

        # Indicators
        ohlcv["ma20"]  = _compute_ma(ohlcv["Close"], MA_PERIOD)
        ohlcv["atr14"] = _compute_atr(ohlcv, ATR_PERIOD)
        ohlcv["above_ma20"] = ohlcv["Close"] > ohlcv["ma20"]

        # Reshape to long format
        ohlcv = ohlcv.reset_index().rename(columns={"index": "date", "Date": "date"})
        ohlcv.rename(
            columns={
                "Open":  "open",
                "High":  "high",
                "Low":   "low",
                "Close": "close",
                "Volume":"volume",
            },
            inplace=True,
        )
        ohlcv.insert(0, "ticker", ticker)

        frames.append(ohlcv)

    if not frames:
        raise RuntimeError("No data loaded for any ticker. Check your internet connection.")

    combined = pd.concat(frames, ignore_index=True)

    # Drop rows where indicators are NaN (warm-up period)
    combined.dropna(subset=["ma20", "atr14"], inplace=True)

    combined.sort_values(["ticker", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(
        f"\n  Loaded {len(combined)} rows across {combined['ticker'].nunique()} ticker(s)."
        f"\n  Date range: {combined['date'].min().date()} → {combined['date'].max().date()}"
    )
    return combined


# ── Coinbase live loader (used by main.py only) ────────────────────────────

def load_coinbase_live(
    tickers: list[str] = DEFAULT_TICKERS,
    days: int = 60,
) -> dict[str, pd.DataFrame]:
    """
    Fetch the last `days` daily candles from the Coinbase Advanced public
    market API for each ticker.  No API key required.

    Returns a dict mapping ticker → DataFrame indexed by date (UTC midnight),
    with columns: open, high, low, close, volume, ma20, atr14, above_ma20.

    60 days is enough for MA20 (needs 20) + ATR14 (needs 14) warm-up with
    room to spare.  Raise RuntimeError immediately on any HTTP failure so the
    caller never silently processes stale data.
    """
    now   = datetime.now(timezone.utc)
    end   = int(now.timestamp())
    start = int((now - timedelta(days=days)).timestamp())

    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        product_id = ticker          # Coinbase uses the same "BTC-USD" format
        url = _COINBASE_CANDLES_URL.format(product_id=product_id)
        params = {
            "start":       str(start),
            "end":         str(end),
            "granularity": _COINBASE_GRANULARITY,
        }

        print(f"  Coinbase: fetching {ticker} daily candles ({days}d)...")
        resp = requests.get(url, params=params, timeout=_COINBASE_TIMEOUT)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Coinbase API error for {ticker}: "
                f"HTTP {resp.status_code} — {resp.text[:200]}"
            )

        payload  = resp.json()
        candles  = payload.get("candles", [])
        if not candles:
            raise RuntimeError(
                f"Coinbase returned 0 candles for {ticker}. "
                f"Response: {resp.text[:200]}"
            )

        rows = []
        for c in candles:
            rows.append(
                {
                    "date":   pd.Timestamp(int(c["start"]), unit="s", tz="UTC").normalize().tz_localize(None),
                    "open":   float(c["open"]),
                    "high":   float(c["high"]),
                    "low":    float(c["low"]),
                    "close":  float(c["close"]),
                    "volume": float(c["volume"]),
                }
            )

        df = pd.DataFrame(rows).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]  # drop any duplicate dates

        # Indicators (lowercase columns)
        df["ma20"]       = df["close"].rolling(MA_PERIOD).mean()
        df["atr14"]      = _compute_atr_lower(df, ATR_PERIOD)
        df["above_ma20"] = df["close"] > df["ma20"]

        # Drop warm-up NaN rows
        df.dropna(subset=["ma20", "atr14"], inplace=True)

        if df.empty:
            raise RuntimeError(
                f"Not enough candles for {ticker} to compute indicators "
                f"(need ≥{MA_PERIOD} days, got {len(rows)})."
            )

        latest = df.index[-1].date()
        print(f"    {len(df)} usable bars  |  latest close: {latest}  ${df['close'].iloc[-1]:,.2f}")
        result[ticker] = df

    return result


# ── Coinbase hourly loader ─────────────────────────────────────────────────

def load_hourly_data(
    tickers: list[str] = DEFAULT_TICKERS,
    days: int = 90,
) -> dict[str, pd.DataFrame]:
    """
    Fetch the last `days` days of hourly OHLCV from Coinbase Advanced API.
    Returns dict mapping ticker → DataFrame indexed by UTC-aware Timestamps.

    Paginates backward in 300-hour windows (API limit per request).
    90 days × 24 hours = 2,160 bars ≈ 8 requests per ticker.
    """
    now          = datetime.now(timezone.utc)
    target_start = now - timedelta(days=days)
    HOURS_PER_REQ = 300

    result: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        url      = _COINBASE_CANDLES_URL.format(product_id=ticker)
        all_rows: list[dict] = []
        window_end = now

        print(f"  Coinbase hourly: fetching {ticker} ({days}d)...", end="", flush=True)

        while window_end > target_start:
            window_start = max(
                window_end - timedelta(hours=HOURS_PER_REQ),
                target_start,
            )
            params = {
                "start":       str(int(window_start.timestamp())),
                "end":         str(int(window_end.timestamp())),
                "granularity": "ONE_HOUR",
            }
            resp = requests.get(url, params=params, timeout=_COINBASE_TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Coinbase hourly API error for {ticker}: "
                    f"HTTP {resp.status_code} — {resp.text[:200]}"
                )
            for c in resp.json().get("candles", []):
                all_rows.append({
                    "datetime": pd.Timestamp(int(c["start"]), unit="s", tz="UTC"),
                    "open":   float(c["open"]),
                    "high":   float(c["high"]),
                    "low":    float(c["low"]),
                    "close":  float(c["close"]),
                    "volume": float(c["volume"]),
                })
            window_end = window_start
            time.sleep(0.3)
            print(".", end="", flush=True)

        print()
        df = (
            pd.DataFrame(all_rows)
            .set_index("datetime")
            .sort_index()
        )
        df = df[~df.index.duplicated(keep="last")]
        print(f"    {len(df)} hourly bars  |  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        result[ticker] = df

    return result


# ── Convenience helpers used by backtest and main ──────────────────────────

def get_ticker_df(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return rows for a single ticker, indexed by date."""
    subset = data[data["ticker"] == ticker].copy()
    subset.set_index("date", inplace=True)
    subset.index = pd.to_datetime(subset.index)
    return subset


def get_fridays(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose weekday is Friday (dayofweek == 4)."""
    return ticker_df[ticker_df.index.dayofweek == 4].copy()
