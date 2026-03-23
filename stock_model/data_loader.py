"""
data_loader.py

Universe and price data utilities for the Stock Swing Bet Model.

Responsibilities:
    - Build/load S&P 500 universe (from Wikipedia + local cache)
    - Download OHLCV price data via yfinance
    - Apply basic price/volume filters
"""

from __future__ import annotations

import os
import time
from io import StringIO
from typing import List

import pandas as pd
import requests
import yfinance as yf


def _clean_tickers(tickers: list[str]) -> list[str]:
    """
    Remove obviously bad tickers such as '0', empty strings, or values
    without any alphabetic characters.
    """
    cleaned: list[str] = []
    for t in tickers:
        if not t:
            continue
        s = str(t).strip().upper()
        if not s:
            continue
        if s == "0":
            continue
        if not any(c.isalpha() for s_char in s for c in [s_char] if s_char.isalpha()):
            continue
        cleaned.append(s)
    return cleaned


# ---------- Universe (S&P 500) ----------

def _scrape_sp500_tickers_wikipedia() -> list[str]:
    """
    Scrape S&P 500 tickers from Wikipedia.
    Uses pandas.read_html for robustness.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("No tables found when scraping S&P 500 constituents.")

    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("Expected 'Symbol' column in S&P 500 table.")

    tickers = (
        df["Symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

    # Convert dots to dashes for Yahoo Finance style
    tickers = [t.replace(".", "-") for t in tickers]

    # Clean any junk symbols that slipped through
    tickers = _clean_tickers(tickers)
    return tickers


def _scrape_sp500_tickers_slickcharts() -> list[str]:
    """
    Scrape S&P 500 tickers from SlickCharts.
    Uses pandas.read_html for robustness.
    """
    url = "https://www.slickcharts.com/sp500"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("No tables found when scraping SlickCharts S&P 500.")

    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("Expected 'Symbol' column in SlickCharts table.")

    tickers = (
        df["Symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

    # Convert dots to dashes for Yahoo Finance style
    tickers = [t.replace(".", "-") for t in tickers]

    # Clean any junk symbols that slipped through
    tickers = _clean_tickers(tickers)
    return tickers


def get_sp500_tickers(
    save_path: str = "Data/raw_data/sp500_tickers.csv",
    force_refresh: bool = False,
    cross_check: bool = True,
) -> list[str]:
    """
    Load S&P 500 tickers from CSV if it exists, otherwise scrape from Wikipedia
    and save. Set force_refresh=True to always re-scrape.
    """
    if os.path.exists(save_path) and not force_refresh:
        df = pd.read_csv(save_path, header=None)
        tickers = (
            df.iloc[:, 0]
            .astype(str)
            .str.upper()
            .str.strip()
            .dropna()
            .unique()
            .tolist()
        )
        tickers = _clean_tickers(tickers)
        return tickers

    wiki_tickers = _scrape_sp500_tickers_wikipedia()

    if not cross_check:
        tickers = wiki_tickers
    else:
        try:
            slick_tickers = _scrape_sp500_tickers_slickcharts()
        except Exception as exc:
            print(f"!!! SlickCharts scrape failed; falling back to Wikipedia only. Error: {exc}")
            slick_tickers = []

        if slick_tickers:
            wiki_only = sorted(set(wiki_tickers) - set(slick_tickers))
            slick_only = sorted(set(slick_tickers) - set(wiki_tickers))
            if wiki_only:
                print(f"    Tickers only in Wikipedia list: {len(wiki_only)}")
            if slick_only:
                print(f"    Tickers only in SlickCharts list: {len(slick_only)}")
            # Use union so we don't miss valid symbols.
            tickers = sorted(set(wiki_tickers) | set(slick_tickers))
        else:
            tickers = wiki_tickers

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.Series(tickers).to_csv(save_path, index=False, header=False)
    return tickers


# ---------- Sector Map ----------

# Maps every GICS sector name (as it appears in the Wikipedia S&P 500 table)
# to the exact key used in the dashboard's SECTOR_COLORS object.
_GICS_TO_DASHBOARD: dict[str, str] = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financials",
    "Consumer Discretionary": "Consumer",
    "Consumer Staples": "Consumer",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Telecom",
}


def get_sector_map(
    save_path: str = "Data/raw_data/sector_map.csv",
    max_age_days: int = 7,
) -> dict[str, str]:
    """
    Return a {ticker: dashboard_sector} mapping for every S&P 500 constituent.

    Source: Wikipedia S&P 500 constituents table, 'GICS Sector' column.
    The result is cached to `save_path` and re-fetched when the file is
    older than `max_age_days` (default 7 days).
    """
    # Use cache if it exists and is fresh enough
    if os.path.exists(save_path):
        age_days = (time.time() - os.path.getmtime(save_path)) / 86_400
        if age_days < max_age_days:
            df = pd.read_csv(save_path)
            return dict(zip(df["ticker"], df["sector"]))

    # Scrape fresh data from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("No tables found when scraping S&P 500 sector map.")

    df = tables[0]
    if "Symbol" not in df.columns or "GICS Sector" not in df.columns:
        raise ValueError(
            f"Expected 'Symbol' and 'GICS Sector' columns; got {list(df.columns)}"
        )

    security_col = "Security" if "Security" in df.columns else None
    cols = ["Symbol", "GICS Sector"] + ([security_col] if security_col else [])
    df = df[cols].copy()
    df["ticker"] = (
        df["Symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".", "-", regex=False)
    )
    df["gics"] = df["GICS Sector"].astype(str).str.strip()
    df["sector"] = df["gics"].map(_GICS_TO_DASHBOARD).fillna("Other")
    df["company"] = df[security_col].astype(str).str.strip() if security_col else ""
    df = df[["ticker", "sector", "company"]]

    # Drop obviously bad rows
    df = df[df["ticker"].apply(
        lambda t: bool(t) and t != "0" and any(c.isalpha() for c in t)
    )]

    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True,
    )
    df.to_csv(save_path, index=False)
    return dict(zip(df["ticker"], df["sector"]))


def get_company_map(
    save_path: str = "Data/raw_data/sector_map.csv",
    max_age_days: int = 7,
) -> dict[str, str]:
    """
    Return a {ticker: company_name} mapping for every S&P 500 constituent.
    Shares the same cache file as get_sector_map().
    """
    # Ensure the cache is fresh (re-fetches if needed via get_sector_map)
    get_sector_map(save_path=save_path, max_age_days=max_age_days)
    df = pd.read_csv(save_path)
    if "company" not in df.columns:
        return {}
    return dict(zip(df["ticker"], df["company"].fillna("")))


# ---------- Price Data ----------

def fetch_price_data(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a list of tickers using yfinance.

    Returns a DataFrame with columns:
      ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    tickers = _clean_tickers(list(tickers))
    if not tickers:
        raise ValueError("No valid tickers provided to fetch_price_data().")

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    frames = []

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.levels[0]:
                continue
            df_t = data[ticker].copy()
            if df_t.empty:
                continue
            df_t["ticker"] = ticker
            frames.append(df_t)
    else:
        # Single ticker case
        df_t = data.copy()
        if not df_t.empty:
            df_t["ticker"] = tickers[0]
            frames.append(df_t)

    if not frames:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    all_data = pd.concat(frames)
    all_data.reset_index(inplace=True)
    all_data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

    all_data["date"] = pd.to_datetime(all_data["date"])
    all_data.sort_values(["ticker", "date"], inplace=True)

    return all_data


def filter_by_price_and_volume(
    price_data: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_volume: float = 1_000_000,
) -> pd.DataFrame:
    """
    Filter price data by minimum price and 20-day average volume.

    Args:
        price_data: DataFrame with columns ['ticker', 'date', 'close', 'volume', ...]
        min_price: Minimum closing price to qualify.
        min_avg_volume: Minimum 20-day average volume to qualify.

    Returns:
        Filtered DataFrame with only eligible tickers and sorted by ['ticker', 'date'].
    """
    if price_data.empty:
        return price_data.copy()

    df = price_data.copy()

    # 20-day rolling average volume per ticker
    df["avg_vol_20"] = (
        df.groupby("ticker")["volume"]
          .rolling(20, min_periods=20)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # Latest row per ticker with price & avg_vol_20
    stats = (
        df.sort_values(["ticker", "date"])
          .groupby("ticker", as_index=False)
          .tail(1)[["ticker", "close", "avg_vol_20"]]
    )
    stats = stats.rename(columns={"close": "latest_close"})

    eligible_tickers = stats[
        (stats["latest_close"] >= min_price) &
        (stats["avg_vol_20"] >= min_avg_volume)
    ]["ticker"].tolist()

    if not eligible_tickers:
        # No names pass the filter; return empty frame with same columns
        return df.iloc[0:0].copy()

    filtered_df = df[df["ticker"].isin(eligible_tickers)].sort_values(["ticker", "date"])
    return filtered_df


def download_ohlcv_data(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
    min_price: float = 5.0,
    min_volume: int = 1_000_000,
) -> pd.DataFrame:
    """
    Convenience wrapper to fetch OHLCV data for tickers and apply
    basic min price and volume (via 20-day average) filters.

    Returns a single long DataFrame with all eligible tickers.
    """
    tickers = _clean_tickers(list(tickers))
    if not tickers:
        raise ValueError("No valid tickers provided to download_ohlcv_data().")

    raw = fetch_price_data(tickers, period=period, interval=interval)
    if raw.empty:
        return raw

    filtered = filter_by_price_and_volume(raw, min_price=min_price, min_avg_volume=min_volume)
    return filtered


if __name__ == "__main__":
    # Simple manual test:
    sp500 = get_sp500_tickers()
    print(f"Loaded {len(sp500)} S&P 500 tickers.")
    df_prices = download_ohlcv_data(sp500[:20], period="3mo", interval="1d")
    print(f"Downloaded filtered OHLCV data shape: {df_prices.shape}")
