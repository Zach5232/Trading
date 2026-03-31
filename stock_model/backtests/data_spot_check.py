"""
data_spot_check.py

One-time data integrity check: sample 30 ticker/date pairs from the yfinance
cache and compare against freshly fetched Yahoo Finance prices.

Run from stock_model/:
    python3 backtests/data_spot_check.py
"""

from __future__ import annotations

import random
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# parents[1] = stock_model/ for files directly in stock_model/backtests/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV     = PROJECT_ROOT / "Data" / "backtest_cache" / "ohlcv_5yr.csv"
OUTPUT_FILE  = PROJECT_ROOT / "Results" / "backtest" / "data_spot_check.txt"

SAMPLE_SIZE        = 30
MIN_PER_YEAR       = 5
SLEEP_BETWEEN      = 0.5   # seconds between yfinance calls
DISCREPANCY_THRESH = 1.0   # % threshold for flagging


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_pairs(df: pd.DataFrame, n: int, min_per_year: int, seed: int = 42) -> pd.DataFrame:
    """
    Return n (ticker, date) rows from df with at least min_per_year from each
    calendar year in 2021-2025.
    """
    df = df[["ticker", "date", "close"]].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    years = [2021, 2022, 2023, 2024, 2025]
    sampled_idx: list[int] = []

    for yr in years:
        yr_rows = df[df["year"] == yr]
        take = min(min_per_year, len(yr_rows))
        sampled_idx.extend(yr_rows.sample(take, random_state=seed).index.tolist())

    already = set(sampled_idx)
    slots_left = n - len(already)
    if slots_left > 0:
        remaining = df[~df.index.isin(already)]
        extra_count = min(slots_left, len(remaining))
        if extra_count > 0:
            sampled_idx.extend(
                remaining.sample(extra_count, random_state=seed).index.tolist()
            )

    result = df.loc[list(dict.fromkeys(sampled_idx))].head(n)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------

def _fetch_live_close(ticker: str, target_date: pd.Timestamp) -> float | None:
    """
    Fetch unadjusted closing price from Yahoo Finance for a specific date.
    Uses a 5-day window to handle weekends/holidays around the target.
    Returns None if the date has no data.
    """
    start = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
    end   = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,   # yfinance >= 0.2.52 kwarg; ignored by older versions
        )
    except TypeError:
        # Older yfinance version — retry without the extra kwarg
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            return None
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    # Flatten MultiIndex columns if present (e.g. yfinance wraps single ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]

    raw.index = pd.to_datetime(raw.index).normalize()
    target_norm = target_date.normalize()

    if target_norm not in raw.index:
        return None

    row = raw.loc[target_norm]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    close_val = row.get("Close") if hasattr(row, "get") else row["Close"]
    if pd.isna(close_val):
        return None
    return float(close_val)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _build_report(results: list[dict]) -> str:
    checked  = [r for r in results if r["live_close"] is not None]
    skipped  = [r for r in results if r["live_close"] is None]
    exact    = [r for r in checked if r["diff_pct"] == 0.0]
    within01 = [r for r in checked if r["diff_pct"] <= 0.1]
    within05 = [r for r in checked if r["diff_pct"] <= 0.5]
    flagged  = [r for r in checked if r["diff_pct"] > DISCREPANCY_THRESH]

    lines: list[str] = []
    lines.append("=== YFINANCE CACHE SPOT CHECK ===")
    lines.append(f"Sampled {SAMPLE_SIZE} ticker/date pairs across 2021-2025")
    lines.append("")

    hdr = f"{'Ticker':<8}  {'Date':<12}  {'Cached Close':>14}  {'Live Close':>12}  {'Diff%':>8}  Status"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in results:
        lc_str   = f"${r['live_close']:.2f}" if r["live_close"] is not None else "N/A"
        diff_str = f"{r['diff_pct']:.4f}%"   if r["diff_pct"]  is not None else "N/A"
        lines.append(
            f"{r['ticker']:<8}  {str(r['date']):<12}  ${r['cached_close']:>13.2f}  "
            f"{lc_str:>12}  {diff_str:>8}  {r['status']}"
        )

    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"Pairs checked:        {len(checked)}")
    lines.append(f"Skipped (no data):    {len(skipped)}")
    lines.append(f"Exact matches:        {len(exact)}")
    lines.append(f"Within 0.1%:          {len(within01)}")
    lines.append(f"Within 0.5%:          {len(within05)}")
    lines.append(f"Discrepancies > 1%:   {len(flagged)}")

    if flagged:
        lines.append("")
        lines.append("  Flagged pairs:")
        for r in flagged:
            lines.append(
                f"    {r['ticker']}  {r['date']}  "
                f"cached=${r['cached_close']:.4f}  live=${r['live_close']:.4f}  "
                f"diff={r['diff_pct']:.4f}%"
            )

    lines.append("")
    lines.append("--- Verdict ---")
    if len(flagged) == 0 and len(checked) > 0:
        lines.append("TRUSTWORTHY: cache matches live data within normal rounding")
    elif len(flagged) <= 2:
        lines.append(
            f"MINOR VARIANCE: {len(flagged)} pair(s) show > 1% discrepancy — see flagged pairs above"
        )
    else:
        lines.append(
            f"INVESTIGATE: {len(flagged)} pairs show > 1% discrepancy — see flagged pairs above"
        )

    lines.append("")
    lines.append("--- Likely causes of any discrepancy ---")
    lines.append("- Split adjustment timing differences")
    lines.append("- Dividend adjustment applied on different dates")
    lines.append("- Data correction applied by Yahoo after original cache pull")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading cache from {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    print(
        f"  {len(df):,} rows — {df['ticker'].nunique()} tickers — "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )

    pairs = _sample_pairs(df, SAMPLE_SIZE, MIN_PER_YEAR)
    print(f"\nSampling {len(pairs)} pairs (>= {MIN_PER_YEAR}/year for 2021-2025) ...")
    print()

    results: list[dict] = []

    for i, row in pairs.iterrows():
        ticker       = str(row["ticker"])
        target_date  = pd.Timestamp(row["date"])
        cached_close = float(row["close"])

        print(
            f"  [{i+1:2d}/{len(pairs)}] {ticker:<6}  {target_date.date()}  "
            f"cached=${cached_close:.4f}",
            end="  ",
            flush=True,
        )

        live_close = _fetch_live_close(ticker, target_date)
        time.sleep(SLEEP_BETWEEN)

        if live_close is None:
            print("SKIP — no live data")
            results.append({
                "ticker": ticker, "date": target_date.date(),
                "cached_close": cached_close, "live_close": None,
                "diff_pct": None, "status": "SKIP",
            })
            continue

        diff_pct = abs(live_close - cached_close) / cached_close * 100

        if diff_pct == 0.0:
            status = "EXACT"
        elif diff_pct <= 0.1:
            status = "PASS (<0.1%)"
        elif diff_pct <= 0.5:
            status = "PASS (<0.5%)"
        elif diff_pct <= 1.0:
            status = "PASS (<1.0%)"
        else:
            status = f"FLAG ({diff_pct:.2f}%)"

        print(f"live=${live_close:.4f}  diff={diff_pct:.4f}%  {status}")
        results.append({
            "ticker": ticker, "date": target_date.date(),
            "cached_close": cached_close, "live_close": live_close,
            "diff_pct": diff_pct, "status": status,
        })

    report = _build_report(results)

    print()
    print(report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
