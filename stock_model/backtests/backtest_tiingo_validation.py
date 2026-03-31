"""
backtest_tiingo_validation.py

Data integrity check: runs the identical V1 momentum backtest logic against
Tiingo-sourced OHLCV data and compares results to the existing yfinance run.

If results match → yfinance is trustworthy for this strategy.
If results diverge → investigate price-data quality before trusting conclusions.

Steps:
  1. Fetch/load daily OHLCV from Tiingo REST API (cached to ohlcv_5yr_tiingo.csv)
  2. Run the exact same backtest_engine.run_backtest() on the Tiingo data
  3. Compare to the already-computed yfinance results
  4. Write data_validation_report.txt

NOTE on PROJECT_ROOT:
  This file lives at stock_model/backtests/backtest_tiingo_validation.py.
  parents[1] => stock_model/ — mirrors backtest_engine.py exactly.
  (The task spec said parents[2], which is a copy-paste error from the
  deeper mean_reversion/ subfolder; parents[2] here resolves to Trading/.)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # stock_model/
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "backtest_cache"
YF_CACHE     = CACHE_DIR / "ohlcv_5yr.csv"
TIINGO_CACHE = DATA_DIR / "raw_data" / "ohlcv_5yr_tiingo.csv"
RESULTS_DIR  = PROJECT_ROOT / "Results" / "backtest" / "tiingo_validation"
TICKER_CSV   = DATA_DIR / "raw_data" / "sp500_tickers.csv"

YF_TRADES_CSV  = PROJECT_ROOT / "Results" / "backtest" / "momentum" / "backtest_trades.csv"
YF_WEEKLY_CSV  = PROJECT_ROOT / "Results" / "backtest" / "momentum" / "backtest_weekly_summary.csv"

# ─── Import backtest engine ───────────────────────────────────────────────────
# backtest_engine.py lives in the same directory as this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import (          # noqa: E402
    run_backtest,
    compute_all_indicators,
    load_tickers,
    ACCOUNT_START,
    START_DATE,
    END_DATE,
)

# ─── Tiingo configuration ─────────────────────────────────────────────────────

TIINGO_API_KEY = "dfbb7048de301ea4f8ccf083c8e7b0385bcbfda2"   # Replace with your Tiingo API key from https://api.tiingo.com

TIINGO_START   = "2021-01-01"
TIINGO_END     = "2026-03-31"
TIINGO_SLEEP   = 0.12              # 0.12 s between calls → ≤10 req/s

TIINGO_BASE    = "https://www.tiingo.com/api/tiingo/daily/{ticker}/prices"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _max_drawdown(equity_series: pd.Series) -> float:
    """Peak-to-trough drawdown as a negative percentage."""
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    dd   = (equity_series - peak) / peak
    return float(dd.min() * 100.0)


def _compute_stats(trades_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict:
    """Standard summary stats dict from a completed backtest run."""
    n = len(trades_df)
    if n == 0:
        return dict(
            total_trades=0, win_rate=0.0, profit_factor=0.0,
            avg_r=0.0, max_dd=0.0, total_return=0.0,
            final_equity=ACCOUNT_START,
        )

    wins    = int((trades_df["pnl_dollars"] > 0).sum())
    gross_w = trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gross_l = trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else float("inf")

    eq         = weekly_df["equity"]
    final_eq   = float(eq.iloc[-1]) if not weekly_df.empty else ACCOUNT_START
    total_ret  = (final_eq / ACCOUNT_START - 1.0) * 100.0

    return dict(
        total_trades  = n,
        win_rate      = round(wins / n * 100.0, 1),
        profit_factor = round(float(pf), 2) if not math.isinf(pf) else 999.0,
        avg_r         = round(float(trades_df["r_multiple"].mean()), 4),
        max_dd        = round(_max_drawdown(eq), 1),
        total_return  = round(total_ret, 1),
        final_equity  = round(final_eq, 2),
    )


# ─── Ticker format translation ───────────────────────────────────────────────

def to_tiingo_ticker(ticker: str) -> str:
    """
    Translate a yfinance-format ticker to Tiingo format.
    yfinance uses hyphens (BRK-B); Tiingo uses periods (BRK.B).
    The original hyphen-format is preserved in the output DataFrame so both
    datasets can be joined on the same key.
    """
    return ticker.replace("-", ".")


# ─── Tiingo data fetcher ──────────────────────────────────────────────────────

def fetch_tiingo_data(
    tickers: list[str],
    api_key: str,
    cache_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch daily OHLCV from Tiingo REST API for all tickers.

    Uses adjClose/adjOpen/adjHigh/adjLow (split+dividend adjusted) to match
    yfinance auto_adjust=True. Falls back to unadjusted with a terminal warning
    if adjusted fields are absent for a ticker.

    Ticker translation: yfinance hyphens → Tiingo periods in the URL only
    (BRK-B fetched as BRK.B; stored in DataFrame as BRK-B for join compatibility).

    If cache_path already exists, loads it and only fetches tickers not yet
    present — enabling incremental updates without a full re-download.

    Returns (df, skipped_tickers).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing cache; find which tickers still need fetching ───────────
    initial_frames: list[pd.DataFrame] = []
    skipped:        list[str]          = []
    already_cached: set[str]           = set()

    if cache_path.exists():
        existing       = pd.read_csv(cache_path, parse_dates=["date"])
        already_cached = set(existing["ticker"].unique())
        tickers_to_fetch = [t for t in tickers if t not in already_cached]

        if not tickers_to_fetch:
            print(f"Loading Tiingo cache from {cache_path.name} ...")
            print(f"  {len(existing):,} rows, {existing['ticker'].nunique()} tickers.")
            return existing, []

        print(f"Cache exists ({existing['ticker'].nunique()} tickers); "
              f"fetching {len(tickers_to_fetch)} missing tickers...")
        initial_frames.append(existing)
    else:
        tickers_to_fetch = list(tickers)

    total = len(tickers_to_fetch)
    print("Using Tiingo adjClose (split+dividend adjusted) to match yfinance auto_adjust=True")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}",
    }

    all_frames: list[pd.DataFrame] = []

    for idx, ticker in enumerate(tickers_to_fetch, start=1):
        print(f"  Fetching ticker {idx}/{total}: {ticker}...", flush=True)

        # Tiingo uses periods where yfinance uses hyphens (BRK-B → brk.b)
        tiingo_ticker = to_tiingo_ticker(ticker).lower()
        url    = TIINGO_BASE.format(ticker=tiingo_ticker)
        params = {
            "startDate": TIINGO_START,
            "endDate":   TIINGO_END,
            "token":     api_key,
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)

            if resp.status_code == 404:
                print(f"    SKIP {ticker}: 404 Not Found")
                skipped.append(ticker)
                time.sleep(TIINGO_SLEEP)
                continue

            if resp.status_code != 200:
                print(f"    SKIP {ticker}: HTTP {resp.status_code}")
                skipped.append(ticker)
                time.sleep(TIINGO_SLEEP)
                continue

            data = resp.json()

            if not data:
                print(f"    SKIP {ticker}: empty response")
                skipped.append(ticker)
                time.sleep(TIINGO_SLEEP)
                continue

        except requests.exceptions.RequestException as exc:
            print(f"    SKIP {ticker}: request error — {exc}")
            skipped.append(ticker)
            time.sleep(TIINGO_SLEEP)
            continue
        except ValueError as exc:
            print(f"    SKIP {ticker}: JSON parse error — {exc}")
            skipped.append(ticker)
            time.sleep(TIINGO_SLEEP)
            continue

        # Check whether adjusted fields are present in the response
        has_adj = (data[0].get("adjClose") is not None) if data else False
        if not has_adj:
            print(f"    WARNING {ticker}: adjClose absent — using unadjusted close")

        # Parse response — prefer adjusted fields; fall back to raw if absent
        rows = []
        for entry in data:
            try:
                rows.append({
                    "ticker": ticker,                         # keep yfinance format (hyphen)
                    "date":   entry["date"],
                    "open":   entry.get("adjOpen")  if has_adj else entry.get("open"),
                    "high":   entry.get("adjHigh")  if has_adj else entry.get("high"),
                    "low":    entry.get("adjLow")   if has_adj else entry.get("low"),
                    "close":  entry.get("adjClose") if has_adj else entry.get("close"),
                    "volume": entry.get("adjVolume") or entry.get("volume"),
                })
            except (KeyError, TypeError):
                continue

        if not rows:
            print(f"    SKIP {ticker}: no parseable rows")
            skipped.append(ticker)
            time.sleep(TIINGO_SLEEP)
            continue

        df_t = pd.DataFrame(rows)
        # Normalize Tiingo ISO timestamps (e.g. 2021-01-04T00:00:00+00:00) to plain dates
        df_t["date"] = (
            pd.to_datetime(df_t["date"], utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        df_t.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        df_t.sort_values("date", inplace=True)

        if df_t.empty:
            print(f"    SKIP {ticker}: all rows dropped after normalization")
            skipped.append(ticker)
            time.sleep(TIINGO_SLEEP)
            continue

        all_frames.append(df_t[["ticker", "date", "open", "high", "low", "close", "volume"]])
        time.sleep(TIINGO_SLEEP)

    if not all_frames and not initial_frames:
        raise RuntimeError("Tiingo fetch returned no data. Check API key and connectivity.")

    df = pd.concat(initial_frames + all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)
    df.to_csv(cache_path, index=False)
    print(f"\n  Saved {len(df):,} Tiingo rows → {cache_path.name}")
    if skipped:
        print(f"  Skipped {len(skipped)} tickers: {', '.join(skipped[:20])}"
              + (" ..." if len(skipped) > 20 else ""))

    return df, skipped


# ─── Price discrepancy check ──────────────────────────────────────────────────

def compute_price_discrepancy(
    df_yf: pd.DataFrame,
    df_tiingo: pd.DataFrame,
    n_samples: int = 20,
    rng_seed: int = 42,
) -> dict:
    """
    Sample n_samples random (ticker, date) pairs present in both datasets
    and compute the close-price percentage difference.

    Returns a dict with avg_pct_diff, max_pct_diff, max_ticker, max_date,
    assessment, and sampled_pairs list.
    """
    # Build lookup keys for both datasets
    df_yf     = df_yf.copy()
    df_tiingo = df_tiingo.copy()
    df_yf["date"]     = pd.to_datetime(df_yf["date"])
    df_tiingo["date"] = pd.to_datetime(df_tiingo["date"])
    df_yf["_key"]     = df_yf["ticker"]     + "|" + df_yf["date"].dt.strftime("%Y-%m-%d")
    df_tiingo["_key"] = df_tiingo["ticker"] + "|" + df_tiingo["date"].dt.strftime("%Y-%m-%d")

    yf_map     = df_yf.set_index("_key")["close"]
    tiingo_map = df_tiingo.set_index("_key")["close"]

    common_keys = list(set(yf_map.index) & set(tiingo_map.index))

    if len(common_keys) < n_samples:
        sample_keys = common_keys
    else:
        rng         = np.random.default_rng(rng_seed)
        sample_keys = list(rng.choice(common_keys, size=n_samples, replace=False))

    diffs: list[float] = []
    details: list[dict] = []
    for key in sample_keys:
        yf_c  = float(yf_map[key])
        ti_c  = float(tiingo_map[key])
        if yf_c > 0:
            pct = abs(yf_c - ti_c) / yf_c * 100.0
            ticker, date_str = key.split("|", 1)
            diffs.append(pct)
            details.append({"ticker": ticker, "date": date_str,
                             "yf_close": yf_c, "tiingo_close": ti_c, "pct_diff": pct})

    if not diffs:
        return dict(
            avg_pct_diff=0.0, max_pct_diff=0.0,
            max_ticker="N/A", max_date="N/A",
            assessment="NO DATA", n_sampled=0, details=[],
        )

    avg_diff = float(np.mean(diffs))
    max_idx  = int(np.argmax(diffs))
    max_d    = details[max_idx]

    if avg_diff < 0.5:
        assessment = "PASS"
    elif avg_diff < 2.0:
        assessment = "WARN"
    else:
        assessment = "FAIL"

    return dict(
        avg_pct_diff = round(avg_diff, 4),
        max_pct_diff = round(float(max_d["pct_diff"]), 4),
        max_ticker   = max_d["ticker"],
        max_date     = max_d["date"],
        assessment   = assessment,
        n_sampled    = len(diffs),
        details      = details,
    )


# ─── Outlier scanner ─────────────────────────────────────────────────────────

def scan_price_outliers(
    df_yf: pd.DataFrame,
    df_tiingo: pd.DataFrame,
    outlier_threshold: float = 10.0,
    max_pairs: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Scan up to max_pairs random (ticker, date) pairs present in both datasets.
    Returns a DataFrame of pairs where abs price difference > outlier_threshold%.

    Columns: ticker, date, yfinance_close, tiingo_close, pct_difference, likely_cause
    """
    df_yf     = df_yf.copy()
    df_tiingo = df_tiingo.copy()
    df_yf["date"]     = pd.to_datetime(df_yf["date"])
    df_tiingo["date"] = pd.to_datetime(df_tiingo["date"])
    df_yf["_key"]     = df_yf["ticker"]     + "|" + df_yf["date"].dt.strftime("%Y-%m-%d")
    df_tiingo["_key"] = df_tiingo["ticker"] + "|" + df_tiingo["date"].dt.strftime("%Y-%m-%d")

    yf_map     = df_yf.set_index("_key")["close"]
    tiingo_map = df_tiingo.set_index("_key")["close"]

    common_keys = list(set(yf_map.index) & set(tiingo_map.index))
    rng = np.random.default_rng(rng_seed)
    sample_keys = (
        list(rng.choice(common_keys, size=max_pairs, replace=False))
        if len(common_keys) > max_pairs
        else common_keys
    )

    outliers: list[dict] = []
    for key in sample_keys:
        yf_c = float(yf_map[key])
        ti_c = float(tiingo_map[key])
        if yf_c <= 0:
            continue
        pct = abs(yf_c - ti_c) / yf_c * 100.0
        if pct > outlier_threshold:
            ticker, date_str = key.split("|", 1)
            cause = (
                "likely split adjustment timing"
                if pct > 40.0
                else "possible dividend adjustment or data gap"
            )
            outliers.append({
                "ticker":          ticker,
                "date":            date_str,
                "yfinance_close":  round(yf_c, 4),
                "tiingo_close":    round(ti_c, 4),
                "pct_difference":  round(pct, 2),
                "likely_cause":    cause,
            })

    if not outliers:
        return pd.DataFrame(columns=[
            "ticker", "date", "yfinance_close", "tiingo_close",
            "pct_difference", "likely_cause",
        ])
    return pd.DataFrame(outliers).sort_values("pct_difference", ascending=False)


# ─── Report builder ──────────────────────────────────────────────────────────

def build_report(
    yf_stats:     dict,
    tiingo_stats: dict,
    yf_tickers:   set,
    tiingo_tickers: set,
    discrepancy:  dict,
    skipped:      list[str],
) -> str:
    missing = sorted(yf_tickers - tiingo_tickers)
    tiingo_only = sorted(tiingo_tickers - yf_tickers)

    lines = ["=== DATA VALIDATION REPORT: yfinance vs Tiingo ===", ""]

    # ── Coverage ──────────────────────────────────────────────────────────────
    lines.append("--- Coverage ---")
    lines.append(f"yfinance tickers:          {len(yf_tickers)}")
    lines.append(f"Tiingo tickers fetched:    {len(tiingo_tickers)}")
    lines.append(f"Tickers missing in Tiingo: {len(missing)}")
    if missing:
        lines.append(f"  Missing: {', '.join(missing[:30])}"
                     + (" ..." if len(missing) > 30 else ""))
    if tiingo_only:
        lines.append(f"Tickers only in Tiingo:  {len(tiingo_only)}")
    lines.append("")

    # ── Price discrepancy ─────────────────────────────────────────────────────
    lines.append("--- Price Discrepancy Check ---")
    lines.append(
        f"Sampled {discrepancy['n_sampled']} random ticker/date pairs — "
        f"avg price difference: {discrepancy['avg_pct_diff']:.2f}%"
    )
    lines.append(
        f"Max single price difference observed: "
        f"{discrepancy['max_pct_diff']:.2f}% "
        f"({discrepancy['max_ticker']} on {discrepancy['max_date']})"
    )
    lines.append(f"Assessment: {discrepancy['assessment']}"
                 + (" (< 0.5% avg)" if discrepancy["assessment"] == "PASS"
                    else " (0.5–2.0% avg)" if discrepancy["assessment"] == "WARN"
                    else " (> 2.0% avg)"))
    lines.append("")

    # ── Backtest results comparison ───────────────────────────────────────────
    lines.append("--- Backtest Results Comparison ---")

    col_w = 16
    hdr   = f"{'Metric':<22} {'yfinance':>{col_w}} {'Tiingo':>{col_w}} {'Difference':>{col_w}}"
    sep   = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    def _diff_str(yf_v, ti_v, is_pct: bool = False) -> str:
        if isinstance(yf_v, int) or (isinstance(yf_v, float) and yf_v == int(yf_v) and not is_pct):
            diff = ti_v - yf_v
            return f"{diff:>+{col_w}d}" if isinstance(diff, int) else f"{diff:>+{col_w}.0f}"
        diff = ti_v - yf_v
        return f"{diff:>+{col_w}.2f}"

    comparison_rows = [
        ("Total trades",   yf_stats["total_trades"],   tiingo_stats["total_trades"],   False, "d"),
        ("Win rate",       yf_stats["win_rate"],        tiingo_stats["win_rate"],        True,  "1%"),
        ("Profit factor",  yf_stats["profit_factor"],   tiingo_stats["profit_factor"],   False, "2f"),
        ("Avg R",          yf_stats["avg_r"],           tiingo_stats["avg_r"],           False, "4f"),
        ("Max drawdown",   yf_stats["max_dd"],          tiingo_stats["max_dd"],          True,  "1%"),
        ("Total return",   yf_stats["total_return"],    tiingo_stats["total_return"],    True,  "1%"),
        ("Final equity",   yf_stats["final_equity"],    tiingo_stats["final_equity"],    False, "2f"),
    ]

    for label, yf_v, ti_v, _is_pct, fmt in comparison_rows:
        if fmt == "d":
            yf_s = f"{int(yf_v):>{col_w}d}"
            ti_s = f"{int(ti_v):>{col_w}d}"
            diff = int(ti_v) - int(yf_v)
            di_s = f"{diff:>+{col_w}d}"
        elif fmt.endswith("%"):
            d    = int(fmt[:-1]) if fmt[:-1].isdigit() else 1
            yf_s = f"{float(yf_v):>{col_w-1}.{d}f}%"
            ti_s = f"{float(ti_v):>{col_w-1}.{d}f}%"
            diff = float(ti_v) - float(yf_v)
            di_s = f"{diff:>+{col_w-1}.{d}f}%"
        else:
            d    = int(fmt[:-1]) if fmt[:-1].isdigit() else 2
            yf_s = f"{float(yf_v):>{col_w}.{d}f}"
            ti_s = f"{float(ti_v):>{col_w}.{d}f}"
            diff = float(ti_v) - float(yf_v)
            di_s = f"{diff:>+{col_w}.{d}f}"
        lines.append(f"{label:<22} {yf_s} {ti_s} {di_s}")

    lines.append("")

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines.append("--- Verdict ---")

    # Divergence: use total return difference as primary signal
    ret_diff  = abs(tiingo_stats["total_return"] - yf_stats["total_return"])
    pf_diff   = abs(tiingo_stats["profit_factor"] - yf_stats["profit_factor"])
    trade_diff_pct = (
        abs(tiingo_stats["total_trades"] - yf_stats["total_trades"])
        / max(yf_stats["total_trades"], 1) * 100.0
    )

    # Updated thresholds:
    #   TRUSTWORTHY:    PF diff < 0.05  AND  total return diff < 3pp
    #   MINOR VARIANCE: PF diff 0.05–0.10  OR  return diff 3–8pp
    #   DISCREPANCY:    PF diff > 0.10  OR  return diff > 8pp
    if (
        discrepancy["assessment"] == "PASS"
        and pf_diff < 0.05
        and ret_diff < 3.0
    ):
        verdict = (
            f"TRUSTWORTHY: Results are within tight tolerance — "
            f"profit factor differs by {pf_diff:.2f} (< 0.05 threshold), "
            f"total return by {ret_diff:.1f}pp (< 3pp threshold), "
            f"trade count by {trade_diff_pct:.1f}%. "
            f"yfinance data is reliable for this backtest. "
            f"Price-level discrepancy: {discrepancy['avg_pct_diff']:.2f}% avg ({discrepancy['assessment']})."
        )
    elif pf_diff <= 0.10 and ret_diff <= 8.0:
        verdict = (
            f"MINOR VARIANCE: Results are broadly consistent but diverge slightly — "
            f"profit factor differs by {pf_diff:.2f}, total return by {ret_diff:.1f}pp, "
            f"trade count by {trade_diff_pct:.1f}%. "
            f"Price-level discrepancy: {discrepancy['avg_pct_diff']:.2f}% avg ({discrepancy['assessment']}). "
            f"Likely caused by missing tickers ({len(missing)}) or minor price feed differences. "
            f"yfinance results are usable — spot-check outlier tickers before drawing conclusions."
        )
    else:
        verdict = (
            f"DISCREPANCY: Results diverge materially — "
            f"profit factor differs by {pf_diff:.2f} (threshold: 0.10), "
            f"total return by {ret_diff:.1f}pp (threshold: 8pp), "
            f"trade count by {trade_diff_pct:.1f}%. "
            f"Price-level discrepancy: {discrepancy['avg_pct_diff']:.2f}% avg ({discrepancy['assessment']}). "
            f"Investigate price differences before trusting backtest conclusions. "
            f"Check price_outliers.csv for specific problem tickers."
        )

    lines.append(verdict)
    lines.append("")

    # ── Data quality notes ────────────────────────────────────────────────────
    lines.append("--- Data Quality Notes ---")
    if skipped:
        lines.append(
            f"Tickers skipped during Tiingo fetch ({len(skipped)}): "
            + ", ".join(skipped[:20])
            + (" ..." if len(skipped) > 20 else "")
        )
    else:
        lines.append("No tickers were skipped during Tiingo fetch.")

    if missing:
        lines.append(
            f"Tickers present in yfinance cache but absent from Tiingo "
            f"({len(missing)}): {', '.join(missing[:20])}"
            + (" ..." if len(missing) > 20 else "")
        )

    if discrepancy["assessment"] != "PASS":
        lines.append(
            f"Price discrepancy WARNING: avg {discrepancy['avg_pct_diff']:.2f}% close-price "
            f"difference across sampled pairs. Largest discrepancy: "
            f"{discrepancy['max_ticker']} on {discrepancy['max_date']} "
            f"({discrepancy['max_pct_diff']:.2f}%). This may reflect dividend/split "
            f"adjustment differences between the two data providers."
        )

    lines.append(
        "Note: Tiingo uses adjClose/adjOpen/adjHigh/adjLow (split+dividend adjusted) "
        "to match yfinance auto_adjust=True. Any remaining systematic differences "
        "likely reflect adjustment timing gaps or rounding differences between providers. "
        "See price_outliers.csv for pairs with > 10% discrepancy."
    )

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if TIINGO_API_KEY == "your_key_here":
        raise ValueError(
            "Set TIINGO_API_KEY at the top of this file before running.\n"
            "Get a free key at https://api.tiingo.com"
        )

    # Step 1: Load ticker list
    tickers = load_tickers()
    print(f"Universe: {len(tickers)} tickers.")

    # Step 2: Fetch / load Tiingo data
    print("\n--- Step 1: Tiingo Data Fetch ---")
    df_tiingo, skipped = fetch_tiingo_data(tickers, TIINGO_API_KEY, TIINGO_CACHE)

    # Step 3: Compute indicators on Tiingo data (same function as backtest_engine.py)
    print("\n--- Step 2: Computing Indicators ---")
    df_tiingo_ind = compute_all_indicators(df_tiingo)

    # Step 4: Run the identical V1 backtest on Tiingo data
    # run_backtest() is imported directly from backtest_engine.py — zero logic replication.
    print("\n--- Step 3: Running V1 Backtest on Tiingo Data ---")
    tiingo_trades, tiingo_weekly = run_backtest(df_tiingo_ind)

    # Step 5: Save Tiingo backtest outputs
    tiingo_trades_path  = RESULTS_DIR / "tiingo_trades.csv"
    tiingo_weekly_path  = RESULTS_DIR / "tiingo_weekly_summary.csv"
    tiingo_trades.to_csv(tiingo_trades_path,  index=False)
    tiingo_weekly.to_csv(tiingo_weekly_path,  index=False)
    print(f"  Saved {len(tiingo_trades)} trades  → {tiingo_trades_path.name}")
    print(f"  Saved {len(tiingo_weekly)} weeks   → {tiingo_weekly_path.name}")

    # Step 6: Load existing yfinance results for comparison
    print("\n--- Step 4: Loading yfinance Results for Comparison ---")
    if not YF_TRADES_CSV.exists() or not YF_WEEKLY_CSV.exists():
        raise FileNotFoundError(
            f"yfinance backtest results not found at {YF_TRADES_CSV.parent}.\n"
            "Run backtest_engine.py first."
        )
    yf_trades  = pd.read_csv(YF_TRADES_CSV,  parse_dates=["week_start", "entry_date", "exit_date"])
    yf_weekly  = pd.read_csv(YF_WEEKLY_CSV,  parse_dates=["week_start"])
    print(f"  Loaded {len(yf_trades)} yfinance trades, {len(yf_weekly)} weeks.")

    # Step 7: Compute stats for both runs
    yf_stats     = _compute_stats(yf_trades,     yf_weekly)
    tiingo_stats = _compute_stats(tiingo_trades, tiingo_weekly)

    # Step 8: Price discrepancy check
    print("\n--- Step 5: Price Discrepancy Check ---")
    if YF_CACHE.exists():
        df_yf_raw = pd.read_csv(YF_CACHE, parse_dates=["date"])
    else:
        print("  Warning: yfinance OHLCV cache not found — skipping price check.")
        df_yf_raw = pd.DataFrame(columns=["ticker", "date", "close"])

    discrepancy = compute_price_discrepancy(df_yf_raw, df_tiingo)
    print(f"  Avg close-price difference: {discrepancy['avg_pct_diff']:.4f}%  "
          f"→ {discrepancy['assessment']}")

    # Scan broader sample for outliers (>10% diff) and save to CSV
    outliers_df = scan_price_outliers(df_yf_raw, df_tiingo)
    outliers_path = RESULTS_DIR / "price_outliers.csv"
    outliers_df.to_csv(outliers_path, index=False)
    if len(outliers_df) > 0:
        print(f"  {len(outliers_df)} outlier pairs (>10% diff) → {outliers_path.name}")
    else:
        print(f"  No outlier pairs found (>10% diff) → {outliers_path.name} (empty)")

    # Step 9: Coverage
    yf_tickers     = set(df_yf_raw["ticker"].unique()) if not df_yf_raw.empty else set(tickers)
    tiingo_tickers = set(df_tiingo["ticker"].unique())

    # Step 10: Build and save report
    print("\n--- Step 6: Building Validation Report ---")
    report = build_report(
        yf_stats, tiingo_stats,
        yf_tickers, tiingo_tickers,
        discrepancy, skipped,
    )
    report_path = RESULTS_DIR / "data_validation_report.txt"
    report_path.write_text(report)
    print(f"  Saved → {report_path.name}")
    print(f"\nAll outputs saved to {RESULTS_DIR}/\n")
    print(report)


if __name__ == "__main__":
    main()
