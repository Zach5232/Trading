"""
crypto/binance_funding_backtest.py
===================================
Funding rate filter backtest using Binance public historical data
from data.binance.vision (coin-margined perpetuals, no auth required).

Data source:
  https://data.binance.vision/data/futures/cm/monthly/fundingRate/
  Symbols: BTCUSD_PERP, ETHUSD_PERP
  Coverage: 2022-07 onward (earlier months return 404)

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Data:          Binance data.binance.vision cm/monthly/fundingRate,
#                BTCUSD_PERP and ETHUSD_PERP, 3,911 records per symbol,
#                Jul 2022 – Feb 2026, 97.6–97.7% Friday coverage.
# Verdict:       DO NOT PROMOTE — filter has nothing to filter
# Key finding:   95%+ of all LONG signals fall in NEUTRAL territory. Zero
#                ELEVATED or EXTREME signals on BTC across the entire window.
#                The three-filter system already acts as a natural funding rate
#                gate — it only fires in confirmed uptrends, and confirmed
#                uptrends coincide with neutral funding. The scenarios where a
#                funding rate filter would be relevant (crowded longs, extreme
#                positive funding) don't overlap with signal dates that pass all
#                three filters.
# Conservative:  Removes 0 BTC trades, 1 ETH trade (+0.007R — noise).
# Aggressive:    Removes nothing on either ticker.
# Status:        Keep as display-only in main.py. Revisit only if the system
#                begins generating signals in elevated funding environments,
#                which would suggest a regime change worth monitoring separately.
# ─────────────────────────────────────────────────────────────────────────────

Method:
  1. Download all available monthly zip files, parse in-memory
  2. Join to Friday signal dates using last settlement at/before 16:00 UTC
  3. Compare Baseline / Conservative / Aggressive on 2022-07–present window

Classification thresholds (per-8h rate, %):
  NEGATIVE  : < -0.01%
  NEUTRAL   : -0.01% to 0.05%
  ELEVATED  : 0.05% to 0.15%
  EXTREME   : > 0.15%

Promotion criteria:
  - Net avg R improvement > 0.02R vs baseline
  - Sample stays above 30 trades
  - Improvement holds on BOTH BTC and ETH independently
"""

import io
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import run_backtest, STARTING_CAPITAL, _calc_metrics

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = (
    "https://data.binance.vision/data/futures/cm/monthly/fundingRate"
    "/{symbol}/{symbol}-fundingRate-{yyyy_mm}.zip"
)
BINANCE_SYMBOL_MAP = {"BTC-USD": "BTCUSD_PERP", "ETH-USD": "ETHUSD_PERP"}
CACHE_DIR    = Path(__file__).parent.parent / "Data" / "backtest_cache"
WINDOW_START = "2022-07-01"
REQUEST_TIMEOUT = 20

# Rate thresholds (per-8h %)
THRESHOLDS = [("NEGATIVE", -999, -0.01),
              ("NEUTRAL",  -0.01,  0.05),
              ("ELEVATED",  0.05,  0.15),
              ("EXTREME",   0.15,  999)]

def _classify(rate_pct: float) -> str:
    for label, lo, hi in THRESHOLDS:
        if lo <= rate_pct < hi:
            return label
    return "EXTREME"


# ── Data loading ─────────────────────────────────────────────────────────────

def _months_in_range(start_ym: tuple, end_ym: tuple) -> list[tuple]:
    """Generate (year, month) tuples from start_ym to end_ym inclusive."""
    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def load_binance_funding_rates(symbol: str) -> pd.DataFrame:
    """
    Download all available monthly zip files for symbol from data.binance.vision.
    Returns DataFrame with columns: timestamp (UTC datetime), rate_pct (float).
    Missing months (404) are silently skipped.
    """
    cache_file = CACHE_DIR / f"binance_funding_{symbol.lower()}.csv"
    if cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df

    now = datetime.now(timezone.utc)
    months = _months_in_range((2022, 7), (now.year, now.month))
    all_rows = []

    print(f"  Fetching {symbol} — {len(months)} months to check...")
    for year, month in months:
        yyyy_mm = f"{year}-{month:02d}"
        url = BASE_URL.format(symbol=symbol, yyyy_mm=yyyy_mm)
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    {yyyy_mm}: request error — {e}")
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    chunk = pd.read_csv(f)
        except Exception as e:
            print(f"    {yyyy_mm}: parse error — {e}")
            continue

        # Normalise column names (Binance sometimes varies capitalisation)
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if "calc_time" not in chunk.columns or "last_funding_rate" not in chunk.columns:
            print(f"    {yyyy_mm}: unexpected columns {list(chunk.columns)}")
            continue

        chunk = chunk[["calc_time", "last_funding_rate"]].copy()
        chunk = chunk.dropna()
        chunk = chunk[chunk["last_funding_rate"] != 0]
        chunk["timestamp"] = pd.to_datetime(
            chunk["calc_time"].astype(np.int64), unit="ms", utc=True
        ).dt.tz_localize(None)
        chunk["rate_pct"] = chunk["last_funding_rate"].astype(float) * 100
        all_rows.append(chunk[["timestamp", "rate_pct"]])
        print(f"    {yyyy_mm}: {len(chunk)} records")
        time.sleep(0.1)

    if not all_rows:
        raise RuntimeError(f"No data downloaded for {symbol}")

    df = pd.concat(all_rows, ignore_index=True).sort_values("timestamp")
    df = df.drop_duplicates("timestamp").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"  Saved {len(df)} records → {cache_file.name}")
    return df


# ── Friday rate lookup ────────────────────────────────────────────────────────

def _get_friday_rate(
    funding_df: pd.DataFrame, fri_date: pd.Timestamp
) -> tuple[float | None, str]:
    """
    Return the most recent funding rate at or before Friday 16:00 UTC.
    If nothing within 24h window (Thu 16:00 – Fri 16:00), return (None, 'NO_DATA').
    """
    cutoff  = pd.Timestamp(fri_date.date()) + pd.Timedelta(hours=16)
    earliest = cutoff - pd.Timedelta(hours=24)
    window = funding_df[
        (funding_df["timestamp"] >= earliest) &
        (funding_df["timestamp"] <= cutoff)
    ]
    if window.empty:
        return None, "NO_DATA"
    rate_pct = float(window.iloc[-1]["rate_pct"])
    return rate_pct, _classify(rate_pct)


# ── Filter application ────────────────────────────────────────────────────────

def _apply_filter(
    trade_log: pd.DataFrame,
    funding_df: pd.DataFrame,
    skip_classes: set[str],
) -> pd.DataFrame:
    """
    Attach funding rate/class to each LONG trade row.
    Trades whose class is in skip_classes become NO_TRADE (direction set, R zeroed).
    Returns a modified copy restricted to WINDOW_START.
    """
    tl = trade_log[trade_log["date"] >= WINDOW_START].copy()
    rates, classes = [], []
    for _, row in tl.iterrows():
        if row["direction"] != "LONG":
            rates.append(None); classes.append("NO_DATA"); continue
        r, c = _get_friday_rate(funding_df, row["date"])
        rates.append(r); classes.append(c)
    tl["funding_rate_pct"] = rates
    tl["funding_class"]    = classes

    if skip_classes:
        mask = (tl["direction"] == "LONG") & (tl["funding_class"].isin(skip_classes))
        tl.loc[mask, "direction"]   = "NO_TRADE"
        tl.loc[mask, "R_multiple"]  = None
        tl.loc[mask, "profit_loss"] = 0.0
    return tl


# ── Reporting ─────────────────────────────────────────────────────────────────

def _funding_distribution(trade_log: pd.DataFrame, funding_df: pd.DataFrame) -> dict:
    """Count funding classes across all LONG signal Fridays in the window."""
    longs = trade_log[
        (trade_log["direction"] == "LONG") & (trade_log["date"] >= WINDOW_START)
    ]
    counts: dict[str, int] = {c: 0 for c in ["NEGATIVE","NEUTRAL","ELEVATED","EXTREME","NO_DATA"]}
    for _, row in longs.iterrows():
        _, cls = _get_friday_rate(funding_df, row["date"])
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _print_results(
    ticker: str,
    versions: dict[str, tuple[pd.DataFrame, dict]],
    baseline_key: str,
    funding_dist: dict,
    funding_df: pd.DataFrame,
) -> dict[str, str]:
    base_m = versions[baseline_key][1]
    print(f"\n{'='*76}")
    print(f"  FUNDING RATE FILTER — {ticker}  (window: {WINDOW_START} – present)")
    print(f"{'='*76}")

    # Funding class distribution
    total = sum(funding_dist.values())
    print(f"  Funding class distribution across LONG signals ({WINDOW_START}+):")
    for cls in ["NEGATIVE","NEUTRAL","ELEVATED","EXTREME","NO_DATA"]:
        n = funding_dist.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"    {cls:<10}: {n:>3}  ({pct:.1f}%)")

    # NO_DATA count
    base_tl = versions[baseline_key][0]
    longs_w = base_tl[
        (base_tl["direction"] == "LONG") & (base_tl["date"] >= WINDOW_START)
    ]
    n_nodata = funding_dist.get("NO_DATA", 0)
    print(f"\n  Fridays with clean funding data : {total - n_nodata} / {total}"
          f"  ({(total-n_nodata)/total*100:.1f}% coverage)")

    # Comparison table
    print(f"\n  {'Version':<14} {'Signals':>8} {'Filtered':>9} {'Trades':>7} "
          f"{'WR%':>6} {'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print("  " + "-" * 72)

    verdicts: dict[str, str] = {}
    for name, (tl, m) in versions.items():
        if not m:
            print(f"  {name:<14}  — no trades —")
            continue
        longs_in_window = len(tl[
            (tl["direction"].isin(["LONG","NO_TRADE"])) &
            (tl["date"] >= WINDOW_START) &
            (tl.get("no_trade_reason", pd.Series()) != "below MA20") &
            (tl.get("no_trade_reason", pd.Series()) != "ATR contracting") &
            (tl.get("no_trade_reason", pd.Series()) != "momentum decelerating")
        ]) if name != baseline_key else len(longs_w)

        n_filtered = int((tl.get("no_trade_reason", pd.Series("")) == "funding filter").sum()) \
            if name != baseline_key else 0
        # simpler: count trades removed by funding
        if name != baseline_key:
            n_filtered = int(
                ((tl["direction"] == "NO_TRADE") &
                 tl.get("funding_class", pd.Series()).isin(
                     {"ELEVATED","EXTREME"} if "Conservative" in name else {"EXTREME"}
                 )).sum()
            ) if "funding_class" in tl.columns else 0

        delta_str = "    base" if name == baseline_key else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {name:<14} {total:>8} {n_filtered:>9} {m['n_trades']:>7} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}{low_n}"
        )

        if name != baseline_key:
            delta  = m["avg_R"] - base_m["avg_R"]
            ok_r   = delta > 0.02
            ok_n   = m["n_trades"] >= 30
            verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
            reasons = []
            if not ok_r:
                reasons.append(f"delta={delta:+.3f}R")
            if not ok_n:
                reasons.append(f"only {m['n_trades']} trades")
            verdicts[name] = verdict

    print()
    for name, verdict in verdicts.items():
        m  = versions[name][1]
        delta = m["avg_R"] - base_m["avg_R"] if m else 0
        ok_r  = delta > 0.02
        ok_n  = m["n_trades"] >= 30 if m else False
        reasons = []
        if not ok_r:
            reasons.append(f"delta={delta:+.3f}R (need >+0.02R)")
        if not ok_n:
            reasons.append(f"only {m['n_trades']} trades (need ≥30)")
        suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
        print(f"  {name:<14}: {verdict}{suffix}")

    return verdicts


def _print_filtered_trades(
    ticker: str, version_name: str,
    filtered_tl: pd.DataFrame, skip_classes: set[str],
) -> None:
    """Print individual R-multiples of filtered trades."""
    # Filtered trades are those whose funding_class is in skip_classes
    # but we need to recover R_multiple from baseline (before zeroing)
    # Instead, we stored them as NO_TRADE — but the R was zeroed already.
    # We cross-reference the dates against the full baseline to get R.
    pass  # handled in caller


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 76)
    print("Binance Funding Rate Backtest  —  Var1+Var2+Var4 baseline")
    print(f"Data: data.binance.vision (coin-margined perpetuals)")
    print(f"Window: {WINDOW_START} – present")
    print("=" * 76)

    # ── Load OHLCV ───────────────────────────────────────────────────────
    print("\nLoading OHLCV data...")
    data = load_crypto_data(["BTC-USD", "ETH-USD"])

    all_verdicts: dict[str, dict[str, str]] = {}
    all_base_logs: dict[str, pd.DataFrame] = {}

    for ticker in ["BTC-USD", "ETH-USD"]:
        symbol = BINANCE_SYMBOL_MAP[ticker]
        print(f"\n{'─'*60}")
        print(f"  {ticker}  ({symbol})")
        print(f"{'─'*60}")

        # ── Download funding data ─────────────────────────────────────────
        funding_df = load_binance_funding_rates(symbol)
        print(f"  Coverage: {funding_df['timestamp'].min().date()} → "
              f"{funding_df['timestamp'].max().date()}  ({len(funding_df)} records)")

        # ── Run full baseline backtest ────────────────────────────────────
        ticker_df = get_ticker_df(data, ticker)
        full_log, _, _ = run_backtest(ticker_df, ticker)
        full_log["date"] = pd.to_datetime(full_log["date"])
        all_base_logs[ticker] = full_log

        # ── Funding class distribution ────────────────────────────────────
        funding_dist = _funding_distribution(full_log, funding_df)

        # ── Build three versions ──────────────────────────────────────────
        # Baseline (window-restricted, no filter applied)
        base_w = full_log[full_log["date"] >= WINDOW_START].copy()
        base_m = _calc_metrics(base_w, STARTING_CAPITAL)

        # Conservative: skip ELEVATED + EXTREME
        cons_tl = _apply_filter(full_log, funding_df, {"ELEVATED", "EXTREME"})
        cons_m  = _calc_metrics(cons_tl, STARTING_CAPITAL)

        # Aggressive: skip EXTREME only
        agg_tl = _apply_filter(full_log, funding_df, {"EXTREME"})
        agg_m  = _calc_metrics(agg_tl, STARTING_CAPITAL)

        versions = {
            "Baseline":     (base_w,  base_m),
            "Conservative": (cons_tl, cons_m),
            "Aggressive":   (agg_tl,  agg_m),
        }

        verdicts = _print_results(
            ticker, versions, "Baseline", funding_dist, funding_df
        )
        all_verdicts[ticker] = verdicts

        # ── Per-class R breakdown ─────────────────────────────────────────
        print(f"\n  Avg R by funding class (LONG signals, {WINDOW_START}+):")
        longs_w = base_w[base_w["direction"] == "LONG"].dropna(subset=["R_multiple"])
        # re-attach funding class to baseline window longs
        classes = []
        for _, row in longs_w.iterrows():
            _, cls = _get_friday_rate(funding_df, row["date"])
            classes.append(cls)
        longs_w = longs_w.copy()
        longs_w["funding_class"] = classes
        for cls in ["NEGATIVE","NEUTRAL","ELEVATED","EXTREME","NO_DATA"]:
            subset = longs_w[longs_w["funding_class"] == cls]
            if subset.empty:
                continue
            wr  = (subset["R_multiple"] > 0).mean() * 100
            avr = subset["R_multiple"].mean()
            print(f"    {cls:<10} n={len(subset):>3}  WR={wr:>5.1f}%  AvgR={avr:>+7.3f}")

        # ── Print individual filtered trades if either version promotes ───
        for vname, vtl in [("Conservative", cons_tl), ("Aggressive", agg_tl)]:
            if verdicts.get(vname) != "PROMOTE":
                continue
            skip = {"ELEVATED","EXTREME"} if "Conservative" in vname else {"EXTREME"}
            # Recover R from baseline for filtered dates
            filtered_dates = set()
            if "funding_class" in vtl.columns:
                filtered_dates = set(
                    vtl[(vtl["direction"] == "NO_TRADE") &
                        vtl["funding_class"].isin(skip)]["date"]
                )
            if filtered_dates:
                print(f"\n  {vname} — filtered trades (individual R from baseline):")
                base_longs = base_w[base_w["direction"] == "LONG"].dropna(
                    subset=["R_multiple"]
                )
                removed = base_longs[base_longs["date"].isin(filtered_dates)]
                for _, row in removed.sort_values("date").iterrows():
                    _, cls = _get_friday_rate(funding_df, row["date"])
                    print(f"    {row['date'].date()}  {cls:<10}  R={row['R_multiple']:>+.3f}")
                print(f"    Mean R of filtered trades: "
                      f"{removed['R_multiple'].mean():>+.3f}")

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*76}")
    for vname in ["Conservative", "Aggressive"]:
        btc_v = all_verdicts.get("BTC-USD", {}).get(vname, "—")
        eth_v = all_verdicts.get("ETH-USD", {}).get(vname, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {vname:<14}: BTC={btc_v}  ETH={eth_v}  →  {combined}")

    print("\nDone.")


if __name__ == "__main__":
    main()
