"""
crypto/fear_greed_test.py
==========================
Two new data sources tested against the Var1+Var2+Var4 baseline:

  Part 1 — Hourly exit simulation:
    Load 90 days of Coinbase hourly OHLCV. Re-simulate exits for the last
    12 LONG trades using hourly bars. Compare result to daily bar simulator.
    Quantifies the daily bar approximation error and validates the TIME_SAT
    time stop logic at hourly resolution.

  Part 2 — Fear & Greed distribution:
    Fetch full F&G history from Alternative.me. Join to Friday signal dates.
    Print distribution across signal days and avg R by F&G bucket.

  Part 3 — Fear & Greed as hard filter (4 variants):
    Avoid Extreme Fear (<25), Avoid Extreme Greed (>75),
    Neutral only (25-75), Tighter neutral (20-80).
    Promotion criteria: >0.02R improvement, ≥30 trades, both instruments.

  Part 4 — Fear & Greed as sizing modifier:
    Extreme Fear (<25) → 0.5× risk, Neutral (25-75) → 1.0×,
    Extreme Greed (>75) → 0.75× risk.
    Metric: ROI / equity growth (not avg_R — see volatility compression test).

──────────────────────────────────────────────────────────────────────────
TEST RESULT — March 2026
──────────────────────────────────────────────────────────────────────────

Test date : March 2026
Verdict   : DO NOT PROMOTE — F&G removes best trades in every configuration;
            sizing modifier downsizes exactly the highest-performing zones.

Baseline (F&G era, 2018-02-01 onward):
  BTC  93 trades  avg R +0.019  WR 25.8%  MaxDD -30.1%
  ETH  94 trades  avg R +0.189  WR 35.1%  MaxDD -22.8%
  Note: BTC baseline lower than full-history backtest — F&G era excludes
  the pre-2018 bull run. ETH era baseline is representative.

────────────────────────────────────────────────────────────────────
PART 1 — HOURLY vs DAILY EXIT COMPARISON
────────────────────────────────────────────────────────────────────
  Coverage: 3 BTC / 2 ETH trades within 90-day hourly window.
  All 5 trades had exit-type mismatches (daily TIME or STOP_BE vs
  hourly TIME_SAT at 20:00 UTC Saturday).

  Hourly exits were consistently higher-priced than daily:
    BTC avg delta : +0.431R  (range: +0.118R to +0.764R)
    ETH avg delta : +0.316R  (range: +0.235R to +0.398R)

  Interpretation: daily simulator exits at Monday close (TIME) or
  Saturday close (STOP_BE), both of which fall below the Saturday
  15:00 ET (20:00 UTC) price captured by hourly. In the recent
  bearish environment, prices drifted down from Saturday afternoon
  through Monday — hourly TIME_SAT captured the higher price.

  Limitation: only 5 trades in the hourly window. Direction is
  consistent (hourly better) but magnitude cannot be extrapolated.
  The daily approximation overstates downside by roughly 0.3–0.4R
  in this sample.

  Future action: run live trades with hourly simulation once 20+
  weekend exits are in the hourly window. Flag if systemic gap
  exceeds 0.15R.

────────────────────────────────────────────────────────────────────
PART 2 — F&G DISTRIBUTION ACROSS SIGNAL DATES
────────────────────────────────────────────────────────────────────
  BTC distribution (93 LONG signal dates with F&G):
    Extreme Fear (<25)   : n=4   avg R +0.486  WR 50%   ← best bucket
    Fear (25-44)         : n=12  avg R -0.034  WR 8%    ← worst bucket
    Neutral (45-54)      : n=16  avg R +0.102  WR 31%
    Greed (55-74)        : n=36  avg R -0.064  WR 22%   ← most common, underperforms
    Extreme Greed (≥75)  : n=25  avg R +0.037  WR 32%

  ETH distribution (94 LONG signal dates with F&G):
    Extreme Fear (<25)   : n=5   avg R +0.283  WR 20%
    Fear (25-44)         : n=17  avg R -0.117  WR 24%   ← worst bucket
    Neutral (45-54)      : n=16  avg R +0.242  WR 38%
    Greed (55-74)        : n=37  avg R +0.150  WR 32%   ← most common, OK
    Extreme Greed (≥75)  : n=19  avg R +0.469  WR 53%   ← best bucket

  Structural observation:
    BTC and ETH perform best in OPPOSITE F&G zones:
      BTC best  : Extreme Fear (market panic → BTC weekend reversal)
      ETH best  : Extreme Greed (market euphoria → ETH momentum extension)
    Fear (25-44) is the worst performing bucket for both instruments.
    This pattern is directionally consistent with the BTC/ETH asymmetry
    found in Tier 2 (market structure, volume, R targets). BTC and ETH
    respond to different macro sentiment conditions.

────────────────────────────────────────────────────────────────────
PART 3 — HARD FILTER RESULTS
────────────────────────────────────────────────────────────────────
  All 4 variants fail on both instruments. Verdicts:
    No Extr.Fear (<25)    : BTC -0.021R  ETH -0.006R  →  DO NOT PROMOTE
    No Extr.Greed (>75)   : BTC -0.021R  ETH -0.032R  →  DO NOT PROMOTE
    Neutral only (25-75)  : BTC -0.049R  ETH -0.040R  →  DO NOT PROMOTE
    Tighter (20-80)       : BTC -0.015R  ETH -0.033R  →  DO NOT PROMOTE

  Removed vs kept analysis — every variant removes ABOVE-average trades:
    BTC: removing Extreme Fear takes out +0.486R trades (vs -0.002R kept)
    BTC: removing Extreme Greed removes +0.095R trades (vs -0.002R kept)
    ETH: removing Extreme Greed removes +0.356R trades (vs +0.157R kept)
    ETH: removing Extreme Fear removes +0.283R trades (vs +0.183R kept)

  Root cause: F&G tails (Extreme Fear + Extreme Greed) contain the
  system's best-performing trades. Any filter that excludes a tail
  removes positive expectancy. No configuration avoids this.

  Note on Fear (25-44): this is the worst bucket (BTC -0.034R,
  ETH -0.117R) but filtering for "only Fear" would leave <15% of
  trades on both instruments — far below the 30-trade minimum.

────────────────────────────────────────────────────────────────────
PART 4 — SIZING MODIFIER (ROI metric)
────────────────────────────────────────────────────────────────────
  Sizing rule: Ext.Fear → 0.5×, Neutral → 1.0×, Ext.Greed → 0.75×

  BTC: flat 1.6% ROI → F&G sizing -4.3% ROI  (delta = -5.9pp)
       MaxDD worsens: -30.1% → -33.8%
  ETH: flat 122.1% ROI → F&G sizing 103.2% ROI  (delta = -18.9pp)
       MaxDD marginal improvement: -22.8% → -22.6%

  Result: sizing DOWN on Extreme Fear hurts BTC (its best bucket).
  Sizing DOWN on Extreme Greed hurts ETH (its best bucket).
  The modifier consistently downsizes the highest-performing zones
  on each instrument — the inverse of what a helpful sizing rule
  should do.

  Alternative sizing not tested: Ext.Fear 1.5× for BTC, Ext.Greed
  1.5× for ETH (instrument-specific). Deferred to post-Phase-1
  when instrument-specific parameter sets are refactored.

────────────────────────────────────────────────────────────────────
STRUCTURAL OBSERVATION
────────────────────────────────────────────────────────────────────
  The F&G test adds a fourth data point to the BTC/ETH asymmetry:
    BTC outperforms in Extreme Fear; ETH outperforms in Extreme Greed.
    This is directionally consistent with BTC's volume-floor finding
    (BTC needs broad-market participation signals) and ETH's Extreme
    Greed environment (risk-on sentiment extends ETH momentum).

  F&G is not a filter candidate given current signal structure.
  It may be valuable as a context variable for instrument-specific
  sizing once per-ticker parameter sets are built out.

Status   : DO NOT PROMOTE
Next test : Phase 1 live trading (system is locked — no further
            Tier 2 tests until 8 weeks of live data)
──────────────────────────────────────────────────────────────────────────
"""

import sys
import warnings
import time
from pathlib import Path

import requests
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, load_hourly_data, get_ticker_df
from backtest_engine import (
    run_backtest, _simulate_exit_hourly, STARTING_CAPITAL, REGIMES
)

TICKERS = ["BTC-USD", "ETH-USD"]
FG_API  = "https://api.alternative.me/fng/?limit=0&format=json"


# ── Fear & Greed loader ───────────────────────────────────────────────────────

def load_fear_greed() -> pd.DataFrame:
    """
    Fetch complete F&G history from Alternative.me (free, no auth).
    Returns DataFrame indexed by UTC-naive date with columns:
        value (int 0-100), classification (str)
    """
    print("  Fetching Fear & Greed Index history...")
    resp = requests.get(FG_API, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"F&G API error: HTTP {resp.status_code} — {resp.text[:200]}"
        )
    data = resp.json()["data"]
    rows = []
    for item in data:
        rows.append({
            "date":           pd.Timestamp(int(item["timestamp"]), unit="s").normalize(),
            "value":          int(item["value"]),
            "classification": item["value_classification"],
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    print(f"    {len(df)} F&G records  |  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def fg_bucket(value: int | None) -> str:
    if value is None:
        return "No Data"
    if value < 25:
        return "Extreme Fear (<25)"
    if value < 45:
        return "Fear (25-44)"
    if value < 55:
        return "Neutral (45-54)"
    if value < 75:
        return "Greed (55-74)"
    return "Extreme Greed (≥75)"


# ── Part 1: Hourly vs Daily exit comparison ───────────────────────────────────

def run_hourly_comparison(
    base_log: pd.DataFrame,
    ticker_df: pd.DataFrame,
    hourly: pd.DataFrame,
    ticker: str,
) -> None:
    longs = (
        base_log[base_log["direction"] == "LONG"]
        .dropna(subset=["R_multiple"])
        .sort_values("date")
        .tail(12)
    )

    if longs.empty:
        print(f"  {ticker}: no LONG trades in hourly window")
        return

    print(f"\n  {ticker} — hourly vs daily exit comparison (last {len(longs)} LONGs):")
    print(f"  {'Date':<12} {'Entry':>8} {'Stop':>8} "
          f"{'Daily_type':<10} {'Daily_px':>9} "
          f"{'Hourly_type':<11} {'Hourly_px':>9} {'Match':>6} {'R_delta':>8}")
    print("  " + "-" * 88)

    mismatches = 0
    r_deltas   = []

    for _, row in longs.iterrows():
        fri_date = row["date"]
        entry    = float(row["entry_price"])
        stop     = float(row["stop_price"])
        target   = float(row["target_price"])
        risk     = entry - stop

        daily_etype = row["exit_type"]
        daily_epx   = float(row["exit_price"])
        daily_r     = float(row["R_multiple"])

        h_price, h_type, _ = _simulate_exit_hourly(
            hourly, entry, stop, target, fri_date
        )

        if h_type == "NO_DATA":
            print(f"  {str(fri_date.date()):<12}  — no hourly data for this weekend —")
            continue

        # Compute hourly R (approximate — no fee recalculation)
        h_r      = (h_price - entry) / risk if risk > 0 else 0.0
        r_delta  = h_r - daily_r
        match    = "✓" if h_type == daily_etype else "✗"
        if h_type != daily_etype:
            mismatches += 1
        r_deltas.append(r_delta)

        print(f"  {str(fri_date.date()):<12} {entry:>8.1f} {stop:>8.1f} "
              f"{daily_etype:<10} {daily_epx:>9.1f} "
              f"{h_type:<11} {h_price:>9.1f} {match:>6} {r_delta:>+8.3f}R")

    print(f"\n  Exit type mismatches : {mismatches} / {len(r_deltas)}")
    if r_deltas:
        print(f"  Avg R delta (hourly - daily) : {np.mean(r_deltas):>+.3f}R")
        print(f"  Max abs R delta              : {max(abs(d) for d in r_deltas):>.3f}R")


# ── Part 2: F&G distribution ──────────────────────────────────────────────────

def fg_distribution(base_log: pd.DataFrame, fg_df: pd.DataFrame, ticker: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"]).copy()
    longs = longs[longs["fear_greed_value"].notna()]

    if longs.empty:
        print(f"  {ticker}: no trades with F&G data")
        return

    longs["bucket"] = longs["fear_greed_value"].apply(
        lambda v: fg_bucket(int(v)) if pd.notna(v) else "No Data"
    )

    buckets_order = [
        "Extreme Fear (<25)", "Fear (25-44)", "Neutral (45-54)",
        "Greed (55-74)", "Extreme Greed (≥75)"
    ]

    print(f"\n  {ticker} — F&G distribution across LONG signal dates:")
    print(f"  {'Bucket':<24} {'n':>5} {'Pct%':>6} {'WR%':>6} {'AvgR':>8} {'PF':>6}")
    print("  " + "-" * 58)

    for bucket in buckets_order:
        sub = longs[longs["bucket"] == bucket]
        if sub.empty:
            continue
        n    = len(sub)
        pct  = n / len(longs) * 100
        wr   = (sub["R_multiple"] > 0).mean() * 100
        avr  = sub["R_multiple"].mean()
        wins = sub[sub["R_multiple"] > 0]["R_multiple"].sum()
        loss = abs(sub[sub["R_multiple"] <= 0]["R_multiple"].sum())
        pf   = wins / loss if loss > 0 else float("inf")
        print(f"  {bucket:<24} {n:>5} {pct:>5.1f}% {wr:>5.1f}% {avr:>+8.3f} {pf:>6.3f}")

    print(f"  {'Total (with F&G data)':<24} {len(longs):>5}")


# ── Part 3: Filter tests ──────────────────────────────────────────────────────

def run_filter_tests(
    ticker_df: pd.DataFrame,
    ticker: str,
    fg_df: pd.DataFrame,
    base_m: dict,
) -> dict[str, str]:
    variants = {
        "No Extr.Fear (<25)":  dict(fear_greed_min=25, fear_greed_max=100),
        "No Extr.Greed (>75)": dict(fear_greed_min=0,  fear_greed_max=75),
        "Neutral only (25-75)": dict(fear_greed_min=25, fear_greed_max=75),
        "Tighter (20-80)":     dict(fear_greed_min=20, fear_greed_max=80),
    }

    # Baseline restricted to F&G era for fair comparison
    fg_start = fg_df.index.min()
    base_era_log, base_era_m, _ = run_backtest(
        ticker_df[ticker_df.index >= fg_start], ticker,
        fear_greed_df=fg_df,  # stores values, no filter (min=0 max=100)
    )

    print(f"\n{'='*80}")
    print(f"  F&G FILTER TESTS — {ticker}  (F&G era: {fg_start.date()} – present)")
    print(f"{'='*80}")

    hdr = (f"  {'Variant':<22} {'Trades':>7} {'WR%':>6} {'AvgR':>8} "
           f"{'PF':>6} {'MaxDD':>7} {'delta_R':>8} {'Removed':>8}")
    print(hdr)
    print("  " + "-" * 74)

    if base_era_m:
        print(f"  {'Baseline (era)':22} {base_era_m['n_trades']:>7} "
              f"{base_era_m['win_rate']:>5}% {base_era_m['avg_R']:>+8.3f} "
              f"{base_era_m['profit_factor']:>6.3f} {base_era_m['max_drawdown']:>6}%"
              f"{'    base':>9}")

    verdicts: dict[str, str] = {}
    for label, kwargs in variants.items():
        vlog, vm, _ = run_backtest(
            ticker_df[ticker_df.index >= fg_start], ticker,
            fear_greed_df=fg_df, **kwargs,
        )
        if not vm or not base_era_m:
            print(f"  {label:<22}  — no trades —")
            continue

        removed  = base_era_m["n_trades"] - vm["n_trades"]
        delta    = vm["avg_R"] - base_era_m["avg_R"]
        ok_r     = delta > 0.02
        ok_n     = vm["n_trades"] >= 30
        verdict  = "PROMOTE" if (ok_r and ok_n) else "DNP"
        verdicts[label] = "PROMOTE" if verdict == "PROMOTE" else "DO NOT PROMOTE"

        print(f"  {label:<22} {vm['n_trades']:>7} {vm['win_rate']:>5}%"
              f" {vm['avg_R']:>+8.3f} {vm['profit_factor']:>6.3f}"
              f" {vm['max_drawdown']:>6}% {delta:>+8.3f} {removed:>8}  [{verdict}]")

    # Filtered vs kept quality
    print(f"\n  {ticker} — avg R: removed trades vs kept (from era baseline):")
    for label, kwargs in variants.items():
        fg_min = kwargs["fear_greed_min"]
        fg_max = kwargs["fear_greed_max"]
        era_longs = base_era_log[
            (base_era_log["direction"] == "LONG") &
            base_era_log["fear_greed_value"].notna()
        ].dropna(subset=["R_multiple"])

        removed = era_longs[
            (era_longs["fear_greed_value"] < fg_min) |
            (era_longs["fear_greed_value"] > fg_max)
        ]
        kept = era_longs[
            (era_longs["fear_greed_value"] >= fg_min) &
            (era_longs["fear_greed_value"] <= fg_max)
        ]
        rem_r  = removed["R_multiple"].mean() if not removed.empty else float("nan")
        kept_r = kept["R_multiple"].mean()    if not kept.empty    else float("nan")
        rem_s  = f"{rem_r:>+8.3f}" if not pd.isna(rem_r) else "     N/A"
        kpt_s  = f"{kept_r:>+8.3f}" if not pd.isna(kept_r) else "     N/A"
        print(f"    {label:<22}  removed n={len(removed):>3} avg={rem_s}  "
              f"kept n={len(kept):>3} avg={kpt_s}")

    return verdicts


# ── Part 4: Sizing modifier ───────────────────────────────────────────────────

def run_sizing_modifier(
    ticker_df: pd.DataFrame,
    ticker: str,
    fg_df: pd.DataFrame,
) -> None:
    fg_start = fg_df.index.min()
    era_df   = ticker_df[ticker_df.index >= fg_start]

    base_log, base_m, _ = run_backtest(era_df, ticker, fear_greed_df=fg_df)
    sz_log,   sz_m,   _ = run_backtest(era_df, ticker,
                                        fear_greed_df=fg_df,
                                        fear_greed_sizing=True)

    if not base_m or not sz_m:
        print(f"  {ticker}: insufficient data for sizing modifier")
        return

    base_roi = base_m["roi_pre_tax"]
    sz_roi   = sz_m["roi_pre_tax"]
    base_dd  = base_m["max_drawdown"]
    sz_dd    = sz_m["max_drawdown"]

    base_eq  = STARTING_CAPITAL * (1 + base_roi / 100)
    sz_eq    = STARTING_CAPITAL * (1 + sz_roi  / 100)

    print(f"\n  {ticker} — F&G sizing modifier (Ext.Fear 0.5× / Neutral 1.0× / Ext.Greed 0.75×):")
    print(f"  {'Version':<22} {'ROI%':>7} {'Final_eq':>10} {'MaxDD%':>8} {'ROI_delta':>10}")
    print("  " + "-" * 60)
    print(f"  {'Flat sizing':22} {base_roi:>6}%  ${base_eq:>9,.2f} {base_dd:>7}%")
    print(f"  {'F&G sizing':22} {sz_roi:>6}%  ${sz_eq:>9,.2f} {sz_dd:>7}%  "
          f"delta={sz_roi - base_roi:>+.1f}pp")

    # Also show how many trades fell in each zone
    era_longs = sz_log[sz_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    if not era_longs.empty and "fear_greed_value" in era_longs.columns:
        fv = era_longs["fear_greed_value"].dropna()
        ef = (fv < 25).sum()
        nu = ((fv >= 25) & (fv <= 75)).sum()
        eg = (fv > 75).sum()
        print(f"\n    Trades by zone: Ext.Fear(0.5×)={ef}  Neutral(1.0×)={nu}  Ext.Greed(0.75×)={eg}")


# ── Part 5: Regime breakdown (if promoted) ────────────────────────────────────

def regime_breakdown(
    ticker: str,
    ticker_df: pd.DataFrame,
    fg_df: pd.DataFrame,
    fg_min: int,
    fg_max: int,
    label: str,
) -> list[str]:
    fg_start = fg_df.index.min()
    era_df   = ticker_df[ticker_df.index >= fg_start]

    _, base_rm, _ = run_backtest(era_df, ticker, fear_greed_df=fg_df)
    _, filt_rm, _ = run_backtest(era_df, ticker, fear_greed_df=fg_df,
                                  fear_greed_min=fg_min, fear_greed_max=fg_max)

    print(f"\n  Regime breakdown — {ticker} {label}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Flt_T':>6} {'Flt_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 67)
    regressions = []
    for name in REGIMES:
        bm  = base_rm.get(name, {})
        mm  = filt_rm.get(name, {})
        b_t = bm.get("n_trades", 0)
        m_t = mm.get("n_trades", 0)
        if b_t == 0 and m_t == 0:
            continue
        b_r   = bm.get("avg_R", float("nan"))
        m_r   = mm.get("avg_R", float("nan"))
        delta = m_r - b_r if (b_t > 0 and m_t > 0) else float("nan")
        ds    = f"{delta:>+8.3f}" if not pd.isna(delta) else "     N/A"
        flag  = "  ← regression" if (not pd.isna(delta) and delta < -0.02) else ""
        print(f"  {name:<22} {b_t:>7} {b_r:>10.3f} {m_t:>6} {m_r:>10.3f} {ds}{flag}")
        if not pd.isna(delta) and delta < -0.02:
            regressions.append(name)
    return regressions


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 80)
    print("Fear & Greed + Hourly Data Test  —  Var1+Var2+Var4 baseline")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading daily data...")
    data = load_crypto_data(TICKERS)

    print("\nLoading hourly data (90 days)...")
    try:
        hourly_data = load_hourly_data(TICKERS, days=90)
        hourly_ok = True
    except Exception as e:
        print(f"  WARNING: hourly data failed — {e}")
        hourly_ok = False

    print("\nLoading Fear & Greed Index...")
    fg_df = load_fear_greed()

    # Run baseline (full history) to store F&G values on rows
    all_base_logs: dict[str, pd.DataFrame] = {}
    all_base_m:    dict[str, dict]         = {}
    for ticker in TICKERS:
        tdf = get_ticker_df(data, ticker)
        log, m, _ = run_backtest(tdf, ticker, fear_greed_df=fg_df)
        all_base_logs[ticker] = log
        all_base_m[ticker]    = m

    # ── PART 1: Hourly vs daily comparison ────────────────────────────────
    print(f"\n{'='*80}")
    print("  PART 1 — HOURLY vs DAILY EXIT COMPARISON (last 12 LONGs)")
    print(f"{'='*80}")
    print("  APPROXIMATION NOTE: daily bar simulator uses full-day H/L for")
    print("  stop/target; hourly shows exact intrabar timing within the hour.")

    if hourly_ok:
        for ticker in TICKERS:
            tdf = get_ticker_df(data, ticker)
            run_hourly_comparison(
                all_base_logs[ticker], tdf,
                hourly_data[ticker], ticker,
            )
    else:
        print("  Skipped — hourly data unavailable.")

    # ── PART 2: F&G distribution ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PART 2 — FEAR & GREED DISTRIBUTION ACROSS SIGNAL DATES")
    print(f"{'='*80}")
    for ticker in TICKERS:
        fg_distribution(all_base_logs[ticker], fg_df, ticker)

    # ── PART 3: Filter tests ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PART 3 — FEAR & GREED AS HARD FILTER")
    print(f"{'='*80}")

    all_filter_verdicts: dict[str, dict[str, str]] = {}
    for ticker in TICKERS:
        tdf = get_ticker_df(data, ticker)
        v   = run_filter_tests(tdf, ticker, fg_df, all_base_m[ticker])
        all_filter_verdicts[ticker] = v

    # Combined verdict
    print(f"\n{'='*80}")
    print("  COMBINED FILTER VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*80}")
    variant_labels = [
        "No Extr.Fear (<25)", "No Extr.Greed (>75)",
        "Neutral only (25-75)", "Tighter (20-80)",
    ]
    promoted_filters = []
    for label in variant_labels:
        btc_v = all_filter_verdicts.get("BTC-USD", {}).get(label, "—")
        eth_v = all_filter_verdicts.get("ETH-USD", {}).get(label, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {label:<22}: BTC={btc_v}  ETH={eth_v}  →  {combined}")
        if combined == "PROMOTE":
            promoted_filters.append(label)

    # Regime breakdown if promoted
    fg_params_map = {
        "No Extr.Fear (<25)":   (25, 100),
        "No Extr.Greed (>75)":  (0,  75),
        "Neutral only (25-75)": (25, 75),
        "Tighter (20-80)":      (20, 80),
    }
    if promoted_filters:
        print(f"\n{'='*80}")
        print("  REGIME BREAKDOWN — PROMOTED FILTERS")
        print(f"{'='*80}")
        for label in promoted_filters:
            fg_min, fg_max = fg_params_map[label]
            for ticker in TICKERS:
                tdf  = get_ticker_df(data, ticker)
                regs = regime_breakdown(ticker, tdf, fg_df, fg_min, fg_max, label)
                if regs:
                    print(f"    Regressions in: {', '.join(regs)}")
                else:
                    print(f"    No regime regressions.")
    else:
        print("\n  No filters promoted — regime breakdown skipped.")

    # ── PART 4: Sizing modifier ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PART 4 — FEAR & GREED SIZING MODIFIER (ROI metric, not avg_R)")
    print(f"{'='*80}")
    print("  Sizing: Extreme Fear (<25) → 0.5×  |  Neutral → 1.0×  |  Greed (>75) → 0.75×")
    for ticker in TICKERS:
        tdf = get_ticker_df(data, ticker)
        run_sizing_modifier(tdf, ticker, fg_df)

    # ── PART 5: Summary ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PART 5 — SUMMARY")
    print(f"{'='*80}")
    print(f"  Hourly data loaded      : {'Yes' if hourly_ok else 'No'}")
    for ticker in TICKERS:
        log = all_base_logs[ticker]
        with_fg = log[log["fear_greed_value"].notna() & (log["direction"] == "LONG")]
        pct     = len(with_fg) / max(len(log[log["direction"]=="LONG"]), 1) * 100
        print(f"  {ticker} F&G coverage    : {len(with_fg)} / "
              f"{len(log[log['direction']=='LONG'])} LONG trades  ({pct:.1f}%)")
    print(f"\n  Filter promotion summary:")
    for label in variant_labels:
        btc_v = all_filter_verdicts.get("BTC-USD", {}).get(label, "—")
        eth_v = all_filter_verdicts.get("ETH-USD", {}).get(label, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"    {label:<22}: {combined}")
    print()


if __name__ == "__main__":
    main()
