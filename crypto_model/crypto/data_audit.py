"""
crypto/data_audit.py
=====================
Three-part data quality audit for the crypto trading pipeline.

Audit 1 — yfinance weekend bar completeness
  For every LONG signal Friday, checks whether Sat/Sun bars exist in the
  loaded data, breaks down by year, and flags anomalous bars.

Audit 2 — yfinance vs Coinbase price divergence (last 60 days)
  Aligns both sources on date and computes close/high/low/ATR14 differences.
  Flags if ATR14 diverges >2% on any recent Friday.

Audit 3 — prior Friday lookup gaps
  Counts how many LONG signals fired the default-allow because no prior
  Friday existed in the data (Filters 2 & 3 silently skipped).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, load_coinbase_live, get_ticker_df, get_fridays
from backtest_engine import run_backtest

TICKERS = ["BTC-USD", "ETH-USD"]
ATR_DIVERGENCE_THRESHOLD = 0.02   # 2% — flag if exceeded


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT 1 — Weekend bar completeness
# ═══════════════════════════════════════════════════════════════════════════════

def audit_weekend_bars(ticker_df: pd.DataFrame, ticker: str) -> str:
    """Returns 'CLEAN' or 'NEEDS ATTENTION'."""
    trade_log, _, _ = run_backtest(ticker_df, ticker)
    longs = trade_log[trade_log["direction"] == "LONG"].copy()
    longs["date"] = pd.to_datetime(longs["date"])

    total = len(longs)
    both_present = sat_only_missing = sun_only_missing = both_missing = 0
    exit_sat = exit_sun = exit_nodata = 0
    by_year: dict[int, dict] = {}
    anomalous_bars: list[str] = []

    for _, row in longs.iterrows():
        fri_date = row["date"]
        year     = fri_date.year

        if year not in by_year:
            by_year[year] = {"signals": 0, "both": 0, "sat_missing": 0,
                             "sun_missing": 0, "both_missing": 0, "no_data": 0}
        by_year[year]["signals"] += 1

        fri_pos   = ticker_df.index.get_loc(fri_date)
        next_bars = ticker_df.iloc[fri_pos + 1 : fri_pos + 4]
        sat_bars  = next_bars[next_bars.index.dayofweek == 5]
        sun_bars  = next_bars[next_bars.index.dayofweek == 6]
        has_sat   = not sat_bars.empty
        has_sun   = not sun_bars.empty

        if has_sat and has_sun:
            both_present += 1
            by_year[year]["both"] += 1
        elif not has_sat and has_sun:
            sat_only_missing += 1
            by_year[year]["sat_missing"] += 1
        elif has_sat and not has_sun:
            sun_only_missing += 1
            by_year[year]["sun_missing"] += 1
        else:
            both_missing += 1
            by_year[year]["both_missing"] += 1

        # Exit day
        etype = row["exit_type"]
        if etype == "NO_DATA":
            exit_nodata += 1
            by_year[year]["no_data"] += 1
        else:
            # Determine which day exit hit
            weekend_bars = next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()
            if not weekend_bars.empty:
                entry = row["entry_price"]
                stop  = row["stop_price"]
                tgt   = row["target_price"]
                exit_day = None
                for bar_date, bar in weekend_bars.iterrows():
                    if etype in ("STOP", "TRAIL") and bar["low"] <= stop:
                        exit_day = bar_date.dayofweek
                        break
                    if etype == "TARGET" and bar["high"] >= tgt:
                        exit_day = bar_date.dayofweek
                        break
                    exit_day = bar_date.dayofweek  # TIME — last bar
                if exit_day == 5:
                    exit_sat += 1
                elif exit_day == 6:
                    exit_sun += 1

        # Anomaly check on present bars
        for bar_date, bar in next_bars[next_bars.index.dayofweek.isin([5, 6])].iterrows():
            issues = []
            if bar["volume"] == 0:
                issues.append("zero volume")
            if bar["high"] == bar["low"]:
                issues.append("flat bar (high==low)")
            if bar["close"] == 0 or bar["open"] == 0:
                issues.append("zero price")
            # Carried-forward check: identical to prior bar
            pos = ticker_df.index.get_loc(bar_date)
            if pos > 0:
                prev = ticker_df.iloc[pos - 1]
                if (bar["close"] == prev["close"] and bar["high"] == prev["high"]
                        and bar["low"] == prev["low"]):
                    issues.append("identical to prior bar (possible carry-forward)")
            if issues:
                anomalous_bars.append(
                    f"    {bar_date.date()} ({ticker})  {', '.join(issues)}"
                )

    needs_attention = (sat_only_missing + sun_only_missing + both_missing > 5
                       or len(anomalous_bars) > 0)
    flag = "NEEDS ATTENTION" if needs_attention else "CLEAN"

    print(f"\n{'─'*60}")
    print(f"  AUDIT 1 — Weekend Bar Completeness: {ticker}")
    print(f"{'─'*60}")
    print(f"  Total LONG signals         : {total}")
    print(f"  Both Sat+Sun present       : {both_present}  ({both_present/total*100:.1f}%)")
    print(f"  Missing Saturday only      : {sat_only_missing}")
    print(f"  Missing Sunday only        : {sun_only_missing}")
    print(f"  Missing both               : {both_missing}")
    print(f"  Exit: NO_DATA              : {exit_nodata}")
    print(f"  Exit on Saturday           : {exit_sat}")
    print(f"  Exit on Sunday             : {exit_sun}")

    print(f"\n  Year breakdown:")
    print(f"  {'Year':>6} {'Signals':>8} {'Both':>6} {'SatMiss':>8} "
          f"{'SunMiss':>8} {'AllMiss':>8} {'NoData':>7}")
    print("  " + "-" * 56)
    for year in sorted(by_year):
        d = by_year[year]
        print(f"  {year:>6} {d['signals']:>8} {d['both']:>6} {d['sat_missing']:>8} "
              f"{d['sun_missing']:>8} {d['both_missing']:>8} {d['no_data']:>7}")

    if anomalous_bars:
        print(f"\n  ⚠  Anomalous bars detected ({len(anomalous_bars)}):")
        for line in anomalous_bars:
            print(line)
    else:
        print(f"\n  No anomalous bars detected.")

    print(f"\n  Result: {flag}")
    return flag


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT 2 — yfinance vs Coinbase price divergence
# ═══════════════════════════════════════════════════════════════════════════════

def audit_price_divergence(
    yf_df: pd.DataFrame,
    cb_df: pd.DataFrame,
    ticker: str,
) -> str:
    """Align on date, compute differences, return flag."""
    # Restrict yfinance to last 60 days to match Coinbase window
    cutoff = cb_df.index.min()
    yf_w   = yf_df[yf_df.index >= cutoff].copy()

    common = yf_w.index.intersection(cb_df.index)
    if len(common) == 0:
        print(f"\n  {ticker}: no overlapping dates between yfinance and Coinbase")
        return "NEEDS ATTENTION"

    yf_c = yf_w.loc[common]
    cb_c = cb_df.loc[common]

    fields = ["close", "high", "low", "atr14"]
    diffs  = {}
    for f in fields:
        if f not in yf_c.columns or f not in cb_c.columns:
            continue
        abs_diff = (yf_c[f] - cb_c[f]).abs()
        pct_diff = abs_diff / cb_c[f].abs().replace(0, np.nan) * 100
        diffs[f] = {"abs": abs_diff, "pct": pct_diff}

    needs_attention = False

    print(f"\n{'─'*60}")
    print(f"  AUDIT 2 — yfinance vs Coinbase Divergence: {ticker}")
    print(f"  Overlap window: {common[0].date()} → {common[-1].date()}  ({len(common)} days)")
    print(f"{'─'*60}")
    print(f"  {'Field':<8} {'Mean abs':>10} {'Max abs':>10} {'Std abs':>10} "
          f"{'Mean %':>8} {'Max %':>8}")
    print("  " + "-" * 54)
    for f, d in diffs.items():
        print(f"  {f:<8} {d['abs'].mean():>10.4f} {d['abs'].max():>10.4f} "
              f"{d['abs'].std():>10.4f} {d['pct'].mean():>7.3f}% {d['pct'].max():>7.3f}%")

    # ATR14 >2% flag
    if "atr14" in diffs:
        atr_exceedances = diffs["atr14"]["pct"][diffs["atr14"]["pct"] > ATR_DIVERGENCE_THRESHOLD * 100]
        if not atr_exceedances.empty:
            needs_attention = True
            print(f"\n  ⚠  ATR14 diverges >2% on {len(atr_exceedances)} day(s):")
            for dt, pct in atr_exceedances.items():
                print(f"     {dt.date()}  {pct:.2f}%")

    # Most recent Friday side-by-side
    fridays_in_window = common[common.dayofweek == 4]
    if not fridays_in_window.empty:
        last_fri = fridays_in_window[-1]
        yf_fri   = yf_c.loc[last_fri]
        cb_fri   = cb_c.loc[last_fri]
        print(f"\n  Most recent Friday: {last_fri.date()}")
        print(f"  {'Metric':<16} {'yfinance':>14} {'Coinbase':>14} {'Diff':>10} {'Diff%':>8}")
        print("  " + "-" * 66)
        for f in ["close", "atr14", "ma20"]:
            if f not in yf_fri.index or f not in cb_fri.index:
                continue
            yv = yf_fri[f]; cv = cb_fri[f]
            diff = yv - cv
            pct  = abs(diff) / abs(cv) * 100 if cv != 0 else float("nan")
            flag_str = "  ⚠" if (f == "atr14" and pct > 2) else ""
            print(f"  {f:<16} {yv:>14,.4f} {cv:>14,.4f} {diff:>10,.4f} {pct:>7.3f}%{flag_str}")

        # Stop distance and units (using backtest params)
        from backtest_engine import ATR_MULT_STOP, RISK_PCT_PER_TRADE, SLIPPAGE_PCT
        ACCOUNT_EQUITY = 500.0
        for label, src in [("yfinance", yf_fri), ("Coinbase", cb_fri)]:
            entry = float(src["close"]) * (1 + SLIPPAGE_PCT)
            atr   = float(src["atr14"])
            stop  = entry - ATR_MULT_STOP * atr
            risk  = entry - stop
            units = (ACCOUNT_EQUITY * RISK_PCT_PER_TRADE) / risk
            print(f"\n  {label} signal metrics:")
            print(f"    Entry={entry:,.2f}  Stop={stop:,.2f}  "
                  f"Risk/unit={risk:,.2f}  Units={units:.6f}")

        # ATR divergence on this specific Friday
        if "atr14" in diffs:
            fri_atr_pct = diffs["atr14"]["pct"].get(last_fri, float("nan"))
            if not pd.isna(fri_atr_pct) and fri_atr_pct > 2:
                needs_attention = True
                print(f"\n  ⚠  ATR14 divergence on last Friday: {fri_atr_pct:.2f}% — CONCERN")
            else:
                print(f"\n  ATR14 divergence on last Friday: "
                      f"{fri_atr_pct:.2f}% — within tolerance")

    flag = "NEEDS ATTENTION" if needs_attention else "CLEAN"
    print(f"\n  Result: {flag}")
    return flag


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT 3 — Prior Friday lookup gaps
# ═══════════════════════════════════════════════════════════════════════════════

def audit_prior_friday_gaps(ticker_df: pd.DataFrame, ticker: str) -> str:
    """Count and report Fridays where no prior Friday existed (default-allow fired)."""
    fridays     = get_fridays(ticker_df)
    all_fri_idx = ticker_df.index[ticker_df.index.dayofweek == 4]
    default_allow_dates = []

    for fri_date, _ in fridays.iterrows():
        prior = all_fri_idx[all_fri_idx < fri_date]
        if len(prior) == 0:
            default_allow_dates.append(fri_date)

    # Cross with LONG signals only
    trade_log, _, _ = run_backtest(ticker_df, ticker)
    longs = set(pd.to_datetime(trade_log[trade_log["direction"] == "LONG"]["date"]))
    default_long_dates = [d for d in default_allow_dates if d in longs]

    n_total  = len(fridays)
    n_gap    = len(default_allow_dates)
    n_traded = len(default_long_dates)

    needs_attention = n_gap > 3

    print(f"\n{'─'*60}")
    print(f"  AUDIT 3 — Prior Friday Lookup Gaps: {ticker}")
    print(f"{'─'*60}")
    print(f"  Total Fridays in data        : {n_total}")
    print(f"  Fridays with no prior Friday : {n_gap}  "
          f"({'CONCERN' if n_gap > 3 else 'acceptable'})")
    print(f"  Of those, generated LONG     : {n_traded}  "
          f"(Filters 2+3 defaulted to ALLOW on these)")

    if default_allow_dates:
        print(f"\n  Dates where default-allow fired:")
        for d in sorted(default_allow_dates):
            traded = "→ LONG (filter skipped)" if d in longs else "→ NO_TRADE"
            print(f"    {d.date()}  {traded}")
    else:
        print(f"\n  No default-allow dates found.")

    flag = "NEEDS ATTENTION" if needs_attention else "CLEAN"
    print(f"\n  Result: {flag}")
    return flag


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Data Pipeline Audit — Three Checks")
    print("=" * 60)

    print("\nLoading yfinance history...")
    data = load_crypto_data(TICKERS)

    print("\nLoading Coinbase live data (last 60 days)...")
    try:
        cb_data = load_coinbase_live(TICKERS, days=60)
        coinbase_ok = True
    except Exception as e:
        print(f"  Coinbase load failed: {e}")
        coinbase_ok = False

    flags: dict[str, str] = {}

    # ── Audit 1 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AUDIT 1 — WEEKEND BAR COMPLETENESS")
    print(f"{'='*60}")
    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)
        flag = audit_weekend_bars(ticker_df, ticker)
        flags[f"A1_{ticker}"] = flag

    # ── Audit 2 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AUDIT 2 — yFINANCE vs COINBASE PRICE DIVERGENCE")
    print(f"{'='*60}")
    if coinbase_ok:
        for ticker in TICKERS:
            yf_df = get_ticker_df(data, ticker)
            cb_df = cb_data[ticker]
            flag = audit_price_divergence(yf_df, cb_df, ticker)
            flags[f"A2_{ticker}"] = flag
    else:
        print("  Skipped — Coinbase data unavailable.")
        for ticker in TICKERS:
            flags[f"A2_{ticker}"] = "NEEDS ATTENTION"

    # ── Audit 3 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AUDIT 3 — PRIOR FRIDAY LOOKUP GAPS")
    print(f"{'='*60}")
    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)
        flag = audit_prior_friday_gaps(ticker_df, ticker)
        flags[f"A3_{ticker}"] = flag

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    labels = {
        "A1_BTC-USD": "Audit 1 — Weekend bars (BTC)",
        "A1_ETH-USD": "Audit 1 — Weekend bars (ETH)",
        "A2_BTC-USD": "Audit 2 — Price divergence (BTC)",
        "A2_ETH-USD": "Audit 2 — Price divergence (ETH)",
        "A3_BTC-USD": "Audit 3 — Prior Fri gaps (BTC)",
        "A3_ETH-USD": "Audit 3 — Prior Fri gaps (ETH)",
    }
    any_concern = False
    for key, label in labels.items():
        f = flags.get(key, "UNKNOWN")
        marker = "⚠ " if f != "CLEAN" else "✓ "
        print(f"  {marker}{label:<34} {f}")
        if f != "CLEAN":
            any_concern = True

    print(f"\n  Overall: {'NEEDS ATTENTION — see flagged items above' if any_concern else 'CLEAN — pipeline looks healthy'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
