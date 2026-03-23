"""
crypto/volatility_compression_test.py
======================================
Tests the volatility compression setup classifier against the
Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — sample too small, sizing metric flawed
# Key finding:   Compression setups show spectacular outperformance. BTC
#                compression avg R +0.698R vs standard +0.078R (delta
#                +0.620R). ETH compression avg R +0.823R vs standard +0.151R
#                (delta +0.672R). Both clear the >0.05R threshold massively.
#                BTC compression had zero stop-out exits across all 14 trades
#                vs 19 stops in 128 standard trades.
# Problem 1:     n=14 BTC and n=7 ETH — approximately 1 compression setup per
#                year. Cannot distinguish genuine mechanism from selection bias
#                concentrated in 2019/2021 bull markets. Minimum meaningful
#                sample is 50+ setups (~3-4 years of accumulation).
# Problem 2:     Sizing rule metric is structurally broken for avg_R — sizing
#                up 1.5× scales numerator and denominator identically, R never
#                changes. Future sizing rule tests must use ROI or equity
#                growth as the metric, not avg_R.
# Passive        vol_compression flag already stored on every trade row in
# tracking:      live trade log — costs nothing to track going forward.
# Revisit        After 50+ compression setups accumulate (~3-4 years), use
# criteria:      ROI as metric not avg_R, check whether zero-stop pattern
#                holds out of sample.
# Status:        Do not add to main.py. Track passively and revisit post-
#                Phase-1 when sample is meaningful.
# ─────────────────────────────────────────────────────────────────────────────

This is a classifier test, not a filter test. All trades execute as normal.
Compression setups are flagged and their performance compared to standard
setups. If compression outperforms sufficiently, a sizing rule (1.5× risk
on compression weeks) is tested as the promotion vehicle.

Compression setup definition:
  ATR14 on the 3 bars immediately before Friday must be strictly declining:
    ATR14(Thu) < ATR14(Wed) < ATR14(Tue)
  All three steps must descend — a flat or rising day breaks the pattern.

Sizing rule tested (if compression outperforms):
  Compression setups : 1.5× normal risk per trade
  Standard setups    : 1.0× normal risk (unchanged)

Promotion criteria for sizing rule (higher bar than filter tests):
  1. Compression setups must outperform standard by > 0.05R avg R
  2. Blended portfolio avg R must improve > 0.02R vs flat baseline
  3. Max drawdown must not increase by more than 5 percentage points
  4. Must hold on both BTC and ETH independently
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import run_backtest, STARTING_CAPITAL, REGIMES

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Subset metrics ────────────────────────────────────────────────────────────

def _subset_metrics(trades: pd.DataFrame, label: str) -> dict:
    """Compute stats for a subset of LONG trades."""
    longs = trades[(trades["direction"] == "LONG")].dropna(subset=["R_multiple"])
    if longs.empty:
        return {}
    n     = len(longs)
    wins  = longs[longs["R_multiple"] > 0]
    gross_profit = wins["R_multiple"].sum()
    gross_loss   = abs(longs[longs["R_multiple"] <= 0]["R_multiple"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    ec = longs["exit_type"].value_counts().to_dict()
    return {
        "label":   label,
        "n":       n,
        "wr":      round(wins.__len__() / n * 100, 1),
        "avg_R":   round(longs["R_multiple"].mean(), 3),
        "pf":      round(pf, 3),
        "exits":   ec,
    }


def _avg_r_by_exit(trades: pd.DataFrame) -> dict[str, float]:
    longs = trades[(trades["direction"] == "LONG")].dropna(subset=["R_multiple"])
    result = {}
    for etype in ["TARGET", "STOP", "TIME", "TIME_MON", "STOP_BE", "TRAIL"]:
        sub = longs[longs["exit_type"] == etype]
        if not sub.empty:
            result[etype] = round(sub["R_multiple"].mean(), 3)
    return result


# ── Classification breakdown ─────────────────────────────────────────────────

def _classification_breakdown(base_log: pd.DataFrame, ticker: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    n     = len(longs)
    comp  = longs[longs["vol_compression"] == True]
    std   = longs[longs["vol_compression"] == False]

    print(f"\n  {ticker} — compression setup classification ({n} LONG signals):")
    print(f"    Compression setups : {len(comp):>3}  ({len(comp)/n*100:.1f}%)")
    print(f"    Standard setups    : {len(std):>3}  ({len(std)/n*100:.1f}%)")


def _regime_distribution(base_log: pd.DataFrame, ticker: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    comp  = longs[longs["vol_compression"] == True]

    print(f"\n  {ticker} — compression setup regime distribution:")
    print(f"  {'Regime':<22} {'Total_T':>8} {'Comp_T':>7} {'Comp%':>7} {'Comp_AvgR':>10} {'Std_AvgR':>10}")
    print("  " + "-" * 68)
    for name, (r_start, r_end) in REGIMES.items():
        sub_all  = longs[(longs["date"] >= r_start)  & (longs["date"] <= r_end)]
        sub_comp = comp[(comp["date"]   >= r_start)  & (comp["date"]  <= r_end)]
        sub_std  = sub_all[sub_all["vol_compression"] == False]
        if sub_all.empty:
            continue
        pct      = len(sub_comp) / len(sub_all) * 100
        comp_r   = sub_comp["R_multiple"].mean() if not sub_comp.empty else float("nan")
        std_r    = sub_std["R_multiple"].mean()  if not sub_std.empty  else float("nan")
        comp_r_s = f"{comp_r:>+10.3f}" if not pd.isna(comp_r) else "       N/A"
        std_r_s  = f"{std_r:>+10.3f}"  if not pd.isna(std_r)  else "       N/A"
        print(f"  {name:<22} {len(sub_all):>8} {len(sub_comp):>7} {pct:>6.1f}% {comp_r_s} {std_r_s}")


# ── Performance comparison ────────────────────────────────────────────────────

def _performance_comparison(base_log: pd.DataFrame, ticker: str) -> tuple[dict, dict]:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    comp  = longs[longs["vol_compression"] == True]
    std   = longs[longs["vol_compression"] == False]

    cm = _subset_metrics(comp, "Compression")
    sm = _subset_metrics(std,  "Standard")

    print(f"\n  {ticker} — compression vs standard performance:")
    print(f"  {'Setup':<14} {'n':>5} {'WR%':>6} {'AvgR':>8} {'PF':>6}")
    print("  " + "-" * 44)
    for m in [cm, sm]:
        if not m:
            continue
        print(f"  {m['label']:<14} {m['n']:>5} {m['wr']:>5}% {m['avg_R']:>+8.3f} {m['pf']:>6.3f}")

    if cm and sm:
        delta = cm["avg_R"] - sm["avg_R"]
        print(f"  {'Delta':>14}        {'':>5} {delta:>+8.3f}")

    # Avg R by exit type
    print(f"\n  {ticker} — avg R by exit type:")
    comp_exits = _avg_r_by_exit(comp)
    std_exits  = _avg_r_by_exit(std)
    exit_types = sorted(set(list(comp_exits.keys()) + list(std_exits.keys())))
    print(f"  {'Exit':<10} {'Comp_AvgR':>10} {'Std_AvgR':>10}")
    print("  " + "-" * 32)
    for et in exit_types:
        c_r = f"{comp_exits[et]:>+10.3f}" if et in comp_exits else "       N/A"
        s_r = f"{std_exits[et]:>+10.3f}"  if et in std_exits  else "       N/A"
        # also print counts
        c_n = len(comp[comp["exit_type"] == et]) if not comp.empty else 0
        s_n = len(std[std["exit_type"] == et])   if not std.empty  else 0
        print(f"  {et:<10} {c_r}  (n={c_n:>2})  {s_r}  (n={s_n:>2})")

    return cm, sm


# ── Sizing rule comparison table ──────────────────────────────────────────────

def _sizing_table(
    ticker: str,
    base_m: dict,
    sized_m: dict,
) -> dict:
    """Compare flat baseline vs 1.5× compression sizing. Returns verdict dict."""
    print(f"\n{'='*80}")
    print(f"  SIZING RULE — {ticker}  (compression 1.5× vs flat 1.0×)")
    print(f"{'='*80}")

    hdr = (f"  {'Version':<22} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 74)

    versions = {"Flat (1.0×)": base_m, "Compression 1.5×": sized_m}
    for label, m in versions.items():
        if not m:
            print(f"  {label:<22}  — no trades —")
            continue
        base_ref = base_m["avg_R"]
        delta_str = "    base" if label == "Flat (1.0×)" else f"{m['avg_R']-base_ref:>+8.3f}"
        print(
            f"  {label:<22} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}"
        )

    # Verdict
    if not sized_m or not base_m:
        return {"verdict": "NO DATA"}

    delta_r  = sized_m["avg_R"] - base_m["avg_R"]
    dd_base  = abs(base_m["max_drawdown"])
    dd_sized = abs(sized_m["max_drawdown"])
    dd_delta = dd_sized - dd_base

    ok_r  = delta_r > 0.02
    ok_dd = dd_delta <= 5.0
    verdict = "PROMOTE" if (ok_r and ok_dd) else "DO NOT PROMOTE"
    reasons = []
    if not ok_r:
        reasons.append(f"delta={delta_r:+.3f}R (need >+0.02R)")
    if not ok_dd:
        reasons.append(f"drawdown increased {dd_delta:+.1f}pp (limit 5pp)")
    suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
    print(f"\n  Compression 1.5×    : {verdict}{suffix}")
    print(f"  Drawdown delta      : {dd_delta:>+.1f} percentage points")
    return {"verdict": verdict, "delta_r": delta_r, "dd_delta": dd_delta}


# ── Regime breakdown ──────────────────────────────────────────────────────────

def _regime_breakdown(ticker: str, base_rm: dict, sized_rm: dict) -> list[str]:
    print(f"\n  Regime breakdown — {ticker} (compression 1.5×):")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Sz_T':>6} {'Sz_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 67)
    regressions = []
    for name in REGIMES:
        bm  = base_rm.get(name, {})
        mm  = sized_rm.get(name, {})
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
    print("Volatility Compression Setup Test  —  Var1+Var2+Var4 baseline")
    print("  Compression: ATR14(Thu) < ATR14(Wed) < ATR14(Tue) before Friday")
    print("  Classifier only — no trades removed; sizing rule tested if outperforms")
    print("=" * 80)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_comp_deltas: dict[str, float] = {}
    all_sizing_verdicts: dict[str, dict] = {}
    all_results: dict[str, dict] = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log,  base_m,  base_rm  = run_backtest(ticker_df, ticker)
        sized_log, sized_m, sized_rm = run_backtest(ticker_df, ticker,
                                                     compression_size_mult=1.5)

        # ── Classification and regime distribution ────────────────────────
        print(f"\n{'─'*60}")
        print(f"  CLASSIFICATION — {ticker}")
        print(f"{'─'*60}")
        _classification_breakdown(base_log, ticker)
        _regime_distribution(base_log, ticker)

        # ── Performance comparison ────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  COMPRESSION vs STANDARD PERFORMANCE — {ticker}")
        print(f"{'─'*60}")
        cm, sm = _performance_comparison(base_log, ticker)

        if cm and sm:
            delta = cm["avg_R"] - sm["avg_R"]
            all_comp_deltas[ticker] = delta
            ok_diff = delta > 0.05
            print(f"\n  Compression outperformance: {delta:>+.3f}R"
                  f"  {'✓ meets >0.05R threshold' if ok_diff else '✗ below >0.05R threshold'}")

        # ── Sizing rule ───────────────────────────────────────────────────
        sizing_verdict = _sizing_table(ticker, base_m, sized_m)
        all_sizing_verdicts[ticker] = sizing_verdict
        all_results[ticker] = {
            "base_rm": base_rm, "sized_rm": sized_rm,
        }

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*80}")

    # Compression outperformance check (>0.05R on both)
    print(f"\n  Step 1 — Compression outperformance threshold (>0.05R both):")
    for ticker in TICKERS:
        d = all_comp_deltas.get(ticker, float("nan"))
        flag = "✓" if d > 0.05 else "✗"
        print(f"    {ticker}: {d:>+.3f}R  {flag}")

    both_outperform = all(all_comp_deltas.get(t, -999) > 0.05 for t in TICKERS)

    # Sizing rule verdict
    print(f"\n  Step 2 — Sizing rule promotion (both instruments):")
    btc_v = all_sizing_verdicts.get("BTC-USD", {}).get("verdict", "—")
    eth_v = all_sizing_verdicts.get("ETH-USD", {}).get("verdict", "—")
    combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
    print(f"    BTC={btc_v}  ETH={eth_v}  →  {combined}")

    # Regime breakdown if promoted
    if combined == "PROMOTE":
        print(f"\n{'='*80}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*80}")
        for ticker in TICKERS:
            regs = _regime_breakdown(
                ticker,
                all_results[ticker]["base_rm"],
                all_results[ticker]["sized_rm"],
            )
            if regs:
                print(f"    Regressions in: {', '.join(regs)}")
            else:
                print(f"    No regime regressions.")
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
