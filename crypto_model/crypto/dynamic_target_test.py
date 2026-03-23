"""
crypto/dynamic_target_test.py
==============================
Tests dynamic R targets against the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — split result on Component A; Components B
#                and C fail outright
# Component A    ETH +0.039R, 96 trades — would promote in isolation. BTC
# (2.5R on       -0.015R — fails. Zero trades on either instrument convert
# expanding):    from TIME to TARGET at 2.5R — the market doesn't run the
#                extra 0.5R within the weekend window frequently enough. BTC
#                loses 10 trades that hit 2.0R but not 2.5R (avg R drops
#                from +1.837 to +0.923 on those 10).
# Component B    Adds positive-expectancy trades (BTC +0.051R, ETH +0.168R
# (1.5R on       standalone) but weaker than expanding universe — dilutes
# contracting):  blended avg R below baseline. BTC -0.041R, ETH -0.014R net.
# Component C    Full dynamic: BTC -0.049R, ETH +0.007R — fails both.
# (dynamic):
# Emerging       This is the second Tier 2 test where ETH promotes in
# pattern:       isolation and BTC fails (market structure filter was the
#                first). ETH appears more amenable to parameter tuning than
#                BTC at current regime.
# Future test:   ETH-only 2.5R target after Phase 1 — meets promotion
#                criteria in isolation. Would require per-ticker target
#                configuration in main.py (separate R_TARGET_BTC and
#                R_TARGET_ETH constants). Low implementation complexity,
#                worth testing after 8 weeks.
# Status:        Do not add to main.py. Fixed 2.0R target remains for both
#                instruments.
# ─────────────────────────────────────────────────────────────────────────────

Three components tested:

  Component A — 2.5R target on existing LONG signals (ATR expanding):
    All current LONG trades pass Filter 2 (ATR expanding), so raising
    the target from 2.0R to 2.5R applies to all of them. Same trades,
    same entries, same stops — just a higher target. Tests whether
    the market carries past 2R when momentum is expanding.

  Component B — ATR-contracting signals with 1.5R target:
    Relax Filter 2; allow LONG when ATR is contracting, but use 1.5R
    target instead of 2.0R as compensation for weaker momentum. All
    other filters still required (MA20 and momentum confirm must pass).
    Tests whether adding these signals adds positive expectancy net of
    the existing baseline.

  Component C — Full dynamic system:
    2.5R when ATR expanding + 1.5R when ATR contracting (B ⊃ A).

Promotion criteria (all must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on both BTC and ETH independently
  For B and C: ATR-contracting trades must themselves have positive
  expectancy — blending in negative-expectancy trades is not valid.
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


# ── Component A conversion analysis ──────────────────────────────────────────

def _comp_a_conversion(
    ticker: str,
    base_log: pd.DataFrame,
    a_log: pd.DataFrame,
) -> None:
    """Show which baseline trades flip exit type when target moves from 2.0R to 2.5R."""
    TIME_EXITS  = {"TIME", "TIME_MON", "STOP_BE"}
    base_longs  = base_log[base_log["direction"] == "LONG"][["date", "exit_type", "R_multiple"]].copy()
    a_longs     = a_log[a_log["direction"] == "LONG"][["date", "exit_type", "R_multiple"]].copy()
    merged      = base_longs.merge(a_longs, on="date", suffixes=("_base", "_a"))

    # Trades that were TARGET at 2.0R but miss 2.5R (hurt by higher target)
    hurt = merged[
        (merged["exit_type_base"] == "TARGET") &
        (~merged["exit_type_a"].isin(["TARGET"]))
    ]
    # Trades that were TIME/TIME_MON/STOP_BE at 2.0R but now hit 2.5R
    benefited = merged[
        (merged["exit_type_base"].isin(TIME_EXITS)) &
        (merged["exit_type_a"] == "TARGET")
    ]

    print(f"\n  {ticker} — Component A conversion analysis (2.0R → 2.5R target):")
    print(f"    Trades hurt   (TARGET→other)  : {len(hurt):>3}"
          f"  avg R_base={hurt['R_multiple_base'].mean():>+7.3f}"
          f"  avg R_new={hurt['R_multiple_a'].mean():>+7.3f}")
    print(f"    Trades gained (TIME→TARGET)   : {len(benefited):>3}"
          f"  avg R_base={benefited['R_multiple_base'].mean():>+7.3f}"
          f"  avg R_new={benefited['R_multiple_a'].mean():>+7.3f}")

    # Avg R by exit type for baseline and CompA
    for label, log in [("Baseline (2.0R)", base_log), ("Comp A  (2.5R)", a_log)]:
        longs = log[log["direction"] == "LONG"].dropna(subset=["R_multiple"])
        tgt   = longs[longs["exit_type"] == "TARGET"]
        tim   = longs[longs["exit_type"].isin(TIME_EXITS)]
        stp   = longs[longs["exit_type"].isin({"STOP", "TRAIL"})]
        tgt_r = tgt["R_multiple"].mean() if not tgt.empty else float("nan")
        tim_r = tim["R_multiple"].mean() if not tim.empty else float("nan")
        stp_r = stp["R_multiple"].mean() if not stp.empty else float("nan")
        print(f"    {label}  TARGET n={len(tgt):>3} avg={tgt_r:>+7.3f}"
              f"  TIME/BE n={len(tim):>3} avg={tim_r:>+7.3f}"
              f"  STOP n={len(stp):>3} avg={stp_r:>+7.3f}")


# ── Component B quality check ─────────────────────────────────────────────────

def _comp_b_quality(ticker: str, b_log: pd.DataFrame, base_log: pd.DataFrame) -> None:
    """Report on the new ATR-contracting trades added by Component B."""
    b_longs   = b_log[b_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    contracting = b_longs[b_longs["atr_expanding"] == "no"]
    expanding   = b_longs[b_longs["atr_expanding"] == "yes"]

    n_base = len(base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"]))

    def _fmt(sub: pd.DataFrame, label: str) -> str:
        if sub.empty:
            return f"  {label}: n=0"
        wr  = (sub["R_multiple"] > 0).mean() * 100
        avr = sub["R_multiple"].mean()
        return (f"  {label}: n={len(sub):>3}  WR={wr:>5.1f}%  AvgR={avr:>+7.3f}"
                f"  {'POSITIVE EXP' if avr > 0 else 'NEGATIVE EXP'}")

    print(f"\n  {ticker} — Component B new ATR-contracting trades (1.5R target):")
    print(f"    Baseline LONG trades       : {n_base}")
    print(f"    Added (ATR contracting)    : {len(contracting)}")
    print(f"    {_fmt(contracting, 'Contracting trades')}")
    print(f"    {_fmt(expanding,   'Expanding trades  ')}")


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_table(
    ticker: str,
    versions: dict[str, tuple[pd.DataFrame, dict]],
    base_key: str,
) -> dict[str, str]:
    base_m = versions[base_key][1]

    print(f"\n{'='*80}")
    print(f"  DYNAMIC R TARGET — {ticker}")
    print(f"{'='*80}")

    hdr = (f"  {'Version':<20} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 74)

    verdicts: dict[str, str] = {}
    for label, (tl, m) in versions.items():
        if not m:
            print(f"  {label:<20}  — no trades —")
            continue
        delta_str = "    base" if label == base_key else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n     = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<20} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}{low_n}"
        )

    # Exit breakdown
    print(f"\n  Exit breakdown:")
    for label, (tl, m) in versions.items():
        if not m:
            continue
        n    = m["n_trades"]
        tgt  = m.get("exit_TARGET",   0)
        stp  = m.get("exit_STOP",     0)
        tim  = m.get("exit_TIME",     0)
        tmon = m.get("exit_TIME_MON", 0)
        be   = m.get("exit_STOP_BE",  0)
        print(f"  {label:<20}  TARGET:{tgt:>3}({tgt/n*100:.0f}%)  STOP:{stp:>3}({stp/n*100:.0f}%)"
              f"  TIME:{tim:>3}({tim/n*100:.0f}%)  TIME_MON:{tmon:>3}  STOP_BE:{be:>3}")

    # Verdicts
    for label, (tl, m) in versions.items():
        if label == base_key or not m:
            continue
        delta   = m["avg_R"] - base_m["avg_R"]
        ok_r    = delta > 0.02
        ok_n    = m["n_trades"] >= 30
        verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
        reasons = []
        if not ok_r:
            reasons.append(f"delta={delta:+.3f}R (need >+0.02R)")
        if not ok_n:
            reasons.append(f"only {m['n_trades']} trades (need ≥30)")
        suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
        print(f"\n  {label:<20}: {verdict}{suffix}")
        verdicts[label] = verdict

    return verdicts


# ── Regime breakdown ──────────────────────────────────────────────────────────

def _regime_breakdown(ticker: str, label: str, base_rm: dict, cmp_rm: dict) -> list[str]:
    print(f"\n  Regime breakdown — {ticker} {label}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Cmp_T':>6} {'Cmp_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 67)
    regressions = []
    for name in REGIMES:
        bm  = base_rm.get(name, {})
        mm  = cmp_rm.get(name, {})
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
    print("Dynamic R Target Test  —  Var1+Var2+Var4 baseline")
    print("  Comp A: 2.5R target on expanding ATR trades (current LONG universe)")
    print("  Comp B: Relax Filter 2 + 1.5R target on contracting ATR trades")
    print("  Comp C: Full dynamic — 2.5R expanding + 1.5R contracting")
    print("=" * 80)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_verdicts: dict[str, dict[str, str]] = {}
    all_results:  dict[str, dict]           = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log, base_m, base_rm = run_backtest(ticker_df, ticker)
        a_log,    a_m,    a_rm    = run_backtest(ticker_df, ticker,
                                                  expanding_r_target=2.5)
        b_log,    b_m,    b_rm    = run_backtest(ticker_df, ticker,
                                                  contracting_r_target=1.5,
                                                  relax_filter2=True)
        c_log,    c_m,    c_rm    = run_backtest(ticker_df, ticker,
                                                  expanding_r_target=2.5,
                                                  contracting_r_target=1.5,
                                                  relax_filter2=True)

        # ── Component A conversion analysis ───────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  COMPONENT A CONVERSION ANALYSIS — {ticker}")
        print(f"{'─'*60}")
        _comp_a_conversion(ticker, base_log, a_log)

        # ── Component B quality check ──────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  COMPONENT B NEW TRADE QUALITY — {ticker}")
        print(f"{'─'*60}")
        _comp_b_quality(ticker, b_log, base_log)

        versions = {
            "Baseline (2.0R)":     (base_log, base_m),
            "Comp A (2.5R exp)":   (a_log,    a_m),
            "Comp B (1.5R ctr)":   (b_log,    b_m),
            "Comp C (dynamic)":    (c_log,    c_m),
        }
        all_results[ticker] = {
            "base_rm": base_rm,
            "a_rm":    a_rm,
            "b_rm":    b_rm,
            "c_rm":    c_rm,
        }

        verdicts = _print_table(ticker, versions, "Baseline (2.0R)")
        all_verdicts[ticker] = verdicts

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*80}")
    promoted = []
    for label in ["Comp A (2.5R exp)", "Comp B (1.5R ctr)", "Comp C (dynamic)"]:
        btc_v    = all_verdicts.get("BTC-USD", {}).get(label, "—")
        eth_v    = all_verdicts.get("ETH-USD", {}).get(label, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {label:<22}: BTC={btc_v}  ETH={eth_v}  →  {combined}")
        if combined == "PROMOTE":
            promoted.append(label)

    # ── Regime breakdown for promoted ─────────────────────────────────────
    if promoted:
        print(f"\n{'='*80}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*80}")
        rm_key_map = {
            "Comp A (2.5R exp)": "a_rm",
            "Comp B (1.5R ctr)": "b_rm",
            "Comp C (dynamic)":  "c_rm",
        }
        for label in promoted:
            rm_key = rm_key_map[label]
            for ticker in TICKERS:
                base_rm = all_results[ticker]["base_rm"]
                cmp_rm  = all_results[ticker][rm_key]
                regs    = _regime_breakdown(ticker, label, base_rm, cmp_rm)
                if regs:
                    print(f"    Regressions in: {', '.join(regs)}")
                else:
                    print(f"    No regime regressions.")
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
