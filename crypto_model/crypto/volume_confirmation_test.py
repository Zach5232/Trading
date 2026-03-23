"""
crypto/volume_confirmation_test.py
===================================
Tests the volume confirmation filter against the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — all variants fail combined test, three-way
#                split pattern continues
# Floor only     BTC +0.067R (83 trades, removes 59 low-volume trades avg R
# (vol > 7-day   +0.045 vs kept +0.206) — would promote in isolation. ETH
# avg):          -0.017R — removes 32 low-volume trades avg R +0.235, better
#                than kept +0.183 — floor works backwards on ETH.
# Ceiling only   BTC -0.023R, ETH -0.018R — hurts both. BTC's 8 highest-
# (vol < 2×avg): volume Fridays avg +0.521R (best trades not exhaustion). ETH
#                ceiling removes the +1.901R winner.
# Combined:      BTC +0.033R, ETH -0.044R — split.
# Overlap with   Minimal (6 BTC, 1 ETH) — these filters find different trades.
# liquidity trap:
# Structural     BTC needs volume participation to follow through on momentum —
# interpretation: low volume Friday moves are noise for BTC. ETH is the
#                opposite — low volume Friday moves are ETH's better trades,
#                possibly because ETH moves on thinner participation and
#                whale-driven momentum doesn't require broad volume
#                confirmation.
# Emerging       Three consecutive BTC/ETH splits in Tier 2: market structure
# pattern:       (ETH yes/BTC no), dynamic R targets (ETH yes/BTC no), volume
#                floor (BTC yes/ETH no). The two instruments have meaningfully
#                different signal characteristics. A unified filter set may be
#                suboptimal — instrument-specific parameters likely needed.
# Future tests   BTC-only volume floor (+0.067R, 83 trades) and ETH-only 2.5R
# after Phase 1: target (+0.039R, 96 trades). Both meet promotion criteria in
#                isolation. Consider testing together as a combined instrument-
#                specific parameter upgrade. Requires per-ticker config in
#                main.py.
# Status:        Do not add to main.py.
# ─────────────────────────────────────────────────────────────────────────────

Filter hypothesis:
  Real momentum participation requires above-average volume (floor), but a
  volume spike suggests exhaustion rather than continuation (ceiling).

  Floor (vol7_floor): volume > 7-day rolling average volume
  Ceiling (vol7_ceil): volume < 2 × 7-day rolling average volume

Four variants tested:
  Baseline     : three-filter system (no volume filter)
  Combined     : floor AND ceiling (both conditions must hold)
  Floor only   : volume > 7-day avg only
  Ceiling only : volume < 2× 7-day avg only (closest to liquidity trap)

Overlap note:
  The liquidity trap test used vol > 2× 20-day avg as part of an AND
  condition with wide candles. This test uses a shorter 7-day window and
  adds a floor. Overlap between the two is reported explicitly.

Promotion criteria (all must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on both BTC and ETH independently
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


# ── Removal rate and overlap analysis ─────────────────────────────────────────

def _removal_preview(base_log: pd.DataFrame, ticker: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"]).copy()
    n = len(longs)
    if n == 0:
        return

    floor_fail = ~longs["vol7_floor_ok"]          # vol below 7-day avg
    ceil_fail  = ~longs["vol7_ceil_ok"]            # vol above 2× 7-day avg
    # Note: floor_fail and ceil_fail are mutually exclusive by definition
    # (a trade can't have vol < avg AND vol > 2× avg simultaneously)
    trap_flag  = longs["is_trap"]                  # liquidity trap flag

    n_floor = int(floor_fail.sum())
    n_ceil  = int(ceil_fail.sum())
    n_both  = int((floor_fail & ceil_fail).sum())  # always 0 by construction
    n_combo = int((floor_fail | ceil_fail).sum())  # removed by combined filter

    # Overlap with liquidity trap
    trap_and_vol = int((trap_flag & (floor_fail | ceil_fail)).sum())
    trap_only    = int((trap_flag & ~(floor_fail | ceil_fail)).sum())
    vol_only     = int((~trap_flag & (floor_fail | ceil_fail)).sum())

    print(f"\n  {ticker} — volume filter removal preview ({n} baseline LONG signals):")
    print(f"    Combined filter removes   : {n_combo:>3}  ({n_combo/n*100:.1f}%)")
    print(f"      → fails floor only      : {n_floor:>3}  ({n_floor/n*100:.1f}%)  vol ≤ 7-day avg")
    print(f"      → fails ceiling only    : {n_ceil:>3}  ({n_ceil/n*100:.1f}%)  vol ≥ 2× 7-day avg")
    print(f"      → fails both (impossible): {n_both:>3}  (confirms mutual exclusion)")
    print(f"    Floor-only filter removes : {n_floor:>3}  ({n_floor/n*100:.1f}%)")
    print(f"    Ceiling-only removes      : {n_ceil:>3}  ({n_ceil/n*100:.1f}%)")

    print(f"\n    Overlap with liquidity trap filter:")
    print(f"      Both vol filter AND trap : {trap_and_vol:>3}  (redundant trades)")
    print(f"      Trap filter only         : {trap_only:>3}  (unique to trap filter)")
    print(f"      Vol filter only          : {vol_only:>3}  (unique to vol filter)")


# ── Filtered-vs-kept quality breakdown ───────────────────────────────────────

def _quality_breakdown(base_log: pd.DataFrame, ticker: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"]).copy()

    def _fmt(sub: pd.DataFrame, label: str) -> str:
        if sub.empty:
            return f"  {label}: n=0"
        wr  = (sub["R_multiple"] > 0).mean() * 100
        avr = sub["R_multiple"].mean()
        return f"  {label}: n={len(sub):>3}  WR={wr:>5.1f}%  AvgR={avr:>+7.3f}"

    floor_fail = ~longs["vol7_floor_ok"]
    ceil_fail  = ~longs["vol7_ceil_ok"]

    kept_by_combined  = longs[~floor_fail & ~ceil_fail]
    floor_removed     = longs[floor_fail]
    ceil_removed      = longs[ceil_fail]

    print(f"\n  {ticker} — avg R: filtered vs kept (from baseline):")
    print(f"    {_fmt(kept_by_combined,  'Kept by combined filter')}")
    print(f"    {_fmt(floor_removed,     'Removed by floor (↓ vol)')}")
    print(f"    {_fmt(ceil_removed,      'Removed by ceiling (↑ vol)')}")


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_table(
    ticker: str,
    versions: dict[str, tuple[pd.DataFrame, dict]],
    base_key: str,
) -> dict[str, str]:
    base_m = versions[base_key][1]

    print(f"\n{'='*80}")
    print(f"  VOLUME CONFIRMATION FILTER — {ticker}")
    print(f"{'='*80}")

    hdr = (f"  {'Version':<22} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 74)

    verdicts: dict[str, str] = {}
    for label, (tl, m) in versions.items():
        if not m:
            print(f"  {label:<22}  — no trades —")
            continue
        delta_str = "    base" if label == base_key else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n     = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<22} {m['n_trades']:>7} {m['trades_per_year']:>5} "
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
        print(f"  {label:<22}  TARGET:{tgt:>3}({tgt/n*100:.0f}%)  STOP:{stp:>3}({stp/n*100:.0f}%)"
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
        print(f"\n  {label:<22}: {verdict}{suffix}")
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
    print("Volume Confirmation Filter Test  —  Var1+Var2+Var4 baseline")
    print("  Floor:   volume > 7-day rolling average (confirms participation)")
    print("  Ceiling: volume < 2× 7-day rolling average (filters exhaustion)")
    print("=" * 80)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_verdicts: dict[str, dict[str, str]] = {}
    all_results:  dict[str, dict]           = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log,  base_m,  base_rm  = run_backtest(ticker_df, ticker)
        comb_log,  comb_m,  comb_rm  = run_backtest(ticker_df, ticker,
                                                      volume_floor_filter=True,
                                                      volume_ceiling_filter=True)
        floor_log, floor_m, floor_rm = run_backtest(ticker_df, ticker,
                                                      volume_floor_filter=True)
        ceil_log,  ceil_m,  ceil_rm  = run_backtest(ticker_df, ticker,
                                                      volume_ceiling_filter=True)

        # ── Removal rate preview ───────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  REMOVAL RATE PREVIEW — {ticker}")
        print(f"{'─'*60}")
        _removal_preview(base_log, ticker)

        # ── Quality breakdown ──────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  FILTERED vs KEPT QUALITY — {ticker}")
        print(f"{'─'*60}")
        _quality_breakdown(base_log, ticker)

        versions = {
            "Baseline":           (base_log,  base_m),
            "Combined (F+C)":     (comb_log,  comb_m),
            "Floor only":         (floor_log, floor_m),
            "Ceiling only":       (ceil_log,  ceil_m),
        }
        all_results[ticker] = {
            "base_rm":  base_rm,
            "comb_rm":  comb_rm,
            "floor_rm": floor_rm,
            "ceil_rm":  ceil_rm,
        }

        verdicts = _print_table(ticker, versions, "Baseline")
        all_verdicts[ticker] = verdicts

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*80}")
    promoted = []
    rm_key_map = {
        "Combined (F+C)": "comb_rm",
        "Floor only":     "floor_rm",
        "Ceiling only":   "ceil_rm",
    }
    for label in rm_key_map:
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
