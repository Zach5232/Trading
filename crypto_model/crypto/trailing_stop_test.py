"""
crypto/trailing_stop_test.py
=============================
Tests dynamic trailing stop management against the fixed-stop Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — fails on ETH, BTC misses bar by 0.001R
# Key finding:   0.5×ATR trail width is too tight for weekend crypto bars.
#                The trail fires on the same bar that triggers the 1.5R upgrade,
#                converting TARGET winners into early TRAIL exits.
#                BTC net +0.019R (just below 0.02R bar).
#                ETH net -0.098R — trail destroys 11 TARGET exits worth ~+1.90R each.
# Exit           BTC: converts 19 STOPs → TRAIL (good) but also kills 19 of 24
# conversion:    TARGETs (bad). ETH: TARGET exits collapse from 13 to 2.
# Revisit:       Test wider trail widths of 1.0×ATR and 1.5×ATR separately —
#                different hypothesis, needs its own test. ETH winners run harder
#                and straighter so may need asymmetric trail logic vs BTC.
# Status:        Do not add to main.py. Fixed stop remains in place.
# ─────────────────────────────────────────────────────────────────────────────

Trailing stop logic (applied intrabar during weekend + Monday hold bars):
  - dynamic_stop starts at original stop (entry - 1.25 × ATR)
  - When bar high reaches 1R profit: move dynamic_stop to breakeven (entry)
  - When bar high reaches 1.5R profit: trail at (bar_high - 0.5 × ATR),
    updating each subsequent bar as price moves higher
  - Target (2R) is unchanged
  - A stop hit under trailing logic exits as "TRAIL" not "STOP"

Hypothesis: reduces losing trade size (TRAIL at BE instead of -1R STOP)
without cutting significantly into winners, improving net expectancy.

Promotion criteria (all must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on both BTC and ETH independently
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import run_backtest, STARTING_CAPITAL, REGIMES

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Per-exit-type avg R ───────────────────────────────────────────────────────

def _avg_r_by_exit(trade_log: pd.DataFrame) -> dict[str, str]:
    completed = trade_log[trade_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    result = {}
    for exit_type in ["TARGET", "STOP", "TRAIL", "TIME", "TIME_MON", "STOP_BE"]:
        subset = completed[completed["exit_type"] == exit_type]
        if subset.empty:
            result[exit_type] = "  —"
        else:
            result[exit_type] = f"{subset['R_multiple'].mean():>+.3f} (n={len(subset)})"
    return result


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_comparison(
    ticker: str,
    base_log: pd.DataFrame, base_m: dict,
    trail_log: pd.DataFrame, trail_m: dict,
) -> str:
    """Print comparison table and return PROMOTE / DO NOT PROMOTE."""
    print(f"\n{'='*72}")
    print(f"  TRAILING STOP TEST — {ticker}")
    print(f"{'='*72}")

    hdr = (f"  {'Version':<12} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 68)

    for label, m in [("Baseline", base_m), ("Trailing", trail_m)]:
        if not m:
            print(f"  {label:<12}  — no trades —")
            continue
        if label == "Baseline":
            delta_str = "    base"
        else:
            delta = m["avg_R"] - base_m["avg_R"]
            delta_str = f"{delta:>+8.3f}"
        print(
            f"  {label:<12} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}"
        )

    # Exit breakdown
    print(f"\n  Exit breakdown:")
    base_exits  = {k: base_m.get(f"exit_{k}", 0)  for k in ["TARGET","STOP","TRAIL","TIME","TIME_MON","STOP_BE"]}
    trail_exits = {k: trail_m.get(f"exit_{k}", 0) for k in ["TARGET","STOP","TRAIL","TIME","TIME_MON","STOP_BE"]}
    base_n  = base_m.get("n_trades", 1)
    trail_n = trail_m.get("n_trades", 1)
    print(f"  {'Exit type':<12} {'Base n':>8} {'Base %':>8} {'Trail n':>9} {'Trail %':>9}")
    print("  " + "-" * 52)
    for k in ["TARGET", "STOP", "TRAIL", "TIME", "TIME_MON", "STOP_BE"]:
        bn = base_exits[k]
        tn = trail_exits[k]
        bp = bn / base_n  * 100
        tp = tn / trail_n * 100
        if bn == 0 and tn == 0:
            continue
        print(f"  {k:<12} {bn:>8} {bp:>7.1f}%  {tn:>8} {tp:>8.1f}%")

    # Avg R by exit type
    print(f"\n  Avg R by exit type:")
    base_r_by  = _avg_r_by_exit(base_log)
    trail_r_by = _avg_r_by_exit(trail_log)
    print(f"  {'Exit type':<12} {'Baseline':>20} {'Trailing':>20}")
    print("  " + "-" * 54)
    for k in ["TARGET", "STOP", "TRAIL", "TIME", "TIME_MON", "STOP_BE"]:
        bv = base_r_by[k]
        tv = trail_r_by[k]
        if bv.strip() == "—" and tv.strip() == "—":
            continue
        print(f"  {k:<12} {bv:>20} {tv:>20}")

    # Verdict
    delta  = trail_m["avg_R"] - base_m["avg_R"]
    ok_r   = delta > 0.02
    ok_n   = trail_m["n_trades"] >= 30
    verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
    reasons = []
    if not ok_r:
        reasons.append(f"delta={delta:+.3f}R (need >+0.02R)")
    if not ok_n:
        reasons.append(f"only {trail_m['n_trades']} trades (need ≥30)")
    suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
    print(f"\n  Verdict: {verdict}{suffix}")
    return verdict


def _print_regime_breakdown(
    ticker: str,
    base_rm: dict, trail_rm: dict,
) -> None:
    print(f"\n  Regime breakdown — {ticker}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Trail_T':>8} {'Trail_AvgR':>11} {'delta_R':>8}")
    print("  " + "-" * 70)
    regressions = []
    for name in REGIMES:
        bm = base_rm.get(name, {})
        cm = trail_rm.get(name, {})
        b_t = bm.get("n_trades", 0)
        c_t = cm.get("n_trades", 0)
        if b_t == 0 and c_t == 0:
            continue
        b_r = bm.get("avg_R", float("nan"))
        c_r = cm.get("avg_R", float("nan"))
        delta = c_r - b_r if (b_t > 0 and c_t > 0) else float("nan")
        delta_str = f"{delta:>+8.3f}" if not pd.isna(delta) else "     N/A"
        flag = "  ← regression" if (not pd.isna(delta) and delta < -0.02) else ""
        print(f"  {name:<22} {b_t:>7} {b_r:>10.3f} {c_t:>8} {c_r:>11.3f} {delta_str}{flag}")
        if not pd.isna(delta) and delta < -0.02:
            regressions.append(name)
    if regressions:
        print(f"\n  Fragility note: trailing stop regresses in {len(regressions)} "
              f"regime(s): {', '.join(regressions)}")
    else:
        print(f"\n  No regime regressions > 0.02R.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Trailing Stop Test  —  Var1+Var2+Var4 baseline")
    print("Breakeven at 1R  |  Trail at bar_high − 0.5×ATR above 1.5R")
    print("=" * 72)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_verdicts: dict[str, str] = {}
    all_results: dict[str, tuple] = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log,  base_m,  base_rm  = run_backtest(ticker_df, ticker, trailing_stop=False)
        trail_log, trail_m, trail_rm = run_backtest(ticker_df, ticker, trailing_stop=True)

        all_results[ticker] = (base_log, base_m, base_rm, trail_log, trail_m, trail_rm)
        verdict = _print_comparison(ticker, base_log, base_m, trail_log, trail_m)
        all_verdicts[ticker] = verdict

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*72}")
    btc_v = all_verdicts.get("BTC-USD", "—")
    eth_v = all_verdicts.get("ETH-USD", "—")
    combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
    print(f"  BTC={btc_v}  ETH={eth_v}  →  {combined}")

    # ── Regime breakdown if promoted ─────────────────────────────────────
    if combined == "PROMOTE":
        print(f"\n{'='*72}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*72}")
        for ticker in TICKERS:
            _, _, base_rm, _, _, trail_rm = all_results[ticker]
            _print_regime_breakdown(ticker, base_rm, trail_rm)
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
