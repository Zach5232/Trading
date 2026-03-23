"""
crypto/liquidity_trap_test.py
==============================
Tests the liquidity trap detection filter against the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — filter too selective to move the needle
# Key finding:   AND condition flags only 9 BTC signals (6.3%) and 1 ETH signal
#                (1.0%) across full history. Removed BTC trades have avg R +0.074
#                vs +0.143 kept — below average but still positive, not the
#                negative-R blowups the hypothesis predicts. ETH removes a single
#                +1.901R winner, net -0.017R.
# BTC:           Net improvement +0.004R (well below 0.02R bar).
# ETH:           Statistically meaningless with 1 trade flagged.
# Revisit:       Test wider thresholds (2.0×ATR candle size, or lower volume
#                multiplier) as a separate test. OR condition would flag 32%/24%
#                of signals but was not tested — may overcorrect.
# Status:        Do not add to main.py.
# ─────────────────────────────────────────────────────────────────────────────

Filter logic (AND condition — both must be true to reject):
  - Candle size: (high - low) > 1.5 × ATR14
  - Volume spike: volume > 2 × rolling 20-day average volume

Hypothesis: Fridays with both a wide candle and a volume spike signal
institutional/news-driven exhaustion that tends to reverse over the weekend.

Promotion criteria (all must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on both BTC and ETH independently

If promoted, regime breakdown checks whether gains are concentrated in
one period (fragile) or spread across multiple regimes (robust).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays
from backtest_engine import run_backtest, STARTING_CAPITAL, REGIMES

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Volume coverage check ─────────────────────────────────────────────────────

def _check_volume_coverage(ticker_df: pd.DataFrame, ticker: str) -> None:
    vol = ticker_df["volume"]
    valid = vol[vol.notna() & (vol > 0)]
    if valid.empty:
        print(f"  {ticker}: NO volume data available")
        return
    print(f"  {ticker}: volume available {valid.index[0].date()} → {valid.index[-1].date()}"
          f"  ({len(valid)} bars, {len(vol)-len(valid)} gaps)")

    # Check how many LONG signals land on valid volume days
    fridays = get_fridays(ticker_df)
    fri_valid = fridays[fridays["volume"].notna() & (fridays["volume"] > 0)]
    print(f"    Friday bars total: {len(fridays)}  with valid volume: {len(fri_valid)}")


def _removal_rate_preview(ticker_df: pd.DataFrame, ticker: str) -> None:
    """Print how many LONG signals (post 3-filter) the trap filter removes at AND vs OR."""
    # Run baseline to get which Fridays passed 3 filters
    base_log, _, _ = run_backtest(ticker_df, ticker, liquidity_trap_filter=False)
    longs = base_log[base_log["direction"] == "LONG"].copy()

    # Compute trap flags on those rows
    vol_avg20 = ticker_df["volume"].rolling(20).mean().shift(1)

    and_removed = or_removed = 0
    for _, row in longs.iterrows():
        fri_date = row["date"]
        if fri_date not in ticker_df.index:
            continue
        fri_row = ticker_df.loc[fri_date]
        candle_range = float(fri_row["high"]) - float(fri_row["low"])
        atr14        = float(fri_row["atr14"])
        vol_today    = float(fri_row["volume"])
        vol_avg      = vol_avg20.get(fri_date, float("nan"))
        wide  = candle_range > 1.5 * atr14
        spike = (not np.isnan(vol_avg)) and vol_avg > 0 and (vol_today > 2.0 * vol_avg)
        if wide and spike:
            and_removed += 1
        if wide or spike:
            or_removed += 1

    n = len(longs)
    print(f"  {ticker}: {n} baseline LONG signals")
    print(f"    AND condition (wide + spike): removes {and_removed} ({and_removed/n*100:.1f}%)")
    print(f"    OR  condition (wide + spike): removes {or_removed} ({or_removed/n*100:.1f}%)")


# ── Filtered vs kept trade quality ───────────────────────────────────────────

def _filtered_vs_kept(base_log: pd.DataFrame) -> None:
    """Using baseline trade log (which has is_trap column), compare trap vs non-trap R."""
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    trap    = longs[longs["is_trap"] == True]
    no_trap = longs[longs["is_trap"] == False]

    print(f"  Trap trades     : n={len(trap)}", end="")
    if not trap.empty:
        wr = (trap["R_multiple"] > 0).mean() * 100
        print(f"  WR={wr:.1f}%  AvgR={trap['R_multiple'].mean():+.3f}"
              f"  (would be filtered)")
    else:
        print()
    print(f"  Non-trap trades : n={len(no_trap)}", end="")
    if not no_trap.empty:
        wr = (no_trap["R_multiple"] > 0).mean() * 100
        print(f"  WR={wr:.1f}%  AvgR={no_trap['R_multiple'].mean():+.3f}"
              f"  (kept by filter)")
    else:
        print()


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_comparison(
    ticker: str,
    base_log: pd.DataFrame, base_m: dict,
    trap_log: pd.DataFrame, trap_m: dict,
) -> str:
    n_removed = int((trap_log["no_trade_reason"] == "liquidity trap — wide candle + volume spike").sum())

    print(f"\n{'='*72}")
    print(f"  LIQUIDITY TRAP FILTER — {ticker}")
    print(f"{'='*72}")
    hdr = (f"  {'Version':<14} {'Removed':>8} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 68)

    for label, m, removed in [("Baseline", base_m, 0), ("Trap filter", trap_m, n_removed)]:
        if not m:
            print(f"  {label:<14}  — no trades —")
            continue
        delta_str = "    base" if label == "Baseline" else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<14} {removed:>8} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}{low_n}"
        )

    # Filtered vs kept quality
    print(f"\n  Quality of filtered vs kept trades (from baseline):")
    _filtered_vs_kept(base_log)

    # Verdict
    delta   = trap_m["avg_R"] - base_m["avg_R"]
    ok_r    = delta > 0.02
    ok_n    = trap_m["n_trades"] >= 30
    verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
    reasons = []
    if not ok_r:
        reasons.append(f"delta={delta:+.3f}R (need >+0.02R)")
    if not ok_n:
        reasons.append(f"only {trap_m['n_trades']} trades (need ≥30)")
    suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
    print(f"\n  Verdict: {verdict}{suffix}")
    return verdict


def _print_regime_breakdown(ticker: str, base_rm: dict, trap_rm: dict) -> list[str]:
    print(f"\n  Regime breakdown — {ticker}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Trap_T':>7} {'Trap_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 68)
    regressions = []
    for name in REGIMES:
        bm = base_rm.get(name, {})
        cm = trap_rm.get(name, {})
        b_t = bm.get("n_trades", 0)
        c_t = cm.get("n_trades", 0)
        if b_t == 0 and c_t == 0:
            continue
        b_r = bm.get("avg_R", float("nan"))
        c_r = cm.get("avg_R", float("nan"))
        delta = c_r - b_r if (b_t > 0 and c_t > 0) else float("nan")
        delta_str = f"{delta:>+8.3f}" if not pd.isna(delta) else "     N/A"
        flag = "  ← regression" if (not pd.isna(delta) and delta < -0.02) else ""
        print(f"  {name:<22} {b_t:>7} {b_r:>10.3f} {c_t:>7} {c_r:>10.3f} {delta_str}{flag}")
        if not pd.isna(delta) and delta < -0.02:
            regressions.append(name)
    return regressions


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Liquidity Trap Filter Test  —  Var1+Var2+Var4 baseline")
    print("Filter: (high−low) > 1.5×ATR14  AND  volume > 2× 20-day avg")
    print("=" * 72)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    # ── Step 1: Volume coverage check ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  VOLUME DATA COVERAGE")
    print(f"{'─'*60}")
    for ticker in TICKERS:
        _check_volume_coverage(get_ticker_df(data, ticker), ticker)

    print(f"\n{'─'*60}")
    print("  SIGNAL REMOVAL RATE PREVIEW (AND vs OR)")
    print(f"{'─'*60}")
    for ticker in TICKERS:
        _removal_rate_preview(get_ticker_df(data, ticker), ticker)

    # ── Steps 2-4: Run comparison ────────────────────────────────────────
    all_verdicts: dict[str, str] = {}
    all_results: dict[str, tuple] = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)
        base_log, base_m, base_rm = run_backtest(ticker_df, ticker,
                                                  liquidity_trap_filter=False)
        trap_log, trap_m, trap_rm = run_backtest(ticker_df, ticker,
                                                  liquidity_trap_filter=True)
        all_results[ticker] = (base_log, base_m, base_rm, trap_log, trap_m, trap_rm)
        verdict = _print_comparison(ticker, base_log, base_m, trap_log, trap_m)
        all_verdicts[ticker] = verdict

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*72}")
    btc_v = all_verdicts.get("BTC-USD", "—")
    eth_v = all_verdicts.get("ETH-USD", "—")
    combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
    print(f"  BTC={btc_v}  ETH={eth_v}  →  {combined}")

    # ── Step 5: Regime breakdown if promoted ─────────────────────────────
    if combined == "PROMOTE":
        print(f"\n{'='*72}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*72}")
        all_regressions: dict[str, list] = {}
        for ticker in TICKERS:
            _, _, base_rm, _, _, trap_rm = all_results[ticker]
            regressions = _print_regime_breakdown(ticker, base_rm, trap_rm)
            all_regressions[ticker] = regressions

        print(f"\n  Fragility summary:")
        for ticker, regressions in all_regressions.items():
            if regressions:
                print(f"    {ticker}: regresses in {', '.join(regressions)}"
                      f" — filter may be fragile")
            else:
                print(f"    {ticker}: no regime regressions — improvement is robust")
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
