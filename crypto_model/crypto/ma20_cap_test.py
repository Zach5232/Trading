"""
crypto/ma20_cap_test.py
========================
Tests the MA20 distance cap as Filter 4 on top of Var1+Var2+Var4.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — all three thresholds fail
# Key finding:   The median LONG signal is already 9.6% above MA20 (BTC) and
#                14.1% (ETH) after the three-filter system. The cap eliminates
#                58–94% of signals depending on threshold, cutting deeply into
#                the core of the signal distribution rather than trimming outliers.
# 8% cap:        Closest result — BTC +0.005R (below 0.02R bar), ETH +0.020R
#                but only 25 trades (below 30-trade floor). Does not promote.
# Underlying     Crypto momentum signals that pass MA20 + ATR expansion +
# reason:        momentum confirmation are already in confirmed uptrends.
#                Distance from MA20 at that point is a characteristic of strong
#                trends, not overextension. A distance cap would need to be
#                tested as a pre-filter before the other three, not after —
#                that is a different hypothesis and requires a separate test if
#                revisited.
# Status:        Do not add to main.py.
# ─────────────────────────────────────────────────────────────────────────────

Filter logic: skip LONG when (close - MA20) / MA20 > threshold
Rationale: overextended entries have compressed R:R and elevated mean-reversion risk.

Versions tested: Baseline / 3% cap / 5% cap / 8% cap
Instruments:     BTC-USD, ETH-USD

Promotion criteria (all three must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on BOTH BTC and ETH independently

If any threshold promotes, regime breakdown is printed to verify the
improvement is not concentrated in a single market period.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import (
    run_backtest,
    STARTING_CAPITAL,
    RISK_PCT_PER_TRADE,
    REGIMES,
    _calc_metrics,
)

CAPS = {
    "Baseline": None,
    "3% cap":   0.03,
    "5% cap":   0.05,
    "8% cap":   0.08,
}

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _n_filtered(trade_log: pd.DataFrame, cap: float | None) -> int:
    """Count LONG signals removed specifically by the MA20 distance cap."""
    if cap is None:
        return 0
    return int(
        (trade_log["no_trade_reason"] == "price overextended above MA20").sum()
    )


def _print_comparison(ticker: str, results: dict[str, tuple]) -> dict[str, str]:
    """
    Print comparison table for one ticker.
    results: {label: (trade_log, metrics, regime_metrics)}
    Returns {label: verdict} for each non-baseline version.
    """
    base_m = results["Baseline"][1]

    print(f"\n{'='*76}")
    print(f"  MA20 DISTANCE CAP — {ticker}")
    print(f"{'='*76}")
    hdr = (f"  {'Version':<12} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'Filtered':>9} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 72)

    verdicts: dict[str, str] = {}
    for label, (tl, m, _) in results.items():
        if not m:
            print(f"  {label:<12}  — no trades —")
            continue
        n_filt = _n_filtered(tl, CAPS[label])
        if label == "Baseline":
            delta_str = "    base"
        else:
            delta = m["avg_R"] - base_m["avg_R"]
            delta_str = f"{delta:>+8.3f}"
        low_n = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<12} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {n_filt:>9} {delta_str}{low_n}"
        )

    # Verdicts
    print()
    for label, (tl, m, _) in results.items():
        if label == "Baseline" or not m:
            continue
        delta  = m["avg_R"] - base_m["avg_R"]
        ok_r   = delta > 0.02
        ok_n   = m["n_trades"] >= 30
        verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
        reasons = []
        if not ok_r:
            reasons.append(f"delta={delta:+.3f}R (need >+0.02R)")
        if not ok_n:
            reasons.append(f"only {m['n_trades']} trades (need ≥30)")
        suffix = f"  [{'; '.join(reasons)}]" if reasons else ""
        print(f"  {label:<12}: {verdict}{suffix}")
        verdicts[label] = verdict

    return verdicts


def _print_regime_breakdown(ticker: str, label: str, regime_metrics: dict,
                             base_regime_metrics: dict) -> None:
    print(f"\n  Regime breakdown — {ticker} {label} vs Baseline:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'Cap_T':>7} {'Cap_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 68)
    for name in REGIMES:
        bm = base_regime_metrics.get(name, {})
        cm = regime_metrics.get(name, {})
        b_t  = bm.get("n_trades", 0)
        b_r  = bm.get("avg_R", float("nan"))
        c_t  = cm.get("n_trades", 0)
        c_r  = cm.get("avg_R", float("nan"))
        if b_t == 0 and c_t == 0:
            continue
        delta = c_r - b_r if (b_t > 0 and c_t > 0) else float("nan")
        delta_str = f"{delta:>+8.3f}" if not pd.isna(delta) else "     N/A"
        print(f"  {name:<22} {b_t:>7} {b_r:>10.3f} {c_t:>7} {c_r:>10.3f} {delta_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 76)
    print("MA20 Distance Cap Filter Test  —  Var1+Var2+Var4 baseline")
    print("=" * 76)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_results: dict[str, dict] = {}   # ticker → {label: (tl, m, rm)}
    all_verdicts: dict[str, dict] = {}  # ticker → {label: verdict}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)
        ticker_results: dict[str, tuple] = {}

        for label, cap in CAPS.items():
            tl, m, rm = run_backtest(ticker_df, ticker,
                                     ma20_distance_cap=cap)
            ticker_results[label] = (tl, m, rm)

        all_results[ticker] = ticker_results
        verdicts = _print_comparison(ticker, ticker_results)
        all_verdicts[ticker] = verdicts

    # ── MA20 distance distribution (informational) ────────────────────────
    print(f"\n{'='*76}")
    print("  MA20 DISTANCE DISTRIBUTION  (LONG signals, all dates)")
    print(f"{'='*76}")
    for ticker in TICKERS:
        tl_base, _, _ = all_results[ticker]["Baseline"]
        longs = tl_base[tl_base["direction"] == "LONG"]["ma20_distance"].dropna()
        if longs.empty:
            continue
        print(f"\n  {ticker}  (n={len(longs)} LONG signals)")
        for pct, label in [(25, "25th pct"), (50, "median"), (75, "75th pct"),
                           (90, "90th pct"), (95, "95th pct")]:
            print(f"    {label}: {longs.quantile(pct/100):.1f}%")
        print(f"    Mean:     {longs.mean():.1f}%")
        print(f"    Max:      {longs.max():.1f}%")
        n_above = {
            "3%": (longs > 3).sum(),
            "5%": (longs > 5).sum(),
            "8%": (longs > 8).sum(),
        }
        for thresh, n in n_above.items():
            print(f"    > {thresh}:  {n} trades ({n/len(longs)*100:.1f}%)")

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*76}")
    promoted = []
    for label in [k for k in CAPS if k != "Baseline"]:
        btc_v = all_verdicts.get("BTC-USD", {}).get(label, "—")
        eth_v = all_verdicts.get("ETH-USD", {}).get(label, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {label:<12}: BTC={btc_v}  ETH={eth_v}  →  {combined}")
        if combined == "PROMOTE":
            promoted.append(label)

    # ── Regime breakdown for any promoted thresholds ──────────────────────
    if promoted:
        print(f"\n{'='*76}")
        print("  REGIME BREAKDOWN FOR PROMOTED THRESHOLD(S)")
        print(f"{'='*76}")
        for label in promoted:
            for ticker in TICKERS:
                _, _, cap_rm   = all_results[ticker][label]
                _, _, base_rm  = all_results[ticker]["Baseline"]
                _print_regime_breakdown(ticker, label, cap_rm, base_rm)

            # Fragility check
            print(f"\n  Fragility check — {label}:")
            for ticker in TICKERS:
                _, _, cap_rm  = all_results[ticker][label]
                _, _, base_rm = all_results[ticker]["Baseline"]
                improvements = []
                regressions  = []
                for name in REGIMES:
                    bm = base_rm.get(name, {})
                    cm = cap_rm.get(name, {})
                    if bm.get("n_trades", 0) > 3 and cm.get("n_trades", 0) > 0:
                        d = cm["avg_R"] - bm["avg_R"]
                        if d > 0.02:
                            improvements.append(name)
                        elif d < -0.02:
                            regressions.append(name)
                print(f"    {ticker}: improves in {len(improvements)} regime(s)"
                      f", regresses in {len(regressions)} regime(s)")
                if regressions:
                    print(f"      Regressions: {', '.join(regressions)}"
                          f"  ← filter is fragile if gains concentrated elsewhere")
    else:
        print("\n  No thresholds promoted — no regime breakdown needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
