"""
crypto/market_structure_test.py
================================
Tests the market structure filter against the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — split result, BTC and ETH disagree
# Removal rate:  AND mode removes 23% BTC / 20% ETH signals.
#                OR  mode removes  6% BTC /  4% ETH signals.
# Critical       Filter removes BTC's BETTER trades. BTC filtered avg R +0.194
# finding:       vs kept +0.122 (AND mode). BTC filtered avg R +0.252 vs kept
#                +0.131 (OR mode). Opposite of hypothesis on BTC — the filter
#                is actively selecting against good BTC trades.
# ETH result:    AND mode would promote in isolation: +0.044R improvement,
#                77 trades, filtered ETH trades avg R +0.021 vs kept +0.244.
#                Correct direction on ETH.
# Asymmetry      BTC makes broader, less structured intraday moves — 10-day
# hypothesis:    high/low structure does not predict weekend edge for BTC.
#                ETH trend quality is better reflected in the 5-bar structure.
# Combined test: FAILS. A filter that helps ETH but hurts BTC cannot be
#                promoted as a system-wide rule.
# Future test:   Instrument-specific filters — apply market structure to ETH
#                only, skip for BTC. Would require separate signal logic per
#                instrument in main.py. Test after Phase 1 is complete.
# Status:        Do not add to main.py.
# ─────────────────────────────────────────────────────────────────────────────

Filter logic — split the 10 daily bars immediately before Friday into halves:
  Recent 5 (days 1-5): recent_high = max(high), recent_low = min(low)
  Prior  5 (days 6-10): prior_high = max(high), prior_low = min(low)

  AND mode (default): LONG only if recent_high > prior_high AND recent_low > prior_low
  OR  mode:           LONG only if recent_high > prior_high OR  recent_low > prior_low

Fewer than 10 prior bars → NO_TRADE (never default-allow).

Promotion criteria (all must hold):
  1. Net avg R improvement > 0.02R vs baseline
  2. Sample stays above 30 trades
  3. Improvement holds on both BTC and ETH independently
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import run_backtest, STARTING_CAPITAL, REGIMES

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Removal rate preview ──────────────────────────────────────────────────────

def _removal_preview(base_log: pd.DataFrame, ticker: str) -> None:
    """Break down how many LONG signals fail HH only, HL only, or both."""
    longs = base_log[base_log["direction"] == "LONG"].copy()
    n = len(longs)
    if n == 0:
        return

    hh_only = int(((~longs["ms_higher_highs"]) & longs["ms_higher_lows"]).sum())
    hl_only = int((longs["ms_higher_highs"] & (~longs["ms_higher_lows"])).sum())
    both    = int(((~longs["ms_higher_highs"]) & (~longs["ms_higher_lows"])).sum())
    either  = int(((~longs["ms_higher_highs"]) | (~longs["ms_higher_lows"])).sum())

    print(f"\n  {ticker} — removal rate preview ({n} baseline LONG signals):")
    print(f"    Filtered by AND (both fail)       : {either:>3}  ({either/n*100:.1f}%)")
    print(f"      → fails HH only                : {hh_only:>3}  ({hh_only/n*100:.1f}%)")
    print(f"      → fails HL only                : {hl_only:>3}  ({hl_only/n*100:.1f}%)")
    print(f"      → fails both HH and HL         : {both:>3}  ({both/n*100:.1f}%)")
    print(f"    Filtered by OR  (both required)   : {both:>3}  ({both/n*100:.1f}%)")


# ── Quality: filtered vs kept ─────────────────────────────────────────────────

def _quality_breakdown(base_log: pd.DataFrame, mode: str) -> None:
    longs = base_log[base_log["direction"] == "LONG"].dropna(subset=["R_multiple"])
    if mode == "AND":
        passes = longs["ms_higher_highs"] & longs["ms_higher_lows"]
    else:
        passes = longs["ms_higher_highs"] | longs["ms_higher_lows"]

    kept     = longs[passes]
    filtered = longs[~passes]

    def _fmt(subset: pd.DataFrame, label: str) -> str:
        if subset.empty:
            return f"  {label}: n=0"
        wr  = (subset["R_multiple"] > 0).mean() * 100
        avr = subset["R_multiple"].mean()
        return f"  {label}: n={len(subset):>3}  WR={wr:>5.1f}%  AvgR={avr:>+7.3f}"

    print(f"    {_fmt(kept,     'Kept by filter  ')}")
    print(f"    {_fmt(filtered, 'Filtered out    ')}")


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_table(
    ticker: str,
    versions: dict[str, tuple[pd.DataFrame, dict]],
    base_key: str,
) -> dict[str, str]:
    base_m = versions[base_key][1]

    print(f"\n{'='*72}")
    print(f"  MARKET STRUCTURE FILTER — {ticker}")
    print(f"{'='*72}")
    hdr = (f"  {'Version':<16} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 68)

    verdicts: dict[str, str] = {}
    for label, (tl, m) in versions.items():
        if not m:
            print(f"  {label:<16}  — no trades —")
            continue
        delta_str = "    base" if label == base_key else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<16} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}{low_n}"
        )

    # Quality breakdown and verdicts for non-baseline versions
    print(f"\n  Avg R — filtered vs kept trades (from baseline):")
    for label, (tl, m) in versions.items():
        if label == base_key or not m:
            continue
        mode = "AND" if "AND" in label else "OR"
        base_tl = versions[base_key][0]
        print(f"    {label} ({mode} mode):")
        _quality_breakdown(base_tl, mode)

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
        print(f"\n  {label:<16}: {verdict}{suffix}")
        verdicts[label] = verdict

    return verdicts


# ── Regime breakdown ──────────────────────────────────────────────────────────

def _regime_breakdown(ticker: str, label: str, base_rm: dict, ms_rm: dict) -> list[str]:
    print(f"\n  Regime breakdown — {ticker} {label}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'MS_T':>6} {'MS_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 67)
    regressions = []
    for name in REGIMES:
        bm = base_rm.get(name, {})
        mm = ms_rm.get(name, {})
        b_t = bm.get("n_trades", 0)
        m_t = mm.get("n_trades", 0)
        if b_t == 0 and m_t == 0:
            continue
        b_r = bm.get("avg_R", float("nan"))
        m_r = mm.get("avg_R", float("nan"))
        delta = m_r - b_r if (b_t > 0 and m_t > 0) else float("nan")
        ds    = f"{delta:>+8.3f}" if not pd.isna(delta) else "     N/A"
        flag  = "  ← regression" if (not pd.isna(delta) and delta < -0.02) else ""
        print(f"  {name:<22} {b_t:>7} {b_r:>10.3f} {m_t:>6} {m_r:>10.3f} {ds}{flag}")
        if not pd.isna(delta) and delta < -0.02:
            regressions.append(name)
    return regressions


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Market Structure Filter Test  —  Var1+Var2+Var4 baseline")
    print("HH+HL check over prior 10 bars (split into two 5-bar halves)")
    print("=" * 72)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_verdicts: dict[str, dict[str, str]] = {}
    all_results:  dict[str, dict] = {}

    # ── Removal preview (uses baseline to inspect ms flags) ──────────────
    print(f"\n{'─'*60}")
    print("  REMOVAL RATE PREVIEW")
    print(f"{'─'*60}")
    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)
        base_log, _, _ = run_backtest(ticker_df, ticker)
        _removal_preview(base_log, ticker)

    # ── Full comparison ───────────────────────────────────────────────────
    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log,  base_m,  base_rm  = run_backtest(ticker_df, ticker)
        and_log,   and_m,   and_rm   = run_backtest(ticker_df, ticker,
                                                     market_structure_filter=True,
                                                     market_structure_mode="AND")
        or_log,    or_m,    or_rm    = run_backtest(ticker_df, ticker,
                                                     market_structure_filter=True,
                                                     market_structure_mode="OR")

        versions = {
            "Baseline":      (base_log, base_m),
            "MS AND":        (and_log,  and_m),
            "MS OR":         (or_log,   or_m),
        }
        all_results[ticker] = {
            "base_rm": base_rm, "and_rm": and_rm, "or_rm": or_rm,
            "base_log": base_log,
        }

        verdicts = _print_table(ticker, versions, "Baseline")
        all_verdicts[ticker] = verdicts

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*72}")
    promoted = []
    for label in ["MS AND", "MS OR"]:
        btc_v = all_verdicts.get("BTC-USD", {}).get(label, "—")
        eth_v = all_verdicts.get("ETH-USD", {}).get(label, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {label:<10}: BTC={btc_v}  ETH={eth_v}  →  {combined}")
        if combined == "PROMOTE":
            promoted.append(label)

    # ── Regime breakdown if promoted ──────────────────────────────────────
    if promoted:
        print(f"\n{'='*72}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*72}")
        for label in promoted:
            rm_key = "and_rm" if "AND" in label else "or_rm"
            for ticker in TICKERS:
                base_rm = all_results[ticker]["base_rm"]
                ms_rm   = all_results[ticker][rm_key]
                regs    = _regime_breakdown(ticker, label, base_rm, ms_rm)
                if regs:
                    print(f"    Regressions in: {', '.join(regs)}")
                else:
                    print(f"    No regime regressions.")
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
