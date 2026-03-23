"""
crypto/time_stop_test.py
========================
Tests the Saturday time stop against the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — fires on 82% of BTC and 81% of ETH trades,
#                far too broad
# Key finding:   The 1R threshold is too high — only ~18% of trades reach 1R
#                on the Saturday bar, so the time stop fires on almost
#                everything. TIME_SAT avg R: BTC -0.089, ETH -0.065. If those
#                same trades held to Sunday: BTC +0.107, ETH +0.110. 58% of
#                TIME_SAT trades on BTC and 56% on ETH would have been positive
#                by Sunday close.
# Structural     Saturday-stagnant trades that recover Sunday ARE the Var4
# problem 1:     edge. Exiting them early forfeits the core source of returns.
# Structural     Kills the Monday hold entirely — all 23 BTC TIME_MON and 36
# problem 2:     STOP_BE exits convert to TIME_SAT, disabling Var4.
# Net delta:     BTC -0.097R. ETH -0.082R.
# Approximation  Daily bar cannot model a 3pm ET intrabar exit — fires any time
# limitation:    price never reached 1R across the full 24-hour Saturday bar.
# Revisit        Test with hourly data and a lower/different threshold (e.g.
# hypothesis:    high < 0.5R AND below entry) after Phase 1 complete, when
#                hourly data sourcing is worth the effort.
# Status:        Do not add to main.py.
# ─────────────────────────────────────────────────────────────────────────────

APPROXIMATION NOTE: This test uses daily bars. The "Saturday 3pm ET"
exit is modelled as: exit at Saturday CLOSE if Saturday HIGH < 1R level.
This is conservative — it only fires when we can confirm with certainty
the trade never reached 1R at ANY point on Saturday. If Saturday HIGH
>=1R, the trade is assumed to have potentially reached 1R and continues
to Sunday. If this test promotes, hourly data would be needed to validate
the 3pm intrabar timing before adding to main.py.

Filter logic:
  1R level = entry + (entry - stop)
  If Saturday HIGH < 1R level → exit Saturday close, type "TIME_SAT"
  If Saturday HIGH >= 1R level → continue to Sunday as normal

Fewer than 2 weekend bars → no change in behavior.

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
from backtest_engine import run_backtest, STARTING_CAPITAL, REGIMES, ATR_MULT_STOP

TICKERS = ["BTC-USD", "ETH-USD"]


# ── Sunday recovery analysis ──────────────────────────────────────────────────

def _sunday_recovery(
    time_sat_trades: pd.DataFrame,
    ticker_df: pd.DataFrame,
) -> None:
    """
    For TIME_SAT trades, look up the Sunday bar and determine whether
    holding through Sunday would have been positive or negative.
    """
    if time_sat_trades.empty:
        print("    No TIME_SAT trades to analyse.")
        return

    positive_sun = 0
    negative_sun = 0
    sun_r_values = []

    for _, row in time_sat_trades.iterrows():
        fri_date  = row["date"]
        entry     = row["entry_price"]
        stop      = row["stop_price"]
        target    = row["target_price"]
        risk_dol  = entry - stop   # risk per unit (used for R calc)

        # Find Friday position in ticker_df
        try:
            fri_pos = ticker_df.index.get_loc(fri_date)
        except KeyError:
            continue

        # Look for Sunday bar (dayofweek==6) in the next 3 bars
        candidates = ticker_df.iloc[fri_pos + 1 : fri_pos + 4]
        sun_bars   = candidates[candidates.index.dayofweek == 6]

        if sun_bars.empty:
            continue

        sun = sun_bars.iloc[0]

        # Determine Sunday outcome
        if sun["low"] <= stop:
            sun_exit = stop
        elif sun["high"] >= target:
            sun_exit = target
        else:
            sun_exit = sun["close"]

        sun_r = (sun_exit - entry) / risk_dol if risk_dol > 0 else 0.0
        sun_r_values.append(sun_r)

        if sun_exit > entry:
            positive_sun += 1
        else:
            negative_sun += 1

    total = positive_sun + negative_sun
    if total == 0:
        print("    No Sunday bars found for TIME_SAT trades.")
        return

    avg_sun_r = float(np.mean(sun_r_values)) if sun_r_values else float("nan")
    print(f"    Sunday outcome for TIME_SAT trades ({total} found):")
    print(f"      Would have been positive Sunday : {positive_sun:>3}  ({positive_sun/total*100:.1f}%)")
    print(f"      Would have been negative Sunday : {negative_sun:>3}  ({negative_sun/total*100:.1f}%)")
    print(f"      Avg R if held to Sunday         : {avg_sun_r:>+7.3f}")


# ── Removal rate preview ──────────────────────────────────────────────────────

def _removal_preview(
    ts_log: pd.DataFrame,
    base_log: pd.DataFrame,
    ticker: str,
    ticker_df: pd.DataFrame,
) -> None:
    longs_base = base_log[base_log["direction"] == "LONG"]
    longs_ts   = ts_log[ts_log["direction"] == "LONG"]
    time_sat   = longs_ts[longs_ts["exit_type"] == "TIME_SAT"]

    n_base = len(longs_base)
    n_sat  = len(time_sat)

    base_time = longs_base[longs_base["exit_type"] == "TIME"]
    ts_time   = longs_ts[longs_ts["exit_type"] == "TIME"]

    print(f"\n  {ticker} — time stop removal preview:")
    print(f"    Baseline LONG trades    : {n_base}")
    print(f"    TIME_SAT exits          : {n_sat}  ({n_sat/n_base*100:.1f}% of all trades)")
    if not base_time.empty:
        base_time_avg = base_time["R_multiple"].dropna().mean()
        print(f"    Baseline TIME exits     : {len(base_time)}  avg R={base_time_avg:>+7.3f}")
    if not ts_time.empty:
        ts_time_avg = ts_time["R_multiple"].dropna().mean()
        print(f"    Remaining TIME (Sunday) : {len(ts_time)}  avg R={ts_time_avg:>+7.3f}")
    if not time_sat.empty:
        sat_avg = time_sat["R_multiple"].dropna().mean()
        print(f"    TIME_SAT avg R          : {sat_avg:>+7.3f}")

    print(f"\n    Sunday recovery analysis on TIME_SAT exits:")
    _sunday_recovery(time_sat, ticker_df)


# ── Comparison table ──────────────────────────────────────────────────────────

def _print_table(
    ticker: str,
    versions: dict[str, tuple[pd.DataFrame, dict]],
    base_key: str,
) -> dict[str, str]:
    base_m = versions[base_key][1]

    print(f"\n{'='*80}")
    print(f"  SATURDAY TIME STOP — {ticker}")
    print(f"{'='*80}")

    hdr = (f"  {'Version':<16} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
           f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    print(hdr)
    print("  " + "-" * 74)

    verdicts: dict[str, str] = {}
    for label, (tl, m) in versions.items():
        if not m:
            print(f"  {label:<16}  — no trades —")
            continue
        delta_str = "    base" if label == base_key else f"{m['avg_R']-base_m['avg_R']:>+8.3f}"
        low_n     = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        print(
            f"  {label:<16} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str}{low_n}"
        )

    # Exit breakdown
    print(f"\n  Exit breakdown:")
    for label, (tl, m) in versions.items():
        if not m:
            continue
        n     = m["n_trades"]
        tgt   = m.get("exit_TARGET", 0)
        stp   = m.get("exit_STOP",   0)
        tim   = m.get("exit_TIME",   0)
        tsat  = m.get("exit_TIME_SAT", 0)
        tmon  = m.get("exit_TIME_MON", 0)
        be    = m.get("exit_STOP_BE",  0)
        print(f"  {label:<16}  TARGET:{tgt:>3}({tgt/n*100:.0f}%)  STOP:{stp:>3}({stp/n*100:.0f}%)"
              f"  TIME:{tim:>3}({tim/n*100:.0f}%)  TIME_SAT:{tsat:>3}({tsat/n*100:.0f}%)"
              f"  TIME_MON:{tmon}  STOP_BE:{be}")

    # Verdict for non-baseline
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
        print(f"\n  {label:<16}: {verdict}{suffix}")
        verdicts[label] = verdict

    return verdicts


# ── Regime breakdown ──────────────────────────────────────────────────────────

def _regime_breakdown(ticker: str, label: str, base_rm: dict, ts_rm: dict) -> list[str]:
    print(f"\n  Regime breakdown — {ticker} {label}:")
    print(f"  {'Regime':<22} {'Base_T':>7} {'Base_AvgR':>10} "
          f"{'TS_T':>6} {'TS_AvgR':>10} {'delta_R':>8}")
    print("  " + "-" * 67)
    regressions = []
    for name in REGIMES:
        bm = base_rm.get(name, {})
        mm = ts_rm.get(name, {})
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
    print("Saturday Time Stop Test  —  Var1+Var2+Var4 baseline")
    print("Exit at Saturday close if Saturday HIGH < 1R level (entry + 1×risk)")
    print("APPROXIMATION: Daily bars used — cannot model exact 3pm ET intrabar exit")
    print("Conservative: only fires when Saturday HIGH confirms trade never reached 1R")
    print("=" * 80)

    print("\nLoading data...")
    data = load_crypto_data(TICKERS)

    all_verdicts: dict[str, dict[str, str]] = {}
    all_results:  dict[str, dict]           = {}

    for ticker in TICKERS:
        ticker_df = get_ticker_df(data, ticker)

        base_log, base_m, base_rm = run_backtest(ticker_df, ticker)
        ts_log,   ts_m,   ts_rm   = run_backtest(ticker_df, ticker,
                                                  saturday_time_stop=True)

        # ── Removal rate preview ───────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  REMOVAL RATE PREVIEW — {ticker}")
        print(f"{'─'*60}")
        _removal_preview(ts_log, base_log, ticker, ticker_df)

        versions = {
            "Baseline":    (base_log, base_m),
            "Time Stop":   (ts_log,   ts_m),
        }
        all_results[ticker] = {
            "base_rm": base_rm, "ts_rm": ts_rm,
        }

        verdicts = _print_table(ticker, versions, "Baseline")
        all_verdicts[ticker] = verdicts

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*80}")
    promoted = []
    label = "Time Stop"
    btc_v = all_verdicts.get("BTC-USD", {}).get(label, "—")
    eth_v = all_verdicts.get("ETH-USD", {}).get(label, "—")
    combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
    print(f"  {label:<12}: BTC={btc_v}  ETH={eth_v}  →  {combined}")
    if combined == "PROMOTE":
        promoted.append(label)

    # ── Regime breakdown if promoted ──────────────────────────────────────
    if promoted:
        print(f"\n{'='*80}")
        print("  REGIME BREAKDOWN")
        print(f"{'='*80}")
        for label in promoted:
            for ticker in TICKERS:
                base_rm = all_results[ticker]["base_rm"]
                ts_rm   = all_results[ticker]["ts_rm"]
                regs    = _regime_breakdown(ticker, label, base_rm, ts_rm)
                if regs:
                    print(f"    Regressions in: {', '.join(regs)}")
                else:
                    print(f"    No regime regressions.")
    else:
        print("\n  Not promoted — regime breakdown skipped.")

    print(f"\n{'─'*80}")
    print("  APPROXIMATION REMINDER")
    print(f"{'─'*80}")
    print("  This test uses daily bars. 'Saturday 3pm ET' is approximated as:")
    print("  EXIT Saturday close IF Saturday HIGH < 1R level.")
    print("  Fires only when we can confirm the trade NEVER reached 1R on Saturday.")
    print("  If promoted, source hourly data to validate before adding to main.py.")
    print()


if __name__ == "__main__":
    main()
