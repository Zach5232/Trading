"""
crypto/parameter_sweep.py
==========================
Parameter grid sweep + ETH standalone backtest + correlation analysis.

Reads backtest_engine.py and data_loader.py — does NOT modify them.
Data is loaded once; the simulation loop is re-implemented internally
with stop_mult and r_target as variables.

Parts:
  1. BTC 5×5 parameter grid  (stop_mult × r_target)
  2. ETH standalone backtest + 5×5 grid
  3. Correlation analysis (BTC vs ETH R multiples on shared signal weeks)

Outputs → Results/crypto_backtest/
  param_sweep_btc.csv
  param_sweep_eth.csv
  eth_standalone_summary.txt
  eth_regime_breakdown.csv
  param_sweep_report.txt
  correlation_analysis.csv
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays

# ── Grid ────────────────────────────────────────────────────────────────────
STOP_MULTS = [0.75, 1.0, 1.25, 1.5, 2.0]
R_TARGETS  = [1.0,  1.5, 2.0,  2.5, 3.0]

# Fixed constants (same as backtest_engine.py)
SLIPPAGE_PCT       = 0.001
STARTING_CAPITAL   = 500.0
RISK_PCT_PER_TRADE = 0.10
TAX_RATE           = 0.32

OUTPUT_DIR = Path(__file__).parent.parent / "Results" / "crypto_backtest"

REGIMES = {
    "2018 Bear":        ("2018-01-01", "2018-12-31"),
    "2019 Recovery":    ("2019-01-01", "2019-12-31"),
    "2020 COVID":       ("2020-01-01", "2020-12-31"),
    "2021 Bull":        ("2021-01-01", "2021-12-31"),
    "2022 Bear":        ("2022-01-01", "2022-12-31"),
    "2023-24 Recovery": ("2023-01-01", "2024-12-31"),
    "2025-Present":     ("2025-01-01", "2099-12-31"),
}


# ── Core simulation loop ────────────────────────────────────────────────────

def _run_sim(
    ticker_df: pd.DataFrame,
    stop_mult: float,
    r_target: float,
    starting_capital: float = STARTING_CAPITAL,
    risk_pct: float = RISK_PCT_PER_TRADE,
) -> pd.DataFrame:
    """
    Walk-forward weekend simulation for a single ticker with given parameters.
    Returns a DataFrame with one row per Friday signal date.
    Columns: date, direction, exit_type, R_multiple, profit_loss
    """
    fridays = get_fridays(ticker_df)
    rows = []
    equity = starting_capital

    for fri_date, fri_row in fridays.iterrows():
        above_ma = bool(fri_row["above_ma20"])

        if not above_ma:
            rows.append(
                {"date": fri_date, "direction": "NO_TRADE",
                 "exit_type": "NO_TRADE", "R_multiple": None, "profit_loss": 0.0}
            )
            continue

        entry         = fri_row["close"] * (1 + SLIPPAGE_PCT)
        atr           = fri_row["atr14"]
        stop          = entry - stop_mult * atr
        target        = entry + r_target * (entry - stop)
        risk_per_unit = entry - stop
        units         = (equity * risk_pct) / risk_per_unit

        # Weekend bars
        fri_pos    = ticker_df.index.get_loc(fri_date)
        next_bars  = ticker_df.iloc[fri_pos + 1 : fri_pos + 3]
        weekend    = next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()

        if weekend.empty:
            rows.append(
                {"date": fri_date, "direction": "LONG",
                 "exit_type": "NO_DATA", "R_multiple": None, "profit_loss": 0.0}
            )
            continue

        exit_price = None
        exit_type  = None
        for _, bar in weekend.iterrows():
            if bar["low"] <= stop:
                exit_price, exit_type = stop, "STOP"
                break
            if bar["high"] >= target:
                exit_price, exit_type = target, "TARGET"
                break

        if exit_price is None:
            exit_price = weekend.iloc[-1]["close"]
            exit_type  = "TIME"

        r_multiple  = (exit_price - entry) / risk_per_unit
        profit_loss = units * (exit_price - entry)
        equity      = max(equity + profit_loss, 0.01)

        rows.append(
            {"date": fri_date, "direction": "LONG",
             "exit_type": exit_type, "R_multiple": round(r_multiple, 4),
             "profit_loss": round(profit_loss, 4)}
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Metrics from sim output ─────────────────────────────────────────────────

def _metrics_from_sim(
    sim: pd.DataFrame,
    stop_mult: float,
    r_target: float,
    starting_capital: float = STARTING_CAPITAL,
) -> dict:
    completed = sim[sim["direction"] == "LONG"].dropna(subset=["R_multiple"])
    if completed.empty:
        return {
            "stop_mult": stop_mult, "r_target": r_target,
            "n_trades": 0, "win_rate": None, "avg_R": None,
            "profit_factor": None, "expectancy_R": None, "max_drawdown": None,
            "pct_target": None, "pct_stop": None, "pct_time": None,
            "roi_pre_tax": None,
        }

    wins   = completed[completed["R_multiple"] > 0]
    losses = completed[completed["R_multiple"] <= 0]

    win_rate      = len(wins) / len(completed)
    avg_R         = completed["R_multiple"].mean()
    avg_win_R     = wins["R_multiple"].mean()   if not wins.empty   else 0.0
    avg_loss_R    = losses["R_multiple"].mean() if not losses.empty else 0.0
    gross_profit  = wins["R_multiple"].sum()
    gross_loss    = abs(losses["R_multiple"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    expectancy_R  = win_rate * avg_win_R + (1 - win_rate) * avg_loss_R

    equity = [starting_capital]
    for pl in completed["profit_loss"]:
        equity.append(max(equity[-1] + pl, 0.01))
    equity_s    = pd.Series(equity)
    running_max = equity_s.cummax()
    max_dd      = float(((equity_s - running_max) / running_max).min())
    total_pl    = equity[-1] - starting_capital
    roi_pre_tax = total_pl / starting_capital

    ec = completed["exit_type"].value_counts().to_dict()
    n  = len(completed)

    return {
        "stop_mult":     stop_mult,
        "r_target":      r_target,
        "n_trades":      n,
        "win_rate":      round(win_rate * 100, 1),
        "avg_R":         round(avg_R, 3),
        "profit_factor": round(min(profit_factor, 999.0), 2),
        "expectancy_R":  round(expectancy_R, 3),
        "max_drawdown":  round(max_dd * 100, 1),
        "pct_target":    round(ec.get("TARGET", 0) / n * 100, 1),
        "pct_stop":      round(ec.get("STOP", 0)   / n * 100, 1),
        "pct_time":      round(ec.get("TIME", 0)   / n * 100, 1),
        "exit_TARGET":   ec.get("TARGET", 0),
        "exit_STOP":     ec.get("STOP", 0),
        "exit_TIME":     ec.get("TIME", 0),
        "roi_pre_tax":   round(roi_pre_tax * 100, 1),
    }


# ── Regime breakdown for a sim result ──────────────────────────────────────

def _regime_breakdown(
    sim: pd.DataFrame,
    stop_mult: float = 1.0,
    r_target: float = 2.0,
    starting_capital: float = STARTING_CAPITAL,
) -> list[dict]:
    rows = []
    for regime_name, (r_start, r_end) in REGIMES.items():
        subset = sim[(sim["date"] >= r_start) & (sim["date"] <= r_end)]
        m = _metrics_from_sim(subset, stop_mult, r_target, starting_capital)
        m["regime"] = regime_name
        rows.append(m)
    return rows


# ── Grid runner ─────────────────────────────────────────────────────────────

def run_grid(
    ticker_df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the 5×5 grid. Returns (grid_results_df, baseline_sim_df).
    baseline_sim_df is the sim for stop=1.0, r_target=2.0.
    """
    grid_rows   = []
    baseline_sim = None

    combos = list(product(STOP_MULTS, R_TARGETS))
    print(f"\n  Running {len(combos)}-combo grid for {label}...")

    for stop_mult, r_target in combos:
        sim = _run_sim(ticker_df, stop_mult, r_target)
        m   = _metrics_from_sim(sim, stop_mult, r_target)
        grid_rows.append(m)

        if stop_mult == 1.0 and r_target == 2.0:
            baseline_sim = sim

        # Progress line
        wr_str  = f"{m['win_rate']}%"  if m["win_rate"]      is not None else "N/A"
        avgr    = f"{m['avg_R']}"      if m["avg_R"]         is not None else "N/A"
        pf_str  = f"{m['profit_factor']}" if m["profit_factor"] is not None else "N/A"
        dd_str  = f"{m['max_drawdown']}%" if m["max_drawdown"]  is not None else "N/A"
        print(
            f"    stop={stop_mult}x  target={r_target}R  →  "
            f"WR={wr_str}  AvgR={avgr}  PF={pf_str}  DD={dd_str}"
        )

    return pd.DataFrame(grid_rows), baseline_sim


# ── Correlation analysis ────────────────────────────────────────────────────

def correlation_analysis(
    btc_sim: pd.DataFrame,
    eth_sim: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Align BTC and ETH LONG trades by Friday date and compute correlation stats.
    """
    btc_long = btc_sim[btc_sim["direction"] == "LONG"].dropna(subset=["R_multiple"])
    eth_long = eth_sim[eth_sim["direction"] == "LONG"].dropna(subset=["R_multiple"])

    merged = btc_long[["date", "R_multiple"]].merge(
        eth_long[["date", "R_multiple"]],
        on="date",
        suffixes=("_btc", "_eth"),
    )

    if merged.empty:
        return pd.DataFrame(), {"n_shared": 0}

    merged["both_win"]  = (merged["R_multiple_btc"] > 0) & (merged["R_multiple_eth"] > 0)
    merged["both_loss"] = (merged["R_multiple_btc"] <= 0) & (merged["R_multiple_eth"] <= 0)
    merged["diverged"]  = ~(merged["both_win"] | merged["both_loss"])

    n = len(merged)
    corr = merged["R_multiple_btc"].corr(merged["R_multiple_eth"])

    stats = {
        "n_shared":    n,
        "both_win":    int(merged["both_win"].sum()),
        "both_loss":   int(merged["both_loss"].sum()),
        "diverged":    int(merged["diverged"].sum()),
        "pct_both_win":  round(merged["both_win"].sum() / n * 100, 1),
        "pct_both_loss": round(merged["both_loss"].sum() / n * 100, 1),
        "pct_diverged":  round(merged["diverged"].sum() / n * 100, 1),
        "r_correlation": round(corr, 3),
    }

    corr_df = merged[["date", "R_multiple_btc", "R_multiple_eth",
                       "both_win", "both_loss", "diverged"]].rename(
        columns={"R_multiple_btc": "btc_R", "R_multiple_eth": "eth_R"}
    )
    return corr_df, stats


# ── Report writer ───────────────────────────────────────────────────────────

def _grid_table(grid: pd.DataFrame) -> list[str]:
    """Format a grid DataFrame as fixed-width text rows."""
    header = (
        f"  {'Stop':>6}  {'Target':>7}  {'Trades':>6}  "
        f"{'WR%':>5}  {'AvgR':>6}  {'PF':>5}  "
        f"{'DD%':>7}  {'T%':>5}  {'S%':>5}  {'Ti%':>5}"
    )
    sep = "  " + "-" * 68
    lines = [header, sep]

    sorted_grid = grid.sort_values("expectancy_R", ascending=False)
    for _, r in sorted_grid.iterrows():
        lines.append(
            f"  {r['stop_mult']:>5.2f}x  {r['r_target']:>6.1f}R  "
            f"{int(r['n_trades']) if r['n_trades'] else 0:>6}  "
            f"{r['win_rate'] if r['win_rate'] is not None else 'N/A':>5}  "
            f"{r['avg_R'] if r['avg_R'] is not None else 'N/A':>6}  "
            f"{r['profit_factor'] if r['profit_factor'] is not None else 'N/A':>5}  "
            f"{r['max_drawdown'] if r['max_drawdown'] is not None else 'N/A':>7}  "
            f"{r['pct_target'] if r['pct_target'] is not None else 'N/A':>5}  "
            f"{r['pct_stop'] if r['pct_stop'] is not None else 'N/A':>5}  "
            f"{r['pct_time'] if r['pct_time'] is not None else 'N/A':>5}"
        )
    return lines


def _top3_commentary(grid: pd.DataFrame, label: str) -> list[str]:
    top3 = grid.sort_values("expectancy_R", ascending=False).head(3)
    lines = [f"\nTop 3 {label} combinations:"]
    for rank, (_, r) in enumerate(top3.iterrows(), 1):
        time_pct = r["pct_time"]
        time_note = (
            "high TIME exits — target may still be too far"
            if time_pct > 60
            else "moderate TIME exits" if time_pct > 40
            else "low TIME exits — target reachable in 48h"
        )
        stop_note = (
            "tight stop — watch for stop-hunts"
            if r["stop_mult"] < 1.0
            else "wide stop — larger dollar risk per trade"
            if r["stop_mult"] >= 1.5
            else "standard stop"
        )
        lines.append(
            f"  #{rank}  stop={r['stop_mult']}x  target={r['r_target']}R  "
            f"→  expectancy={r['expectancy_R']}R  |  "
            f"{time_note}; {stop_note}"
        )
    return lines


def write_report(
    btc_grid: pd.DataFrame,
    eth_grid: pd.DataFrame,
    btc_baseline_m: dict,
    eth_baseline_m: dict,
    corr_stats: dict,
    out_dir: Path,
) -> None:
    lines = [
        "=" * 65,
        "PARAMETER SWEEP REPORT — BTC + ETH Weekend MA20 System",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "PART 1 — BTC PARAMETER GRID",
        "(sorted by expectancy_R, best to worst)",
        "",
    ]
    lines += _grid_table(btc_grid)
    lines += _top3_commentary(btc_grid, "BTC")

    lines += [
        "",
        "-" * 65,
        "PART 2 — ETH PARAMETER GRID",
        "(sorted by expectancy_R, best to worst)",
        "",
    ]
    lines += _grid_table(eth_grid)
    lines += _top3_commentary(eth_grid, "ETH")

    # BTC vs ETH at baseline
    def _fmt(m: dict, key: str, suffix: str = "") -> str:
        v = m.get(key)
        return f"{v}{suffix}" if v is not None else "N/A"

    lines += [
        "",
        "-" * 65,
        "PART 3 — BTC vs ETH AT V1 BASELINE (stop=1.0x, target=2.0R)",
        "",
        f"  BTC: WR={_fmt(btc_baseline_m,'win_rate','%')}  "
        f"AvgR={_fmt(btc_baseline_m,'avg_R')}  "
        f"PF={_fmt(btc_baseline_m,'profit_factor')}  "
        f"DD={_fmt(btc_baseline_m,'max_drawdown','%')}  "
        f"ExpR={_fmt(btc_baseline_m,'expectancy_R')}",
        f"  ETH: WR={_fmt(eth_baseline_m,'win_rate','%')}  "
        f"AvgR={_fmt(eth_baseline_m,'avg_R')}  "
        f"PF={_fmt(eth_baseline_m,'profit_factor')}  "
        f"DD={_fmt(eth_baseline_m,'max_drawdown','%')}  "
        f"ExpR={_fmt(eth_baseline_m,'expectancy_R')}",
        "",
    ]

    btc_exp = btc_baseline_m.get("expectancy_R", 0) or 0
    eth_exp = eth_baseline_m.get("expectancy_R", 0) or 0
    if abs(btc_exp - eth_exp) < 0.02:
        better = "Comparable"
    elif btc_exp > eth_exp:
        better = "BTC"
    else:
        better = "ETH"
    lines.append(f"  Better instrument at baseline: {better}")

    # Correlation section
    n = corr_stats.get("n_shared", 0)
    corr_val = corr_stats.get("r_correlation", 0) or 0
    if corr_val > 0.6:
        verdict = "High correlation — trading both primarily doubles risk, not diversification"
    elif corr_val > 0.3:
        verdict = "Moderate correlation — partial diversification, size both conservatively"
    else:
        verdict = "Low correlation — real diversification benefit when trading both"

    lines += [
        "",
        "-" * 65,
        "PART 4 — CORRELATION ANALYSIS",
        "",
        f"  Weeks both gave LONG signal : {n}",
        f"  Both won  : {corr_stats.get('both_win', 0)}  ({corr_stats.get('pct_both_win', 0)}%)",
        f"  Both lost : {corr_stats.get('both_loss', 0)}  ({corr_stats.get('pct_both_loss', 0)}%)",
        f"  Diverged  : {corr_stats.get('diverged', 0)}  ({corr_stats.get('pct_diverged', 0)}%)",
        f"  R correlation: {corr_stats.get('r_correlation', 'N/A')}",
        "",
        f"  Verdict: {verdict}",
    ]

    # Recommendation
    btc_best  = btc_grid.sort_values("expectancy_R", ascending=False).iloc[0]
    eth_best  = eth_grid.sort_values("expectancy_R", ascending=False).iloc[0]
    trade_both = "No — correlation too high; size one at a time" if corr_val > 0.6 else "Yes — with reduced per-trade risk (e.g. 5% each instead of 10%)"

    v1_time_pct = btc_baseline_m.get("pct_time")
    best_time_pct = btc_best["pct_time"]
    time_delta = (
        f"{v1_time_pct}% → {best_time_pct}% TIME exits"
        if v1_time_pct is not None else "N/A"
    )

    v1_dd  = btc_baseline_m.get("max_drawdown", "N/A")
    best_dd = btc_best["max_drawdown"]

    lines += [
        "",
        "-" * 65,
        "PART 5 — RECOMMENDATION",
        "",
        f"  Recommended BTC parameters : stop={btc_best['stop_mult']}x  target={btc_best['r_target']}R",
        f"  Recommended ETH parameters : stop={eth_best['stop_mult']}x  target={eth_best['r_target']}R",
        f"  Trade both instruments?    : {trade_both}",
        "",
        f"  Reasoning:",
        f"    - Best BTC combo has expectancy {btc_best['expectancy_R']}R "
        f"vs V1 baseline {btc_exp}R",
        f"    - Best ETH combo has expectancy {eth_best['expectancy_R']}R "
        f"vs V1 baseline {eth_exp}R",
        f"    - BTC at baseline: {btc_baseline_m.get('pct_time', 'N/A')}% TIME exits "
        f"→ target may be reachable at lower R with tighter stop",
        "",
        f"  Key change from V1 baseline       : stop {btc_baseline_m.get('stop_mult', 1.0) if 'stop_mult' in btc_baseline_m else 1.0}x→{btc_best['stop_mult']}x  "
        f"target 2.0R→{btc_best['r_target']}R",
        f"  Expected TIME exit rate change    : {time_delta}",
        f"  Expected max drawdown change      : {v1_dd}% → {best_dd}%",
        "=" * 65,
    ]

    report_path = out_dir / "param_sweep_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\n  Sweep report    → {report_path}")


def write_eth_summary(
    eth_baseline_m: dict,
    eth_regime_rows: list[dict],
    eth_baseline_sim: pd.DataFrame,
    out_dir: Path,
) -> None:
    n_total    = len(eth_baseline_sim)
    n_long     = len(eth_baseline_sim[eth_baseline_sim["direction"] == "LONG"])
    n_no_trade = n_total - n_long

    lines = [
        "=" * 65,
        "ETH STANDALONE BACKTEST — MA20 Weekend System",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Parameters: stop=1.0×ATR14  target=2.0R  slippage=0.1%",
        "=" * 65,
        "",
        "SIGNAL OVERVIEW",
        "-" * 40,
        f"  Total Fridays analyzed:      {n_total}",
        f"  LONG signals (above MA20):   {n_long}  ({n_long/n_total*100:.1f}%)",
        f"  NO_TRADE (below MA20):       {n_no_trade}  ({n_no_trade/n_total*100:.1f}%)",
        "",
        "OVERALL PERFORMANCE",
        "-" * 40,
    ]

    m = eth_baseline_m
    if m and m.get("n_trades", 0) > 0:
        lines += [
            f"  Trades completed:      {m['n_trades']}",
            f"  Win Rate:              {m['win_rate']}%",
            f"  Average R:             {m['avg_R']}",
            f"  Avg Win R:             {m.get('avg_win_R', 'N/A')}",
            f"  Avg Loss R:            {m.get('avg_loss_R', 'N/A')}",
            f"  Profit Factor:         {m['profit_factor']}",
            f"  Expectancy (R):        {m['expectancy_R']}",
            f"  Max Drawdown:          {m['max_drawdown']}%",
            f"  ROI Pre-Tax:           {m['roi_pre_tax']}%",
            "",
            f"  Exit breakdown:",
            f"    Hit target:          {m['exit_TARGET']}  ({m['pct_target']}%)",
            f"    Stopped out:         {m['exit_STOP']}  ({m['pct_stop']}%)",
            f"    Timed out:           {m['exit_TIME']}  ({m['pct_time']}%)",
        ]
    else:
        lines.append("  No completed trades.")

    lines += ["", "REGIME BREAKDOWN", "-" * 40]
    for r in eth_regime_rows:
        regime = r.get("regime", "?")
        if r.get("n_trades", 0) > 0:
            lines.append(
                f"  {regime:<22}  "
                f"Trades:{r['n_trades']:>3}  "
                f"Win%:{r['win_rate']:>5}%  "
                f"AvgR:{r['avg_R']:>6}  "
                f"PF:{r['profit_factor']:>5}  "
                f"MaxDD:{r['max_drawdown']:>6}%"
            )
        else:
            lines.append(f"  {regime:<22}  No trades in range")

    lines += ["", "=" * 65]

    summary_path = out_dir / "eth_standalone_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"  ETH summary     → {summary_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PARAMETER SWEEP — BTC + ETH Weekend MA20 System")
    print("=" * 65)

    # ── Load data once ────────────────────────────────────────────────────
    print("\nLoading data...")
    data = load_crypto_data(["BTC-USD", "ETH-USD"])
    btc  = get_ticker_df(data, "BTC-USD")
    eth  = get_ticker_df(data, "ETH-USD")

    # ── Part 1: BTC grid ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 1 — BTC PARAMETER GRID")
    print("=" * 65)
    btc_grid, btc_baseline_sim = run_grid(btc, "BTC")

    btc_sweep_path = OUTPUT_DIR / "param_sweep_btc.csv"
    cols_out = ["stop_mult","r_target","n_trades","win_rate","avg_R",
                "profit_factor","expectancy_R","max_drawdown",
                "pct_target","pct_stop","pct_time","roi_pre_tax"]
    btc_grid[cols_out].to_csv(btc_sweep_path, index=False)
    print(f"\n  BTC grid saved  → {btc_sweep_path}")

    btc_baseline_m = _metrics_from_sim(btc_baseline_sim, 1.0, 2.0)

    # ── Part 2: ETH grid ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 2 — ETH PARAMETER GRID")
    print("=" * 65)
    eth_grid, eth_baseline_sim = run_grid(eth, "ETH")

    eth_sweep_path = OUTPUT_DIR / "param_sweep_eth.csv"
    eth_grid[cols_out].to_csv(eth_sweep_path, index=False)
    print(f"\n  ETH grid saved  → {eth_sweep_path}")

    eth_baseline_m = _metrics_from_sim(eth_baseline_sim, 1.0, 2.0)

    # ETH regime breakdown
    eth_regime_rows = _regime_breakdown(eth_baseline_sim)
    eth_regime_df   = pd.DataFrame(eth_regime_rows)
    eth_regime_path = OUTPUT_DIR / "eth_regime_breakdown.csv"
    eth_regime_df.to_csv(eth_regime_path, index=False)
    print(f"  ETH regime csv  → {eth_regime_path}")

    write_eth_summary(eth_baseline_m, eth_regime_rows, eth_baseline_sim, OUTPUT_DIR)

    # ── Part 3: Correlation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 3 — CORRELATION ANALYSIS")
    print("=" * 65)
    corr_df, corr_stats = correlation_analysis(btc_baseline_sim, eth_baseline_sim)
    print(f"  Shared LONG weeks : {corr_stats.get('n_shared', 0)}")
    print(f"  Both won          : {corr_stats.get('both_win', 0)}  ({corr_stats.get('pct_both_win', 0)}%)")
    print(f"  Both lost         : {corr_stats.get('both_loss', 0)}  ({corr_stats.get('pct_both_loss', 0)}%)")
    print(f"  Diverged          : {corr_stats.get('diverged', 0)}  ({corr_stats.get('pct_diverged', 0)}%)")
    print(f"  R correlation     : {corr_stats.get('r_correlation', 'N/A')}")

    if not corr_df.empty:
        corr_path = OUTPUT_DIR / "correlation_analysis.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"  Correlation csv   → {corr_path}")

    # ── Final report ──────────────────────────────────────────────────────
    print("\nWriting sweep report...")
    write_report(btc_grid, eth_grid, btc_baseline_m, eth_baseline_m, corr_stats, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done. Review param_sweep_report.txt for recommendations.")


if __name__ == "__main__":
    main()
