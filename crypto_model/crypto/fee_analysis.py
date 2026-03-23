"""
crypto/fee_analysis.py
=======================
Exchange-specific fee modeling on the full historical backtest.

Runs the same simulation logic as parameter_sweep.py (copied, not imported)
at the locked live parameters, then applies two exchange fee structures.

Exchanges modeled
  Kraken Pro      : 0.26% taker on both legs (entry + exit)
  Coinbase Advanced: 0.60% taker on both legs (entry + exit)

Entry uses stop-limit → taker fee.
Exit uses stop-loss or limit take-profit → taker fee.
Fee per trade = (entry_price × units + exit_price × units) × taker_rate

Live parameters (locked — do not change)
  ATR_MULT_STOP    = 1.25
  R_TARGET         = 2.0
  RISK_PCT         = 0.05   (5% per instrument)
  STARTING_CAPITAL = 500.0

Instruments: BTC-USD and ETH-USD independently, plus combined portfolio view.

Outputs → Results/crypto_backtest/
  fee_analysis_trades_btc.csv    — per-trade fee breakdown for BTC
  fee_analysis_trades_eth.csv    — per-trade fee breakdown for ETH
  fee_analysis_summary.txt       — full side-by-side report

Do NOT modify: backtest_engine.py, data_loader.py, main.py, parameter_sweep.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays

# ── Locked live parameters ──────────────────────────────────────────────────
ATR_MULT_STOP    = 1.25
R_TARGET         = 2.0
RISK_PCT         = 0.05
STARTING_CAPITAL = 500.0
SLIPPAGE_PCT     = 0.001

# ── Exchange fee structures (taker rate, applied to both legs) ───────────────
EXCHANGES = {
    "Kraken Pro":          0.0026,   # 0.26%
    "Coinbase Advanced":   0.0060,   # 0.60%
}

OUTPUT_DIR = Path(__file__).parent.parent / "Results" / "crypto_backtest"


# ── Simulation loop (self-contained — does not import from other files) ──────

def _run_sim(
    ticker_df: pd.DataFrame,
    stop_mult: float = ATR_MULT_STOP,
    r_target:  float = R_TARGET,
    risk_pct:  float = RISK_PCT,
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    """
    Walk-forward weekend simulation.
    Returns one row per Friday with all data needed for fee calculations.
    Columns: date, instrument, direction, exit_type,
             entry_price, stop_price, target_price, exit_price,
             units, gross_pnl, gross_R, entry_value, exit_value
    """
    fridays = get_fridays(ticker_df)
    rows = []
    equity = starting_capital

    for fri_date, fri_row in fridays.iterrows():
        above_ma = bool(fri_row["above_ma20"])

        if not above_ma:
            rows.append({
                "date": fri_date, "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                "entry_price": None, "stop_price": None, "target_price": None,
                "exit_price": None, "units": None,
                "gross_pnl": 0.0, "gross_R": None,
                "entry_value": None, "exit_value": None,
            })
            continue

        entry         = fri_row["close"] * (1 + SLIPPAGE_PCT)
        atr           = fri_row["atr14"]
        stop          = entry - stop_mult * atr
        target        = entry + r_target * (entry - stop)
        risk_per_unit = entry - stop
        units         = (equity * risk_pct) / risk_per_unit

        fri_pos   = ticker_df.index.get_loc(fri_date)
        next_bars = ticker_df.iloc[fri_pos + 1 : fri_pos + 3]
        weekend   = next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()

        if weekend.empty:
            rows.append({
                "date": fri_date, "direction": "LONG", "exit_type": "NO_DATA",
                "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                "target_price": round(target, 4), "exit_price": None,
                "units": round(units, 8),
                "gross_pnl": 0.0, "gross_R": None,
                "entry_value": round(entry * units, 4), "exit_value": None,
            })
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

        gross_pnl   = units * (exit_price - entry)
        gross_R     = (exit_price - entry) / risk_per_unit
        entry_value = entry * units
        exit_value  = exit_price * units

        equity = max(equity + gross_pnl, 0.01)

        rows.append({
            "date": fri_date, "direction": "LONG", "exit_type": exit_type,
            "entry_price":  round(entry, 4),
            "stop_price":   round(stop, 4),
            "target_price": round(target, 4),
            "exit_price":   round(exit_price, 4),
            "units":        round(units, 8),
            "gross_pnl":    round(gross_pnl, 6),
            "gross_R":      round(gross_R, 4),
            "entry_value":  round(entry_value, 4),
            "exit_value":   round(exit_value, 4),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Apply fees ───────────────────────────────────────────────────────────────

def apply_fees(sim: pd.DataFrame, taker_rate: float) -> pd.DataFrame:
    """
    Add fee columns to a simulation DataFrame.
    Only LONG trades with completed exits get fees applied.
    NO_TRADE and NO_DATA rows carry zero fees.
    """
    df = sim.copy()
    tradeable = (df["direction"] == "LONG") & (df["exit_type"] != "NO_DATA") & df["entry_value"].notna() & df["exit_value"].notna()

    df["fee"] = 0.0
    df.loc[tradeable, "fee"] = (
        (df.loc[tradeable, "entry_value"] + df.loc[tradeable, "exit_value"]) * taker_rate
    ).round(6)

    df["net_pnl"] = df["gross_pnl"] - df["fee"]

    # Net R: net_pnl / risk_dollars, where risk_dollars = units × (entry - stop)
    df["net_R"] = None
    mask = tradeable & df["units"].notna() & df["entry_price"].notna() & df["stop_price"].notna()
    risk_dollars = df.loc[mask, "units"] * (df.loc[mask, "entry_price"] - df.loc[mask, "stop_price"])
    df.loc[mask, "net_R"] = (df.loc[mask, "net_pnl"] / risk_dollars).round(4)

    # Flag trades that flipped gross-profit → net-loss after fees
    df["flipped"] = (df["gross_pnl"] > 0) & (df["net_pnl"] <= 0)

    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(df: pd.DataFrame, label: str, taker_rate: float) -> dict:
    """
    Compute gross and net-of-fee performance metrics for completed LONG trades.
    """
    completed = df[(df["direction"] == "LONG") & (df["exit_type"] != "NO_DATA")].dropna(subset=["gross_R"])

    if completed.empty:
        return {"label": label, "taker_rate": taker_rate, "n_trades": 0}

    n = len(completed)

    # ── Gross ────────────────────────────────────────────────────────────────
    gross_wins   = completed[completed["gross_pnl"] > 0]
    gross_losses = completed[completed["gross_pnl"] <= 0]
    gross_wr     = len(gross_wins) / n
    gross_avg_R  = completed["gross_R"].mean()
    gross_gw     = gross_wins["gross_R"].sum()
    gross_gl     = abs(gross_losses["gross_R"].sum())
    gross_pf     = gross_gw / gross_gl if gross_gl > 0 else np.inf
    gross_aw     = gross_wins["gross_R"].mean()   if not gross_wins.empty   else 0.0
    gross_al     = gross_losses["gross_R"].mean() if not gross_losses.empty else 0.0
    gross_exp    = gross_wr * gross_aw + (1 - gross_wr) * gross_al

    # ── Net-of-fee ────────────────────────────────────────────────────────────
    net_wins   = completed[completed["net_pnl"] > 0]
    net_losses = completed[completed["net_pnl"] <= 0]
    net_wr     = len(net_wins) / n
    net_avg_R  = completed["net_R"].mean()
    net_gw     = net_wins["net_R"].sum()
    net_gl     = abs(net_losses["net_R"].sum())
    net_pf     = net_gw / net_gl if net_gl > 0 else np.inf
    net_aw     = net_wins["net_R"].mean()   if not net_wins.empty   else 0.0
    net_al     = net_losses["net_R"].mean() if not net_losses.empty else 0.0
    net_exp    = net_wr * net_aw + (1 - net_wr) * net_al

    # ── Fee totals ────────────────────────────────────────────────────────────
    total_fees     = completed["fee"].sum()
    total_gross_pnl= completed["gross_pnl"].sum()
    total_net_pnl  = completed["net_pnl"].sum()
    flipped        = int(completed["flipped"].sum())

    # ── Break-even fee rate ───────────────────────────────────────────────────
    # fee_be × Σ(entry_val + exit_val) = Σ(gross_pnl)
    # Only meaningful when gross P&L is positive
    total_notional = (completed["entry_value"] + completed["exit_value"]).sum()
    be_rate = (total_gross_pnl / total_notional) if total_notional > 0 and total_gross_pnl > 0 else 0.0

    return {
        "label":          label,
        "taker_rate":     taker_rate,
        "n_trades":       n,
        "n_no_trade":     int((df["direction"] == "NO_TRADE").sum()),
        # Gross
        "gross_wr":       round(gross_wr * 100, 1),
        "gross_avg_R":    round(gross_avg_R, 3),
        "gross_pf":       round(min(gross_pf, 999.0), 2),
        "gross_exp":      round(gross_exp, 3),
        "gross_total_pnl":round(total_gross_pnl, 2),
        # Net
        "net_wr":         round(net_wr * 100, 1),
        "net_avg_R":      round(net_avg_R, 3) if net_avg_R is not None else None,
        "net_pf":         round(min(net_pf, 999.0), 2),
        "net_exp":        round(net_exp, 3),
        "net_total_pnl":  round(total_net_pnl, 2),
        # Fees
        "total_fees":     round(total_fees, 2),
        "fee_pct_of_gross": round(total_fees / abs(total_gross_pnl) * 100, 1) if total_gross_pnl != 0 else None,
        "flipped_trades": flipped,
        "flipped_pct":    round(flipped / n * 100, 1),
        # Break-even
        "breakeven_rate": round(be_rate * 100, 4),   # as percent per leg
        "breakeven_roundtrip": round(be_rate * 2 * 100, 4),  # round-trip %
        "fee_headroom_kraken":  round((be_rate - 0.0026) * 100, 4),
        "fee_headroom_coinbase": round((be_rate - 0.0060) * 100, 4),
    }


# ── Combined portfolio (weeks both instruments gave LONG signals) ────────────

def combined_portfolio(
    btc_df_fees: pd.DataFrame,
    eth_df_fees: pd.DataFrame,
    label: str,
) -> dict:
    btc_long = btc_df_fees[
        (btc_df_fees["direction"] == "LONG") &
        (btc_df_fees["exit_type"] != "NO_DATA")
    ].dropna(subset=["net_pnl"])[["date", "gross_pnl", "net_pnl", "fee"]].copy()

    eth_long = eth_df_fees[
        (eth_df_fees["direction"] == "LONG") &
        (eth_df_fees["exit_type"] != "NO_DATA")
    ].dropna(subset=["net_pnl"])[["date", "gross_pnl", "net_pnl", "fee"]].copy()

    merged = btc_long.merge(eth_long, on="date", suffixes=("_btc", "_eth"))
    if merged.empty:
        return {"label": label, "n_shared": 0}

    merged["combined_gross"] = merged["gross_pnl_btc"] + merged["gross_pnl_eth"]
    merged["combined_net"]   = merged["net_pnl_btc"]   + merged["net_pnl_eth"]
    merged["combined_fees"]  = merged["fee_btc"]        + merged["fee_eth"]
    merged["both_win"]       = (merged["net_pnl_btc"] > 0) & (merged["net_pnl_eth"] > 0)
    merged["both_loss"]      = (merged["net_pnl_btc"] <= 0) & (merged["net_pnl_eth"] <= 0)
    merged["diverged"]       = ~(merged["both_win"] | merged["both_loss"])

    n = len(merged)
    return {
        "label":            label,
        "n_shared":         n,
        "both_win":         int(merged["both_win"].sum()),
        "pct_both_win":     round(merged["both_win"].sum() / n * 100, 1),
        "both_loss":        int(merged["both_loss"].sum()),
        "pct_both_loss":    round(merged["both_loss"].sum() / n * 100, 1),
        "diverged":         int(merged["diverged"].sum()),
        "pct_diverged":     round(merged["diverged"].sum() / n * 100, 1),
        "total_gross_pnl":  round(merged["combined_gross"].sum(), 2),
        "total_net_pnl":    round(merged["combined_net"].sum(), 2),
        "total_fees":       round(merged["combined_fees"].sum(), 2),
        "avg_net_pnl_week": round(merged["combined_net"].mean(), 2),
    }


# ── Build trades CSV with fee columns ────────────────────────────────────────

def build_trades_csv(
    sim: pd.DataFrame,
    ticker: str,
    kraken_rate: float,
    coinbase_rate: float,
) -> pd.DataFrame:
    """
    Produce a single per-trade DataFrame with both exchange fee columns.
    """
    k = apply_fees(sim, kraken_rate)
    c = apply_fees(sim, coinbase_rate)

    out = sim[["date", "direction", "exit_type",
               "entry_price", "stop_price", "target_price", "exit_price",
               "units", "gross_pnl", "gross_R"]].copy()
    out["instrument"] = ticker

    out["kraken_fee"]    = k["fee"]
    out["kraken_net_pnl"]= k["net_pnl"]
    out["kraken_net_R"]  = k["net_R"]
    out["kraken_flipped"]= k["flipped"]

    out["coinbase_fee"]    = c["fee"]
    out["coinbase_net_pnl"]= c["net_pnl"]
    out["coinbase_net_R"]  = c["net_R"]
    out["coinbase_flipped"]= c["flipped"]

    # Reorder
    cols = ["date", "instrument", "direction", "exit_type",
            "entry_price", "stop_price", "target_price", "exit_price",
            "units", "gross_pnl", "gross_R",
            "kraken_fee", "kraken_net_pnl", "kraken_net_R", "kraken_flipped",
            "coinbase_fee", "coinbase_net_pnl", "coinbase_net_R", "coinbase_flipped"]
    return out[[c for c in cols if c in out.columns]]


# ── Report writer ─────────────────────────────────────────────────────────────

def _metric_block(m: dict) -> list[str]:
    if m.get("n_trades", 0) == 0:
        return [f"  {m['label']}: No completed trades."]

    lines = [
        f"  Exchange          : {m['label']}  (taker={m['taker_rate']*100:.2f}%)",
        f"  Trades completed  : {m['n_trades']}  |  NO_TRADE weeks: {m.get('n_no_trade',0)}",
        "",
        f"  {'Metric':<26} {'GROSS':>10}  {'NET-OF-FEE':>10}",
        f"  {'-'*50}",
        f"  {'Win Rate':<26} {m['gross_wr']:>9.1f}%  {m['net_wr']:>9.1f}%",
        f"  {'Avg R':<26} {m['gross_avg_R']:>10.3f}  {m['net_avg_R'] if m['net_avg_R'] is not None else float('nan'):>10.3f}",
        f"  {'Profit Factor':<26} {m['gross_pf']:>10.2f}  {m['net_pf']:>10.2f}",
        f"  {'Expectancy R':<26} {m['gross_exp']:>10.3f}  {m['net_exp']:>10.3f}",
        f"  {'Total P&L ($)':<26} {m['gross_total_pnl']:>10.2f}  {m['net_total_pnl']:>10.2f}",
        "",
        f"  Total fees paid   : ${m['total_fees']:.2f}",
        f"  Fee drag          : {m.get('fee_pct_of_gross','N/A')}% of gross P&L",
        f"  Trades flipped    : {m['flipped_trades']} / {m['n_trades']}  ({m['flipped_pct']}%)  [gross win → net loss after fees]",
    ]
    return lines


def _breakeven_block(btc_m: dict, eth_m: dict, exch: str) -> list[str]:
    def row(ticker, m):
        be     = m.get("breakeven_rate", 0)
        be_rt  = m.get("breakeven_roundtrip", 0)
        rate   = m["taker_rate"] * 100
        margin = be - rate
        status = "POSITIVE HEADROOM" if margin > 0 else "EDGE CONSUMED"
        color  = "+" if margin > 0 else ""
        return [
            f"  {ticker:<8}  break-even taker rate : {be:.4f}% / leg  "
            f"({be_rt:.4f}% round-trip)",
            f"          actual {exch} rate  : {rate:.2f}% / leg",
            f"          fee headroom       : {color}{margin:.4f}% / leg  →  {status}",
        ]

    lines = [f"\n  ── {exch} ──"]
    lines += row("BTC", btc_m)
    lines += [""]
    lines += row("ETH", eth_m)
    return lines


def _portfolio_block(port: dict) -> list[str]:
    if port.get("n_shared", 0) == 0:
        return [f"  {port['label']}: No shared LONG weeks."]
    return [
        f"  {port['label']}",
        f"  Shared LONG weeks   : {port['n_shared']}",
        f"  Both won (net)      : {port['both_win']}  ({port['pct_both_win']}%)",
        f"  Both lost (net)     : {port['both_loss']}  ({port['pct_both_loss']}%)",
        f"  Diverged            : {port['diverged']}  ({port['pct_diverged']}%)",
        f"  Total gross P&L     : ${port['total_gross_pnl']:.2f}",
        f"  Total fees (2×leg)  : ${port['total_fees']:.2f}",
        f"  Total net P&L       : ${port['total_net_pnl']:.2f}",
        f"  Avg net P&L / week  : ${port['avg_net_pnl_week']:.2f}",
    ]


def write_summary(
    btc_metrics: dict,  # exchange_name → metrics dict
    eth_metrics: dict,
    portfolios:  dict,  # exchange_name → combined dict
    out_dir: Path,
) -> None:
    lines = [
        "=" * 65,
        "FEE IMPACT ANALYSIS — BTC + ETH Weekend MA20 System",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Parameters: stop={ATR_MULT_STOP}×ATR  target={R_TARGET}R  "
        f"risk={RISK_PCT*100:.0f}%/instrument  capital=${STARTING_CAPITAL}",
        "=" * 65,
        "",
        "PART 1 — BTC-USD  (per-exchange, gross vs net-of-fee)",
        "=" * 65,
    ]

    for exch in EXCHANGES:
        lines += [""]
        lines += _metric_block(btc_metrics[exch])
        lines += ["-" * 65]

    lines += [
        "",
        "PART 2 — ETH-USD  (per-exchange, gross vs net-of-fee)",
        "=" * 65,
    ]
    for exch in EXCHANGES:
        lines += [""]
        lines += _metric_block(eth_metrics[exch])
        lines += ["-" * 65]

    lines += [
        "",
        "PART 3 — SIDE-BY-SIDE COMPARISON AT BASELINE PARAMS",
        "=" * 65,
        "",
        f"  {'':30} {'Kraken Pro':>14}  {'Coinbase Adv':>14}",
        f"  {'':30} {'(0.26% taker)':>14}  {'(0.60% taker)':>14}",
        f"  {'-'*62}",
    ]

    def side(ticker, metrics):
        k = metrics["Kraken Pro"]
        c = metrics["Coinbase Advanced"]
        lines = []
        for label, kv, cv in [
            ("Total fees paid ($)",    f"${k['total_fees']:.2f}",    f"${c['total_fees']:.2f}"),
            ("Net expectancy R",       f"{k['net_exp']:.3f}R",       f"{c['net_exp']:.3f}R"),
            ("Net profit factor",      f"{k['net_pf']:.2f}",         f"{c['net_pf']:.2f}"),
            ("Net total P&L ($)",      f"${k['net_total_pnl']:.2f}", f"${c['net_total_pnl']:.2f}"),
            ("Trades flipped",         f"{k['flipped_trades']}",     f"{c['flipped_trades']}"),
        ]:
            lines.append(f"  {ticker+' — '+label:<30} {kv:>14}  {cv:>14}")
        return lines

    lines += side("BTC", btc_metrics)
    lines += [""]
    lines += side("ETH", eth_metrics)

    lines += [
        "",
        "PART 4 — BREAK-EVEN FEE RATE",
        "=" * 65,
        "  Break-even = the taker fee rate per leg that would zero out",
        "  all gross profit over the full backtest period.",
        "  If actual fee > break-even: edge is fully consumed by fees.",
        "",
    ]
    for exch, rate in EXCHANGES.items():
        lines += _breakeven_block(btc_metrics[exch], eth_metrics[exch], exch)

    lines += [
        "",
        "PART 5 — COMBINED WEEKEND PORTFOLIO (both instruments LONG same week)",
        "=" * 65,
    ]
    for exch in EXCHANGES:
        lines += [""]
        lines += _portfolio_block(portfolios[exch])

    lines += [
        "",
        "PART 6 — VERDICT",
        "=" * 65,
        "",
    ]

    # Auto-generate verdict
    for exch, rate in EXCHANGES.items():
        btc_be   = btc_metrics[exch].get("breakeven_rate", 0)
        eth_be   = eth_metrics[exch].get("breakeven_rate", 0)
        btc_net  = btc_metrics[exch].get("net_exp", 0)
        eth_net  = eth_metrics[exch].get("net_exp", 0)
        rate_pct = rate * 100

        btc_ok = btc_be > rate_pct
        eth_ok = eth_be > rate_pct
        both_ok = btc_ok and eth_ok

        lines.append(f"  {exch}  ({rate_pct:.2f}% taker)")
        lines.append(f"    BTC net expectancy : {btc_net:+.3f}R  →  "
                     + ("EDGE SURVIVES FEES" if btc_net > 0 else "EDGE CONSUMED"))
        lines.append(f"    ETH net expectancy : {eth_net:+.3f}R  →  "
                     + ("EDGE SURVIVES FEES" if eth_net > 0 else "EDGE CONSUMED"))
        lines.append(f"    Verdict: {'VIABLE on this exchange' if both_ok else 'MARGINAL — review before live deployment'}")
        lines.append("")

    lines += ["=" * 65]

    out_path = out_dir / "fee_analysis_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"  Summary → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Fee Analysis — BTC + ETH Weekend MA20 System")
    print("=" * 65)

    print("\nLoading historical data (yfinance)...")
    data = load_crypto_data(["BTC-USD", "ETH-USD"])
    btc  = get_ticker_df(data, "BTC-USD")
    eth  = get_ticker_df(data, "ETH-USD")

    # ── Run gross simulations once each ──────────────────────────────────────
    print("\nRunning gross simulation — BTC...")
    btc_sim = _run_sim(btc)
    print(f"  {len(btc_sim[btc_sim['direction']=='LONG'])} LONG signals")

    print("Running gross simulation — ETH...")
    eth_sim = _run_sim(eth)
    print(f"  {len(eth_sim[eth_sim['direction']=='LONG'])} LONG signals")

    # ── Apply fees and compute metrics ────────────────────────────────────────
    btc_metrics: dict[str, dict] = {}
    eth_metrics: dict[str, dict] = {}
    portfolios:  dict[str, dict] = {}

    print("\nApplying fee structures...")
    for exch, rate in EXCHANGES.items():
        print(f"  {exch}  ({rate*100:.2f}% taker)")

        btc_fee = apply_fees(btc_sim, rate)
        eth_fee = apply_fees(eth_sim, rate)

        btc_metrics[exch] = calc_metrics(btc_fee, exch, rate)
        eth_metrics[exch] = calc_metrics(eth_fee, exch, rate)
        portfolios[exch]  = combined_portfolio(btc_fee, eth_fee, exch)

    # ── Print break-even table to terminal ───────────────────────────────────
    print()
    print("=" * 65)
    print("BREAK-EVEN FEE RATES")
    print("=" * 65)
    for exch, rate in EXCHANGES.items():
        btc_be = btc_metrics[exch].get("breakeven_rate", 0)
        eth_be = eth_metrics[exch].get("breakeven_rate", 0)
        print(f"\n  {exch}  (actual: {rate*100:.2f}%/leg)")
        print(f"    BTC break-even: {btc_be:.4f}%/leg  "
              f"headroom: {btc_be - rate*100:+.4f}%  "
              f"{'OK' if btc_be > rate*100 else '*** EDGE GONE'}")
        print(f"    ETH break-even: {eth_be:.4f}%/leg  "
              f"headroom: {eth_be - rate*100:+.4f}%  "
              f"{'OK' if eth_be > rate*100 else '*** EDGE GONE'}")

    # ── Save trades CSVs ─────────────────────────────────────────────────────
    print("\nSaving outputs...")
    btc_csv = build_trades_csv(btc_sim, "BTC-USD",
                               EXCHANGES["Kraken Pro"],
                               EXCHANGES["Coinbase Advanced"])
    eth_csv = build_trades_csv(eth_sim, "ETH-USD",
                               EXCHANGES["Kraken Pro"],
                               EXCHANGES["Coinbase Advanced"])

    btc_path = OUTPUT_DIR / "fee_analysis_trades_btc.csv"
    eth_path = OUTPUT_DIR / "fee_analysis_trades_eth.csv"
    btc_csv.to_csv(btc_path, index=False)
    eth_csv.to_csv(eth_path, index=False)
    print(f"  BTC trades → {btc_path}")
    print(f"  ETH trades → {eth_path}")

    write_summary(btc_metrics, eth_metrics, portfolios, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
