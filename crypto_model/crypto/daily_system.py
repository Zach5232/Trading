"""
daily_system.py
---------------
Tests the MA20 momentum system running on every daily close (not just Fridays),
with multiple hold periods and a day-of-week analysis.
Also produces the master full_system_comparison.txt.
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATR_MULT_STOP    = 1.25
R_TARGET         = 2.0
RISK_PCT         = 0.05
STARTING_CAPITAL = 500.0
SLIPPAGE_PCT     = 0.001
KRAKEN_TAKER     = 0.0026   # 0.26%

CACHE_PATH = Path(__file__).parent.parent / "Data" / "backtest_cache" / "ohlcv_5yr.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "Results" / "crypto_backtest"

HOLD_PERIODS = [1, 2, 3, 5]   # bars (daily candles) to hold

REGIMES = {
    "2018 Bear":         ("2018-01-01", "2018-12-31"),
    "2019 Recovery":     ("2019-01-01", "2019-12-31"),
    "2020 COVID":        ("2020-01-01", "2020-12-31"),
    "2021 Bull":         ("2021-01-01", "2021-12-31"),
    "2022 Bear":         ("2022-01-01", "2022-12-31"),
    "2023-24 Recovery":  ("2023-01-01", "2024-12-31"),
    "2025-Present":      ("2025-01-01", "2099-12-31"),
}

V1_BASELINE = {
    "BTC-USD": {"gross_avg_R": 0.169, "net_avg_R": 0.029, "net_pf": 1.12,
                "trades_per_year": 60},
    "ETH-USD": {"gross_avg_R": 0.182, "net_avg_R": 0.090, "net_pf": 1.42,
                "trades_per_year": 40},
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict[str, pd.DataFrame]:
    if CACHE_PATH.exists():
        raw = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    else:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = load_crypto_data(["BTC-USD", "ETH-USD"])
        raw.to_csv(CACHE_PATH, index=False)
    return {
        "BTC-USD": get_ticker_df(raw, "BTC-USD"),
        "ETH-USD": get_ticker_df(raw, "ETH-USD"),
    }


def prep_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr5_ago"] = df["atr14"].shift(5)
    df["daily_atr_expanding"] = df["atr14"] > df["atr5_ago"]
    return df.dropna(subset=["atr5_ago"])


# ---------------------------------------------------------------------------
# Exit simulation
# ---------------------------------------------------------------------------

def _simulate_daily_exit(ticker_df, sig_pos, entry, stop, target, hold_bars):
    """
    Walk the next `hold_bars` bars checking stop/target.
    Force exit at last bar's close if neither triggers.
    Returns (exit_price, exit_type) or (None, "NO_DATA") if insufficient bars.
    """
    bars = ticker_df.iloc[sig_pos + 1 : sig_pos + 1 + hold_bars]
    if len(bars) < hold_bars:
        return None, "NO_DATA"
    for _, bar in bars.iterrows():
        if bar["low"] <= stop:
            return stop, "STOP"
        if bar["high"] >= target:
            return target, "TARGET"
    return bars.iloc[-1]["close"], "TIME"


# ---------------------------------------------------------------------------
# Fee helper
# ---------------------------------------------------------------------------

def _fee(entry_px, exit_px, units):
    return (entry_px * units + exit_px * units) * KRAKEN_TAKER


# ---------------------------------------------------------------------------
# Metrics calculator
# ---------------------------------------------------------------------------

def _calc_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics from a list of trade dicts.
    Each dict must have: gross_pnl, gross_R, net_pnl, net_R, entry_value, exit_value, exit_type.
    Only completed (non-NO_DATA, non-NO_TRADE) trades are included.
    """
    df = pd.DataFrame(trades)
    comp = df[df["direction"] == "LONG"].dropna(subset=["gross_R"])
    if comp.empty:
        return {}
    n = len(comp)
    gross_wins   = comp[comp["gross_pnl"] > 0]
    gross_losses = comp[comp["gross_pnl"] <= 0]
    gwr = len(gross_wins) / n
    g_avg_R = comp["gross_R"].mean()
    gpf = (
        gross_wins["gross_R"].sum() / abs(gross_losses["gross_R"].sum())
        if not gross_losses.empty and gross_losses["gross_R"].sum() != 0
        else np.inf
    )
    g_exp = (
        gwr * (gross_wins["gross_R"].mean() if not gross_wins.empty else 0)
        + (1 - gwr) * (gross_losses["gross_R"].mean() if not gross_losses.empty else 0)
    )

    net_wins   = comp[comp["net_pnl"] > 0]
    net_losses = comp[comp["net_pnl"] <= 0]
    nwr = len(net_wins) / n
    n_avg_R = comp["net_R"].mean()
    npf = (
        net_wins["net_R"].sum() / abs(net_losses["net_R"].sum())
        if not net_losses.empty and net_losses["net_R"].sum() != 0
        else np.inf
    )
    n_exp = (
        nwr * (net_wins["net_R"].mean() if not net_wins.empty else 0)
        + (1 - nwr) * (net_losses["net_R"].mean() if not net_losses.empty else 0)
    )

    # equity curve for max drawdown
    equity = [STARTING_CAPITAL]
    for pl in comp["gross_pnl"]:
        equity.append(max(equity[-1] + pl, 0.01))
    equity_s = pd.Series(equity)
    max_dd = float(((equity_s - equity_s.cummax()) / equity_s.cummax()).min() * 100)

    ec = comp["exit_type"].value_counts().to_dict()

    # break-even fee rate
    total_gpnl = comp["gross_pnl"].sum()
    total_notional = (comp["entry_value"] + comp["exit_value"]).sum()
    be_rate = (total_gpnl / total_notional * 100) if total_notional > 0 and total_gpnl > 0 else 0.0

    # trades per year estimate (5yr data → ~1825 bars → divide by 5)
    tpy = round(n / 5.0, 0)

    return {
        "n_trades": n, "n_no_trade": len(df) - n,
        "gross_wr": round(gwr * 100, 1), "gross_avg_R": round(g_avg_R, 3),
        "gross_pf": round(min(gpf, 99.9), 2), "gross_exp": round(g_exp, 3),
        "net_wr": round(nwr * 100, 1), "net_avg_R": round(n_avg_R, 3),
        "net_pf": round(min(npf, 99.9), 2), "net_exp": round(n_exp, 3),
        "max_dd": round(max_dd, 1),
        "exit_TARGET": ec.get("TARGET", 0), "exit_STOP": ec.get("STOP", 0), "exit_TIME": ec.get("TIME", 0),
        "breakeven_rate": round(be_rate, 4),
        "trades_per_year": int(tpy),
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_daily(ticker_df: pd.DataFrame, ticker: str, hold_bars: int) -> list[dict]:
    """
    Run daily system for one ticker at one hold period.
    Signal: close > MA20 AND ATR14 > ATR14 five bars ago.
    Entry: signal bar close x (1 + 0.001).
    Exit: stop, target, or forced TIME after hold_bars bars.
    """
    rows = []
    equity = STARTING_CAPITAL

    for i in range(len(ticker_df)):
        row = ticker_df.iloc[i]
        sig_date = ticker_df.index[i]

        # Both filters must pass
        above_ma = bool(row["above_ma20"])
        atr_exp  = bool(row["daily_atr_expanding"]) if not pd.isna(row.get("daily_atr_expanding", float("nan"))) else False

        if not above_ma or not atr_exp:
            rows.append({
                "date": sig_date, "ticker": ticker, "hold_bars": hold_bars,
                "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                "day_of_week": sig_date.dayofweek,
                "gross_pnl": 0.0, "gross_R": None, "net_R": None, "net_pnl": 0.0,
                "entry_value": None, "exit_value": None,
            })
            continue

        entry  = row["close"] * (1 + SLIPPAGE_PCT)
        atr    = row["atr14"]
        stop   = entry - ATR_MULT_STOP * atr
        target = entry + R_TARGET * (entry - stop)
        rpu    = entry - stop
        units  = (equity * RISK_PCT) / rpu

        exit_px, exit_type = _simulate_daily_exit(ticker_df, i, entry, stop, target, hold_bars)

        if exit_px is None:  # NO_DATA
            rows.append({
                "date": sig_date, "ticker": ticker, "hold_bars": hold_bars,
                "direction": "LONG", "exit_type": "NO_DATA",
                "day_of_week": sig_date.dayofweek,
                "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                "target_price": round(target, 4), "exit_price": None,
                "units": round(units, 8), "gross_pnl": 0.0, "gross_R": None,
                "entry_value": round(entry * units, 4), "exit_value": None,
                "fee": None, "net_pnl": 0.0, "net_R": None,
            })
            continue

        gross_pnl = units * (exit_px - entry)
        gross_R   = (exit_px - entry) / rpu
        fee       = _fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / (units * rpu)
        equity    = max(equity + gross_pnl, 0.01)

        rows.append({
            "date": sig_date, "ticker": ticker, "hold_bars": hold_bars,
            "direction": "LONG", "exit_type": exit_type,
            "day_of_week": sig_date.dayofweek,
            "entry_price": round(entry, 4), "stop_price": round(stop, 4),
            "target_price": round(target, 4), "exit_price": round(exit_px, 4),
            "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
            "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
            "exit_value": round(exit_px * units, 4),
            "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4),
        })

    return rows


# ---------------------------------------------------------------------------
# Day-of-week breakdown
# ---------------------------------------------------------------------------

def _dow_breakdown(rows: list[dict]) -> dict[int, dict]:
    """Compute metrics per day of week (0=Mon..6=Sun) for completed LONG trades."""
    result = {}
    df = pd.DataFrame(rows)
    comp = df[(df["direction"] == "LONG") & df["gross_R"].notna()]
    for dow in range(7):
        sub = comp[comp["day_of_week"] == dow]
        if sub.empty:
            result[dow] = {"n": 0}
            continue
        net_wins   = sub[sub["net_pnl"] > 0]
        net_losses = sub[sub["net_pnl"] <= 0]
        if not net_losses.empty:
            n_exp = (
                len(net_wins) / len(sub) * (net_wins["net_R"].mean() if not net_wins.empty else 0)
                + (1 - len(net_wins) / len(sub)) * net_losses["net_R"].mean()
            )
        else:
            n_exp = (
                len(net_wins) / len(sub) * (net_wins["net_R"].mean() if not net_wins.empty else 0)
            )
        result[dow] = {
            "n": len(sub),
            "gross_wr": round(len(sub[sub["gross_pnl"] > 0]) / len(sub) * 100, 1),
            "gross_avg_R": round(sub["gross_R"].mean(), 3),
            "net_avg_R": round(sub["net_R"].mean(), 3),
            "net_exp": round(n_exp, 3),
        }
    return result


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------

def _regime_breakdown(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for regime_name, (r_start, r_end) in REGIMES.items():
        subset = [
            r for r in rows
            if r.get("date") is not None
            and pd.Timestamp(r["date"]) >= pd.Timestamp(r_start)
            and pd.Timestamp(r["date"]) <= pd.Timestamp(r_end)
        ]
        result[regime_name] = _calc_metrics(subset)
    return result


# ---------------------------------------------------------------------------
# Load variation metrics from prior CSVs
# ---------------------------------------------------------------------------

def _load_variation_metrics(csv_path: Path, ticker: str) -> dict[str, dict]:
    """Load per-variation metrics from a trade CSV. Returns {variation: metrics_dict}."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df = df[df["ticker"] == ticker]
    result = {}
    for var, grp in df.groupby("variation"):
        comp = grp[(grp["direction"].isin(["LONG", "SHORT"])) & grp["gross_R"].notna()]
        if comp.empty:
            continue
        n = len(comp)
        gross_wins   = comp[comp["gross_pnl"] > 0]
        gross_losses = comp[comp["gross_pnl"] <= 0]
        gwr = len(gross_wins) / n
        g_avg_R = comp["gross_R"].mean()
        gpf = (
            gross_wins["gross_R"].sum() / abs(gross_losses["gross_R"].sum())
            if not gross_losses.empty and gross_losses["gross_R"].sum() != 0
            else np.inf
        )
        net_wins   = comp[comp["net_pnl"] > 0]
        net_losses = comp[comp["net_pnl"] <= 0]
        nwr = len(net_wins) / n
        n_avg_R = comp["net_R"].mean()
        npf = (
            net_wins["net_R"].sum() / abs(net_losses["net_R"].sum())
            if not net_losses.empty and net_losses["net_R"].sum() != 0
            else np.inf
        )
        n_exp = (
            nwr * (net_wins["net_R"].mean() if not net_wins.empty else 0)
            + (1 - nwr) * (net_losses["net_R"].mean() if not net_losses.empty else 0)
        )
        tpy = round(n / 5.0, 0)
        result[var] = {
            "gross_avg_R": round(g_avg_R, 3), "net_avg_R": round(n_avg_R, 3),
            "gross_wr": round(gwr * 100, 1), "net_pf": round(min(npf, 99.9), 2),
            "net_exp": round(n_exp, 3), "n_trades": n, "trades_per_year": int(tpy),
        }
    return result


# ---------------------------------------------------------------------------
# Full system comparison writer
# ---------------------------------------------------------------------------

def write_full_comparison(
    daily_results: dict,   # {ticker: {hold_bars: {"metrics": ..., "dow": ..., "regime": ...}}}
    output_dir: Path,
) -> None:
    """
    Write full_system_comparison.txt combining weekend variation results
    with daily hold period results. Loads CSVs if available; falls back
    to V1_BASELINE hardcoded values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "full_system_comparison.txt"

    sig_imp_btc   = output_dir / "signal_improvements_btc.csv"
    sig_comb_btc  = output_dir / "signal_combinations_btc.csv"
    sig_imp_eth   = output_dir / "signal_improvements_eth.csv"
    sig_comb_eth  = output_dir / "signal_combinations_eth.csv"

    # Load weekend variation data for BTC
    wknd_btc = {}
    wknd_btc.update(_load_variation_metrics(sig_imp_btc, "BTC-USD"))
    wknd_btc.update(_load_variation_metrics(sig_comb_btc, "BTC-USD"))

    # Load weekend variation data for ETH
    wknd_eth = {}
    wknd_eth.update(_load_variation_metrics(sig_imp_eth, "ETH-USD"))
    wknd_eth.update(_load_variation_metrics(sig_comb_eth, "ETH-USD"))

    btc_v1 = V1_BASELINE["BTC-USD"]
    eth_v1 = V1_BASELINE["ETH-USD"]

    lines = []
    lines.append("=" * 65)
    lines.append("FULL SYSTEM COMPARISON — Daily MA20 Momentum + Weekend Signals")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1: Weekend variations ranked by net_avg_R
    # ------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("WEEKEND VARIATIONS — BTC-USD  (ranked by net avg R)")
    lines.append("=" * 65)
    if wknd_btc:
        sorted_btc = sorted(wknd_btc.items(), key=lambda x: x[1].get("net_avg_R", -999), reverse=True)
        lines.append(f"{'Variation':<25}  {'Trades':>6}  {'Trd/yr':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}  {'NetPF':>6}  {'NetExp':>7}")
        lines.append("-" * 80)
        for var, m in sorted_btc:
            lines.append(
                f"{var:<25}  {m.get('n_trades',0):>6}  {m.get('trades_per_year',0):>6}  "
                f"{m.get('gross_wr', 0):>5.1f}  {m.get('gross_avg_R', 0):>7.3f}  "
                f"{m.get('net_avg_R', 0):>7.3f}  {m.get('net_pf', 0):>6.2f}  "
                f"{m.get('net_exp', 0):>7.3f}"
            )
    else:
        lines.append("  (run signal_improvements.py and signal_combinations.py first)")
        lines.append(f"  V1 Baseline: gross={btc_v1['gross_avg_R']}  net={btc_v1['net_avg_R']}  "
                     f"pf={btc_v1['net_pf']}  tpy=~{btc_v1['trades_per_year']}")
    lines.append("")

    lines.append("=" * 65)
    lines.append("WEEKEND VARIATIONS — ETH-USD  (ranked by net avg R)")
    lines.append("=" * 65)
    if wknd_eth:
        sorted_eth = sorted(wknd_eth.items(), key=lambda x: x[1].get("net_avg_R", -999), reverse=True)
        lines.append(f"{'Variation':<25}  {'Trades':>6}  {'Trd/yr':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}  {'NetPF':>6}  {'NetExp':>7}")
        lines.append("-" * 80)
        for var, m in sorted_eth:
            lines.append(
                f"{var:<25}  {m.get('n_trades',0):>6}  {m.get('trades_per_year',0):>6}  "
                f"{m.get('gross_wr', 0):>5.1f}  {m.get('gross_avg_R', 0):>7.3f}  "
                f"{m.get('net_avg_R', 0):>7.3f}  {m.get('net_pf', 0):>6.2f}  "
                f"{m.get('net_exp', 0):>7.3f}"
            )
    else:
        lines.append("  (run signal_improvements.py and signal_combinations.py first)")
        lines.append(f"  V1 Baseline: gross={eth_v1['gross_avg_R']}  net={eth_v1['net_avg_R']}  "
                     f"pf={eth_v1['net_pf']}  tpy=~{eth_v1['trades_per_year']}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Daily hold period results
    # ------------------------------------------------------------------
    for ticker in ["BTC-USD", "ETH-USD"]:
        short = ticker.split("-")[0]
        lines.append("=" * 65)
        lines.append(f"DAILY SYSTEM — {ticker}  (all hold periods, Kraken net)")
        lines.append("=" * 65)
        lines.append(f"{'Hold':>4}  {'Trades':>6}  {'Trd/yr':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}  {'NetPF':>6}  {'NetExp':>7}  {'MaxDD':>6}")
        lines.append("-" * 70)
        for h in HOLD_PERIODS:
            m = daily_results.get(ticker, {}).get(h, {}).get("metrics", {})
            if m:
                lines.append(
                    f"{h:>4}  {m['n_trades']:>6}  {m['trades_per_year']:>6}  "
                    f"{m['gross_wr']:>5.1f}  {m['gross_avg_R']:>7.3f}  "
                    f"{m['net_avg_R']:>7.3f}  {m['net_pf']:>6.2f}  "
                    f"{m['net_exp']:>7.3f}  {m['max_dd']:>6.1f}%"
                )
            else:
                lines.append(f"{h:>4}  {'N/A':>6}")
        lines.append("")

        # DOW breakdown for 2-bar hold
        dow_data = daily_results.get(ticker, {}).get(2, {}).get("dow", {})
        lines.append(f"  Day-of-week breakdown (2-bar hold) — {ticker}:")
        lines.append(f"  {'Day':<5}  {'Trades':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}  {'NetExp':>7}")
        lines.append("  " + "-" * 48)
        for dow in range(7):
            d = dow_data.get(dow, {"n": 0})
            if d.get("n", 0) == 0:
                lines.append(f"  {DAYS[dow]:<5}  {'0':>6}  {'---':>5}  {'---':>7}  {'---':>7}  {'---':>7}")
            else:
                lines.append(
                    f"  {DAYS[dow]:<5}  {d['n']:>6}  {d['gross_wr']:>5.1f}  "
                    f"{d['gross_avg_R']:>7.3f}  {d['net_avg_R']:>7.3f}  {d['net_exp']:>7.3f}"
                )
        lines.append("")

        # Regime breakdown for 2-bar hold
        regime_data = daily_results.get(ticker, {}).get(2, {}).get("regime", {})
        lines.append(f"  Regime breakdown (2-bar hold) — {ticker}:")
        lines.append(f"  {'Regime':<22}  {'N':>5}  {'NetR':>7}  {'NetPF':>6}  {'NetExp':>7}  {'MaxDD':>6}")
        lines.append("  " + "-" * 58)
        for regime_name in REGIMES:
            rm = regime_data.get(regime_name, {})
            if rm:
                lines.append(
                    f"  {regime_name:<22}  {rm.get('n_trades',0):>5}  "
                    f"{rm.get('net_avg_R',0):>7.3f}  {rm.get('net_pf',0):>6.2f}  "
                    f"{rm.get('net_exp',0):>7.3f}  {rm.get('max_dd',0):>6.1f}%"
                )
            else:
                lines.append(f"  {regime_name:<22}  {'---':>5}")
        lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Weekend vs Daily comparison table
    # ------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("WEEKEND vs DAILY SYSTEM COMPARISON")
    lines.append("=" * 65)

    # Best weekend variation
    def _best_wknd(wknd_dict):
        if not wknd_dict:
            return None, None
        best_var = max(wknd_dict, key=lambda x: wknd_dict[x].get("net_avg_R", -999))
        return best_var, wknd_dict[best_var]

    btc_best_var, btc_best_wknd = _best_wknd(wknd_btc)
    eth_best_var, eth_best_wknd = _best_wknd(wknd_eth)

    # Best daily hold period (by net_avg_R)
    def _best_daily(ticker):
        best_h, best_m = None, None
        for h in HOLD_PERIODS:
            m = daily_results.get(ticker, {}).get(h, {}).get("metrics", {})
            if m and (best_m is None or m.get("net_avg_R", -999) > best_m.get("net_avg_R", -999)):
                best_h, best_m = h, m
        return best_h, best_m

    btc_best_h, btc_best_daily = _best_daily("BTC-USD")
    eth_best_h, eth_best_daily = _best_daily("ETH-USD")

    # Var2+Var4 or fallback label
    btc_wknd_label = btc_best_var if btc_best_var else "Var2+Var4"
    eth_wknd_label = eth_best_var if eth_best_var else "Var2+Var4"
    daily_btc_label = f"Daily {btc_best_h}-bar" if btc_best_h else "Daily Best"
    daily_eth_label = f"Daily {eth_best_h}-bar" if eth_best_h else "Daily Best"

    col_w1 = max(len(btc_wknd_label), len(eth_wknd_label), len("Weekend Var2+Var4")) + 2
    col_w2 = max(len(daily_btc_label), len(daily_eth_label), len("Daily Best")) + 2

    header_var = btc_wknd_label
    lines.append(f"{'':28}  {'Weekend V1':>12}  {header_var:>{col_w1}}  {'Daily Best':>{col_w2}}")

    def _fmt(val, fallback="N/A"):
        return f"{val:.3f}" if val is not None else fallback

    def _fmt2(val, fallback="N/A"):
        return f"{val:.2f}" if val is not None else fallback

    def _fmt_tpy(val, fallback="N/A"):
        return f"~{int(val)}" if val is not None else fallback

    btc_wknd_gross  = btc_best_wknd.get("gross_avg_R") if btc_best_wknd else None
    btc_wknd_net    = btc_best_wknd.get("net_avg_R") if btc_best_wknd else None
    btc_wknd_pf     = btc_best_wknd.get("net_pf") if btc_best_wknd else None
    btc_wknd_tpy    = btc_best_wknd.get("trades_per_year") if btc_best_wknd else None

    eth_wknd_gross  = eth_best_wknd.get("gross_avg_R") if eth_best_wknd else None
    eth_wknd_net    = eth_best_wknd.get("net_avg_R") if eth_best_wknd else None
    eth_wknd_pf     = eth_best_wknd.get("net_pf") if eth_best_wknd else None
    eth_wknd_tpy    = eth_best_wknd.get("trades_per_year") if eth_best_wknd else None

    btc_daily_gross = btc_best_daily.get("gross_avg_R") if btc_best_daily else None
    btc_daily_net   = btc_best_daily.get("net_avg_R") if btc_best_daily else None
    btc_daily_pf    = btc_best_daily.get("net_pf") if btc_best_daily else None
    btc_daily_tpy   = btc_best_daily.get("trades_per_year") if btc_best_daily else None

    eth_daily_gross = eth_best_daily.get("gross_avg_R") if eth_best_daily else None
    eth_daily_net   = eth_best_daily.get("net_avg_R") if eth_best_daily else None
    eth_daily_pf    = eth_best_daily.get("net_pf") if eth_best_daily else None
    eth_daily_tpy   = eth_best_daily.get("trades_per_year") if eth_best_daily else None

    wknd_note = "(run X.py first)" if btc_best_wknd is None else ""

    lines.append(f"{'BTC gross avg R':<28}  {btc_v1['gross_avg_R']:>12.3f}  "
                 f"{_fmt(btc_wknd_gross, wknd_note):>{col_w1}}  {_fmt(btc_daily_gross):>{col_w2}}")
    lines.append(f"{'BTC net avg R Kraken':<28}  {btc_v1['net_avg_R']:>12.3f}  "
                 f"{_fmt(btc_wknd_net, wknd_note):>{col_w1}}  {_fmt(btc_daily_net):>{col_w2}}")
    lines.append(f"{'BTC net profit factor':<28}  {btc_v1['net_pf']:>12.2f}  "
                 f"{_fmt2(btc_wknd_pf, wknd_note):>{col_w1}}  {_fmt2(btc_daily_pf):>{col_w2}}")
    lines.append(f"{'BTC trades/year':<28}  {_fmt_tpy(btc_v1['trades_per_year']):>12}  "
                 f"{_fmt_tpy(btc_wknd_tpy, wknd_note):>{col_w1}}  {_fmt_tpy(btc_daily_tpy):>{col_w2}}")
    lines.append(f"{'ETH gross avg R':<28}  {eth_v1['gross_avg_R']:>12.3f}  "
                 f"{_fmt(eth_wknd_gross, wknd_note):>{col_w1}}  {_fmt(eth_daily_gross):>{col_w2}}")
    lines.append(f"{'ETH net avg R Kraken':<28}  {eth_v1['net_avg_R']:>12.3f}  "
                 f"{_fmt(eth_wknd_net, wknd_note):>{col_w1}}  {_fmt(eth_daily_net):>{col_w2}}")
    lines.append(f"{'ETH net profit factor':<28}  {eth_v1['net_pf']:>12.2f}  "
                 f"{_fmt2(eth_wknd_pf, wknd_note):>{col_w1}}  {_fmt2(eth_daily_pf):>{col_w2}}")
    lines.append(f"{'ETH trades/year':<28}  {_fmt_tpy(eth_v1['trades_per_year']):>12}  "
                 f"{_fmt_tpy(eth_wknd_tpy, wknd_note):>{col_w1}}  {_fmt_tpy(eth_daily_tpy):>{col_w2}}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 4: Answers to the 5 key questions
    # ------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("KEY QUESTIONS")
    lines.append("=" * 65)
    lines.append("")

    # Q1: Is the weekend effect real?
    lines.append("Q1: Is the weekend effect real?")
    lines.append("    (Compare Sat/Sun DOW entries vs Mon-Thu in daily system)")
    for ticker in ["BTC-USD", "ETH-USD"]:
        dow_data = daily_results.get(ticker, {}).get(2, {}).get("dow", {})
        weekday_nets = [dow_data.get(d, {}).get("net_avg_R") for d in range(5) if dow_data.get(d, {}).get("n", 0) > 0]
        weekend_nets = [dow_data.get(d, {}).get("net_avg_R") for d in [5, 6] if dow_data.get(d, {}).get("n", 0) > 0]
        if weekday_nets and weekend_nets:
            avg_wd = np.mean([x for x in weekday_nets if x is not None])
            avg_we = np.mean([x for x in weekend_nets if x is not None])
            effect = "YES — weekend net R is higher" if avg_we > avg_wd else "NO — weekday net R is equal or higher"
            lines.append(f"    {ticker}: weekday avg net R = {avg_wd:.3f}  |  weekend avg net R = {avg_we:.3f}  -> {effect}")
        else:
            lines.append(f"    {ticker}: insufficient data for comparison")
    lines.append("")

    # Q2: Best single entry day
    lines.append("Q2: Best single entry day (highest net_avg_R, 2-bar hold)?")
    for ticker in ["BTC-USD", "ETH-USD"]:
        dow_data = daily_results.get(ticker, {}).get(2, {}).get("dow", {})
        best_dow = None
        best_net = -999.0
        for d in range(7):
            d_info = dow_data.get(d, {})
            if d_info.get("n", 0) > 0 and d_info.get("net_avg_R", -999) > best_net:
                best_dow = d
                best_net = d_info["net_avg_R"]
        if best_dow is not None:
            lines.append(f"    {ticker}: {DAYS[best_dow]} (net avg R = {best_net:.3f})")
        else:
            lines.append(f"    {ticker}: N/A")
    lines.append("")

    # Q3: Does increasing hold period improve or hurt net R?
    lines.append("Q3: Does increasing hold period improve or hurt net R?")
    for ticker in ["BTC-USD", "ETH-USD"]:
        net_rs = []
        for h in HOLD_PERIODS:
            m = daily_results.get(ticker, {}).get(h, {}).get("metrics", {})
            net_rs.append((h, m.get("net_avg_R")))
        valid = [(h, r) for h, r in net_rs if r is not None]
        if len(valid) >= 2:
            trend = "IMPROVES" if valid[-1][1] > valid[0][1] else "HURTS (or flat)"
            vals = "  ".join([f"{h}-bar={r:.3f}" for h, r in valid])
            lines.append(f"    {ticker}: {vals}  -> Hold period {trend} net R")
        else:
            lines.append(f"    {ticker}: insufficient data")
    lines.append("")

    # Q4: Is the daily system viable after Kraken fees?
    lines.append("Q4: Is the daily system viable after Kraken fees?")
    lines.append("    (net expectancy > 0 threshold)")
    for ticker in ["BTC-USD", "ETH-USD"]:
        viable_holds = []
        for h in HOLD_PERIODS:
            m = daily_results.get(ticker, {}).get(h, {}).get("metrics", {})
            if m and m.get("net_exp", -999) > 0:
                viable_holds.append(f"{h}-bar (exp={m['net_exp']:.3f})")
        if viable_holds:
            lines.append(f"    {ticker}: YES — viable hold periods: {', '.join(viable_holds)}")
        else:
            lines.append(f"    {ticker}: NO — no hold period achieves positive net expectancy")
    lines.append("")

    # Q5: Best combination across all tests (top 3 by net_exp)
    lines.append("Q5: Best combination across all tests (top 3 by net_exp)?")
    all_systems = []
    for ticker in ["BTC-USD", "ETH-USD"]:
        short = ticker.split("-")[0]
        # Weekend variations
        wknd_dict = wknd_btc if ticker == "BTC-USD" else wknd_eth
        for var, m in wknd_dict.items():
            all_systems.append({
                "label": f"{short} Weekend {var}",
                "net_exp": m.get("net_exp", -999),
                "net_avg_R": m.get("net_avg_R", -999),
                "net_pf": m.get("net_pf", 0),
                "trades_per_year": m.get("trades_per_year", 0),
            })
        # Daily hold periods
        for h in HOLD_PERIODS:
            m = daily_results.get(ticker, {}).get(h, {}).get("metrics", {})
            if m:
                all_systems.append({
                    "label": f"{short} Daily {h}-bar",
                    "net_exp": m.get("net_exp", -999),
                    "net_avg_R": m.get("net_avg_R", -999),
                    "net_pf": m.get("net_pf", 0),
                    "trades_per_year": m.get("trades_per_year", 0),
                })
        # V1 baseline
        v1 = V1_BASELINE[ticker]
        all_systems.append({
            "label": f"{short} Weekend V1",
            "net_exp": v1["net_avg_R"],   # proxy
            "net_avg_R": v1["net_avg_R"],
            "net_pf": v1["net_pf"],
            "trades_per_year": v1["trades_per_year"],
        })

    top3 = sorted(all_systems, key=lambda x: x["net_exp"], reverse=True)[:3]
    for rank, sys in enumerate(top3, 1):
        lines.append(
            f"    #{rank}: {sys['label']:<35}  net_exp={sys['net_exp']:.3f}  "
            f"net_avg_R={sys['net_avg_R']:.3f}  pf={sys['net_pf']:.2f}  "
            f"tpy=~{sys['trades_per_year']}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 5: Final recommendation
    # ------------------------------------------------------------------
    lines.append("=" * 65)
    lines.append("FINAL RECOMMENDATION")
    lines.append("=" * 65)
    lines.append("")

    if top3:
        top = top3[0]
        lines.append(f"  Best system to paper trade: {top['label']}")
        lines.append(f"  Parameters:")
        lines.append(f"    Signal:   close > MA20 AND ATR14 > ATR14[5 bars ago]")
        lines.append(f"    Entry:    close * (1 + {SLIPPAGE_PCT}) at signal bar")
        lines.append(f"    Stop:     entry - {ATR_MULT_STOP} * ATR14")
        lines.append(f"    Target:   entry + {R_TARGET}R")
        lines.append(f"    Risk:     {RISK_PCT*100:.0f}% of equity per trade")
        lines.append(f"    Exchange: Kraken (taker {KRAKEN_TAKER*100:.2f}%)")
        lines.append("")

    # Separate recommendation for weekend vs daily
    btc_daily_viable = any(
        daily_results.get("BTC-USD", {}).get(h, {}).get("metrics", {}).get("net_exp", -999) > 0
        for h in HOLD_PERIODS
    )
    eth_daily_viable = any(
        daily_results.get("ETH-USD", {}).get(h, {}).get("metrics", {}).get("net_exp", -999) > 0
        for h in HOLD_PERIODS
    )

    lines.append("  System viability summary:")
    lines.append(f"    BTC Weekend V1:    net_avg_R={btc_v1['net_avg_R']:.3f}  pf={btc_v1['net_pf']:.2f}  tpy=~{btc_v1['trades_per_year']}")
    lines.append(f"    ETH Weekend V1:    net_avg_R={eth_v1['net_avg_R']:.3f}  pf={eth_v1['net_pf']:.2f}  tpy=~{eth_v1['trades_per_year']}")
    if btc_best_daily:
        lines.append(f"    BTC Daily ({btc_best_h}-bar):  net_avg_R={btc_daily_net:.3f}  pf={btc_daily_pf:.2f}  tpy=~{btc_daily_tpy}  {'VIABLE' if btc_daily_viable else 'NOT viable'}")
    if eth_best_daily:
        lines.append(f"    ETH Daily ({eth_best_h}-bar):  net_avg_R={eth_daily_net:.3f}  pf={eth_daily_pf:.2f}  tpy=~{eth_daily_tpy}  {'VIABLE' if eth_daily_viable else 'NOT viable'}")
    lines.append("")
    lines.append("  Next steps:")
    lines.append("    1. Paper trade the top-ranked system for 30 days")
    lines.append("    2. Focus entries on the best DOW identified in Q2")
    lines.append("    3. Monitor regime — avoid bear market entries if net_exp < 0")
    lines.append("    4. Confirm breakeven fee rate exceeds Kraken taker before going live")
    lines.append("")
    lines.append("=" * 65)
    lines.append("END OF REPORT")
    lines.append("=" * 65)

    out_path.write_text("\n".join(lines))
    print(f"\n[+] Written: {out_path}")


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def _print_ticker_table(ticker: str, results: dict) -> None:
    """Print summary table for one ticker across all hold periods."""
    print()
    print("=" * 65)
    print(f"DAILY SYSTEM — {ticker}  (all hold periods, Kraken net)")
    print("=" * 65)
    print(f"{'Hold':>4}  {'Trades':>6}  {'Trd/yr':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}  {'NetPF':>6}  {'NetExp':>7}  {'MaxDD':>6}")
    print("-" * 68)
    for h in HOLD_PERIODS:
        m = results.get(h, {}).get("metrics", {})
        if m:
            print(
                f"{h:>4}  {m['n_trades']:>6}  {m['trades_per_year']:>6}  "
                f"{m['gross_wr']:>5.1f}%  {m['gross_avg_R']:>7.3f}  "
                f"{m['net_avg_R']:>7.3f}  {m['net_pf']:>6.2f}  "
                f"{m['net_exp']:>7.3f}  {m['max_dd']:>6.1f}%"
            )
        else:
            print(f"{h:>4}  {'N/A':>6}")

    # DOW table for 2-bar hold
    print()
    print(f"Day-of-week breakdown (2-bar hold):")
    dow_data = results.get(2, {}).get("dow", {})
    print(f"  {'Day':<5}  {'Trades':>6}  {'WR%':>5}  {'GrossR':>7}  {'NetR':>7}")
    print("  " + "-" * 38)
    for dow in range(7):
        d = dow_data.get(dow, {"n": 0})
        if d.get("n", 0) == 0:
            print(f"  {DAYS[dow]:<5}  {'0':>6}  {'---':>5}  {'---':>7}  {'---':>7}")
        else:
            print(
                f"  {DAYS[dow]:<5}  {d['n']:>6}  {d['gross_wr']:>5.1f}%  "
                f"{d['gross_avg_R']:>7.3f}  {d['net_avg_R']:>7.3f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[*] Loading data ...")
    raw_data = load_data()

    print("[*] Prepping tickers ...")
    tickers_prepped = {
        "BTC-USD": prep_ticker(raw_data["BTC-USD"]),
        "ETH-USD": prep_ticker(raw_data["ETH-USD"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_results = {}   # {ticker: {hold_bars: {"metrics": ..., "dow": ..., "regime": ...}}}
    all_rows_btc  = []
    all_rows_eth  = []

    for ticker, ticker_df in tickers_prepped.items():
        print(f"\n[*] Running daily system for {ticker} ...")
        daily_results[ticker] = {}

        for h in HOLD_PERIODS:
            print(f"    hold={h} bars ...", end=" ", flush=True)
            rows = run_daily(ticker_df, ticker, h)
            metrics = _calc_metrics(rows)
            dow     = _dow_breakdown(rows)
            regime  = _regime_breakdown(rows)
            daily_results[ticker][h] = {"metrics": metrics, "dow": dow, "regime": regime, "rows": rows}
            n = metrics.get("n_trades", 0)
            print(f"{n} trades")

            if ticker == "BTC-USD":
                all_rows_btc.extend(rows)
            else:
                all_rows_eth.extend(rows)

    # Terminal tables
    for ticker in ["BTC-USD", "ETH-USD"]:
        _print_ticker_table(ticker, daily_results[ticker])

    # Save CSVs
    print(f"\n[*] Saving CSVs to {OUTPUT_DIR} ...")
    cols = [
        "date", "ticker", "hold_bars", "direction", "exit_type", "day_of_week",
        "entry_price", "stop_price", "target_price", "exit_price",
        "units", "gross_pnl", "gross_R", "entry_value", "exit_value",
        "fee", "net_pnl", "net_R",
    ]

    btc_df = pd.DataFrame(all_rows_btc)
    for c in cols:
        if c not in btc_df.columns:
            btc_df[c] = None
    btc_df[cols].to_csv(OUTPUT_DIR / "daily_system_btc.csv", index=False)
    print(f"    Saved daily_system_btc.csv  ({len(btc_df)} rows)")

    eth_df = pd.DataFrame(all_rows_eth)
    for c in cols:
        if c not in eth_df.columns:
            eth_df[c] = None
    eth_df[cols].to_csv(OUTPUT_DIR / "daily_system_eth.csv", index=False)
    print(f"    Saved daily_system_eth.csv  ({len(eth_df)} rows)")

    # Full system comparison
    print("\n[*] Writing full_system_comparison.txt ...")
    write_full_comparison(daily_results, OUTPUT_DIR)

    print("\n[+] Done.")


if __name__ == "__main__":
    main()
