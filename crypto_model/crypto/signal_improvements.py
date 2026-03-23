"""
crypto/signal_improvements.py
==============================
Four signal improvements tested independently against the V1 baseline.

V1 baseline (fee_analysis.py):
  BTC: gross avg R 0.169 | net avg R 0.029 on Kraken | win rate 57.3%
  ETH: gross avg R 0.182 | net avg R 0.090 on Kraken | win rate 55.5%
  Parameters: stop=1.25x ATR, target=2.0R, risk=5%/instrument

Variations (each tested in isolation):
  Var1 — Momentum Confirmation : Friday close > prior Friday close
  Var2 — Volatility Expansion  : ATR14 Friday > ATR14 prior Friday
  Var3 — Short Side            : SHORT when below MA20 (mirror of LONG)
  Var4 — Monday Extended Hold  : extend profitable TIME exits to Monday close

Fees: Kraken Pro 0.26% taker on both legs (entry + exit).

Data: loads from Data/backtest_cache/ohlcv_5yr.csv; downloads and saves on
      first run so subsequent runs are instant.

Outputs → Results/crypto_backtest/
  signal_improvements_btc.csv
  signal_improvements_eth.csv
  signal_improvements_summary.txt

Do NOT modify: backtest_engine.py, data_loader.py, fee_analysis.py,
               parameter_sweep.py, main.py, backtest_variations.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays

# ── Locked parameters ───────────────────────────────────────────────────────
ATR_MULT_STOP    = 1.25
R_TARGET         = 2.0
RISK_PCT         = 0.05
STARTING_CAPITAL = 500.0
SLIPPAGE_PCT     = 0.001
KRAKEN_TAKER     = 0.0026   # 0.26%

CACHE_PATH = Path(__file__).parent.parent / "Data" / "backtest_cache" / "ohlcv_5yr.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "Results" / "crypto_backtest"

# V1 baseline (from fee_analysis.py output) for delta reporting
V1_BASELINE = {
    "BTC-USD": {"gross_wr": 57.3, "gross_avg_R": 0.169, "net_avg_R": 0.029,
                "gross_pf": 1.94, "net_pf": 1.12, "gross_exp": 0.169, "net_exp": 0.029},
    "ETH-USD": {"gross_wr": 55.5, "gross_avg_R": 0.182, "net_avg_R": 0.090,
                "gross_pf": 2.07, "net_pf": 1.42, "gross_exp": 0.182, "net_exp": 0.090},
}


# ── Data loading with cache ─────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    """Load from CSV cache; download + save if cache missing."""
    if CACHE_PATH.exists():
        print(f"  Loading from cache: {CACHE_PATH}")
        raw = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        tickers_found = raw["ticker"].unique().tolist()
        print(f"  Cached tickers: {tickers_found}  rows: {len(raw)}")
    else:
        print("  Cache not found — downloading from yfinance (one-time)...")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = load_crypto_data(["BTC-USD", "ETH-USD"])
        raw.to_csv(CACHE_PATH, index=False)
        print(f"  Saved to {CACHE_PATH}")

    return {
        "BTC-USD": get_ticker_df(raw, "BTC-USD"),
        "ETH-USD": get_ticker_df(raw, "ETH-USD"),
    }


# ── Exit simulation helpers ─────────────────────────────────────────────────

def _exit_long(weekend: pd.DataFrame, entry: float, stop: float, target: float):
    """Returns (exit_price, exit_type) for a LONG position."""
    for _, bar in weekend.iterrows():
        if bar["low"] <= stop:
            return stop, "STOP"
        if bar["high"] >= target:
            return target, "TARGET"
    return weekend.iloc[-1]["close"], "TIME"


def _exit_short(weekend: pd.DataFrame, entry: float, stop: float, target: float):
    """Returns (exit_price, exit_type) for a SHORT position.
    stop  = entry + 1.25×ATR  (above entry)
    target= entry - 2.0×risk   (below entry)
    """
    for _, bar in weekend.iterrows():
        if bar["high"] >= stop:
            return stop, "STOP"
        if bar["low"] <= target:
            return target, "TARGET"
    return weekend.iloc[-1]["close"], "TIME"


def _get_weekend(ticker_df: pd.DataFrame, fri_pos: int) -> pd.DataFrame:
    next_bars = ticker_df.iloc[fri_pos + 1: fri_pos + 3]
    return next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()


def _get_monday(ticker_df: pd.DataFrame, fri_pos: int) -> Optional[pd.Series]:
    """Return Monday bar (3 bars after Friday) or None if missing/wrong day."""
    try:
        mon = ticker_df.iloc[fri_pos + 3]
        return mon if mon.name.dayofweek == 0 else None
    except IndexError:
        return None


# ── Fee application ─────────────────────────────────────────────────────────

def _apply_kraken_fee(entry_px: float, exit_px: float, units: float) -> float:
    return (entry_px * units + exit_px * units) * KRAKEN_TAKER


# ── Metrics calculator ──────────────────────────────────────────────────────

def _calc_metrics(trades: list[dict], variation: str, ticker: str,
                  n_v1_trades: int) -> dict:
    df = pd.DataFrame(trades)
    completed = df[df["direction"].isin(["LONG", "SHORT"]) &
                   (df["exit_type"] != "NO_DATA")].dropna(subset=["gross_R"])

    n = len(completed)
    n_filtered = n_v1_trades - len(completed[completed["direction"] == "LONG"]) \
        if variation not in ("Var3-Short", "Var3-Combined") else 0

    if n == 0:
        return {"variation": variation, "ticker": ticker, "n_trades": 0,
                "n_filtered": n_filtered}

    # Gross
    gross_wins   = completed[completed["gross_pnl"] > 0]
    gross_losses = completed[completed["gross_pnl"] <= 0]
    gross_wr     = len(gross_wins) / n
    gross_avg_R  = completed["gross_R"].mean()
    gw_sum       = gross_wins["gross_R"].sum()
    gl_sum       = abs(gross_losses["gross_R"].sum())
    gross_pf     = gw_sum / gl_sum if gl_sum > 0 else np.inf
    gross_aw     = gross_wins["gross_R"].mean()   if not gross_wins.empty   else 0.0
    gross_al     = gross_losses["gross_R"].mean() if not gross_losses.empty else 0.0
    gross_exp    = gross_wr * gross_aw + (1 - gross_wr) * gross_al

    # Net (Kraken)
    net_wins   = completed[completed["net_pnl"] > 0]
    net_losses = completed[completed["net_pnl"] <= 0]
    net_wr     = len(net_wins) / n
    net_avg_R  = completed["net_R"].mean() if "net_R" in completed else None
    nw_sum     = net_wins["net_R"].sum()   if "net_R" in completed else 0
    nl_sum     = abs(net_losses["net_R"].sum()) if "net_R" in completed else 0
    net_pf     = nw_sum / nl_sum if nl_sum > 0 else np.inf
    net_aw     = net_wins["net_R"].mean()   if not net_wins.empty   and "net_R" in completed else 0.0
    net_al     = net_losses["net_R"].mean() if not net_losses.empty and "net_R" in completed else 0.0
    net_exp    = net_wr * net_aw + (1 - net_wr) * net_al

    # Max drawdown on dollar equity curve
    equity = [STARTING_CAPITAL]
    for pl in completed["gross_pnl"]:
        equity.append(max(equity[-1] + pl, 0.01))
    equity_s    = pd.Series(equity)
    running_max = equity_s.cummax()
    max_dd      = float(((equity_s - running_max) / running_max).min() * 100)

    # Exit breakdown (LONG trades only for exit pct)
    long_comp = completed[completed["direction"] == "LONG"]
    n_long    = len(long_comp)
    ec        = long_comp["exit_type"].value_counts().to_dict() if n_long else {}

    # Break-even fee rate (gross P&L / total round-trip notional)
    total_gross_pnl = completed["gross_pnl"].sum()
    total_notional  = (completed["entry_value"] + completed["exit_value"]).sum()
    be_rate = (total_gross_pnl / total_notional * 100) if total_notional > 0 and total_gross_pnl > 0 else 0.0

    return {
        "variation":   variation,
        "ticker":      ticker,
        "n_trades":    n,
        "n_long":      n_long,
        "n_short":     len(completed[completed["direction"] == "SHORT"]),
        "n_filtered":  n_filtered,
        "gross_wr":    round(gross_wr * 100, 1),
        "gross_avg_R": round(gross_avg_R, 3),
        "gross_pf":    round(min(gross_pf, 99.9), 2),
        "gross_exp":   round(gross_exp, 3),
        "net_wr":      round(net_wr * 100, 1),
        "net_avg_R":   round(net_avg_R, 3) if net_avg_R is not None else None,
        "net_pf":      round(min(net_pf, 99.9), 2),
        "net_exp":     round(net_exp, 3),
        "max_dd":      round(max_dd, 1),
        "pct_target":  round(ec.get("TARGET", 0) / n_long * 100, 1) if n_long else 0,
        "pct_stop":    round(ec.get("STOP", 0)   / n_long * 100, 1) if n_long else 0,
        "pct_time":    round((ec.get("TIME", 0) + ec.get("TIME_MON", 0)) / n_long * 100, 1) if n_long else 0,
        "pct_target_mon": round(ec.get("TARGET_MON", 0) / n_long * 100, 1) if n_long else 0,
        "pct_be_mon":     round(ec.get("BE_MON", 0)     / n_long * 100, 1) if n_long else 0,
        "breakeven_rate": round(be_rate, 4),
    }


# ── V1 Baseline ─────────────────────────────────────────────────────────────

def run_v1(ticker_df: pd.DataFrame, ticker: str) -> tuple[list[dict], int]:
    """Standard MA20 long-only backtest at locked params."""
    fridays = get_fridays(ticker_df)
    rows = []
    equity = STARTING_CAPITAL
    n_long_completed = 0

    for fri_date, fri_row in fridays.iterrows():
        if not fri_row["above_ma20"]:
            rows.append({"date": fri_date, "variation": "V1", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        entry = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop  = entry - ATR_MULT_STOP * fri_row["atr14"]
        target= entry + R_TARGET * (entry - stop)
        rpu   = entry - stop
        units = (equity * RISK_PCT) / rpu

        weekend = _get_weekend(ticker_df, ticker_df.index.get_loc(fri_date))
        if weekend.empty:
            rows.append({"date": fri_date, "variation": "V1", "ticker": ticker,
                         "direction": "LONG", "exit_type": "NO_DATA",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": entry * units, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)
        gross_pnl = units * (exit_px - entry)
        gross_R   = (exit_px - entry) / rpu
        fee       = _apply_kraken_fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / (units * rpu)
        equity    = max(equity + gross_pnl, 0.01)
        n_long_completed += 1

        rows.append({"date": fri_date, "variation": "V1", "ticker": ticker,
                     "direction": "LONG", "exit_type": exit_type,
                     "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                     "target_price": round(target, 4), "exit_price": round(exit_px, 4),
                     "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
                     "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
                     "exit_value": round(exit_px * units, 4),
                     "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4)})

    return rows, n_long_completed


# ── Var 1: Momentum Confirmation ────────────────────────────────────────────

def run_var1_momentum(ticker_df: pd.DataFrame, ticker: str) -> list[dict]:
    """LONG only if Friday close > prior Friday close (higher weekly close)."""
    fridays = get_fridays(ticker_df)
    rows = []
    equity = STARTING_CAPITAL

    for i, (fri_date, fri_row) in enumerate(fridays.iterrows()):
        if not fri_row["above_ma20"]:
            rows.append({"date": fri_date, "variation": "Var1-Momentum", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        # Need prior Friday
        if i == 0:
            rows.append({"date": fri_date, "variation": "Var1-Momentum", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "FILTERED_NO_PRIOR",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        prior_fri_close = fridays.iloc[i - 1]["close"]
        if fri_row["close"] <= prior_fri_close:
            # Above MA20 but lower weekly close — skip
            rows.append({"date": fri_date, "variation": "Var1-Momentum", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "FILTERED_MOMENTUM",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        entry = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop  = entry - ATR_MULT_STOP * fri_row["atr14"]
        target= entry + R_TARGET * (entry - stop)
        rpu   = entry - stop
        units = (equity * RISK_PCT) / rpu

        weekend = _get_weekend(ticker_df, ticker_df.index.get_loc(fri_date))
        if weekend.empty:
            rows.append({"date": fri_date, "variation": "Var1-Momentum", "ticker": ticker,
                         "direction": "LONG", "exit_type": "NO_DATA",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": entry * units, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)
        gross_pnl = units * (exit_px - entry)
        gross_R   = (exit_px - entry) / rpu
        fee       = _apply_kraken_fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / (units * rpu)
        equity    = max(equity + gross_pnl, 0.01)

        rows.append({"date": fri_date, "variation": "Var1-Momentum", "ticker": ticker,
                     "direction": "LONG", "exit_type": exit_type,
                     "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                     "target_price": round(target, 4), "exit_price": round(exit_px, 4),
                     "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
                     "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
                     "exit_value": round(exit_px * units, 4),
                     "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4)})

    return rows


# ── Var 2: Volatility Expansion ─────────────────────────────────────────────

def run_var2_vol(ticker_df: pd.DataFrame, ticker: str) -> list[dict]:
    """LONG only if ATR14 Friday > ATR14 prior Friday (expanding volatility)."""
    fridays = get_fridays(ticker_df)
    rows = []
    equity = STARTING_CAPITAL

    for i, (fri_date, fri_row) in enumerate(fridays.iterrows()):
        if not fri_row["above_ma20"]:
            rows.append({"date": fri_date, "variation": "Var2-VolExpansion", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        if i == 0:
            rows.append({"date": fri_date, "variation": "Var2-VolExpansion", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "FILTERED_NO_PRIOR",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        prior_atr = fridays.iloc[i - 1]["atr14"]
        if fri_row["atr14"] <= prior_atr:
            rows.append({"date": fri_date, "variation": "Var2-VolExpansion", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "FILTERED_VOL_CONTRACTION",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        entry = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop  = entry - ATR_MULT_STOP * fri_row["atr14"]
        target= entry + R_TARGET * (entry - stop)
        rpu   = entry - stop
        units = (equity * RISK_PCT) / rpu

        weekend = _get_weekend(ticker_df, ticker_df.index.get_loc(fri_date))
        if weekend.empty:
            rows.append({"date": fri_date, "variation": "Var2-VolExpansion", "ticker": ticker,
                         "direction": "LONG", "exit_type": "NO_DATA",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": entry * units, "exit_value": None, "net_R": None, "net_pnl": 0.0})
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)
        gross_pnl = units * (exit_px - entry)
        gross_R   = (exit_px - entry) / rpu
        fee       = _apply_kraken_fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / (units * rpu)
        equity    = max(equity + gross_pnl, 0.01)

        rows.append({"date": fri_date, "variation": "Var2-VolExpansion", "ticker": ticker,
                     "direction": "LONG", "exit_type": exit_type,
                     "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                     "target_price": round(target, 4), "exit_price": round(exit_px, 4),
                     "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
                     "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
                     "exit_value": round(exit_px * units, 4),
                     "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4)})

    return rows


# ── Var 3: Short Side ────────────────────────────────────────────────────────

def _make_short_row(fri_date, fri_row, ticker_df, ticker, equity, variation):
    """Simulate a single SHORT trade. Returns a row dict and updated equity."""
    entry = fri_row["close"] * (1 - SLIPPAGE_PCT)  # short at slight discount
    rpu   = ATR_MULT_STOP * fri_row["atr14"]        # risk per unit = stop distance
    stop  = entry + rpu                              # stop above entry
    target= entry - R_TARGET * rpu                  # target below entry
    units = (equity * RISK_PCT) / rpu

    weekend = _get_weekend(ticker_df, ticker_df.index.get_loc(fri_date))
    if weekend.empty:
        return {"date": fri_date, "variation": variation, "ticker": ticker,
                "direction": "SHORT", "exit_type": "NO_DATA",
                "gross_pnl": 0.0, "gross_R": None,
                "entry_value": entry * units, "exit_value": None,
                "net_R": None, "net_pnl": 0.0}, equity

    exit_px, exit_type = _exit_short(weekend, entry, stop, target)
    gross_pnl = units * (entry - exit_px)       # positive when price falls
    gross_R   = (entry - exit_px) / rpu
    fee       = _apply_kraken_fee(entry, exit_px, units)
    net_pnl   = gross_pnl - fee
    net_R     = net_pnl / (units * rpu)
    equity    = max(equity + gross_pnl, 0.01)

    return {"date": fri_date, "variation": variation, "ticker": ticker,
            "direction": "SHORT", "exit_type": exit_type,
            "entry_price": round(entry, 4), "stop_price": round(stop, 4),
            "target_price": round(target, 4), "exit_price": round(exit_px, 4),
            "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
            "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
            "exit_value": round(exit_px * units, 4),
            "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4)}, equity


def run_var3_short(ticker_df: pd.DataFrame, ticker: str) -> dict[str, list[dict]]:
    """
    Returns three trade lists:
      'short_only'  — SHORT when below MA20, skip when above
      'long_only'   — same as V1 (LONG when above, skip when below)
      'combined'    — LONG above MA20, SHORT below MA20
    """
    fridays = get_fridays(ticker_df)
    short_only = []
    combined   = []
    equity_s   = STARTING_CAPITAL
    equity_c   = STARTING_CAPITAL

    for fri_date, fri_row in fridays.iterrows():
        above_ma = bool(fri_row["above_ma20"])

        # ── SHORT-ONLY ────────────────────────────────────────────────────
        if above_ma:
            short_only.append({"date": fri_date, "variation": "Var3-Short", "ticker": ticker,
                               "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                               "gross_pnl": 0.0, "gross_R": None,
                               "entry_value": None, "exit_value": None,
                               "net_R": None, "net_pnl": 0.0})
        else:
            row_s, equity_s = _make_short_row(fri_date, fri_row, ticker_df,
                                              ticker, equity_s, "Var3-Short")
            short_only.append(row_s)

        # ── COMBINED ──────────────────────────────────────────────────────
        if above_ma:
            # LONG leg (mirror of V1 with combined equity)
            entry = fri_row["close"] * (1 + SLIPPAGE_PCT)
            stop  = entry - ATR_MULT_STOP * fri_row["atr14"]
            target= entry + R_TARGET * (entry - stop)
            rpu   = entry - stop
            units = (equity_c * RISK_PCT) / rpu

            weekend = _get_weekend(ticker_df, ticker_df.index.get_loc(fri_date))
            if weekend.empty:
                combined.append({"date": fri_date, "variation": "Var3-Combined", "ticker": ticker,
                                 "direction": "LONG", "exit_type": "NO_DATA",
                                 "gross_pnl": 0.0, "gross_R": None,
                                 "entry_value": entry * units, "exit_value": None,
                                 "net_R": None, "net_pnl": 0.0})
            else:
                exit_px, exit_type = _exit_long(weekend, entry, stop, target)
                gross_pnl = units * (exit_px - entry)
                gross_R   = (exit_px - entry) / rpu
                fee       = _apply_kraken_fee(entry, exit_px, units)
                net_pnl   = gross_pnl - fee
                net_R     = net_pnl / (units * rpu)
                equity_c  = max(equity_c + gross_pnl, 0.01)
                combined.append({"date": fri_date, "variation": "Var3-Combined", "ticker": ticker,
                                 "direction": "LONG", "exit_type": exit_type,
                                 "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                                 "target_price": round(target, 4), "exit_price": round(exit_px, 4),
                                 "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
                                 "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
                                 "exit_value": round(exit_px * units, 4),
                                 "fee": round(fee, 4), "net_pnl": round(net_pnl, 4),
                                 "net_R": round(net_R, 4)})
        else:
            row_c, equity_c = _make_short_row(fri_date, fri_row, ticker_df,
                                              ticker, equity_c, "Var3-Combined")
            combined.append(row_c)

    return {"Var3-Short": short_only, "Var3-Combined": combined}


# ── Var 4: Monday Extended Hold ──────────────────────────────────────────────

def run_var4_monday(ticker_df: pd.DataFrame, ticker: str) -> tuple[list[dict], dict]:
    """
    Extend profitable TIME exits into Monday.
    Stop moves to breakeven (entry). Force-close at Monday close.

    Returns (rows, extension_stats).
    """
    fridays = get_fridays(ticker_df)
    rows = []
    equity = STARTING_CAPITAL

    stats = {
        "n_extended": 0,
        "n_mon_target": 0,
        "n_mon_be": 0,
        "n_mon_time": 0,
        "pnl_gain_vs_sunday": 0.0,  # sum of (monday_pnl - sunday_pnl) for extended trades
    }

    for fri_date, fri_row in fridays.iterrows():
        if not fri_row["above_ma20"]:
            rows.append({"date": fri_date, "variation": "Var4-Monday", "ticker": ticker,
                         "direction": "NO_TRADE", "exit_type": "NO_TRADE",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": None, "exit_value": None,
                         "net_R": None, "net_pnl": 0.0})
            continue

        entry = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop  = entry - ATR_MULT_STOP * fri_row["atr14"]
        target= entry + R_TARGET * (entry - stop)
        rpu   = entry - stop
        units = (equity * RISK_PCT) / rpu

        fri_pos = ticker_df.index.get_loc(fri_date)
        weekend = _get_weekend(ticker_df, fri_pos)

        if weekend.empty:
            rows.append({"date": fri_date, "variation": "Var4-Monday", "ticker": ticker,
                         "direction": "LONG", "exit_type": "NO_DATA",
                         "gross_pnl": 0.0, "gross_R": None,
                         "entry_value": entry * units, "exit_value": None,
                         "net_R": None, "net_pnl": 0.0})
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)

        # Extend only if TIME exit AND currently profitable
        if exit_type == "TIME" and exit_px > entry:
            sunday_pnl = units * (exit_px - entry)
            monday_bar = _get_monday(ticker_df, fri_pos)

            if monday_bar is not None:
                stats["n_extended"] += 1
                be_stop = entry  # breakeven stop

                if monday_bar["low"] <= be_stop:
                    # Stopped at breakeven
                    exit_px    = be_stop
                    exit_type  = "BE_MON"
                    stats["n_mon_be"] += 1
                elif monday_bar["high"] >= target:
                    # Hit 2R target on Monday
                    exit_px   = target
                    exit_type = "TARGET_MON"
                    stats["n_mon_target"] += 1
                else:
                    # Timed out Monday close
                    exit_px   = monday_bar["close"]
                    exit_type = "TIME_MON"
                    stats["n_mon_time"] += 1

                monday_pnl = units * (exit_px - entry)
                stats["pnl_gain_vs_sunday"] += round(monday_pnl - sunday_pnl, 4)

        gross_pnl = units * (exit_px - entry)
        gross_R   = (exit_px - entry) / rpu
        fee       = _apply_kraken_fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / (units * rpu)
        equity    = max(equity + gross_pnl, 0.01)

        rows.append({"date": fri_date, "variation": "Var4-Monday", "ticker": ticker,
                     "direction": "LONG", "exit_type": exit_type,
                     "entry_price": round(entry, 4), "stop_price": round(stop, 4),
                     "target_price": round(target, 4), "exit_price": round(exit_px, 4),
                     "units": round(units, 8), "gross_pnl": round(gross_pnl, 4),
                     "gross_R": round(gross_R, 4), "entry_value": round(entry * units, 4),
                     "exit_value": round(exit_px * units, 4),
                     "fee": round(fee, 4), "net_pnl": round(net_pnl, 4), "net_R": round(net_R, 4)})

    return rows, stats


# ── Terminal table ───────────────────────────────────────────────────────────

def _row(m: dict, v1: dict) -> str:
    if m.get("n_trades", 0) == 0:
        return f"  {m['variation']:<22}  {'0':>5}  {'N/A':>7}  {'N/A':>5}  {'N/A':>6}  {'N/A':>6}  {'N/A':>5}  {'N/A':>5}"

    def d(key, fmt=".3f"):
        val = m.get(key)
        ref = v1.get(key)
        if val is None or ref is None:
            return "  N/A"
        delta = val - ref
        sign  = "+" if delta >= 0 else ""
        return f"{val:{fmt}} ({sign}{delta:{fmt}})"

    wr_delta   = f"{m['gross_wr']:.1f}%({m['gross_wr']-v1['gross_wr']:+.1f})"
    gross_r    = d("gross_avg_R")
    net_r      = d("net_avg_R")
    net_pf_str = d("net_pf", ".2f")
    dd_str     = f"{m['max_dd']:.1f}%"

    return (f"  {m['variation']:<22}  {m['n_trades']:>5}  {m.get('n_filtered',0):>7}  "
            f"{wr_delta:>14}  {gross_r:>20}  {net_r:>20}  {net_pf_str:>16}  {dd_str:>5}")


def print_table(all_metrics: list[dict], ticker: str, v1_m: dict) -> None:
    print()
    print("=" * 115)
    print(f"  SIGNAL IMPROVEMENT RESULTS — {ticker}")
    print("=" * 115)
    print(f"  {'Variation':<22}  {'Trades':>5}  {'Filterd':>7}  "
          f"{'WR% (delta)':>14}  {'GrossR (delta)':>20}  "
          f"{'NetR (delta)':>20}  {'NetPF (delta)':>16}  {'DD%':>5}")
    print(f"  {'-'*110}")
    for m in all_metrics:
        print(_row(m, v1_m))
    print()


# ── Report writer ─────────────────────────────────────────────────────────────

def _fmt_m(m: dict, v1: dict) -> list[str]:
    if m.get("n_trades", 0) == 0:
        return [f"  {m['variation']}: No completed trades."]

    def delta(key, fmt=".3f"):
        val = m.get(key)
        ref = v1.get(key)
        if val is None or ref is None:
            return "N/A"
        d = val - ref
        return f"{val:{fmt}} ({d:+{fmt}})"

    lines = [
        f"\n  ── {m['variation']} ──",
        f"  Trades completed : {m['n_trades']}  "
        f"(LONG: {m.get('n_long', m['n_trades'])}  SHORT: {m.get('n_short', 0)}  "
        f"Filtered vs V1: {m.get('n_filtered', 0)})",
        f"  {'Metric':<26}  {'GROSS':>10}  {'NET-KRAKEN':>10}  {'V1 DELTA':>12}",
        f"  {'-'*62}",
        f"  {'Win Rate':<26}  {m['gross_wr']:>9.1f}%  {m['net_wr']:>9.1f}%  "
        f"  {m['gross_wr'] - v1['gross_wr']:>+10.1f}%",
        f"  {'Avg R':<26}  {m['gross_avg_R']:>10.3f}  "
        f"{m['net_avg_R'] if m['net_avg_R'] is not None else 'N/A':>10}  "
        f"  {(m['net_avg_R'] - v1['net_avg_R'])if m['net_avg_R'] is not None else float('nan'):>+10.3f}",
        f"  {'Profit Factor':<26}  {m['gross_pf']:>10.2f}  {m['net_pf']:>10.2f}  "
        f"  {m['net_pf'] - v1['net_pf']:>+10.2f}",
        f"  {'Expectancy R':<26}  {m['gross_exp']:>10.3f}  {m['net_exp']:>10.3f}  "
        f"  {m['net_exp'] - v1['net_exp']:>+10.3f}",
        f"  {'Max Drawdown':<26}  {m['max_dd']:>9.1f}%",
        f"  Exit: TARGET {m['pct_target']:.1f}%  STOP {m['pct_stop']:.1f}%  "
        f"TIME {m['pct_time']:.1f}%"
        + (f"  TARGET_MON {m['pct_target_mon']:.1f}%  BE_MON {m['pct_be_mon']:.1f}%"
           if m.get("pct_target_mon", 0) or m.get("pct_be_mon", 0) else ""),
        f"  Break-even fee rate : {m['breakeven_rate']:.4f}%/leg  "
        f"(Kraken actual: 0.26%  headroom: {m['breakeven_rate'] - 0.26:+.4f}%)",
    ]
    return lines


def write_summary(
    all_metrics_btc: list[dict],
    all_metrics_eth: list[dict],
    mon_stats: dict,   # {"BTC-USD": stats_dict, "ETH-USD": stats_dict}
    out_dir: Path,
) -> None:
    v1_btc = V1_BASELINE["BTC-USD"]
    v1_eth = V1_BASELINE["ETH-USD"]

    lines = [
        "=" * 65,
        "SIGNAL IMPROVEMENT ANALYSIS — BTC + ETH Weekend MA20 System",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Fees      : Kraken Pro 0.26% taker (both legs)",
        f"Params    : stop={ATR_MULT_STOP}×ATR  target={R_TARGET}R  risk={RISK_PCT*100:.0f}%",
        "=" * 65,
        "",
        "BTC-USD — ALL VARIATIONS vs V1 BASELINE",
        "=" * 65,
    ]
    for m in all_metrics_btc:
        lines += _fmt_m(m, v1_btc)

    lines += ["", "ETH-USD — ALL VARIATIONS vs V1 BASELINE", "=" * 65]
    for m in all_metrics_eth:
        lines += _fmt_m(m, v1_eth)

    # Monday stats detail
    lines += ["", "VAR4 MONDAY EXTENSION — DETAIL", "=" * 65]
    for ticker, s in mon_stats.items():
        lines += [
            f"\n  {ticker}",
            f"  Profitable TIME exits extended   : {s['n_extended']}",
            f"  Hit 2R target on Monday          : {s['n_mon_target']}  "
            f"({s['n_mon_target']/max(s['n_extended'],1)*100:.1f}%)",
            f"  Hit breakeven stop on Monday     : {s['n_mon_be']}  "
            f"({s['n_mon_be']/max(s['n_extended'],1)*100:.1f}%)",
            f"  Timed out Monday close           : {s['n_mon_time']}  "
            f"({s['n_mon_time']/max(s['n_extended'],1)*100:.1f}%)",
            f"  Net P&L gain vs Sunday close     : ${s['pnl_gain_vs_sunday']:.2f}",
        ]

    # Recommendation
    lines += ["", "RECOMMENDATION", "=" * 65, ""]

    # Find best net_avg_R for BTC and ETH
    def best_var(metrics):
        candidates = [m for m in metrics if m.get("net_avg_R") is not None and m.get("n_trades", 0) >= 20]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m["net_avg_R"])

    best_btc = best_var(all_metrics_btc)
    best_eth = best_var(all_metrics_eth)

    if best_btc:
        lines.append(f"  BTC best net R   : {best_btc['variation']}  "
                     f"net avg R = {best_btc['net_avg_R']:.3f}  "
                     f"(V1 was {v1_btc['net_avg_R']:.3f}  delta {best_btc['net_avg_R']-v1_btc['net_avg_R']:+.3f})")
    if best_eth:
        lines.append(f"  ETH best net R   : {best_eth['variation']}  "
                     f"net avg R = {best_eth['net_avg_R']:.3f}  "
                     f"(V1 was {v1_eth['net_avg_R']:.3f}  delta {best_eth['net_avg_R']-v1_eth['net_avg_R']:+.3f})")

    lines += [
        "",
        "  Notes:",
        "  - Var1 and Var2 both reduce trade count. Check N >= 30 before concluding.",
        "  - Var3 short side adds trades but in bear regimes only — see regime breakdown",
        "    in btc_regime_breakdown.csv for bear-period context.",
        "  - Var4 Monday extension is structural: it only helps profitable TIME exits.",
        "    If pnl_gain_vs_sunday is negative the extension is a noise source, not edge.",
        "  - Target net avg R > 0.20R after Kraken fees to justify live deployment.",
        "=" * 65,
    ]

    out_path = out_dir / "signal_improvements_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"\n  Summary → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Signal Improvement Analysis — BTC + ETH Weekend MA20")
    print("=" * 65)

    data = load_data()
    btc  = data["BTC-USD"]
    eth  = data["ETH-USD"]

    all_rows_btc: list[dict] = []
    all_rows_eth: list[dict] = []
    all_metrics_btc: list[dict] = []
    all_metrics_eth: list[dict] = []
    mon_stats: dict = {}

    for ticker, ticker_df in [("BTC-USD", btc), ("ETH-USD", eth)]:
        print(f"\n{'='*40}")
        print(f"  Running variations for {ticker}...")
        all_rows = all_rows_btc if ticker == "BTC-USD" else all_rows_eth
        all_metrics = all_metrics_btc if ticker == "BTC-USD" else all_metrics_eth
        v1_ref = V1_BASELINE[ticker]

        # V1
        print("    V1 Baseline...")
        v1_rows, n_v1 = run_v1(ticker_df, ticker)
        m_v1 = _calc_metrics(v1_rows, "V1-Baseline", ticker, n_v1)
        all_metrics.append(m_v1)
        all_rows.extend(v1_rows)

        # Var1
        print("    Var1 Momentum...")
        v1_m_rows = run_var1_momentum(ticker_df, ticker)
        m_v1m = _calc_metrics(v1_m_rows, "Var1-Momentum", ticker, n_v1)
        all_metrics.append(m_v1m)
        all_rows.extend(v1_m_rows)

        # Var2
        print("    Var2 Vol Expansion...")
        v2_rows = run_var2_vol(ticker_df, ticker)
        m_v2 = _calc_metrics(v2_rows, "Var2-VolExpansion", ticker, n_v1)
        all_metrics.append(m_v2)
        all_rows.extend(v2_rows)

        # Var3
        print("    Var3 Short Side...")
        v3_dict = run_var3_short(ticker_df, ticker)
        m_v3s = _calc_metrics(v3_dict["Var3-Short"],    "Var3-Short",    ticker, n_v1)
        m_v3c = _calc_metrics(v3_dict["Var3-Combined"], "Var3-Combined", ticker, n_v1)
        all_metrics.extend([m_v3s, m_v3c])
        all_rows.extend(v3_dict["Var3-Short"])
        all_rows.extend(v3_dict["Var3-Combined"])

        # Var4
        print("    Var4 Monday Hold...")
        v4_rows, v4_stats = run_var4_monday(ticker_df, ticker)
        m_v4 = _calc_metrics(v4_rows, "Var4-MondayHold", ticker, n_v1)
        all_metrics.append(m_v4)
        all_rows.extend(v4_rows)
        mon_stats[ticker] = v4_stats

    # ── Terminal tables ───────────────────────────────────────────────────────
    print_table(all_metrics_btc, "BTC-USD", V1_BASELINE["BTC-USD"])
    print_table(all_metrics_eth, "ETH-USD", V1_BASELINE["ETH-USD"])

    # Monday extension detail
    for ticker in ["BTC-USD", "ETH-USD"]:
        s = mon_stats[ticker]
        print(f"  Var4 {ticker}: {s['n_extended']} extended  "
              f"TARGET_MON={s['n_mon_target']}  BE_MON={s['n_mon_be']}  "
              f"TIME_MON={s['n_mon_time']}  "
              f"net P&L gain vs Sunday=${s['pnl_gain_vs_sunday']:.2f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\nSaving trade logs...")

    csv_cols = ["date", "variation", "ticker", "direction", "exit_type",
                "entry_price", "stop_price", "target_price", "exit_price",
                "units", "gross_pnl", "gross_R",
                "entry_value", "exit_value", "fee", "net_pnl", "net_R"]

    btc_df = pd.DataFrame(all_rows_btc)
    eth_df = pd.DataFrame(all_rows_eth)

    for df, fname in [(btc_df, "signal_improvements_btc.csv"),
                      (eth_df, "signal_improvements_eth.csv")]:
        out_cols = [c for c in csv_cols if c in df.columns]
        path = OUTPUT_DIR / fname
        df[out_cols].to_csv(path, index=False)
        print(f"  {fname} → {path}")

    write_summary(all_metrics_btc, all_metrics_eth, mon_stats, OUTPUT_DIR)
    print("\nDone. Review signal_improvements_summary.txt for full comparison.")


if __name__ == "__main__":
    main()
