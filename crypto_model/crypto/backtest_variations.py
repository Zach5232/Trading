"""
crypto/backtest_variations.py
==============================
Two filter variations against the V1 baseline backtest.

Variation A — Fear/Greed Filter:
  Skip weekends where the Crypto Fear & Greed Index is in
  Extreme Fear (<= 25) or Extreme Greed (>= 75) on Friday close.
  Fallback if API unavailable: skip when BTC 7-day return > +15% or < -15%.

Variation B — ETH Confirmation Filter:
  Only take the BTC LONG signal if ETH is ALSO above its own MA20.

Outputs saved to Results/crypto_backtest/:
  variation_a_trades.csv
  variation_b_trades.csv
  variation_comparison.txt
"""

import sys
import warnings
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays
from backtest_engine import (
    _simulate_exit,
    _calc_metrics,
    SLIPPAGE_PCT,
    ATR_MULT_STOP,
    R_TARGET,
    STARTING_CAPITAL,
    RISK_PCT_PER_TRADE,
    TAX_RATE,
    OUTPUT_DIR,
)

FNG_API_URL = "https://api.alternative.me/fng/?limit=365&format=json"
MOMENTUM_EXTREME_PCT = 0.15   # ±15% 7-day return as fallback threshold


# ── Fear & Greed loader ────────────────────────────────────────────────────

def _fetch_fng() -> pd.Series | None:
    """
    Fetch Crypto Fear & Greed Index.
    Returns a Series indexed by date, values = integer 0-100.
    Returns None on failure.
    """
    try:
        resp = requests.get(FNG_API_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", [])
        if not records:
            return None

        entries = {}
        for item in records:
            ts  = int(item["timestamp"])
            val = int(item["value"])
            dt  = pd.Timestamp(ts, unit="s").normalize()
            entries[dt] = val

        series = pd.Series(entries).sort_index()
        print(f"  Fear & Greed: {len(series)} days loaded "
              f"({series.index[0].date()} → {series.index[-1].date()})")
        return series

    except Exception as exc:
        print(f"  WARNING: Fear & Greed API unavailable ({exc}). Using momentum fallback.")
        return None


def _build_fng_fallback(btc: pd.DataFrame) -> pd.Series:
    """
    Proxy Fear & Greed using 7-day BTC return.
    Returns a Series indexed by date, values in [-1, 1] (raw return fraction).
    """
    ret_7d = btc["close"].pct_change(7)
    return ret_7d


# ── Shared trade simulator ─────────────────────────────────────────────────

def _run_filtered_backtest(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    filter_fn,         # callable(fri_date, fri_row) → bool  (True = take trade)
    label: str,
    starting_capital: float = STARTING_CAPITAL,
    risk_pct: float = RISK_PCT_PER_TRADE,
) -> pd.DataFrame:
    """
    Run the base backtest with an additional filter applied on top of MA20.

    filter_fn(fri_date, fri_row) returns True if the trade should be taken.
    """
    fridays = get_fridays(btc)
    trade_rows = []
    equity = starting_capital

    for fri_date, fri_row in fridays.iterrows():
        above_ma = bool(fri_row["above_ma20"])

        if not above_ma:
            trade_rows.append(
                _no_trade_row(fri_date, "Below MA20 — no trade")
            )
            continue

        # Additional variation filter
        if not filter_fn(fri_date, fri_row):
            trade_rows.append(
                _no_trade_row(fri_date, f"Filtered out by {label}")
            )
            continue

        entry  = fri_row["close"] * (1 + SLIPPAGE_PCT)
        atr    = fri_row["atr14"]
        stop   = entry - ATR_MULT_STOP * atr
        target = entry + R_TARGET * (entry - stop)

        risk_per_unit = entry - stop
        risk_dollars  = equity * risk_pct
        units         = risk_dollars / risk_per_unit

        fri_pos   = btc.index.get_loc(fri_date)
        next_bars = btc.iloc[fri_pos + 1 : fri_pos + 3]
        weekend   = next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()

        if weekend.empty:
            trade_rows.append(
                {
                    "date": fri_date, "instrument": "BTC-USD",
                    "entry_price": round(entry, 2), "stop_price": round(stop, 2),
                    "target_price": round(target, 2), "exit_price": None,
                    "direction": "LONG", "exit_type": "NO_DATA",
                    "R_multiple": None, "profit_loss": 0.0,
                    "btc_above_20ma": "yes", "weekend_notes": "No weekend bars",
                }
            )
            continue

        exit_price, exit_type, notes = _simulate_exit(weekend, entry, stop, target)
        r_multiple  = (exit_price - entry) / risk_per_unit
        profit_loss = units * (exit_price - entry)
        equity      = max(equity + profit_loss, 0.01)

        trade_rows.append(
            {
                "date": fri_date, "instrument": "BTC-USD",
                "entry_price":  round(entry, 2),
                "stop_price":   round(stop, 2),
                "target_price": round(target, 2),
                "exit_price":   round(exit_price, 2),
                "direction":    "LONG",
                "exit_type":    exit_type,
                "R_multiple":   round(r_multiple, 3),
                "profit_loss":  round(profit_loss, 2),
                "btc_above_20ma": "yes",
                "weekend_notes": notes,
            }
        )

    df = pd.DataFrame(trade_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _no_trade_row(fri_date, notes: str) -> dict:
    return {
        "date": fri_date, "instrument": "BTC-USD",
        "entry_price": None, "stop_price": None,
        "target_price": None, "exit_price": None,
        "direction": "NO_TRADE", "exit_type": "NO_TRADE",
        "R_multiple": None, "profit_loss": 0.0,
        "btc_above_20ma": "no", "weekend_notes": notes,
    }


# ── Variation A — Fear/Greed ───────────────────────────────────────────────

def run_variation_a(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    print("\nVariation A: Fear/Greed Filter")
    fng = _fetch_fng()

    if fng is not None:
        def filter_fn(fri_date, fri_row):
            # Find closest prior FNG reading
            prior = fng[fng.index <= fri_date]
            if prior.empty:
                return True  # no data → allow trade
            val = int(prior.iloc[-1])
            if val <= 25:
                print(f"    {fri_date.date()}: FNG={val} (Extreme Fear) → skipped")
                return False
            if val >= 75:
                print(f"    {fri_date.date()}: FNG={val} (Extreme Greed) → skipped")
                return False
            return True
    else:
        fallback = _build_fng_fallback(btc)

        def filter_fn(fri_date, fri_row):
            try:
                ret = fallback.loc[fri_date]
            except KeyError:
                return True
            if abs(ret) > MOMENTUM_EXTREME_PCT:
                direction = "up" if ret > 0 else "down"
                print(f"    {fri_date.date()}: 7d-return={ret:.1%} (momentum extreme {direction}) → skipped")
                return False
            return True

    return _run_filtered_backtest(btc, eth, filter_fn, "FearGreed", starting_capital)


# ── Variation B — ETH Confirmation ────────────────────────────────────────

def run_variation_b(
    btc: pd.DataFrame,
    eth: pd.DataFrame,
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    print("\nVariation B: ETH Confirmation Filter (ETH also above MA20)")

    def filter_fn(fri_date, fri_row):
        # Find ETH's Friday row
        eth_fridays = eth[eth.index.dayofweek == 4]
        if fri_date not in eth_fridays.index:
            # Find closest ETH Friday on or before signal date
            prior = eth_fridays[eth_fridays.index <= fri_date]
            if prior.empty:
                return False
            eth_row = prior.iloc[-1]
        else:
            eth_row = eth_fridays.loc[fri_date]

        eth_above = bool(eth_row["above_ma20"])
        if not eth_above:
            print(f"    {fri_date.date()}: ETH below MA20 → skipped")
        return eth_above

    return _run_filtered_backtest(btc, eth, filter_fn, "ETH_Confirmation", starting_capital)


# ── Comparison report ──────────────────────────────────────────────────────

def _metrics_block(label: str, m: dict, n_filtered: int = 0) -> list[str]:
    if not m:
        return [f"  {label}: No completed trades."]

    lines = [
        f"  {label}",
        f"    Trades completed : {m['n_trades']}",
        f"    Trades filtered  : {n_filtered}",
        f"    Win Rate         : {m['win_rate']}%",
        f"    Avg R            : {m['avg_R']}",
        f"    Profit Factor    : {m['profit_factor']}",
        f"    Expectancy R     : {m['expectancy_R']}",
        f"    Max Drawdown     : {m['max_drawdown']}%",
        f"    ROI Pre-Tax      : {m['roi_pre_tax']}%",
        f"    ROI Post-Tax 32% : {m['roi_post_tax_estimated']}%",
        f"    Exit TARGET      : {m['exit_TARGET']}",
        f"    Exit STOP        : {m['exit_STOP']}",
        f"    Exit TIME        : {m['exit_TIME']}",
    ]
    return lines


def _write_comparison(
    v1_trades: pd.DataFrame,
    va_trades: pd.DataFrame,
    vb_trades: pd.DataFrame,
    out_dir: Path,
) -> None:
    m_v1 = _calc_metrics(v1_trades)
    m_va = _calc_metrics(va_trades)
    m_vb = _calc_metrics(vb_trades)

    n1 = len(v1_trades[v1_trades["direction"] == "LONG"])
    na = len(va_trades[va_trades["direction"] == "LONG"])
    nb = len(vb_trades[vb_trades["direction"] == "LONG"])

    lines = [
        "=" * 65,
        "VARIATION COMPARISON — BTC Weekend MA20 System",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
    ]

    lines += _metrics_block("V1 Baseline (MA20 only)", m_v1)
    lines += [""]
    lines += _metrics_block("Var A: Fear/Greed Filter", m_va, n1 - na)
    lines += [""]
    lines += _metrics_block("Var B: ETH Confirmation", m_vb, n1 - nb)
    lines += [""]

    # ── Delta summary ──────────────────────────────────────────────────────
    lines += ["DELTA vs V1 BASELINE", "-" * 40]
    for label, m in [("Var A", m_va), ("Var B", m_vb)]:
        if m and m_v1:
            dwr  = round(m["win_rate"]   - m_v1["win_rate"],  1)
            davg = round(m["avg_R"]      - m_v1["avg_R"],     3)
            dpf  = round(m["profit_factor"] - m_v1["profit_factor"], 2)
            droi = round(m["roi_post_tax_estimated"] - m_v1["roi_post_tax_estimated"], 1)
            lines.append(
                f"  {label:<8}  ΔWin%={dwr:+.1f}  ΔAvgR={davg:+.3f}  "
                f"ΔPF={dpf:+.2f}  ΔROI(post-tax)={droi:+.1f}%"
            )

    # ── Recommendation ─────────────────────────────────────────────────────
    lines += [
        "",
        "RECOMMENDATION",
        "-" * 40,
    ]

    best = "V1 Baseline"
    best_exp = m_v1.get("expectancy_R", -99) if m_v1 else -99
    for label, m in [("Var A", m_va), ("Var B", m_vb)]:
        if m and m.get("expectancy_R", -99) > best_exp:
            best_exp = m["expectancy_R"]
            best = label

    lines += [
        f"  Highest expectancy R: {best} ({best_exp:.3f}R)",
        "",
        "  Notes:",
        "  - Filters reduce trade count; verify statistical significance (N >= 30).",
        "  - Fear/Greed API data only goes back ~1 year; fallback uses momentum proxy.",
        "  - ETH confirmation most useful during high-correlation regimes.",
        "  - Run 8+ weeks paper trading on chosen variant before live deployment.",
        "=" * 65,
    ]

    comp_path = out_dir / "variation_comparison.txt"
    comp_path.write_text("\n".join(lines))
    print(f"  Comparison  → {comp_path}")


# ── CLI entry ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("BTC Backtest Variations — Phase 1")
    print("=" * 65)

    # Load data once
    data = load_crypto_data(["BTC-USD", "ETH-USD"])
    btc  = get_ticker_df(data, "BTC-USD")
    eth  = get_ticker_df(data, "ETH-USD")

    # Run V1 baseline (re-use engine's backtest but capture trades)
    from backtest_engine import run_backtest
    v1_trades, _, _, _ = run_backtest(data)

    # Run variations
    va_trades = run_variation_a(btc, eth)
    vb_trades = run_variation_b(btc, eth)

    # Save CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    va_path = OUTPUT_DIR / "variation_a_trades.csv"
    vb_path = OUTPUT_DIR / "variation_b_trades.csv"
    va_trades.to_csv(va_path, index=False)
    vb_trades.to_csv(vb_path, index=False)
    print(f"\n  Var A trades → {va_path}")
    print(f"  Var B trades → {vb_path}")

    # Write comparison
    _write_comparison(v1_trades, va_trades, vb_trades, OUTPUT_DIR)
    print("\nDone. Review variation_comparison.txt for recommendations.")


if __name__ == "__main__":
    main()
