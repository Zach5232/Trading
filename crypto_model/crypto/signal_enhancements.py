"""
crypto/signal_enhancements.py
==============================
Two additional backtest filters tested on top of the Var1+Var2+Var4 system.

Section 1 — Funding Rate Filter
  Hypothesis: avoid LONG signals when perpetual futures funding rate is
  highly positive (crowded longs, flush risk).
  Data: Binance perpetual funding rate history (no API key required).
  Paginated from https://fapi.binance.com/fapi/v1/fundingRate

Section 2 — Liquidity Trap Filter
  Hypothesis: reject trades where Friday candle range > 1.5-2x ATR14 AND
  volume > 1.5-2.5x 10-day average (exhaustion breakout signature).

Base system: Var1+Var2+Var4 (from signal_combinations.py logic).
Fees: Kraken Pro 0.26% taker on both legs.
Data: loads from Data/backtest_cache/ohlcv_5yr.csv.

Do NOT import from: backtest_engine, fee_analysis, parameter_sweep,
                    signal_improvements, signal_combinations, daily_system.

Outputs -> Results/crypto_backtest/:
  funding_rate_analysis.csv    — one row per Friday with funding metadata
  funding_rate_summary.txt     — comparison table + recommendations
  liquidity_trap_analysis.csv  — one row per Friday with trap flag
  liquidity_trap_summary.txt   — threshold comparison + recommendations
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df, get_fridays

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ATR_MULT_STOP    = 1.25
R_TARGET         = 2.0
RISK_PCT         = 0.05
STARTING_CAPITAL = 500.0
SLIPPAGE_PCT     = 0.001
KRAKEN_TAKER     = 0.0026

CACHE_PATH         = Path(__file__).parent.parent / "Data" / "backtest_cache" / "ohlcv_5yr.csv"
FUNDING_CACHE_BTC  = Path(__file__).parent.parent / "Data" / "backtest_cache" / "funding_btc.csv"
FUNDING_CACHE_ETH  = Path(__file__).parent.parent / "Data" / "backtest_cache" / "funding_eth.csv"
OUTPUT_DIR         = Path(__file__).parent.parent / "Results" / "crypto_backtest"

# OKX public funding rate history (no auth, accessible from US)
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"
OKX_SYMBOL_MAP  = {"BTC-USD": "BTC-USDT-SWAP", "ETH-USD": "ETH-USDT-SWAP"}
OKX_TIMEOUT     = 10

REGIMES = {
    "2018 Bear":         ("2018-01-01", "2018-12-31"),
    "2019 Recovery":     ("2019-01-01", "2019-12-31"),
    "2020 COVID":        ("2020-01-01", "2020-12-31"),
    "2021 Bull":         ("2021-01-01", "2021-12-31"),
    "2022 Bear":         ("2022-01-01", "2022-12-31"),
    "2023-24 Recovery":  ("2023-01-01", "2024-12-31"),
    "2025-Present":      ("2025-01-01", "2099-12-31"),
}

# Var1+Var2+Var4 baseline net expectancy (from signal_combinations.py output)
COMBO_BASELINE = {
    "BTC-USD": {"net_avg_R": 0.139, "net_pf": 1.50, "net_exp": 0.139, "n_trades": 142},
    "ETH-USD": {"net_avg_R": 0.200, "net_pf": 1.91, "net_exp": 0.200, "n_trades":  96},
}

# Funding regime thresholds (rate in % per 8h period)
FUNDING_LEVELS = [
    (-999.0, -0.01, "NEGATIVE"),
    (-0.01,   0.05, "NEUTRAL"),
    ( 0.05,   0.15, "ELEVATED"),
    ( 0.15,  999.0, "EXTREME"),
]

# Liquidity trap threshold combinations to test
TRAP_COMBOS = [
    (1.5, 1.5), (1.5, 2.0), (1.5, 2.5),
    (2.0, 1.5), (2.0, 2.0), (2.0, 2.5),
]   # (atr_mult, vol_mult)

# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def _exit_long(weekend, entry, stop, target):
    """Simulate exit over weekend bars: stop, target, or time-exit."""
    for _, bar in weekend.iterrows():
        if bar["low"] <= stop:
            return stop, "STOP"
        if bar["high"] >= target:
            return target, "TARGET"
    return weekend.iloc[-1]["close"], "TIME"


def _get_weekend(ticker_df, fri_pos):
    """Return Saturday and Sunday bars immediately after Friday at fri_pos."""
    next_bars = ticker_df.iloc[fri_pos + 1: fri_pos + 3]
    return next_bars[next_bars.index.dayofweek.isin([5, 6])].sort_index()


def _get_monday(ticker_df, fri_pos):
    """Return Monday bar after the weekend, or None if unavailable."""
    try:
        mon = ticker_df.iloc[fri_pos + 3]
        return mon if mon.name.dayofweek == 0 else None
    except IndexError:
        return None


def _fee(entry_px, exit_px, units):
    """Compute round-trip Kraken taker fee."""
    return (entry_px * units + exit_px * units) * KRAKEN_TAKER


def _nt(fri_date, ticker, exit_type="NO_TRADE", extra=None):
    """Build a no-trade row dict."""
    row = {
        "date": fri_date,
        "ticker": ticker,
        "direction": "NO_TRADE",
        "exit_type": exit_type,
        "gross_pnl": 0.0,
        "gross_R": None,
        "entry_value": None,
        "exit_value": None,
        "net_R": None,
        "net_pnl": 0.0,
    }
    if extra:
        row.update(extra)
    return row

# ---------------------------------------------------------------------------
# METRICS CALCULATOR
# ---------------------------------------------------------------------------

def _calc_metrics(rows, label: str, ticker: str) -> dict:
    """
    Compute performance metrics from a list of trade row dicts.

    Filters to completed LONG trades (direction=="LONG", gross_R not None,
    exit_type not in {"NO_DATA", "NO_TRADE"}).

    Returns a dict with keys: label, ticker, n_trades, gross_wr, gross_avg_R,
    gross_pf, gross_exp, net_wr, net_avg_R, net_pf, net_exp, max_dd,
    n_target, n_stop, n_time, breakeven_rate.
    Returns {"label": label, "ticker": ticker, "n_trades": 0} if no trades.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return {"label": label, "ticker": ticker, "n_trades": 0}

    mask = (
        (df["direction"] == "LONG") &
        (df["gross_R"].notna()) &
        (~df["exit_type"].isin(["NO_DATA", "NO_TRADE"]))
    )
    trades = df[mask].copy()

    if trades.empty:
        return {"label": label, "ticker": ticker, "n_trades": 0}

    n = len(trades)

    # Gross metrics
    wins_g  = (trades["gross_R"] > 0).sum()
    gross_wr    = wins_g / n
    gross_avg_R = trades["gross_R"].mean()
    gross_pos   = trades.loc[trades["gross_R"] > 0, "gross_R"].sum()
    gross_neg   = trades.loc[trades["gross_R"] < 0, "gross_R"].abs().sum()
    gross_pf    = gross_pos / gross_neg if gross_neg > 0 else np.inf
    gross_exp   = gross_avg_R

    # Net metrics
    has_net = "net_R" in trades.columns and trades["net_R"].notna().any()
    if has_net:
        net_trades  = trades[trades["net_R"].notna()]
        wins_n      = (net_trades["net_R"] > 0).sum()
        net_wr      = wins_n / len(net_trades)
        net_avg_R   = net_trades["net_R"].mean()
        net_pos     = net_trades.loc[net_trades["net_R"] > 0, "net_R"].sum()
        net_neg     = net_trades.loc[net_trades["net_R"] < 0, "net_R"].abs().sum()
        net_pf      = net_pos / net_neg if net_neg > 0 else np.inf
        net_exp     = net_avg_R
    else:
        net_wr = net_avg_R = net_pf = net_exp = None

    # Max drawdown from gross_pnl equity curve
    equity = STARTING_CAPITAL + trades["gross_pnl"].cumsum()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    max_dd = float(dd.min()) if not dd.empty else 0.0

    # Exit breakdown
    n_target = int((trades["exit_type"] == "TARGET").sum())
    n_stop   = int((trades["exit_type"] == "STOP").sum())
    n_time   = int(trades["exit_type"].isin(["TIME", "TIME_MON"]).sum())

    # Breakeven rate: WR needed for 0 expectancy given avg winner / avg loser
    avg_win  = trades.loc[trades["gross_R"] > 0, "gross_R"].mean() if wins_g > 0 else 0.0
    avg_loss = trades.loc[trades["gross_R"] < 0, "gross_R"].abs().mean() if (n - wins_g) > 0 else 0.0
    if avg_win > 0 and avg_loss > 0:
        breakeven_rate = avg_loss / (avg_win + avg_loss)
    else:
        breakeven_rate = None

    return {
        "label":          label,
        "ticker":         ticker,
        "n_trades":       n,
        "gross_wr":       gross_wr,
        "gross_avg_R":    gross_avg_R,
        "gross_pf":       gross_pf,
        "gross_exp":      gross_exp,
        "net_wr":         net_wr,
        "net_avg_R":      net_avg_R,
        "net_pf":         net_pf,
        "net_exp":        net_exp,
        "max_dd":         max_dd,
        "n_target":       n_target,
        "n_stop":         n_stop,
        "n_time":         n_time,
        "breakeven_rate": breakeven_rate,
    }

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Load OHLCV from cache; download on first run."""
    if CACHE_PATH.exists():
        print(f"  Loading OHLCV from cache: {CACHE_PATH}")
        raw = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    else:
        print("  Cache not found — downloading (one-time)...")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = load_crypto_data(["BTC-USD", "ETH-USD"])
        raw.to_csv(CACHE_PATH, index=False)
    return {
        "BTC-USD": get_ticker_df(raw, "BTC-USD"),
        "ETH-USD": get_ticker_df(raw, "ETH-USD"),
    }

# ===========================================================================
# SECTION 1 — FUNDING RATE FILTER
# ===========================================================================

def _fetch_all_funding(symbol: str) -> pd.DataFrame:
    """
    Paginate OKX public funding rate history API backwards from now.
    OKX endpoint requires no authentication and is accessible from US.

    Pagination: cursor-based using `before` param (fundingTime in ms).
    Each page returns up to 100 records (8h intervals → ~33 days/page).
    Stops when the API returns an empty data list.

    Returns DataFrame with columns: fundingTime (UTC datetime), fundingRate (%)
    """
    all_records = []
    after_ms: int | None = None   # OKX: "after" = older than this fundingTime
    page = 0
    MAX_PAGES = 500
    prev_oldest_ts: int | None = None
    while page < MAX_PAGES:
        page += 1
        params: dict = {"instId": symbol, "limit": 100}
        if after_ms is not None:
            params["after"] = str(after_ms)
        resp = requests.get(OKX_FUNDING_URL, params=params, timeout=OKX_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"OKX API error {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error: {payload.get('msg', payload)}")
        batch = payload.get("data", [])
        if not batch:
            break
        all_records.extend(batch)
        # OKX returns newest-first; oldest record in batch is last element
        oldest_ts = int(batch[-1]["fundingTime"])
        earliest_dt = pd.to_datetime(oldest_ts, unit="ms", utc=True).date()
        print(f"    Page {page}: {len(batch)} records  (earliest: {earliest_dt})")
        # Stop if cursor isn't moving (OKX history exhausted)
        if oldest_ts == prev_oldest_ts:
            print("    Cursor not advancing — history exhausted.")
            break
        prev_oldest_ts = oldest_ts
        if len(batch) < 100:
            break   # reached start of history
        after_ms = oldest_ts - 1
        time.sleep(0.2)

    if not all_records:
        raise RuntimeError(f"No funding records returned for {symbol}")

    df = pd.DataFrame(all_records)[["fundingTime", "fundingRate"]]
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float) * 100   # decimal → %
    df = df.sort_values("fundingTime").drop_duplicates("fundingTime").reset_index(drop=True)
    print(f"    Total: {len(df)} records  "
          f"{df['fundingTime'].iloc[0].date()} → {df['fundingTime'].iloc[-1].date()}")
    return df


def load_funding(ticker: str) -> pd.DataFrame:
    """Load funding history from cache; fetch from OKX on first run."""
    cache = FUNDING_CACHE_BTC if "BTC" in ticker else FUNDING_CACHE_ETH
    symbol = OKX_SYMBOL_MAP[ticker]
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["fundingTime"])
        if df["fundingTime"].dt.tz is None:
            df["fundingTime"] = df["fundingTime"].dt.tz_localize("UTC")
        print(f"  Funding cache: {len(df)} records for {ticker}")
    else:
        print(f"  Fetching funding history for {ticker} ({symbol})...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df = _fetch_all_funding(symbol)
        df.to_csv(cache, index=False)
        print(f"  Saved to {cache}")
    return df


def _classify_funding(rate_pct: float) -> str:
    """Map a funding rate (in %) to a regime label."""
    for lo, hi, label in FUNDING_LEVELS:
        if lo <= rate_pct < hi:
            return label
    return "EXTREME"


def _get_friday_avg_funding(funding_df: pd.DataFrame, fri_date: pd.Timestamp) -> Optional[float]:
    """
    Average funding rate over the 7 days BEFORE fri_date (21 periods at 8h each).
    fri_date should be timezone-naive; we convert to UTC for comparison.
    Returns None if fewer than 3 records found (no data for this period).
    """
    fri_ts   = pd.Timestamp(fri_date).tz_localize("UTC") if fri_date.tzinfo is None else fri_date
    start_ts = fri_ts - pd.Timedelta(days=7)
    mask     = (funding_df["fundingTime"] >= start_ts) & (funding_df["fundingTime"] < fri_ts)
    subset   = funding_df.loc[mask, "fundingRate"]
    return float(subset.mean()) if len(subset) >= 3 else None


def run_var1v2v4_with_funding(ticker_df: pd.DataFrame, ticker: str,
                               funding_df: pd.DataFrame) -> list:
    """
    Full Var1+Var2+Var4 simulation with funding rate metadata attached.

    The simulation logic mirrors signal_combinations.py run_var1_var2_var4:
      Var1 = MA20 filter (above MA20 only)
      Var2 = ATR14 expansion (current ATR14 > prior Friday ATR14)
      Var4 = Monday hold (if TIME exit on weekend, check Monday bar for stop/target)

    For each Friday (traded or not), attaches:
      avg_funding_rate  — 7-day average prior funding rate (None if pre-2019)
      funding_class     — NEGATIVE/NEUTRAL/ELEVATED/EXTREME/NO_DATA

    The funding filter is NOT applied here — rows are tagged for post-hoc analysis.
    Returns list of row dicts.
    """
    fridays = get_fridays(ticker_df)
    rows    = []
    equity  = STARTING_CAPITAL

    for i, (fri_date, fri_row) in enumerate(fridays.iterrows()):
        fri_pos = ticker_df.index.get_loc(fri_date)

        # Look up funding metadata for this Friday
        avg_rate = _get_friday_avg_funding(funding_df, fri_date)
        fund_cls = _classify_funding(avg_rate) if avg_rate is not None else "NO_DATA"
        fund_meta = {"avg_funding_rate": avg_rate, "funding_class": fund_cls}

        # Var1: MA20 filter
        if not bool(fri_row["above_ma20"]):
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_MA20", extra=fund_meta))
            continue

        # Need a prior Friday for Var2 + Var1 momentum
        if i == 0:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_NO_PRIOR", extra=fund_meta))
            continue

        prior_fri_row = fridays.iloc[i - 1]

        # Var2: ATR14 expansion
        if fri_row["atr14"] <= prior_fri_row["atr14"]:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_VOL_CONTRACTION", extra=fund_meta))
            continue

        # Var1 momentum: close > prior Friday close
        if fri_row["close"] <= prior_fri_row["close"]:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_MOMENTUM", extra=fund_meta))
            continue

        # Entry setup
        entry  = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop   = entry - ATR_MULT_STOP * fri_row["atr14"]
        target = entry + R_TARGET * (entry - stop)
        risk   = entry - stop
        dollar_risk = equity * RISK_PCT
        units  = dollar_risk / risk if risk > 0 else 0

        if units <= 0:
            rows.append(_nt(fri_date, ticker, exit_type="NO_UNITS", extra=fund_meta))
            continue

        # Simulate weekend exit
        weekend = _get_weekend(ticker_df, fri_pos)
        if weekend.empty:
            rows.append(_nt(fri_date, ticker, exit_type="NO_DATA", extra=fund_meta))
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)

        # Var4: Monday hold — if TIME exit and profitable, hold to Monday with BE stop
        if exit_type == "TIME" and exit_px > entry:
            monday = _get_monday(ticker_df, fri_pos)
            if monday is not None:
                be_stop = entry
                if monday["low"] <= be_stop:
                    exit_px, exit_type = be_stop, "BE_MON"
                elif monday["high"] >= target:
                    exit_px, exit_type = target, "TARGET_MON"
                else:
                    exit_px, exit_type = monday["close"], "TIME_MON"

        gross_pnl = (exit_px - entry) * units
        gross_R   = (exit_px - entry) / risk if risk > 0 else 0.0
        fee       = _fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / dollar_risk if dollar_risk > 0 else 0.0

        equity += net_pnl

        rows.append({
            "date":             fri_date,
            "ticker":           ticker,
            "direction":        "LONG",
            "exit_type":        exit_type,
            "entry_price":      entry,
            "stop_price":       stop,
            "target_price":     target,
            "exit_price":       exit_px,
            "units":            units,
            "gross_pnl":        gross_pnl,
            "gross_R":          gross_R,
            "entry_value":      entry * units,
            "exit_value":       exit_px * units,
            "fee":              fee,
            "net_pnl":          net_pnl,
            "net_R":            net_R,
            "avg_funding_rate": avg_rate,
            "funding_class":    fund_cls,
        })

    return rows


def run_funding_analysis(ticker_df: pd.DataFrame, ticker: str,
                          funding_df: pd.DataFrame) -> tuple:
    """
    Run Var1+Var2+Var4 with funding metadata and compute filter comparisons.

    Returns:
        (all_rows, metrics_baseline, metrics_conservative, metrics_aggressive,
         regime_by_funding)

    Groups:
      Baseline:     all completed LONG trades
      Conservative: exclude ELEVATED + EXTREME (keep NEGATIVE, NEUTRAL, NO_DATA)
      Aggressive:   exclude EXTREME only (keep NEGATIVE, NEUTRAL, ELEVATED, NO_DATA)

    Note: NO_DATA trades (pre-2019, no funding history) are INCLUDED in both
    filtered sets — we cannot classify them as dangerous, so we do not exclude them.
    This is noted in the summary.

    regime_by_funding: dict mapping funding class -> metrics dict for trades in that class.
    """
    rows = run_var1v2v4_with_funding(ticker_df, ticker, funding_df)

    # Completed trades only
    trades = [r for r in rows
              if r.get("direction") == "LONG"
              and r.get("gross_R") is not None
              and r.get("exit_type") not in ("NO_DATA", "NO_TRADE")]

    m_base = _calc_metrics(trades, "Baseline", ticker)

    # Conservative: NEGATIVE + NEUTRAL + NO_DATA
    keep_cons = {"NEGATIVE", "NEUTRAL", "NO_DATA"}
    trades_cons = [r for r in trades if r.get("funding_class") in keep_cons]
    m_cons = _calc_metrics(trades_cons, "Conservative", ticker)

    # Aggressive: NOT EXTREME
    keep_agg = {"NEGATIVE", "NEUTRAL", "ELEVATED", "NO_DATA"}
    trades_agg = [r for r in trades if r.get("funding_class") in keep_agg]
    m_agg = _calc_metrics(trades_agg, "Aggressive", ticker)

    # Per-regime breakdown
    regime_by_funding = {}
    for cls in ["NEGATIVE", "NEUTRAL", "ELEVATED", "EXTREME", "NO_DATA"]:
        subset = [r for r in trades if r.get("funding_class") == cls]
        regime_by_funding[cls] = _calc_metrics(subset, cls, ticker)

    return rows, m_base, m_cons, m_agg, regime_by_funding


def _fmt_funding_table(ticker: str, m_base: dict, m_cons: dict, m_agg: dict,
                        regime_by_funding: dict, all_rows: list) -> str:
    """Format the funding rate comparison table and answer questions."""
    lines = []
    lines.append("=" * 65)
    lines.append(f"FUNDING RATE FILTER — {ticker}")
    lines.append("=" * 65)

    def safe(val, fmt=".3f"):
        if val is None:
            return "  N/A "
        return format(val, fmt)

    header = f"{'Filter':<18} {'Trades':>7} {'Filtered':>9} {'WR%':>6} {'GrossR':>8} {'NetR':>8} {'NetPF':>7} {'NetExp':>8}"
    lines.append(header)
    lines.append("-" * 65)

    def row_str(m, n_filtered=0):
        if m.get("n_trades", 0) == 0:
            return f"{'N/A':<18} {'0':>7} {n_filtered:>9} {'':>6} {'':>8} {'':>8} {'':>7} {'':>8}"
        wr_pct = (m["gross_wr"] * 100) if m.get("gross_wr") is not None else float("nan")
        return (
            f"{m['label']:<18} {m['n_trades']:>7} {n_filtered:>9}"
            f" {wr_pct:>5.1f}%"
            f" {safe(m.get('gross_avg_R')):>8}"
            f" {safe(m.get('net_avg_R')):>8}"
            f" {safe(m.get('net_pf')):>7}"
            f" {safe(m.get('net_exp')):>8}"
        )

    n_base = m_base.get("n_trades", 0)
    n_cons = m_cons.get("n_trades", 0)
    n_agg  = m_agg.get("n_trades", 0)

    lines.append(row_str(m_base, 0))
    lines.append(row_str(m_cons, n_base - n_cons))
    lines.append(row_str(m_agg,  n_base - n_agg))

    lines.append("")
    lines.append("Funding regime breakdown (Baseline trades):")

    # Count trades per class
    completed = [r for r in all_rows
                 if r.get("direction") == "LONG"
                 and r.get("gross_R") is not None
                 and r.get("exit_type") not in ("NO_DATA", "NO_TRADE")]
    total = len(completed)

    for cls in ["NEGATIVE", "NEUTRAL", "ELEVATED", "EXTREME", "NO_DATA"]:
        rm = regime_by_funding.get(cls, {})
        n_cls = rm.get("n_trades", 0)
        pct = 100.0 * n_cls / total if total > 0 else 0.0
        if cls == "NO_DATA":
            lines.append(f"  {cls:<12} N={n_cls:>3}  (pre-2019 — no funding history)  ({pct:.0f}% of trades)")
        else:
            wr_str  = f"{rm['gross_wr']*100:.0f}%" if rm.get("gross_wr") is not None else " N/A"
            netr_s  = f"{rm['net_avg_R']:.3f}"     if rm.get("net_avg_R") is not None else " N/A"
            lines.append(f"  {cls:<12} N={n_cls:>3}  WR={wr_str:>4}  NetR={netr_s:>6}  ({pct:.0f}% of trades)")

    # --- Answer four questions ---
    lines.append("")
    lines.append("ANALYSIS — Four Questions")
    lines.append("-" * 65)

    base_net = m_base.get("net_avg_R") or 0.0
    cons_net = m_cons.get("net_avg_R") or 0.0
    agg_net  = m_agg.get("net_avg_R")  or 0.0
    delta_cons = cons_net - base_net
    delta_agg  = agg_net  - base_net
    n_filtered_cons = n_base - n_cons
    pct_filtered = (n_filtered_cons / n_base * 100) if n_base > 0 else 0.0

    # Q1
    improved = delta_cons > 0.02
    lines.append(f"Q1. Does the funding rate filter improve net expectancy by >0.02R?")
    lines.append(f"    Conservative vs Baseline delta: {delta_cons:+.3f}R  "
                 f"-> {'YES' if improved else 'NO'}")
    lines.append(f"    Aggressive  vs Baseline delta: {delta_agg:+.3f}R")

    # Q2
    lines.append(f"Q2. Is the improvement statistically significant given reduced sample size?")
    lines.append(f"    Conservative removes {n_filtered_cons} trades ({pct_filtered:.1f}% of baseline).")
    if pct_filtered > 25:
        lines.append("    Reduced sample size is LARGE (>25%) — treat improvement with caution.")
    elif pct_filtered < 10:
        lines.append("    Sample size impact is SMALL (<10%) — improvement is reasonably reliable.")
    else:
        lines.append("    Sample size impact is MODERATE (10-25%) — interpret improvement carefully.")

    # Q3
    lines.append(f"Q3. Should funding rate be used as a hard filter or position sizing modifier?")
    if improved and pct_filtered < 20:
        lines.append("    Recommendation: HARD FILTER — excluding ELEVATED+EXTREME improves edge")
        lines.append("    without sacrificing too many trades.")
    elif improved and pct_filtered >= 20:
        lines.append("    Recommendation: POSITION SIZING MODIFIER — improvement exists but")
        lines.append("    hard filter removes too large a fraction of the trade population.")
        lines.append("    Consider reducing size by 50% when funding is ELEVATED,")
        lines.append("    skip entirely only when EXTREME.")
    else:
        lines.append("    Recommendation: MONITOR ONLY — funding filter does not meaningfully")
        lines.append("    improve net expectancy in this dataset. Collect more data before")
        lines.append("    implementing as a hard filter.")

    # Q4
    lines.append(f"Q4. Recommended implementation:")
    lines.append("    1. Cache Binance funding data weekly (run load_funding() before Friday close).")
    lines.append("    2. Compute 7-day average funding rate at signal time.")
    lines.append("    3. If funding_class == EXTREME (>0.15% per 8h): SKIP the trade entirely.")
    lines.append("    4. If funding_class == ELEVATED (0.05-0.15%): reduce position size by 40%.")
    lines.append("    5. NEGATIVE/NEUTRAL/NO_DATA: trade at full size per the base system.")
    lines.append("    Note: NO_DATA rows (pre-2019) are included in all filter sets — we cannot")
    lines.append("    classify them as dangerous and therefore do not exclude them.")
    lines.append("    Note: equity curve is NOT recomputed for filtered subsets (approximate")
    lines.append("    analysis). Error is small when filtered trades < 20% of total.")

    return "\n".join(lines)

# ===========================================================================
# SECTION 2 — LIQUIDITY TRAP FILTER
# ===========================================================================

def _is_liquidity_trap(fri_row: pd.Series, vol_avg10: float,
                        atr_mult: float, vol_mult: float) -> bool:
    """
    Returns True if BOTH conditions are simultaneously true:
      1. Candle range (high - low) > atr_mult x ATR14
      2. Volume > vol_mult x 10-day avg volume
    Requires fri_row to have columns: high, low, atr14, volume.
    """
    candle_range = fri_row["high"] - fri_row["low"]
    range_flag   = candle_range > atr_mult * fri_row["atr14"]
    vol_flag     = fri_row["volume"] > vol_mult * vol_avg10
    return bool(range_flag and vol_flag)


def run_var1v2v4_base(ticker_df: pd.DataFrame, ticker: str) -> list:
    """
    Standard Var1+Var2+Var4 simulation with candle range and volume metadata.

    Pre-computes vol_avg10 = 10-day rolling average volume shifted by 1
    (prior 10 bars, no lookahead bias).

    Each completed LONG trade row includes:
      candle_range, range_vs_atr, vol_avg10, vol_vs_avg
    NO_TRADE rows include these fields as None.

    Returns list of row dicts.
    """
    # Pre-compute vol_avg10 with 1-period shift to avoid lookahead bias
    vol_avg10_series = ticker_df["volume"].rolling(10).mean().shift(1)

    fridays = get_fridays(ticker_df)
    rows    = []
    equity  = STARTING_CAPITAL

    for i, (fri_date, fri_row) in enumerate(fridays.iterrows()):
        fri_pos   = ticker_df.index.get_loc(fri_date)
        vol_avg10 = vol_avg10_series.get(fri_date, float("nan"))

        trap_meta = {
            "candle_range": None,
            "range_vs_atr": None,
            "vol_avg10":    None,
            "vol_vs_avg":   None,
        }

        # Var1: MA20 filter
        if not bool(fri_row["above_ma20"]):
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_MA20", extra=trap_meta))
            continue

        # Need a prior Friday for Var2 + Var1 momentum
        if i == 0:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_NO_PRIOR", extra=trap_meta))
            continue

        prior_fri_row = fridays.iloc[i - 1]

        # Var2: ATR14 expansion
        if fri_row["atr14"] <= prior_fri_row["atr14"]:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_VOL_CONTRACTION", extra=trap_meta))
            continue

        # Var1 momentum: close > prior Friday close
        if fri_row["close"] <= prior_fri_row["close"]:
            rows.append(_nt(fri_date, ticker, exit_type="FILTERED_MOMENTUM", extra=trap_meta))
            continue

        # Compute trap metadata for this Friday
        candle_range = fri_row["high"] - fri_row["low"]
        atr14_val    = fri_row["atr14"]
        range_vs_atr = candle_range / atr14_val if atr14_val > 0 else None
        vol_val      = fri_row["volume"]
        vol_vs_avg   = vol_val / vol_avg10 if (vol_avg10 and vol_avg10 > 0) else None

        # Entry setup
        entry  = fri_row["close"] * (1 + SLIPPAGE_PCT)
        stop   = entry - ATR_MULT_STOP * atr14_val
        target = entry + R_TARGET * (entry - stop)
        risk   = entry - stop
        dollar_risk = equity * RISK_PCT
        units  = dollar_risk / risk if risk > 0 else 0

        trap_with_data = {
            "candle_range": candle_range,
            "range_vs_atr": range_vs_atr,
            "vol_avg10":    float(vol_avg10) if pd.notna(vol_avg10) else None,
            "vol_vs_avg":   vol_vs_avg,
        }

        if units <= 0:
            rows.append(_nt(fri_date, ticker, exit_type="NO_UNITS", extra=trap_with_data))
            continue

        # Simulate weekend exit
        weekend = _get_weekend(ticker_df, fri_pos)
        if weekend.empty:
            rows.append(_nt(fri_date, ticker, exit_type="NO_DATA", extra=trap_with_data))
            continue

        exit_px, exit_type = _exit_long(weekend, entry, stop, target)

        # Var4: Monday hold — if TIME exit and profitable, hold to Monday with BE stop
        if exit_type == "TIME" and exit_px > entry:
            monday = _get_monday(ticker_df, fri_pos)
            if monday is not None:
                be_stop = entry
                if monday["low"] <= be_stop:
                    exit_px, exit_type = be_stop, "BE_MON"
                elif monday["high"] >= target:
                    exit_px, exit_type = target, "TARGET_MON"
                else:
                    exit_px, exit_type = monday["close"], "TIME_MON"

        gross_pnl = (exit_px - entry) * units
        gross_R   = (exit_px - entry) / risk if risk > 0 else 0.0
        fee       = _fee(entry, exit_px, units)
        net_pnl   = gross_pnl - fee
        net_R     = net_pnl / dollar_risk if dollar_risk > 0 else 0.0

        equity += net_pnl

        rows.append({
            "date":         fri_date,
            "ticker":       ticker,
            "direction":    "LONG",
            "exit_type":    exit_type,
            "entry_price":  entry,
            "stop_price":   stop,
            "target_price": target,
            "exit_price":   exit_px,
            "units":        units,
            "gross_pnl":    gross_pnl,
            "gross_R":      gross_R,
            "entry_value":  entry * units,
            "exit_value":   exit_px * units,
            "fee":          fee,
            "net_pnl":      net_pnl,
            "net_R":        net_R,
            "candle_range": candle_range,
            "range_vs_atr": range_vs_atr,
            "vol_avg10":    float(vol_avg10) if pd.notna(vol_avg10) else None,
            "vol_vs_avg":   vol_vs_avg,
        })

    return rows


def run_trap_analysis(ticker_df: pd.DataFrame, ticker: str) -> tuple:
    """
    Run Var1+Var2+Var4 base simulation and analyse liquidity trap filter
    for all 6 threshold combinations in TRAP_COMBOS.

    For each combination (atr_mult, vol_mult):
      - Tags each completed LONG trade as trap True/False
      - Computes metrics for: all trades (baseline), non-trap trades only
      - Computes n_trap, pct_trap, wr_trap, avg_R_trap (gross/net),
        wr_nontrap, avg_R_nontrap (gross/net), net_exp_improvement

    Returns (base_rows, list_of_combo_results)
    where each combo_result is a dict.
    """
    base_rows = run_var1v2v4_base(ticker_df, ticker)

    completed = [r for r in base_rows
                 if r.get("direction") == "LONG"
                 and r.get("gross_R") is not None
                 and r.get("exit_type") not in ("NO_DATA", "NO_TRADE")]

    combo_results = []
    for atr_mult, vol_mult in TRAP_COMBOS:
        # Tag each completed trade
        trap_flags = []
        for r in completed:
            if (r.get("candle_range") is not None
                    and r.get("range_vs_atr") is not None
                    and r.get("vol_vs_avg") is not None
                    and r.get("vol_avg10") is not None):
                is_trap = (
                    r["range_vs_atr"] > atr_mult
                    and r["vol_vs_avg"] > vol_mult
                )
            else:
                is_trap = False
            trap_flags.append(is_trap)

        trap_trades    = [r for r, t in zip(completed, trap_flags) if t]
        nontrap_trades = [r for r, t in zip(completed, trap_flags) if not t]

        n_total = len(completed)
        n_trap  = len(trap_trades)
        pct_trap = 100.0 * n_trap / n_total if n_total > 0 else 0.0

        # Metrics for trap trades
        m_trap = _calc_metrics(trap_trades, "Trap", ticker) if trap_trades else {}
        # Metrics for non-trap trades
        m_nontrap = _calc_metrics(nontrap_trades, "NonTrap", ticker) if nontrap_trades else {}
        # Baseline
        m_all = _calc_metrics(completed, "All", ticker)

        base_net_exp = m_all.get("net_exp") or 0.0
        nont_net_exp = m_nontrap.get("net_exp") or 0.0
        improvement  = nont_net_exp - base_net_exp

        combo_results.append({
            "atr_mult":       atr_mult,
            "vol_mult":       vol_mult,
            "n_total":        n_total,
            "n_trap":         n_trap,
            "pct_trap":       pct_trap,
            "m_trap":         m_trap,
            "m_nontrap":      m_nontrap,
            "m_all":          m_all,
            "net_exp_delta":  improvement,
            "trap_flags":     trap_flags,
        })

    return base_rows, combo_results


def _fmt_trap_table(ticker: str, base_rows: list, combo_results: list) -> str:
    """Format the liquidity trap comparison table and answer questions."""
    lines = []
    lines.append("=" * 75)
    lines.append(f"LIQUIDITY TRAP FILTER — {ticker}")
    lines.append("=" * 75)

    header = (
        f"{'ATR_mult':>8} {'Vol_mult':>8} {'N_trap':>7} {'Pct%':>5} "
        f"{'Trap_WR%':>9} {'Trap_NetR':>10} {'NonTrap_NetR':>13} {'NetExp_d':>9}"
    )
    lines.append(header)
    lines.append("-" * 75)

    def safe(val, fmt=".3f"):
        if val is None:
            return "  N/A  "
        return format(val, fmt)

    best_delta = -999.0
    best_combo = None

    for cr in combo_results:
        am   = cr["atr_mult"]
        vm   = cr["vol_mult"]
        nt   = cr["n_trap"]
        pct  = cr["pct_trap"]
        mt   = cr["m_trap"]
        mnt  = cr["m_nontrap"]
        delta = cr["net_exp_delta"]

        trap_wr_pct  = mt.get("gross_wr", 0) * 100 if mt.get("gross_wr") is not None else float("nan")
        trap_netr    = mt.get("net_avg_R")
        nontrap_netr = mnt.get("net_avg_R")

        lines.append(
            f"  {am:.1f}x    {vm:.1f}x    {nt:>6} {pct:>5.1f}%"
            f" {trap_wr_pct:>8.1f}%"
            f" {safe(trap_netr):>10}"
            f" {safe(nontrap_netr):>13}"
            f" {delta:>+9.3f}"
        )

        if delta > best_delta:
            best_delta = delta
            best_combo = cr

    # --- Answer three questions ---
    lines.append("")
    lines.append("ANALYSIS — Three Questions")
    lines.append("-" * 75)

    # Q1
    lines.append("Q1. What threshold combination produces the best risk-adjusted improvement?")
    if best_combo:
        lines.append(
            f"    Best: ATR_mult={best_combo['atr_mult']:.1f}x, "
            f"Vol_mult={best_combo['vol_mult']:.1f}x  "
            f"(NetExp delta = {best_combo['net_exp_delta']:+.3f}R, "
            f"removes {best_combo['n_trap']} trades / {best_combo['pct_trap']:.1f}%)"
        )
    else:
        lines.append("    Insufficient data to determine best combination.")

    # Q2
    lines.append("Q2. Are liquidity trap trades meaningfully worse than baseline (is the trap edge real)?")
    if best_combo:
        mt   = best_combo["m_trap"]
        mall = best_combo["m_all"]
        trap_netr  = mt.get("net_avg_R")
        all_netr   = mall.get("net_avg_R")
        if trap_netr is not None and all_netr is not None:
            diff = trap_netr - all_netr
            if diff < -0.05:
                lines.append(f"    YES — trap trades underperform baseline by {diff:.3f}R net.")
                lines.append("    The exhaustion breakout signature appears to identify weaker setups.")
            elif diff < 0:
                lines.append(f"    MARGINALLY — trap trades underperform by only {diff:.3f}R net.")
                lines.append("    Effect is small; interpret with caution given sample size.")
            else:
                lines.append(f"    NO — trap trades do not underperform baseline (delta = {diff:.3f}R).")
                lines.append("    Liquidity trap signature has no demonstrated edge in this dataset.")
        else:
            lines.append("    Insufficient data for comparison.")

    # Q3
    lines.append("Q3. Should this be added as a hard filter? Threshold recommendation.")
    if best_combo and best_combo["net_exp_delta"] > 0.02:
        am = best_combo["atr_mult"]
        vm = best_combo["vol_mult"]
        pct_f = best_combo["pct_trap"]
        if pct_f < 20:
            lines.append(
                f"    YES — recommended hard filter: skip if range > {am:.1f}x ATR14 "
                f"AND volume > {vm:.1f}x 10-day avg."
            )
            lines.append(f"    This removes {pct_f:.1f}% of trades while improving net expectancy by "
                         f"{best_combo['net_exp_delta']:+.3f}R.")
        else:
            lines.append(
                f"    CONDITIONAL — improvement exists but {pct_f:.1f}% of trades are removed."
            )
            lines.append(f"    Consider a position size reduction (50%) rather than a hard skip.")
            lines.append(f"    Preferred threshold if using as a filter: ATR {am:.1f}x, Vol {vm:.1f}x.")
    else:
        lines.append("    NO — filter does not reliably improve net expectancy by >0.02R.")
        lines.append("    Continue monitoring; revisit after 50+ additional Fridays of live data.")
        lines.append("    If implemented, use the most permissive threshold (ATR 2.0x, Vol 2.5x)")
        lines.append("    to minimise trade attrition while still flagging extreme exhaustion candles.")

    return "\n".join(lines)

# ===========================================================================
# CSV BUILDERS
# ===========================================================================

def _build_funding_csv(all_rows: list) -> pd.DataFrame:
    """Select and order columns for funding_rate_analysis.csv."""
    cols = [
        "date", "ticker", "direction", "exit_type",
        "avg_funding_rate", "funding_class",
        "gross_R", "net_R", "gross_pnl", "net_pnl",
        "entry_price", "exit_price", "units",
    ]
    df = pd.DataFrame(all_rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _build_trap_csv(all_rows: list, all_combo_results_by_ticker: dict) -> pd.DataFrame:
    """
    Build liquidity_trap_analysis.csv with is_trap_* columns for all 6 combos.

    all_combo_results_by_ticker: {ticker: (base_rows, combo_results)}
    """
    rows_out = []
    for ticker, (base_rows, combo_results) in all_combo_results_by_ticker.items():
        completed_indices = []
        completed_rows    = []
        for idx, r in enumerate(base_rows):
            if (r.get("direction") == "LONG"
                    and r.get("gross_R") is not None
                    and r.get("exit_type") not in ("NO_DATA", "NO_TRADE")):
                completed_indices.append(idx)
                completed_rows.append(r)

        # Build a dict: index -> {combo_key -> bool}
        trap_by_idx: dict = {idx: {} for idx in completed_indices}
        for cr in combo_results:
            am, vm = cr["atr_mult"], cr["vol_mult"]
            key = f"is_trap_{am:.1f}_{vm:.1f}".replace(".0", "")
            # The trap_flags list aligns with completed_rows
            for i, (idx, flag) in enumerate(zip(completed_indices, cr["trap_flags"])):
                trap_by_idx[idx][key] = flag

        # Build output rows for ALL rows (not just completed)
        combo_keys = [
            f"is_trap_{am:.1f}_{vm:.1f}".replace(".0", "")
            for am, vm in TRAP_COMBOS
        ]

        ci_set = set(completed_indices)
        for idx, r in enumerate(base_rows):
            row_out = {
                "date":       r.get("date"),
                "ticker":     r.get("ticker"),
                "direction":  r.get("direction"),
                "exit_type":  r.get("exit_type"),
                "candle_range": r.get("candle_range"),
                "range_vs_atr": r.get("range_vs_atr"),
                "vol_vs_avg":   r.get("vol_vs_avg"),
                "gross_R":    r.get("gross_R"),
                "net_R":      r.get("net_R"),
                "gross_pnl":  r.get("gross_pnl"),
                "net_pnl":    r.get("net_pnl"),
            }
            if idx in ci_set:
                for key in combo_keys:
                    row_out[key] = trap_by_idx[idx].get(key, False)
            else:
                for key in combo_keys:
                    row_out[key] = None
            rows_out.append(row_out)

    base_cols = [
        "date", "ticker", "direction", "exit_type",
        "candle_range", "range_vs_atr", "vol_vs_avg",
    ]
    trap_cols = [
        f"is_trap_{am:.1f}_{vm:.1f}".replace(".0", "")
        for am, vm in TRAP_COMBOS
    ]
    result_cols = ["gross_R", "net_R", "gross_pnl", "net_pnl"]

    df = pd.DataFrame(rows_out)
    all_cols = base_cols + trap_cols + result_cols
    for c in all_cols:
        if c not in df.columns:
            df[c] = None
    return df[all_cols]

# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Signal Enhancements — Funding Rate + Liquidity Trap Filters")
    print("=" * 65)

    # Load OHLCV
    print("\nLoading OHLCV data...")
    data = load_data()

    # -----------------------------------------------------------------------
    # SECTION 1: FUNDING RATE FILTER
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SECTION 1 — FUNDING RATE FILTER")
    print("=" * 65)

    all_funding_rows   = []
    funding_summaries  = []

    for ticker in ["BTC-USD", "ETH-USD"]:
        print(f"\n  Loading funding data for {ticker}...")
        try:
            funding_df = load_funding(ticker)
        except Exception as exc:
            print(f"  ERROR loading funding for {ticker}: {exc}")
            print("  Skipping funding analysis for this ticker.")
            continue

        print(f"  Running Var1+Var2+Var4 + funding analysis for {ticker}...")
        rows, m_base, m_cons, m_agg, regime_by_funding = run_funding_analysis(
            data[ticker], ticker, funding_df
        )
        all_funding_rows.extend(rows)

        table_str = _fmt_funding_table(
            ticker, m_base, m_cons, m_agg, regime_by_funding, rows
        )
        print("\n" + table_str)
        funding_summaries.append(table_str)

    # Save funding CSV
    if all_funding_rows:
        funding_csv_path = OUTPUT_DIR / "funding_rate_analysis.csv"
        _build_funding_csv(all_funding_rows).to_csv(funding_csv_path, index=False)
        print(f"\n  Saved: {funding_csv_path}")

        funding_txt_path = OUTPUT_DIR / "funding_rate_summary.txt"
        with open(funding_txt_path, "w") as f:
            f.write("Signal Enhancements — Section 1: Funding Rate Filter\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("\n\n".join(funding_summaries))
        print(f"  Saved: {funding_txt_path}")

    # -----------------------------------------------------------------------
    # SECTION 2: LIQUIDITY TRAP FILTER
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SECTION 2 — LIQUIDITY TRAP FILTER")
    print("=" * 65)

    all_combo_results_by_ticker: dict = {}
    trap_summaries = []

    for ticker in ["BTC-USD", "ETH-USD"]:
        print(f"\n  Running liquidity trap analysis for {ticker}...")
        base_rows, combo_results = run_trap_analysis(data[ticker], ticker)
        all_combo_results_by_ticker[ticker] = (base_rows, combo_results)

        table_str = _fmt_trap_table(ticker, base_rows, combo_results)
        print("\n" + table_str)
        trap_summaries.append(table_str)

    # Save trap CSV
    if all_combo_results_by_ticker:
        trap_csv_path = OUTPUT_DIR / "liquidity_trap_analysis.csv"
        _build_trap_csv([], all_combo_results_by_ticker).to_csv(trap_csv_path, index=False)
        print(f"\n  Saved: {trap_csv_path}")

        trap_txt_path = OUTPUT_DIR / "liquidity_trap_summary.txt"
        with open(trap_txt_path, "w") as f:
            f.write("Signal Enhancements — Section 2: Liquidity Trap Filter\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("\n\n".join(trap_summaries))
        print(f"  Saved: {trap_txt_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
