"""
main.py

Entry point for the Stock Swing Bet Model.

Daily flow:
    1) Build / load S&P 500 universe
    2) Download recent OHLCV data
    3) Compute signals (momentum, RSI, ATR, liquidity filters)
    4) Select top trade candidates
    5) Size positions based on account risk rules
    6) Save a daily candidates CSV (and optionally append to execution_log.csv)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd

from data_loader import (
    get_sp500_tickers,
    fetch_price_data,
    get_sector_map,
    get_company_map,
    SECTOR_ETF_MAP,
    fetch_sector_etf_return_3w,
    fetch_next_week_earnings,
)
from model_logic import (
    add_signals,
    select_top_candidates,
    classify_current_regime,
    compute_atr,
    MIN_PRICE,
    MIN_AVG_VOL,
    MAX_STOP_DIST_PCT,
    MIN_RET_3W,
    MIN_VOL_SURGE,
    MIN_DIST_52W,
    EXCLUDED_SECTOR,
    ATR_STOP_MULTIPLE,
)
from position_sizing import (
    RiskConfig,
    compute_stop_price_long,
    compute_position_size_long,
)
from trade_logger import append_execution_log


# ------------------- CONFIG -------------------

# Approximate account equity (update this as needed)
ACCOUNT_EQUITY = 10_000.0

# Risk settings (can be tuned later). max_dollar_risk_per_trade is a
# placeholder here — the V2 daily pipeline overrides it per-candidate with
# RISK_NORMAL / RISK_BOOSTED (see run_daily_model). Backfill mode (which
# can't compute filters_passed) uses this flat default.
RISK_CONFIG = RiskConfig(
    account_equity=ACCOUNT_EQUITY,
    max_risk_per_trade_pct=0.005,      # 0.5% of equity
    max_dollar_risk_per_trade=50.0,    # $50 max per trade (backfill default)
    max_notional_per_trade_pct=0.10,   # 10% of equity
)

# Signal / candidate settings
TOP_N_CANDIDATES = 3      # V2: top 3 (was top 5 in V1)
# V2 needs a full year of history for the 52-week-high filter (F5); V1 only
# needed 6mo. This is a data-volume change, not a change to the download
# mechanism itself.
PRICE_HISTORY_PERIOD = "1y"    # used by fetch_price_data
PRICE_HISTORY_INTERVAL = "1d"
FORCE_REFRESH_TICKERS = True

# V2 week-level / candidate-level gates
# Recalibrated for full-universe percentile ranking (was 0.05, calibrated
# for the old eligible-subset-only ranking where scores spread 0-1 among a
# small pool; top candidates now cluster tightly near the top of the full
# 500+ ticker distribution, so the old threshold almost never fires).
SCORE_GAP_THRESHOLD = 0.01
RISK_NORMAL = 35.0
RISK_BOOSTED = 70.0
FILTERS_BOOSTED_THRESHOLD = 8   # filters_passed >= this -> 2x risk sizing
ETF_MIN_3W_RETURN = 0.01

MAX_ENTRY_PREMIUM_PCT = 0.006  # 0.6% above signal price
                                # Beyond this the edge disappears
                                # Backtested: PF drops below 1.0
                                # at 0.6% premium (0.15x ATR)

# File paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent   # Trading/stock_model/
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
RAW_DATA_DIR = DATA_DIR / "raw_data"
EXECUTION_LOG_PATH = RESULTS_DIR / "execution_log.csv"
CANDIDATE_DIR = RESULTS_DIR / "candidates"
CONFIG_DIR = PROJECT_ROOT / "config"
FIRESTORE_KEY_PATH = CONFIG_DIR / "firebase_service_account.json"
FIRESTORE_USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"

# ------------------- CONCENTRATION / CORRELATION WARNINGS -------------------
# Informational only — never excludes, reorders, or resizes a candidate.
# Shared between the V2 systematic slate and the Bucket 2 opportunity scan.
CRYPTO_CLUSTER = {
    "MSTR", "MARA", "RIOT", "COIN", "HOOD", "BMNR", "BLSH", "CRCL",
    "BITO", "FBTC", "GBTC", "IBIT",
}
GOLD_CLUSTER = {"GLD", "GDX", "GDXJ", "SLV", "AU", "NEM", "AEM", "WPM", "RGLD"}
BIOTECH_TICKERS = {"XBI", "IBB", "LABU", "ARKG"}
BIOTECH_ATR_PCT_THRESHOLD = 0.05

CONCENTRATION_EXPLANATIONS = {
    "crypto-sentiment": "one bet not two",
    "gold": "correlated commodity exposure",
    "biotech": "correlated biotech volatility",
}

# ------------------- BUCKET 2 OPPORTUNITY SCANNER CONFIG (--opportunities) -------------------
# Entirely separate from the V2 systematic model above: expanded universe,
# different signal definitions, fixed (not risk-based) sizing. Never mixed
# into V2's own candidates CSV, Firestore writes, or filter logic.
MASTER_UNIVERSE_PATH = RAW_DATA_DIR / "master_universe.json"
SP500_CACHE_PATH = RAW_DATA_DIR / "sp500_tickers.csv"
OPP_DOWNLOAD_BATCH_SIZE = 100

OPP_MOMENTUM_MIN_RET_3W = 0.30   # Signal 1: 3W return floor
OPP_VOLUME_MIN_SURGE = 5.0       # Signal 2: volume-vs-20d-avg floor
OPP_VOLUME_MIN_RET_3W = 0.10     # Signal 2: 3W return floor (looser than Signal 1)
OPP_MIN_AVG_DV_M = 50.0          # both signals: min $M avg daily dollar volume
OPP_MIN_PRICE = 10.0             # both signals: min price
OPP_MAX_ATR_PCT = 0.20           # both signals: max ATR% (volatility cap)
OPP_TOP_N = 10

OPP_SIZING_NOTE = "$300-500 deployed (fixed, not risk-based)"

OPP_CSV_COLUMNS = [
    "Rank", "Ticker", "Universe", "Signal", "Tier", "Score",
    "Return 3W", "Return 1W", "Vol Surge", "Entry Price",
    "Stop Price", "Target Price", "Avg DV ($M)", "Accel",
    "Dist 52W", "ATR Pct", "Warning",
]

# Conviction tiers (Change 4) — purely a display label, never a filter.
OPP_TIER_STRONG_RET_3W = 0.40
OPP_TIER_STRONG_VOL = 8.0
OPP_TIER_SOLID_RET_3W = 0.30
OPP_TIER_SOLID_VOL = 5.0
OPP_TIER_WATCH_RET_3W = 0.20


def opportunity_tier(ret_3w: float, vol_surge: float) -> str:
    ret_3w = ret_3w if pd.notna(ret_3w) else 0.0
    vol_surge = vol_surge if pd.notna(vol_surge) else 0.0
    if ret_3w >= OPP_TIER_STRONG_RET_3W or vol_surge >= OPP_TIER_STRONG_VOL:
        return "STRONG"
    if ret_3w >= OPP_TIER_SOLID_RET_3W or vol_surge >= OPP_TIER_SOLID_VOL:
        return "SOLID"
    if ret_3w >= OPP_TIER_WATCH_RET_3W:
        return "WATCH"
    return ""


def ensure_directories() -> None:
    """Create required folders if they don't exist."""
    for p in [DATA_DIR, RESULTS_DIR, RAW_DATA_DIR, CANDIDATE_DIR]:
        os.makedirs(p, exist_ok=True)


# ------------------- CORE PIPELINE -------------------

def build_universe() -> list[str]:
    """
    Get S&P 500 tickers (from cache if present, otherwise scrape & save).
    """
    tickers = get_sp500_tickers(
        save_path=str(RAW_DATA_DIR / "sp500_tickers.csv"),
        force_refresh=FORCE_REFRESH_TICKERS,
        cross_check=True,
    )
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers
    return tickers


def load_price_data(tickers: list[str]) -> pd.DataFrame:
    """
    Download recent OHLCV data for the full universe.

    V2 does NOT pre-filter by price/volume here. The real hard filters
    (MIN_PRICE, MIN_AVG_VOL, etc.) live in model_logic.add_signals(), which
    needs to see every ticker — including cheap/thin ones — so the
    candidate-pool diagnostic can report a genuine rejection reason for
    each instead of silently dropping it before signals are ever computed.
    A ticker only ends up missing from the returned data if yfinance
    couldn't return anything for it at all (bad symbol, delisted, etc.);
    save_candidate_pool() reconciles those as 'rejected_no_data'.
    """
    df = fetch_price_data(
        tickers=tickers,
        period=PRICE_HISTORY_PERIOD,
        interval=PRICE_HISTORY_INTERVAL,
    )
    if df.empty:
        return df

    # Cache the filtered OHLCV snapshot for inspection/re-runs.
    today_str = datetime.today().strftime("%Y-%m-%d")
    cache_path = RAW_DATA_DIR / f"ohlcv_filtered_{today_str}.csv"
    df.to_csv(cache_path, index=False)

    # Drop tickers whose latest date is stale vs the global max date.
    max_date = df["date"].max()
    latest_by_ticker = df.groupby("ticker")["date"].max()
    stale_tickers = latest_by_ticker[latest_by_ticker < max_date].index.tolist()
    if stale_tickers:
        df = df[~df["ticker"].isin(stale_tickers)].copy()
        print(f"    Dropped stale tickers: {len(stale_tickers)}")
    print(f"    Latest price date: {max_date.date()}")
    return df


def size_positions(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Given a candidates DataFrame (from model_logic.select_top_candidates),
    compute stop levels and position sizes for each candidate.

    Used by backfill mode. The V2 daily pipeline (run_daily_model) does its
    own sizing pass instead, since it needs per-candidate dynamic risk
    dollars (RISK_NORMAL / RISK_BOOSTED) based on filters_passed.
    """
    df = candidates.copy()

    if df.empty:
        return df

    # Use latest close as our notional "entry" price for planning
    df["entry_price"] = df["close"]

    # Ensure ATR is present (add_signals should have added 'atr_14')
    if "atr_14" not in df.columns:
        raise ValueError("Expected 'atr_14' in candidates DataFrame. Check model_logic.add_signals().")

    # Compute stop price (ATR-based) and position size
    stop_prices: list[float] = []
    shares_list: list[int] = []
    targets: list[float] = []

    for _, row in df.iterrows():
        entry = float(row["entry_price"])
        atr = float(row["atr_14"])

        stop = compute_stop_price_long(
            entry_price=entry, atr=atr, atr_multiple=ATR_STOP_MULTIPLE,
            max_stop_dist_pct=MAX_STOP_DIST_PCT,
        )

        # None means stop distance exceeded the cap — reject this candidate
        if stop is None:
            stop_prices.append(None)
            shares_list.append(0)
            targets.append(None)
            continue

        shares = compute_position_size_long(
            entry_price=entry,
            stop_price=stop,
            risk_config=RISK_CONFIG,
        )

        # Simple 2:1 reward-to-risk target
        risk_per_share = max(entry - stop, 0.01)
        target = entry + 2.0 * risk_per_share

        stop_prices.append(stop)
        shares_list.append(shares)
        targets.append(target)

    df["stop_price"] = stop_prices
    df["shares"] = shares_list
    df["target_price"] = targets

    # Filter out cases where position size is zero (too volatile / too expensive)
    df = df[df["shares"] > 0].reset_index(drop=True)

    return df


# Full V2 output column order (Change 10). Company is appended at the end —
# it isn't part of the spec'd order but was present in V1 output, so it's
# kept as an additive trailing column rather than dropped. Warning (added
# for the concentration-warning feature) is appended after Company for the
# same reason — additive, not a reordering of the existing schema.
CANDIDATE_COLUMN_ORDER = [
    "Rank", "Ticker", "Signal Score", "Entry Price", "Max Entry", "Stop Price", "Target Price",
    "Shares", "Return 1W", "RSI 14", "Regime", "Sector",
    "Filters Passed", "Score Gap", "Dist 52W", "Sector ETF", "Sector ETF 3W", "Accel",
    "Company", "Warning",
]


def compute_concentration_warnings(df: pd.DataFrame) -> pd.Series:
    """
    Given a candidates DataFrame with at least a lowercase 'ticker' column
    (and, where available, 'sector' / 'atr_pct'), returns a Series of
    semicolon-joined warning labels aligned to df.index — empty string
    where nothing is flagged. Purely informational: never excludes,
    reorders, or resizes anything.

    Checks four types of concentration risk within the SAME slate/scan:
      A — 2+ candidates share a sector (skipped if no 'sector' column —
          Bucket 2's scan has no sector data)
      B — 2+ candidates are in the crypto-sentiment cluster
      C — 2+ candidates are in the gold/commodity cluster
      D — 2+ candidates are biotech (explicit ETF tickers, or — where
          'sector'/'atr_pct' are available — Healthcare sector with
          ATR% > 5%)
    """
    if df.empty:
        return pd.Series([], dtype=str, index=df.index)

    warnings: dict[int, list[str]] = {idx: [] for idx in df.index}
    tickers_in_slate = set(df["ticker"])

    # Type A — same sector
    if "sector" in df.columns:
        for sector, grp in df.groupby("sector"):
            if sector and sector != "Other" and len(grp) >= 2:
                for idx in grp.index:
                    warnings[idx].append(f"SECTOR: {sector}")

    # Type B — crypto-sentiment cluster
    crypto_hits = tickers_in_slate & CRYPTO_CLUSTER
    if len(crypto_hits) >= 2:
        for idx in df[df["ticker"].isin(crypto_hits)].index:
            warnings[idx].append("CORR: crypto-sentiment")

    # Type C — gold/commodity cluster
    gold_hits = tickers_in_slate & GOLD_CLUSTER
    if len(gold_hits) >= 2:
        for idx in df[df["ticker"].isin(gold_hits)].index:
            warnings[idx].append("CORR: gold")

    # Type D — biotech cluster (explicit tickers, plus Healthcare+high-ATR
    # where sector data exists)
    biotech_mask = df["ticker"].isin(BIOTECH_TICKERS)
    if "sector" in df.columns and "atr_pct" in df.columns:
        biotech_mask = biotech_mask | (
            (df["sector"] == "Healthcare") & (df["atr_pct"] > BIOTECH_ATR_PCT_THRESHOLD)
        )
    biotech_hits = df[biotech_mask]
    if len(biotech_hits) >= 2:
        for idx in biotech_hits.index:
            warnings[idx].append("CORR: biotech")

    return pd.Series({idx: "; ".join(v) for idx, v in warnings.items()})


def print_concentration_warnings(df: pd.DataFrame, ticker_col: str = "ticker", warning_col: str = "Warning") -> None:
    """
    Prints the grouped "⚠ Concentration warnings:" section after a
    candidate table. Prints nothing at all if there are no warnings.
    """
    if df.empty or warning_col not in df.columns:
        return

    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        label = row[warning_col]
        if not label:
            continue
        for single in str(label).split("; "):
            groups.setdefault(single, []).append(str(row[ticker_col]))

    if not groups:
        return

    print("⚠ Concentration warnings:")
    for label, tickers in groups.items():
        names = " + ".join(tickers)
        if label.startswith("CORR: "):
            kind = label[len("CORR: "):]
            explain = CONCENTRATION_EXPLANATIONS.get(kind, "correlated moves")
            print(f"  {names}: CORR {kind} — {explain}")
        elif label.startswith("SECTOR: "):
            sector = label[len("SECTOR: "):]
            print(f"  {names}: SECTOR {sector} — correlated moves")

# Filenames matching exactly "candidates_YYYY-MM-DD.csv" — excludes the
# _backup_, _backfill_, and candidate_pool_ variants.
_CANDIDATES_FILENAME_RE = re.compile(r"^candidates_(\d{4}-\d{2}-\d{2})\.csv$")


def save_sitout_csv(regime: str, score_gap: float, reason: str) -> Path:
    """
    Write a single sit-out row so the dashboard can detect the regime (empty
    Ticker/Signal Score on row 1 = sit-out week, per Change 12/13).
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = CANDIDATE_DIR / f"candidates_{today_str}.csv"

    sit_out = pd.DataFrame([{
        "Ticker":       "",
        "Signal Score": "",
        "Entry Price":  "",
        "Stop Price":   "",
        "Target Price": "",
        "Shares":       "",
        "Return 1W":    "",
        "RSI 14":       "",
        "Regime":       regime,
        "Score Gap":    round(score_gap, 3),
        "Reason":       reason,
    }])
    sit_out.to_csv(out_path, index=False)
    return out_path


def save_candidates_csv(candidates: pd.DataFrame, regime: str, score_gap: float) -> Path:
    """
    Save the final, sized V2 candidate slate to a dated CSV and return the path.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = CANDIDATE_DIR / f"candidates_{today_str}.csv"

    df = candidates.reset_index(drop=True).copy()
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["regime"] = regime
    df["score_gap"] = round(score_gap, 3)

    df = df.rename(columns={
        "ticker": "Ticker",
        "signal_score": "Signal Score",
        "entry_price": "Entry Price",
        "max_entry_price": "Max Entry",
        "stop_price": "Stop Price",
        "target_price": "Target Price",
        "shares": "Shares",
        "ret_1w": "Return 1W",
        "rsi_14": "RSI 14",
        "regime": "Regime",
        "sector": "Sector",
        "filters_passed": "Filters Passed",
        "score_gap": "Score Gap",
        "dist_52w": "Dist 52W",
        "sector_etf": "Sector ETF",
        "sector_etf_3w": "Sector ETF 3W",
        "f_accel": "Accel",
        "company": "Company",
    })

    for col in ["Signal Score", "Entry Price", "Max Entry", "Stop Price", "Target Price", "Return 1W", "RSI 14"]:
        df[col] = df[col].astype(float).round(2)
    for col in ["Dist 52W", "Sector ETF 3W"]:
        df[col] = df[col].astype(float).round(4)

    present_cols = [c for c in CANDIDATE_COLUMN_ORDER if c in df.columns]
    df = df[present_cols]

    validate_csv_schema(df)
    df.to_csv(out_path, index=False)
    return out_path


def get_last_week_tickers(candidate_dir: Path, today_str: str) -> set[str]:
    """
    Change 4 (repeat-appearance filter): read the Ticker column from the
    most recent candidates_YYYY-MM-DD.csv (excluding today's file, and
    excluding _backup_/_backfill_/candidate_pool_ variants). Returns an
    empty set if no prior file exists (first run).
    """
    dated: list[tuple[str, Path]] = []
    for p in candidate_dir.glob("candidates_*.csv"):
        m = _CANDIDATES_FILENAME_RE.match(p.name)
        if m and m.group(1) != today_str:
            dated.append((m.group(1), p))

    if not dated:
        return set()

    dated.sort(key=lambda x: x[0], reverse=True)
    latest_path = dated[0][1]

    try:
        df = pd.read_csv(latest_path)
    except Exception as exc:
        print(f"    Warning: could not read {latest_path.name} for repeat filter: {exc}")
        return set()

    if "Ticker" not in df.columns:
        return set()

    tickers = df["Ticker"].dropna().astype(str)
    tickers = tickers[tickers.str.strip() != ""]
    print(f"    Repeat filter: comparing against {latest_path.name} ({len(tickers)} tickers)")
    return set(tickers.str.upper())


def maybe_log_trades(candidates: pd.DataFrame, log_trades: bool) -> None:
    """
    Optionally append candidate rows to execution_log.csv.
    In practice, you might only log trades after you actually take them.
    For now this is off by default.
    """
    if not log_trades or candidates.empty:
        return

    trades: list[dict] = []
    today = datetime.today()

    for _, row in candidates.iterrows():
        trades.append(
            {
                "date": today,
                "ticker": row["ticker"],
                "direction": row.get("direction", "LONG"),
                "entry_price": row["entry_price"],
                "stop_price": row["stop_price"],
                "target_price": row["target_price"],
                "shares": int(row["shares"]),
                "signal_score": row.get("signal_score", None),
                "rsi_14": row.get("rsi_14", None),
                "ret_1w": row.get("ret_1w", None),
                "ret_3w": row.get("ret_3w", None),
                "notes": "AUTO-GENERATED CANDIDATE (verify before trading)",
            }
        )

    append_execution_log(str(EXECUTION_LOG_PATH), trades)


# ------------------- CANDIDATE POOL -------------------

def _rejection_status(row: pd.Series) -> str:
    """
    Return the primary rejection reason for a signals-row that did not make
    the final slate. Filters are checked in priority order so each ticker
    gets exactly one label. Mirrors the V2 hard filters (F1-F5) applied in
    model_logic.add_signals(); does NOT reflect the pipeline-level filters
    (score gap, repeat, sector ETF, earnings) since those depend on live
    pipeline state not available at this diagnostic-pool stage.

    RSI is intentionally NOT checked here — it's no longer a hard filter
    (see model_logic.add_signals), just a reported/diagnostic column.
    """
    price        = row.get("price", 0) or 0
    avg_vol      = row.get("avg_vol_20", 0) or 0
    stop_dist    = row.get("stop_dist_pct", float("nan"))
    ret_3w       = row.get("ret_3w", float("nan"))
    vol_surge    = row.get("vol_surge", float("nan"))
    sector       = row.get("sector", "")
    dist_52w     = row.get("dist_52w", float("nan"))

    if pd.isna(price) or price < MIN_PRICE:
        return "rejected_price"
    if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOL:
        return "rejected_volume"
    if pd.isna(stop_dist) or stop_dist > MAX_STOP_DIST_PCT:
        return "rejected_stop"
    if pd.isna(ret_3w) or ret_3w < MIN_RET_3W:
        return "rejected_ret_3w"
    if pd.isna(vol_surge) or vol_surge < MIN_VOL_SURGE:
        return "rejected_vol_surge"
    if sector == EXCLUDED_SECTOR:
        return "rejected_sector"
    if pd.isna(dist_52w) or dist_52w < MIN_DIST_52W:
        return "rejected_52w_high"
    return "not_selected"


def save_candidate_pool(
    signals: pd.DataFrame,
    final_candidates: pd.DataFrame,
    sector_map: dict,
    company_map: dict,
    universe_tickers: list[str] | None = None,
) -> Path:
    """
    Write Results/candidates/candidate_pool_YYYY-MM-DD.csv with every ticker
    in the universe, ranked by signal_score, and their status.
    Also prints a rejection-reason summary to the terminal.

    universe_tickers, if given, is reconciled against `signals` — any
    universe ticker with no row in `signals` at all (yfinance returned
    nothing for it, or it was dropped upstream for stale/insufficient
    history) is still written to the pool with Status='rejected_no_data'.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = CANDIDATE_DIR / f"candidate_pool_{today_str}.csv"

    final_tickers = set(final_candidates["ticker"].tolist()) if not final_candidates.empty else set()

    # SPY is only present for regime classification, not a real candidate —
    # exclude it so the pool reflects the S&P 500 constituent universe only.
    pool = signals[signals["ticker"] != "SPY"].sort_values("signal_score", ascending=False).copy()

    pool["status"] = pool.apply(
        lambda r: "passed" if r["ticker"] in final_tickers else _rejection_status(r),
        axis=1,
    )
    pool["company"]      = pool["ticker"].map(company_map).fillna("")
    pool["sector"]       = pool["ticker"].map(sector_map).fillna("Other")
    pool["entry_price"]  = pool["close"]
    pool["stop_price"]   = pool["entry_price"] - ATR_STOP_MULTIPLE * pool["atr_14"]
    pool["stop_pct"]     = (pool["stop_dist_pct"] * 100).round(2)
    pool["target_price"] = pool["entry_price"] + 2.0 * (pool["entry_price"] - pool["stop_price"])

    out = pool[[
        "ticker", "company", "sector",
        "entry_price", "stop_price", "stop_pct", "target_price",
        "signal_score", "status",
        "ret_1w", "ret_3w", "vol_surge", "rsi_14", "atr_14",
    ]].rename(columns={
        "ticker":       "Ticker",
        "company":      "Company",
        "sector":       "Sector",
        "entry_price":  "Entry Price",
        "stop_price":   "Stop Price",
        "stop_pct":     "Stop Pct",
        "target_price": "Target Price",
        "signal_score": "Signal Score",
        "status":       "Status",
        "ret_1w":       "Return 1W",
        "ret_3w":       "Return 3W",
        "vol_surge":    "Vol Surge",
        "rsi_14":       "RSI 14",
        "atr_14":       "ATR 14",
    })

    round_cols = [
        "Entry Price", "Stop Price", "Stop Pct", "Target Price", "Signal Score",
        "Return 1W", "Return 3W", "Vol Surge", "RSI 14", "ATR 14",
    ]
    for col in round_cols:
        out[col] = out[col].round(4)

    # Reconcile against the full universe: tickers with no row in `signals`
    # at all couldn't be evaluated (no usable price data) — show them too.
    no_data_count = 0
    if universe_tickers:
        present = set(pool["ticker"])
        missing = [t for t in universe_tickers if t != "SPY" and t not in present]
        no_data_count = len(missing)
        if missing:
            missing_rows = pd.DataFrame([{
                "Ticker": t,
                "Company": company_map.get(t, ""),
                "Sector": sector_map.get(t, "Other"),
                "Entry Price": "", "Stop Price": "", "Stop Pct": "", "Target Price": "",
                "Signal Score": "", "Status": "rejected_no_data",
                "Return 1W": "", "Return 3W": "", "Vol Surge": "", "RSI 14": "", "ATR 14": "",
            } for t in missing])
            out = pd.concat([out, missing_rows], ignore_index=True)

    out.to_csv(out_path, index=False)

    # Terminal summary
    counts = pool["status"].value_counts()
    status_order = [
        "passed", "not_selected",
        "rejected_52w_high", "rejected_sector", "rejected_vol_surge", "rejected_ret_3w",
        "rejected_stop", "rejected_volume", "rejected_price", "rejected_no_data",
    ]
    print(f">>> Candidate pool summary (all {len(out)} tickers, ranked by score):")
    for label in status_order:
        n = no_data_count if label == "rejected_no_data" else int(counts.get(label, 0))
        if n:
            print(f"    {label}: {n}")

    return out_path


def print_candidate_table(selected: pd.DataFrame, score_gap: float) -> None:
    """Change 11: terminal output for the final selected slate."""
    print(f"\n{'Rank':<5}{'Ticker':<8}{'Score':>7}{'Entry':>9}{'MaxEntry':>10}{'Stop':>9}{'Target':>9}{'Shares':>8}{'Filters':>9}{'Accel':>7}")
    print("-" * 81)
    for i, row in enumerate(selected.itertuples(), start=1):
        accel_mark = "✓" if row.f_accel else "✗"
        print(
            f"{i:<5}{row.ticker:<8}{row.signal_score:>7.2f}{row.entry_price:>9.2f}"
            f"{row.max_entry_price:>10.2f}{row.stop_price:>9.2f}{row.target_price:>9.2f}{int(row.shares):>8}"
            f"{int(row.filters_passed):>6}/9{accel_mark:>7}"
        )

    print(f"\nScore gap this week: {score_gap:.3f}")

    normal_n = int((selected["filters_passed"] < FILTERS_BOOSTED_THRESHOLD).sum())
    boosted_n = int((selected["filters_passed"] >= FILTERS_BOOSTED_THRESHOLD).sum())
    print(f"Sizing: {normal_n} candidates at normal (${RISK_NORMAL:.0f} risk)")
    print(f"Sizing: {boosted_n} candidates at 2x (${RISK_BOOSTED:.0f} risk) — {FILTERS_BOOSTED_THRESHOLD}+ filters passed")

    flag_labels = [
        ("vol", "f_vol"), ("gap", "f_gap"), ("fresh", "f_fresh"), ("sector", "f_sector"),
        ("price", "f_price"), ("etf", "f_etf"), ("52w", "f_52w"), ("earn", "f_earn"), ("accel", "f_accel"),
    ]
    for row in selected.itertuples():
        row_d = row._asdict()
        parts = " ".join(f"{name}{'✓' if row_d[flag] else '✗'}" for name, flag in flag_labels)
        print(f"{row.ticker}: {parts} ({int(row.filters_passed)}/9)")


def print_premarket_check(selected: pd.DataFrame) -> None:
    """Prints the pre-market (9:15am Monday/Tuesday) entry-ceiling reference table."""
    print(">>> " + "=" * 60)
    print(">>> PRE-MARKET CHECK (9:15am Monday or Tuesday)")
    print(">>> " + "=" * 60)
    print(f">>> Edge disappears at {MAX_ENTRY_PREMIUM_PCT*100:.1f}% above signal price (backtested)")
    print(">>>")
    print(">>> Ticker  Signal Price  Max Entry  Action if above Max Entry")
    for row in selected.itertuples():
        print(f">>> {row.ticker:<8}{row.entry_price:>10.2f}{row.max_entry_price:>11.2f}  SKIP — do not chase")
    print(">>>")
    print(">>> If price <= Max Entry at 9:15am: enter market order at 9:30am")
    print(">>> If price > Max Entry at 9:15am: skip this candidate")
    print(">>> Place GTC stop immediately after every fill — not optional")


def run_daily_model(log_trades: bool = False) -> Path | None:
    """
    Convenience wrapper to run the full V2 daily pipeline end-to-end.
    Returns the path to the candidates CSV, or None if no price data.
    """
    ensure_directories()
    today_str = datetime.today().strftime("%Y-%m-%d")

    print(">>> Building S&P 500 universe...")
    tickers = build_universe()
    sp500_tickers = [t for t in tickers if t != "SPY"]
    print(f"    Universe size: {len(tickers)} tickers ({len(sp500_tickers)} S&P 500 constituents)")

    print(">>> Downloading price data...")
    price_data = load_price_data(tickers)
    print(f"    Price data shape: {price_data.shape}")

    if price_data.empty:
        print("!!! No price data returned after filters. Exiting.")
        return None

    # --- Regime: still calculated & reported (V2 no longer blocks on it) ---
    regime = classify_current_regime(price_data)
    print(f">>> Current market regime: {regime}")

    print(">>> Looking up sectors and company names...")
    sector_map = get_sector_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))
    company_map = get_company_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))

    print(">>> Computing signals & applying hard filters (F1-F5)...")
    signals = add_signals(price_data, sector_map=sector_map)
    pool = signals[(signals["direction"] == "LONG") & (signals["is_eligible"])].copy()
    print(f"    Candidates passing hard filters: {len(pool)}")

    if pool.empty:
        print("!!! No candidates passed hard filters this week")
        out_path = save_sitout_csv(regime=regime, score_gap=0.0, reason="NO_CANDIDATES")
        print(f"    Sit-out file saved to: {out_path}")
        pool_path = save_candidate_pool(signals, pool, sector_map, company_map, universe_tickers=sp500_tickers)
        print(f">>> Candidate pool saved to: {pool_path}")
        return out_path

    # --- Change 3: score gap gate (week-level) ---
    score_gap = float(pool["signal_score"].max() - pool["signal_score"].min())
    print(f">>> Score gap this week: {score_gap:.3f}")
    if score_gap < SCORE_GAP_THRESHOLD:
        print(
            f"Score gap {score_gap:.3f} below threshold {SCORE_GAP_THRESHOLD:.2f} — "
            f"field too bunched, no clear conviction this week"
        )
        out_path = save_sitout_csv(regime=regime, score_gap=score_gap, reason="LOW_SCORE_GAP")
        print(f"    Sit-out file saved to: {out_path}")
        pool_path = save_candidate_pool(signals, pd.DataFrame(), sector_map, company_map, universe_tickers=sp500_tickers)
        print(f">>> Candidate pool saved to: {pool_path}")
        return out_path

    # --- Change 4: repeat-appearance filter ---
    last_week_tickers = get_last_week_tickers(CANDIDATE_DIR, today_str)
    if last_week_tickers:
        before = len(pool)
        pool = pool[~pool["ticker"].isin(last_week_tickers)].copy()
        print(f"    Repeat filter: excluded {before - len(pool)} ticker(s) seen last week")

    # --- Change 5: sector ETF momentum filter ---
    if not pool.empty:
        print(">>> Checking sector ETF momentum...")
        etf_cache: dict[str, float | None] = {}
        keep_mask, etf_list, etf_ret_list = [], [], []
        for row in pool.itertuples():
            etf = SECTOR_ETF_MAP.get(row.sector)
            if etf is None:
                # No ETF mapping for this sector (e.g. 'Other') — fail open.
                keep_mask.append(True)
                etf_list.append("")
                etf_ret_list.append(float("nan"))
                continue
            etf_ret = fetch_sector_etf_return_3w(etf, etf_cache)
            etf_list.append(etf)
            etf_ret_list.append(etf_ret if etf_ret is not None else float("nan"))
            if etf_ret is not None and etf_ret < ETF_MIN_3W_RETURN:
                print(f"    {row.ticker} excluded — sector ETF {etf} only up {etf_ret*100:.1f}% 3W")
                keep_mask.append(False)
            else:
                keep_mask.append(True)
        pool["sector_etf"] = etf_list
        pool["sector_etf_3w"] = etf_ret_list
        pool = pool[keep_mask].copy()

    # --- Change 6: earnings exclusion filter (only for survivors so far) ---
    if not pool.empty:
        print(">>> Checking upcoming earnings dates...")
        week_start = pd.Timestamp(datetime.today().date())
        next_monday = week_start + pd.Timedelta(days=7)
        window_end = next_monday + pd.Timedelta(days=7)

        earnings_cache: dict[str, list | None] = {}
        keep_mask = []
        for row in pool.itertuples():
            has_earnings = fetch_next_week_earnings(row.ticker, next_monday, window_end, earnings_cache)
            if has_earnings:
                print(f"    {row.ticker} excluded — earnings next week")
            keep_mask.append(not has_earnings)
        pool = pool[keep_mask].copy()

    if pool.empty:
        print("!!! No candidates passed all filters this week")
        out_path = save_sitout_csv(regime=regime, score_gap=score_gap, reason="NO_CANDIDATES")
        print(f"    Sit-out file saved to: {out_path}")
        pool_path = save_candidate_pool(signals, pd.DataFrame(), sector_map, company_map, universe_tickers=sp500_tickers)
        print(f">>> Candidate pool saved to: {pool_path}")
        return out_path

    # --- Change 7: filter count per candidate (0-9) ---
    pool["f_vol"]    = pool["vol_surge"] >= MIN_VOL_SURGE
    pool["f_gap"]    = score_gap >= SCORE_GAP_THRESHOLD
    pool["f_fresh"]  = ~pool["ticker"].isin(last_week_tickers)
    pool["f_sector"] = pool["sector"] != EXCLUDED_SECTOR
    pool["f_price"]  = pool["close"] >= MIN_PRICE
    pool["f_etf"]    = pool["sector_etf_3w"].fillna(ETF_MIN_3W_RETURN) >= ETF_MIN_3W_RETURN
    pool["f_52w"]    = pool["dist_52w"] >= MIN_DIST_52W
    pool["f_earn"]   = True  # already filtered above; recorded for the audit trail
    pool["f_accel"]  = pool["ret_1w"] > (pool["ret_3w"] / 3.0)  # sizing signal only, never excludes

    filter_cols = ["f_vol", "f_gap", "f_fresh", "f_sector", "f_price", "f_etf", "f_52w", "f_earn", "f_accel"]
    pool["filters_passed"] = pool[filter_cols].sum(axis=1).astype(int)

    # --- Change 8: show every candidate passing 7+ filters, ranked by score ---
    # No artificial top-N cutoff — TOP_N_CANDIDATES is still used by backfill
    # mode (untouched), just not here. In practice everything reaching this
    # point already has filters_passed in {8, 9} (F1/F2/F3/F4/F5/fresh/etf/
    # earn are all hard-enforced by row-drops earlier in the pipeline; only
    # f_accel varies), so "7+" is a genuine floor, not a no-op guess.
    pool = pool.sort_values("signal_score", ascending=False).reset_index(drop=True)
    selected = pool[pool["filters_passed"] >= 7].copy().reset_index(drop=True)
    print(f">>> Selected {len(selected)} candidate(s) after all filters — showing all")

    # --- Change 9: stop / target recalculation + dynamic position sizing ---
    selected["entry_price"] = selected["close"]
    selected["max_entry_price"] = (selected["entry_price"] * (1 + MAX_ENTRY_PREMIUM_PCT)).round(2)
    stop_prices, targets, shares_list = [], [], []
    for row in selected.itertuples():
        entry = float(row.entry_price)
        atr = float(row.atr_14)
        stop = compute_stop_price_long(
            entry_price=entry, atr=atr, atr_multiple=ATR_STOP_MULTIPLE,
            max_stop_dist_pct=MAX_STOP_DIST_PCT,
        )
        if stop is None:
            stop_prices.append(None); targets.append(None); shares_list.append(0)
            continue
        risk_per_share = max(entry - stop, 0.01)
        target = entry + 2.0 * risk_per_share
        risk_dollars = RISK_BOOSTED if row.filters_passed >= FILTERS_BOOSTED_THRESHOLD else RISK_NORMAL
        # Override both the flat dollar cap AND the equity-percentage cap to
        # risk_dollars — compute_position_size_long() takes min(pct-of-equity,
        # dollar cap), so leaving max_risk_per_trade_pct at its stale 0.5%
        # default (=$50 on a $10k account) would silently clamp RISK_BOOSTED
        # ($70) back down to $50 via that field instead.
        risk_cfg = replace(
            RISK_CONFIG,
            max_dollar_risk_per_trade=risk_dollars,
            max_risk_per_trade_pct=risk_dollars / ACCOUNT_EQUITY,
        )
        shares = compute_position_size_long(entry_price=entry, stop_price=stop, risk_config=risk_cfg)
        stop_prices.append(stop); targets.append(target); shares_list.append(shares)

    selected["stop_price"] = stop_prices
    selected["target_price"] = targets
    selected["shares"] = shares_list
    selected = selected[selected["shares"] > 0].reset_index(drop=True)

    if selected.empty:
        print("!!! No candidates passed all filters this week")
        out_path = save_sitout_csv(regime=regime, score_gap=score_gap, reason="NO_CANDIDATES")
        print(f"    Sit-out file saved to: {out_path}")
        pool_path = save_candidate_pool(signals, pd.DataFrame(), sector_map, company_map, universe_tickers=sp500_tickers)
        print(f">>> Candidate pool saved to: {pool_path}")
        return out_path

    selected["company"] = selected["ticker"].map(company_map).fillna("")
    selected["Warning"] = compute_concentration_warnings(selected)

    out_path = save_candidates_csv(selected, regime=regime, score_gap=score_gap)
    print(f">>> Daily candidates saved to: {out_path}")

    print_candidate_table(selected, score_gap)
    print_concentration_warnings(selected)
    print_premarket_check(selected)

    pool_path = save_candidate_pool(signals, selected, sector_map, company_map, universe_tickers=sp500_tickers)
    print(f">>> Candidate pool saved to: {pool_path}")

    maybe_log_trades(selected, log_trades=log_trades)

    return out_path


# ------------------- BACKFILL MODE -------------------

def run_backfill(backfill_date_str: str) -> Path | None:
    """
    Generate a candidate slate for a specific past date without the regime filter.

    Uses the dated ohlcv_filtered snapshot when available (has SPY + adj_close),
    falls back to Data/backtest_cache/ohlcv_5yr.csv otherwise.
    Entry price is the open on the backfill date; falls back to prior-day close
    if the open is missing for a ticker.
    """
    ensure_directories()

    try:
        backfill_dt = datetime.strptime(backfill_date_str, "%Y-%m-%d")
    except ValueError:
        print(f"!!! Invalid date format: '{backfill_date_str}'. Use YYYY-MM-DD.")
        return None

    backfill_ts = pd.Timestamp(backfill_dt)

    print(f"\n{'='*60}")
    print(f"  [BACKFILL] {backfill_date_str}")
    print(f"{'='*60}")

    # Prefer the dated filtered snapshot (has SPY for regime + adj_close).
    # Fall back to the 5yr backtest cache if no snapshot exists for this date.
    filtered_path = RAW_DATA_DIR / f"ohlcv_filtered_{backfill_date_str}.csv"
    cache_path = DATA_DIR / "backtest_cache" / "ohlcv_5yr.csv"

    if filtered_path.exists():
        print(f">>> Loading dated snapshot: {filtered_path.name}")
        price_data = pd.read_csv(filtered_path, parse_dates=["date"])
    elif cache_path.exists():
        print(f">>> Loading 5yr cache (no dated snapshot for {backfill_date_str}): {cache_path.name}")
        price_data = pd.read_csv(cache_path, parse_dates=["date"])
        if "adj_close" not in price_data.columns:
            price_data["adj_close"] = price_data["close"]
    else:
        print("!!! No cached data found. Run the normal pipeline at least once to populate the cache.")
        return None

    # Slice to only the data the model would have seen on the backfill date.
    price_data = price_data[price_data["date"] <= backfill_ts].copy()
    if price_data.empty:
        print(f"!!! No price data available on or before {backfill_date_str}")
        return None
    print(f"    Price data shape: {price_data.shape}  (up to {backfill_date_str})")

    # --- Regime: classify and report, but do NOT filter ---
    regime = classify_current_regime(price_data)
    print(f"\n!!! BACKFILL MODE — Regime filter bypassed for historical tracking")
    print(f"    Regime on {backfill_date_str}: {regime}")
    if regime in ("BULL_TREND", "BULL_VOLATILE"):
        print(f"    These candidates would NOT have been traded (sit-out week)")
    print()

    # --- Sector metadata (needed by add_signals for the Materials exclusion) ---
    sector_map = get_sector_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))
    company_map = get_company_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))

    # --- Score and select candidates ---
    # NOTE: backfill applies the V2 hard filters (F1-F5, via add_signals) and
    # the V2 stop/target formula (via size_positions, below), since those are
    # deterministic functions of price history. It intentionally does NOT
    # apply the score-gap gate, repeat-appearance filter, sector ETF filter,
    # or earnings filter — those depend on live pipeline state (last week's
    # CSV, current ETF/earnings data) that can't be reconstructed accurately
    # for a past date, and backfill instead uses a flat risk amount rather
    # than the V2 filters_passed-based sizing.
    print(">>> [BACKFILL] Computing signals & selecting candidates...")
    signals = add_signals(price_data, sector_map=sector_map)
    candidates = select_top_candidates(signals, max_trades=TOP_N_CANDIDATES)
    print(f"    Candidates before sizing: {len(candidates)}")

    if candidates.empty:
        print(f"!!! No trade candidates found for {backfill_date_str}")
        return None

    # --- Entry price: use open on backfill date; fall back to prior close ---
    open_prices = (
        price_data[price_data["date"] == backfill_ts]
        .set_index("ticker")["open"]
        .to_dict()
    )

    cands_for_sizing = candidates.copy()
    for idx, row in cands_for_sizing.iterrows():
        ticker = row["ticker"]
        if ticker in open_prices and pd.notna(open_prices[ticker]):
            cands_for_sizing.at[idx, "close"] = open_prices[ticker]
        else:
            print(f"    Warning: no open for {ticker} on {backfill_date_str} — using prior close as entry")

    # --- Size positions (uses the overridden close as entry) ---
    print(">>> [BACKFILL] Sizing positions...")
    sized = size_positions(cands_for_sizing)
    print(f"    Candidates after sizing: {len(sized)}")

    if sized.empty:
        print("!!! No viable candidates after position sizing (ATR cap or zero shares).")
        return None

    # sector/company columns for reference (sized["sector"] is already set by
    # add_signals via sector_map; company still needs to be added)
    sized["company"] = sized["ticker"].map(company_map).fillna("")

    # --- Build output DataFrame with canonical schema ---
    out_df = sized.rename(columns={
        "ticker":       "Ticker",
        "ret_1w":       "Return 1W",
        "rsi_14":       "RSI 14",
        "signal_score": "Signal Score",
        "entry_price":  "Entry Price",
        "stop_price":   "Stop Price",
        "shares":       "Shares",
        "target_price": "Target Price",
    }).copy()
    out_df["Regime"] = regime
    out_df = out_df.reset_index(drop=True)
    out_df.insert(0, "Rank", range(1, len(out_df) + 1))

    export_cols = [
        "Rank", "Ticker", "Signal Score", "Entry Price", "Stop Price",
        "Target Price", "Shares", "Return 1W", "RSI 14", "Regime",
    ]
    present_cols = [c for c in export_cols if c in out_df.columns]
    out_df = out_df[present_cols]
    for col in ["Signal Score", "Entry Price", "Stop Price", "Target Price", "Return 1W", "RSI 14"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].round(2)

    # --- Save CSV ---
    out_path = CANDIDATE_DIR / f"candidates_backfill_{backfill_date_str}.csv"
    out_df.to_csv(out_path, index=False)
    print(f">>> [BACKFILL] Saved: {out_path.name}")

    # --- Terminal summary table ---
    print(f"\n{'='*60}")
    print(f"  [BACKFILL] Top candidates — {backfill_date_str}  (Regime: {regime})")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Ticker':<8} {'Score':>7} {'Entry':>8} {'Stop':>8} {'Target':>8} {'Shares':>6}")
    print(f"  {'-'*54}")
    for _, row in out_df.iterrows():
        print(
            f"  {int(row['Rank']):<5} {str(row['Ticker']):<8}"
            f" {float(row.get('Signal Score', 0)):>7.3f}"
            f" {float(row.get('Entry Price', 0)):>8.2f}"
            f" {float(row.get('Stop Price', 0)):>8.2f}"
            f" {float(row.get('Target Price', 0)):>8.2f}"
            f" {int(row.get('Shares', 0)):>6}"
        )
    print()

    return out_path


# ------------------- BUCKET 2 OPPORTUNITY SCANNER (--opportunities) -------------------
#
# Runs AFTER the V2 model, over an expanded universe (Russell 1000 names not
# already in the S&P 500, Russell Midcap, and a curated ETF list), using two
# independent momentum/volume-surge signals. Entirely separate from V2:
# different universe, different filters, discretionary fixed-dollar sizing
# instead of risk-based sizing. Never touches V2's candidates CSV, Firestore,
# or filter logic — only ever reads master_universe.json and the cached
# sp500_tickers.csv, and writes its own opportunities_YYYY-MM-DD.csv.

_opp_price_cache: pd.DataFrame | None = None  # per-process download cache


def load_sp500_cache() -> list[str]:
    """Reads the S&P 500 ticker cache written by build_universe() — avoids a second scrape."""
    if not SP500_CACHE_PATH.exists():
        return []
    return pd.read_csv(SP500_CACHE_PATH, header=None)[0].dropna().astype(str).tolist()


def load_opportunity_universe() -> tuple[list[str], dict[str, set[str]]]:
    """
    Builds the Bucket 2 scan universe from master_universe.json: R1000 extra
    + MidCap + ETFs, deduplicated, excluding anything already in the S&P 500
    (V2's own universe) to avoid double-counting.

    Returns (opp_universe, universe_sets) — universe_sets holds the raw
    (unfiltered) per-bucket sets, used later to label each candidate's
    source universe in the output CSV.
    """
    if not MASTER_UNIVERSE_PATH.exists():
        print(f"!!! {MASTER_UNIVERSE_PATH} not found — run stock_model/scripts/build_universe.py first")
        return [], {"r1000_extra": set(), "midcap": set(), "etfs": set()}

    with open(MASTER_UNIVERSE_PATH) as f:
        universe = json.load(f)

    r1000_extra = universe.get("r1000_extra", [])
    midcap = universe.get("midcap", [])
    etfs = universe.get("etfs", [])

    sp500_set = set(load_sp500_cache())
    combined = list(dict.fromkeys(r1000_extra + midcap + etfs))  # de-dupe, preserve order
    opp_universe = [t for t in combined if t not in sp500_set]

    print(f">>> Opportunity universe: {len(opp_universe)} tickers")
    print(f"     (R1000 extra: {len(r1000_extra)}, MidCap: {len(midcap)}, ETFs: {len(etfs)})")

    return opp_universe, {
        "r1000_extra": set(r1000_extra),
        "midcap": set(midcap),
        "etfs": set(etfs),
    }


def download_opportunity_universe(tickers: list[str]) -> pd.DataFrame:
    """
    Downloads OHLCV for the opportunity universe in batches of
    OPP_DOWNLOAD_BATCH_SIZE (same fetch_price_data/yfinance pattern the V2
    model uses). Cached at module scope for the lifetime of the process.
    """
    global _opp_price_cache
    if _opp_price_cache is not None:
        return _opp_price_cache

    print(">>> Downloading opportunity universe data...")
    batches = [
        tickers[i:i + OPP_DOWNLOAD_BATCH_SIZE]
        for i in range(0, len(tickers), OPP_DOWNLOAD_BATCH_SIZE)
    ]
    frames = []
    for i, batch in enumerate(batches, start=1):
        print(f"    Batch {i}/{len(batches)} ({len(batch)} tickers)...")
        df = fetch_price_data(tickers=batch, period=PRICE_HISTORY_PERIOD, interval=PRICE_HISTORY_INTERVAL)
        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    valid_tickers = combined["ticker"].nunique() if not combined.empty else 0
    print(f"    Tickers with valid data: {valid_tickers}")

    _opp_price_cache = combined
    return combined


def compute_opportunity_signals(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes ret_3w, ret_1w, vol_surge, atr_14, atr_pct, dist_52w, accel,
    and avg_dv_m for the most recent trading day of each ticker.
    """
    if price_data.empty:
        return pd.DataFrame()

    frames = []
    for ticker, grp in price_data.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        g["ret_1w"] = g["close"].pct_change(5, fill_method=None)
        g["ret_3w"] = g["close"].pct_change(15, fill_method=None)
        g["atr_14"] = compute_atr(g, period=14)
        g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()
        g["high_252"] = g["high"].rolling(252, min_periods=1).max()
        frames.append(g)

    df = pd.concat(frames)
    df.sort_values(["ticker", "date"], inplace=True)
    latest = df.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)

    latest["price"] = latest["close"]
    latest["vol_surge"] = latest["volume"] / latest["avg_vol_20"].replace(0, np.nan)
    latest["atr_pct"] = latest["atr_14"] / latest["price"]
    latest["dist_52w"] = (latest["close"] - latest["high_252"]) / latest["high_252"]
    latest["accel"] = latest["ret_1w"] > (latest["ret_3w"] / 3.0)
    latest["avg_dv_m"] = (latest["avg_vol_20"] * latest["price"]) / 1e6

    return latest


def apply_opportunity_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Applies the two Bucket 2 signal definitions and labels qualifying rows."""
    if signals.empty:
        return signals

    df = signals.copy()

    momentum = (
        (df["ret_3w"] >= OPP_MOMENTUM_MIN_RET_3W)
        & df["accel"].fillna(False)
        & (df["avg_dv_m"] >= OPP_MIN_AVG_DV_M)
        & (df["price"] >= OPP_MIN_PRICE)
        & (df["atr_pct"] <= OPP_MAX_ATR_PCT)
    ).fillna(False)

    volume_break = (
        (df["vol_surge"] >= OPP_VOLUME_MIN_SURGE)
        & (df["ret_3w"] >= OPP_VOLUME_MIN_RET_3W)
        & (df["avg_dv_m"] >= OPP_MIN_AVG_DV_M)
        & (df["price"] >= OPP_MIN_PRICE)
        & (df["atr_pct"] <= OPP_MAX_ATR_PCT)
    ).fillna(False)

    def label(is_momentum: bool, is_volume: bool) -> str | None:
        if is_momentum and is_volume:
            return "MOMENTUM+VOL"
        if is_momentum:
            return "MOMENTUM"
        if is_volume:
            return "VOLUME_BREAK"
        return None

    df["signal"] = [label(m, v) for m, v in zip(momentum, volume_break)]
    return df[df["signal"].notna()].copy()


def score_and_rank_opportunities(candidates: pd.DataFrame, top_n: int = OPP_TOP_N) -> pd.DataFrame:
    """Scores by momentum_score, sorts descending, returns the top N across both signal types."""
    if candidates.empty:
        return candidates

    df = candidates.copy()
    df["momentum_score"] = (
        df["ret_3w"].fillna(0.0) * 0.40
        + df["ret_1w"].fillna(0.0) * 0.30
        + (df["vol_surge"].fillna(0.0) / 10) * 0.20
        + df["accel"].fillna(False).astype(int) * 0.10
    )
    df = df.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def compute_opportunity_prices(candidates: pd.DataFrame) -> pd.DataFrame:
    """Entry/stop/target using the same 0.75x-ATR stop and 2R target as the V2 model."""
    if candidates.empty:
        return candidates

    df = candidates.copy()
    df["entry_price"] = df["price"]
    df["stop_price"] = df["entry_price"] - (ATR_STOP_MULTIPLE * df["atr_14"])
    df["target_price"] = df["entry_price"] + (2.0 * (df["entry_price"] - df["stop_price"]))
    return df


def which_universe(ticker: str, universe_sets: dict[str, set[str]]) -> str:
    if ticker in universe_sets["r1000_extra"]:
        return "R1000"
    if ticker in universe_sets["midcap"]:
        return "MidCap"
    if ticker in universe_sets["etfs"]:
        return "ETF"
    return "Other"


def get_bucket2_open_count() -> int | None:
    """
    Reads users/{uid}/data/opportunisticPlays from Firestore (same
    ADC-then-service-account pattern friday_close.py uses for
    candidatePool) and counts plays with status == 'open'.

    Returns None — never 0 — if Firebase isn't installed, isn't
    configured, or the read fails for any reason. Callers must treat
    None as "unknown" and skip the warning, not treat it as "0 open":
    this must never block or fabricate a false all-clear on candidate
    output.
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        return None

    if not FIRESTORE_USER_CONFIG_PATH.exists():
        return None

    try:
        with open(FIRESTORE_USER_CONFIG_PATH) as f:
            uid = json.load(f).get("uid")
        if not uid:
            return None

        if not firebase_admin._apps:
            try:
                firebase_admin.initialize_app()
                firestore.client()  # ADC is resolved lazily — force it now
            except Exception:
                for app in list(firebase_admin._apps.values()):
                    firebase_admin.delete_app(app)
                if FIRESTORE_KEY_PATH.exists():
                    cred = credentials.Certificate(str(FIRESTORE_KEY_PATH))
                    firebase_admin.initialize_app(cred)
                else:
                    return None

        db = firestore.client()
        doc_ref = (
            db.collection("users").document(uid)
            .collection("data").document("opportunisticPlays")
        )
        doc = doc_ref.get()
        items = doc.to_dict().get("items", []) if doc.exists else []
        return sum(1 for it in items if it.get("status") == "open")
    except Exception as exc:
        print(f"    (couldn't read Bucket 2 open-position count from Firestore: {exc})")
        return None


def print_bucket2_open_warning(open_count: int | None) -> None:
    """Change 3: graduated warning, never a hard cap. Prints nothing at 0-1 open or unknown count."""
    if open_count is None or open_count <= 1:
        return
    if open_count <= 3:
        print(f">>> ⚠ {open_count} opportunistic plays currently open — adding more increases concentration")
    else:
        print(f">>> ⚠ {open_count} opportunistic plays open — high concentration, be selective")


def print_opportunity_table(
    candidates: pd.DataFrame,
    universe_sets: dict[str, set[str]],
    open_count: int | None,
) -> None:
    print(">>> " + "=" * 60)
    print(">>> BUCKET 2 — OPPORTUNITY SCANNER")
    print(">>> " + "=" * 60)
    print(">>> Signal 1 (MOMENTUM): 3W>=30% + accel + dv>$50M")
    print(">>> Signal 2 (VOLUME_BREAK): vol>=5x + 3W>=10% + dv>$50M")
    print(">>> Universe: R1000 extra + MidCap + ETFs")
    print(">>>")

    if candidates.empty:
        print(">>> No Bucket 2 opportunities this week")
        return

    print(
        f">>> {'Rank':<6}{'Ticker':<8}{'Universe':<9}{'Signal':<14}{'Tier':<8}{'3W Ret':<8}"
        f"{'1W Ret':<8}{'Vol':<6}{'Entry':<8}{'Stop':<8}{'Target':<8}{'DV($M)':<8}{'Accel':<7}{'Warning'}"
    )
    print(">>> " + "-" * 115)
    for i, row in enumerate(candidates.itertuples(), start=1):
        accel_mark = "✓" if row.accel else "✗"
        ret_3w_s = f"{row.ret_3w*100:.1f}%"
        ret_1w_s = f"{row.ret_1w*100:.1f}%"
        vol_s = f"{row.vol_surge:.1f}x"
        universe = which_universe(row.ticker, universe_sets)
        warning = getattr(row, "Warning", "") or ""
        print(
            f">>> {i:<6}{row.ticker:<8}{universe:<9}{row.signal:<14}{row.Tier:<8}"
            f"{ret_3w_s:<8}{ret_1w_s:<8}{vol_s:<6}"
            f"{row.entry_price:<8.2f}{row.stop_price:<8.2f}{row.target_price:<8.2f}"
            f"{row.avg_dv_m:<8.0f}{accel_mark:<7}{warning}"
        )
    print(">>>")
    print_concentration_warnings(candidates)
    print(">>>")
    print(f">>> Sizing: {OPP_SIZING_NOTE.split(' (')[0]} per play (fixed)")
    print_bucket2_open_warning(open_count)
    print(">>> Size each play at $300-500 deployed — your judgment on how many to hold simultaneously")
    print(">>> No mandatory Friday exit — hold until thesis breaks")
    print(">>> These are NOT systematic V2 trades — discretionary only")


def save_opportunities_csv(
    candidates: pd.DataFrame,
    date_str: str,
    universe_sets: dict[str, set[str]],
) -> Path:
    """Saves Results/candidates/opportunities_YYYY-MM-DD.csv (headers-only if no candidates)."""
    out_path = CANDIDATE_DIR / f"opportunities_{date_str}.csv"

    if candidates.empty:
        pd.DataFrame(columns=OPP_CSV_COLUMNS).to_csv(out_path, index=False)
        return out_path

    df = candidates.reset_index(drop=True).copy()
    out = pd.DataFrame({
        "Rank":         range(1, len(df) + 1),
        "Ticker":       df["ticker"],
        "Universe":     df["ticker"].apply(lambda t: which_universe(t, universe_sets)),
        "Signal":       df["signal"],
        "Tier":         df["Tier"],
        "Score":        df["momentum_score"].round(4),
        "Return 3W":    df["ret_3w"].round(4),
        "Return 1W":    df["ret_1w"].round(4),
        "Vol Surge":    df["vol_surge"].round(2),
        "Entry Price":  df["entry_price"].round(2),
        "Stop Price":   df["stop_price"].round(2),
        "Target Price": df["target_price"].round(2),
        "Avg DV ($M)":  df["avg_dv_m"].round(1),
        "Accel":        df["accel"],
        "Dist 52W":     df["dist_52w"].round(4),
        "ATR Pct":      df["atr_pct"].round(4),
        "Warning":      df["Warning"],
    })
    out.to_csv(out_path, index=False)
    return out_path


def run_opportunity_scanner(dry_run: bool = False) -> Path | None:
    """
    Entry point for --opportunities. Reads master_universe.json + the S&P
    500 cache (never re-scrapes), scans, prints, and saves — independent of
    everything run_daily_model() just did.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")

    opp_universe, universe_sets = load_opportunity_universe()

    if opp_universe:
        price_data = download_opportunity_universe(opp_universe)
        signals = compute_opportunity_signals(price_data)
        flagged = apply_opportunity_signals(signals)
        ranked = score_and_rank_opportunities(flagged)
        priced = compute_opportunity_prices(ranked)
    else:
        priced = pd.DataFrame()

    if not priced.empty:
        priced["Tier"] = priced.apply(lambda r: opportunity_tier(r["ret_3w"], r["vol_surge"]), axis=1)
        priced["Warning"] = compute_concentration_warnings(priced)

    open_count = get_bucket2_open_count()
    print_opportunity_table(priced, universe_sets, open_count)

    if dry_run:
        print(">>> DRY RUN — opportunities CSV not saved")
        return None

    out_path = save_opportunities_csv(priced, today_str, universe_sets)
    print(f">>> Opportunities saved to: {out_path}")
    return out_path


# ------------------- CSV SCHEMA VALIDATION -------------------

REQUIRED_COLUMNS = [
    "Ticker",
    "Sector",
    "Signal Score",
    "Entry Price",
    "Stop Price",
    "Target Price",
    "Shares",
    "Return 1W",
    "RSI 14",
    "Regime",
]


def validate_csv_schema(df: pd.DataFrame) -> None:
    """
    Confirm the candidates DataFrame contains every column the dashboard
    expects before writing to disk.  Raises ValueError listing any missing
    columns so the pipeline fails loudly — not silently on the dashboard.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV schema broken — missing columns: {missing}")
    print("✓ CSV schema validated — all required columns present")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Swing Bet Model")
    parser.add_argument(
        "--backfill",
        metavar="DATE",
        help="Generate backfill candidates for a past date (YYYY-MM-DD). Regime filter is bypassed.",
    )
    parser.add_argument(
        "--opportunities",
        action="store_true",
        help="After the V2 model, run the Bucket 2 opportunity scanner (R1000 extra + MidCap + ETFs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --opportunities, print results without saving opportunities_YYYY-MM-DD.csv.",
    )
    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.backfill)
    else:
        # Set log_trades=True if you want to automatically append to execution_log.csv
        run_daily_model(log_trades=False)
        if args.opportunities:
            run_opportunity_scanner(dry_run=args.dry_run)
