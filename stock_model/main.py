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

from collections import Counter
from datetime import datetime
from pathlib import Path
import os

import pandas as pd

from data_loader import get_sp500_tickers, download_ohlcv_data, get_sector_map, get_company_map
from model_logic import add_signals, select_top_candidates, classify_current_regime
from position_sizing import (
    RiskConfig,
    compute_stop_price_long,
    compute_position_size_long,
)
from trade_logger import append_execution_log


# ------------------- CONFIG -------------------

# Approximate account equity (update this as needed)
ACCOUNT_EQUITY = 10_000.0

# Risk settings (can be tuned later)
RISK_CONFIG = RiskConfig(
    account_equity=ACCOUNT_EQUITY,
    max_risk_per_trade_pct=0.005,      # 0.5% of equity
    max_dollar_risk_per_trade=50.0,    # $50 max per trade
    max_notional_per_trade_pct=0.10,   # 10% of equity
)

# Signal / candidate settings
MAX_TRADES_PER_DAY = 5
CANDIDATE_POOL_SIZE = 20   # rows written to candidate_pool_YYYY-MM-DD.csv
PRICE_HISTORY_PERIOD = "6mo"   # used by download_ohlcv_data
PRICE_HISTORY_INTERVAL = "1d"
FORCE_REFRESH_TICKERS = True

# File paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent   # Trading/stock_model/
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
RAW_DATA_DIR = DATA_DIR / "raw_data"
EXECUTION_LOG_PATH = RESULTS_DIR / "execution_log.csv"
CANDIDATE_DIR = RESULTS_DIR / "candidates"


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
    Download recent OHLCV data and apply basic price/volume filters.
    Returns a long-format DataFrame with all eligible tickers.
    """
    df = download_ohlcv_data(
        tickers=tickers,
        period=PRICE_HISTORY_PERIOD,
        interval=PRICE_HISTORY_INTERVAL,
        min_price=5.0,
        min_volume=1_000_000,
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


def compute_trade_candidates(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signals and select the top trade candidates.
    """
    signals = add_signals(price_data)
    candidates = select_top_candidates(signals, max_trades=MAX_TRADES_PER_DAY)
    return candidates


def size_positions(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Given a candidates DataFrame (from model_logic.select_top_candidates),
    compute stop levels and position sizes for each candidate.
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

        stop = compute_stop_price_long(entry_price=entry, atr=atr, atr_multiple=1.5)

        # None means stop distance exceeded the 3% cap — reject this candidate
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


def save_daily_candidates(candidates: pd.DataFrame, regime: str = "UNKNOWN") -> Path:
    """
    Save the sized candidates to a dated CSV and return the path.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = CANDIDATE_DIR / f"candidates_{today_str}.csv"

    if candidates.empty:
        # Still create an empty file so you know the model ran
        candidates.to_csv(out_path, index=False)
    else:
        # Tag with current regime
        candidates["regime"] = regime
        # Round all numeric columns to 2 decimals for readability
        for col in candidates.select_dtypes(include=["float", "int"]).columns:
            candidates[col] = candidates[col].round(2)
        # Rename columns for readability
        candidates = candidates.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
            "ticker": "Ticker",
            "company": "Company",
            "sector": "Sector",
            "avg_vol_20": "20d Avg Volume",
            "ret_1w": "Return 1W",
            "ret_3w": "Return 3W",
            "rsi_14": "RSI 14",
            "atr_14": "ATR 14",
            "price": "Price",
            "atr_pct": "ATR %",
            "is_eligible": "Eligible",
            "score_raw": "Raw Score",
            "signal_score": "Signal Score",
            "direction": "Direction",
            "entry_price": "Entry Price",
            "stop_price": "Stop Price",
            "shares": "Shares",
            "target_price": "Target Price",
            "rank_1w": "Rank 1W",
            "rank_3w": "Rank 3W",
            "rank_vol_surge": "Rank Vol Surge",
            "regime": "Regime",
        })
        # Add rank as first column
        candidates = candidates.reset_index(drop=True)
        candidates.insert(0, 'Rank', range(1, len(candidates) + 1))
        # Reorder columns: Rank, Ticker, Sector, Regime, Shares, Entry Price, Stop Price, Target Price, then the rest
        priority_cols = ["Rank", "Ticker", "Company", "Sector", "Regime", "Shares", "Entry Price", "Stop Price", "Target Price"]
        remaining_cols = [col for col in candidates.columns if col not in priority_cols]
        candidates = candidates[priority_cols + remaining_cols]
        validate_csv_schema(candidates)
        candidates.to_csv(out_path, index=False)

    return out_path


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
    the final slate.  Filters are checked in priority order so each ticker
    gets exactly one label.
    """
    price        = row.get("price", 0) or 0
    avg_vol      = row.get("avg_vol_20", 0) or 0
    rsi          = row.get("rsi_14", float("nan"))
    stop_dist    = row.get("stop_dist_pct", float("nan"))

    if pd.isna(price) or price < 5.0 or price > 300.0:
        return "rejected_price"
    if pd.isna(avg_vol) or avg_vol < 1_000_000:
        return "rejected_volume"
    if pd.isna(rsi) or rsi < 30 or rsi > 70:
        return "rejected_rsi"
    if pd.isna(stop_dist) or stop_dist > 0.03:
        return "rejected_stop"
    return "not_selected"


def save_candidate_pool(
    signals: pd.DataFrame,
    final_candidates: pd.DataFrame,
    sector_map: dict,
    company_map: dict,
) -> Path:
    """
    Write Results/candidates/candidate_pool_YYYY-MM-DD.csv with the top
    CANDIDATE_POOL_SIZE tickers by signal_score and their status.
    Also prints a rejection-reason summary to the terminal.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = CANDIDATE_DIR / f"candidate_pool_{today_str}.csv"

    final_tickers = set(final_candidates["ticker"].tolist()) if not final_candidates.empty else set()

    pool = signals.sort_values("signal_score", ascending=False).head(CANDIDATE_POOL_SIZE).copy()

    pool["status"] = pool.apply(
        lambda r: "passed" if r["ticker"] in final_tickers else _rejection_status(r),
        axis=1,
    )
    pool["company"]      = pool["ticker"].map(company_map).fillna("")
    pool["sector"]       = pool["ticker"].map(sector_map).fillna("Other")
    pool["entry_price"]  = pool["close"]
    pool["stop_price"]   = pool["entry_price"] - 1.5 * pool["atr_14"]
    pool["stop_pct"]     = (pool["stop_dist_pct"] * 100).round(2)
    pool["target_price"] = pool["entry_price"] + 2.0 * (pool["entry_price"] - pool["stop_price"])

    out = pool[[
        "ticker", "company", "sector",
        "entry_price", "stop_price", "stop_pct", "target_price",
        "signal_score", "status",
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
    })

    for col in ["Entry Price", "Stop Price", "Stop Pct", "Target Price", "Signal Score"]:
        out[col] = out[col].round(2)

    out.to_csv(out_path, index=False)

    # Terminal summary
    counts = pool["status"].value_counts()
    status_order = ["passed", "not_selected", "rejected_stop", "rejected_rsi", "rejected_volume", "rejected_price"]
    print(f">>> Candidate pool summary (top {CANDIDATE_POOL_SIZE} by score):")
    for label in status_order:
        n = int(counts.get(label, 0))
        if n:
            print(f"    {label}: {n}")

    return out_path


def run_daily_model(log_trades: bool = False) -> Path | None:
    """
    Convenience wrapper to run the full daily pipeline end-to-end.
    Returns the path to the candidates CSV, or None if no candidates.
    """
    ensure_directories()

    print(">>> Building S&P 500 universe...")
    tickers = build_universe()
    print(f"    Universe size: {len(tickers)} tickers")

    print(">>> Downloading price data...")
    price_data = load_price_data(tickers)
    print(f"    Price data shape: {price_data.shape}")

    if price_data.empty:
        print("!!! No price data returned after filters. Exiting.")
        return None

    # --- Regime filter (Variation A) ---
    regime = classify_current_regime(price_data)
    print(f">>> Current market regime: {regime}")

    if regime in ("BULL_TREND", "BULL_VOLATILE"):
        print("!!! REGIME FILTER ACTIVE — No trades this week.")
        print(f"    Regime '{regime}' has historically shown no edge (PF < 1.05).")
        print("    Sit out this week. Re-run next Monday.")
        empty = pd.DataFrame()
        out_path = save_daily_candidates(empty, regime=regime)
        print(f"    Empty candidates file saved to: {out_path}")
        return out_path

    print(">>> Computing signals & selecting candidates...")
    signals = add_signals(price_data)
    candidates = select_top_candidates(signals, max_trades=MAX_TRADES_PER_DAY)
    print(f"    Candidates before sizing: {len(candidates)}")

    if candidates.empty:
        print("!!! No trade candidates found today.")
        out_path = save_daily_candidates(candidates, regime=regime)
        print(f"    Empty candidates file saved to: {out_path}")
        return out_path

    print(">>> Sizing positions...")
    sized_candidates = size_positions(candidates)
    print(f"    Candidates after sizing (shares > 0): {len(sized_candidates)}")

    print(">>> Looking up sectors and company names...")
    sector_map = get_sector_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))
    company_map = get_company_map(save_path=str(RAW_DATA_DIR / "sector_map.csv"))
    sized_candidates["sector"] = sized_candidates["ticker"].map(sector_map).fillna("Other")
    sized_candidates["company"] = sized_candidates["ticker"].map(company_map).fillna("")
    print(f"    Sectors: {sized_candidates[['ticker','sector']].set_index('ticker')['sector'].to_dict()}")

    out_path = save_daily_candidates(sized_candidates, regime=regime)
    print(f">>> Daily candidates saved to: {out_path}")

    # Sector concentration check
    sector_counts = Counter(sized_candidates['sector'])
    most_common_sector, most_common_count = sector_counts.most_common(1)[0]

    if most_common_count >= 3:
        print(f"\n⚠️  SECTOR CONCENTRATION: {most_common_count}/5 candidates are {most_common_sector}")
        print(f"   Main slate saved normally.")
        print(f"\n   Want a diversified backup slate from ranks 6-10?")
        print(f"   1 = Yes    2 = No")
        choice = input("   Enter 1 or 2: ").strip()

        if choice == '1':
            df_all_sorted = signals.sort_values('signal_score', ascending=False).reset_index(drop=True)
            df_remaining = df_all_sorted.iloc[MAX_TRADES_PER_DAY:MAX_TRADES_PER_DAY + 15].reset_index(drop=True)
            backup_raw = df_remaining.head(5).copy()

            backup_sized = size_positions(backup_raw)

            if not backup_sized.empty:
                backup_sized['sector'] = backup_sized['ticker'].map(sector_map).fillna('Other')
                backup_sized['company'] = backup_sized['ticker'].map(company_map).fillna('')

                backup_sector_counts = Counter(backup_sized['sector'])
                backup_top_sector, backup_top_count = backup_sector_counts.most_common(1)[0]
                if backup_top_count >= 4:
                    print(f"\n⚠️  Backup slate is also heavily concentrated ({backup_top_count}/5 are {backup_top_sector})")
                    print(f"   This may be a sector rotation week.")
                    print(f"   Recommendation: take only ranks 1-2 or 1-3 from the main slate instead of a full 5-trade week.")
                    print(f"   Saving backup anyway for reference.\n")

                backup_sized['regime'] = regime
                backup_sized = backup_sized.rename(columns={
                    "ticker": "Ticker", "company": "Company", "sector": "Sector",
                    "ret_1w": "Return 1W", "rsi_14": "RSI 14",
                    "signal_score": "Signal Score", "entry_price": "Entry Price",
                    "stop_price": "Stop Price", "shares": "Shares",
                    "target_price": "Target Price", "regime": "Regime",
                })
                backup_sized = backup_sized.reset_index(drop=True)
                backup_sized.insert(0, 'Rank', range(6, 6 + len(backup_sized)))

                backup_cols = ["Rank", "Ticker", "Signal Score", "Entry Price", "Stop Price",
                               "Target Price", "Shares", "Return 1W", "RSI 14", "Regime"]
                backup_cols_present = [c for c in backup_cols if c in backup_sized.columns]
                backup_out = backup_sized[backup_cols_present].copy()

                for col in ["Signal Score", "Entry Price", "Stop Price", "Target Price", "Return 1W", "RSI 14"]:
                    if col in backup_out.columns:
                        backup_out[col] = backup_out[col].round(2)

                today_str = datetime.today().strftime("%Y-%m-%d")
                backup_path = CANDIDATE_DIR / f"candidates_backup_{today_str}.csv"
                backup_out.to_csv(backup_path, index=False)
                print(f"\n✅ Backup slate saved: candidates_backup_{today_str}.csv")
                print(f"   Load via Load Monday Slate button if you want an alternative this week.")
            else:
                print("   No viable backup candidates found after sizing.")
        else:
            print("Done.")

    pool_path = save_candidate_pool(signals, sized_candidates, sector_map, company_map)
    print(f">>> Candidate pool saved to: {pool_path}")

    maybe_log_trades(sized_candidates, log_trades=log_trades)

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
    # Set log_trades=True if you want to automatically append to execution_log.csv
    run_daily_model(log_trades=False)
