"""
backtest_bucket2_chase_threshold.py

Standalone research backtest for Bucket 2 (the opportunistic momentum /
volume-break scanner, `--opportunities` in main.py). Answers one question:

    How far above the signal-day closing price can you pay to enter a
    Bucket 2 trade (a "chase premium") before the strategy's edge degrades
    or disappears?

This exists because MAX_ENTRY_PREMIUM_PCT = 0.006 (0.6%) in stock_model/main.py
was carried over from the older, separate V2 (S&P 500) systematic model and
has never actually been validated against Bucket 2's own signal set, which
trades much higher-volatility names (4-7%+ daily ATR candidates). This script
does NOT touch that constant or any other existing file — it only reads
data and writes new output files under Results/backtest/bucket2_chase/.

Signal logic, stop/target formula, and universe construction are reproduced
here (not imported from main.py, to avoid dragging in its argparse/Firebase
pipeline) but are copied verbatim from:
    stock_model/main.py          (constants ~line 95-160, plus
                                   compute_opportunity_signals,
                                   apply_opportunity_signals,
                                   compute_opportunity_prices)
    stock_model/model_logic.py   (compute_atr -- imported directly, not
                                   reimplemented, so the ATR math can never
                                   silently drift from the live scanner)
as of 2026-08-25. If those change later, re-diff this file against them.

THE EXPERIMENT
---------------
For every Bucket 2 signal event (a maximal run of consecutive qualifying
days collapses to one event, anchored to the first day):

  1. Look forward up to FILL_WINDOW_DAYS (5) trading days for a day whose
     LOW touches signal_price * (1 + threshold) or lower.
  2. If found: fill at min(that day's open, the capped price) -- i.e. a
     limit-style fill at the cap or better. If not found in the window,
     the event is MISSED at that threshold.
  3. From the fill day, walk forward using daily OHLC until stop hit,
     target hit, or 21 calendar days elapse -- whichever comes first.
  4. Size at a fixed $35 risk (same convention as backtest_engine.py),
     shares = 35 / (signal_price - stop) -- sized off SIGNAL-day risk,
     not fill-day risk, since that's what you'd have decided pre-trade.
     0.1% slippage per side, also matching backtest_engine.py.

Swept thresholds: 0%, 0.5%, 1%, 1.5%, 2%, 3%, 4%, 5%, 7%, 10%, uncapped.

KNOWN, DELIBERATE SIMPLIFICATIONS (read before trusting the numbers)
---------------------------------------------------------------------
- No discretionary "thesis broke" exit is simulated -- only stop / target /
  21-day time. A real discretionary trader might exit earlier for reasons
  a backtest can't see. This could bias results either direction; we don't
  attempt to correct for it.
- Stop/target are anchored to the SIGNAL day's close and ATR, exactly as
  the live scanner computes them (its Stop/Target columns are fixed at
  scan time) -- even though the actual fill price differs once a chase
  premium is introduced. This is intentional, not an oversight.
- Events too close to "today" to have had a full fill-window + hold-period
  play out are excluded from the aggregate stats (flagged `too_recent` in
  trades.csv) rather than force-exited early, which would bias results.
- A ticker's data ending mid-hold (e.g. delisting) is also excluded from
  aggregate stats (`data_end_censored`) for the same reason, rather than
  silently booking a fabricated exit.
- Sample sizes are what they are. This script does not pad small buckets
  with any statistical trickery -- read the fill counts before trusting a
  profit factor.

Run:
    python3 stock_model/backtests/backtest_bucket2_chase_threshold.py

Outputs -> stock_model/Results/backtest/bucket2_chase/
    threshold_summary.csv   one row per (threshold, cut) with aggregate stats
    trades.csv               one row per (signal event, threshold) attempt,
                              including MISSED / excluded rows, fully auditable
    summary.txt               human-readable report (also printed to stdout)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Make model_logic importable (it lives in stock_model/, this script sits
#    in stock_model/backtests/). compute_atr is reused verbatim per the
#    task spec so ATR can never silently drift from the live scanner's.
STOCK_MODEL_DIR = Path(__file__).resolve().parents[1]
if str(STOCK_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(STOCK_MODEL_DIR))
from model_logic import compute_atr  # noqa: E402


# ─── Paths (all NEW files/dirs -- nothing here overlaps existing outputs) ──
PROJECT_ROOT = STOCK_MODEL_DIR
DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
CACHE_DIR = DATA_DIR / "backtest_cache"
CACHE_FILE = CACHE_DIR / "bucket2_chase_ohlcv_3yr.csv"
MASTER_UNIVERSE_PATH = RAW_DATA_DIR / "master_universe.json"
SP500_CACHE_PATH = RAW_DATA_DIR / "sp500_tickers.csv"
RESULTS_DIR = PROJECT_ROOT / "Results" / "backtest" / "bucket2_chase"

# ─── Lookback window ────────────────────────────────────────────────────────
END_DATE = "2026-08-25"     # today
START_DATE = "2023-08-25"   # 3 years back, per task spec
DOWNLOAD_BATCH_SIZE = 100   # mirrors OPP_DOWNLOAD_BATCH_SIZE in main.py

# ─── Bucket 2 signal definitions (copied verbatim from main.py) ────────────
OPP_MOMENTUM_MIN_RET_3W = 0.30
OPP_VOLUME_MIN_SURGE = 5.0
OPP_VOLUME_MIN_RET_3W = 0.10
OPP_MIN_AVG_DV_M = 50.0
OPP_MIN_PRICE = 10.0
OPP_MAX_ATR_PCT = 0.20
ATR_STOP_MULTIPLE = 0.75    # matches model_logic.py's V2 constant, same value Bucket 2 uses
REWARD_RISK = 2.0

# ─── Trade management / experiment knobs ───────────────────────────────────
FILL_WINDOW_DAYS = 5              # trading days to look for a fill after signal
TIME_EXIT_CALENDAR_DAYS = 21
RISK_PER_TRADE = 35.0             # matches backtest_engine.py's RISK_PER_TRADE
SLIPPAGE = 0.001                  # 0.1% per side, matches backtest_engine.py
MIN_TRUSTWORTHY_FILLED = 20       # below this, flag the PF as unreliable

CHASE_THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, float("inf")]


def threshold_label(t: float) -> str:
    return "uncapped" if math.isinf(t) else f"{t * 100:.1f}%"


# ─── Universe ────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    """R1000-extra + MidCap + ETFs, minus anything already in the S&P 500
    (V2's universe) -- same construction as main.py's load_opportunity_universe()."""
    with open(MASTER_UNIVERSE_PATH) as f:
        universe = json.load(f)

    r1000_extra = universe.get("r1000_extra", [])
    midcap = universe.get("midcap", [])
    etfs = universe.get("etfs", [])

    sp500 = set(pd.read_csv(SP500_CACHE_PATH, header=None)[0].dropna().astype(str))
    combined = list(dict.fromkeys(r1000_extra + midcap + etfs))
    opp_universe = [t for t in combined if t not in sp500]

    print(f">>> Bucket 2 universe: {len(opp_universe)} tickers "
          f"(R1000 extra {len(r1000_extra)}, MidCap {len(midcap)}, ETFs {len(etfs)}, "
          f"minus {len(sp500)} S&P 500 names)")
    return opp_universe


# ─── Data download (adapted from backtest_engine.py's proven yfinance parser) ─

def _parse_yf_download(raw: pd.DataFrame, batch: list[str]) -> list[pd.DataFrame]:
    """Parse a yfinance multi-ticker download into per-ticker DataFrames.
    Handles both the >=0.2.37 (price-type-in-level-0) and older APIs."""
    frames: list[pd.DataFrame] = []

    if not isinstance(raw.columns, pd.MultiIndex):
        df_t = raw.copy().reset_index()
        df_t.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Adj Close": "adj_close",
            "Volume": "volume",
        }, inplace=True)
        if "close" in df_t.columns and not df_t["close"].isna().all():
            df_t["ticker"] = batch[0]
            needed = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if all(c in df_t.columns for c in needed):
                frames.append(df_t[needed])
        return frames

    lvl0 = raw.columns.get_level_values(0).unique().tolist()

    if any(x in lvl0 for x in ["Close", "Open", "High", "Low"]):
        for ticker in batch:
            try:
                df_t = raw.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                continue
            if df_t.empty or df_t.get("Close", pd.Series(dtype=float)).isna().all():
                continue
            df_t.index.name = "date"
            df_t = df_t.reset_index()
            df_t.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Adj Close": "adj_close",
                "Volume": "volume",
            }, inplace=True)
            df_t["ticker"] = ticker
            needed = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if all(c in df_t.columns for c in needed):
                frames.append(df_t[needed])
    else:
        for ticker in batch:
            if ticker not in lvl0:
                continue
            try:
                df_t = raw[ticker].copy()
            except KeyError:
                continue
            if df_t.empty:
                continue
            df_t.index.name = "date"
            df_t = df_t.reset_index()
            df_t.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Adj Close": "adj_close",
                "Volume": "volume",
            }, inplace=True)
            df_t["ticker"] = ticker
            needed = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if all(c in df_t.columns for c in needed):
                frames.append(df_t[needed])

    return frames


def download_universe(tickers: list[str]) -> pd.DataFrame:
    """Downloads 3yr daily OHLCV in batches of 100, caches to a new CSV
    (doesn't touch backtest_engine.py's own ohlcv_5yr.csv cache)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print(f">>> Loading cached price data from {CACHE_FILE.name} ...")
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        print(f"    {len(df):,} rows, {df['ticker'].nunique()} tickers")
        return df

    print(f">>> Downloading {START_DATE} -> {END_DATE} OHLCV for {len(tickers)} tickers...")
    batches = [tickers[i:i + DOWNLOAD_BATCH_SIZE] for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE)]
    frames: list[pd.DataFrame] = []
    failed_batches = 0

    for i, batch in enumerate(batches, start=1):
        print(f"    Batch {i}/{len(batches)} ({len(batch)} tickers)...", flush=True)
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(
                    tickers=batch, start=START_DATE, end=END_DATE, interval="1d",
                    auto_adjust=False, progress=False, threads=True,
                )
                break
            except Exception as exc:
                print(f"    ! batch {i} attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        if raw is None or raw.empty:
            failed_batches += 1
            continue
        frames.extend(_parse_yf_download(raw, batch))

    if not frames:
        raise RuntimeError("No data downloaded -- check network connectivity.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)
    downloaded_tickers = df["ticker"].nunique()
    print(f"    Downloaded {len(df):,} rows for {downloaded_tickers}/{len(tickers)} tickers "
          f"({failed_batches} batch(es) failed outright)")
    df.to_csv(CACHE_FILE, index=False)
    return df


# ─── Feature computation (mirrors compute_opportunity_signals, but for the
#     FULL history per ticker, not just the latest row -- we need signal
#     days across the whole 3yr window, not just "as of today") ────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    print(">>> Computing features (ret_1w/3w, ATR, volume surge, 52w high, accel)...")
    frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").reset_index(drop=True).copy()
        g["ret_1w"] = g["close"].pct_change(5, fill_method=None)
        g["ret_3w"] = g["close"].pct_change(15, fill_method=None)
        g["atr_14"] = compute_atr(g, period=14)
        g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()
        g["high_252"] = g["high"].rolling(252, min_periods=1).max()
        g["vol_surge"] = g["volume"] / g["avg_vol_20"].replace(0, np.nan)
        g["atr_pct"] = g["atr_14"] / g["close"]
        g["dist_52w"] = (g["close"] - g["high_252"]) / g["high_252"]
        g["accel"] = g["ret_1w"] > (g["ret_3w"] / 3.0)
        g["avg_dv_m"] = (g["avg_vol_20"] * g["close"]) / 1e6
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["ticker", "date"], inplace=True)
    return out


def flag_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Applies both Bucket 2 signal definitions row-wise across full history
    (mirrors apply_opportunity_signals, but keeps every day, not just hits,
    since we need to run-length-encode consecutive flagged days below)."""
    df = df.copy()
    df["is_momentum"] = (
        (df["ret_3w"] >= OPP_MOMENTUM_MIN_RET_3W)
        & df["accel"].fillna(False)
        & (df["avg_dv_m"] >= OPP_MIN_AVG_DV_M)
        & (df["close"] >= OPP_MIN_PRICE)
        & (df["atr_pct"] <= OPP_MAX_ATR_PCT)
    ).fillna(False)

    df["is_volume_break"] = (
        (df["vol_surge"] >= OPP_VOLUME_MIN_SURGE)
        & (df["ret_3w"] >= OPP_VOLUME_MIN_RET_3W)
        & (df["avg_dv_m"] >= OPP_MIN_AVG_DV_M)
        & (df["close"] >= OPP_MIN_PRICE)
        & (df["atr_pct"] <= OPP_MAX_ATR_PCT)
    ).fillna(False)

    return df


# ─── Signal-event extraction ────────────────────────────────────────────────

def extract_events(ticker_frames: dict[str, pd.DataFrame]) -> list[dict]:
    """
    For each ticker and EACH signal type independently, collapses maximal
    runs of consecutive qualifying days into ONE event anchored to the
    first day of the run (mirrors acting on the first day you saw the
    flag, not re-buying every day it stays flagged).

    MOMENTUM and VOLUME_BREAK are run-length-encoded separately, so a
    ticker can produce a MOMENTUM event and a VOLUME_BREAK event starting
    on the same day if both conditions happen to kick in together -- they
    are tracked as two independent events. This keeps event definition
    unambiguous (no need to decide how to split a run when the label
    flips between MOMENTUM / MOMENTUM+VOL / VOLUME_BREAK from day to day)
    and satisfies the spec's instruction to track, pool, and cut results
    by signal type.
    """
    events: list[dict] = []
    for ticker, g in ticker_frames.items():
        n = len(g)
        for col, label in [("is_momentum", "MOMENTUM"), ("is_volume_break", "VOLUME_BREAK")]:
            flag_arr = g[col].to_numpy()
            if len(flag_arr) == 0:
                continue
            prev = np.empty_like(flag_arr)
            prev[0] = False
            prev[1:] = flag_arr[:-1]
            run_start = flag_arr & ~prev
            for idx in np.where(run_start)[0]:
                row = g.iloc[idx]
                atr = row["atr_14"]
                if pd.isna(atr) or atr <= 0:
                    continue
                signal_price = float(row["close"])
                stop = signal_price - ATR_STOP_MULTIPLE * float(atr)
                if stop >= signal_price:
                    continue
                target = signal_price + REWARD_RISK * (signal_price - stop)
                too_recent = (idx + FILL_WINDOW_DAYS) >= n  # not enough trailing data yet
                events.append({
                    "event_id": f"{ticker}_{label}_{row['date'].date()}",
                    "ticker": ticker,
                    "signal_type": label,
                    "signal_date": row["date"],
                    "signal_idx": int(idx),
                    "signal_price": signal_price,
                    "atr_14": float(atr),
                    "stop_price": stop,
                    "target_price": target,
                    "risk_per_share": signal_price - stop,
                    "too_recent": bool(too_recent),
                })
    return events


# ─── Per-event, per-threshold simulation ───────────────────────────────────

def simulate_event(ticker_df: pd.DataFrame, signal_idx: int, signal_price: float,
                    stop: float, target: float, threshold: float) -> dict:
    """Simulates ONE (event, threshold) attempt. Returns filled=False if no
    fill within the window, else the full fill/exit/pnl detail."""
    n = len(ticker_df)
    cap = signal_price * (1 + threshold) if not math.isinf(threshold) else float("inf")

    fill_idx = None
    for offset in range(1, FILL_WINDOW_DAYS + 1):
        idx = signal_idx + offset
        if idx >= n:
            break
        low = ticker_df["low"].iat[idx]
        if pd.notna(low) and low <= cap:
            fill_idx = idx
            break

    if fill_idx is None:
        return {"filled": False}

    fill_date = ticker_df["date"].iat[fill_idx]
    open_px = float(ticker_df["open"].iat[fill_idx])
    raw_fill = open_px if math.isinf(cap) else min(open_px, cap)
    fill_price = raw_fill * (1 + SLIPPAGE)  # buy-side slippage
    days_to_fill = fill_idx - signal_idx
    risk_per_share = signal_price - stop
    shares = max(1, math.floor(RISK_PER_TRADE / risk_per_share))

    time_limit = fill_date + pd.Timedelta(days=TIME_EXIT_CALENDAR_DAYS)

    exit_price = exit_date = exit_reason = None
    last_idx_in_window = None
    data_end_censored = False

    idx = fill_idx
    while idx < n:
        date_i = ticker_df["date"].iat[idx]
        if date_i > time_limit:
            break
        last_idx_in_window = idx
        low_i = ticker_df["low"].iat[idx]
        high_i = ticker_df["high"].iat[idx]
        if pd.notna(low_i) and low_i <= stop:
            exit_price = stop * (1 - SLIPPAGE)
            exit_date = date_i
            exit_reason = "STOP"
            break
        if pd.notna(high_i) and high_i >= target:
            exit_price = target * (1 - SLIPPAGE)
            exit_date = date_i
            exit_reason = "TARGET"
            break
        idx += 1
    else:
        # loop exhausted the ticker's available data before hitting the
        # time limit or a stop/target -- can't know the real outcome
        # (e.g. ticker delisted mid-hold). Censored, not a real exit.
        data_end_censored = True

    if exit_price is None and not data_end_censored:
        close_i = ticker_df["close"].iat[last_idx_in_window]
        exit_date = ticker_df["date"].iat[last_idx_in_window]
        exit_price = float(close_i) * (1 - SLIPPAGE)
        exit_reason = "TIME"
    elif exit_price is None and data_end_censored:
        exit_reason = "DATA_END"

    if exit_reason == "DATA_END":
        return {
            "filled": True, "fill_date": fill_date, "fill_price": fill_price,
            "days_to_fill": days_to_fill, "exit_date": None, "exit_price": None,
            "exit_reason": "DATA_END", "shares": shares, "pnl": None,
        }

    pnl = shares * (exit_price - fill_price)
    return {
        "filled": True, "fill_date": fill_date, "fill_price": fill_price,
        "days_to_fill": days_to_fill, "exit_date": exit_date, "exit_price": exit_price,
        "exit_reason": exit_reason, "shares": shares, "pnl": pnl,
    }


# ─── Aggregation ────────────────────────────────────────────────────────────

def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cuts = ["ALL", "MOMENTUM", "VOLUME_BREAK"]
    for threshold in CHASE_THRESHOLDS:
        label = threshold_label(threshold)
        for cut in cuts:
            sub = trades[trades["threshold"] == label]
            if cut != "ALL":
                sub = sub[sub["signal_type"] == cut]

            total_events = len(sub)
            resolved = sub[sub["include_in_stats"]]
            filled = resolved[resolved["filled"]].copy()
            filled["pnl"] = filled["pnl"].astype(float)
            n_filled = len(filled)

            fill_rate = (n_filled / len(resolved) * 100) if len(resolved) else np.nan
            wins = filled[filled["pnl"] > 0]
            losses = filled[filled["pnl"] <= 0]
            gross_win = float(wins["pnl"].sum())
            gross_loss = float(losses["pnl"].abs().sum())
            if gross_loss > 0:
                pf = gross_win / gross_loss
            elif gross_win > 0:
                pf = np.inf
            else:
                pf = np.nan
            total_pnl = float(filled["pnl"].sum()) if n_filled else np.nan
            avg_pnl = float(filled["pnl"].mean()) if n_filled else np.nan
            win_rate = (len(wins) / n_filled * 100) if n_filled else np.nan
            avg_days_to_fill = float(filled["days_to_fill"].astype(float).mean()) if n_filled else np.nan

            filled_sorted = filled.sort_values("fill_date")
            cum = filled_sorted["pnl"].cumsum()
            running_peak = cum.cummax()
            dd = cum - running_peak
            max_dd = float(dd.min()) if not dd.empty else 0.0

            rows.append({
                "threshold": label,
                "cut": cut,
                "signal_events_total": total_events,
                "signal_events_resolved": len(resolved),
                "filled": n_filled,
                "fill_rate_pct": round(fill_rate, 1) if pd.notna(fill_rate) else np.nan,
                "win_rate_pct": round(win_rate, 1) if pd.notna(win_rate) else np.nan,
                "gross_win_$": round(gross_win, 2),
                "gross_loss_$": round(gross_loss, 2),
                "profit_factor": round(pf, 3) if np.isfinite(pf) else pf,
                "total_pnl_$": round(total_pnl, 2) if pd.notna(total_pnl) else np.nan,
                "avg_pnl_per_trade_$": round(avg_pnl, 2) if pd.notna(avg_pnl) else np.nan,
                "max_drawdown_$": round(max_dd, 2),
                "avg_days_to_fill": round(avg_days_to_fill, 2) if pd.notna(avg_days_to_fill) else np.nan,
                "trustworthy_sample": n_filled >= MIN_TRUSTWORTHY_FILLED,
            })
    return pd.DataFrame(rows)


# ─── Report ─────────────────────────────────────────────────────────────────

def build_report(events: list[dict], trades: pd.DataFrame, summary: pd.DataFrame,
                  requested_tickers: list[str], raw_prices: pd.DataFrame) -> str:
    n_events = len(events)
    n_too_recent = sum(e["too_recent"] for e in events)
    n_data_end = int((trades["excluded_reason"] == "data_end_censored").sum())
    n_momentum = sum(1 for e in events if e["signal_type"] == "MOMENTUM")
    n_volbreak = sum(1 for e in events if e["signal_type"] == "VOLUME_BREAK")

    tickers_with_data = raw_prices["ticker"].nunique()
    date_min = raw_prices["date"].min()
    date_max = raw_prices["date"].max()

    lines = []
    lines.append("=== BUCKET 2 CHASE-THRESHOLD BACKTEST ===")
    lines.append(f"Universe requested:   {len(requested_tickers)} tickers")
    lines.append(f"Universe with data:   {tickers_with_data} tickers")
    lines.append(f"Date range used:      {date_min.date()} to {date_max.date()}")
    lines.append(f"Raw signal events:    {n_events}  (MOMENTUM: {n_momentum}, VOLUME_BREAK: {n_volbreak})")
    lines.append(f"  excluded, too recent to have a resolved outcome: {n_too_recent}")
    lines.append(f"  excluded, ticker data ended mid-hold (delist/gap): {n_data_end}")
    lines.append("")
    lines.append("--- Threshold sweep (pooled, ALL signal types) ---")
    pooled = summary[summary["cut"] == "ALL"].copy()
    lines.append(pooled.to_string(index=False))
    lines.append("")
    lines.append("--- Threshold sweep, MOMENTUM only ---")
    lines.append(summary[summary["cut"] == "MOMENTUM"].to_string(index=False))
    lines.append("")
    lines.append("--- Threshold sweep, VOLUME_BREAK only ---")
    lines.append(summary[summary["cut"] == "VOLUME_BREAK"].to_string(index=False))
    lines.append("")

    low_n = pooled[~pooled["trustworthy_sample"]]
    if not low_n.empty:
        lines.append(f"CAVEAT: {len(low_n)} threshold(s) have fewer than "
                      f"{MIN_TRUSTWORTHY_FILLED} filled trades (pooled) -- their profit "
                      f"factor / win rate should not be trusted as precise:")
        for _, r in low_n.iterrows():
            lines.append(f"    {r['threshold']:>10}: {int(r['filled'])} filled trades")
    lines.append("")
    lines.append("See trades.csv for the full per-event, per-threshold audit trail "
                  "(includes MISSED and excluded rows).")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_universe()
    raw = download_universe(tickers)
    if raw.empty:
        raise RuntimeError("No price data available -- aborting.")

    feat = compute_features(raw)
    flagged = flag_signals(feat)

    ticker_frames = {
        t: g.sort_values("date").reset_index(drop=True)
        for t, g in flagged.groupby("ticker", sort=False)
    }

    print(">>> Extracting signal events (maximal consecutive-day runs, per signal type)...")
    events = extract_events(ticker_frames)
    n_too_recent = sum(e["too_recent"] for e in events)
    print(f"    {len(events)} raw signal events "
          f"({n_too_recent} too-recent / insufficient trailing data, excluded from stats)")

    print(">>> Running chase-threshold sweep...")
    trade_rows = []
    for e in events:
        ticker_df = ticker_frames[e["ticker"]]
        for threshold in CHASE_THRESHOLDS:
            label = threshold_label(threshold)
            base = {
                "event_id": e["event_id"], "ticker": e["ticker"], "signal_type": e["signal_type"],
                "signal_date": e["signal_date"].date(), "signal_price": round(e["signal_price"], 4),
                "atr_14": round(e["atr_14"], 4), "stop_price": round(e["stop_price"], 4),
                "target_price": round(e["target_price"], 4),
                "risk_per_share": round(e["risk_per_share"], 4),
                "threshold": label,
            }

            if e["too_recent"]:
                base.update({
                    "filled": False, "excluded_reason": "too_recent", "fill_date": None,
                    "fill_price": None, "days_to_fill": None, "exit_date": None,
                    "exit_price": None, "exit_reason": None, "shares": None, "pnl": None,
                })
                trade_rows.append(base)
                continue

            result = simulate_event(ticker_df, e["signal_idx"], e["signal_price"],
                                      e["stop_price"], e["target_price"], threshold)

            if not result["filled"]:
                base.update({
                    "filled": False, "excluded_reason": None, "fill_date": None, "fill_price": None,
                    "days_to_fill": None, "exit_date": None, "exit_price": None,
                    "exit_reason": None, "shares": None, "pnl": None,
                })
            else:
                excluded_reason = "data_end_censored" if result["exit_reason"] == "DATA_END" else None
                fd = result["fill_date"]
                xd = result["exit_date"]
                base.update({
                    "filled": True,
                    "excluded_reason": excluded_reason,
                    "fill_date": fd.date() if hasattr(fd, "date") else fd,
                    "fill_price": round(result["fill_price"], 4),
                    "days_to_fill": result["days_to_fill"],
                    "exit_date": (xd.date() if hasattr(xd, "date") else xd) if xd is not None else None,
                    "exit_price": round(result["exit_price"], 4) if result["exit_price"] is not None else None,
                    "exit_reason": result["exit_reason"],
                    "shares": result["shares"],
                    "pnl": round(result["pnl"], 2) if result["pnl"] is not None else None,
                })
            trade_rows.append(base)

    trades_df = pd.DataFrame(trade_rows)
    trades_df["include_in_stats"] = trades_df["excluded_reason"].isna()

    trades_path = RESULTS_DIR / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    summary_df = summarize(trades_df)
    summary_path = RESULTS_DIR / "threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report = build_report(events, trades_df, summary_df, tickers, raw)
    report_path = RESULTS_DIR / "summary.txt"
    report_path.write_text(report)

    print(f"\nOutputs saved to {RESULTS_DIR}/")
    print(f"  {trades_path.name}: {len(trades_df)} rows")
    print(f"  {summary_path.name}: {len(summary_df)} rows")
    print(f"  {report_path.name}")
    print()
    print(report)


if __name__ == "__main__":
    main()
