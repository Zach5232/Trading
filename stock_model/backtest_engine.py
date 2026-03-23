"""
backtest_engine.py

Standalone walk-forward weekly backtest for the Stock Swing Bet Model.

Version 1 rules:
  - Monday open entry, Friday close forced exit
  - Stop: entry - 1.5 x ATR(14)
  - Target: entry + 2.0 x risk_per_share
  - Max 5 trades/week, long-only
  - $35 risk per trade, $10,000 starting equity
  - 0.1% slippage per side

Outputs to Results/backtest/ — never writes to existing output files.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "backtest_cache"
RESULTS_DIR  = PROJECT_ROOT / "Results" / "backtest"
TICKER_CSV   = DATA_DIR / "raw_data" / "sp500_tickers.csv"
CACHE_FILE   = CACHE_DIR / "ohlcv_5yr.csv"


# ─── Backtest parameters ──────────────────────────────────────────────────────

START_DATE     = "2021-01-01"
END_DATE       = "2026-03-12"
RISK_PER_TRADE = 35.0
ACCOUNT_START  = 10_000.0
SLIPPAGE       = 0.001        # 0.1% per side
ATR_MULT       = 1.5
REWARD_RISK    = 2.0
MAX_TRADES     = 5
MIN_ELIGIBLE   = 3

# Filters
MIN_PRICE    = 5.0
MIN_AVG_VOL  = 500_000
RSI_LOW      = 30
RSI_HIGH     = 70
ATR_PCT_LOW  = 0.005
ATR_PCT_HIGH = 0.06

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "HD", "MA", "AVGO", "XOM", "CVX", "LLY", "ABBV",
    "MRK", "PEP", "KO", "BAC", "PFE", "TMO", "COST", "MCD", "WMT", "DIS",
    "CSCO", "ACN", "ABT", "DHR", "NFLX", "ADBE", "CRM", "VZ", "CMCSA", "NKE",
    "INTC", "TXN", "NEE", "PM", "RTX", "HON", "LIN", "QCOM", "IBM", "GS",
]


# ─── Indicator helpers ────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0.0)
    loss     = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr       = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tickers() -> list[str]:
    if TICKER_CSV.exists():
        tickers = (
            pd.read_csv(TICKER_CSV, header=None)
            .iloc[:, 0]
            .astype(str)
            .str.upper()
            .str.strip()
            .tolist()
        )
        tickers = [t for t in tickers if t and t != "0" and any(c.isalpha() for c in t)]
        return tickers
    print("  sp500_tickers.csv not found — using 50-ticker fallback list.")
    return list(FALLBACK_TICKERS)


def _parse_yf_download(raw: pd.DataFrame, batch: list[str]) -> list[pd.DataFrame]:
    """
    Parse a yfinance multi-ticker download (0.2.x API) into per-ticker DataFrames.

    yfinance >=0.2.37: MultiIndex with level 0 = price type, level 1 = ticker.
    Falls back to level 0 = ticker for older behaviour.
    """
    frames: list[pd.DataFrame] = []

    if not isinstance(raw.columns, pd.MultiIndex):
        # Single-ticker result
        df_t = raw.copy().reset_index()
        df_t.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Adj Close": "adj_close",
            "Volume": "volume",
        }, inplace=True)
        df_t.columns = [str(c).strip() for c in df_t.columns]
        if "close" in df_t.columns and not df_t["close"].isna().all():
            df_t["ticker"] = batch[0]
            needed = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if all(c in df_t.columns for c in needed):
                frames.append(df_t[needed])
        return frames

    lvl0 = raw.columns.get_level_values(0).unique().tolist()

    # Detect API version: new = price type in level 0
    if any(x in lvl0 for x in ["Close", "Open", "High", "Low"]):
        # New yfinance (>=0.2.37): xs by ticker on level 1
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
        # Old yfinance: level 0 = ticker
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


def download_and_cache(tickers: list[str]) -> pd.DataFrame:
    """Download 5-year OHLCV for all tickers, or load from cache if present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE.name} ...")
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers.")
        return df

    print(f"Downloading 5yr data for {len(tickers)} tickers...")
    batch_size  = 100
    total_batches = math.ceil(len(tickers) / batch_size)
    all_frames: list[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        b_num = i // batch_size + 1
        print(f"  Batch {b_num}/{total_batches}  ({len(batch)} tickers)...", flush=True)
        try:
            raw = yf.download(
                tickers=batch,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            print(f"  ! Batch {b_num} download error: {exc}")
            continue

        if raw is None or raw.empty:
            continue

        batch_frames = _parse_yf_download(raw, batch)
        all_frames.extend(batch_frames)

    if not all_frames:
        raise RuntimeError("No data downloaded. Check internet connection.")

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  Saved {len(df):,} rows to {CACHE_FILE.name}")
    return df


# ─── Indicator computation ────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling indicators for every ticker across the full date range.
    Returns long DataFrame with indicator columns added.
    """
    print("Computing indicators for all tickers...")
    frames: list[pd.DataFrame] = []

    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date").copy()
        g["ret_1w"]     = g["close"].pct_change(5,  fill_method=None)
        g["ret_3w"]     = g["close"].pct_change(15, fill_method=None)
        g["rsi_14"]     = compute_rsi(g["close"])
        g["atr_14"]     = compute_atr(g)
        g["avg_vol_20"] = g["volume"].rolling(20, min_periods=20).mean()
        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["date", "ticker"], inplace=True)
    print(f"  Done. {len(result):,} rows with indicators.")
    return result


# ─── Walk-forward backtest ────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward weekly simulation.

    Signals computed as of prior Friday (no lookahead).
    Entry: Monday open + slippage.
    Exit: stop / target / Friday close.

    Returns (trades_df, weekly_summary_df).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Build fast lookup: index by (date, ticker)
    df.set_index(["date", "ticker"], inplace=True)
    df.sort_index(inplace=True)

    date_set  = set(df.index.get_level_values("date").unique())
    mondays   = pd.date_range(start=START_DATE, end=END_DATE, freq="W-MON")
    mondays   = [m for m in mondays if m <= pd.Timestamp(END_DATE)]

    all_trades:   list[dict] = []
    weekly_rows:  list[dict] = []
    equity        = ACCOUNT_START
    trades_so_far = 0
    total_weeks   = len(mondays)

    for week_num, monday in enumerate(mondays, start=1):

        if week_num % 50 == 0:
            print(f"  Week {week_num}/{total_weeks}... (trades so far: {trades_so_far})")

        # ── Find last trading day before Monday (prior Friday) ────────────────
        prior_day = monday - timedelta(days=3)   # start at Friday
        found_prior = False
        for _ in range(7):
            if prior_day in date_set:
                found_prior = True
                break
            prior_day -= timedelta(days=1)

        if not found_prior:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        # ── Signal snapshot as of prior Friday ───────────────────────────────
        try:
            signal_rows = df.loc[prior_day]
        except KeyError:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T

        signal_rows = signal_rows.copy()
        signal_rows["atr_pct"] = signal_rows["atr_14"] / signal_rows["close"].replace(0, np.nan)

        # ── Apply filters ─────────────────────────────────────────────────────
        cond = (
            (signal_rows["close"]      >= MIN_PRICE)
            & (signal_rows["avg_vol_20"] >= MIN_AVG_VOL)
            & (signal_rows["rsi_14"]     >= RSI_LOW)
            & (signal_rows["rsi_14"]     <= RSI_HIGH)
            & (signal_rows["atr_pct"]    >= ATR_PCT_LOW)
            & (signal_rows["atr_pct"]    <= ATR_PCT_HIGH)
            & signal_rows["ret_1w"].notna()
            & signal_rows["ret_3w"].notna()
            & signal_rows["atr_14"].notna()
        )
        eligible = signal_rows[cond]

        if len(eligible) < MIN_ELIGIBLE:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        # ── Percentile rank scoring ───────────────────────────────────────────
        eligible = eligible.copy()
        eligible["rank_1w"]       = eligible["ret_1w"].rank(pct=True)
        eligible["rank_3w"]       = eligible["ret_3w"].rank(pct=True)
        vol_ratio                  = eligible["volume"] / eligible["avg_vol_20"].replace(0, np.nan)
        eligible["rank_vol_surge"] = vol_ratio.fillna(0.0).rank(pct=True)
        eligible["signal_score"]   = (
            0.40 * eligible["rank_1w"]
            + 0.40 * eligible["rank_3w"]
            + 0.20 * eligible["rank_vol_surge"]
        )

        top5 = eligible[eligible["signal_score"] > 0].nlargest(MAX_TRADES, "signal_score")
        if top5.empty:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        # ── Monday open: check data availability ──────────────────────────────
        if monday not in date_set:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        try:
            monday_data = df.loc[monday]
        except KeyError:
            _append_empty_week(weekly_rows, monday, equity)
            continue

        if isinstance(monday_data, pd.Series):
            monday_data = monday_data.to_frame().T

        # ── Collect Tue–Fri OHLCV for this week ──────────────────────────────
        friday = monday + timedelta(days=4)
        week_days = [monday + timedelta(days=d) for d in range(1, 5)]
        week_ohlcv: dict = {}
        for d in week_days:
            if d in date_set:
                try:
                    day_data = df.loc[d]
                    week_ohlcv[d] = day_data if not isinstance(day_data, pd.Series) \
                                             else day_data.to_frame().T
                except KeyError:
                    pass

        # ── Simulate each selected trade ──────────────────────────────────────
        week_pnl     = 0.0
        week_r       = 0.0
        week_winners = 0
        week_losers  = 0
        week_trades  = 0
        equity_start = equity

        for ticker in top5.index:
            if ticker not in monday_data.index:
                continue

            mon_open = monday_data.loc[ticker, "open"]
            if pd.isna(mon_open) or mon_open <= 0:
                continue

            entry_price  = mon_open * (1.0 + SLIPPAGE)
            atr_14       = float(top5.loc[ticker, "atr_14"])
            if pd.isna(atr_14) or atr_14 <= 0:
                continue

            stop_price   = entry_price - ATR_MULT * atr_14
            risk_ps      = entry_price - stop_price
            if risk_ps <= 0:
                continue

            shares       = max(1, math.floor(RISK_PER_TRADE / risk_ps))
            target_price = entry_price + REWARD_RISK * risk_ps

            # ── Exit: walk Tue → Fri ──────────────────────────────────────────
            exit_price  = None
            exit_date   = None
            exit_reason = None

            sorted_days = sorted(week_ohlcv.keys())
            for idx, day in enumerate(sorted_days):
                is_last = (idx == len(sorted_days) - 1)
                day_df  = week_ohlcv[day]

                if ticker not in day_df.index:
                    if is_last:
                        break   # ticker missing on last day — no exit recorded
                    continue

                row       = day_df.loc[ticker]
                day_low   = float(row.get("low",   np.nan))
                day_high  = float(row.get("high",  np.nan))
                day_close = float(row.get("close", np.nan))

                if not np.isnan(day_low) and day_low <= stop_price:
                    exit_price  = stop_price * (1.0 - SLIPPAGE)
                    exit_date   = day
                    exit_reason = "stop"
                    break
                elif not np.isnan(day_high) and day_high >= target_price:
                    exit_price  = target_price * (1.0 - SLIPPAGE)
                    exit_date   = day
                    exit_reason = "target"
                    break
                elif is_last:   # last available day this week → forced exit
                    if not np.isnan(day_close):
                        exit_price  = day_close * (1.0 - SLIPPAGE)
                        exit_date   = day
                        exit_reason = "time"
                    break

            if exit_price is None:
                continue

            pnl        = (exit_price - entry_price) * shares
            r_multiple = (exit_price - entry_price) / risk_ps
            sig        = top5.loc[ticker]

            all_trades.append({
                "week_start":    monday.date(),
                "ticker":        ticker,
                "entry_date":    monday.date(),
                "exit_date":     exit_date.date() if hasattr(exit_date, "date") else exit_date,
                "entry_price":   round(entry_price,  4),
                "exit_price":    round(exit_price,   4),
                "stop_price":    round(stop_price,   4),
                "target_price":  round(target_price, 4),
                "shares":        shares,
                "risk_per_share": round(risk_ps,     4),
                "pnl_dollars":   round(pnl,          2),
                "r_multiple":    round(r_multiple,   4),
                "exit_reason":   exit_reason,
                "ret_1w":        round(float(sig["ret_1w"]),       6) if pd.notna(sig["ret_1w"])  else None,
                "ret_3w":        round(float(sig["ret_3w"]),       6) if pd.notna(sig["ret_3w"])  else None,
                "rsi_14":        round(float(sig["rsi_14"]),       4) if pd.notna(sig["rsi_14"])  else None,
                "atr_14":        round(float(sig["atr_14"]),       4) if pd.notna(sig["atr_14"])  else None,
                "signal_score":  round(float(sig["signal_score"]), 6),
            })

            week_pnl     += pnl
            week_r       += r_multiple
            week_trades  += 1
            trades_so_far += 1
            if pnl > 0:
                week_winners += 1
            else:
                week_losers += 1

        equity += week_pnl
        weekly_return = ((equity / equity_start) - 1.0) * 100.0 if equity_start > 0 else 0.0

        weekly_rows.append({
            "week_start":        monday.date(),
            "trades":            week_trades,
            "winners":           week_winners,
            "losers":            week_losers,
            "win_rate":          round(week_winners / week_trades, 4) if week_trades > 0 else np.nan,
            "total_pnl":         round(week_pnl, 2),
            "total_r":           round(week_r, 4),
            "equity":            round(equity, 2),
            "weekly_return_pct": round(weekly_return, 4),
        })

    return pd.DataFrame(all_trades), pd.DataFrame(weekly_rows)


def _append_empty_week(rows: list[dict], monday: pd.Timestamp, equity: float) -> None:
    rows.append({
        "week_start": monday.date(), "trades": 0, "winners": 0, "losers": 0,
        "win_rate": np.nan, "total_pnl": 0.0, "total_r": 0.0,
        "equity": round(equity, 2), "weekly_return_pct": 0.0,
    })


# ─── Summary ─────────────────────────────────────────────────────────────────

def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    peak = equity_series.expanding().max()
    dd   = (equity_series - peak) / peak
    return float(dd.min() * 100.0)


def build_summary(trades_df: pd.DataFrame, weekly_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "=== BACKTEST SUMMARY ===\nNo trades were simulated."

    total  = len(trades_df)
    wins   = int((trades_df["pnl_dollars"] > 0).sum())
    losses = int((trades_df["pnl_dollars"] <= 0).sum())
    wr     = wins / total * 100.0
    avg_r  = float(trades_df["r_multiple"].mean())

    gross_w = trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum()
    gross_l = trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].abs().sum()
    pf      = (gross_w / gross_l) if gross_l > 0 else float("inf")

    max_dd      = _max_drawdown(weekly_df["equity"])
    final_eq    = float(weekly_df["equity"].iloc[-1]) if not weekly_df.empty else ACCOUNT_START
    total_ret   = (final_eq / ACCOUNT_START - 1.0) * 100.0
    total_weeks = len(weekly_df)
    weeks_w_tr  = int((weekly_df["trades"] > 0).sum())

    dr_start = str(trades_df["entry_date"].min())
    dr_end   = str(trades_df["exit_date"].max())

    ec = trades_df["exit_reason"].value_counts()
    t_n = int(ec.get("target", 0))
    s_n = int(ec.get("stop",   0))
    f_n = int(ec.get("time",   0))

    lines = [
        "=== BACKTEST SUMMARY (5 Year, Version 1 Rules) ===",
        f"Date range:         {dr_start} to {dr_end}",
        f"Total weeks:        {total_weeks}",
        f"Weeks with trades:  {weeks_w_tr}",
        f"Total trades:       {total}",
        f"Win rate:           {wr:.1f}%",
        f"Avg R per trade:    {avg_r:.2f}",
        f"Profit factor:      {pf:.2f}  (gross wins / gross losses)",
        f"Max drawdown:       {max_dd:.1f}%",
        f"Final equity:       ${final_eq:,.2f}",
        f"Total return:       {total_ret:.1f}%",
        f"Slippage assumed:   0.1% per side",
        f"Risk per trade:     $35 flat",
        "",
        "Exit breakdown:",
        f"  Target hit:       {t_n}  ({t_n/total*100:.1f}%)",
        f"  Stop hit:         {s_n}  ({s_n/total*100:.1f}%)",
        f"  Time (Friday):    {f_n}  ({f_n/total*100:.1f}%)",
    ]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Tickers & raw data
    tickers = load_tickers()
    df_raw  = download_and_cache(tickers)

    # Step 2: Indicators
    df_ind = compute_all_indicators(df_raw)

    # Step 3: Walk-forward backtest
    print("Running walk-forward backtest...")
    trades_df, weekly_df = run_backtest(df_ind)

    # Step 4: Save outputs
    trades_path  = RESULTS_DIR / "backtest_trades.csv"
    weekly_path  = RESULTS_DIR / "backtest_weekly_summary.csv"
    summary_path = RESULTS_DIR / "backtest_summary.txt"

    trades_df.to_csv(trades_path,  index=False)
    weekly_df.to_csv(weekly_path,  index=False)
    summary = build_summary(trades_df, weekly_df)
    summary_path.write_text(summary)

    print(f"\nOutputs saved to {RESULTS_DIR}/")
    print(f"  {trades_path.name}:  {len(trades_df)} trade rows")
    print(f"  {weekly_path.name}: {len(weekly_df)} week rows")
    print(f"  {summary_path.name}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
