"""
friday_close.py

Standalone helper: reads the current week's candidates CSV, downloads this
week's OHLC via yfinance, and prints a hypothetical R-multiple summary table
for each candidate. Read-only reporting only — does not touch main.py logic,
Firestore, or any dashboard state.

Usage:
    python3 stock_model/friday_close.py                # most recent trade week
    python3 stock_model/friday_close.py 2026-08-21      # a specific candidates_<date>.csv
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_DIR = PROJECT_ROOT / "Results" / "candidates"

# Same convention as main.py: excludes _backup_/_backfill_/candidate_pool_ variants.
_CANDIDATES_FILENAME_RE = re.compile(r"^candidates_(\d{4}-\d{2}-\d{2})\.csv$")


def _has_trade_rows(path: Path) -> bool:
    df = pd.read_csv(path, dtype=str)
    if df.empty or "Ticker" not in df.columns:
        return False
    return df["Ticker"].fillna("").str.strip().ne("").any()


def find_candidates_csv(target_date: str | None) -> Path:
    if not CANDIDATE_DIR.exists():
        raise SystemExit(f"Candidates directory not found: {CANDIDATE_DIR}")

    matches = []
    for f in CANDIDATE_DIR.iterdir():
        m = _CANDIDATES_FILENAME_RE.match(f.name)
        if m:
            matches.append((m.group(1), f))
    if not matches:
        raise SystemExit(f"No candidates_YYYY-MM-DD.csv files found in {CANDIDATE_DIR}")
    matches.sort(key=lambda x: x[0])

    if target_date:
        for date_str, f in matches:
            if date_str == target_date:
                if not _has_trade_rows(f):
                    raise SystemExit(f"{f.name} is a sit-out week — nothing to report on.")
                return f
        raise SystemExit(f"No candidates_{target_date}.csv found in {CANDIDATE_DIR}")

    for date_str, f in reversed(matches):
        if _has_trade_rows(f):
            return f
    raise SystemExit("No non-sit-out candidates CSV found in " + str(CANDIDATE_DIR))


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Ticker"].notna() & (df["Ticker"].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


def week_bounds(date_str: str) -> tuple[datetime, datetime]:
    """Monday-through-Friday of the calendar week containing date_str."""
    anchor = datetime.strptime(date_str, "%Y-%m-%d")
    monday = anchor - timedelta(days=anchor.weekday())
    friday = monday + timedelta(days=4)
    end = friday + timedelta(days=1)  # yfinance 'end' is exclusive
    return monday, end


def fetch_week_ohlc(tickers: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    result: dict[str, pd.DataFrame] = {}
    if data.empty:
        return result

    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                df_t = data[t].dropna(how="all")
                if not df_t.empty:
                    result[t] = df_t
    else:
        if len(tickers) == 1:
            result[tickers[0]] = data.dropna(how="all")

    return result


def compute_outcome(entry: float, stop: float, target: float, week_df: pd.DataFrame) -> tuple[float, float, str]:
    """
    Returns (exit_price, r_multiple, outcome_label).

    If the target was crossed intraweek (daily high >= target on any day),
    the exit price is the target itself — more conservative than crediting
    the actual high, which can reflect a fleeting intraday spike.
    Otherwise the exit is the week's final close.
    """
    risk_per_share = entry - stop
    target_hit = bool((week_df["High"] >= target).any())

    if target_hit:
        exit_price = target
        outcome = "TARGET HIT"
    else:
        exit_price = float(week_df["Close"].iloc[-1])
        outcome = "Win" if exit_price > entry else "Loss"

    r_multiple = (exit_price - entry) / risk_per_share if risk_per_share > 0 else 0.0
    return exit_price, r_multiple, outcome


def fmt_r(r: float) -> str:
    return f"{'+' if r >= 0 else ''}{r:.2f}R"


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = find_candidates_csv(target_date)
    candidates = load_candidates(csv_path)

    date_str = _CANDIDATES_FILENAME_RE.match(csv_path.name).group(1)
    start, end = week_bounds(date_str)

    tickers = candidates["Ticker"].tolist()
    week_data = fetch_week_ohlc(tickers, start, end)

    rows = []
    for _, r in candidates.iterrows():
        ticker = r["Ticker"]
        week_df = week_data.get(ticker)
        if week_df is None or week_df.empty:
            print(f"  ! No price data for {ticker} — skipping")
            continue
        entry = float(r["Entry Price"])
        stop = float(r["Stop Price"])
        target = float(r["Target Price"])
        exit_price, r_multiple, outcome = compute_outcome(entry, stop, target, week_df)
        rows.append({
            "ticker": ticker, "entry": entry, "stop": stop, "target": target,
            "exit": exit_price, "r": r_multiple, "outcome": outcome,
        })

    if not rows:
        print("No price data available for any candidates this week.")
        return

    print(f"=== FRIDAY CLOSE — Week of {date_str} ===")
    print(f"{'Ticker':<8}{'Entry':<8}{'Stop':<8}{'Target':<8}{'Exit':<8}{'R':<8}{'Outcome'}")
    for row in rows:
        print(
            f"{row['ticker']:<8}{row['entry']:<8.2f}{row['stop']:<8.2f}"
            f"{row['target']:<8.2f}{row['exit']:<8.2f}{fmt_r(row['r']):<8}{row['outcome']}"
        )

    wins = sum(1 for row in rows if row["r"] > 0)
    losses = sum(1 for row in rows if row["r"] <= 0)
    avg_r = sum(row["r"] for row in rows) / len(rows)

    print()
    print("=== SUMMARY ===")
    print(f"Wins: {wins} | Losses: {losses} | Avg R: {fmt_r(avg_r)}")
    print("Log these exits in the dashboard candidate pool tab.")


if __name__ == "__main__":
    main()
