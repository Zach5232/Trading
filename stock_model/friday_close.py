"""
friday_close.py

Friday-close helper for the Stock Swing Bet Model. Reads this week's
candidates CSV, downloads the week's OHLC via yfinance, computes a
hypothetical R-multiple per ticker, prints a summary table, and writes
the outcomes (exitR / outcome / exitPrice) back to the matching pending
entries in the user's Firestore candidatePool doc.

Only ever reads/writes users/{uid}/data/candidatePool — no other
Firestore collection, and never touches main.py or the dashboard's
candidatePool tab code.

Usage:
    python3 stock_model/friday_close.py                # most recent week
    python3 stock_model/friday_close.py 2026-08-21      # a specific candidates_<date>.csv

First-time setup (writes stock_model/config/user_config.json):
    python3 stock_model/setup_firebase.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import firebase_admin
    from firebase_admin import firestore
except ImportError:
    print("Firebase Admin SDK not installed.")
    print("Run: pip3 install firebase-admin --break-system-packages")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_DIR = PROJECT_ROOT / "Results" / "candidates"
CONFIG_PATH = PROJECT_ROOT / "config" / "user_config.json"

# Same convention as main.py: excludes _backup_/_backfill_/candidate_pool_ variants.
_CANDIDATES_FILENAME_RE = re.compile(r"^candidates_(\d{4}-\d{2}-\d{2})\.csv$")


def _has_trade_rows(df: pd.DataFrame) -> bool:
    if df.empty or "Ticker" not in df.columns:
        return False
    return df["Ticker"].fillna("").str.strip().ne("").any()


def find_candidates_csv(target_date: str | None) -> Path:
    """Locate the candidates CSV to process: the most recent one, or a specific date."""
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
                return f
        raise SystemExit(f"No candidates_{target_date}.csv found in {CANDIDATE_DIR}")

    return matches[-1][1]


def week_bounds(date_str: str) -> tuple[datetime, datetime, datetime]:
    """Monday, Friday, and the yfinance-exclusive end date for the calendar week of date_str."""
    anchor = datetime.strptime(date_str, "%Y-%m-%d")
    monday = anchor - timedelta(days=anchor.weekday())
    friday = monday + timedelta(days=4)
    end = friday + timedelta(days=1)
    return monday, friday, end


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


def compute_outcome(entry: float, stop: float, target: float, week_df: pd.DataFrame, friday: datetime):
    """
    Returns (exit_price, r_multiple, outcome_label, target_hit), using the
    closest trading day on or before Friday as the exit day — so a Friday
    market holiday correctly falls back to Thursday's close instead of
    silently reusing whatever the last cached row happens to be.

    Returns None if no trading day on or before `friday` is in week_df yet
    (i.e. the market hasn't closed for this week when the script was run).
    """
    on_or_before_friday = week_df.index[week_df.index <= pd.Timestamp(friday)]
    if len(on_or_before_friday) == 0:
        return None
    exit_date = on_or_before_friday[-1]

    risk_per_share = entry - stop
    target_hit = bool((week_df.loc[:exit_date, "High"] >= target).any())

    if target_hit:
        # Target is defined as exactly entry + 2*risk (see main.py), so
        # crediting the target price itself — rather than the actual high,
        # which can reflect a fleeting intraday spike — always yields +2.00R.
        exit_price = target
        r_multiple = 2.0
        outcome = "Win ★TARGET HIT"
    else:
        exit_price = float(week_df.loc[exit_date, "Close"])
        r_multiple = (exit_price - entry) / risk_per_share if risk_per_share > 0 else 0.0
        outcome = "Win" if exit_price > entry else "Loss"

    return exit_price, r_multiple, outcome, target_hit


def fmt_r(r: float) -> str:
    return f"{'+' if r >= 0 else ''}{r:.2f}R"


def load_uid() -> str | None:
    """Returns the configured Firebase UID, or None (after printing setup instructions)."""
    if not CONFIG_PATH.exists():
        print("=== Firebase not configured ===")
        print("Run this once to connect friday_close.py to your dashboard:")
        print("  python3 stock_model/setup_firebase.py")
        return None
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"=== Could not read {CONFIG_PATH}: {e} ===")
        print("Run: python3 stock_model/setup_firebase.py")
        return None
    uid = cfg.get("uid")
    if not uid:
        print(f"=== {CONFIG_PATH} is missing a 'uid' field ===")
        print("Run: python3 stock_model/setup_firebase.py")
        return None
    return uid


def get_firestore_client():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def update_candidate_pool(uid: str, date_str: str, rows: list[dict]) -> int:
    """
    Writes exitR / outcome / exitPrice into the matching pending
    (exitR is None) candidatePool entries for this week. Reads and writes
    the candidatePool doc exactly once — never touches any other
    Firestore collection or document.
    """
    db = get_firestore_client()
    doc_ref = db.collection("users").document(uid).collection("data").document("candidatePool")
    doc = doc_ref.get()
    items = doc.to_dict().get("items", []) if doc.exists else []

    written = 0
    for row in rows:
        matched = False
        outcome = "win" if row["r"] > 0 else "loss"
        for item in items:
            if (
                item.get("ticker") == row["ticker"]
                and item.get("date") == date_str
                and item.get("exitR") is None
            ):
                item["exitR"] = round(row["r"], 2)
                item["outcome"] = outcome
                item["exitPrice"] = round(row["exit"], 2)
                matched = True
                written += 1
        if matched:
            print(f"✓ Firestore updated: {row['ticker']} → {fmt_r(row['r'])} ({outcome})")
        else:
            print(f"  ! No matching pending candidatePool entry for {row['ticker']} (week {date_str}) — skipped")

    if written:
        doc_ref.set({"items": items})
    return written


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = find_candidates_csv(target_date)
    date_str = _CANDIDATES_FILENAME_RE.match(csv_path.name).group(1)

    raw = pd.read_csv(csv_path)
    if not _has_trade_rows(raw):
        print(f"=== Sit-out week — no outcomes to log (candidates_{date_str}.csv) ===")
        return

    monday, friday, end = week_bounds(date_str)
    if datetime.now() < friday:
        print(f"Market hasn't closed yet for week of {date_str}.")
        print("Run this on Friday after 4pm ET.")
        return

    candidates = raw[raw["Ticker"].notna() & (raw["Ticker"].astype(str).str.strip() != "")].reset_index(drop=True)
    uid = load_uid()

    tickers = candidates["Ticker"].tolist()
    week_data = fetch_week_ohlc(tickers, monday, end)

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
        outcome_data = compute_outcome(entry, stop, target, week_df, friday)
        if outcome_data is None:
            print(f"Market hasn't closed yet for week of {date_str}.")
            print("Run this on Friday after 4pm ET.")
            return
        exit_price, r_multiple, outcome, _target_hit = outcome_data
        rows.append({
            "ticker": ticker, "entry": entry, "stop": stop, "target": target,
            "exit": exit_price, "r": r_multiple, "outcome": outcome,
        })

    if not rows:
        print("No price data available for any candidates this week.")
        return

    written = update_candidate_pool(uid, date_str, rows) if uid else 0

    header = f"{'Ticker':<8}{'Entry':<9}{'Stop':<9}{'Target':<9}{'Exit':<9}{'R':<9}{'Outcome'}"
    print(f"\n=== FRIDAY CLOSE — Week of {date_str} ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['ticker']:<8}{row['entry']:<9.2f}{row['stop']:<9.2f}"
            f"{row['target']:<9.2f}{row['exit']:<9.2f}{fmt_r(row['r']):<9}{row['outcome']}"
        )

    wins = sum(1 for row in rows if row["r"] > 0)
    losses = sum(1 for row in rows if row["r"] <= 0)
    avg_r = sum(row["r"] for row in rows) / len(rows)

    print()
    print("=== SUMMARY ===")
    print(f"Wins: {wins} | Losses: {losses} | Avg R: {fmt_r(avg_r)}")
    if uid:
        print(f"Firestore updated: {written} record{'s' if written != 1 else ''} written")
    else:
        print("Firestore not configured — run python3 stock_model/setup_firebase.py to enable auto-write")


if __name__ == "__main__":
    main()
