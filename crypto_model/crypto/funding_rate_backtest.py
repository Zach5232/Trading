"""
crypto/funding_rate_backtest.py
================================
Tests the funding rate as a hard filter on top of the Var1+Var2+Var4 baseline.

# ── TEST RESULT (March 2026) ─────────────────────────────────────────────────
# Verdict:       DO NOT PROMOTE — insufficient data
# Reason:        OKX funding rate history only covers Dec 2025–Mar 2026.
#                In the 2021-present backtest window this yields only 3 BTC and
#                2 ETH trades with actual funding data out of 56–57 signals.
#                All remaining trades are classified NO_DATA, so Conservative
#                and Aggressive filters produce identical results to Baseline.
#                The filter cannot be meaningfully evaluated.
# Revisit:       After Phase 1 complete (~May 2026), or if a historical dataset
#                from Coinalyze or CryptoQuant covering 2021+ becomes available.
# Current status: Funding rate remains display-only in main.py.
#                 Monitor but do not filter.
# ─────────────────────────────────────────────────────────────────────────────

Method:
  1. Fetch historical funding rates from OKX for BTC-USDT-SWAP and ETH-USDT-SWAP.
     OKX settles 3× daily (00:00, 08:00, 16:00 UTC). We use the 16:00 UTC rate
     on each Friday as the signal-day rate (most recent before Friday close).
  2. Join funding rates to the Var1+Var2+Var4 trade log on Friday date.
  3. Run three versions on the 2021-present window (OKX history starts ~2021):
       Baseline     — no funding filter
       Conservative — skip ELEVATED + EXTREME (keep NEGATIVE/NEUTRAL/NO_DATA)
       Aggressive   — skip EXTREME only
  4. Print comparison table and PROMOTE / DO NOT PROMOTE verdict.

Promotion bar:
  - net avg R improvement > 0.02R vs baseline
  - sample stays above 30 trades
  - improvement holds on BOTH BTC and ETH independently
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_crypto_data, get_ticker_df
from backtest_engine import run_backtest, STARTING_CAPITAL, RISK_PCT_PER_TRADE, _calc_metrics

# ── Constants ────────────────────────────────────────────────────────────────
OKX_HIST_URL   = "https://www.okx.com/api/v5/public/funding-rate-history"
OKX_SYMBOL_MAP = {"BTC-USD": "BTC-USDT-SWAP", "ETH-USD": "ETH-USDT-SWAP"}
OKX_TIMEOUT    = 15
CACHE_DIR      = Path(__file__).parent.parent / "Data" / "backtest_cache"

OUTPUT_DIR     = Path(__file__).parent.parent / "Results" / "crypto_backtest"

# Date window: OKX history starts mid-2021 at earliest
WINDOW_START = "2021-01-01"

# Funding rate thresholds (per-8h rate, %)
THRESHOLDS = {
    "NEGATIVE":  (-999,  -0.01),
    "NEUTRAL":   (-0.01,  0.05),
    "ELEVATED":  ( 0.05,  0.15),
    "EXTREME":   ( 0.15,  999),
}


def _classify_funding(rate_pct: float) -> str:
    for label, (lo, hi) in THRESHOLDS.items():
        if lo <= rate_pct < hi:
            return label
    return "EXTREME"


# ── OKX fetch ────────────────────────────────────────────────────────────────

def _fetch_okx_funding(symbol: str) -> pd.DataFrame:
    """
    Paginate OKX funding rate history backwards. Returns DataFrame with
    columns: timestamp (UTC datetime), rate_pct (float).
    """
    cache_file = CACHE_DIR / f"funding_{symbol.replace('-','_').lower()}.csv"
    if cache_file.exists():
        print(f"  Loading funding cache: {cache_file.name}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        return df

    print(f"  Fetching OKX funding history for {symbol}...")
    all_records = []
    after_ms: int | None = None
    page = 0
    prev_oldest_ts: int | None = None
    MAX_PAGES = 500

    while page < MAX_PAGES:
        page += 1
        params: dict = {"instId": symbol, "limit": 100}
        if after_ms is not None:
            params["after"] = str(after_ms)

        resp = requests.get(OKX_HIST_URL, params=params, timeout=OKX_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"OKX error {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error: {payload.get('msg', payload)}")

        batch = payload.get("data", [])
        if not batch:
            break

        all_records.extend(batch)
        oldest_ts = int(batch[-1]["fundingTime"])
        earliest_dt = pd.to_datetime(oldest_ts, unit="ms", utc=True).date()
        print(f"    Page {page}: {len(batch)} records  (earliest: {earliest_dt})")

        if oldest_ts == prev_oldest_ts:
            print("    Cursor not advancing — history exhausted.")
            break
        prev_oldest_ts = oldest_ts

        if len(batch) < 100:
            break

        after_ms = oldest_ts - 1
        time.sleep(0.2)

    if not all_records:
        raise RuntimeError(f"No funding records for {symbol}")

    df = pd.DataFrame(all_records)[["fundingTime", "fundingRate"]]
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["rate_pct"]  = df["fundingRate"].astype(float) * 100
    df = df[["timestamp", "rate_pct"]].sort_values("timestamp").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"    Saved {len(df)} records → {cache_file}")
    return df


def _get_friday_rate(funding_df: pd.DataFrame, fri_date: pd.Timestamp) -> tuple[float | None, str]:
    """
    Return the most recent 16:00 UTC funding rate on or before Friday close.
    If no data, return (None, 'NO_DATA').
    """
    if funding_df is None or funding_df.empty:
        return None, "NO_DATA"

    # The 16:00 UTC settlement on fri_date
    target_ts = pd.Timestamp(fri_date.date()) + pd.Timedelta(hours=16)

    eligible = funding_df[funding_df["timestamp"] <= target_ts]
    if eligible.empty:
        return None, "NO_DATA"

    rate_pct = float(eligible.iloc[-1]["rate_pct"])
    return rate_pct, _classify_funding(rate_pct)


# ── Filter runner ────────────────────────────────────────────────────────────

def _apply_funding_filter(
    trade_log: pd.DataFrame,
    funding_df: pd.DataFrame,
    filter_name: str,
    skip_classes: set[str],
) -> pd.DataFrame:
    """
    Attach funding rate to each LONG trade. Trades with skip_classes become NO_TRADE.
    Returns a modified copy of trade_log (only within WINDOW_START).
    """
    tl = trade_log[trade_log["date"] >= WINDOW_START].copy()

    rates   = []
    classes = []
    for _, row in tl.iterrows():
        if row["direction"] != "LONG":
            rates.append(None)
            classes.append("NO_DATA")
            continue
        rate, cls = _get_friday_rate(funding_df, row["date"])
        rates.append(rate)
        classes.append(cls)

    tl["funding_rate_pct"] = rates
    tl["funding_class"]    = classes

    if skip_classes:
        mask = (tl["direction"] == "LONG") & (tl["funding_class"].isin(skip_classes))
        tl.loc[mask, "direction"]  = "NO_TRADE"
        tl.loc[mask, "R_multiple"] = None
        tl.loc[mask, "profit_loss"] = 0.0

    return tl


# ── Comparison table ─────────────────────────────────────────────────────────

def _print_comparison(ticker: str, versions: dict[str, dict], baseline_key: str) -> str:
    """Print comparison table, return PROMOTE/DO NOT PROMOTE verdict."""
    base = versions[baseline_key]
    lines = []
    lines.append(f"\n{'='*72}")
    lines.append(f"  FUNDING RATE FILTER — {ticker}  (window: {WINDOW_START} – present)")
    lines.append(f"{'='*72}")
    header = (f"  {'Version':<14} {'Trades':>7} {'T/yr':>5} {'WR%':>6} "
              f"{'AvgR':>7} {'PF':>6} {'MaxDD':>7} {'delta_R':>8}")
    lines.append(header)
    lines.append("  " + "-" * 68)

    for name, m in versions.items():
        if not m:
            lines.append(f"  {name:<14}  — no trades —")
            continue
        delta = round(m["avg_R"] - base["avg_R"], 3) if name != baseline_key else 0.0
        delta_str = f"{delta:+.3f}" if name != baseline_key else "  base"
        flag = "  *** LOW SAMPLE ***" if m["n_trades"] < 30 else ""
        lines.append(
            f"  {name:<14} {m['n_trades']:>7} {m['trades_per_year']:>5} "
            f"{m['win_rate']:>5}% {m['avg_R']:>7.3f} {m['profit_factor']:>6.3f} "
            f"{m['max_drawdown']:>6}% {delta_str:>8}{flag}"
        )

    # Verdict
    lines.append("")
    verdicts = {}
    for name, m in versions.items():
        if name == baseline_key or not m:
            continue
        delta = m["avg_R"] - base["avg_R"]
        ok_r  = delta > 0.02
        ok_n  = m["n_trades"] >= 30
        verdict = "PROMOTE" if (ok_r and ok_n) else "DO NOT PROMOTE"
        reason  = []
        if not ok_r:
            reason.append(f"delta={delta:+.3f}R (need >+0.02R)")
        if not ok_n:
            reason.append(f"only {m['n_trades']} trades (need ≥30)")
        lines.append(f"  {name:<14}: {verdict}"
                     + (f"  [{'; '.join(reason)}]" if reason else ""))
        verdicts[name] = verdict

    for line in lines:
        print(line)
    return verdicts


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("Funding Rate Filter Backtest  —  Var1+Var2+Var4 baseline")
    print(f"Window: {WINDOW_START} – present")
    print("=" * 72)

    print("\nLoading OHLCV data...")
    data = load_crypto_data(["BTC-USD", "ETH-USD"])

    # Collect per-ticker verdicts for combined verdict at the end
    all_verdicts: dict[str, dict[str, str]] = {}

    for ticker in ["BTC-USD", "ETH-USD"]:
        print(f"\n{'─'*60}")
        print(f"  {ticker}")
        print(f"{'─'*60}")

        # ── Full backtest ─────────────────────────────────────────────────
        ticker_df = get_ticker_df(data, ticker)
        full_log, _, _ = run_backtest(ticker_df, ticker)
        full_log["date"] = pd.to_datetime(full_log["date"])

        # ── Funding data ──────────────────────────────────────────────────
        symbol = OKX_SYMBOL_MAP[ticker]
        funding_df = _fetch_okx_funding(symbol)

        # ── Three versions ────────────────────────────────────────────────
        versions: dict[str, dict] = {}

        # Baseline (window-restricted, no filter)
        base_log = full_log[full_log["date"] >= WINDOW_START].copy()
        base_log["funding_class"] = base_log.apply(
            lambda r: _get_friday_rate(funding_df, r["date"])[1]
            if r["direction"] == "LONG" else "NO_DATA",
            axis=1,
        )
        versions["Baseline"] = _calc_metrics(base_log, STARTING_CAPITAL)

        # Conservative: skip ELEVATED + EXTREME
        cons_log = _apply_funding_filter(
            full_log, funding_df, "Conservative", {"ELEVATED", "EXTREME"}
        )
        versions["Conservative"] = _calc_metrics(cons_log, STARTING_CAPITAL)

        # Aggressive: skip EXTREME only
        agg_log = _apply_funding_filter(
            full_log, funding_df, "Aggressive", {"EXTREME"}
        )
        versions["Aggressive"] = _calc_metrics(agg_log, STARTING_CAPITAL)

        # ── Funding class distribution ────────────────────────────────────
        print(f"\n  Funding class breakdown (LONG trades, {WINDOW_START}+):")
        if not base_log.empty:
            long_rows = base_log[base_log["direction"] == "LONG"]
            if "funding_class" in base_log.columns:
                for cls in ["NEGATIVE", "NEUTRAL", "ELEVATED", "EXTREME", "NO_DATA"]:
                    subset = long_rows[long_rows["funding_class"] == cls]
                    if not subset.empty:
                        wr  = (subset["R_multiple"] > 0).mean() * 100
                        avr = subset["R_multiple"].mean()
                        print(f"    {cls:<10} N={len(subset):>3}  WR={wr:>5.1f}%  AvgR={avr:>7.3f}")

        verdicts = _print_comparison(ticker, versions, "Baseline")
        all_verdicts[ticker] = verdicts

    # ── Combined verdict ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  COMBINED VERDICT  (must hold on BOTH tickers)")
    print(f"{'='*72}")
    for version in ["Conservative", "Aggressive"]:
        btc_v = all_verdicts.get("BTC-USD", {}).get(version, "—")
        eth_v = all_verdicts.get("ETH-USD", {}).get(version, "—")
        combined = "PROMOTE" if btc_v == eth_v == "PROMOTE" else "DO NOT PROMOTE"
        print(f"  {version:<14}: BTC={btc_v}  ETH={eth_v}  →  {combined}")

    print("\nDone.")


if __name__ == "__main__":
    main()
