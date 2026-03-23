"""
crypto/main.py
==============
Weekly pipeline entry point — run every Friday after market close.

Phase 1 — Paper trading only.
  No real capital is deployed until ALL gates are cleared:
    [ ] 8 weeks of paper trades logged
    [ ] Win rate > 50%
    [ ] Positive expectancy after 32% short-term cap-gains tax

Flow:
  1. Load BTC-USD and ETH-USD data from Coinbase Advanced public API
  2. For each instrument independently — ALL THREE filters must pass for LONG:
       a. Filter 1: Friday close > MA20
       b. Filter 2: ATR14 today > ATR14 prior Friday (volatility expanding)
       c. Filter 3: Friday close > prior Friday close (momentum confirming)
       d. Failures: NO_TRADE — below MA20 / ATR contracting / momentum decelerating
  3. Validate candidate CSV schema
  4. Save both rows to Results/crypto_candidates/candidate_YYYY-MM-DD.csv
  5. Print clean terminal summary for each instrument
"""

import sys
import warnings
from pathlib import Path
from datetime import date, datetime, timezone

import requests
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_coinbase_live
from dashboard_generator import generate_dashboard

# ── Config ──────────────────────────────────────────────────────────────────
PHASE              = 1
ACCOUNT_EQUITY     = 500.0   # USD — update after Phase 1 validation
RISK_PCT_PER_TRADE = 0.05    # 5% per instrument (10% total if both trade)
SLIPPAGE_PCT       = 0.001
ATR_MULT_STOP      = 1.25
R_TARGET           = 2.0
TAX_RATE           = 0.32

INSTRUMENTS      = ["BTC-USD", "ETH-USD"]
CANDIDATES_DIR   = Path(__file__).parent.parent / "Results" / "crypto_candidates"
KILL_SWITCH_PATH = Path(__file__).parent.parent / "Results" / "crypto_backtest" / "kill_switch_log.csv"
KILL_SWITCH_THRESHOLD = -2.0   # sum of last 5 R-multiples below this → activate

# OKX public funding rate endpoint (no auth, accessible from US)
_OKX_FUNDING_URL   = "https://www.okx.com/api/v5/public/funding-rate"
_OKX_SYMBOL_MAP    = {"BTC-USD": "BTC-USDT-SWAP", "ETH-USD": "ETH-USDT-SWAP"}
_FUNDING_TIMEOUT   = 5   # seconds

# Funding rate classification thresholds (% per 8h period)
_FUNDING_LEVELS = [
    (-999.0, -0.01, "NEGATIVE"),   # shorts crowded — squeeze fuel — strong long
    (-0.01,   0.05, "NEUTRAL"),    # balanced — long OK
    ( 0.05,   0.15, "ELEVATED"),   # longs slightly crowded — caution
    ( 0.15,  999.0, "EXTREME"),    # longs heavily crowded — avoid
]

REQUIRED_COLUMNS = [
    "date",
    "instrument",
    "entry_price",
    "stop_price",
    "target_price",
    "direction",
    "above_ma20",
    "R_target",
    "risk_dollars",
    "units",
    "atr_expanding",
    "momentum_confirm",
    "all_filters_pass",
]


# ── Kill switch ──────────────────────────────────────────────────────────────

def check_kill_switch() -> bool:
    """
    Read kill_switch_log.csv (manually maintained by trader) and check whether
    the sum of the last 5 closed R-multiples falls below KILL_SWITCH_THRESHOLD.

    File format: date, instrument, R_multiple
    Template is created automatically if missing.

    Returns True if kill switch is ACTIVE (suppress all signals this weekend).
    """
    # Create template if missing so the trader knows the format
    if not KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        KILL_SWITCH_PATH.write_text(
            "date,instrument,R_multiple\n"
            "# Fill in each closed paper trade below — one row per instrument per week.\n"
            "# Example: 2026-03-01,BTC-USD,2.0\n"
        )
        print("  Kill switch log created (template) →", KILL_SWITCH_PATH)
        return False

    try:
        ks = pd.read_csv(KILL_SWITCH_PATH, comment="#")
        ks.columns = ks.columns.str.strip()
        if "R_multiple" not in ks.columns or len(ks) < 5:
            print(f"  KILL SWITCH  : inactive  (< 5 trades logged)")
            return False
        last5 = float(ks["R_multiple"].tail(5).sum())
        if last5 < KILL_SWITCH_THRESHOLD:
            print()
            print("!" * 65)
            print(f"  KILL SWITCH  : ACTIVE")
            print(f"  Last 5R sum  : {last5:.3f}  (threshold {KILL_SWITCH_THRESHOLD:.1f})")
            print(f"  Action       : sit out this weekend — 2-week cooling period")
            print("!" * 65)
            return True
        else:
            print(f"  KILL SWITCH  : inactive  (last 5R = {last5:+.3f}  threshold {KILL_SWITCH_THRESHOLD:.1f})")
            return False
    except Exception as exc:
        print(f"  Kill switch check error: {exc} — proceeding normally")
        return False


# ── Funding rate helpers ──────────────────────────────────────────────────────

def _classify_funding(rate_pct: float) -> str:
    """Classify a funding rate percentage into a named regime."""
    for lo, hi, label in _FUNDING_LEVELS:
        if lo <= rate_pct < hi:
            return label
    return "EXTREME"   # catch-all above


def _fetch_current_funding_rate(ticker: str) -> tuple[float | None, str]:
    """
    Fetch the single most recent funding rate for a ticker from Binance.
    Returns (rate_pct, classification) or (None, 'N/A') on any failure.
    Displayed for information only — not used as a hard filter in main.py.
    """
    symbol = _OKX_SYMBOL_MAP.get(ticker)
    if not symbol:
        return None, "N/A"
    try:
        resp = requests.get(
            _OKX_FUNDING_URL,
            params={"instId": symbol},
            timeout=_FUNDING_TIMEOUT,
        )
        if resp.status_code != 200:
            return None, "FETCH_ERROR"
        payload = resp.json()
        if payload.get("code") != "0" or not payload.get("data"):
            return None, "NO_DATA"
        rate_pct = float(payload["data"][0]["fundingRate"]) * 100   # decimal → %
        return round(rate_pct, 4), _classify_funding(rate_pct)
    except Exception:
        return None, "FETCH_ERROR"


# ── Schema validator ─────────────────────────────────────────────────────────

def validate_csv_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Candidate CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    print("  Schema validation: PASSED")


# ── Phase gate ───────────────────────────────────────────────────────────────

def _phase_gate_warning() -> None:
    if PHASE == 1:
        print()
        print("=" * 65)
        print("  PHASE 1 — PAPER TRADING MODE")
        print("  No real capital should be deployed until ALL gates clear:")
        print("    [ ] 8 weeks of paper trades logged")
        print("    [ ] Live win rate > 50%")
        print("    [ ] Positive expectancy after 32% short-term tax")
        print("=" * 65)
        print()


# ── Signal logic ─────────────────────────────────────────────────────────────

def _signal_for_ticker(
    ticker_df: pd.DataFrame,
    ticker: str,
    account_equity: float,
) -> dict:
    """
    Compute the signal row for a single instrument using the most recent
    available bar, regardless of day of week. When run after market close
    on the signal day itself, the latest row is always the correct signal bar.

    All three filters must pass for a LONG signal:
      1. Close > MA20
      2. ATR14 today > ATR14 prior Friday (volatility expanding)
      3. Close today > close prior Friday (momentum confirming)

    Returns a dict matching REQUIRED_COLUMNS plus internal _ keys.
    """
    if ticker_df.empty:
        raise RuntimeError(f"No data found for {ticker}.")

    fri_date = ticker_df.index[-1]
    fri_row  = ticker_df.iloc[-1]
    above_ma = bool(fri_row["above_ma20"])

    # ── Prior Friday lookups (ATR + close) ──────────────────────────────────
    all_fridays_idx = ticker_df.index[ticker_df.index.dayofweek == 4]
    prior_fridays   = all_fridays_idx[all_fridays_idx < fri_date]

    if len(prior_fridays) >= 1:
        prior_fri_row   = ticker_df.loc[prior_fridays[-1]]
        prior_fri_atr   = float(prior_fri_row["atr14"])
        prior_fri_close = float(prior_fri_row["close"])
        atr_expanding    = bool(fri_row["atr14"] > prior_fri_atr)
        momentum_confirm = bool(fri_row["close"] > prior_fri_close)
    else:
        prior_fri_atr    = None
        prior_fri_close  = None
        atr_expanding    = True   # can't verify — default allow
        momentum_confirm = True   # can't verify — default allow

    base = {
        "date":               fri_date.date(),
        "instrument":         ticker,
        "above_ma20":         "yes" if above_ma else "no",
        "R_target":           R_TARGET,
        # CSV columns (not stripped)
        "atr_expanding":      "yes" if atr_expanding    else "no",
        "momentum_confirm":   "yes" if momentum_confirm else "no",
        # internal keys (stripped before CSV)
        "_fri_close":         fri_row["close"],
        "_ma20":              fri_row["ma20"],
        "_atr14":             fri_row["atr14"],
        "_prior_fri_atr":     prior_fri_atr,
        "_prior_fri_close":   prior_fri_close,
        "_atr_expanding":     atr_expanding,
        "_momentum_confirm":  momentum_confirm,
        "_fri_date":          fri_date,
    }

    if not above_ma:
        return {
            **base,
            "entry_price":      None,
            "stop_price":       None,
            "target_price":     None,
            "direction":        "NO_TRADE",
            "risk_dollars":     None,
            "units":            None,
            "all_filters_pass": "no",
            "_no_trade_reason": "below MA20",
        }

    if not atr_expanding:
        return {
            **base,
            "entry_price":      None,
            "stop_price":       None,
            "target_price":     None,
            "direction":        "NO_TRADE",
            "risk_dollars":     None,
            "units":            None,
            "all_filters_pass": "no",
            "_no_trade_reason": "ATR contracting",
        }

    if not momentum_confirm:
        return {
            **base,
            "entry_price":      None,
            "stop_price":       None,
            "target_price":     None,
            "direction":        "NO_TRADE",
            "risk_dollars":     None,
            "units":            None,
            "all_filters_pass": "no",
            "_no_trade_reason": "momentum decelerating",
        }

    entry        = fri_row["close"] * (1 + SLIPPAGE_PCT)
    atr          = fri_row["atr14"]
    stop         = entry - ATR_MULT_STOP * atr
    target       = entry + R_TARGET * (entry - stop)
    risk_dollars = account_equity * RISK_PCT_PER_TRADE
    units        = risk_dollars / (entry - stop)

    return {
        **base,
        "entry_price":      round(entry, 2),
        "stop_price":       round(stop, 2),
        "target_price":     round(target, 2),
        "direction":        "LONG",
        "risk_dollars":     round(risk_dollars, 2),
        "units":            round(units, 6),
        "all_filters_pass": "yes",
        "_no_trade_reason": None,
    }


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_instrument_block(row: dict, account_equity: float) -> None:
    ticker   = row["instrument"]
    label    = ticker.replace("-USD", "")
    fri_date = row["_fri_date"]

    print(f"\n  ── {ticker}  ({fri_date.date()}) ──")
    print(f"  Friday close : ${row['_fri_close']:>12,.2f}")
    print(f"  MA20         : ${row['_ma20']:>12,.2f}")

    # Filter 1 — MA20
    f1 = "PASS" if row["above_ma20"] == "yes" else "FAIL"
    print(f"  Filter 1 (MA20)     : {f1}")

    # Filter 2 — ATR expansion
    if row["_prior_fri_atr"] is not None:
        f2     = "PASS" if row["_atr_expanding"] else "FAIL"
        cmp2   = ">" if row["_atr_expanding"] else "≤"
        print(f"  Filter 2 (ATR exp)  : {f2}"
              f"  (ATR14 ${row['_atr14']:,.2f} {cmp2} prev ${row['_prior_fri_atr']:,.2f})")
    else:
        print(f"  Filter 2 (ATR exp)  : N/A  (no prior Friday in data)")

    # Filter 3 — Momentum
    if row["_prior_fri_close"] is not None:
        f3     = "PASS" if row["_momentum_confirm"] else "FAIL"
        cmp3   = ">" if row["_momentum_confirm"] else "≤"
        print(f"  Filter 3 (Momentum) : {f3}"
              f"  (close ${row['_fri_close']:,.2f} {cmp3} prev Fri ${row['_prior_fri_close']:,.2f})")
    else:
        print(f"  Filter 3 (Momentum) : N/A  (no prior Friday in data)")

    # Funding rate (display-only — not a hard filter in Phase 1)
    fr_rate  = row.get("_funding_rate")
    fr_class = row.get("_funding_class", "N/A")
    if fr_rate is not None:
        print(f"  Funding rate : {fr_rate:>+10.4f}%  ({fr_class})")
    else:
        print(f"  Funding rate :   N/A  (fetch failed — check network)")

    # Kill switch override
    if row.get("_kill_switch_active"):
        if row["direction"] == "LONG":
            print(f"  SIGNAL       : NO_TRADE — kill switch active (computed: LONG suppressed)")
        else:
            print(f"  SIGNAL       : NO_TRADE — kill switch active")
        return

    if row["direction"] == "LONG":
        print(f"  SIGNAL       : LONG  ✓ all filters pass")
        print(f"  Entry        : ${row['entry_price']:>12,.2f}  (+0.1% slippage)")
        print(f"  Stop         : ${row['stop_price']:>12,.2f}  (-{ATR_MULT_STOP}× ATR14)")
        print(f"  Target       : ${row['target_price']:>12,.2f}  (+{R_TARGET}R)")
        print(f"  Risk $       : ${row['risk_dollars']:>12,.2f}  ({RISK_PCT_PER_TRADE*100:.0f}% of equity)")
        print(f"  Units        :  {row['units']:>12.6f} {label}")
    else:
        reason = row.get("_no_trade_reason", "")
        if reason == "ATR contracting":
            print(f"  SIGNAL       : NO_TRADE — ATR contracting, sit out")
        elif reason == "momentum decelerating":
            print(f"  SIGNAL       : NO_TRADE — momentum decelerating (close below prior Friday)")
        else:
            print(f"  SIGNAL       : NO_TRADE — below MA20, sit out")


def _print_summary(rows: list[dict], account_equity: float) -> None:
    print()
    print("=" * 65)
    print("  CRYPTO WEEKEND SYSTEM — Friday Signals")
    print(f"  Account equity : ${account_equity:,.2f}")
    print(f"  Weekend: stop={ATR_MULT_STOP}×ATR  target={R_TARGET}R  risk={RISK_PCT_PER_TRADE*100:.0f}%"
          f" — filters: MA20 + ATR expand + momentum")
    print(f"  Daily:   stop={ATR_MULT_STOP}×ATR  target={R_TARGET}R  risk={RISK_PCT_PER_TRADE*100:.0f}%"
          f" — BTC Monday / ETH Thursday 48hr hold")
    print("=" * 65)

    for row in rows:
        _print_instrument_block(row, account_equity)

    active = [r for r in rows if r["direction"] == "LONG"]
    kill_on = any(r.get("_kill_switch_active") for r in rows)
    print()
    if kill_on:
        print(f"  *** KILL SWITCH ACTIVE — {len(active)} LONG signal(s) suppressed ***")
        print(f"  Active trades  : 0 / {len(rows)}  (kill switch)")
    else:
        total_risk = sum(r["risk_dollars"] for r in active)
        print(f"  Active trades  : {len(active)} / {len(rows)}")
        if active:
            print(f"  Total $ at risk: ${total_risk:,.2f}  ({total_risk/account_equity*100:.1f}% of equity)")
    print()
    print("  NOTE: Paper trade only. Record outcomes by Sunday close.")
    print("=" * 65)


# ── Daily signal block ───────────────────────────────────────────────────────

# BTC Monday and ETH Thursday setups (from day-of-week analysis).
# Map: weekday → (ticker, entry_day_label, exit_day_label)
_DAILY_SETUPS = {
    0: ("BTC-USD", "Monday",   "Wednesday"),   # 48h hold → exit Wed close
    3: ("ETH-USD", "Thursday", "Saturday"),    # 48h hold → exit Sat close
}


def _print_daily_signals(
    live_data: dict,
    account_equity: float,
    kill_switch_active: bool = False,
) -> None:
    today_dow = date.today().weekday()   # 0=Mon … 6=Sun

    print()
    print("=" * 65)
    print("  DAILY MOMENTUM SIGNALS")
    print("=" * 65)

    if kill_switch_active:
        print("  KILL SWITCH ACTIVE — daily signals suppressed")
        print("=" * 65)
        return

    if today_dow not in _DAILY_SETUPS:
        print("  Not a signal day today (BTC Monday / ETH Thursday only)")
        print("=" * 65)
        return

    ticker, entry_day, exit_day = _DAILY_SETUPS[today_dow]
    label     = ticker.replace("-USD", "")
    ticker_df = live_data[ticker]
    row       = ticker_df.iloc[-1]

    above_ma = bool(row["above_ma20"])

    # ATR expansion vs 5 bars ago
    if len(ticker_df) >= 6:
        atr5ago = float(ticker_df["atr14"].iloc[-6])
        atr_exp = bool(row["atr14"] > atr5ago)
    else:
        atr5ago = None
        atr_exp = True   # insufficient history — default allow

    print(f"\n  ── {ticker} {entry_day} Daily Setup ──")
    print(f"  Close        : ${row['close']:>12,.2f}")
    print(f"  MA20         : ${row['ma20']:>12,.2f}")
    print(f"  Filter 1 (MA20)    : {'PASS' if above_ma else 'FAIL'}")
    if atr5ago is not None:
        cmp = ">" if atr_exp else "≤"
        print(f"  Filter 2 (ATR exp) : {'PASS' if atr_exp else 'FAIL'}"
              f"  (ATR14 ${row['atr14']:,.2f} {cmp} 5d-ago ${atr5ago:,.2f})")
    else:
        print(f"  Filter 2 (ATR exp) : N/A  (insufficient history)")

    if above_ma and atr_exp:
        entry    = row["close"] * (1 + SLIPPAGE_PCT)
        atr      = row["atr14"]
        stop     = entry - ATR_MULT_STOP * atr
        target   = entry + R_TARGET * (entry - stop)
        risk_d   = account_equity * RISK_PCT_PER_TRADE
        units    = risk_d / (entry - stop)
        print(f"  SIGNAL       : LONG — 48h hold (exit {exit_day} close)")
        print(f"  Entry        : ${entry:>12,.2f}  (+0.1% slippage)")
        print(f"  Stop         : ${stop:>12,.2f}  (-{ATR_MULT_STOP}× ATR14)")
        print(f"  Target       : ${target:>12,.2f}  (+{R_TARGET}R)")
        print(f"  Risk $       : ${risk_d:>12,.2f}  ({RISK_PCT_PER_TRADE*100:.0f}% of equity)")
        print(f"  Units        :  {units:>12.6f} {label}")
    else:
        reasons = []
        if not above_ma:
            reasons.append("below MA20")
        if not atr_exp:
            reasons.append("ATR not expanding")
        print(f"  SIGNAL       : NO_TRADE — {', '.join(reasons)}")

    print("=" * 65)


# ── Firebase signal writer ────────────────────────────────────────────────────

def _write_signal_to_firebase(rows: list[dict], fg_value: int | None, fg_label: str | None) -> None:
    """
    Write the Friday signal to Firestore via Firebase Admin SDK (service account auth).
    Authenticates using service_account.json; UID from firebase_config.json determines
    the document path: users/{uid}/crypto_signals/{fri_date}.
    Re-running on the same Friday overwrites cleanly (set with merge=False).
    Never crashes main.py — all errors are caught and printed.
    """
    import json as _json
    import firebase_admin
    from firebase_admin import credentials, firestore as _fs

    crypto_dir  = Path(__file__).parent
    sa_path     = crypto_dir / "service_account.json"
    config_path = crypto_dir / "firebase_config.json"

    if not sa_path.exists():
        print("  Firebase write : skipped (service_account.json not found)")
        return
    if not config_path.exists():
        print("  Firebase write : skipped (firebase_config.json not found)")
        return

    try:
        uid = _json.loads(config_path.read_text())["uid"]
    except Exception as exc:
        print(f"  Firebase write : config error — {exc}")
        return

    fri_date = rows[0]["_fri_date"].date().isoformat() if rows else str(date.today())

    def _row_to_dict(row: dict) -> dict:
        """Convert signal row to plain Python dict for Firestore Admin SDK."""
        is_long = row["direction"] == "LONG"
        d: dict = {
            "direction":          row["direction"],
            "above_ma20":         row.get("above_ma20", "no"),
            "filter1_pass":       row.get("above_ma20") == "yes",
            "filter1_close":      float(row.get("_fri_close", 0) or 0),
            "filter1_ma20":       float(row.get("_ma20", 0) or 0),
            "filter2_pass":       bool(row.get("_atr_expanding", False)),
            "filter2_atr":        float(row.get("_atr14", 0) or 0),
            "filter2_prev_atr":   float(row.get("_prior_fri_atr") or 0),
            "filter3_pass":       bool(row.get("_momentum_confirm", False)),
            "filter3_close":      float(row.get("_fri_close", 0) or 0),
            "filter3_prev_close": float(row.get("_prior_fri_close") or 0),
            "funding_rate":       float(row.get("_funding_rate") or 0),
            "funding_class":      row.get("_funding_class", "N/A"),
            "fear_greed_value":   fg_value,
            "fear_greed_class":   fg_label,
        }
        if is_long:
            d.update({
                "entry":           float(row.get("entry_price") or 0),
                "stop":            float(row.get("stop_price")  or 0),
                "target":          float(row.get("target_price") or 0),
                "units":           float(row.get("units") or 0),
                "risk_dollars":    float(row.get("risk_dollars") or 0),
                "no_trade_reason": None,
            })
        else:
            d["no_trade_reason"] = row.get("_no_trade_reason", "filters not met")
        return d

    try:
        # Initialise app once per process (guard against re-runs in same interpreter)
        app_name = "crypto_signal_writer"
        try:
            app = firebase_admin.get_app(app_name)
        except ValueError:
            cred = credentials.Certificate(str(sa_path))
            app  = firebase_admin.initialize_app(cred, name=app_name)

        db       = _fs.client(app)
        doc_ref  = db.collection("users").document(uid) \
                     .collection("crypto_signals").document(fri_date)

        payload: dict = {
            "date":         fri_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        for row in rows:
            key = "btc" if "BTC" in row["instrument"] else "eth"
            payload[key] = _row_to_dict(row)

        doc_ref.set(payload)
        print(f"  Firebase write : signal/{fri_date} written to Firestore")
    except Exception as exc:
        print(f"  Firebase write : error — {exc}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    _phase_gate_warning()
    kill_switch_active = check_kill_switch()

    # Note: ATR14 from Coinbase live data diverges ~2-5% from yfinance backtest data
    # due to different venue OHLC aggregation. Live stops will be slightly wider than
    # backtest predicts. This is expected — see data_audit_2026-03.txt for details.
    print(f"\nLoading data from Coinbase (today = {date.today()})...")
    live_data = load_coinbase_live(INSTRUMENTS)   # dict: ticker → DataFrame

    rows = []
    for ticker in INSTRUMENTS:
        row = _signal_for_ticker(live_data[ticker], ticker, ACCOUNT_EQUITY)
        # Attach current funding rate (display-only — fetched live from Binance)
        fr_rate, fr_class = _fetch_current_funding_rate(ticker)
        row["_funding_rate"]       = fr_rate
        row["_funding_class"]      = fr_class
        row["_kill_switch_active"] = kill_switch_active
        rows.append(row)

    # Build candidate DataFrame (drop internal _ keys before saving)
    csv_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in rows
    ]
    candidate = pd.DataFrame(csv_rows)

    # Validate schema
    validate_csv_schema(candidate, REQUIRED_COLUMNS)

    # Save
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    fri_date  = rows[0]["_fri_date"].date()
    save_path = CANDIDATES_DIR / f"candidate_{fri_date}.csv"
    candidate.to_csv(save_path, index=False)
    print(f"  Candidate saved → {save_path}")

    # Write signal to Firebase (non-blocking, never crashes)
    from dashboard_generator import _fetch_fear_greed
    try:
        _fg_val, _ = _fetch_fear_greed()
        _fg_lbl = None
        if _fg_val is not None:
            if _fg_val < 25: _fg_lbl = "Extreme Fear"
            elif _fg_val < 45: _fg_lbl = "Fear"
            elif _fg_val < 55: _fg_lbl = "Neutral"
            elif _fg_val < 75: _fg_lbl = "Greed"
            else: _fg_lbl = "Extreme Greed"
    except Exception:
        _fg_val, _fg_lbl = None, None
    _write_signal_to_firebase(rows, _fg_val, _fg_lbl)

    _print_summary(rows, ACCOUNT_EQUITY)
    _print_daily_signals(live_data, ACCOUNT_EQUITY, kill_switch_active)

    # Dashboard — generated last so any failure doesn't block signal output
    try:
        generate_dashboard(
            rows=rows,
            live_data=live_data,
            account_equity=ACCOUNT_EQUITY,
            weeks_logged=1,          # UPDATE: increment each Sunday after closing week
            candidates_dir=CANDIDATES_DIR,
        )
    except Exception as exc:
        print(f"  Dashboard generation failed (non-fatal): {exc}")


if __name__ == "__main__":
    main()
