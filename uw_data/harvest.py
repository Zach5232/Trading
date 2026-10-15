"""
Unusual Whales data harvester -- pull everything the API plan allows into a local raw archive,
so future backtests and ideas never need the API again.

Every response is stored as-is (gzipped JSON, one file per request) under uw_data/data/raw/.
Nothing is parsed or thrown away at harvest time; loader.py turns the archive into DataFrames.

Resumable: a file that already exists is never requested again, and permanent failures
(403 outside the plan's lookback, 404, 422) leave a .err marker so they are not retried.
Run it once a day until the trial ends -- it stops on its own at the daily request limit.

    export UW_API_KEY=...            # never commit the key
    python3 uw_data/harvest.py --plan          # show what would be pulled, no requests
    python3 uw_data/harvest.py                 # harvest (snapshots -> P1 -> P2 -> P3)
    python3 uw_data/harvest.py --only p1       # one tier
    python3 uw_data/harvest.py --probe-full-tape   # check whether the plan includes the full tape

Tiers (in run order; newest dates first inside each tier):
  snap  Data with no history (prediction markets, calendars) -- snapshot it every day.
  p1    One request per ticker returns its whole history (options volume, GEX, IV rank, skew, daily bars).
  p2    Market-wide, one request per trading day (market tide, sector/ETF tide, OI change,
        net impact, unusual activity, daily report, flow alerts, SPY/QQQ/IWM GEX by strike).
  p3    Per ticker per trading day for the FOCUS list (dark pool, net premium ticks, OI change).
"""

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests

BASE = "https://api.unusualwhales.com"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
LOG = HERE / "data" / "harvest_log.jsonl"
STATE = HERE / "data" / "state.json"

DATA_START = dt.date(2022, 1, 1)  # UW history begins here; the plan's lookback may cut it shorter
DAILY_RESERVE = 200               # leave this many requests of the daily limit unused
# If the API doesn't report its daily limit in headers, stop each run at this many requests.
FALLBACK_RUN_CAP = int(os.environ.get("UW_RUN_CAP", "14000"))
# Stop cleanly after this many minutes so a CI job saves its work before the runner's time limit.
MAX_MINUTES = float(os.environ.get("UW_MAX_MINUTES", "0")) or None
STARTED = time.monotonic()

SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
           "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology",
           "Utilities"]
ETF_TIDE = ["SPY", "QQQ", "IWM", "DIA"]
INDEX_GEX = ["SPY", "QQQ", "IWM"]
EXTRA_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB",
                 "XLU", "XLRE", "XLC", "SMH", "TLT", "GLD", "IBIT", "COIN", "MSTR"]
# P3 costs one request per ticker per day per dataset, so it only covers a focus list.
FOCUS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
         "NFLX", "JPM", "COIN", "MSTR", "PLTR", "SMCI", "BA", "UNH", "LLY", "XOM", "COST", "WMT", "DIS"]
PRED_CATEGORIES = ["Crypto", "Culture", "Finance", "Politics", "Science", "Sports", "Technology"]


# ---------------------------------------------------------------- HTTP + rate limiting
class OutOfBudget(Exception):
    pass


def _is_quota_error(r):
    """A request/rate quota refusal (stop the run) vs. a plan-tier refusal (mark the endpoint .err)."""
    body = r.text.lower()
    if any(w in body for w in ("tier", "premium", "access", "invalid", "not found")):
        return False
    return "quota" in body or "too many" in body or ("limit" in body and any(
        w in body for w in ("daily", "request limit", "rate limit", "exceeded", "reached")))


class Client:
    def __init__(self, key, max_requests=None):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.max_requests = max_requests
        self.sent = 0
        self.daily_count = None
        self.daily_limit = None
        self.headers_shown = False

    def get(self, path, params=None, stream=False):
        if MAX_MINUTES and time.monotonic() - STARTED > MAX_MINUTES * 60:
            raise OutOfBudget(f"time budget of {MAX_MINUTES:.0f} min used -- next run continues")
        if self.max_requests is not None and self.sent >= self.max_requests:
            raise OutOfBudget(f"--max-requests {self.max_requests} reached")
        if self.daily_limit and self.daily_count is not None and self.daily_count >= self.daily_limit - DAILY_RESERVE:
            raise OutOfBudget(f"daily limit nearly used ({self.daily_count}/{self.daily_limit}); "
                              "resets 8 PM ET -- run again then")
        r = None
        for attempt in range(5):
            self.sent += 1
            try:
                r = self.s.get(BASE + path, params=params, timeout=60, stream=stream)
            except requests.RequestException as e:
                time.sleep(2 ** attempt)
                last = e
                continue
            self._read_limits(r)
            if not self.headers_shown:
                self.headers_shown = True
                print("  rate headers: " + (", ".join(f"{k}={v}" for k, v in r.headers.items()
                                                       if "req" in k.lower() or "limit" in k.lower()) or "none"))
            if self.daily_limit is None and self.sent >= FALLBACK_RUN_CAP:
                raise OutOfBudget(f"run cap {FALLBACK_RUN_CAP} reached (daily limit not reported)")
            if r.status_code in (403, 422) and _is_quota_error(r):
                raise OutOfBudget(f"API quota hit ({r.status_code}): {r.text[:200]}")
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 60))
                print(f"  429 rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            self._pace(r)
            return r
        if r is not None and r.status_code == 429:
            raise OutOfBudget("still rate limited after 5 tries")
        raise RuntimeError(f"giving up on {path}: {locals().get('last', 'repeated 5xx')}")

    def _read_limits(self, r):
        h = r.headers
        if "x-uw-daily-req-count" in h:
            self.daily_count = int(h["x-uw-daily-req-count"])
        if "x-uw-token-req-limit" in h:
            try:
                self.daily_limit = int(h["x-uw-token-req-limit"])
            except ValueError:
                pass

    @staticmethod
    def _pace(r):
        rem = r.headers.get("x-uw-req-per-minute-remaining")
        reset = r.headers.get("x-uw-req-per-minute-reset")
        if rem is not None and int(rem) <= 1 and reset:
            time.sleep(int(reset) / 1000 + 0.5)


def _data(payload):
    rows = payload.get("data", payload.get("result", [])) if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


# ---------------------------------------------------------------- storage
def out_path(dataset, key):
    safe = key.replace("/", "_").replace(" ", "_")
    return RAW / dataset / f"{safe}.json.gz"


def _load_done_list():
    """Paths (relative to RAW) already stored elsewhere -- set by the GitHub Action, whose checkout
    leaves the archive's files out to stay small. UW_DONE_LIST is a file of `git ls-tree` paths."""
    f = os.environ.get("UW_DONE_LIST")
    return set(Path(f).read_text().split()) if f and Path(f).exists() else set()


DONE_LIST = _load_done_list()


def done(p):
    if p.exists() or p.with_suffix(".err").exists():
        return True
    rel = p.relative_to(RAW).as_posix()
    return rel in DONE_LIST or rel[:-len(".json.gz")] + ".err" in DONE_LIST


def save(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(payload, f)
    os.replace(tmp, p)


def log(**rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec["at"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def fetch(client, dataset, key, path, params=None):
    """One request -> one file. Returns 'ok', 'skip' or 'err'."""
    p = out_path(dataset, key)
    if done(p):
        return "skip"
    r = client.get(path, params)
    if r.status_code == 200:
        try:
            payload = r.json()
        except ValueError:
            payload = {"_text": r.text}
        save(p, {"_request": {"path": path, "params": params, "fetched": dt.datetime.now().isoformat()},
                 **(payload if isinstance(payload, dict) else {"data": payload})})
        return "ok"
    if r.status_code == 401:
        sys.exit("401 Unauthorized -- check UW_API_KEY (expired trial or wrong key).")
    # permanent: outside lookback / not in plan / bad params -- remember it, don't retry
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".err").write_text(f"{r.status_code} {r.text[:500]}")
    log(dataset=dataset, key=key, status=r.status_code, body=r.text[:300])
    return "err"


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


# ---------------------------------------------------------------- calendar + lookback
def weekdays(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def trading_days(client, st, end):
    """Real sessions from SPY daily bars (one request, cached); weekdays as a fallback."""
    if st.get("trading_days"):
        days = [dt.date.fromisoformat(x) for x in st["trading_days"]]
        if days and days[-1] >= end - dt.timedelta(days=4):
            return days
    r = client.get("/api/stock/SPY/ohlc/1d", {"timeframe": "5Y", "limit": 2500})
    days = []
    if r.status_code == 200:
        for row in _data(r.json()):
            stamp = row.get("date") or row.get("market_time") or row.get("start_time") or ""
            try:
                days.append(dt.date.fromisoformat(str(stamp)[:10]))
            except ValueError:
                pass
    days = sorted({d for d in days if DATA_START <= d <= end})
    if len(days) < 50:
        print("  (could not read SPY sessions -- falling back to weekdays; holidays just return empty)")
        days = weekdays(DATA_START, end)
    st["trading_days"] = [d.isoformat() for d in days]
    return days


def find_lookback(client, st, days):
    """Earliest date the plan serves, by binary search on market tide (~10 requests, cached)."""
    if st.get("earliest"):
        return dt.date.fromisoformat(st["earliest"])
    ok = lambda d: client.get("/api/market/market-tide", {"date": d.isoformat()}).status_code == 200
    if ok(days[0]):
        earliest = days[0]
    else:
        lo, hi = 0, len(days) - 1          # days[lo] fails, days[hi] (latest) assumed ok
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(days[mid]):
                hi = mid
            else:
                lo = mid
        earliest = days[hi]
    st["earliest"] = earliest.isoformat()
    print(f"  plan lookback reaches back to {earliest}")
    return earliest


def load_universe():
    sys.path.insert(0, str(HERE.parent / "stock_model"))
    try:
        from data_loader import get_sp500_tickers
        tickers = get_sp500_tickers(save_path=str(HERE.parent / "stock_model/Data/raw_data/sp500_tickers.csv"))
    except Exception:  # noqa: BLE001 -- no stock_model here (e.g. the uw-data repo); read Wikipedia directly
        try:
            import io
            import pandas as pd
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
            tickers = pd.read_html(io.StringIO(html))[0]["Symbol"].astype(str).str.strip().tolist()
        except Exception as e:  # noqa: BLE001 -- universe is best-effort, extras still run
            print(f"  could not load S&P 500 list ({e}); using focus + extras only")
            tickers = []
    seen, out = set(), []
    for t in list(tickers) + EXTRA_TICKERS + FOCUS:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------- task lists
def snap_tasks(today):
    d = today.isoformat()
    t = [("pred_smart_money", f"{d}_all", "/api/predictions/smart-money", None),
         ("pred_whales", d, "/api/predictions/whales", None),
         ("pred_insiders", d, "/api/predictions/insiders", None),
         ("market_movers", d, "/api/market/movers", None),
         ("economic_calendar", d, "/api/market/economic-calendar", None),
         ("fda_calendar", d, "/api/market/fda-calendar", {"limit": 500}),
         ("congress_unusual", d, "/api/congress/unusual-trades", {"limit": 500}),
         ("insider_buy_sells", d, "/api/market/insider-buy-sells", {"limit": 500}),
         ("sector_etfs", d, "/api/market/sector-etfs", None)]
    for c in PRED_CATEGORIES:
        t.append(("pred_smart_money", f"{d}_{c}", "/api/predictions/smart-money", {"categories": c}))
        for off in (0, 50, 100):
            t.append(("pred_unusual", f"{d}_{c}_{off}", "/api/predictions/unusual",
                      {"categories": c, "limit": 50, "offset": off}))
    return t


def p1_tasks(tickers, earliest, today):
    months = max(1, (today.year - earliest.year) * 12 + today.month - earliest.month + 1)
    tf = f"{months}M"
    t = [("total_options_volume", today.isoformat(), "/api/market/total-options-volume", {"limit": 500})]
    for s in tickers:
        q = urllib.parse.quote(s)
        t += [("options_volume", s, f"/api/stock/{q}/options-volume", {"limit": 500}),
              ("greek_exposure", s, f"/api/stock/{q}/greek-exposure", {"timeframe": tf}),
              ("ohlc_1d", s, f"/api/stock/{q}/ohlc/1d", {"timeframe": tf, "limit": 2500}),
              ("short_volume", s, f"/api/shorts/{q}/volume-and-ratio", None)]
        # iv-rank returns at most a year per request: walk back a year at a time from today
        end = today
        while end > earliest:
            t.append(("iv_rank", f"{s}_{end.isoformat()}", f"/api/stock/{q}/iv-rank",
                      {"date": end.isoformat(), "timespan": "1y"}))
            end = end - dt.timedelta(days=365)
    return t


def p2_tasks(days):
    t = []
    for d in reversed(days):
        ds = d.isoformat()
        t += [("market_tide", ds, "/api/market/market-tide", {"date": ds, "interval_5m": "true"}),
              ("market_tide_otm", ds, "/api/market/market-tide", {"date": ds, "interval_5m": "true", "otm_only": "true"}),
              ("market_oi_change", ds, "/api/market/oi-change", {"date": ds, "limit": 200}),
              ("top_net_impact", ds, "/api/market/top-net-impact", {"date": ds, "limit": 100}),
              ("unusual_activity", ds, "/api/option-activity/unusual", {"date": ds, "limit": 200}),
              ("daily_report", ds, "/api/market/daily-report", {"date": ds, "limit": 100})]
        for sec in SECTORS:
            t.append(("sector_tide", f"{ds}_{sec}", f"/api/market/{urllib.parse.quote(sec)}/sector-tide", {"date": ds}))
        for e in ETF_TIDE:
            t.append(("etf_tide", f"{ds}_{e}", f"/api/market/{e}/etf-tide", {"date": ds}))
        for e in INDEX_GEX:
            t.append(("gex_by_strike", f"{ds}_{e}", f"/api/stock/{e}/greek-exposure/strike", {"date": ds}))
        t.append(("flow_alerts", ds, None, None))  # paginated, handled by harvest_flow_alerts
    return t


def p3_tasks(days, focus):
    t = []
    for d in reversed(days):
        ds = d.isoformat()
        for s in focus:
            t += [("net_prem_ticks", f"{ds}_{s}", f"/api/stock/{s}/net-prem-ticks", {"date": ds}),
                  ("darkpool_top", f"{ds}_{s}", f"/api/darkpool/{s}", {"date": ds, "limit": 500, "order_by": "premium"}),
                  ("ticker_oi_change", f"{ds}_{s}", f"/api/stock/{s}/oi-change", {"date": ds, "limit": 500})]
    return t


# ---------------------------------------------------------------- flow alerts (paginated by time)
MAX_ALERT_PAGES = 60


def _row_ms(row):
    for k in ("created_at", "executed_at", "start_time", "timestamp"):
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return int(v if v > 1e12 else v * 1000)
        try:
            return int(dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            continue
    return None


def harvest_flow_alerts(client, d):
    """All flow alerts for one session, newest page first, walking older_than back to the open."""
    day_dir = RAW / "flow_alerts" / d.isoformat()
    if (day_dir / "complete").exists() or f"flow_alerts/{d.isoformat()}/complete" in DONE_LIST \
            or done(out_path("flow_alerts", d.isoformat())):
        return "skip"
    start = int(dt.datetime(d.year, d.month, d.day, 4, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    end = int((dt.datetime(d.year, d.month, d.day, 4, 0, tzinfo=dt.timezone.utc) + dt.timedelta(days=1)).timestamp() * 1000)
    older = end
    existing = sorted(day_dir.glob("p*.json.gz"))
    if existing:  # resume after the last saved page
        with gzip.open(existing[-1], "rt") as f:
            older = json.load(f)["_request"]["next_older_than"]
    page = len(existing)
    while page < MAX_ALERT_PAGES:
        params = {"newer_than": start, "older_than": older, "limit": 200}
        r = client.get("/api/option-trades/flow-alerts", params)
        if r.status_code != 200:
            if page == 0:
                p = out_path("flow_alerts", d.isoformat())
                p.parent.mkdir(parents=True, exist_ok=True)
                p.with_suffix(".err").write_text(f"{r.status_code} {r.text[:500]}")
                log(dataset="flow_alerts", key=d.isoformat(), status=r.status_code, body=r.text[:300])
            return "err"
        rows = _data(r.json())
        stamps = [m for m in (_row_ms(x) for x in rows) if m]
        nxt = min(stamps) if stamps else None
        save(day_dir / f"p{page:03d}.json.gz",
             {"_request": {"params": params, "next_older_than": nxt}, "data": rows})
        page += 1
        if len(rows) < 200 or not nxt or nxt >= older:
            break
        older = nxt
    (day_dir / "complete").write_text(str(page))
    return "ok"


# ---------------------------------------------------------------- main
def run_tasks(client, name, tasks):
    counts = {"ok": 0, "skip": 0, "err": 0}
    todo = [t for t in tasks if t[2] is None or not done(out_path(t[0], t[1]))]
    print(f"[{name}] {len(tasks)} tasks, {len(todo)} left")
    for i, (ds, key, path, params) in enumerate(todo, 1):
        if path is None:
            res = harvest_flow_alerts(client, dt.date.fromisoformat(key))
        else:
            res = fetch(client, ds, key, path, params)
        counts[res] += 1
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  {counts}  daily {client.daily_count}/{client.daily_limit}")
    print(f"[{name}] done {counts}")


def probe_full_tape(client, days):
    d = days[-2].isoformat()
    r = client.get(f"/api/option-trades/full-tape/{d}", stream=True)
    size = r.headers.get("content-length")
    r.close()
    print(f"full tape {d}: HTTP {r.status_code}, size {int(size) / 1e9:.2f} GB" if size
          else f"full tape {d}: HTTP {r.status_code} (size unknown)")
    print("  200 = your plan includes it; download days with:\n"
          "  wget --header=\"Authorization: Bearer $UW_API_KEY\" "
          f"{BASE}/api/option-trades/full-tape/{d} -O full_tape_{d}.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["snap", "p1", "p2", "p3"], action="append",
                    help="run only these tiers (repeatable); default all")
    ap.add_argument("--plan", action="store_true", help="print task counts and exit (no data requests)")
    ap.add_argument("--max-requests", type=int, help="stop after this many requests this run")
    ap.add_argument("--focus", help="comma-separated P3 tickers (default: built-in focus list)")
    ap.add_argument("--probe-full-tape", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("UW_API_KEY")
    if not key:
        sys.exit("Set UW_API_KEY first (export UW_API_KEY=...). Never commit it.")
    client = Client(key, a.max_requests)
    st = load_state()
    today = dt.date.today()
    tiers = a.only or ["snap", "p1", "p2", "p3"]
    try:
        days = trading_days(client, st, today)
        earliest = find_lookback(client, st, days)
        save_state(st)
        days = [d for d in days if d >= earliest and d < today]  # today's session isn't final yet
        if a.probe_full_tape:
            probe_full_tape(client, days)
            return
        focus = a.focus.split(",") if a.focus else FOCUS
        plan = {}
        if "snap" in tiers:
            plan["snap"] = snap_tasks(today)
        if "p1" in tiers:
            plan["p1"] = p1_tasks(load_universe(), earliest, today)
        if "p2" in tiers:
            plan["p2"] = p2_tasks(days)
        if "p3" in tiers:
            plan["p3"] = p3_tasks(days, focus)
        if a.plan:
            print(f"sessions {days[0]} .. {days[-1]} ({len(days)})")
            for k, v in plan.items():
                left = sum(1 for t in v if t[2] is None or not done(out_path(t[0], t[1])))
                print(f"  {k}: {len(v)} tasks, {left} not yet stored")
            print(f"daily limit seen: {client.daily_limit} (used {client.daily_count})")
            return
        for k, v in plan.items():
            run_tasks(client, k, v)
        print("All tiers complete.")
    except OutOfBudget as e:
        print(f"Stopping: {e}")
    finally:
        save_state(st)
        print(f"requests this run: {client.sent}; daily {client.daily_count}/{client.daily_limit}")


if __name__ == "__main__":
    main()
