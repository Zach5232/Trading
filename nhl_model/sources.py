"""Live data sources: NHL API (schedule/results), MoneyPuck (goalie logs), The Odds API (lines),
DailyFaceoff (starting goalies)."""
import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (NHLModel pipeline)"}
NHL = "https://api-web.nhle.com/v1"
MP = "https://moneypuck.com/moneypuck/playerData"
ODDS = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"


def get(url, params=None, tries=3, timeout=30):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# ---------------- NHL API ----------------

# Every franchise abbreviation used since 2020-21 (ARI became UTA in 2024-25)
TEAMS = ("ANA ARI BOS BUF CGY CAR CHI COL CBJ DAL DET EDM FLA LAK MIN MTL NSH NJD NYI NYR "
         "OTT PHI PIT SJS SEA STL TBL TOR VAN VGK WSH WPG UTA").split()


def season_games(season, teams=TEAMS):
    """All regular-season games of a season from each team's schedule ({id: game})."""
    games = {}

    def one(t):
        r = get(f"{NHL}/club-schedule-season/{t}/{season}")
        return r.json().get("games", []) if r.status_code == 200 else []
    with ThreadPoolExecutor(8) as ex:
        for lst in ex.map(one, teams):
            for g in lst:
                if g.get("gameType") != 2:
                    continue
                games[g["id"]] = dict(
                    id=g["id"], season=season, date=g["gameDate"], start=g.get("startTimeUTC"),
                    state=g.get("gameState"), away=g["awayTeam"]["abbrev"], home=g["homeTeam"]["abbrev"],
                    ag=g["awayTeam"].get("score"), hg=g["homeTeam"].get("score"),
                    last=(g.get("gameOutcome") or {}).get("lastPeriodType"))
    return games


def schedule_day(date):
    """Games on an ET date 'YYYY-MM-DD' with team names (for matching odds feeds)."""
    d = get(f"{NHL}/schedule/{date}").json()
    out = []
    for wk in d.get("gameWeek", []):
        if wk["date"] != date:
            continue
        for g in wk["games"]:
            if g.get("gameType") != 2:
                continue
            side = {}
            for k in ("awayTeam", "homeTeam"):
                t = g[k]
                side[k] = dict(abbrev=t["abbrev"], place=t.get("placeName", {}).get("default", ""),
                               name=t.get("commonName", {}).get("default", ""), score=t.get("score"))
            out.append(dict(id=g["id"], date=date, start=g["startTimeUTC"], state=g["gameState"],
                            away=side["awayTeam"], home=side["homeTeam"],
                            last=(g.get("gameOutcome") or {}).get("lastPeriodType")))
    return out


def final_score(game_id):
    """(away goals, home goals, state) from the NHL API."""
    d = get(f"{NHL}/gamecenter/{game_id}/landing").json()
    return d["awayTeam"].get("score"), d["homeTeam"].get("score"), d.get("gameState")


# ---------------- MoneyPuck ----------------

def moneypuck_goalie_logs(seasons):
    """Goalie game logs (situation == 'all') for every goalie who played in `seasons` (MoneyPuck years)."""
    ids, names = set(), {}
    for y in seasons:
        r = get(f"{MP}/seasonSummary/{y}/regular/goalies.csv")
        if r.status_code != 200:
            continue
        s = pd.read_csv(io.StringIO(r.text))
        ids |= set(s.playerId)
        names.update(dict(zip(s.playerId, s.name)))

    def one(pid):
        r = get(f"{MP}/careers/gameByGame/regular/goalies/{pid}.csv")
        if r.status_code != 200 or not r.text.strip():
            return None
        d = pd.read_csv(io.StringIO(r.text))
        return d[(d.situation == "all") & d.season.isin(list(seasons))]
    with ThreadPoolExecutor(8) as ex:
        logs = [d for d in ex.map(one, sorted(ids)) if d is not None and len(d)]
    g = pd.concat(logs, ignore_index=True)
    return g[["playerId", "season", "name", "gameId", "playerTeam", "opposingTeam", "gameDate",
              "icetime", "xGoals", "goals"]], names


# ---------------- The Odds API ----------------

def odds_snapshot(api_key, bookmakers):
    """Current NHL moneylines. One request = 1 credit when <= 10 bookmakers are listed."""
    r = get(ODDS, params=dict(apiKey=api_key, markets="h2h", oddsFormat="american",
                              bookmakers=",".join(bookmakers)))
    if r.status_code != 200:
        raise RuntimeError(f"Odds API {r.status_code}: {r.text[:200]}")
    remaining = r.headers.get("x-requests-remaining")
    events = []
    for ev in r.json():
        books = {}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                px = {o["name"]: o["price"] for o in mk["outcomes"]}
                if ev["home_team"] in px and ev["away_team"] in px:
                    books[bk["key"]] = dict(home=px[ev["home_team"]], away=px[ev["away_team"]],
                                            updated=mk.get("last_update"))
        events.append(dict(id=ev["id"], start=ev["commence_time"], home_team=ev["home_team"],
                           away_team=ev["away_team"], books=books))
    return events, remaining


def match_event(events, game):
    """Find the odds event for an NHL API game by team nickname (e.g. 'Mammoth', 'Canadiens')."""
    hn, an = norm(game["home"]["name"]), norm(game["away"]["name"])
    for ev in events:
        if hn and an and hn in norm(ev["home_team"]) and an in norm(ev["away_team"]):
            return ev
    return None


# ---------------- DailyFaceoff starting goalies ----------------

def dailyfaceoff_goalies(date):
    """[{away_team, home_team, away_goalie, home_goalie, away_status, home_status}] for an ET date.

    Statuses are DailyFaceoff's ('Confirmed', 'Likely', 'Unconfirmed', ...). Returns [] if the
    page can't be read, so the pipeline falls back to projected starters.
    """
    try:
        r = get(f"https://www.dailyfaceoff.com/starting-goalies/{date}")
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text, re.S)
        if not m:
            return []
        data = json.loads(m.group(1))
    except Exception as e:  # network / layout change
        print(f"[goalies] DailyFaceoff unavailable: {e}")
        return []

    found = []

    def walk(x):
        if isinstance(x, dict):
            if "homeGoalieName" in x and "awayGoalieName" in x:
                found.append(x)
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    out = []
    for g in found:
        out.append(dict(
            home_team=g.get("homeTeamName") or g.get("homeTeamSlug") or "",
            away_team=g.get("awayTeamName") or g.get("awayTeamSlug") or "",
            home_goalie=g.get("homeGoalieName"), away_goalie=g.get("awayGoalieName"),
            home_status=g.get("homeNewsStrengthName") or "", away_status=g.get("awayNewsStrengthName") or "",
        ))
    if not out:
        print("[goalies] DailyFaceoff page parsed but no goalie entries found (layout changed?)")
    return out
