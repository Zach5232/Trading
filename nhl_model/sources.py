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


def starting_goalies(game_id):
    """{'home': playerId, 'away': playerId} of the goalies who actually started (after the game)."""
    d = get(f"{NHL}/gamecenter/{game_id}/boxscore").json()
    out = {}
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        for g in d.get("playerByGameStats", {}).get(key, {}).get("goalies", []):
            if g.get("starter"):
                out[side] = g["playerId"]
    return out


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


# ---------------- Starting goalie sources ----------------
#
# Every reader returns a list of entries:
#   {source, home, away (NHL abbrevs, or None), home_team, away_team (full names, for matching),
#    home_goalie, away_goalie (names), home_status, away_status ('confirmed' | 'likely' | 'unconfirmed'),
#    home_prob, away_prob (MoneyPuck's start probability, else None), home_note, away_note}
# A reader that fails returns [] and prints why, so one broken site never stops the board.

ABBREV_ALIAS = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "WAS": "WSH", "VEG": "VGK",
                "MON": "MTL", "CLS": "CBJ", "NAS": "NSH", "CAL": "CGY", "UTAH": "UTA", "WPJ": "WPG"}


def _abbrev(a):
    a = (a or "").upper().strip()
    return ABBREV_ALIAS.get(a, a) or None


def _status(raw):
    r = (raw or "").lower()
    if "confirm" in r:
        return "confirmed"
    if any(k in r for k in ("likely", "expected", "probable", "projected")):
        return "likely"
    return "unconfirmed"


def goalies_dailyfaceoff(date):
    try:
        r = get(f"https://www.dailyfaceoff.com/starting-goalies/{date}")
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', r.text, re.S)
        games = json.loads(m.group(1))["props"]["pageProps"]["data"]
    except Exception as e:
        print(f"[goalies] DailyFaceoff unavailable: {e}")
        return []
    out = []
    for g in games:
        e = dict(source="dailyfaceoff", home=None, away=None)
        for side in ("home", "away"):
            e[f"{side}_team"] = g.get(f"{side}TeamName") or ""
            e[f"{side}_goalie"] = g.get(f"{side}GoalieName")
            e[f"{side}_status"] = _status(g.get(f"{side}NewsStrengthName"))
            e[f"{side}_prob"] = None
            e[f"{side}_note"] = g.get(f"{side}NewsSourceName") or None
        out.append(e)
    return out


def goalies_rotowire(date):
    try:
        games = get("https://www.rotowire.com/hockey/tables/projected-goalies.php", params={"date": date}).json()
    except Exception as e:
        print(f"[goalies] RotoWire unavailable: {e}")
        return []
    out = []
    for g in games:
        out.append(dict(
            source="rotowire", home=_abbrev(g.get("hometeam")), away=_abbrev(g.get("visitteam")),
            home_team="", away_team="",
            home_goalie=g.get("homePlayer"), away_goalie=g.get("visitPlayer"),
            home_status=_status(g.get("homeStatus")), away_status=_status(g.get("visitStatus")),
            home_prob=None, away_prob=None, home_note=None, away_note=None))
    return out


def goalies_moneypuck(date):
    """MoneyPuck's games page: confirmed starters ('Starter: X', with the source) or start probabilities."""
    try:
        t = get(f"https://moneypuck.com/moneypuck/dates/{date.replace('-', '')}.htm").text
    except Exception as e:
        print(f"[goalies] MoneyPuck unavailable: {e}")
        return []
    t = re.sub(r"\s+", " ", t)
    out = []
    for row in re.findall(r"<tr>(.*?)</tr>", t, re.S | re.I):
        teams = re.findall(r"logos/([A-Z.]+)\.png", row)
        gid = re.search(r"preview\.htm\?id=(\d+)", row)
        cells = re.findall(r"<h2>\s*[\d.]+%\s*</h2>(.*?)</td>", row, re.S | re.I)
        if len(teams) != 2 or not gid or len(cells) != 2:
            continue
        e = dict(source="moneypuck", game_id=int(gid.group(1)), away=_abbrev(teams[0]), home=_abbrev(teams[1]),
                 home_team="", away_team="")
        for side, cell in (("away", cells[0]), ("home", cells[1])):
            text = re.sub(r"<[^>]+>", " ", cell)
            text = re.sub(r"\s+", " ", text).strip()
            conf = re.search(r"Starter:\s*([^:]+?)\s+Source:\s*(\S+)", text)
            prob = re.search(r"Starter:\s*([^:]+?):\s*([\d.]+)%", text)
            if conf:
                e[f"{side}_goalie"], e[f"{side}_status"], e[f"{side}_prob"], e[f"{side}_note"] = \
                    conf.group(1).strip(), "confirmed", 1.0, conf.group(2)
            elif prob:
                p = float(prob.group(2)) / 100
                e[f"{side}_goalie"], e[f"{side}_status"], e[f"{side}_prob"], e[f"{side}_note"] = \
                    prob.group(1).strip(), "likely" if p >= 0.5 else "unconfirmed", p, None
            else:
                e[f"{side}_goalie"], e[f"{side}_status"], e[f"{side}_prob"], e[f"{side}_note"] = None, "unconfirmed", None, None
        out.append(e)
    return out


GOALIE_SOURCES = {"dailyfaceoff": goalies_dailyfaceoff, "rotowire": goalies_rotowire, "moneypuck": goalies_moneypuck}


def all_goalie_sources(date):
    return {name: fn(date) for name, fn in GOALIE_SOURCES.items()}
