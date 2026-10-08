"""Download backtest inputs into nhl_model/data/.

- NHL API: regular-season schedule + final scores, 2020-21..2024-25
- MoneyPuck: team and goalie game-by-game logs (situation == 'all')
"""
import glob
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")
SEASONS = ["20202021", "20212022", "20222023", "20232024", "20242025"]
TEAMS = ("ANA ARI BOS BUF CGY CAR CHI COL CBJ DAL DET EDM FLA LAK MIN MTL NSH NJD NYI NYR "
         "OTT PHI PIT SJS SEA STL TBL TOR VAN VGK WSH WPG UTA").split()
MP = "https://moneypuck.com/moneypuck/playerData"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_nhl_games():
    games = {}
    for season in SEASONS:
        for t in TEAMS:
            try:
                d = json.loads(get(f"https://api-web.nhle.com/v1/club-schedule-season/{t}/{season}"))
            except Exception:
                continue  # team didn't exist that season
            for g in d.get("games", []):
                if g.get("gameType") != 2:
                    continue
                games[g["id"]] = dict(
                    id=g["id"], season=season, date=g["gameDate"], start=g.get("startTimeUTC"),
                    away=g["awayTeam"]["abbrev"], home=g["homeTeam"]["abbrev"],
                    ag=g["awayTeam"].get("score"), hg=g["homeTeam"].get("score"),
                    last=(g.get("gameOutcome") or {}).get("lastPeriodType"))
        print(season, len(games))
    json.dump(list(games.values()), open(os.path.join(DATA, "nhl_games.json"), "w"))


def fetch_moneypuck():
    years = range(2020, 2025)
    for y in years:
        for kind in ("teams", "goalies"):
            open(os.path.join(DATA, f"{kind}_{y}.csv"), "wb").write(
                get(f"{MP}/seasonSummary/{y}/regular/{kind}.csv"))
    raw = os.path.join(DATA, "all_teams_raw.csv")
    open(raw, "wb").write(get(f"{MP}/careers/gameByGame/all_teams.csv"))
    t = pd.read_csv(raw)
    t = t[t.season.between(2020, 2024) & (t.situation == "all") & (t.playoffGame == 0)]
    t.to_csv(os.path.join(DATA, "teams_gbg.csv"), index=False)
    os.remove(raw)

    ids = sorted(set(pd.concat(pd.read_csv(os.path.join(DATA, f"goalies_{y}.csv")) for y in years).playerId))
    gdir = os.path.join(DATA, "goalies")
    os.makedirs(gdir, exist_ok=True)

    def one(pid):
        open(os.path.join(gdir, f"{pid}.csv"), "wb").write(
            get(f"{MP}/careers/gameByGame/regular/goalies/{pid}.csv"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, ids))
    g = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(gdir, "*.csv")))
    g = g[g.season.between(2020, 2024) & (g.situation == "all")]
    g[["playerId", "season", "name", "gameId", "playerTeam", "opposingTeam", "home_or_away",
       "gameDate", "icetime", "xGoals", "goals", "ongoal"]].to_csv(
        os.path.join(DATA, "goalies_gbg.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    fetch_nhl_games()
    fetch_moneypuck()
