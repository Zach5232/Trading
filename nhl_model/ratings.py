"""Pre-game team and goalie ratings from MoneyPuck game logs (improved model).

Everything is computed walk-forward: a game's ratings only use games played on
earlier dates. Empty-net goals are stripped by measuring goals and xG through
the goalies' own logs (a goalie is never on the ice for an empty-net goal).

Ratings per team:
  O  = goals-for rate   blend of GF/GP and xGF/GP (weight `a` on actual goals)
  D  = xG-against rate  team shot suppression, goalie-independent
Each is (current-season sum + k * prior) / (games played + k), i.e. the single
weight formula w = gp / (gp + k). The prior is last season's rate regressed
toward the league mean by `rho`.

Goalie factor f = (GA + kg) / (xGA + kg) over a goalie's decayed career history
(each past season multiplied by `decay`), so f < 1 means the goalie saves more
than expected. Goalies with no history get `f_new`.

Expected goals:  lam_home = O_home * D_away / L * f_away_goalie   (and vice versa)
"""
import json
import math
import os
from collections import defaultdict

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")
NORM = {"N.J": "NJD", "T.B": "TBL", "S.J": "SJS", "L.A": "LAK"}
# Franchise continuity for priors (ARI -> UTA in 2024-25)
PREV_ABBREV = {"UTA": "ARI"}

# kg=150 chosen by the 2021-24 hyperparameter search (backtest.py --search)
# hl: in-season recency half-life in games (None = all games weighted equally)
# sva: use MoneyPuck score/venue-adjusted team xG instead of xG faced by goalies
DEFAULTS = dict(a=0.5, k=20.0, rho=0.6, kg=150.0, decay=0.6, f_new=1.02, hl=None, sva=False)


def build_games(goalie_gbg, games, team_gbg=None):
    """One row per completed game with team/goalie inputs.

    goalie_gbg: MoneyPuck goalie game logs (situation == 'all')
    games:      list of dicts from the NHL API (id, season 'YYYYYYYY', date, away, home, ag, hg, last)
    team_gbg:   optional MoneyPuck team game logs, only needed for the `sva` option
    """
    g = goalie_gbg.copy()
    g["playerTeam"] = g.playerTeam.replace(NORM)
    per_team = g.groupby(["gameId", "playerTeam"]).agg(
        GA=("goals", "sum"), xGA=("xGoals", "sum")).reset_index()
    starters = (g.sort_values("icetime", ascending=False)
                 .drop_duplicates(["gameId", "playerTeam"])[["gameId", "playerTeam", "playerId", "name"]])
    goalie_games = g[["gameId", "playerId", "name", "playerTeam", "goals", "xGoals"]]

    games = pd.DataFrame(games)
    games["season"] = games.season.astype(str).str[:4].astype(int)
    games["date"] = pd.to_datetime(games.date)
    games = games.sort_values(["date", "id"]).reset_index(drop=True)

    sva = None
    if team_gbg is not None:
        t = team_gbg.copy()
        t["team"] = t.team.replace(NORM)
        sva = t.set_index(["gameId", "team"]).scoreVenueAdjustedxGoalsFor
    pt = per_team.set_index(["gameId", "playerTeam"])
    st = starters.set_index(["gameId", "playerTeam"])
    rows = []
    for r in games.itertuples():
        try:
            h, a = pt.loc[(r.id, r.home)], pt.loc[(r.id, r.away)]
        except KeyError:
            continue  # no MoneyPuck goalie data for this game (yet)
        rows.append(dict(
            gameId=r.id, season=r.season, date=r.date, home=r.home, away=r.away,
            hg=r.hg, ag=r.ag, last=r.last,
            h_GA=h.GA, h_xGA=h.xGA, a_GA=a.GA, a_xGA=a.xGA,
            h_sxGF=sva.get((r.id, r.home), float("nan")) if sva is not None else float("nan"),
            a_sxGF=sva.get((r.id, r.away), float("nan")) if sva is not None else float("nan"),
            h_goalie=st.loc[(r.id, r.home)].playerId, a_goalie=st.loc[(r.id, r.away)].playerId,
        ))
    df = pd.DataFrame(rows)
    rest = rest_features(df[["date", "home", "away"]])
    for k, v in rest.items():
        df[k] = v
    return df, goalie_games


def rest_features(sched):
    """Back-to-back, days of rest (capped at 4), 3 games in 4 nights, from the schedule itself.

    sched must be sorted by date and include every game each team played before the rows of interest.
    """
    played = defaultdict(list)
    feats = {k: [] for k in ("h_b2b", "a_b2b", "h_rest", "a_rest", "h_3in4", "a_3in4")}
    for r in sched.itertuples():
        for team, side in ((r.home, "h"), (r.away, "a")):
            prev = played[team]
            days = (r.date - prev[-1]).days if prev else 4
            feats[f"{side}_b2b"].append(days == 1)
            feats[f"{side}_rest"].append(min(days, 4))
            feats[f"{side}_3in4"].append(len(prev) >= 2 and (r.date - prev[-2]).days <= 3)
        for team in (r.home, r.away):
            played[team].append(r.date)
    return feats


def load_games(seasons=range(2020, 2025)):
    """Backtest inputs from nhl_model/data (see fetch_data.py)."""
    g = pd.read_csv(os.path.join(DATA, "goalies_gbg.csv"))
    g = g[g.season.isin(list(seasons))]
    games = [x for x in json.load(open(os.path.join(DATA, "nhl_games.json")))
             if int(str(x["season"])[:4]) in seasons]
    t = pd.read_csv(os.path.join(DATA, "teams_gbg.csv"))
    return build_games(g, games, t)


def compute(df, goalie_games, p=None, as_of_season=None):
    """Return df with pre-game lam_home / lam_away / goalie factors under params p.

    With as_of_season set, also return the ratings state after the last game:
    {"teams": {team: (O, D)}, "goalies": {playerId: f}, "L": league xG rate, "f_new": ...}.
    If as_of_season is later than the last season in df, the season rollover is applied
    so the state holds next season's preseason priors.
    """
    p = {**DEFAULTS, **(p or {})}
    a, k, rho, kg, decay, f_new = p["a"], p["k"], p["rho"], p["kg"], p["decay"], p["f_new"]
    shrink = 0.5 ** (1 / p["hl"]) if p["hl"] else 1.0

    gg = goalie_games.set_index("gameId")
    goalie_ga = defaultdict(float)
    goalie_xga = defaultdict(float)
    st = dict(season=None, prior={}, L=2.95, xL=2.95, cur=None)

    def rollover(new_season):
        if st["season"] is not None:
            cur = st["cur"]
            gp_all = sum(c["gp"] for c in cur.values())
            st["L"] = sum(c["GF"] for c in cur.values()) / gp_all
            st["xL"] = sum(c["xGF"] for c in cur.values()) / gp_all
            st["prior"] = {t: ((a * c["GF"] + (1 - a) * c["xGF"]) / c["gp"], c["xGA"] / c["gp"])
                           for t, c in cur.items()}
            for gid in list(goalie_ga):
                goalie_ga[gid] *= decay
                goalie_xga[gid] *= decay
        st["season"] = new_season
        st["cur"] = defaultdict(lambda: dict(gp=0, GF=0.0, xGF=0.0, xGA=0.0))

    def team_rating(t):
        prior, L, xL, c = st["prior"], st["L"], st["xL"], st["cur"][t]
        Lo = a * L + (1 - a) * xL
        pr = prior.get(t) or prior.get(PREV_ABBREV.get(t, ""))
        po, pd_ = pr if pr else (Lo, xL)
        po = Lo + rho * (po - Lo)
        pd_ = xL + rho * (pd_ - xL)
        o = (a * c["GF"] + (1 - a) * c["xGF"] + k * po) / (c["gp"] + k)
        d = (c["xGA"] + k * pd_) / (c["gp"] + k)
        return o, d

    def goalie_factor(gid):
        if goalie_xga[gid] <= 0:
            return f_new
        return (goalie_ga[gid] + kg) / (goalie_xga[gid] + kg)

    out = {c: [] for c in ("O_h", "O_a", "D_h", "D_a", "f_h", "f_a", "L")}
    for date, day in df.groupby("date", sort=True):
        if day.season.iloc[0] != st["season"]:
            rollover(day.season.iloc[0])
        for r in day.itertuples():
            oh, dh = team_rating(r.home)
            oa, da = team_rating(r.away)
            out["O_h"].append(oh); out["O_a"].append(oa)
            out["D_h"].append(dh); out["D_a"].append(da)
            out["f_h"].append(goalie_factor(r.h_goalie)); out["f_a"].append(goalie_factor(r.a_goalie))
            out["L"].append(st["xL"])
        # update after the whole day (ratings never see same-day results)
        for r in day.itertuples():
            if p["sva"]:
                xg_h, xg_a = r.h_sxGF, r.a_sxGF  # xG created by home / away
            else:
                xg_h, xg_a = r.a_xGA, r.h_xGA
            for t, gf, xgf, xga in ((r.home, r.a_GA, xg_h, xg_a), (r.away, r.h_GA, xg_a, xg_h)):
                c = st["cur"][t]
                for key in c:
                    c[key] *= shrink
                c["gp"] += 1; c["GF"] += gf; c["xGF"] += xgf; c["xGA"] += xga
            for _, gr in gg.loc[[r.gameId]].iterrows():
                goalie_ga[gr.playerId] += gr.goals
                goalie_xga[gr.playerId] += gr.xGoals

    res = df.copy()
    for c, v in out.items():
        res[c] = v
    res["lam_h"] = res.O_h * res.D_a / res.L * res.f_a
    res["lam_a"] = res.O_a * res.D_h / res.L * res.f_h
    res["x"] = (res.lam_h / res.lam_a).apply(math.log)
    if as_of_season is None:
        return res

    if as_of_season != st["season"]:
        rollover(as_of_season)
    teams = set(st["cur"]) | set(st["prior"]) | set(PREV_ABBREV)
    state = dict(
        season=as_of_season, L=st["xL"], f_new=f_new,
        teams={t: team_rating(t) for t in teams},
        gp={t: st["cur"][t]["gp"] for t in teams},
        goalies={int(g): goalie_factor(g) for g in goalie_xga if goalie_xga[g] > 0},
    )
    return res, state


def expected_goals(state, home, away, home_goalie=None, away_goalie=None):
    """lam_home, lam_away for a future game from a ratings state (unknown goalie -> f_new)."""
    oh, dh = state["teams"][home]
    oa, da = state["teams"][away]
    fh = state["goalies"].get(home_goalie, state["f_new"]) if home_goalie else 1.0
    fa = state["goalies"].get(away_goalie, state["f_new"]) if away_goalie else 1.0
    L = state["L"]
    return oh * da / L * fa, oa * dh / L * fh
