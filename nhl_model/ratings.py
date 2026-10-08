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

DEFAULTS = dict(a=0.5, k=20.0, rho=0.6, kg=40.0, decay=0.6, f_new=1.02)


def load_games():
    """One row per regular-season game 2020-21..2024-25 with team/goalie inputs."""
    g = pd.read_csv(os.path.join(DATA, "goalies_gbg.csv"))
    g["playerTeam"] = g.playerTeam.replace(NORM)
    g = g[g.season.between(2020, 2024)]
    per_team = g.groupby(["gameId", "playerTeam"]).agg(
        GA=("goals", "sum"), xGA=("xGoals", "sum")).reset_index()
    starters = (g.sort_values("icetime", ascending=False)
                 .drop_duplicates(["gameId", "playerTeam"])[["gameId", "playerTeam", "playerId", "name"]])
    goalie_games = g[["gameId", "playerId", "name", "playerTeam", "goals", "xGoals"]]

    games = pd.DataFrame(json.load(open(os.path.join(DATA, "nhl_games.json"))))
    games["season"] = games.season.str[:4].astype(int)
    games = games[games.season.between(2020, 2024)]
    games["date"] = pd.to_datetime(games.date)
    games = games.sort_values(["date", "id"]).reset_index(drop=True)

    pt = per_team.set_index(["gameId", "playerTeam"])
    st = starters.set_index(["gameId", "playerTeam"])
    rows = []
    for r in games.itertuples():
        try:
            h, a = pt.loc[(r.id, r.home)], pt.loc[(r.id, r.away)]
        except KeyError:
            continue  # no MoneyPuck goalie data for this game
        rows.append(dict(
            gameId=r.id, season=r.season, date=r.date, home=r.home, away=r.away,
            hg=r.hg, ag=r.ag, last=r.last,
            h_GA=h.GA, h_xGA=h.xGA, a_GA=a.GA, a_xGA=a.xGA,
            h_goalie=st.loc[(r.id, r.home)].playerId, a_goalie=st.loc[(r.id, r.away)].playerId,
        ))
    df = pd.DataFrame(rows)
    # back-to-back from the schedule itself
    last_played = {}
    hb2b, ab2b = [], []
    for r in df.itertuples():
        for team, out in ((r.home, hb2b), (r.away, ab2b)):
            prev = last_played.get(team)
            out.append(prev is not None and (r.date - prev).days == 1)
        for team in (r.home, r.away):
            last_played[team] = r.date
    df["h_b2b"], df["a_b2b"] = hb2b, ab2b
    return df, goalie_games


def compute(df, goalie_games, p=None):
    """Return df with pre-game lam_home / lam_away / goalie factors under params p."""
    p = {**DEFAULTS, **(p or {})}
    a, k, rho, kg, decay, f_new = p["a"], p["k"], p["rho"], p["kg"], p["decay"], p["f_new"]

    gg = goalie_games.set_index("gameId")
    goalie_ga = defaultdict(float)
    goalie_xga = defaultdict(float)

    out = {c: [] for c in ("O_h", "O_a", "D_h", "D_a", "f_h", "f_a", "L")}
    season = None
    for date, day in df.groupby("date", sort=True):
        s = day.season.iloc[0]
        if s != season:
            if season is not None:
                # finalize prior-season rates, league mean, decay goalie history
                prior = {}
                gp_all = sum(cur[t]["gp"] for t in cur)
                L_new = sum(cur[t]["GF"] for t in cur) / gp_all
                xL_new = sum(cur[t]["xGF"] for t in cur) / gp_all
                for t, c in cur.items():
                    o = (a * c["GF"] + (1 - a) * c["xGF"]) / c["gp"]
                    d = c["xGA"] / c["gp"]
                    prior[t] = (o, d)
                L, xL = L_new, xL_new
                for gid in list(goalie_ga):
                    goalie_ga[gid] *= decay
                    goalie_xga[gid] *= decay
            else:
                prior, L, xL = {}, 2.95, 2.95
            season = s
            cur = defaultdict(lambda: dict(gp=0, GF=0.0, xGF=0.0, xGA=0.0))
            Lo = a * L + (1 - a) * xL

        def team_rating(t):
            pr = prior.get(t) or prior.get(PREV_ABBREV.get(t, ""))
            po, pd_ = pr if pr else (Lo, xL)
            po = Lo + rho * (po - Lo)
            pd_ = xL + rho * (pd_ - xL)
            c = cur[t]
            o = (a * c["GF"] + (1 - a) * c["xGF"] + k * po) / (c["gp"] + k)
            d = (c["xGA"] + k * pd_) / (c["gp"] + k)
            return o, d

        def goalie_factor(gid):
            if goalie_xga[gid] <= 0:
                return f_new
            return (goalie_ga[gid] + kg) / (goalie_xga[gid] + kg)

        for r in day.itertuples():
            oh, dh = team_rating(r.home)
            oa, da = team_rating(r.away)
            out["O_h"].append(oh); out["O_a"].append(oa)
            out["D_h"].append(dh); out["D_a"].append(da)
            out["f_h"].append(goalie_factor(r.h_goalie)); out["f_a"].append(goalie_factor(r.a_goalie))
            out["L"].append(xL)
        # update after the whole day (ratings never see same-day results)
        for r in day.itertuples():
            for t, gf, xgf, xga in ((r.home, r.a_GA, r.a_xGA, r.h_xGA), (r.away, r.h_GA, r.h_xGA, r.a_xGA)):
                c = cur[t]
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
    return res
