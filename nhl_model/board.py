"""Live NHL board: one run = rebuild ratings if stale, snapshot lines, price today's games,
log flagged picks, and grade finished games with closing-line value.

Run by GitHub Actions on a schedule (see .github/workflows/board.yml); safe to run any time.
  ODDS_API_KEY   The Odds API key (without it, games are priced without a market blend)

Outputs (committed by the workflow, served by Netlify from site/):
  state/ratings.json             current team/goalie ratings (rebuilt once per day)
  site/data/board.json           today's games, prices and flags
  site/data/picks.json           every game that crossed the edge threshold with confirmed goalies
  site/data/games.json           every priced game: model vs market at first look and at close, result
  site/data/snapshots/DATE.json  every line change seen per game and book (for closing lines / CLV)
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratings as R  # noqa: E402
import sources as S  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(ROOT, "state")
SITE_DATA = os.path.join(ROOT, "site", "data")
CFG = json.load(open(os.path.join(ROOT, "model_config.json")))
ET = ZoneInfo("America/New_York")

SHARP_BOOK = "pinnacle"
# Books you bet at (juiceReel history). Pinnacle is the market reference only.
BET_BOOKS = ["betonlineag", "espnbet", "williamhill_us", "betrivers", "betmgm", "fanduel", "draftkings",
             "fanatics", "ballybet"]
BOOK_NAMES = {"betonlineag": "BetOnline", "espnbet": "ESPN BET/theScore", "williamhill_us": "Caesars",
              "betrivers": "BetRivers", "betmgm": "BetMGM", "fanduel": "FanDuel", "draftkings": "DraftKings",
              "fanatics": "Fanatics", "ballybet": "Bally Bet", "pinnacle": "Pinnacle"}
ODDS_BOOKS = [SHARP_BOOK] + BET_BOOKS  # 10 books = 1 Odds API credit per snapshot


# ---------------- helpers ----------------

def load(path, default):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def implied(o):
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / -o


def novig_home(px):
    ih, ia = implied(px["home"]), implied(px["away"])
    return ih / (ih + ia)


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def mp_season(today):
    """MoneyPuck season year (2026 = 2026-27). New season from September."""
    return today.year if today.month >= 9 else today.year - 1


# ---------------- ratings ----------------

def rebuild_ratings(today):
    y = mp_season(today)
    seasons = list(range(y - 5, y + 1))
    print(f"[ratings] rebuilding for {y}-{y + 1} from seasons {seasons[0]}..{y}")
    logs, names = S.moneypuck_goalie_logs(seasons)
    games = []
    for s in seasons:
        g = S.season_games(f"{s}{s + 1}")
        games += [x for x in g.values() if x["state"] in ("OFF", "FINAL") and x["hg"] is not None]
    df, gg = R.build_games(logs, games)
    _, st = R.compute(df, gg, CFG["params"], as_of_season=y)

    # projected starter per team = most starts in the team's last 10 games (this season, else last season)
    logs = logs.copy()
    logs["playerTeam"] = logs.playerTeam.replace(R.NORM)
    starters = (logs.sort_values("icetime", ascending=False)
                    .drop_duplicates(["gameId", "playerTeam"]).sort_values("gameId"))
    team_goalies = {}
    for t, d in starters.groupby("playerTeam"):
        cur = d[d.season == y]
        recent = (cur if len(cur) else d[d.season == y - 1]).tail(10)
        counts = recent.groupby(["playerId", "name"]).size().sort_values(ascending=False)
        team_goalies[t] = [[int(pid), nm, int(n)] for (pid, nm), n in counts.items()]
    # most recent team for each goalie (trades / call-ups)
    last_team = starters.drop_duplicates("playerId", keep="last").set_index("playerId").playerTeam.to_dict()

    state = dict(built=today.isoformat(), season=y, L=st["L"], f_new=st["f_new"],
                 teams={t: list(v) for t, v in st["teams"].items()}, gp=st["gp"],
                 goalies={str(k): v for k, v in st["goalies"].items()},
                 goalie_names={str(k): v for k, v in names.items()},
                 goalie_team={str(k): v for k, v in last_team.items()},
                 team_goalies=team_goalies, games_used=len(df))
    save(os.path.join(STATE, "ratings.json"), state)
    print(f"[ratings] {len(df)} games, {len(state['goalies'])} goalies rated")
    return state


def ratings_state(today):
    st = load(os.path.join(STATE, "ratings.json"), None)
    if st is None or st.get("built") != today.isoformat() or "--rebuild" in sys.argv:
        st = rebuild_ratings(today)
    st["goalies"] = {int(k): v for k, v in st["goalies"].items()}
    return st


def find_goalie(st, name, team):
    """MoneyPuck playerId for a goalie name (prefers goalies last seen on `team`)."""
    if not name:
        return None
    n = S.norm(name)
    cands = [int(pid) for pid, nm in st["goalie_names"].items() if S.norm(nm) == n]
    if not cands:
        last = n.split()[-1]
        cands = [int(pid) for pid, nm in st["goalie_names"].items()
                 if S.norm(nm).split()[-1] == last and st["goalie_team"].get(pid) == team]
    if len(cands) > 1:
        cands = [c for c in cands if st["goalie_team"].get(str(c)) == team] or cands
    return cands[0] if cands else None


# ---------------- pricing ----------------

def goalies_for(st, game, df_goalies):
    """{'home': (id, name, status), 'away': ...}; status 'confirmed' only when DailyFaceoff says so."""
    out = {}
    entry = None
    for e in df_goalies:
        if S.norm(game["home"]["name"]) in S.norm(e["home_team"]) and S.norm(game["away"]["name"]) in S.norm(e["away_team"]):
            entry = e
            break
    for side in ("home", "away"):
        team = game[side]["abbrev"]
        name = entry and entry[f"{side}_goalie"]
        status = (entry[f"{side}_status"] if entry else "").lower()
        if name:
            gid = find_goalie(st, name, team)
            out[side] = (gid, name, "confirmed" if "confirm" in status else (status or "unconfirmed"))
        else:
            proj = (st["team_goalies"].get(team) or [[None, None, 0]])[0]
            out[side] = (proj[0], proj[1], "projected")
    return out


def price_game(st, game, ev, goalies, b2b, now):
    cfg = CFG
    h, a = game["home"]["abbrev"], game["away"]["abbrev"]
    lam_h, lam_a = R.expected_goals(st, h, a, goalies["home"][0], goalies["away"][0])
    p_model = sigmoid(cfg["home"] + cfg["slope"] * math.log(lam_h / lam_a)
                      + cfg["b2b"] * (float(b2b[a]) - float(b2b[h])))
    books = ev["books"] if ev else {}
    if SHARP_BOOK in books:
        p_mkt, mkt_src = novig_home(books[SHARP_BOOK]), "Pinnacle"
    elif books:
        p_mkt, mkt_src = float(np.median([novig_home(px) for px in books.values()])), "consensus"
    else:
        p_mkt, mkt_src = None, None
    p = sigmoid(cfg["w_model"] * logit(p_model) + cfg["w_market"] * logit(p_mkt)) if p_mkt else p_model

    best = {}
    for side in ("home", "away"):
        offers = [(px[side], bk) for bk, px in books.items() if bk in BET_BOOKS]
        if offers:
            price, bk = max(offers, key=lambda x: dec(x[0]))
            best[side] = dict(price=price, book=bk)
    sides = []
    for side, prob in (("home", p), ("away", 1 - p)):
        if side in best:
            d = dec(best[side]["price"])
            edge = prob * d - 1
            kelly = max(0.0, (prob * d - 1) / (d - 1)) * cfg["kelly_fraction"] * cfg["unit"]
            sides.append(dict(side=side, team=game[side]["abbrev"], prob=prob, edge=edge,
                              stake=round(kelly, 2), **best[side]))
    pick = max(sides, key=lambda s: s["edge"]) if sides else None
    started = game["state"] not in ("FUT", "PRE") or datetime.fromisoformat(game["start"].replace("Z", "+00:00")) <= now
    confirmed = goalies["home"][2] == "confirmed" and goalies["away"][2] == "confirmed"
    flag = bool(pick and pick["edge"] >= cfg["edge_threshold"] and p_mkt is not None)
    status = ("started" if started else
              "BET" if flag and confirmed else
              "edge - wait for goalies" if flag else "pass")
    return dict(
        id=game["id"], date=game["date"], start=game["start"], away=a, home=h,
        goalies={s: dict(id=v[0], name=v[1], status=v[2],
                         factor=round(st["goalies"].get(v[0], st["f_new"]), 3) if v[0] else None)
                 for s, v in goalies.items()},
        b2b=dict(home=b2b[h], away=b2b[a]), lam=dict(home=round(lam_h, 2), away=round(lam_a, 2)),
        p_model=p_model, p_market=p_mkt, market_source=mkt_src, p_blend=p, sides=sides, pick=pick,
        status=status, flag=flag and confirmed and not started,
        books={bk: px for bk, px in books.items()},
    )


# ---------------- snapshots / picks / grading ----------------

def record_snapshot(date, now, priced):
    path = os.path.join(SITE_DATA, "snapshots", f"{date}.json")
    snaps = load(path, [])
    last = {}
    for s in snaps:
        last[(s["game"], s["book"])] = (s["away"], s["home"])
    t = now.isoformat(timespec="seconds")
    added = 0
    for g in priced:
        if g["status"] == "started":
            continue
        for bk, px in g["books"].items():
            if last.get((g["id"], bk)) != (px["away"], px["home"]):
                snaps.append(dict(t=t, game=g["id"], book=bk, away=px["away"], home=px["home"]))
                added += 1
    save(path, snaps)
    return added


def closing(date, game_id, start, book):
    """Last price seen for a game/book before puck drop."""
    snaps = load(os.path.join(SITE_DATA, "snapshots", f"{date}.json"), [])
    rows = [s for s in snaps if s["game"] == game_id and s["book"] == book and s["t"] < start.replace("Z", "+00:00")]
    return rows[-1] if rows else None


def update_logs(priced, now):
    picks = load(os.path.join(SITE_DATA, "picks.json"), {})
    games = load(os.path.join(SITE_DATA, "games.json"), {})
    t = now.isoformat(timespec="seconds")
    for g in priced:
        if g["status"] == "started":
            continue
        k = str(g["id"])
        rec = games.setdefault(k, dict(id=g["id"], date=g["date"], start=g["start"], away=g["away"], home=g["home"],
                                       first=dict(t=t, p_model=g["p_model"], p_market=g["p_market"])))
        rec["last"] = dict(t=t, p_model=g["p_model"], p_market=g["p_market"], p_blend=g["p_blend"],
                           goalies={s: v["name"] for s, v in g["goalies"].items()},
                           goalie_status={s: v["status"] for s, v in g["goalies"].items()})
        pk = g["pick"]
        if g["flag"]:
            if k not in picks:
                picks[k] = dict(id=g["id"], date=g["date"], start=g["start"], away=g["away"], home=g["home"],
                                side=pk["side"], team=pk["team"], flagged_at=t, book=pk["book"], price=pk["price"],
                                prob=pk["prob"], edge=pk["edge"], stake=pk["stake"],
                                goalies={s: v["name"] for s, v in g["goalies"].items()})
            picks[k]["latest"] = dict(t=t, price=pk["price"] if pk["side"] == picks[k]["side"] else None,
                                      edge=pk["edge"] if pk["side"] == picks[k]["side"] else None, still_flagged=True)
        elif k in picks:
            same = [s for s in g["sides"] if s["side"] == picks[k]["side"]]
            picks[k]["latest"] = dict(t=t, price=same[0]["price"] if same else None,
                                      edge=same[0]["edge"] if same else None, still_flagged=False)
    save(os.path.join(SITE_DATA, "picks.json"), picks)
    save(os.path.join(SITE_DATA, "games.json"), games)
    return picks, games


def grade(now):
    """Fill results, closing lines and CLV for finished games."""
    picks = load(os.path.join(SITE_DATA, "picks.json"), {})
    games = load(os.path.join(SITE_DATA, "games.json"), {})
    changed = False
    for k, rec in games.items():
        if "result" in rec or datetime.fromisoformat(rec["start"].replace("Z", "+00:00")) > now - timedelta(hours=3):
            continue
        ag, hg, state = S.final_score(rec["id"])
        if state not in ("OFF", "FINAL"):
            continue
        rec["result"] = dict(away=ag, home=hg, home_win=int(hg > ag))
        cl = closing(rec["date"], rec["id"], rec["start"], SHARP_BOOK)
        rec["close"] = dict(p_market=novig_home(cl)) if cl else None
        changed = True
        if k in picks:
            pk = picks[k]
            won = (pk["side"] == "home") == bool(rec["result"]["home_win"])
            pk["result"] = "win" if won else "loss"
            pk["pl_flat"] = round(dec(pk["price"]) - 1 if won else -1, 4)
            pk["pl_stake"] = round(pk["stake"] * (dec(pk["price"]) - 1) if won else -pk["stake"], 2)
            same_book = closing(pk["date"], pk["id"], pk["start"], pk["book"])
            pk["close_price"] = same_book[pk["side"]] if same_book else None
            if cl:
                p_close = novig_home(cl) if pk["side"] == "home" else 1 - novig_home(cl)
                pk["close_prob"] = p_close
                pk["clv"] = p_close * dec(pk["price"]) - 1  # EV of the flagged price at Pinnacle's no-vig close
    if changed:
        save(os.path.join(SITE_DATA, "games.json"), games)
        save(os.path.join(SITE_DATA, "picks.json"), picks)


def summary(picks):
    done = [p for p in picks.values() if "result" in p]
    clv = [p["clv"] for p in done if p.get("clv") is not None]
    return dict(picks=len(picks), graded=len(done),
                wins=sum(p["result"] == "win" for p in done), losses=sum(p["result"] == "loss" for p in done),
                roi_flat=(sum(p["pl_flat"] for p in done) / len(done)) if done else None,
                pl_stake=round(sum(p["pl_stake"] for p in done), 2),
                staked=round(sum(p["stake"] for p in done), 2),
                avg_clv=(sum(clv) / len(clv)) if clv else None,
                beat_close=(sum(c > 0 for c in clv) / len(clv)) if clv else None)


# ---------------- main ----------------

def main():
    now = datetime.now(timezone.utc)
    today = now.astimezone(ET).date()
    st = ratings_state(today)
    grade(now)

    date = today.isoformat()
    sched = S.schedule_day(date)
    yesterday = S.schedule_day((today - timedelta(days=1)).isoformat())
    played_yday = {g[s]["abbrev"] for g in yesterday for s in ("home", "away")}
    b2b = {g[s]["abbrev"]: g[s]["abbrev"] in played_yday for g in sched for s in ("home", "away")}

    key = os.environ.get("ODDS_API_KEY", "")
    events, remaining = [], None
    if key and any(g["state"] in ("FUT", "PRE") for g in sched):
        try:
            events, remaining = S.odds_snapshot(key, ODDS_BOOKS)
        except Exception as e:
            print(f"[odds] {e}")
    elif not key:
        print("[odds] no ODDS_API_KEY set -- pricing without a market line")
    df_goalies = S.dailyfaceoff_goalies(date) if sched else []

    priced = []
    for g in sched:
        if g["home"]["abbrev"] not in st["teams"] or g["away"]["abbrev"] not in st["teams"]:
            print(f"[board] no ratings for {g['away']['abbrev']}@{g['home']['abbrev']}")
            continue
        priced.append(price_game(st, g, S.match_event(events, g), goalies_for(st, g, df_goalies), b2b, now))

    added = record_snapshot(date, now, priced)
    picks, _ = update_logs(priced, now)
    board = dict(updated=now.isoformat(timespec="seconds"), date=date, odds_credits_remaining=remaining,
                 ratings_built=st["built"], goalie_source="DailyFaceoff" if df_goalies else "projected (last 10 starts)",
                 config={k: CFG[k] for k in ("edge_threshold", "kelly_fraction", "unit", "w_model", "w_market")},
                 book_names=BOOK_NAMES, games=priced, record=summary(picks))
    save(os.path.join(SITE_DATA, "board.json"), board)
    print(f"[board] {date}: {len(priced)} games, {sum(g['flag'] for g in priced)} bets, "
          f"{added} line changes, odds credits left {remaining}")


if __name__ == "__main__":
    main()
