"""Backtest: old sheet model vs improved model, 2022-23..2024-25.

Usage:  python3 nhl_model/backtest.py [--search]

Constants (home, B2B, shrink, market weight) are fit by logistic regression and
always evaluated out-of-sample: each test season is scored with constants fit on
the other seasons (leave-one-season-out, LOSO). 2024-25 is also scored strictly
walk-forward (fit on 2021-22..2023-24 only). Rating hyperparameters are chosen
on 2021-22..2023-24 only when --search is passed.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import ratings as R  # noqa: E402

DATA = R.DATA
TEST = [2022, 2023, 2024]
SHEET_SEASON = {"2022-2023": 2022, "2023-2024": 2023, "2024-2025": 2024}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(X, y, intercept=True):
    X = np.column_stack([np.ones(len(X)), X]) if intercept else X

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z)
    b = minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    return b


def predict(b, X, intercept=True):
    X = np.column_stack([np.ones(len(X)), X]) if intercept else X
    return sigmoid(X @ b)


def model_X(df):
    return np.column_stack([df.x.values, df.a_b2b.astype(float).values - df.h_b2b.astype(float).values])


def american_to_prob(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def american_to_dec(o):
    o = np.asarray(o, dtype=float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / -o)


def loso_predict(df, cols_fn, seasons_test=TEST, train_pool=(2021, 2022, 2023, 2024), intercept=True):
    """Out-of-sample predictions: each test season uses constants fit on the others."""
    out = pd.Series(np.nan, index=df.index)
    coefs = {}
    for s in seasons_test:
        tr = df[df.season.isin([t for t in train_pool if t != s])]
        te = df[df.season == s]
        b = fit_logistic(cols_fn(tr), tr.hw.values, intercept)
        out.loc[te.index] = predict(b, cols_fn(te), intercept)
        coefs[s] = b
    return out, coefs


def load_ratings(params=None):
    df, gg = R.load_games()
    r = R.compute(df, gg, params)
    r["hw"] = (r.hg > r.ag).astype(int)
    return r


def search(df_games, gg):
    """Coordinate search of rating hyperparameters on 2021-22..2023-24 (LOSO within those)."""
    grid = dict(a=[0.0, 0.25, 0.5, 0.75], k=[10, 20, 35, 60], rho=[0.4, 0.6, 0.8],
                kg=[20, 40, 80, 150], decay=[0.4, 0.6, 0.8, 1.0], f_new=[1.0, 1.03, 1.06])
    best = dict(R.DEFAULTS)

    def score(p):
        r = R.compute(df_games, gg, p)
        r["hw"] = (r.hg > r.ag).astype(int)
        r = r[r.season.isin([2021, 2022, 2023])]
        pr, _ = loso_predict(r, model_X, seasons_test=[2021, 2022, 2023], train_pool=(2021, 2022, 2023))
        return logloss(pr.values, r.hw.values)
    cur = score(best)
    for _ in range(2):
        for key, vals in grid.items():
            for v in vals:
                if v == best[key]:
                    continue
                trial = {**best, key: v}
                s = score(trial)
                if s < cur - 1e-5:
                    cur, best = s, trial
            print(f"  {key}: {best[key]}  logloss {cur:.5f}", flush=True)
    return best, cur


def bet_sim(prob_home, df, threshold, kelly_frac=0.125):
    """Bet the side with edge > threshold at the sheet line (P/Q columns)."""
    dec_h, dec_a = american_to_dec(df.ph), american_to_dec(df.pa)
    ev_h = prob_home * dec_h - 1
    ev_a = (1 - prob_home) * dec_a - 1
    side_home = ev_h >= ev_a
    ev = np.where(side_home, ev_h, ev_a)
    dec = np.where(side_home, dec_h, dec_a)
    p = np.where(side_home, prob_home, 1 - prob_home)
    won = np.where(side_home, df.hw == 1, df.hw == 0)
    bet = ev > threshold
    flat = np.where(won, dec - 1, -1)[bet]
    kelly = np.clip((p * dec - 1) / (dec - 1), 0, None) * kelly_frac * 100  # $ on $100 unit
    k_pl = np.where(won, kelly * (dec - 1), -kelly)[bet]
    n = int(bet.sum())
    roi = flat.mean() if n else np.nan
    se = flat.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(bets=n, win_pct=float(won[bet].mean()) if n else np.nan, flat_roi=roi,
                roi_lo=roi - 1.96 * se, roi_hi=roi + 1.96 * se,
                kelly_pl=float(k_pl.sum()), kelly_staked=float(kelly[bet].sum()),
                home_share=float(side_home[bet].mean()) if n else np.nan)


def main():
    df_games, gg = R.load_games()
    params = dict(R.DEFAULTS)
    pfile = os.path.join(DATA, "best_params.json")
    if "--search" in sys.argv:
        params, ll = search(df_games, gg)
        json.dump(params, open(pfile, "w"), indent=1)
        print("best params", params, round(ll, 5))
    elif os.path.exists(pfile):
        params = json.load(open(pfile))
    r = R.compute(df_games, gg, params)
    r["hw"] = (r.hg > r.ag).astype(int)

    report = {"params": params}
    # ---------- 1. Model alone, all regular-season games ----------
    r["p_new"], coefs = loso_predict(r, model_X)
    wf = r[r.season.isin([2021, 2022, 2023])]
    b_wf = fit_logistic(model_X(wf), wf.hw.values)
    t24 = r.season == 2024
    r.loc[t24, "p_new_wf"] = predict(b_wf, model_X(r[t24]))
    allg = r[r.season.isin(TEST)]
    report["constants_fit_2021_23"] = dict(
        home_logodds=b_wf[0], home_odds_mult=float(np.exp(b_wf[0])), slope_on_log_lam_ratio=b_wf[1],
        b2b_logodds=b_wf[2])
    rows = []
    for s in TEST:
        x = allg[allg.season == s]
        rows.append(dict(season=s, games=len(x), home_win=x.hw.mean(),
                         ll_home_rate_only=logloss(np.full(len(x), x.hw.mean()), x.hw.values),
                         ll_new=logloss(x.p_new.values, x.hw.values),
                         acc_new=float(((x.p_new > .5) == (x.hw == 1)).mean())))
    report["all_games"] = rows
    report["ll_new_2024_walkforward"] = logloss(r[t24].p_new_wf.values, r[t24].hw.values)

    # ---------- 2. Sheet games: old vs new vs market ----------
    sh = pd.read_csv(os.path.join(DATA, "games.csv"))
    sh = sh[sh.gameId.notna()].copy()
    sh["gameId"] = sh.gameId.astype(int)
    sh = sh.rename(columns={"aw": "sheet_aw", "hw": "sheet_hw"})
    m = sh.merge(r[["gameId", "season", "hw", "x", "h_b2b", "a_b2b", "p_new", "lam_h", "lam_a", "f_h", "f_a"]],
                 on="gameId", how="inner")
    # old model: model-only (before market blend) and final (after blend)
    m["old_model"] = np.where(m.sheet == "2022-2023", m.fh, m.sh)
    m["old_final"] = m.fh
    ok_line = m.pa.notna() & m.ph.notna() & (m.pa.abs() >= 100) & (m.ph.abs() >= 100)
    ih, ia = american_to_prob(m.ph), american_to_prob(m.pa)
    m["hold"] = ih + ia - 1
    ok_line &= m.hold.between(0, 0.08)
    m["mkt"] = ih / (ih + ia)
    m["uba_mkt"] = np.where(m.ubh.notna() & m.uba.notna(),
                            american_to_prob(m.ubh) / (american_to_prob(m.ubh) + american_to_prob(m.uba)), np.nan)
    ok = ok_line & m.old_model.between(0.01, 0.99) & m.old_final.between(0.01, 0.99)
    m = m[ok].copy()

    # improved blend: logit(p) = w*logit(model) + (1-w)*logit(market), fit LOSO on sheet games
    def blend_X(d):
        return np.column_stack([logit(d.p_new.values), logit(d.mkt.values)])
    m["p_blend"], bcoef = loso_predict(m, blend_X, train_pool=TEST, intercept=False)
    # old model, same blend fit (to see if it adds anything over market)
    m["old_blend"], ocoef = loso_predict(
        m, lambda d: np.column_stack([logit(d.old_model.values), logit(d.mkt.values)]),
        train_pool=TEST, intercept=False)
    report["blend_weights_by_test_season"] = {s: dict(model=bcoef[s][0], market=bcoef[s][1]) for s in TEST}
    report["old_blend_weights"] = {s: dict(model=ocoef[s][0], market=ocoef[s][1]) for s in TEST}

    rows = []
    for s in TEST + ["all"]:
        x = m if s == "all" else m[m.season == s]
        y = x.hw.values
        rows.append(dict(season=s, games=len(x),
                         market=logloss(x.mkt.values, y),
                         old_model=logloss(x.old_model.values, y),
                         old_final=logloss(x.old_final.values, y),
                         new_model=logloss(x.p_new.values, y),
                         new_blend=logloss(x.p_blend.values, y),
                         old_refit_blend=logloss(x.old_blend.values, y),
                         old_mean_home=x.old_model.mean(), new_mean_home=x.p_new.mean(), actual=y.mean()))
    report["sheet_games_logloss"] = rows

    # ---------- 3. Betting simulation at the sheet line ----------
    sims = []
    for th in (0.0, 0.02, 0.04, 0.06):
        for name in ("old_final", "p_blend", "p_new", "old_model"):
            for s in TEST + ["all"]:
                x = m if s == "all" else m[m.season == s]
                sims.append(dict(model=name, threshold=th, season=s, **bet_sim(x[name].values, x, th)))
    report["bets"] = sims
    # edge buckets for the new blend vs old final
    buckets = []
    for name in ("old_final", "p_blend"):
        p = m[name].values
        dh, da = american_to_dec(m.ph), american_to_dec(m.pa)
        ev = np.maximum(p * dh - 1, (1 - p) * da - 1)
        for lo, hi in ((0, .02), (.02, .04), (.04, .06), (.06, .09), (.09, 1)):
            sel = (ev > lo) & (ev <= hi)
            res = bet_sim(m[name].values[sel], m[sel], -1)
            buckets.append(dict(model=name, edge=f"{lo:.0%}-{hi:.0%}", **res))
    report["edge_buckets"] = buckets

    m.to_csv(os.path.join(DATA, "backtest_games.csv"), index=False)
    json.dump(report, open(os.path.join(DATA, "backtest_report.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(json.dumps({k: report[k] for k in ("params", "constants_fit_2021_23", "ll_new_2024_walkforward",
                                             "blend_weights_by_test_season", "old_blend_weights")},
                     indent=1, default=float))
    print(pd.DataFrame(report["all_games"]).round(4).to_string(index=False))
    print(pd.DataFrame(report["sheet_games_logloss"]).round(4).to_string(index=False))
    b = pd.DataFrame(report["bets"])
    print(b[b.season == "all"].round(3).to_string(index=False))
    print(pd.DataFrame(buckets).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
