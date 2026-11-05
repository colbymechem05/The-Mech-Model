"""Efficiency power ratings (the core engine).

For every D1 team we solve, all at once:
  O = offense: points per 100 possessions above an average D1 offense
  D = defense: points per 100 ALLOWED above average (negative = good)
  T = tempo: possessions per 40 minutes above average
  plus one league-wide home court value (measured from the games).

Each game gives two equations, one per offense:
  home pts/100 = avg + O_home + D_away + h/2
  away pts/100 = avg + O_away + D_home - h/2      (h = full home edge per 100)

Solved by ridge regression centered on a preseason prior: each team's prior
counts as `prior_games` games of evidence, so real games take over quickly.
Net rating = O - D = points per 100 better than an average D1 team.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import norm

from data import d1_games

AVG_PRIOR_GAMES = 1500  # last season's league averages count as 1,500 games
DEFAULT_HCA = 4.5       # home edge per 100 possessions when nothing is known (~3 pts)


@dataclass
class Dials:
    prior_games: float = 3.0      # preseason prior is worth N games
    carryover: float = 1.0        # share of the prior rating kept
    half_life_days: float = 0.0   # recency: game weight halves every N days (0 = off)
    margin_cap: float = 0.0       # cap per-game efficiency gap per 100 (0 = off)
    hca_games: float = 5000.0     # how firmly home court sticks to its prior value
    sigma: dict = field(default_factory=lambda: {"mbb": 11.7, "wbb": 12.4})  # spread SD


def _ridge(X, y, w, prior_mean, strength):
    """Weighted ridge regression around prior_mean: minimizes
    sum w*(y - X b)^2 + sum strength*(b - prior_mean)^2."""
    if X.shape[0] == 0:
        return prior_mean.copy()
    sw = np.sqrt(w)
    A = sparse.vstack([X.multiply(sw[:, None]).tocsr(), sparse.diags(np.sqrt(strength))])
    b = np.concatenate([(y - X @ prior_mean) * sw, np.zeros(len(prior_mean))])
    return prior_mean + lsqr(A, b, atol=1e-10, btol=1e-10)[0]


def fit(games: pd.DataFrame, asof: pd.Timestamp, prior: pd.DataFrame | None,
        dials: Dials) -> tuple[pd.DataFrame, dict]:
    """Ratings from D1-vs-D1 games played BEFORE `asof` (no peeking).
    `prior`: DataFrame indexed by team_id with columns O, D, T; its attrs may
    carry last season's hca / avg_eff / avg_tempo."""
    g = d1_games(games[games.date < asof])
    has_prior = prior is not None
    attrs = prior.attrs if has_prior else {}
    teams = sorted(set(g.home_id) | set(g.away_id) | (set(prior.index) if has_prior else set()))
    idx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)

    # league averages: this season blended with last season's (worth AVG_PRIOR_GAMES)
    tempo40 = (g.poss * 40 / g.minutes).values
    k = AVG_PRIOR_GAMES if has_prior else 0
    prev_eff, prev_tempo = attrs.get("avg_eff", 100.0), attrs.get("avg_tempo", 68.0)
    cur_eff = 100 * (g.home_pts.sum() + g.away_pts.sum()) / (2 * g.poss.sum()) if m else prev_eff
    cur_tempo = tempo40.mean() if m else prev_tempo
    avg_eff = (k * prev_eff + m * cur_eff) / (k + m) if k + m else prev_eff
    avg_tempo = (k * prev_tempo + m * cur_tempo) / (k + m) if k + m else prev_tempo

    w = (0.5 ** ((asof - g.date).dt.days.values / dials.half_life_days)
         if dials.half_life_days and m else np.ones(m))
    eff_h = 100 * g.home_pts.values / g.poss.values
    eff_a = 100 * g.away_pts.values / g.poss.values
    if dials.margin_cap:  # dampen blowouts around each game's midpoint
        mid = (eff_h + eff_a) / 2
        gap = np.clip(eff_h - eff_a, -dials.margin_cap, dials.margin_cap)
        eff_h, eff_a = mid + gap / 2, mid - gap / 2

    hi, ai = g.home_id.map(idx).values, g.away_id.map(idx).values
    home = (~g.neutral.values).astype(float)
    rows = np.tile(np.arange(2 * m), 3)
    cols = np.concatenate([hi, ai, n + ai, n + hi, np.full(2 * m, 2 * n)])
    vals = np.concatenate([np.ones(4 * m), home / 2, -home / 2])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 * n + 1))
    y = np.concatenate([eff_h, eff_a]) - avg_eff

    p = np.zeros(2 * n + 1)
    pt = np.zeros(n)
    if has_prior:
        pr = prior.reindex(teams)
        p[:n] = dials.carryover * pr.O.fillna(0).values
        p[n:2 * n] = dials.carryover * pr.D.fillna(0).values
        pt = dials.carryover * pr["T"].fillna(0).values
    p[2 * n] = attrs.get("hca", DEFAULT_HCA)
    strength = np.concatenate([np.full(2 * n, dials.prior_games), [dials.hca_games]])
    beta = _ridge(X, y, np.concatenate([w, w]), p, strength)

    Xt = sparse.csr_matrix((np.ones(2 * m), (np.tile(np.arange(m), 2), np.concatenate([hi, ai]))),
                           shape=(m, n))
    T = _ridge(Xt, tempo40 - avg_tempo, w, pt, np.full(n, dials.prior_games))

    played = pd.concat([g.home_id, g.away_id]).value_counts()
    out = pd.DataFrame({"O": beta[:n], "D": beta[n:2 * n], "T": T}, index=teams)
    out["net"] = out.O - out.D
    out["games"] = played.reindex(out.index).fillna(0).astype(int)
    league = {"avg_eff": avg_eff, "avg_tempo": avg_tempo, "hca": beta[2 * n]}
    return out, league


def predict(r: pd.DataFrame, league: dict, home_ids, away_ids, neutral, sigma: float) -> pd.DataFrame:
    """Vectorized predictions (home perspective; spread > 0 = home favored)."""
    h, a = r.loc[list(home_ids)], r.loc[list(away_ids)]
    neutral = np.asarray(neutral, dtype=bool)
    tempo = league["avg_tempo"] + h["T"].values + a["T"].values
    hc = np.where(neutral, 0.0, league["hca"] / 2)
    eff_h = league["avg_eff"] + h.O.values + a.D.values + hc
    eff_a = league["avg_eff"] + a.O.values + h.D.values - hc
    pts_h, pts_a = eff_h * tempo / 100, eff_a * tempo / 100
    spread = pts_h - pts_a
    return pd.DataFrame({"home_pts": pts_h, "away_pts": pts_a, "spread_home": spread,
                         "total": pts_h + pts_a, "tempo": tempo,
                         "p_home": norm.cdf(spread / sigma)})


def hca_points(league: dict) -> float:
    """Home court in points for an average-tempo game."""
    return league["hca"] * league["avg_tempo"] / 100


def final_prior(games: pd.DataFrame, season: int, dials: Dials,
                prev_prior: pd.DataFrame | None = None) -> pd.DataFrame:
    """End-of-season ratings for `season` (next season's starting point).
    A full season is plenty to measure home court, so it's left free here."""
    s = games[games.season == season]
    r, league = fit(s, s.date.max() + pd.Timedelta(days=1), prev_prior, replace(dials, hca_games=1.0))
    out = r[["O", "D", "T"]].copy()
    out.attrs = {"hca": league["hca"], "avg_eff": league["avg_eff"], "avg_tempo": league["avg_tempo"]}
    return out


def backtest(games: pd.DataFrame, season: int, prior: pd.DataFrame | None, dials: Dials,
             league_key: str, refit_every: int = 1, end: str | None = None) -> pd.DataFrame:
    """Walk forward through a season: each day, fit on games before that day,
    then predict that day's games."""
    s = games[games.season == season]
    d1 = d1_games(s)
    if end:
        d1 = d1[d1.date <= end]
    out, r, league = [], None, None
    for k, (day, todays) in enumerate(d1.groupby("date", sort=True)):
        if r is None or k % refit_every == 0:
            r, league = fit(s, day, prior, dials)
        ok = todays.home_id.isin(r.index) & todays.away_id.isin(r.index)
        t = todays[ok]
        p = predict(r, league, t.home_id, t.away_id, t.neutral, dials.sigma[league_key])
        p.index = t.index
        out.append(pd.concat([t[["date", "game_id", "home", "away", "neutral", "margin"]], p], axis=1))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(bt: pd.DataFrame, start: str, end: str) -> dict:
    b = bt[(bt.date >= start) & (bt.date <= end)]
    won = (b.margin > 0).astype(float)
    return {"games": len(b),
            "spread_mae": round((b.spread_home - b.margin).abs().mean(), 2),
            "pick_acc": round(((b.p_home > .5) == (b.margin > 0)).mean(), 3),
            "brier": round(((b.p_home - won) ** 2).mean(), 4),
            "resid_sd": round((b.margin - b.spread_home).std(), 2)}
