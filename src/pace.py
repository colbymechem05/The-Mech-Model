"""Pace profile for the website's Pace tab (this season, games so far).

For every game, each team has a preferred pace (its rated tempo vs. an average opponent).
  Control: when the two teams want different speeds, whose pace does the game end up at?
           50% = meets in the middle, 100% = always theirs. Shrunk toward 50% (K_CONTROL),
           so it means little until a team has played several mismatched games.
  Consistency: how much the team's actual pace moves from game to game (std. dev.).
Payoff (better at your own pace?) is not published: it tested as noise (analysis/pace_control.py).
"""
import numpy as np
import pandas as pd

import data as D

MIN_GAP = 2.5      # possessions: smaller pace gaps say nothing about who controlled it
K_CONTROL = 60.0   # shrink control toward 50% (in squared-possession units)


def team_games(g: pd.DataFrame, r: pd.DataFrame, lg: dict) -> pd.DataFrame:
    """One row per team per game: own/opp preferred pace, actual pace, residual (points)."""
    pace = (g.poss * 40 / g.minutes).values
    ph = lg["avg_tempo"] + r.loc[g.home_id, "T"].values
    pa = lg["avg_tempo"] + r.loc[g.away_id, "T"].values
    exp = r.loc[g.home_id, "net"].values - r.loc[g.away_id, "net"].values + np.where(g.neutral, 0, lg["hca"])
    res = (100 * g.margin.values / g.poss.values - exp) * lg["avg_tempo"] / 100
    home = pd.DataFrame({"team": g.home_id.values, "own": ph, "opp": pa, "pace": pace, "res": res,
                         "won": g.margin.values > 0, "gid": g.game_id.values})
    away = pd.DataFrame({"team": g.away_id.values, "own": pa, "opp": ph, "pace": pace, "res": -res,
                         "won": g.margin.values < 0, "gid": g.game_id.values})
    tg = pd.concat([home, away], ignore_index=True)
    tg["gap"] = tg.own - tg.opp
    tg["tilt"] = tg.pace - (tg.own + tg.opp) / 2          # + = toward own pace
    return tg


def control(tg: pd.DataFrame) -> pd.Series:
    """Pace control %, shrunk toward 50."""
    num = (tg.gap * tg.tilt).groupby(tg.team).sum()
    den = (tg.gap ** 2).groupby(tg.team).sum() + K_CONTROL
    return (50 + 100 * num / den).rename("control")


def profile(league: str, day: pd.Timestamp, r: pd.DataFrame, lg: dict) -> pd.DataFrame:
    """Per-team pace numbers from this season's D1-vs-D1 games before `day`."""
    g = D.d1_games(D.load_games(league))
    g = g[(g.season == D.CURRENT) & (g.date < day) & g.home_id.isin(r.index) & g.away_id.isin(r.index)]
    out = pd.DataFrame(index=r.index)
    out["tempo"] = lg["avg_tempo"] + r["T"]
    if g.empty:
        return out.assign(pace_avg=np.nan, pace_sd=np.nan, control=np.nan, battles=0, games=0)
    tg = team_games(g, r, lg)
    by = tg.groupby("team")
    out["pace_avg"] = by.pace.mean()
    out["pace_sd"] = by.pace.std()
    out["control"] = control(tg)
    out["battles"] = tg[tg.gap.abs() >= MIN_GAP].groupby("team").size()
    out["games"] = by.size()
    return out.fillna({"battles": 0, "games": 0})
