"""Roster-talent prior: preseason team ratings from who's actually on the roster
(returning players + incoming transfers + newcomers).

1. Team truth: each season's opponent-adjusted net rating.
2. Player box value: Hollinger Game Score per 100 possessions on court. The data
   only learns one slope + intercept that turns it into team strength.
3. Player value (net points per 100, per player-slot) = box value + `alpha` x the
   player's share of what box stats missed about their team (defense, scheme,
   schedule). Low-minute players are shrunk toward LOW_VALUE.
4. Roster value: every rostered player brings last season's value and minutes
   share from whatever D1 team they played for; newcomers (freshmen, non-D1,
   international) fill the leftover minutes at `new_value`.
5. Team preseason net = a + b x roster value + c x last season's team net.
"""
import numpy as np
import pandas as pd

import players as P
import ratings as R

GAMESCORE = {"points": 1.0, "field_goals_made": 0.4, "field_goals_attempted": -0.7,
             "free_throws_made": 0.4, "free_throws_attempted": -0.4,
             "offensive_rebounds": 0.7, "defensive_rebounds": 0.3, "steals": 1.0,
             "assists": 0.7, "blocks": 0.7, "fouls": -0.4, "turnovers": -1.0}
SHRINK_MINUTES = 250   # a player's value is shrunk as if he'd also played 250 min at LOW_VALUE
LOW_VALUE = -3.0


def team_truth(games: pd.DataFrame, seasons) -> pd.DataFrame:
    """Opponent-adjusted net per team-season (no prior; home court measured freely),
    plus total team possessions in all games so player stats line up."""
    out = []
    for s in seasons:
        g = games[games.season == s]
        r, _ = R.fit(g, g.date.max() + pd.Timedelta(days=1), None, R.Dials(hca_games=1.0))
        poss = pd.concat([g[["home_id", "poss"]].set_axis(["team_id", "poss"], axis=1),
                          g[["away_id", "poss"]].set_axis(["team_id", "poss"], axis=1)]
                         ).groupby("team_id").poss.sum()
        out.append(pd.DataFrame({"team_id": r.index, "season": s, "net": r.net.values,
                                 "team_poss": poss.reindex(r.index).values}))
    return pd.concat(out, ignore_index=True)


def player_table(league: str, seasons, truth: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: minutes share (sums to 5 per team) and Game Score per 100."""
    p = pd.concat([P.season_stats(league, s) for s in seasons], ignore_index=True)
    p = p.merge(truth, on=["season", "team_id"])  # D1 teams only
    slot_minutes = p.groupby(["season", "team_id"]).minutes.transform("sum") / 5
    p["min_share"] = np.where(slot_minutes > 0, p.minutes / slot_minutes, 0.0)
    poss_on = p.team_poss * p.min_share
    gs = sum(w * p[c] for c, w in GAMESCORE.items())
    p["gs100"] = np.where(poss_on > 0, 100 * gs / poss_on, 0.0)
    return p


class TalentPrior:
    """Fit on training seasons, then build a preseason prior for any season."""

    def __init__(self, league: str, games: pd.DataFrame, seasons, train,
                 alpha: float = 1.0, new_value: float = -4.0):
        self.league, self.new_value = league, new_value
        self.d1 = set(games[games.home_d1].home_id) | set(games[games.away_d1].away_id)
        self.truth = team_truth(games, seasons)
        p = player_table(league, seasons, self.truth)

        # 2. scale Game Score into team strength: team net ~ c0 + slope * team GS per 100
        tr = p[p.season.isin(train)]
        team_gs = (tr.min_share * tr.gs100).groupby([tr.season, tr.team_id]).sum()
        net = tr.groupby(["season", "team_id"]).net.first().reindex(team_gs.index)
        self.slope, self.c0 = np.polyfit(team_gs.values, net.values, 1)

        # 3. player values
        p["v_box"] = self.slope * p.gs100 + self.c0 / 5
        team_box = (p.min_share * p.v_box).groupby([p.season, p.team_id]).transform("sum")
        raw = p.v_box + alpha * (p.net - team_box) / 5
        p["value"] = (p.minutes * raw + SHRINK_MINUTES * LOW_VALUE) / (p.minutes + SHRINK_MINUTES)
        self.players = p

        # 5. team model, fitted on training seasons that have a previous season
        fs = [self._features(s, P.season_stats(league, s)) for s in train if s - 1 in seasons]
        f = pd.concat(fs, ignore_index=True)
        f = f.merge(self.truth[["season", "team_id", "net"]], on=["season", "team_id"])
        A = np.column_stack([np.ones(len(f)), f.roster_value, f.last_net])
        self.coef = np.linalg.lstsq(A, f.net.values, rcond=None)[0]

    def roster_value(self, season: int, roster_now: pd.DataFrame) -> pd.DataFrame:
        """4. Expected roster value per team for `season`."""
        last = self.players[(self.players.season == season - 1) & (self.players.minutes > 0)]
        last = (last.sort_values("minutes").drop_duplicates("athlete_id", keep="last")
                    .set_index("athlete_id")[["value", "min_share", "team_id"]]
                    .rename(columns={"team_id": "old_team"}))
        r = roster_now[["athlete_id", "team_id"]].drop_duplicates().join(last, on="athlete_id")
        r = r[r.team_id.isin(self.d1)]
        known = r.value.notna()
        r["share"] = r.min_share.clip(upper=1.0).where(known, 0.0)
        r["transfer"] = known & (r.old_team != r.team_id)
        slots = r.groupby("team_id").share.sum()
        scale = (5 / slots).clip(upper=1.0)                 # more than 5 slots back: scale down
        r["share"] *= r.team_id.map(scale)
        filled = r.groupby("team_id").share.sum()
        rv = (r.share * r.value.fillna(0)).groupby(r.team_id).sum() + (5 - filled) * self.new_value
        return pd.DataFrame({"roster_value": rv, "known_slots": filled,
                             "transfer_slots": (r.share * r.transfer).groupby(r.team_id).sum()})

    def _features(self, season: int, roster_now: pd.DataFrame) -> pd.DataFrame:
        f = self.roster_value(season, roster_now)
        last = self.truth[self.truth.season == season - 1].set_index("team_id").net
        f["last_net"] = last.reindex(f.index).fillna(last.quantile(0.2))  # new to D1: low
        f["season"] = season
        return f.rename_axis("team_id").reset_index()

    def prior(self, season: int, base: pd.DataFrame, roster_now: pd.DataFrame | None = None) -> pd.DataFrame:
        """Preseason prior: `base` (last season's O/D/T + attrs) with each team's net
        replaced by the talent prediction, keeping last season's O/D balance.
        Teams with no roster yet get a roster value estimated from last season."""
        roster_now = P.season_stats(self.league, season) if roster_now is None else roster_now
        f = self._features(season, roster_now).set_index("team_id")
        last = self.truth[self.truth.season == season - 1].set_index("team_id").net
        missing = base.index.difference(f.index).intersection(list(self.d1))
        if len(missing):
            slope, icpt = np.polyfit(f.last_net, f.roster_value, 1)
            ln = last.reindex(missing).fillna(last.quantile(0.2))
            f = pd.concat([f, pd.DataFrame({"roster_value": icpt + slope * ln, "last_net": ln},
                                           index=missing)])
        a, b, c = self.coef
        pred = a + b * f.roster_value + c * f.last_net
        out = base.copy()
        idx = out.index.intersection(pred.index)
        shift = (pred.loc[idx] - (out.loc[idx, "O"] - out.loc[idx, "D"])) / 2
        out.loc[idx, "O"] += shift
        out.loc[idx, "D"] -= shift
        out.attrs = dict(base.attrs)
        return out
