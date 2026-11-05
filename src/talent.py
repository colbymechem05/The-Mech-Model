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

import data as D
import players as P
import ratings as R
import recruits as RC

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
                 alpha: float = 1.0, new_value: float = -4.0,
                 new_by_program: bool = False, conf_feature: bool = False,
                 recruit_mode: str | None = None, recruit_source: str = "avg"):
        """new_by_program: newcomers' value depends on the program (learned from how past
        newcomers played vs. their team's previous-season rating) instead of one number.
        conf_feature: the team model also uses the conference's average last-season rating."""
        self.league, self.new_value = league, new_value
        self.new_by_program, self.conf_feature = new_by_program, conf_feature
        self.new_coef = (new_value, 0.0)
        self.confs = {s: D.conferences(league, s) for s in seasons} if conf_feature else {}
        # recruit_mode: None (off), "roster" (ranked freshmen's expected value added into the
        # roster value) or "feature" (a separate term in the team model, weight learned)
        self.recruit_mode = recruit_mode
        self.recruits = RC.load(league, recruit_source) if recruit_mode else None
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

        # 4b. newcomer value by program: newcomers' minutes-weighted value vs. team's last net
        if new_by_program:
            rows = []
            for s in train:
                if s - 1 not in seasons:
                    continue
                cur, prev = p[p.season == s], p[p.season == s - 1]
                new = cur[~cur.athlete_id.isin(prev.athlete_id) & (cur.minutes > 0)]
                agg = new.groupby("team_id").apply(
                    lambda x: pd.Series({"v": np.average(x.value, weights=x.minutes), "w": x.minutes.sum()}),
                    include_groups=False)
                last = self.truth[self.truth.season == s - 1].set_index("team_id").net
                agg["last_net"] = last.reindex(agg.index)
                rows.append(agg.dropna())
            nt = pd.concat(rows)
            A = np.column_stack([np.ones(len(nt)), nt.last_net]) * np.sqrt(nt.w.values)[:, None]
            self.new_coef = tuple(np.linalg.lstsq(A, nt.v.values * np.sqrt(nt.w.values), rcond=None)[0])

        # 4c. recruiting curves: what a freshman in each RSCI tier actually adds (minutes share
        # and share x value), relative to a typical unranked newcomer, from training seasons
        if recruit_mode and not self.recruits[self.recruits.season.isin(train)].empty:
            rows = []
            for s in train:
                m = RC.match(self.recruits[self.recruits.season == s], P.season_stats(league, s))
                m = m.merge(p[p.season == s][["athlete_id", "team_id", "min_share", "value"]]
                            .rename(columns={"team_id": "played_team"}), on=["athlete_id", "played_team"], how="left")
                rows.append(m)
            x = pd.concat(rows)
            x["min_share"] = x.min_share.fillna(0.0)
            x["contrib"] = x.min_share * x.value.fillna(0.0)
            new = []
            for s in train:
                if s - 1 in seasons:
                    a, b = p[p.season == s], p[p.season == s - 1]
                    new.append(a[~a.athlete_id.isin(b.athlete_id) & (a.minutes > 0) & ~a.athlete_id.isin(x.athlete_id.dropna())])
            n = pd.concat(new)
            self.v_typical = float(np.average(n.value, weights=n.minutes))
            t = x.groupby("tier", observed=False).agg(share=("min_share", "mean"), contrib=("contrib", "mean"))
            self.tier_boost = (t.contrib - t.share * self.v_typical).rename("boost").fillna(0.0)
            self.tier_table = t.assign(boost=self.tier_boost)
        elif recruit_mode:   # no recruiting data in the training seasons: the term stays at 0
            self.tier_boost = pd.Series(0.0, index=RC.LABELS)
            self.tier_table, self.v_typical = None, float("nan")

        # 5. team model, fitted on training seasons that have a previous season
        fs = [self._features(s, P.season_stats(league, s)) for s in train if s - 1 in seasons]
        f = pd.concat(fs, ignore_index=True)
        f = f.merge(self.truth[["season", "team_id", "net"]], on=["season", "team_id"])
        self.coef = np.linalg.lstsq(self._design(f), f.net.values, rcond=None)[0]

    def _design(self, f: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(f)), f.roster_value.values, f.last_net.values]
        if self.conf_feature:
            cols.append(f.conf_last.values)
        if self.recruit_mode == "feature":
            cols.append(f.recruit.values)
        return np.column_stack(cols)

    def recruit_boost(self, season: int, roster_now: pd.DataFrame) -> pd.Series:
        """Per team: sum of its ranked freshmen's expected boost. A recruit counts for the team
        he's actually playing for if he's on a roster, otherwise the school he signed with."""
        r = self.recruits[self.recruits.season == season]
        if r.empty:
            return pd.Series(dtype=float)
        m = RC.match(r, roster_now)
        team = m.played_team.fillna(m.team_id)
        b = m.tier.map(self.tier_boost).astype(float)
        return b.groupby(team).sum()

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
        last = self.truth[self.truth.season == season - 1].set_index("team_id").net
        new_v = self.new_coef[0] + self.new_coef[1] * last.reindex(filled.index).fillna(last.quantile(0.2))
        rv = (r.share * r.value.fillna(0)).groupby(r.team_id).sum() + (5 - filled) * new_v
        return pd.DataFrame({"roster_value": rv, "known_slots": filled,
                             "transfer_slots": (r.share * r.transfer).groupby(r.team_id).sum()})

    def _features(self, season: int, roster_now: pd.DataFrame) -> pd.DataFrame:
        f = self.roster_value(season, roster_now)
        last = self.truth[self.truth.season == season - 1].set_index("team_id").net
        f["last_net"] = last.reindex(f.index).fillna(last.quantile(0.2))  # new to D1: low
        if self.conf_feature:
            f["conf_last"] = self._conf_last(season, f.last_net)
        if self.recruit_mode:
            boost = self.recruit_boost(season, roster_now).reindex(f.index).fillna(0.0)
            if self.recruit_mode == "roster":
                f["roster_value"] += boost
            else:
                f["recruit"] = boost
        f["season"] = season
        return f.rename_axis("team_id").reset_index()

    def _conf_last(self, season: int, last_net: pd.Series) -> pd.Series:
        """Average last-season rating of the OTHER teams in each team's current conference."""
        if season not in self.confs:
            self.confs[season] = D.conferences(self.league, season)
        conf = self.confs[season].reindex(last_net.index)
        tot = last_net.groupby(conf).transform("sum")
        cnt = last_net.groupby(conf).transform("count")
        out = (tot - last_net) / (cnt - 1)
        return out.where(cnt > 1).fillna(last_net.median())

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
            if self.conf_feature:
                f["conf_last"] = self._conf_last(season, f.last_net)
            if self.recruit_mode:
                boost = self.recruit_boost(season, roster_now).reindex(missing).fillna(0.0)
                if self.recruit_mode == "roster":
                    f.loc[missing, "roster_value"] += boost
                else:
                    f.loc[missing, "recruit"] = boost
        pred = pd.Series(self._design(f) @ self.coef, index=f.index)
        out = base.copy()
        idx = out.index.intersection(pred.index)
        shift = (pred.loc[idx] - (out.loc[idx, "O"] - out.loc[idx, "D"])) / 2
        out.loc[idx, "O"] += shift
        out.loc[idx, "D"] -= shift
        out.attrs = dict(base.attrs)
        return out
