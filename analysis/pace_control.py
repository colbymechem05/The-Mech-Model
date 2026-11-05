"""Pace control backfill, 2021-22 to 2025-26 (both leagues).

For every team-season, from games BEFORE the NCAA tournament:
  Control: when the two teams want different speeds, whose pace does the game end up at?
           50% = meets in the middle (what the line model assumes), 100% = always theirs.
  Payoff:  points per game better than expected (vs. ratings) in games played at their
           own pace, minus games played at the opponent's pace. + = better at own pace.
Then checks: is either number stable (split-half, year to year)? does the pace a game
ends up at get pushed around by the score (garbage time)? and do the 2x2 quadrants beat
expectations in the NCAA tournament?

Run: python analysis/pace_control.py   ->  prints results, writes output/pace_profiles.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D        # noqa: E402
import ratings as R     # noqa: E402

SEASONS = [2022, 2023, 2024, 2025, 2026]
NCAA = 22                 # ESPN tournament_id for the NCAA tournament (both leagues)
MIN_GAP = 2.5             # possessions: smaller pace gaps say nothing about who controlled it
K_CONTROL = 60.0          # shrink control toward 50% (in squared-possession units)
K_PAYOFF = 4.0            # shrink each side's average toward 0 (in games)
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Big East", "Pac-12"}
QUAD = {(True, True): "Enforcer", (True, False): "Stubborn",
        (False, True): "Chameleon", (False, False): "Pushover"}


def schedule(league, season):
    return pd.read_csv(D.RAW / f"{league}_schedule_{season}.csv", low_memory=False, usecols=[
        "game_id", "tournament_id", "home_id", "away_id", "home_conference_id", "away_conference_id",
        "groups_id", "groups_short_name", "conference_competition"])


def conf_names(sc):
    names = (sc[sc.conference_competition == True]  # noqa: E712
             .drop_duplicates("groups_id").set_index("groups_id").groups_short_name)
    out = pd.concat([sc.set_index("home_id").home_conference_id, sc.set_index("away_id").away_conference_id])
    out = out[~out.index.duplicated()].map(names)
    return out


def team_games(g, r, lg):
    """One row per team per game: own/opp preferred pace, actual pace, residual."""
    pace = (g.poss * 40 / g.minutes).values
    ph = lg["avg_tempo"] + r.loc[g.home_id, "T"].values
    pa = lg["avg_tempo"] + r.loc[g.away_id, "T"].values
    exp = r.loc[g.home_id, "net"].values - r.loc[g.away_id, "net"].values + np.where(g.neutral, 0, lg["hca"])
    res = 100 * g.margin.values / g.poss.values - exp           # per 100, home side
    k = lg["avg_tempo"] / 100                                    # per 100 -> points
    home = pd.DataFrame({"team": g.home_id.values, "own": ph, "opp": pa, "pace": pace, "res": res * k,
                         "won": g.margin.values > 0, "gid": g.game_id.values})
    away = pd.DataFrame({"team": g.away_id.values, "own": pa, "opp": ph, "pace": pace, "res": -res * k,
                         "won": g.margin.values < 0, "gid": g.game_id.values})
    tg = pd.concat([home, away], ignore_index=True)
    tg["gap"] = tg.own - tg.opp
    tg["tilt"] = (tg.pace - (tg.own + tg.opp) / 2)               # + = toward own pace
    return tg


def profile(tg):
    """Control % and Payoff (pts/game) per team, shrunk."""
    def one(t):
        c = 50 + 100 * (t.gap * t.tilt).sum() / ((t.gap ** 2).sum() + K_CONTROL)
        big = t[t.gap.abs() >= MIN_GAP]
        mine = big[big.tilt * np.sign(big.gap) > 0]
        theirs = big[big.tilt * np.sign(big.gap) <= 0]
        avg = lambda x: x.res.sum() / (len(x) + K_PAYOFF)  # noqa: E731
        return pd.Series({"control": c, "payoff": avg(mine) - avg(theirs),
                          "n_own": len(mine), "n_opp": len(theirs), "games": len(t)})
    return tg.groupby("team").apply(one, include_groups=False)


def split_half(tg):
    """Correlation of each number between odd and even games (Spearman-Brown corrected)."""
    tg = tg.sort_values("gid").copy()
    tg["half"] = tg.groupby("team").cumcount() % 2
    a, b = profile(tg[tg.half == 0]), profile(tg[tg.half == 1])
    out = {}
    for col in ["control", "payoff"]:
        r = a[col].corr(b[col].reindex(a.index))
        out[col] = 2 * r / (1 + r)
    return out


def season(league, s):
    games = D.load_games(league)
    sc = schedule(league, s)
    ncaa_ids = set(sc.loc[sc.tournament_id == NCAA, "game_id"])
    g_all = D.d1_games(games[games.season == s])
    t_games = g_all[g_all.game_id.isin(ncaa_ids)]
    start = t_games.date.min()
    pre = g_all[g_all.date < start]
    r, lg = R.fit(pre, start, None, R.Dials(hca_games=1.0))
    tg = team_games(pre, r, lg)
    prof = profile(tg)
    prof["tempo"] = lg["avg_tempo"] + r["T"].reindex(prof.index)
    prof["net"] = r["net"].reindex(prof.index)
    prof["conf"] = conf_names(sc).reindex(prof.index)
    prof["season"], prof["league"] = s, league
    prof["team_name"] = D.team_names(games).reindex(prof.index)
    # tournament games graded with pre-tournament ratings
    tt = team_games(t_games[t_games.home_id.isin(r.index) & t_games.away_id.isin(r.index)], r, lg)
    tt["season"], tt["league"] = s, league
    # garbage-time check: in games with a real pace gap, does the side that won by a lot get
    # "its" pace more often? (the score changes pace, not just style)
    big = tg[tg.gap.abs() >= MIN_GAP]
    toward = (big.tilt * np.sign(big.gap) > 0)
    mech = {"slow side wins big -> own pace %": 100 * toward[(big.gap < 0) & (big.res > 8)].mean(),
            "slow side loses big -> own pace %": 100 * toward[(big.gap < 0) & (big.res < -8)].mean()}
    return prof, tt, split_half(tg), mech


def main():
    pd.set_option("display.width", 220)
    allp, allt = [], []
    for league in D.LEAGUES:
        rel, mechs = [], []
        for s in SEASONS:
            prof, tt, sh, mech = season(league, s)
            allp.append(prof)
            allt.append(tt)
            rel.append({"season": s, **sh})
            mechs.append(mech)
        print(f"\n################ {league.upper()} ################")
        rel = pd.DataFrame(rel)
        print("Stability within a season (split-half reliability; 1 = rock solid, 0 = pure noise):")
        print(rel.round(2).to_string(index=False))
        print("Score pushes pace around? " + ", ".join(f"{k}: {np.mean([m[k] for m in mechs]):.0f}" for k in mechs[0]))
    P = pd.concat(allp).reset_index().rename(columns={"index": "team"})
    P["dictates"] = P.control > 50
    P["better_own"] = P.payoff > 0
    P["quadrant"] = [QUAD[(d, b)] for d, b in zip(P.dictates, P.better_own)]
    P["power"] = P.conf.isin(POWER)
    P.round(2).to_csv(D.ROOT / "output" / "pace_profiles.csv", index=False)

    # year-to-year: same team, next season
    for league in D.LEAGUES:
        p = P[P.league == league].set_index(["team", "season"])
        nxt = p.rename(index=lambda s: s - 1, level="season")
        j = p.join(nxt[["control", "payoff"]], rsuffix="_next", how="inner")
        print(f"\n{league.upper()} year-to-year correlation: control {j.control.corr(j.control_next):.2f}, "
              f"payoff {j.payoff.corr(j.payoff_next):.2f}   (teams: {len(j)})")
        print(f"  payoff vs. preferred tempo (slow teams look better at own pace?): "
              f"corr {p.payoff.corr(p.tempo):.2f}")

    # tournament
    T = pd.concat(allt)
    T = T.merge(P[["team", "season", "league", "quadrant", "power", "control", "payoff"]],
                on=["team", "season", "league"])
    print("\n######## NCAA tournament, 2022-2026: points per game vs. expectation (pre-tourney ratings) ########")
    for league in D.LEAGUES:
        t = T[T.league == league]
        q = t.groupby("quadrant").res.agg(["size", "mean", "sem"]).round(2)
        q.columns = ["team-games", "vs expected", "± (1 SE)"]
        print(f"\n{league.upper()} by quadrant\n{q.to_string()}")
        q2 = t.groupby(["power", "quadrant"]).res.agg(["size", "mean", "sem"]).round(2)
        q2.columns = ["team-games", "vs expected", "± (1 SE)"]
        q2.index = q2.index.set_levels(["non-P5/6", "Power"], level=0)
        print(f"\n{league.upper()} by conference type x quadrant\n{q2.to_string()}")

    for league in D.LEAGUES:
        p = P[(P.league == league) & (P.season == 2026) & (P.net > P[(P.league == league) & (P.season == 2026)].net.quantile(0.8))]
        print(f"\n{league.upper()} 2025-26, top-20% teams, most extreme profiles:")
        cols = ["team_name", "conf", "tempo", "control", "payoff", "quadrant"]
        print("  biggest dictators:\n" + p.nlargest(5, "control")[cols].round(1).to_string(index=False))
        print("  biggest chameleons:\n" + p.nsmallest(5, "control")[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
