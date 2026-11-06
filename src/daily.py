"""Daily run: refresh data, rate every D1 team, make lines for a date's games,
and append them to a permanent log (saved before tip-off; never overwritten).

Usage:  python src/daily.py [YYYY-MM-DD] [--no-refresh]     (default: today)
Outputs per league (mbb, wbb):
  output/ratings_{lg}.csv          power ratings, all D1
  output/lines_{lg}_{date}.csv     that day's lines
  logs/predictions_{lg}.csv        every line ever made (append-only)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
import data as D  # noqa: E402
import players as P  # noqa: E402
import ratings as R  # noqa: E402
import rosters as RS  # noqa: E402
import talent as T  # noqa: E402

OUT = D.ROOT / "output"
LOGS = D.ROOT / "logs"
HISTORY = [s for s in D.SEASONS if s < D.CURRENT]
# Tuned on 2021-22..2024-25, checked on 2025-26 (see build log; analysis/differentiators.py).
SETTINGS = {"mbb": dict(new_value=-4.0, prior_games=6, recruits="feature"),   # RSCI top 100 freshmen
            "wbb": dict(new_value=-4.0, prior_games=4, recruits="feature")}  # avg of ESPN/On3/247 (2025+)
CONF_GAMES = 30   # in-season conference pooling (analysis/buy_game_fix.py)
# Buy games: Power team at home vs. a non-Power visitor in Nov-Dec. Extra points for the home
# team = a + b x our spread (analysis/buy_game_term.py, fitted on 2021-22..2025-26).
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Big East", "Pac-12"}
BUY_BOOST = {"mbb": (2.33, 0.0), "wbb": (0.61, 0.08)}


def buy_boost(league: str, day: pd.Timestamp, home_ids, away_ids, neutral, spread, conf) -> np.ndarray:
    """Points added to the home side in buy games (0 everywhere else)."""
    if day.month not in (11, 12):
        return np.zeros(len(spread))
    a, b = BUY_BOOST[league]
    buy = (pd.Series(list(home_ids)).map(conf).isin(POWER).values
           & ~pd.Series(list(away_ids)).map(conf).isin(POWER).values & ~np.asarray(neutral, dtype=bool))
    return np.where(buy, np.maximum(a + b * np.asarray(spread), 0.0), 0.0)


def todays_games(league: str, day: pd.Timestamp) -> pd.DataFrame:
    s = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                    usecols=["game_id", "game_date", "game_date_time", "home_id", "away_id",
                             "home_display_name", "away_display_name", "neutral_site"])
    s = s[pd.to_datetime(s.game_date) == day]
    return s.assign(neutral=s.neutral_site.fillna(False).astype(bool))


def rate(league: str, day: pd.Timestamp, refresh: bool):
    cfg = SETTINGS[league]
    games = D.build(league, refresh_current=refresh)
    roster_now = P.season_stats(league, D.CURRENT, refresh=refresh)
    talent = T.TalentPrior(league, games, HISTORY, train=HISTORY, new_value=cfg["new_value"],
                           conf_feature=True, recruit_mode=cfg["recruits"])
    prior = talent.prior(D.CURRENT, R.final_prior(games, D.CURRENT - 1, R.Dials()), roster_now)
    prior.attrs["team_hca"] = R.team_hca(games, D.CURRENT)   # each team's own extra home edge
    prior.attrs["conf"] = D.conferences(league, D.CURRENT)    # for conference pooling
    dials = R.Dials(prior_games=cfg["prior_games"], conf_games=CONF_GAMES)
    r, lg_info = R.fit(games[games.season == D.CURRENT], day, prior, dials)

    d1 = D.d1_games(games)
    r = r[r.index.isin(set(d1.home_id) | set(d1.away_id))].sort_values("net", ascending=False)
    r.insert(0, "rank", range(1, len(r) + 1))
    r["team"] = D.team_names(games).reindex(r.index).values
    r["tempo"] = lg_info["avg_tempo"] + r["T"]
    r["preseason"] = (prior.O - prior.D).reindex(r.index)
    r["returning_pct"] = 100 * RS.returning(league, D.CURRENT, roster_now).reindex(r.index)
    extra = prior.attrs["team_hca"].reindex(r.index).fillna(0)
    r["home_court"] = (lg_info["hca"] + extra) * lg_info["avg_tempo"] / 100   # points
    return r, lg_info, dials, prior.attrs["conf"]


def make_lines(league: str, day: pd.Timestamp, r: pd.DataFrame, lg_info: dict,
               dials: R.Dials, conf: pd.Series | None = None) -> pd.DataFrame:
    g = todays_games(league, day)
    g = g[g.home_id.isin(r.index) & g.away_id.isin(r.index)]   # D1 vs D1 only
    if g.empty:
        return pd.DataFrame()
    p = R.predict(r, lg_info, g.home_id, g.away_id, g.neutral, dials.sigma[league])
    if conf is not None:
        extra = buy_boost(league, day, g.home_id, g.away_id, g.neutral, p.spread_home.values, conf)
        p["home_pts"] += extra / 2
        p["away_pts"] -= extra / 2
        p["spread_home"] += extra
        p["p_home"] = norm.cdf(p.spread_home / dials.sigma[league])
    g, fav_home = g.reset_index(drop=True), p.spread_home >= 0
    return pd.DataFrame({
        "date": day.date(), "game_id": g.game_id,
        "tip": pd.to_datetime(g.game_date_time.str[:19]).dt.strftime("%H:%M").values,  # Eastern
        "away": g.away_display_name, "home": g.home_display_name, "neutral": g.neutral,
        "away_rank": r.loc[g.away_id, "rank"].values, "home_rank": r.loc[g.home_id, "rank"].values,
        "favorite": g.home_display_name.where(fav_home, g.away_display_name),
        "my_line": -p.spread_home.abs().round(1),
        "fav_win_pct": (100 * p.p_home.where(fav_home, 1 - p.p_home)).round(1),
        "proj_score": p.away_pts.round().astype(int).astype(str) + "-" +
                      p.home_pts.round().astype(int).astype(str),
        "proj_tempo": p.tempo.round(1),
        "spread_home": p.spread_home.round(2), "p_home": p.p_home.round(4),
        "vegas_line": None, "torvik_line": None,   # filled in once running on GitHub
        "made_at": pd.Timestamp.now(tz="America/Chicago").isoformat(timespec="minutes"),
    })


def append_log(league: str, lines: pd.DataFrame) -> int:
    """Add lines for games not already logged. Returns how many were added."""
    log = LOGS / f"predictions_{league}.csv"
    old = pd.read_csv(log) if log.exists() else None
    if old is not None:
        lines = lines[~lines.game_id.isin(old.game_id)]
    if not len(lines):
        return 0
    if old is None:
        lines.to_csv(log, index=False)
    elif list(lines.columns) == list(old.columns):
        lines.to_csv(log, mode="a", header=False, index=False)
    else:  # columns changed since the log started: rewrite with the union (old rows untouched)
        cols = list(old.columns) + [c for c in lines.columns if c not in old.columns]
        pd.concat([old, lines], ignore_index=True)[cols].to_csv(log, index=False)
    return len(lines)


SNAP_COLS = ["rank", "net", "O", "D", "tempo", "games"]


def snapshot(league: str, day: pd.Timestamp, r: pd.DataFrame) -> None:
    """Daily ratings history (logs/ratings_history_{lg}.csv) for team pages and trend charts.
    One block of rows per date; a rerun on the same day replaces that day's block."""
    f = LOGS / f"ratings_history_{league}.csv"
    snap = r[SNAP_COLS].round(1).copy()
    snap.insert(0, "team_id", r.index)
    snap.insert(0, "date", str(day.date()))
    if f.exists():
        old = pd.read_csv(f)
        snap = pd.concat([old[old.date != str(day.date())], snap], ignore_index=True)
    snap.sort_values(["date", "rank"]).to_csv(f, index=False)


def run(day: pd.Timestamp, refresh: bool = True, force: bool = False) -> None:
    """A second run on the same day (backup schedule, manual click) keeps the day's lines
    as first made, so the site always matches the log. --force remakes them."""
    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    for league in D.LEAGUES:
        r, lg_info, dials, conf = rate(league, day, refresh)
        r[["rank", "team", "net", "O", "D", "tempo", "games", "preseason", "returning_pct", "home_court"]
          ].round(1).to_csv(OUT / f"ratings_{league}.csv", index_label="team_id")
        snapshot(league, day, r)
        f = OUT / f"lines_{league}_{day.date()}.csv"
        if f.exists() and f.stat().st_size > 1 and not force:
            lines = pd.read_csv(f)
            print(f"{league}: keeping today's lines as first made ({len(lines)} games)")
        else:
            lines = make_lines(league, day, r, lg_info, dials, conf)
            lines.to_csv(f, index=False)
        added = append_log(league, lines) if len(lines) else 0
        print(f"{league}: {len(r)} teams | {len(lines)} lines for {day.date()} "
              f"({added} new to log) | home court {R.hca_points(lg_info):.2f} pts")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(pd.Timestamp(args[0]) if args else pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None),
        refresh="--no-refresh" not in sys.argv, force="--force" in sys.argv)
