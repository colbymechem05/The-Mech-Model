"""Daily run: refresh data, rate every D1 team, make lines for a date's games,
and append them to a permanent log (saved before tip-off; never overwritten).

Usage:  python src/daily.py [YYYY-MM-DD] [--no-refresh] [--force]
        (default date: today; before 6 AM Central, still last night's slate: data.default_day)
Outputs per league (mbb, wbb):
  output/ratings_{lg}.csv          power ratings, all D1
  output/lines_{lg}_{date}.csv     that day's lines
  output/projections_{lg}.csv      every remaining game projected from today's ratings (team pages)
  logs/predictions_{lg}.csv        every line ever made (append-only)
  logs/ratings_history_{lg}.csv    one ratings snapshot per day
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
import data as D  # noqa: E402
import pace as PACE  # noqa: E402
import players as P  # noqa: E402
import ratings as R  # noqa: E402
import rosters as RS  # noqa: E402
import talent as T  # noqa: E402

OUT = D.ROOT / "output"
LOGS = D.ROOT / "logs"
HISTORY = [s for s in D.SEASONS if s < D.CURRENT]
# Tuned on 2021-22..2024-25, checked on 2025-26 (see build log; analysis/differentiators.py).
SETTINGS = {"mbb": dict(new_value=-4.0, prior_games=6, recruits="feature"),   # top-100 freshmen (CBBD 247 composite)
            "wbb": dict(new_value=-4.0, prior_games=4, recruits="feature")}  # avg of ESPN/On3/247 (2025+)
CONF_GAMES = 30   # in-season conference pooling (analysis/buy_game_fix.py)
# Buy games: Power team at home vs. a non-Power visitor in Nov-Dec. Extra points for the home
# team = a + b x our spread (analysis/buy_game_term.py, fitted on 2021-22..2025-26).
POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Big East", "Pac-12"}
BUY_BOOST = {"mbb": (2.33, 0.0), "wbb": (0.61, 0.08)}   # note: refit incl. 2025-26 (after the holdout check)


def NOW() -> pd.Timestamp:   # patched in tests
    return pd.Timestamp.now(tz="UTC")


def buy_boost(league: str, day: pd.Timestamp, home_ids, away_ids, neutral, spread, conf) -> np.ndarray:
    """Points added to the home side in buy games (0 everywhere else)."""
    if day.month not in (11, 12):
        return np.zeros(len(spread))
    a, b = BUY_BOOST[league]
    buy = (pd.Series(list(home_ids)).map(conf).isin(POWER).values
           & ~pd.Series(list(away_ids)).map(conf).isin(POWER).values & ~np.asarray(neutral, dtype=bool))
    return np.where(buy, np.maximum(a + b * np.asarray(spread), 0.0), 0.0)


def todays_games(league: str, day: pd.Timestamp) -> pd.DataFrame:
    s = D.schedule(league, cols=["game_id", "game_date", "game_date_time", "home_id", "away_id",
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


def score_text(home_pts, away_pts) -> list:
    """"away-home" in whole points whose difference equals the rounded spread (so a 12.0 line
    never shows an 11-point score). The favorite always wins the projection by at least 1, so a
    0.1 and a 0.4 favorite both show a 1-point win instead of one win and one tie."""
    out = []
    for h, a in zip(home_pts, away_pts):
        d, tot = max(1, round(abs(h - a))), round(h + a)
        if (tot + d) % 2:
            tot += 1 if h + a > tot else -1
        hi, lo = (tot + d) // 2, (tot - d) // 2
        out.append(f"{lo}-{hi}" if h >= a else f"{hi}-{lo}")
    return out


def make_lines(league: str, day: pd.Timestamp, r: pd.DataFrame, lg_info: dict,
               dials: R.Dials, conf: pd.Series | None = None) -> pd.DataFrame:
    g = todays_games(league, day)
    g = g[g.home_id.isin(r.index) & g.away_id.isin(r.index)]   # D1 vs D1 only
    # never make a line for a game that has already tipped (a late run would otherwise log
    # lines for games in progress; audit Nov 6)
    tip = pd.to_datetime(g.game_date_time, utc=True, errors="coerce")
    g = g[tip.isna() | (tip > NOW())]
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
        "proj_score": score_text(p.home_pts.values, p.away_pts.values),
        "proj_tempo": p.tempo.round(1),
        "spread_home": p.spread_home.round(2), "p_home": p.p_home.round(4),
        "made_at": pd.Timestamp.now(tz="America/Chicago").isoformat(timespec="minutes"),
    })


def project_schedule(league: str, day: pd.Timestamp, r: pd.DataFrame, lg_info: dict,
                     dials: R.Dials, conf: pd.Series | None = None) -> pd.DataFrame:
    """Every remaining D1-vs-D1 game this season (today on), rated as the teams stand today.
    Same math as the daily lines, including each date's buy-game boost."""
    s = D.schedule(league, cols=["game_id", "game_date", "home_id", "away_id", "neutral_site", "status_type_completed"])
    s["date"] = pd.to_datetime(s.game_date)
    g = s[(s.date >= day) & ~s.status_type_completed.fillna(False).astype(bool)
          & s.home_id.isin(r.index) & s.away_id.isin(r.index)].reset_index(drop=True)
    if g.empty:
        return pd.DataFrame(columns=["game_id", "spread_home", "p_home"])
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    p = R.predict(r, lg_info, g.home_id, g.away_id, g.neutral, dials.sigma[league])
    spread = p.spread_home.values.copy()
    if conf is not None:
        for d, idx in g.groupby("date").groups.items():
            idx = list(idx)
            spread[idx] += buy_boost(league, d, g.home_id[idx], g.away_id[idx], g.neutral[idx],
                                     spread[idx], conf)
    return pd.DataFrame({"game_id": g.game_id, "spread_home": spread.round(2),
                         "p_home": norm.cdf(spread / dials.sigma[league]).round(4)})


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


def model_params(league: str, r: pd.DataFrame, lg_info: dict, dials: R.Dials) -> dict:
    """Everything the website's matchup tool needs to rebuild a Meech line (same math as
    make_lines): league constants + per-team O, D, T and home edge, per 100 possessions."""
    a, b = BUY_BOOST[league]
    t = pd.DataFrame({"O": r.O, "D": r.D, "T": r["T"],
                      "hc": r.home_court * 100 / lg_info["avg_tempo"]}).round(3)
    return {"avg_eff": round(lg_info["avg_eff"], 3), "avg_tempo": round(lg_info["avg_tempo"], 3),
            "sigma": dials.sigma[league], "buy": [a, b], "power": sorted(POWER),
            "teams": {int(k): v for k, v in t.T.to_dict(orient="list").items()}}


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


def canonical(league: str, lines: pd.DataFrame) -> pd.DataFrame:
    """The permanent log wins: any game already logged is published exactly as logged."""
    log = LOGS / f"predictions_{league}.csv"
    if not len(lines) or not log.exists():
        return lines
    old = pd.read_csv(log)
    old = old[old.game_id.isin(lines.game_id)]
    if old.empty:
        return lines
    keep = lines[~lines.game_id.isin(old.game_id)]
    return pd.concat([old[[c for c in lines.columns if c in old.columns]], keep],
                     ignore_index=True).sort_values("tip", kind="stable").reset_index(drop=True)


def run(day: pd.Timestamp, refresh: bool = True, force: bool = False) -> None:
    """A second run on the same day (backup schedule, manual click) keeps the day's lines
    as first made, so the site always matches the log. --force remakes the file, but only
    games not yet logged can change: logged games are always published as logged."""
    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    for league in D.LEAGUES:
        r, lg_info, dials, conf = rate(league, day, refresh)
        r[["rank", "team", "net", "O", "D", "tempo", "games", "preseason", "returning_pct", "home_court"]
          ].round(1).to_csv(OUT / f"ratings_{league}.csv", index_label="team_id")
        snapshot(league, day, r)
        project_schedule(league, day, r, lg_info, dials, conf).to_csv(OUT / f"projections_{league}.csv", index=False)
        PACE.profile(league, day, r, lg_info).round(2).to_csv(OUT / f"pace_{league}.csv", index_label="team_id")
        (OUT / f"model_{league}.json").write_text(json.dumps(model_params(league, r, lg_info, dials)))
        f = OUT / f"lines_{league}_{day.date()}.csv"
        if f.exists() and f.stat().st_size > 1 and not force:
            lines = pd.read_csv(f)
            print(f"{league}: keeping today's lines as first made ({len(lines)} games)")
        else:
            lines = canonical(league, make_lines(league, day, r, lg_info, dials, conf))
            lines.to_csv(f, index=False)
        added = append_log(league, lines) if len(lines) else 0
        print(f"{league}: {len(r)} teams | {len(lines)} lines for {day.date()} "
              f"({added} new to log) | home court {R.hca_points(lg_info):.2f} pts")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(pd.Timestamp(args[0]) if args else D.default_day(),
        refresh="--no-refresh" not in sys.argv, force="--force" in sys.argv)
