"""Export everything the website dashboard (docs/index.html) needs as JSON.

Run after daily.py:  python src/export.py [YYYY-MM-DD]
Output: docs/data/*.json (one file per dashboard dataset; GitHub Pages serves docs/)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
import data as D  # noqa: E402
import odds  # noqa: E402

OUT = D.ROOT / "output"
LOGS = D.ROOT / "logs"
UPSET_DOG = 15.0   # biggest-upsets tab: winners that were Meech underdogs by this many points or more
PICK_EDGE = 3.0  # a "pick" vs. the Vegas number = Meech line differs by 3+ points
# 2025-26 walk-forward check (settings chosen on 2021-22..2024-25; buy-game boost not included,
# and it was later refit with 2025-26 in it). The live 2026-27 record is the real test.
BENCHMARK = {"mbb": {"avg_miss": 9.14, "winner_pct": 72.1}, "wbb": {"avg_miss": 9.65, "winner_pct": 76.6}}


def conferences(league: str) -> pd.Series:
    """team_id -> conference (this season's alignment; last season's for teams not seen yet)."""
    cur, prev = D.conferences(league, D.CURRENT), D.conferences(league, D.CURRENT - 1)
    return pd.concat([cur, prev[~prev.index.isin(cur.index)]])


def tip_times(league: str) -> pd.Series:
    """game_id -> scheduled tip (ISO string with offset)."""
    f = D.RAW / f"{league}_schedule_{D.CURRENT}.csv"
    if not f.exists():
        return pd.Series(dtype=object)
    s = pd.read_csv(f, low_memory=False, usecols=["game_id", "game_date_time"])
    return s.drop_duplicates("game_id").set_index("game_id").game_date_time


def grade(league: str) -> pd.DataFrame:
    """Logged lines joined to final results."""
    log = LOGS / f"predictions_{league}.csv"
    if not log.exists():
        return pd.DataFrame()
    g = D.load_games(league)[["game_id", "home_pts", "away_pts", "margin"]]
    x = pd.read_csv(log).merge(g, on="game_id")
    # only lines made before tip count (one Nov 4 women's line was logged after tip: audit Nov 6)
    if "made_at" in x:
        tip = pd.to_datetime(x.game_id.map(tip_times(league)), utc=True, errors="coerce")
        made = pd.to_datetime(x.made_at, utc=True, errors="coerce")
        x = x[~(made >= tip)]
    x["miss"] = (x.spread_home - x.margin).abs()
    x["correct"] = (x.spread_home >= 0) == (x.margin > 0)
    x["fav_p"] = np.where(x.spread_home >= 0, x.p_home, 1 - x.p_home)
    v = odds.load(league)
    v["is_close"] = v.close_home.notna()
    v["vegas_home"] = v.close_home.fillna(v.vegas_home)   # grade vs. the CLOSING line when we have it
    x = x.merge(v[["game_id", "vegas_home", "is_close"]], on="game_id", how="left")
    x["vegas_line"] = vegas_for_favorite(x.spread_home, x.vegas_home)
    x["vegas_miss"] = (x.vegas_home - x.margin).abs()
    # pick vs. the number: side the Meech line likes, graded against the Vegas spread
    edge = x.spread_home - x.vegas_home
    x["pick"] = edge.abs() >= PICK_EDGE
    cover = np.sign(x.margin - x.vegas_home) * np.sign(edge)  # +1 win, -1 loss, 0 push
    x["pick_result"] = np.select([cover > 0, cover < 0], ["W", "L"], "P")
    return x


def clean_score(proj_score: str, spread_home: float) -> str:
    """Projected score for display, made consistent with the line: whole points, the favorite
    ahead by the rounded line (at least 1). Logged lines from before Nov 7, 2026 rounded each
    team separately (a 0.1 favorite at 62-61 next to a 0.4 favorite at 65-65)."""
    try:
        a, h = (int(x) for x in str(proj_score).split("-"))
    except ValueError:
        return proj_score
    d, tot = max(1, round(abs(spread_home))), a + h
    if (tot + d) % 2:
        tot += 1
    hi, lo = (tot + d) // 2, (tot - d) // 2
    return f"{lo}-{hi}" if spread_home >= 0 else f"{hi}-{lo}"


def conf_vs(league: str) -> pd.DataFrame:
    """This season's record of each conference against each other conference (completed D1 vs.
    D1 games between teams in different conferences; neutral, home and road all count)."""
    f = D.RAW / f"{league}_schedule_{D.CURRENT}.csv"
    if not f.exists():
        return pd.DataFrame()
    s = pd.read_csv(f, low_memory=False, usecols=["game_id", "home_id", "away_id", "home_score", "away_score",
                                                  "status_type_completed"]).drop_duplicates("game_id")
    s = s[s.status_type_completed.fillna(False).astype(bool) & (s.home_score != s.away_score)]
    conf = conferences(league)
    s = s.assign(hc=s.home_id.map(conf), ac=s.away_id.map(conf)).dropna(subset=["hc", "ac"])
    s = s[s.hc != s.ac]
    hw = s.home_score > s.away_score
    rows = pd.concat([pd.DataFrame({"conf": s.hc, "opp": s.ac, "w": hw}),
                      pd.DataFrame({"conf": s.ac, "opp": s.hc, "w": ~hw})])
    out = rows.groupby(["conf", "opp"]).w.agg(w="sum", g="size").reset_index()
    out["l"] = out.g - out.w
    return out[["conf", "opp", "w", "l"]].astype({"w": int, "l": int})


def upsets(league: str) -> pd.DataFrame:
    """Logged games where a Meech underdog of UPSET_DOG+ points won outright (from the winner's side)."""
    x = grade(league)
    if x.empty:
        return pd.DataFrame()
    hd = x.spread_home < 0                                   # home team was the Meech underdog
    won = np.where(hd, x.margin > 0, x.margin < 0)
    x, hd = x[(x.spread_home.abs() >= UPSET_DOG) & won], hd[(x.spread_home.abs() >= UPSET_DOG) & won]
    if x.empty:
        return pd.DataFrame()
    neutral = x.neutral.astype(str).str.lower() == "true"
    return pd.DataFrame({
        "game_id": x.game_id, "date": x.date,
        "winner": np.where(hd, x.home, x.away), "loser": np.where(hd, x.away, x.home),
        "winner_rank": np.where(hd, x.home_rank, x.away_rank), "loser_rank": np.where(hd, x.away_rank, x.home_rank),
        "site": np.where(neutral, "N", np.where(hd, "vs", "at")),     # from the winner's side
        "meech": x.spread_home.abs(), "win_pct": 100 * np.where(hd, x.p_home, 1 - x.p_home),
        "vegas": np.where(hd, -x.vegas_home, x.vegas_home),           # + = Vegas had the winner as an underdog
        "winner_pts": np.where(hd, x.home_pts, x.away_pts).astype(int),
        "loser_pts": np.where(hd, x.away_pts, x.home_pts).astype(int),
        "swing": x.spread_home.abs() + x.margin.abs(),                # points better than the Meech line said
    }).sort_values("date", ascending=False).round(1)


def vegas_for_favorite(spread_home, vegas_home):
    """Vegas spread shown from the Meech favorite's side (e.g. -6.5 = Vegas has them by 6.5)."""
    return np.where(spread_home >= 0, -vegas_home, vegas_home)


def record(league: str) -> dict:
    """Graded games plus summary, monthly and calibration tables (rows)."""
    x = grade(league)
    summary = {"kind": "summary", "league": league, "graded": len(x),
               "avg_miss": round(x.miss.mean(), 2) if len(x) else None,
               "winner_pct": round(100 * x.correct.mean(), 1) if len(x) else None,
               "bench_miss": BENCHMARK[league]["avg_miss"], "bench_winner": BENCHMARK[league]["winner_pct"],
               "vegas_graded": 0, "my_miss_v": None, "vegas_miss": None, "closer_pct": None,
               "pick_w": 0, "pick_l": 0, "pick_p": 0, "pick_pct": None}
    v = x[x.vegas_home.notna()] if len(x) else x
    if len(v):
        picks = v[v.pick].pick_result.value_counts()
        w, l_, p = (int(picks.get(k, 0)) for k in "WLP")
        summary.update(vegas_graded=len(v), vegas_closing=int(v.is_close.fillna(False).sum()),
                       my_miss_v=round(v.miss.mean(), 2),
                       vegas_miss=round(v.vegas_miss.mean(), 2),
                       closer_pct=round(100 * ((v.miss < v.vegas_miss) + 0.5 * (v.miss == v.vegas_miss)).mean(), 1),
                       pick_w=w, pick_l=l_, pick_p=p, pick_pct=round(100 * w / (w + l_), 1) if w + l_ else None)
    if x.empty:
        return {"summary": summary, "games": [], "months": [], "calibration": []}
    x["month"] = pd.to_datetime(x.date).dt.strftime("%Y-%m")
    x["miss_v"] = x.miss.where(x.vegas_home.notna())   # Meech miss on games that had a Vegas line
    months = x.groupby("month").agg(games=("miss", "size"), avg_miss=("miss", "mean"),
                                    winner_pct=("correct", "mean"), vegas_games=("vegas_miss", "count"),
                                    my_miss_v=("miss_v", "mean"), vegas_miss=("vegas_miss", "mean")).reset_index()
    months["winner_pct"] *= 100
    labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    buckets = pd.cut(x.fav_p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=labels, include_lowest=True)
    cal = (x.groupby(buckets, observed=True)
             .agg(games=("correct", "size"), said=("fav_p", "mean"), won=("correct", "mean")))
    cal["bucket"] = cal.index.astype(str)
    cal[["said", "won"]] *= 100
    x["final"] = x.away_pts.astype(int).astype(str) + "-" + x.home_pts.astype(int).astype(str)
    x["winner"] = np.where(x.correct, "Yes", "No")
    has_v = x.vegas_home.notna()
    x["closer"] = np.where(~has_v, None, np.where(x.miss < x.vegas_miss, "Meech",
                           np.where(x.miss > x.vegas_miss, "Vegas", "Tie")))
    games = x.sort_values(["date", "miss"], ascending=[False, True])[[
        "game_id", "date", "away", "home", "favorite", "my_line", "vegas_line", "fav_win_pct", "final",
        "miss", "vegas_miss", "winner", "closer"]]
    tag = lambda df, kind: (df.assign(kind=kind, league=league).round(2)  # noqa: E731
                            .astype(object).where(lambda d: d.notna(), None).to_dict("records"))
    return {"summary": summary, "games": tag(games, "game"), "months": tag(months, "month"),
            "calibration": tag(cal.reset_index(drop=True), "calibration")}


def records(league: str) -> pd.DataFrame:
    """Overall and conference W-L per team (all completed games, any opponent)."""
    s = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                    usecols=["home_id", "away_id", "home_score", "away_score", "status_type_completed",
                             "conference_competition"])
    s = s[s.status_type_completed.fillna(False).astype(bool)]
    conf = s.conference_competition.fillna(False).astype(bool)
    home_won = s.home_score > s.away_score
    rows = pd.concat([pd.DataFrame({"team_id": s.home_id, "w": home_won, "c": conf}),
                      pd.DataFrame({"team_id": s.away_id, "w": ~home_won, "c": conf})])
    out = rows.groupby("team_id").agg(wins=("w", "sum"), games_played=("w", "size"))
    cr = rows[rows.c].groupby("team_id").agg(conf_wins=("w", "sum"), conf_games=("w", "size"))
    out = out.join(cr).fillna(0).astype(int)
    out["losses"] = out.games_played - out.wins
    out["conf_losses"] = out.conf_games - out.conf_wins
    return out[["wins", "losses", "conf_wins", "conf_losses"]]


def league_tables(league: str, day: pd.Timestamp) -> dict:
    conf = conferences(league)
    r = pd.read_csv(OUT / f"ratings_{league}.csv")
    r["conf"] = r.team_id.map(conf).fillna("Ind.")
    r["vs_preseason"] = (r.net - r.preseason).round(1)
    r = r.join(records(league), on="team_id")
    sc = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                     usecols=["home_id", "home_location"]).drop_duplicates("home_id").set_index("home_id")
    r["school"] = r.team_id.map(sc.home_location).fillna(r.team)   # short name ("Michigan") for tight spots
    rc = ["wins", "losses", "conf_wins", "conf_losses"]
    r[rc] = r[rc].fillna(0).astype(int)
    strength = (r.groupby("conf").agg(teams=("net", "size"), avg=("net", "mean"), best=("net", "max"))
                  .query("teams >= 4").sort_values("avg", ascending=False).round(1).reset_index())
    strength.insert(0, "rank", range(1, len(strength) + 1))

    f = OUT / f"lines_{league}_{day.date()}.csv"
    lines = pd.read_csv(f) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()
    if len(lines):
        ids = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                          usecols=["game_id", "home_id", "away_id"]).set_index("game_id")
        lines["home_conf"] = lines.game_id.map(ids.home_id).map(conf).fillna("Ind.")
        lines["away_conf"] = lines.game_id.map(ids.away_id).map(conf).fillna("Ind.")
        vegas = odds.load(league).set_index("game_id")
        lines["vegas_line"] = vegas_for_favorite(lines.spread_home, lines.game_id.map(vegas.vegas_home))
        lines["vegas_total"] = lines.game_id.map(vegas.vegas_total)
        lines["gap"] = (lines.my_line - lines.vegas_line).round(1)
        lines["tip24"] = lines.tip
        lines["tip"] = pd.to_datetime(lines.tip, format="%H:%M").dt.strftime("%-I:%M %p")
        lines["site"] = np.where(lines.neutral, "Neutral", "")
        lines["proj_score"] = [clean_score(s, h) for s, h in zip(lines.proj_score, lines.spread_home)]
        lines = lines.drop(columns=["date", "made_at", "neutral"])
    sc = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                     usecols=["home_display_name", "home_color", "away_display_name", "away_color"])
    colors = pd.concat([sc[["home_display_name", "home_color"]].set_axis(["team", "color"], axis=1),
                        sc[["away_display_name", "away_color"]].set_axis(["team", "color"], axis=1)])
    colors = colors.dropna().drop_duplicates("team")
    colors["color"] = "#" + colors.color.astype(str).str.lstrip("#")
    rec = record(league)  # one file: summary rows always exist, so it never comes up empty
    out = {"lines": lines, "rankings": r, "conferences": strength, "colors": colors,
           "upsets": upsets(league), "conf_vs": conf_vs(league), "pace": pace_table(league), "pace_history": pace_history(league), "homecourt": homecourt(league),
           "record": [rec["summary"]] + rec["months"] + rec["calibration"] + rec["games"]}
    return {k: (v if isinstance(v, list) else v.assign(league=league).replace({np.nan: None})
                .to_dict("records")) for k, v in out.items()}


def pace_table(league: str) -> pd.DataFrame:
    f = OUT / f"pace_{league}.csv"
    if not f.exists():
        return pd.DataFrame()
    p = pd.read_csv(f)
    p[["battles", "games"]] = p[["battles", "games"]].fillna(0).astype(int)
    return p.round(1)


def pace_history(league: str, seasons_needed: int = 4) -> pd.DataFrame:
    """Teams that dictated (or adapted to) the pace in at least `seasons_needed` of the last five
    seasons (output/pace_profiles.csv, from analysis/pace_control.py)."""
    f = OUT / "pace_profiles.csv"
    if not f.exists():
        return pd.DataFrame()
    p = pd.read_csv(f)
    p = p[p.league == league]
    g = p.groupby("team").agg(seasons=("season", "size"), above=("control", lambda c: int((c > 50).sum())),
                              avg=("control", "mean"))
    g["team_id"] = g.index
    dic = g[g.above >= seasons_needed].assign(kind="dictator").sort_values("avg", ascending=False)
    cham = g[(g.seasons - g.above) >= seasons_needed].assign(kind="chameleon").sort_values("avg")
    # every qualifying team (the site shows the top 12 after its conference / search filters)
    return pd.concat([dic, cham])[["team_id", "kind", "seasons", "above", "avg"]].round(1)


def homecourt(league: str) -> pd.DataFrame:
    """Home court page: each team's home venue, how many home games its home edge was measured
    on (last 4 full seasons, COVID year skipped, as in ratings.team_hca), arena capacity and
    elevation (data/arenas/arenas.csv, from scripts/fetch_arenas.py), this season's home/road record."""
    g = D.d1_games(D.load_games(league))
    used = g[(g.season >= max(2022, D.CURRENT - 4)) & (g.season < D.CURRENT) & ~g.neutral]
    out = pd.DataFrame({"home_games": used.groupby("home_id").size()})
    cur = g[(g.season == D.CURRENT) & ~g.neutral]
    out["home_w"] = cur[cur.margin > 0].groupby("home_id").size()
    out["home_l"] = cur[cur.margin < 0].groupby("home_id").size()
    out["road_w"] = cur[cur.margin < 0].groupby("away_id").size()
    out["road_l"] = cur[cur.margin > 0].groupby("away_id").size()
    out = out.fillna(0).astype(int)
    v = venues(league)
    out = out.join(v, how="outer")
    out = out[out.index.isin(pd.read_csv(OUT / f"ratings_{league}.csv").team_id)]
    f = D.ROOT / "data" / "arenas" / "arenas.csv"
    if f.exists():
        a = pd.read_csv(f)
        a = a[a.league == league].set_index("team_id")[["capacity", "elevation_ft"]]
        out = out.join(a)
    out.index.name = "team_id"
    return out.reset_index()


def venues(league: str) -> pd.DataFrame:
    """Each team's usual home venue: the most common site of its non-neutral home games,
    this season and last."""
    rows = []
    for season in (D.CURRENT - 1, D.CURRENT):
        f = D.RAW / f"{league}_schedule_{season}.csv"
        if f.exists():
            rows.append(pd.read_csv(f, low_memory=False, usecols=[
                "home_id", "neutral_site", "venue_full_name", "venue_address_city", "venue_address_state"]))
    s = pd.concat(rows)
    s = s[~s.neutral_site.fillna(False).astype(bool)].dropna(subset=["venue_full_name"])
    top = (s.groupby(["home_id", "venue_full_name", "venue_address_city", "venue_address_state"], dropna=False)
             .size().rename("n").reset_index().sort_values("n", ascending=False).drop_duplicates("home_id"))
    return (top.set_index("home_id")[["venue_full_name", "venue_address_city", "venue_address_state"]]
               .set_axis(["venue", "city", "state"], axis=1))


def team_pages(league: str, day: pd.Timestamp) -> dict:
    """Per-team schedule (results, logged or projected Meech line, win chance) + stat leaders,
    keyed by team_id. Loaded by the site only when a team page is opened."""
    r = pd.read_csv(OUT / f"ratings_{league}.csv").set_index("team_id")
    s = pd.read_csv(D.RAW / f"{league}_schedule_{D.CURRENT}.csv", low_memory=False,
                    usecols=["game_id", "game_date", "home_id", "away_id", "home_display_name",
                             "away_display_name", "home_score", "away_score", "neutral_site",
                             "status_type_completed", "conference_competition"])
    s["done"] = s.status_type_completed.fillna(False).astype(bool)
    s["neutral"] = s.neutral_site.fillna(False).astype(bool)
    spread, prob = pd.Series(dtype=float), pd.Series(dtype=float)
    f = OUT / f"projections_{league}.csv"
    if f.exists():
        pj = pd.read_csv(f).set_index("game_id")
        spread, prob = pj.spread_home, pj.p_home
    log = LOGS / f"predictions_{league}.csv"
    if log.exists():   # past games: the line we actually logged before tip
        lg = pd.read_csv(log).drop_duplicates("game_id").set_index("game_id")
        spread = pd.concat([lg.spread_home, spread[~spread.index.isin(lg.index)]])
        prob = pd.concat([lg.p_home, prob[~prob.index.isin(lg.index)]])
    s["sp"], s["ph"] = s.game_id.map(spread), s.game_id.map(prob)
    pages = {int(t): {"g": [], "l": [], "h": []} for t in r.index}
    hist = LOGS / f"ratings_history_{league}.csv"
    if hist.exists():   # rating trend chart: one point per day
        h = pd.read_csv(hist)
        for t, grp in h[h.team_id.isin(r.index)].groupby("team_id"):
            pages[int(t)]["h"] = grp[["date", "net", "rank"]].values.tolist()
    for x in s.sort_values("game_date").itertuples():
        for home in (True, False):
            me, opp = (x.home_id, x.away_id) if home else (x.away_id, x.home_id)
            if me not in pages:
                continue
            row = {"d": str(x.game_date)[:10], "s": "N" if x.neutral else ("vs" if home else "at"),
                   "c": bool(x.conference_competition) if pd.notna(x.conference_competition) else False}
            if opp in r.index:
                row["o"] = int(opp)
            else:
                row["on"] = x.away_display_name if home else x.home_display_name
            if pd.notna(x.sp):
                row["ln"] = round(float(x.sp if home else -x.sp), 1)
                row["p"] = round(float(x.ph if home else 1 - x.ph), 3)
            if x.done:
                us, them = (x.home_score, x.away_score) if home else (x.away_score, x.home_score)
                row["r"] = "W" if us > them else "L"
                row["sc"] = f"{int(us)}-{int(them)}"
            pages[int(me)]["g"].append(row)
    ps_file = D.RAW / f"{league}_pstats_{D.CURRENT}.csv"
    if ps_file.exists():
        ps = pd.read_csv(ps_file)
        ps = ps[ps.team_id.isin(r.index) & (ps.games > 0)]
        ps = ps[ps.games >= ps.groupby("team_id").games.transform("max") / 2]   # regulars only
        for c, cols in {"pts": ["points"], "reb": ["offensive_rebounds", "defensive_rebounds"],
                        "ast": ["assists"], "min": ["minutes"]}.items():
            ps[c] = (ps[cols].sum(axis=1) / ps.games).round(1)
        for t, grp in ps.sort_values("pts", ascending=False).groupby("team_id"):
            pages[int(t)]["l"] = grp.head(8)[["name", "games", "pts", "reb", "ast", "min"]].to_dict("records")
    return pages


def export(day: pd.Timestamp) -> Path:
    """One JSON file per dashboard dataset (both leagues, with a league column)."""
    folder = D.ROOT / "docs" / "data"
    folder.mkdir(parents=True, exist_ok=True)
    tables = {}
    for lg in D.LEAGUES:
        for name, rows in league_tables(lg, day).items():
            tables.setdefault(name, []).extend(rows)
    tables["info"] = [{"model": "The Meech Model", "date": str(day.date()),
                       "date_label": day.strftime("%a, %b %-d"),
                       "updated": pd.Timestamp.now(tz="America/Chicago").strftime("%b %-d, %-I:%M %p CT")}]
    for name, rows in tables.items():
        (folder / f"{name}.json").write_text(json.dumps(rows, default=str))
    (folder / "model.json").write_text(json.dumps(
        {lg: json.loads((OUT / f"model_{lg}.json").read_text()) for lg in D.LEAGUES
         if (OUT / f"model_{lg}.json").exists()}, separators=(",", ":")))
    for lg in D.LEAGUES:
        (folder / f"teams_{lg}.json").write_text(json.dumps(team_pages(lg, day), default=str,
                                                            separators=(",", ":")))
    return folder


def write_venues() -> None:
    """data/arenas/venues.csv: input for scripts/fetch_arenas.py (run once a year)."""
    rows = []
    for lg in D.LEAGUES:
        v = venues(lg)
        v = v[v.index.isin(pd.read_csv(OUT / f"ratings_{lg}.csv").team_id)]
        sc = (pd.read_csv(D.RAW / f"{lg}_schedule_{D.CURRENT}.csv", low_memory=False,
                          usecols=["home_id", "home_location", "home_display_name"])
                .drop_duplicates("home_id").set_index("home_id"))
        rows.append(v.assign(school=sc.home_location.reindex(v.index), team=sc.home_display_name.reindex(v.index),
                             league=lg).rename_axis("team_id").reset_index())
    (D.ROOT / "data" / "arenas").mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(D.ROOT / "data" / "arenas" / "venues.csv", index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--venues" in sys.argv:
        write_venues()
        sys.exit()
    folder = export(pd.Timestamp(args[0]) if args else pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None))
    for f in sorted(folder.glob("*.json")):
        print(f"{f.name:20s} {f.stat().st_size / 1024:6.1f} KB")
