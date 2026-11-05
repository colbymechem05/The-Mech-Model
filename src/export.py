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
PICK_EDGE = 3.0  # a "pick" vs. the Vegas number = Mech line differs by 3+ points
# 2025-26 holdout (never used for tuning): the benchmark the live record is compared to
BENCHMARK = {"mbb": {"avg_miss": 9.15, "winner_pct": 72.0}, "wbb": {"avg_miss": 9.65, "winner_pct": 76.6}}


def conferences(league: str) -> pd.Series:
    """team_id -> conference short name (this season's alignment)."""
    rows = []
    for s in (D.CURRENT, D.CURRENT - 1):  # current season first, last season as fallback
        D.download(league, s, refresh=False)  # no-op when already saved
        sc = pd.read_csv(D.RAW / f"{league}_schedule_{s}.csv", low_memory=False, usecols=[
            "home_id", "home_conference_id", "away_id", "away_conference_id",
            "groups_id", "groups_short_name", "conference_competition"])
        names = (sc[sc.conference_competition == True]  # noqa: E712
                 .drop_duplicates("groups_id").set_index("groups_id").groups_short_name)
        for side in ("home", "away"):
            t = sc[[f"{side}_id", f"{side}_conference_id"]].dropna().drop_duplicates(f"{side}_id")
            rows.append(pd.Series(t[f"{side}_conference_id"].map(names).values, index=t[f"{side}_id"]))
    conf = pd.concat(rows).dropna()
    return conf[~conf.index.duplicated(keep="first")]


def grade(league: str) -> pd.DataFrame:
    """Logged lines joined to final results."""
    log = LOGS / f"predictions_{league}.csv"
    if not log.exists():
        return pd.DataFrame()
    g = D.load_games(league)[["game_id", "home_pts", "away_pts", "margin"]]
    x = pd.read_csv(log).merge(g, on="game_id")
    x["miss"] = (x.spread_home - x.margin).abs()
    x["correct"] = (x.spread_home >= 0) == (x.margin > 0)
    x["fav_p"] = np.where(x.spread_home >= 0, x.p_home, 1 - x.p_home)
    x = x.merge(odds.load(league)[["game_id", "vegas_home"]], on="game_id", how="left")
    x["vegas_line"] = vegas_for_favorite(x.spread_home, x.vegas_home)
    x["vegas_miss"] = (x.vegas_home - x.margin).abs()
    # pick vs. the number: side the Mech line likes, graded against the Vegas spread
    edge = x.spread_home - x.vegas_home
    x["pick"] = edge.abs() >= PICK_EDGE
    cover = np.sign(x.margin - x.vegas_home) * np.sign(edge)  # +1 win, -1 loss, 0 push
    x["pick_result"] = np.select([cover > 0, cover < 0], ["W", "L"], "P")
    return x


def vegas_for_favorite(spread_home, vegas_home):
    """Vegas spread shown from the Mech favorite's side (e.g. -6.5 = Vegas has them by 6.5)."""
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
        summary.update(vegas_graded=len(v), my_miss_v=round(v.miss.mean(), 2),
                       vegas_miss=round(v.vegas_miss.mean(), 2),
                       closer_pct=round(100 * ((v.miss < v.vegas_miss) + 0.5 * (v.miss == v.vegas_miss)).mean(), 1),
                       pick_w=w, pick_l=l_, pick_p=p, pick_pct=round(100 * w / (w + l_), 1) if w + l_ else None)
    if x.empty:
        return {"summary": summary, "games": [], "months": [], "calibration": []}
    x["month"] = pd.to_datetime(x.date).dt.strftime("%Y-%m")
    months = x.groupby("month").agg(games=("miss", "size"), avg_miss=("miss", "mean"),
                                    winner_pct=("correct", "mean"), vegas_miss=("vegas_miss", "mean")).reset_index()
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
    x["closer"] = np.where(~has_v, None, np.where(x.miss < x.vegas_miss, "Mech",
                           np.where(x.miss > x.vegas_miss, "Vegas", "Tie")))
    x["pick_result"] = x.pick_result.where(has_v & x.pick)
    games = x.sort_values(["date", "miss"], ascending=[False, True])[[
        "game_id", "date", "away", "home", "favorite", "my_line", "vegas_line", "fav_win_pct", "final",
        "miss", "vegas_miss", "winner", "closer", "pick_result"]]
    tag = lambda df, kind: (df.assign(kind=kind, league=league).round(2)  # noqa: E731
                            .astype(object).where(lambda d: d.notna(), None).to_dict("records"))
    return {"summary": summary, "games": tag(games, "game"), "months": tag(months, "month"),
            "calibration": tag(cal.reset_index(drop=True), "calibration")}


def league_tables(league: str, day: pd.Timestamp) -> dict:
    conf = conferences(league)
    r = pd.read_csv(OUT / f"ratings_{league}.csv")
    r["conf"] = r.team_id.map(conf).fillna("Ind.")
    r["vs_preseason"] = (r.net - r.preseason).round(1)
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
        lines = lines.drop(columns=["date", "made_at", "neutral"])
    rec = record(league)  # one file: summary rows always exist, so it never comes up empty
    out = {"lines": lines, "rankings": r, "conferences": strength,
           "record": [rec["summary"]] + rec["months"] + rec["calibration"] + rec["games"]}
    return {k: (v if isinstance(v, list) else v.assign(league=league).replace({np.nan: None})
                .to_dict("records")) for k, v in out.items()}


def export(day: pd.Timestamp) -> Path:
    """One JSON file per dashboard dataset (both leagues, with a league column)."""
    folder = D.ROOT / "docs" / "data"
    folder.mkdir(parents=True, exist_ok=True)
    tables = {}
    for lg in D.LEAGUES:
        for name, rows in league_tables(lg, day).items():
            tables.setdefault(name, []).extend(rows)
    tables["info"] = [{"model": "The Mech Model", "date": str(day.date()),
                       "date_label": day.strftime("%a, %b %-d"),
                       "updated": pd.Timestamp.now(tz="America/Chicago").strftime("%b %-d, %-I:%M %p CT")}]
    for name, rows in tables.items():
        (folder / f"{name}.json").write_text(json.dumps(rows, default=str))
    return folder


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = export(pd.Timestamp(args[0]) if args else pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None))
    for f in sorted(folder.glob("*.json")):
        print(f"{f.name:20s} {f.stat().st_size / 1024:6.1f} KB")
