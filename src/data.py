"""Game data: download ESPN box scores (via sportsdataverse) and build one clean
games table per league. Season 2027 = the 2026-27 season.

Output: data/games_{league}.csv, one row per game (home perspective).
"""
from pathlib import Path
import time
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
LEAGUES = {"mbb": ("espn_mens_college_basketball", 2),    # regulation: 2 halves
           "wbb": ("espn_womens_college_basketball", 4)}  # 4 quarters
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026, 2027]
CURRENT = SEASONS[-1]
FT_FACTOR = 0.475  # share of free throw attempts that end a possession
COLUMNS = ["game_id", "season", "date", "home_id", "away_id", "home", "away",
           "home_pts", "away_pts", "margin", "poss", "minutes", "neutral"]


def fetch(remote: str, path: Path, refresh: bool = False, tries: int = 5) -> Path:
    """Download a sportsdataverse release file (cached unless refresh). Retries with
    growing waits (GitHub sometimes answers 504), and writes to a temp file first so a
    failed download never damages the cached copy."""
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(f"{BASE}/{remote}", tmp)
            tmp.replace(path)
            return path
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == tries - 1:
                raise
            time.sleep([5, 15, 30, 60][min(attempt, 3)])
    return path


def download(league: str, season: int, refresh: bool = False) -> None:
    prefix, _ = LEAGUES[league]
    fetch(f"{prefix}_team_boxscores/team_box_{season}.csv",
          RAW / f"{league}_team_box_{season}.csv", refresh)
    fetch(f"{prefix}_schedules/{league}_schedule_{season}.csv",
          RAW / f"{league}_schedule_{season}.csv", refresh)


def possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + FT_FACTOR * fta


def load_season(league: str, season: int) -> pd.DataFrame:
    _, reg_periods = LEAGUES[league]
    box = pd.read_csv(RAW / f"{league}_team_box_{season}.csv", low_memory=False)
    sched = pd.read_csv(RAW / f"{league}_schedule_{season}.csv", low_memory=False,
                        usecols=["game_id", "neutral_site", "status_period"])
    box["poss"] = possessions(box.field_goals_attempted, box.offensive_rebounds,
                              box.turnovers, box.free_throws_attempted)
    keep = ["game_id", "team_id", "team_display_name", "team_score", "poss"]
    home = box[box.team_home_away == "home"][keep + ["game_date"]]
    away = box[box.team_home_away == "away"][keep]
    g = home.merge(away, on="game_id", suffixes=("_h", "_a")).merge(sched, on="game_id", how="left")
    g = g.rename(columns={"team_id_h": "home_id", "team_id_a": "away_id",
                          "team_display_name_h": "home", "team_display_name_a": "away",
                          "team_score_h": "home_pts", "team_score_a": "away_pts"})
    g["season"] = season
    g["date"] = pd.to_datetime(g.game_date)          # ESPN local (Eastern) date
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    overtimes = (g.status_period.fillna(reg_periods) - reg_periods).clip(lower=0)
    g["minutes"] = 40 + 5 * overtimes
    g["poss"] = (g.poss_h + g.poss_a) / 2            # game possessions
    g["margin"] = g.home_pts - g.away_pts
    g = g[(g.home_pts + g.away_pts > 0) & (g.poss_h > 30) & (g.poss_a > 30)]
    return g[COLUMNS]


def d1_teams(league: str) -> set:
    """Division I = teams ESPN lists in a conference (D2/NAIA schools have none)."""
    ids = set()
    for s in SEASONS:
        f = RAW / f"{league}_schedule_{s}.csv"
        if f.exists():
            sc = pd.read_csv(f, low_memory=False, usecols=[
                "home_id", "home_conference_id", "away_id", "away_conference_id"])
            ids |= set(sc.loc[sc.home_conference_id.notna(), "home_id"])
            ids |= set(sc.loc[sc.away_conference_id.notna(), "away_id"])
    return ids


def build(league: str, refresh_current: bool = True) -> pd.DataFrame:
    for s in SEASONS:
        download(league, s, refresh=refresh_current and s == CURRENT)
    games = pd.concat([load_season(league, s) for s in SEASONS], ignore_index=True)
    d1 = d1_teams(league)
    games["home_d1"] = games.home_id.isin(d1)
    games["away_d1"] = games.away_id.isin(d1)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games.to_csv(ROOT / "data" / f"games_{league}.csv", index=False)
    return games


def conferences(league: str, season: int) -> pd.Series:
    """team_id -> conference short name for a season (from that season's schedule)."""
    download(league, season, refresh=False)
    sc = pd.read_csv(RAW / f"{league}_schedule_{season}.csv", low_memory=False, usecols=[
        "home_id", "home_conference_id", "away_id", "away_conference_id",
        "groups_id", "groups_short_name", "conference_competition"])
    names = (sc[sc.conference_competition == True]  # noqa: E712
             .drop_duplicates("groups_id").set_index("groups_id").groups_short_name)
    ids = pd.concat([sc.set_index("home_id").home_conference_id, sc.set_index("away_id").away_conference_id])
    ids = ids[ids.notna() & ~ids.index.duplicated()]
    return ids.map(names).dropna().rename("conf")


def load_games(league: str) -> pd.DataFrame:
    """Read the games table built by build() (builds it from saved raw files if missing)."""
    f = ROOT / "data" / f"games_{league}.csv"
    if not f.exists():
        build(league, refresh_current=False)
    return pd.read_csv(f, parse_dates=["date"])


def d1_games(games: pd.DataFrame) -> pd.DataFrame:
    return games[games.home_d1 & games.away_d1]


def team_names(games: pd.DataFrame) -> pd.Series:
    n = pd.concat([games[["home_id", "home"]].set_axis(["id", "team"], axis=1),
                   games[["away_id", "away"]].set_axis(["id", "team"], axis=1)])
    return n.drop_duplicates("id", keep="last").set_index("id").team


if __name__ == "__main__":
    for lg in LEAGUES:
        g = build(lg, refresh_current=False)
        d1 = d1_games(g)
        print(f"{lg}: {len(g)} games ({len(d1)} D1 vs D1) | {g.date.min().date()} to "
              f"{g.date.max().date()} | avg poss {d1.poss.mean():.1f}")


def default_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The slate a run without a date works on. Before 6 AM Central it's still last night's slate:
    the new day's lines wait for the morning run, after last night's results are in the data
    (a manual run at 12:06 AM on Nov 7, 2026 logged that day's lines without Nov 6's games)."""
    now = pd.Timestamp.now(tz="America/Chicago") if now is None else now
    day = now.normalize().tz_localize(None)
    return day - pd.Timedelta(days=1) if now.hour < 6 else day
