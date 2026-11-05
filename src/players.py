"""Player season stats: one row per player per team per season (every player
listed in a box score, including did-not-play rows, so it doubles as a roster).

Raw ESPN player box files are ~100 MB per season; only this summary is kept:
data/raw/{league}_pstats_{season}.csv
"""
import pandas as pd

from data import LEAGUES, RAW, fetch

STATS = ["minutes", "points", "field_goals_made", "field_goals_attempted",
         "free_throws_made", "free_throws_attempted", "offensive_rebounds",
         "defensive_rebounds", "assists", "steals", "blocks", "turnovers", "fouls"]


def season_stats(league: str, season: int, refresh: bool = False) -> pd.DataFrame:
    small = RAW / f"{league}_pstats_{season}.csv"
    if small.exists() and not refresh:
        return pd.read_csv(small)
    prefix, _ = LEAGUES[league]
    big = fetch(f"{prefix}_player_boxscores/player_box_{season}.csv",
                RAW / f"{league}_player_box_{season}.csv", refresh=True)
    p = pd.read_csv(big, usecols=["athlete_id", "athlete_display_name", "team_id"] + STATS,
                    low_memory=False)
    p[STATS] = p[STATS].apply(pd.to_numeric, errors="coerce").fillna(0)
    p["games"] = (p.minutes > 0).astype(int)
    t = (p.groupby(["athlete_id", "team_id"], as_index=False)
          .agg({**{c: "sum" for c in STATS + ["games"]}, "athlete_display_name": "last"})
          .rename(columns={"athlete_display_name": "name"}))
    t["season"] = season
    t.to_csv(small, index=False)
    big.unlink()
    return t
