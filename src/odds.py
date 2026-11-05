"""Vegas lines from ESPN's public scoreboard (odds come from ESPN's sportsbook partner).

Run:  python src/odds.py [YYYY-MM-DD]      (default: today, Central time)
Keeps logs/vegas_{lg}.csv: the latest pre-tip line for every game (one row per game).
A game's row is only updated while the game hasn't started, so the saved number is
the last line seen before tip-off.

ESPN is reachable from GitHub's servers (not from Claude's workspace).
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run as a script
import data as D  # noqa: E402

LOGS = D.ROOT / "logs"
SPORT = {"mbb": "mens-college-basketball", "wbb": "womens-college-basketball"}
URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/{sport}/scoreboard"
       "?dates={ymd}&groups=50&limit=500")
COLUMNS = ["game_id", "date", "vegas_home", "vegas_total", "provider", "fetched_at"]


def get_json(url: str, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "the-meech-model"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def home_margin(odds: dict, home_abbr: str, away_abbr: str):
    """Vegas expected home margin (+ = home favored), or None if no usable line.

    ESPN's `details` reads like "DUKE -6.5" or "EVEN"; the favorite flags back it up.
    """
    details = str(odds.get("details") or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"^(.*\S)\s+([+-]?\d+(?:\.\d+)?)$", details)
    if m:
        abbr, line = m.group(1).upper(), abs(float(m.group(2)))
        if abbr == str(home_abbr).upper():
            return line
        if abbr == str(away_abbr).upper():
            return -line
    spread = odds.get("spread")
    if spread is None:
        return None
    if (odds.get("homeTeamOdds") or {}).get("favorite"):
        return abs(float(spread))
    if (odds.get("awayTeamOdds") or {}).get("favorite"):
        return -abs(float(spread))
    return None


def parse(board: dict, day: pd.Timestamp) -> pd.DataFrame:
    """Pre-game lines from one ESPN scoreboard response."""
    rows, now = [], pd.Timestamp.now(tz="America/Chicago").isoformat(timespec="minutes")
    for ev in board.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if (comp.get("status") or ev.get("status") or {}).get("type", {}).get("state") != "pre":
            continue  # started or final: keep whatever was saved before tip
        odds = (comp.get("odds") or [None])[0]
        if not odds:
            continue
        side = {c.get("homeAway"): c.get("team", {}) for c in comp.get("competitors", [])}
        if "home" not in side or "away" not in side:
            continue
        line = home_margin(odds, side["home"].get("abbreviation"), side["away"].get("abbreviation"))
        if line is None:
            continue
        rows.append({"game_id": int(ev["id"]), "date": str(day.date()), "vegas_home": line,
                     "vegas_total": odds.get("overUnder"),
                     "provider": (odds.get("provider") or {}).get("name"), "fetched_at": now})
    return pd.DataFrame(rows, columns=COLUMNS)


def save(league: str, new: pd.DataFrame) -> Path:
    """Upsert into the Vegas log: newest pre-tip line wins."""
    LOGS.mkdir(exist_ok=True)
    f = LOGS / f"vegas_{league}.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLUMNS)
    both = pd.concat([old[~old.game_id.isin(new.game_id)], new], ignore_index=True)
    both.sort_values(["date", "game_id"]).to_csv(f, index=False)
    return f


def load(league: str) -> pd.DataFrame:
    f = LOGS / f"vegas_{league}.csv"
    v = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLUMNS)
    return v.astype({"game_id": "int64", "vegas_home": "float64", "vegas_total": "float64"})


def update(day: pd.Timestamp) -> None:
    for league in D.LEAGUES:
        try:
            board = get_json(URL.format(sport=SPORT[league], ymd=day.strftime("%Y%m%d")))
        except Exception as e:  # odds are a bonus: never sink the daily run
            print(f"{league}: Vegas lines unavailable ({e})")
            continue
        new = parse(board, day)
        save(league, new)
        print(f"{league}: {len(board.get('events', []))} games on ESPN, {len(new)} with a pre-game line")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    update(pd.Timestamp(args[0]) if args else
           pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None))
