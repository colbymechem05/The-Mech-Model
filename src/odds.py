"""Vegas lines from ESPN's public scoreboard (odds come from ESPN's sportsbook partner).

Run:  python src/odds.py [YYYY-MM-DD]      (default: today, Central time)
Keeps logs/vegas_{lg}.csv, one row per game:
  vegas_home / vegas_total   latest pre-tip line seen on the scoreboard (updated only before tip)
  close_home / close_total   the CLOSING line, from ESPN's odds feed once the game is over
Grading uses the closing line when it exists, else the last line captured before tip.

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
CORE = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/{sport}"
        "/events/{gid}/competitions/{gid}/odds")
COLUMNS = ["game_id", "date", "vegas_home", "vegas_total", "provider", "fetched_at",
           "close_home", "close_total", "close_fetched"]
FINAL_AFTER = pd.Timedelta(hours=3)   # a game this long past tip is over: its close is final


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
    """Upsert into the Vegas log: newest pre-tip line wins; closing-line columns are kept."""
    LOGS.mkdir(exist_ok=True)
    f = LOGS / f"vegas_{league}.csv"
    old = load(league)
    keep = ["close_home", "close_total", "close_fetched"]
    new = new.drop(columns=[c for c in keep if c in new], errors="ignore").merge(
        old[["game_id"] + keep], on="game_id", how="left")
    both = pd.concat([old[~old.game_id.isin(new.game_id)], new], ignore_index=True)
    both[COLUMNS].sort_values(["date", "game_id"]).to_csv(f, index=False)
    return f


def parse_close(odds_feed: dict):
    """(close_home, close_total) from ESPN's core odds feed for one game.
    close_home: home expected margin (+ = home favored), from the home side's closing point spread."""
    items = odds_feed.get("items") or []
    if not items:
        return None, None
    it = items[0]
    ps = (((it.get("homeTeamOdds") or {}).get("close") or {}).get("pointSpread") or {}).get("american")
    if ps is None:
        return None, it.get("overUnder")
    ps = str(ps).strip().upper()
    if ps in ("EVEN", "PK", "PICK"):
        return 0.0, it.get("overUnder")
    try:
        return -float(ps), it.get("overUnder")
    except ValueError:
        return None, it.get("overUnder")


def fill_closing(league: str, now: pd.Timestamp | None = None, lookback_days: int = 10) -> int:
    """Closing lines for finished games we made a line for (or saw a Vegas line for)
    that don't have one yet. One small request per game, politely spaced."""
    now = now or pd.Timestamp.now(tz="UTC")
    sched = D.RAW / f"{league}_schedule_{D.CURRENT}.csv"
    if not sched.exists():
        return 0
    tips = (pd.read_csv(sched, low_memory=False, usecols=["game_id", "game_date", "game_date_time"])
              .drop_duplicates("game_id").set_index("game_id"))
    v = load(league)
    pred = LOGS / f"predictions_{league}.csv"
    ids = set(v.game_id)
    if pred.exists():
        ids |= set(pd.read_csv(pred, usecols=["game_id"]).game_id)
    have = set(v[v.close_home.notna()].game_id)
    tip = pd.to_datetime(tips.game_date_time.reindex(list(ids - have)), utc=True, errors="coerce").dropna()
    todo = tip[(tip + FINAL_AFTER < now) & (tip > now - pd.Timedelta(days=lookback_days))].index
    rows, stamp = [], pd.Timestamp.now(tz="America/Chicago").isoformat(timespec="minutes")
    for gid in todo:
        try:
            home, total = parse_close(get_json(CORE.format(sport=SPORT[league], gid=int(gid)), tries=2))
        except Exception:
            continue
        time.sleep(0.3)
        if home is not None:
            rows.append({"game_id": int(gid), "close_home": home, "close_total": total, "close_fetched": stamp,
                         "date": str(tips.game_date.get(gid, ""))[:10]})
    if not rows:
        return 0
    c = pd.DataFrame(rows).set_index("game_id")
    v = v.set_index("game_id")
    for col in ["close_home", "close_total", "close_fetched"]:
        v[col] = v[col].astype(object)
        v.loc[v.index.intersection(c.index), col] = c[col]
    extra = c.loc[c.index.difference(v.index)]          # games with a close but no captured line
    v = pd.concat([v, extra.assign(provider="Draft Kings")]).reset_index()
    v[COLUMNS].sort_values(["date", "game_id"]).to_csv(LOGS / f"vegas_{league}.csv", index=False)
    return len(rows)


def load(league: str) -> pd.DataFrame:
    f = LOGS / f"vegas_{league}.csv"
    v = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in v:
            v[c] = None
    return v[COLUMNS].astype({"game_id": "int64", "vegas_home": "float64", "vegas_total": "float64",
                              "close_home": "float64", "close_total": "float64"})


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
        try:
            print(f"{league}: {fill_closing(league)} closing lines saved")
        except Exception as e:
            print(f"{league}: closing lines unavailable ({e})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    update(pd.Timestamp(args[0]) if args else
           pd.Timestamp.now(tz="America/Chicago").normalize().tz_localize(None))
