"""Recruiting rankings for incoming freshmen (men's: RSCI consensus top 100).

data/recruits/rsci_{class}.csv comes from scripts/fetch_rsci.py (GitHub Actions).
High-school class Y plays its freshman season in season Y + 1 (class of 2026 -> 2026-27).
"""
import re
import unicodedata

import pandas as pd

import data as D

FOLDER = D.ROOT / "data" / "recruits"
ALIAS = {"Central Florida": "UCF", "Florida Atlantic University": "Florida Atlantic",
         "Howard University": "Howard", "Kent State University": "Kent State", "Miami (FL)": "Miami",
         "Pitt": "Pittsburgh", "Rutgers University": "Rutgers", "Samford University": "Samford",
         "UNC": "North Carolina", "UW-Milwaukee": "Milwaukee",
         "University of California, Irvine": "UC Irvine"}
BINS = [0, 5, 10, 25, 50, 75, 101]
LABELS = ["1-5", "6-10", "11-25", "26-50", "51-75", "76-100"]


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def school_ids(league: str = "mbb") -> dict:
    """ESPN school name (location) -> team_id."""
    out = {}
    for s in D.SEASONS:
        f = D.RAW / f"{league}_schedule_{s}.csv"
        if f.exists():
            sc = pd.read_csv(f, low_memory=False, usecols=["home_id", "home_location"])
            out.update(dict(zip(sc.home_location, sc.home_id)))
    return out


WBB_SOURCES = ["espn_rank", "on3_rank", "247_rank"]   # On3 = Rivals' parent; Rivals itself is blank


def load(league: str = "mbb", source: str = "avg") -> pd.DataFrame:
    """All recruits: class, season (freshman year), rank, player, team_id (NaN if unknown).
    Men's: RSCI consensus top 100. Women's: On3's page with each service's rank (classes 2025+);
    `source` picks ESPN, On3/Rivals, 247 or the average of whichever ranked the player."""
    if league == "wbb":
        files = sorted(FOLDER.glob("wbb_on3_*.csv"))
        if not files:
            return pd.DataFrame(columns=["class", "season", "rank", "tier", "player", "college", "team_id", "key"])
        r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        raw = r[WBB_SOURCES].mean(axis=1) if source == "avg" else r[source]
        r = r[raw.notna()].copy()
        r["rank"] = raw[raw.notna()].groupby(r["class"]).rank(method="first").astype(int)
        r = r[r["rank"] <= 100]
        ids = school_ids(league)
        r["team_id"] = r.college.map(lambda c: ids.get(c) if isinstance(c, str) else None)
        r["season"] = r["class"] + 1
        r["key"] = r.player.map(norm_name)
        r["tier"] = pd.cut(r["rank"], BINS, labels=LABELS)
        return r[["class", "season", "rank", "tier", "player", "college", "team_id", "key"]]
    files = sorted(FOLDER.glob("rsci_*.csv"))
    if not files:
        return pd.DataFrame(columns=["class", "season", "rank", "player", "college", "team_id", "key"])
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ids = school_ids(league)
    r["team_id"] = r.college.map(lambda c: ids.get(ALIAS.get(c, c)) if isinstance(c, str) else None)
    r["season"] = r["class"] + 1
    r["key"] = r.player.map(norm_name)
    r["tier"] = pd.cut(r["rank"], BINS, labels=LABELS)
    return r[["class", "season", "rank", "tier", "player", "college", "team_id", "key"]]


def _variants(name: str) -> list:
    """Name keys to try: as listed, without a quoted nickname, and nickname + last name."""
    out = [norm_name(re.sub(r'"[^"]*"', " ", name))]
    m = re.search(r'"([^"]+)"', name)
    if m:
        last = norm_name(name.split('"')[-1])
        out.append(norm_name(f"{m.group(1)} {last}"))
    out.append(norm_name(name))
    return list(dict.fromkeys(out))


def match(rec: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Find each recruit in a season's player stats (any D1 team: decommits and late
    switches land where they actually played). Adds athlete_id and played_team (NaN if
    not found: went pro, redshirted with no box score, or a name we can't match)."""
    st = stats[["athlete_id", "team_id", "name", "minutes"]].copy()
    st["key"] = st.name.map(norm_name)
    st["last"] = st.key.str.split().str[-1]
    st["first1"] = st.key.str[:1]
    by_key = st.groupby("key")
    out = []
    for _, r in rec.iterrows():
        hit = None
        for k in _variants(r.player):
            if k in by_key.groups:
                c = by_key.get_group(k)
                if len(c) > 1 and (c.team_id == r.team_id).any():
                    c = c[c.team_id == r.team_id]
                hit = c.sort_values("minutes").iloc[-1]
                break
        if hit is None:
            k = _variants(r.player)[0]
            toks = k.split()
            if toks:
                same = st[(st.team_id == r.team_id) & (st.key.str.startswith(k) |
                          ((st["last"] == toks[-1]) & (st.first1 == k[:1])))]
                if len(same) == 1:
                    hit = same.iloc[0]
        out.append({"athlete_id": None if hit is None else hit.athlete_id,
                    "played_team": None if hit is None else hit.team_id})
    return pd.concat([rec.reset_index(drop=True),
                      pd.DataFrame(out, columns=["athlete_id", "played_team"])], axis=1)
