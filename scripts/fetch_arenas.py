"""Arena capacity and elevation for the Home Court page. Run from GitHub Actions (the Claude
workspace can't reach these sites):  python scripts/fetch_arenas.py

Inputs:  data/arenas/venues.csv  (each team's usual home venue from ESPN schedules, written by
         `python src/export.py --venues`)
Sources: capacity  - Wikipedia "List of NCAA Division I basketball arenas" (CC BY-SA 4.0)
         elevation - Open-Meteo geocoding API (city elevation, meters -> feet)
Output:  data/arenas/arenas.csv  (team_id, league, capacity, elevation_ft, arena_wiki, matched_by)
"""
import io
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parents[1] / "data" / "arenas"
UA = {"User-Agent": "TheMeechModel/1.0 (college basketball ratings site; meechmodel.com)"}
WIKI = ("https://en.wikipedia.org/w/api.php?action=parse&format=json&formatversion=2&prop=text"
        "&page=List_of_NCAA_Division_I_basketball_arenas")
GEO = "https://geocoding-api.open-meteo.com/v1/search?count=10&language=en&format=json&countryCode=US&name="
STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
          "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
          "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
          "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
          "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
          "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
          "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
          "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
          "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
          "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}


def get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return json.load(r)


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


def wiki_arenas() -> pd.DataFrame:
    html = get(WIKI)["parse"]["text"]
    tables = [t for t in pd.read_html(io.StringIO(html)) if {"Arena", "Capacity"} <= set(map(str, t.columns))]
    t = pd.concat(tables, ignore_index=True)
    t["capacity"] = pd.to_numeric(t["Capacity"].astype(str).str.replace(r"\[.*?\]", "", regex=True)
                                  .str.replace(r"[^0-9]", "", regex=True), errors="coerce")
    return t.assign(arena=t["Arena"].astype(str), team=t["Team"].astype(str),
                    city=t.get("City", ""), state=t.get("State", ""))[["arena", "team", "city", "state", "capacity"]]


def match(v: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    w = w.assign(na=w.arena.map(norm), nt=w.team.map(norm))
    out = []
    for x in v.itertuples():
        na, school = norm(x.venue), norm(x.school)
        hit, how = w[w.na == na], "arena name"
        if len(hit) != 1:
            hit, how = w[w.na.map(lambda a: bool(a) and (a in na or na in a))], "arena name (partial)"
        if len(hit) != 1:
            hit, how = w[w.nt.map(lambda t: t == school or t.startswith(school + " "))], "school"
        if len(hit) > 1:   # school with several venues: the one in the same city
            hit = hit[hit.city.map(norm) == norm(x.city)]
        row = {"team_id": x.team_id, "league": x.league, "capacity": None, "arena_wiki": None, "matched_by": None}
        if len(hit) == 1:
            row.update(capacity=hit.capacity.iloc[0], arena_wiki=hit.arena.iloc[0], matched_by=how)
        out.append(row)
    return pd.DataFrame(out)


def elevations(v: pd.DataFrame) -> dict:
    out = {}
    pairs = v[["city", "state"]].dropna().drop_duplicates()
    for i, (city, st) in enumerate(pairs.itertuples(index=False)):
        if i % 50 == 0:
            print(f"elevation {i}/{len(pairs)}", flush=True)
        try:
            res = get(GEO + urllib.parse.quote(str(city))).get("results", [])
        except Exception:
            res = []
        res = [r for r in res if r.get("admin1") == STATES.get(st, st)]
        if res and res[0].get("elevation") is not None:
            out[(city, st)] = round(res[0]["elevation"] * 3.28084)
        time.sleep(0.25)
    return out


if __name__ == "__main__":
    v = pd.read_csv(FOLDER / "venues.csv")
    w = wiki_arenas()
    print(f"wikipedia: {len(w)} arenas", flush=True)
    m = match(v, w)
    print(f"matched {m.capacity.notna().sum()} of {len(m)}", flush=True)
    elev = elevations(v)
    m["elevation_ft"] = [elev.get((c, s)) for c, s in zip(v.city, v.state)]
    m.to_csv(FOLDER / "arenas.csv", index=False)
    print(f"{len(m)} teams | capacity matched {m.capacity.notna().sum()} | elevation {m.elevation_ft.notna().sum()}")
    print("no capacity:", ", ".join(v.loc[m.capacity.isna().values, "team"].astype(str).head(40)))
