"""Men's recruiting rankings from CollegeBasketballData.com (247Sports composite).

Needs the API key in the CBBD_API_KEY environment variable (a GitHub secret; never commit it).
Writes data/recruits/cbbd_{class}.csv: class, rank, player, college, rating, stars, position.
Run: python scripts/fetch_cbbd_recruits.py 2019 2026
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.collegebasketballdata.com/recruiting/players?year={year}"
FOLDER = Path(__file__).resolve().parents[1] / "data" / "recruits"


def get(year: int, key: str) -> list:
    req = urllib.request.Request(URL.format(year=year), headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
        "User-Agent": "TheMeechModel/1.0 (college basketball ratings)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  {year}: {e}; retrying", flush=True)
            time.sleep(5 * (attempt + 1))
    return []


def pick(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def rows(data: list, year: int) -> pd.DataFrame:
    out = []
    for p in data:
        to = pick(p, "committedTo", "committed_to") or {}
        out.append({"class": int(pick(p, "year") or year), "rank": pick(p, "ranking"), "player": pick(p, "name"),
                    "college": to.get("name") if isinstance(to, dict) else None,
                    "rating": pick(p, "rating"), "stars": pick(p, "stars"), "position": pick(p, "position")})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.dropna(subset=["rank", "player"])
    df["rank"] = df["rank"].astype(int)
    return df.sort_values("rank")


if __name__ == "__main__":
    key = os.environ.get("CBBD_API_KEY", "").strip()
    if not key:
        sys.exit("CBBD_API_KEY is not set")
    first, last = int(sys.argv[1]), int(sys.argv[2])
    FOLDER.mkdir(parents=True, exist_ok=True)
    for year in range(first, last + 1):
        data = get(year, key)
        if year == first and data:
            print("fields:", sorted(data[0].keys()), flush=True)
        df = rows(data, year)
        if df.empty:
            print(f"{year}: no recruits returned", flush=True)
            continue
        df.to_csv(FOLDER / f"cbbd_{year}.csv", index=False)
        print(f"{year}: {len(df)} recruits, ranks {df['rank'].min()}-{df['rank'].max()}, "
              f"committed {df.college.notna().sum()}", flush=True)
        time.sleep(1)
