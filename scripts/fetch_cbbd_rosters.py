"""Team rosters (with date of birth and first season) from CollegeBasketballData.com (men's),
for the older-newcomer / JUCO test (analysis/newcomer_age_test.py). Not used by the model.
Needs CBBD_API_KEY. Writes data/rosters/cbbd_roster_{season}.csv (season = spring year).
Run: python scripts/fetch_cbbd_rosters.py 2021 2027
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.collegebasketballdata.com/teams/roster?season={season}"
FOLDER = Path(__file__).resolve().parents[1] / "data" / "rosters"

if __name__ == "__main__":
    key = os.environ.get("CBBD_API_KEY", "").strip()
    if not key:
        sys.exit("CBBD_API_KEY is not set")
    FOLDER.mkdir(parents=True, exist_ok=True)
    for season in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        req = urllib.request.Request(URL.format(season=season), headers={
            "Authorization": f"Bearer {key}", "Accept": "application/json",
            "User-Agent": "TheMeechModel/1.0 (college basketball ratings)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        rows = []
        for t in data:
            for p in t.get("players") or []:
                rows.append({"season": season, "team": t.get("team"), "conference": t.get("conference"),
                             "player_id": p.get("id"), "name": p.get("name"), "position": p.get("position"),
                             "height": p.get("height"), "dob": p.get("dateOfBirth"),
                             "start_season": p.get("startSeason"), "end_season": p.get("endSeason")})
        if data and data[0].get("players"):
            print("example:", json.dumps(data[0]["players"][0])[:400], flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(FOLDER / f"cbbd_roster_{season}.csv", index=False)
        print(f"{season}: {len(data)} teams, {len(df)} players, dob {df.dob.notna().sum() if len(df) else 0}, "
              f"start_season {df.start_season.notna().sum() if len(df) else 0}", flush=True)
        time.sleep(1)
