"""Transfer portal entries from CollegeBasketballData.com (men's), for the JUCO / non-D1
transfer test (analysis/portal_test.py). Not used by the model.
Needs CBBD_API_KEY. Writes data/transfers/cbbd_portal_{year}.csv.
Run: python scripts/fetch_cbbd_portal.py 2021 2026
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://api.collegebasketballdata.com/recruiting/portal?year={year}"
FOLDER = Path(__file__).resolve().parents[1] / "data" / "transfers"

if __name__ == "__main__":
    key = os.environ.get("CBBD_API_KEY", "").strip()
    if not key:
        sys.exit("CBBD_API_KEY is not set")
    FOLDER.mkdir(parents=True, exist_ok=True)
    for year in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
        req = urllib.request.Request(URL.format(year=year), headers={
            "Authorization": f"Bearer {key}", "Accept": "application/json",
            "User-Agent": "TheMeechModel/1.0 (college basketball ratings)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        if data:
            print("example:", json.dumps(data[0])[:600], flush=True)
        rows = []
        for p in data:
            o, d = p.get("origin") or {}, p.get("destination") or {}
            el = p.get("eligibility")
            rows.append({"year": p.get("year", year), "first": p.get("firstName"), "last": p.get("lastName"),
                         "position": p.get("position"), "origin": o.get("name"), "origin_id": o.get("id"),
                         "origin_conf": o.get("conference"), "dest": d.get("name"), "dest_id": d.get("id"),
                         "dest_conf": d.get("conference"), "stars": p.get("stars"), "rating": p.get("rating"),
                         "years_left": p.get("yearsRemaining"),
                         "eligibility": json.dumps(el) if isinstance(el, (dict, list)) else el})
        df = pd.DataFrame(rows)
        df.to_csv(FOLDER / f"cbbd_portal_{year}.csv", index=False)
        print(f"{year}: {len(df)} entries, with destination {df.dest.notna().sum() if len(df) else 0}, "
              f"rated {df.rating.notna().sum() if len(df) else 0}", flush=True)
        time.sleep(1)
