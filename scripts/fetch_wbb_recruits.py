"""Women's recruiting rankings from On3's ranking pages (industry view includes each
service's rank). On3 only has women's classes from 2025 on. Run from GitHub Actions.
python scripts/fetch_wbb_recruits.py 2025 2026"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "recruits"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
URL = "https://www.on3.com/db/rankings/industry-player/womens-basketball/{y}/?page={p}"


def page(y, p):
    with urllib.request.urlopen(urllib.request.Request(URL.format(y=y, p=p), headers=UA), timeout=60) as r:
        h = r.read().decode("utf-8", "replace")
    d = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))
    return d["props"]["pageProps"]["playerData"]


def rows(y):
    out, p = [], 1
    while True:
        pd_ = page(y, p)
        for it in pd_.get("list", []):
            person = it.get("person") or {}
            name = person.get("name") or " ".join(x for x in (person.get("firstName"), person.get("lastName")) if x)
            college = (it.get("collegeTeam") or {}).get("fullName") if isinstance(it.get("collegeTeam"), dict) else None
            committed = next((pr.get("organization", {}).get("fullName") for pr in person.get("predictions", [])
                              if pr.get("committed")), None)
            row = {"class": y, "player": name, "college": college or committed,
                   "industry_rank": it.get("consensusOverallRank"), "on3_rank": it.get("overallRank")}
            for rt in it.get("ratings") or []:
                row[f"{str(rt.get('type')).lower()}_rank"] = rt.get("overallRank")
            out.append(row)
        pg = pd_.get("pagination") or {}
        if p >= (pg.get("pageCount") or 1):
            break
        p += 1
        time.sleep(6)
    return pd.DataFrame(out)


if __name__ == "__main__":
    a, b = (int(x) for x in sys.argv[1:3])
    OUT.mkdir(parents=True, exist_ok=True)
    for y in range(a, b + 1):
        df = rows(y)
        df.to_csv(OUT / f"wbb_on3_{y}.csv", index=False)
        print(y, len(df), "players; columns:", list(df.columns))
        print(df.head(5).to_string())
        time.sleep(6)
