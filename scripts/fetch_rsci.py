"""Download RSCI top-100 recruiting rankings (men's) from Basketball-Reference, one page
per high-school class, politely (one request every 6 seconds). Run from GitHub Actions:
the Claude workspace can't reach the site.  python scripts/fetch_rsci.py 2020 2026"""
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "recruits"
URL = "https://www.basketball-reference.com/awards/recruit_rankings_{}.html"


def fetch(year: int) -> pd.DataFrame:
    req = urllib.request.Request(URL.format(year), headers={"User-Agent": "Mozilla/5.0 (the-mech-model; personal research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    t = pd.read_html(io.StringIO(html))[0]
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = [c[-1] for c in t.columns]
    t = t[t["RSCI"].astype(str).str.fullmatch(r"T?\d+")].copy()
    t["rank"] = t["RSCI"].astype(str).str.lstrip("T").astype(int)
    t["player"] = t["Player"].astype(str).str.replace(r"\s*\(college\)\s*$", "", regex=True).str.strip()
    t["class"] = year
    keep = ["class", "rank", "player", "College"] + [c for c in ("ESPN", "Rivals", "247Sports", "On3") if c in t]
    return t[keep].rename(columns={"College": "college"})


if __name__ == "__main__":
    a, b = (int(x) for x in sys.argv[1:3])
    OUT.mkdir(parents=True, exist_ok=True)
    for y in range(a, b + 1):
        df = fetch(y)
        df.to_csv(OUT / f"rsci_{y}.csv", index=False)
        print(y, len(df), "recruits;", df.college.isna().sum(), "with no college listed")
        time.sleep(6)
