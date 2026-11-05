"""Probe On3 women's ranking JSON structure from GitHub Actions; print structure only."""
import json, re, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def nd(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40) as r:
        h = r.read().decode("utf-8", "replace")
    return json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))

def find_lists(o, path="", out=None):
    out = [] if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            find_lists(v, f"{path}.{k}", out)
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        out.append((path, len(o), list(o[0].keys())[:40]))
        for i, v in enumerate(o[:1]):
            find_lists(v, f"{path}[0]", out)
    return out

for kind in ["industry-player", "player"]:
    for y in [2020, 2022, 2025]:
        u = f"https://www.on3.com/db/rankings/{kind}/womens-basketball/{y}/"
        try:
            d = nd(u)
            pp = d.get("props", {}).get("pageProps", {})
            print("URL", u, "| pageProps keys:", list(pp.keys())[:20])
            for path, n, keys in find_lists(pp)[:6]:
                print("   LIST", path, n, keys)
            s = json.dumps(pp)
            for k in ["pagination", "totalCount", "pageCount", "\"count\""]:
                i = s.find(k)
                if i >= 0:
                    print("   ", k, s[i:i + 200])
        except Exception as e:
            print("URL", u, "| ERROR", e)
# one full player record from industry 2025
d = nd("https://www.on3.com/db/rankings/industry-player/womens-basketball/2025/")
s = json.dumps(d["props"]["pageProps"])
i = s.find('"rankings"')
print("RECORD", s[i: i + 2500])
