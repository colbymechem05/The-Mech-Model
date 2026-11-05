"""Probe women's recruiting ranking pages from GitHub Actions; print structure only."""
import json, re, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
URLS = ["https://www.espn.com/high-school/girls-basketball/recruiting/rankings/_/class/2022",
        "https://www.on3.com/rivals/rankings/player/womens-basketball/2025/",
        "https://www.on3.com/rivals/rankings/industry-player/womens-basketball/2025/",
        "https://www.on3.com/db/rankings/industry-player/womens-basketball/2025/",
        "https://www.on3.com/db/rankings/player/womens-basketball/2025/"]
for u in URLS:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40) as r:
            h = r.read().decode("utf-8", "replace")
        t = re.search(r"<title>(.*?)</title>", h, re.S)
        print("URL", u, "| len", len(h), "| title", (t.group(1).strip()[:90] if t else None),
              "| tables", h.count("<table"), "| next_data", "__NEXT_DATA__" in h)
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S)
        if m:
            d = json.loads(m.group(1))
            s = json.dumps(d)
            i = s.find("fullName")
            print("   NEXT sample:", s[max(0, i - 300): i + 600].replace("\n", " ")[:900])
        else:
            i = h.find("<table")
            print("   HTML sample:", re.sub(r"\s+", " ", h[i: i + 1500]) if i > 0 else re.sub(r"\s+", " ", h[:600]))
    except Exception as e:
        print("URL", u, "| ERROR", e)
