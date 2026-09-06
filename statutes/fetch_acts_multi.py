#!/usr/bin/env python3
"""Fetch multiple Bangladesh acts from bdlaws (official) — TOC + section texts per act.
Modern acts come in Bangla; older in English. All stored verbatim with provenance."""
import json, re, time, urllib.request, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (AdalatAI research; contact rudrasarker130@gmail.com)"}
BASE = "http://bdlaws.minlaw.gov.bd"

ACTS = {
    "75":  "The Code of Criminal Procedure, 1898",
    "46":  "The Negotiable Instruments Act, 1881",
    "24":  "The Evidence Act, 1872",
    "26":  "The Contract Act, 1872",
    "607": "The Dowry Prohibition Act, 1980",
    "835": "নারী ও শিশু নির্যাতন দমন আইন, ২০০০",
    "1256":"যৌতুক নিরোধ আইন, ২০১৮",
    "1261":"ডিজিটাল নিরাপত্তা আইন, ২০১৮",
    "1276":"মাদকদ্রব্য নিয়ন্ত্রণ আইন, ২০১৮",
    "901": "অর্থ ঋণ আদালত আইন, ২০০৩",
    "914": "দুর্নীতি দমন কমিশন আইন, ২০০৪",
    "938": "গ্রাম আদালত আইন, ২০০৬",
    "950": "তথ্য ও যোগাযোগ প্রযুক্তি আইন, ২০০৬",
    "654": "The Motor Vehicles Ordinance, 1983",
    "883": "এসিড অপরাধ দমন আইন, ২০০২",
}

def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=50).read()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
            return raw.decode(enc, errors="replace")
        except Exception as e:
            print(f"  retry {i+1} {url[-40:]}: {str(e)[:50]}", flush=True)
            time.sleep(2 * (i + 1))
    return None

def strip_html(s):
    s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_secdec(html):
    m = re.search(r'id="sec-dec"[^>]*>([\s\S]*?)(?:<div class="|</div>\s*<div id="|<!-- )', html)
    if not m:
        m = re.search(r'id="sec-dec"[^>]*>([\s\S]*?)</div>', html)
    if not m:
        return "", ""
    raw = re.sub(r"<script[\s\S]*?</script>", " ", m.group(1))
    body = re.sub(r"<[^>]+>", " ", raw)
    body = re.sub(r"\s+", " ", body).strip()
    foot = ""
    fm = re.search(r"\s\d+\s(?=(Throughout|Substituted|Inserted|Omitted|Subs\.|Added|Amended|১৭|১৮))", body)
    if fm:
        body, foot = body[:fm.start()].strip(), body[fm.start():].strip()
    return body, foot

def fetch_act(act_id):
    html = fetch(f"{BASE}/act-{act_id}.html")
    if not html:
        return None
    pairs = re.findall(r'<a[^>]+href="/?act-\d+/section-(\d+)\.html"[^>]*>([\s\S]*?)</a>', html)
    toc = []
    seen = set()
    for sid, title in pairs:
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", title)).strip()
        m = re.match(r"([\d০-৯]+[A-Zক-ৎ]*)[.।৴-৹\s]+(.+)", title)
        if m and sid not in seen:
            seen.add(sid)
            toc.append({"sec_id": sid, "number": m.group(1), "title": m.group(2)})
    sections = []
    for p in toc:
        html2 = fetch(f"{BASE}/act-{act_id}/section-{p['sec_id']}.html")
        if not html2:
            continue
        body, foot = extract_secdec(html2)
        if len(body) >= 15:
            sections.append({**p, "text": body, "footnotes": foot,
                             "url": f"{BASE}/act-{act_id}/section-{p['sec_id']}.html",
                             "fetched": "2026-09-06"})
        time.sleep(0.3)
    return {"act_id": act_id, "title": ACTS[act_id], "source": BASE, "fetched": "2026-09-06",
            "section_count": len(sections), "toc_size": len(toc), "sections": sections}

def main():
    only_failed = "--failed" in sys.argv
    all_out = {}
    if only_failed and os.path.exists(os.path.join(HERE, "acts_multi.json")):
        all_out = json.load(open(os.path.join(HERE, "acts_multi.json"), encoding="utf-8"))
    for act_id in (ACTS if not only_failed else [a for a in ACTS if a not in all_out]):
        print(f"== act-{act_id}: {ACTS[act_id]}", flush=True)
        data = fetch_act(act_id)
        if data and data["section_count"] > 0:
            all_out[act_id] = data
            print(f"   {data['section_count']} sections saved", flush=True)
        else:
            print("   FAILED/empty", flush=True)
        time.sleep(0.5)
    json.dump(all_out, open(os.path.join(HERE, "acts_multi.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    total = sum(d["section_count"] for d in all_out.values())
    print(f"DONE: {len(all_out)} acts, {total} additional sections", flush=True)

if __name__ == "__main__":
    main()
