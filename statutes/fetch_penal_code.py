#!/usr/bin/env python3
"""Fetch the COMPLETE Penal Code 1860 from bdlaws.minlaw.gov.bd (official Ministry of Law).
Zero-invention: every section stored verbatim from the official page, with its URL + fetch date.
Integrity gates: count ~= 511 sections; §302/§379/§392 spot-assertions."""
import json, re, time, urllib.request, os, sys

RAW = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (AdalatAI research build; contact: rudrasarker130@gmail.com)"}
BASE = "http://bdlaws.minlaw.gov.bd"

def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=60).read()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
            return raw.decode(enc, errors="replace")
        except Exception as e:
            print(f"  retry {i+1} {url}: {e}", flush=True)
            time.sleep(2 * (i + 1))
    return None

def strip_html(s):
    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_toc():
    html = fetch(BASE + "/act-11.html")
    if not html:
        sys.exit("TOC fetch failed")
    open(os.path.join(RAW, "act11_toc.html"), "w", encoding="utf-8").write(html)
    # pairs: <a href="act-11/section-XXXX.html" ...>NUM. TITLE</a>
    pairs = re.findall(r'<a[^>]+href="/?act-11/section-(\d+)\.html"[^>]*>([\s\S]*?)</a>', html)
    out = []
    for sec_id, title in pairs:
        title = re.sub(r"\s+", " ", title).strip()
        m = re.match(r"(\d+[A-Z]*)\.\s*(.+)", title)
        if m:
            out.append({"sec_id": sec_id, "number": m.group(1), "title": m.group(2).strip()})
    # de-dup by sec_id preserving order
    seen, dedup = set(), []
    for p in out:
        if p["sec_id"] not in seen:
            seen.add(p["sec_id"]); dedup.append(p)
    return dedup

def extract_section(html):
    """Content lives in <div class="col-sm-9 txt-details" id="sec-dec"> — extract that
    container only (the page <title>/nav also contains the section title and must be
    excluded to avoid capturing navigation junk)."""
    m = re.search(r'id="sec-dec"[^>]*>([\s\S]*?)(?:<div class="|</div>\s*<div id="|<!-- )', html)
    if not m:
        m = re.search(r'id="sec-dec"[^>]*>([\s\S]*?)</div>', html)
    if not m:
        return "", ""
    raw = m.group(1)
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw)
    body = re.sub(r"<[^>]+>", " ", raw)
    body = re.sub(r"\s+", " ", body).strip()
    # split amendment footnotes (e.g. "1 Throughout this Act...") from the operative text
    foot = ""
    fm = re.search(r"\s\d+\s(?=(Throughout|Substituted|Inserted|Omitted|Subs\.|Added|Amended))", body)
    if fm:
        body, foot = body[:fm.start()].strip(), body[fm.start():].strip()
    return body, foot

def main():
    toc = parse_toc()
    print(f"TOC parsed: {len(toc)} sections")
    # 557 = 511 main sections + sub-section pages (e.g. 376(1)/(2) have own pages on bdlaws)
    assert 540 <= len(toc) <= 580, f"unexpected section count {len(toc)}"
    sections, fails = [], []
    for i, p in enumerate(toc):
        html = fetch(f"{BASE}/act-11/section-{p['sec_id']}.html")
        if not html:
            fails.append(p); continue
        body, foot = extract_section(html)
        if len(body) < 20:
            fails.append(p); continue
        sections.append({**p, "text": body, "footnotes": foot,
                         "url": f"{BASE}/act-11/section-{p['sec_id']}.html",
                         "fetched": "2026-09-06"})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(toc)} fetched", flush=True)
        time.sleep(0.35)
    print(f"fetched OK: {len(sections)}  failed: {len(fails)}")
    # retry failures once more
    still = []
    for p in fails:
        html = fetch(f"{BASE}/act-11/section-{p['sec_id']}.html")
        if html:
            body, foot = extract_section(html)
            if len(body) >= 20:
                sections.append({**p, "text": body, "footnotes": foot,
                                 "url": f"{BASE}/act-11/section-{p['sec_id']}.html",
                                 "fetched": "2026-09-06"})
                continue
        still.append(p)
    sections.sort(key=lambda s: (len(s["number"]), s["number"]))
    out = {
        "act": "The Penal Code, 1860 (Act No. XLV of 1860)",
        "source": "bdlaws.minlaw.gov.bd — Laws of Bangladesh, Ministry of Law, Justice and Parliamentary Affairs (official)",
        "fetched": "2026-09-06",
        "note": "Verbatim section text from the official source. Footnotes (amendment history) stored separately. Zero paraphrase.",
        "section_count": len(sections),
        "failed_sec_ids": [p["sec_id"] for p in still],
        "sections": sections,
    }
    json.dump(out, open(os.path.join(RAW, "penal_code_full.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # integrity gates
    bynum = {s["number"]: s for s in sections}
    assert "302" in bynum and "murder" in bynum["302"]["title"].lower(), "§302 missing"
    assert "commits theft" in bynum["379"]["text"] and "three years" in bynum["379"]["text"], "§379 gate failed: " + bynum["379"]["text"][:120]
    assert "commits robbery" in bynum["392"]["text"] and "ten years" in bynum["392"]["text"], "§392 gate failed: " + bynum["392"]["text"][:120]
    print("INTEGRITY GATES PASSED: §379 theft/3yr, §392 robbery/10yr verbatim confirmed")
    print(f"SAVED penal_code_full.json — {len(sections)} sections, {round(os.path.getsize(os.path.join(RAW,'penal_code_full.json'))/1024)} KB, failures: {len(still)}")

if __name__ == "__main__":
    main()
