#!/usr/bin/env python3
"""AdalatAI evidence harvest: 100 papers on AI in law/justice — OpenAlex search across
8 themes, Crossref verification for every candidate (title-sim >= 0.8), curated to 100."""
import json, time, urllib.request, urllib.parse, difflib, os

RAW = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "AdalatAI-research/1.0 (mailto:rudrasarker130@gmail.com)"}

THEMES = {
    "T1_legal_judgment_prediction": ["legal judgment prediction", "court judgment prediction NLP", "case outcome prediction machine learning"],
    "T2_legal_nlp": ["legal natural language processing", "legal text summarization", "legal document classification"],
    "T3_legal_retrieval": ["legal information retrieval case law", "precedent retrieval", "semantic search legal documents"],
    "T4_ai_courts_ethics": ["artificial intelligence courts judiciary", "algorithmic decision-making justice ethics", "AI judge human rights"],
    "T5_access_to_justice": ["technology access to justice", "legal chatbot self-represented", "online dispute resolution"],
    "T6_sentencing_recidivism": ["sentencing prediction algorithm", "recidivism prediction COMPAS bias", "risk assessment criminal justice fairness"],
    "T7_bangla_lowresource_legal": ["Bangla NLP legal", "low-resource language legal NLP", "Bangladesh judiciary digitalization"],
    "T8_ecourt_digitization": ["e-court digitalization judiciary", "court case management system evaluation", "digital justice system developing country"],
}

def get_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  retry {i+1}: {str(e)[:70]}", flush=True)
            time.sleep(2 * (i + 1))
    return None

def norm(t):
    return "".join(ch.lower() for ch in (t or "") if ch.isalnum() or ch.isspace()).strip()

def oalex(q, n=14):
    d = get_json("https://api.openalex.org/works?search=" + urllib.parse.quote(q) +
                 f"&per-page={n}&mailto=rudrasarker130@gmail.com")
    out = []
    for w in (d or {}).get("results", []):
        out.append({
            "title": w.get("title"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:6],
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "cited_by": w.get("cited_by_count"),
            "query": q,
        })
    return out

def crossref_ok(doi, title):
    if not doi:
        return False
    d = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not d or not d.get("message", {}).get("title"):
        return False
    sim = difflib.SequenceMatcher(None, norm(d["message"]["title"][0]), norm(title)).ratio()
    return sim >= 0.80

def main():
    cands = []
    for theme, queries in THEMES.items():
        for q in queries:
            print(f"search: {q}", flush=True)
            for c in oalex(q):
                c["theme"] = theme
                cands.append(c)
            time.sleep(0.8)
    seen, uniq = set(), []
    for c in cands:
        key = c["doi"] or norm(c["title"])
        if key and key not in seen:
            seen.add(key); uniq.append(c)
    print(f"candidates: {len(cands)} -> unique {len(uniq)}", flush=True)
    verified = []
    for i, c in enumerate(uniq):
        if crossref_ok(c["doi"], c["title"]):
            c["crossref_ok"] = True
            verified.append(c)
        if (i + 1) % 40 == 0:
            print(f"  verified so far: {len(verified)} ({i+1}/{len(uniq)})", flush=True)
        time.sleep(0.4)
        if len(verified) >= 190:  # enough to curate 100
            break
    json.dump(verified, open(os.path.join(RAW, "harvest_law_raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"DONE: {len(verified)} Crossref-verified legal-AI papers", flush=True)

if __name__ == "__main__":
    main()
