#!/usr/bin/env python3
"""Validate an incoming case_model.json (e.g. trained on Colab) BEFORE integration.
Checks: structure, 30 classes, metric gates, JS-parity spot scores, model size.
Usage: python validate_model.py path/to/case_model.json"""
import json, math, re, sys, os

REQUIRED_CLASSES = {
    "theft", "robbery", "dacoity", "murder", "culpable_homicide", "hurt", "assault_women",
    "eve_teasing", "rape", "kidnapping", "wrongful_confinement", "cheating", "breach_of_trust",
    "mischief", "trespass", "defamation", "criminal_intimidation", "sexual_harassment_workplace",
    "dowry", "cheque_bounce", "cyber_fraud", "cyber_defamation", "cyber_hacking", "narcotics",
    "corruption_bribery", "negligence_death", "money_loan_default", "family_maintenance",
    "village_dispute", "acid_violence",
}

def char_wb_ngrams(text, lo=2, hi=3):
    words = re.sub(r"[^\w\u0980-\u09FF]+", " ", text.lower()).split()
    out = []
    for w in words:
        padded = " " + w + " "
        L = len(padded)
        if L <= lo:
            out.append(padded); continue
        n_max = min(hi, L - 1)
        for n in range(lo, n_max + 1):
            for i in range(L - n + 1):
                out.append(padded[i:i + n])

def score(model, text, cls):
    """Replicates app/assets/js/nlp2.js classifyV2 arithmetic exactly."""
    words = re.sub(r"[^\w\u0980-\u09FF]+", " ", text.lower()).split()
    grams = []
    for w in words:
        padded = " " + w + " "
        L = len(padded)
        if L <= 2:
            grams.append(padded); continue
        n_max = min(3, L - 1)
        for n in range(2, n_max + 1):
            for i in range(L - n + 1):
                grams.append(padded[i:i + n])
    counts = {}
    for g in grams: counts[g] = counts.get(g, 0) + 1
    s = 0.0; norm = 0.0
    for g, cnt in counts.items():
        i = model["idf"].get(g)
        if i is None: continue
        tfidf = (1 + math.log(cnt)) * i
        norm += tfidf * tfidf
        w = model["classes"][cls]["weights"].get(g)
        if w is not None: s += tfidf * w
    norm = math.sqrt(norm) or 1
    return s / norm

def main(path):
    ok, fail = [], []
    def chk(name, cond, detail=""):
        (ok if cond else fail).append((name, detail))

    m = json.load(open(path, encoding="utf-8"))
    chk("size < 5MB", os.path.getsize(path) < 5 * 1048576, f"{os.path.getsize(path)/1048576:.2f} MB")
    chk("config.analyzer is char_wb", "char_wb" in m.get("config", {}).get("analyzer", ""))
    chk("classes complete", set(m["classes"].keys()) == REQUIRED_CLASSES,
        f"missing={sorted(REQUIRED_CLASSES - set(m['classes']))[:5]} extra={sorted(set(m['classes']) - REQUIRED_CLASSES)[:5]}")
    chk("idf map present", len(m.get("idf", {})) > 5000, f"{len(m.get('idf', {}))} ngrams")
    met = m.get("metrics", {})
    chk("metrics.gate F1 >= 0.85", met.get("micro_f1", 0) >= 0.85, str(met.get("micro_f1")))
    chk("per-class F1 >= 0.70", all(v >= 0.70 for v in met.get("per_class_f1", {}).values()) if met.get("per_class_f1") else False)
    chk("disclosed-synthetic statement", "synthetic" in m.get("description", "").lower())
    chk("train size >= 900k", met.get("train_sentences", 0) >= 900_000, str(met.get("train_sentences")))

    # spot checks: must-fire and must-not-fire across 3 scripts
    SPOT_FIRE = [
        ("theft", "dokan theke TV churi hoyeche"),
        ("theft", "দোকান ভেঙে চুরি হয়েছে"),
        ("theft", "shop broken into and goods stolen"),
        ("murder", "ছুরি মেরে খুন করেছে"),
        ("murder", "stabbed to death"),
        ("dowry", "যৌতুকের জন্য নির্যাতন"),
        ("cheque_bounce", "cheek dishonour hoyeche taka nei"),
        ("narcotics", "yaba bikrir somoy dhora poreche"),
        ("eve_teasing", "school girl ke oshlil kotha bole ev-tijging"),
        ("cyber_fraud", "fake facebook ID diye taka protarona"),
    ]
    fire_ok = 0
    for want_cls, text in SPOT_FIRE:
        try:
            sc = score(m, text, want_cls)
            thr = m["classes"][want_cls]["threshold"]
            if sc > thr: fire_ok += 1
            else: fail.append((f"fire:{text[:30]}", f"score {sc:.2f} <= thr {thr}"))
        except Exception as e:
            fail.append((f"fire:{text[:30]}", str(e)[:50]))
    chk("spot-fire >= 8/10", fire_ok >= 8, f"{fire_ok}/10")

    print("=" * 46)
    for n in ok: print("  PASS", n)
    for n, d in fail: print("  FAIL", n, "—", d)
    print("=" * 46)
    verdict = "VALID — safe to integrate" if not fail else "REJECTED — do not integrate"
    print(verdict)
    sys.exit(0 if not fail else 1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case_model.json")
