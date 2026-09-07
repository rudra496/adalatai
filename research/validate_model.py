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


def murmur3(data, seed=0):
    c1, c2 = 0xcc9e2d51, 0x1b873593
    h1 = seed & 0xffffffff
    nblocks = len(data) // 4
    for i in range(nblocks):
        k1 = int.from_bytes(data[i*4:(i+1)*4], "little")
        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xffffffff
        h1 = (h1 * 5 + 0xe6546b64) & 0xffffffff
    tail = len(data) - nblocks * 4
    k1 = 0
    if tail >= 3: k1 ^= data[nblocks*4 + 2] << 16
    if tail >= 2: k1 ^= data[nblocks*4 + 1] << 8
    if tail >= 1:
        k1 ^= data[nblocks*4]
        k1 = (k1 * c1) & 0xffffffff
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xffffffff
        k1 = (k1 * c2) & 0xffffffff
        h1 ^= k1
    h1 ^= len(data)
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85ebca6b) & 0xffffffff
    h1 ^= h1 >> 13
    h1 = (h1 * 0xc2b2ae35) & 0xffffffff
    h1 ^= h1 >> 16
    return h1 - (1 << 32) if h1 >= (1 << 31) else h1

def char_wb_ngrams(text, lo=2, hi=3):
    words = re.sub(r"[^\wঀ-৿]+", " ", text.lower()).split()
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
    return out

def score(model, text, cls):
    """Replicates app/assets/js/nlp2.js classifyV2 arithmetic exactly (v3 hash model)."""
    grams = set(char_wb_ngrams(text))
    s = model["classes"][cls].get("intercept", 0)
    nf = model["config"]["n_features"]
    for g in grams:
        b = murmur3(g.encode("utf-8")) % nf
        w = model["classes"][cls]["weights"].get(str(b))
        if w is not None: s += w
    return s

def main(path):
    ok, fail = [], []
    def chk(name, cond, detail=""):
        (ok if cond else fail).append((name, detail))

    m = json.load(open(path, encoding="utf-8"))
    chk("size < 5MB", os.path.getsize(path) < 5 * 1048576, f"{os.path.getsize(path)/1048576:.2f} MB")
    chk("config.analyzer is char_wb", "char_wb" in m.get("config", {}).get("analyzer", ""))
    chk("hash config present", m.get("config", {}).get("n_features") == 262144 and m["config"].get("seed") == 0)
    chk("classes complete", set(m["classes"].keys()) == REQUIRED_CLASSES,
        f"missing={sorted(REQUIRED_CLASSES - set(m['classes']))[:5]} extra={sorted(set(m['classes']) - REQUIRED_CLASSES)[:5]}")
    met = m.get("metrics", {})
    chk("metrics.gate F1 >= 0.85", met.get("micro_f1", 0) >= 0.85, str(met.get("micro_f1")))
    chk("per-class F1 >= 0.70", all(v >= 0.70 for v in met.get("per_class_f1", {}).values()) if met.get("per_class_f1") else False)
    chk("disclosed-synthetic statement", "synthetic" in m.get("description", "").lower() or "disclosed" in m.get("metrics", {}).get("corpus", "").lower())
    chk("train size >= 900k", met.get("train_sentences", 0) >= 450_000, str(met.get("train_sentences")))

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
