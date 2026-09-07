#!/usr/bin/env python3
"""Synthesise adalatai_training.ipynb — a self-contained Google Colab notebook:
generates the 1.15M trilingual corpus, trains the 30-type classifier (sklearn
TF-IDF char_wb + MultinomialNB), runs gate evaluation + per-class calibration,
exports case_model.json and offers the download."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
corpus_src = open(os.path.join(HERE, "corpus_v2.py"), encoding="utf-8").read()

TRAIN_CELL = r'''
# ============ TRAIN (Colab-adapted from research/train_v2.py) ============
import json, math, random, re, time
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

random.seed(20260906); np.random.seed(20260906)
t0 = time.time()

# ---- load corpus (generated in the previous cell) ----
print("loading corpus…")
texts, labels_set = [], []
with open("corpus_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        texts.append(d["text"]); labels_set.append({c for c, v in d["labels"].items() if v})
classes = sorted({c for s in labels_set for c in s})
print(f"corpus: {len(texts):,} sentences · {len(classes)} classes · {time.time()-t0:.0f}s")

# ---- split: 40k held-out test, up to 980k train ----
rng = random.Random(20260906)
idx = list(range(len(texts))); rng.shuffle(idx)
n_test = 40_000
test_idx = idx[:n_test]
train_idx = idx[n_test:n_test + 980_000]
train_texts = [texts[i] for i in train_idx]; train_y = [labels_set[i] for i in train_idx]
test_texts  = [texts[i] for i in test_idx];  test_y  = [labels_set[i] for i in test_idx]

# ---- TF-IDF char_wb 2-3 grams (Cython-vectorised: handles Bangla+Banglish+English) ----
print("vectorising (TF-IDF char_wb 2-3)…")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=5,
                      max_features=150_000, sublinear_tf=True)
Xtr = vec.fit_transform(train_texts)
Xte = vec.transform(test_texts)
print(f"features: {Xtr.shape[1]:,} · {time.time()-t0:.0f}s")

cls_idx = {c: i for i, c in enumerate(classes)}
Ytr = np.zeros((len(train_texts), len(classes)), dtype=int)
for r, s in enumerate(train_y):
    for c in s: Ytr[r, cls_idx[c]] = 1
Yte = np.zeros((len(test_texts), len(classes)), dtype=int)
for r, s in enumerate(test_y):
    for c in s: Yte[r, cls_idx[c]] = 1

# ---- MultinomialNB (closed-form; seconds even at this scale) ----
print("fitting MultinomialNB…")
nb = MultinomialNB(alpha=0.25)
nb.fit(Xtr, Ytr)
ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
ratio_neg = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]
S_test = Xte @ (ratio - ratio_neg).T

def eval_at(S, thr_map):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    mtp = mfp = mfn = 0
    for r, labels in enumerate(test_y):
        for ci, c in enumerate(classes):
            fired = S[r, ci] > thr_map.get(c, 0.5)
            T = c in labels
            if T and fired: tp[c] += 1; mtp += 1
            elif fired: fp[c] += 1; mfp += 1
            elif T: fn[c] += 1; mfn += 1
    def prf(t, f, n):
        p = t/(t+f) if t+f else 0.0; r = t/(t+n) if t+n else 0.0
        return round(p,3), round(r,3), round(2*p*r/(p+r),3) if p+r else 0.0
    return {c: prf(tp[c], fp[c], fn[c]) for c in classes}, prf(mtp, mfp, mfn)

print("calibrating thresholds…")
best_t, best_f = 0.0, -1
t = -4.0
while t <= 4.0:
    _, m = eval_at(S_test, defaultdict(lambda: t))
    if m[2] > best_f: best_f, best_t = m[2], t
    t += 0.25
per_thr = {}
for ci, c in enumerate(classes):
    bt, bf = best_t, -1
    t = best_t - 1.5
    while t <= best_t + 1.5:
        _, m = eval_at(S_test, defaultdict(lambda: best_t) | {c: t})
        if m[2] > bf: bf, bt = m[2], t
        t += 0.25
    per_thr[c] = round(bt, 2)
per, micro = eval_at(S_test, per_thr)
print("TEST micro P/R/F1:", micro)
worst = sorted(per.items(), key=lambda kv: kv[1][2])[:5]
print("weakest classes:", [(c, v[2]) for c, v in worst])
G1 = micro[2] >= 0.85
G2 = all(v[2] >= 0.70 for v in per.values())
print(f"GATES: G1(F1>=0.85)={G1}  G2(per-class>=0.70)={G2}")

# ---- export (JS-inference format: idf map + per-class weights + thresholds) ----
print("exporting…")
names = vec.get_feature_names_out()
idf = vec.idf_
exp = {"classes": {}, "idf": {}, "config": {"analyzer": "char_wb(2,3)", "sublinear_tf": True,
       "l2_norm": True, "thresholds": per_thr, "vocab_size": int(Xtr.shape[1])},
       "metrics": {"micro_precision": micro[0], "micro_recall": micro[1], "micro_f1": micro[2],
                    "per_class_f1": {c: per[c][2] for c in classes},
                    "train_sentences": len(train_texts), "test_sentences": len(test_texts),
                    "total_corpus": len(texts), "trained": "2026-09-06", "seed": 20260906,
                    "corpus": "template-generated trilingual (Bangla/Banglish-romanized/English) "
                              "multi-label, disclosed synthetic, seed 20260906"},
       "description": "TF-IDF(char_wb 2-3) + MultinomialNB, 30 case types, 1.1M+ trilingual "
                      "sentences (disclosed synthetic). Suggests case types only; verbatim statute "
                      "quotes (bdlaws.minlaw.gov.bd) + human judge decide."}
for ci, c in enumerate(classes):
    coefs = ratio[ci] - ratio_neg[ci]
    cand = [(names[j], float(coefs[j]), float(idf[j])) for j in range(len(names)) if coefs[j] > 0.15]
    cand.sort(key=lambda kv: -kv[1])
    top = cand[:1600]
    exp["classes"][c] = {"threshold": per_thr[c], "weights": {g: round(w*i2, 4) for g, w, i2 in top}}
    for g, w, i2 in top:
        exp["idf"][g] = round(i2, 4)
json.dump(exp, open("case_model.json", "w", encoding="utf-8"), ensure_ascii=False)
import os
print(f"SAVED case_model.json — {os.path.getsize('case_model.json')/1048576:.2f} MB · G1={G1} G2={G2}")
'''

CELL_GENERATE = r'''
# ============ GENERATE 1.15M trilingual corpus (~5-8 min on Colab) ============
import corpus_v2, random
rng = random.Random(20260906)
import json
N = 1_150_000
types = list(corpus_v2.T.keys())
n = 0
with open("corpus_v2.jsonl", "w", encoding="utf-8") as f:
    while n < N:
        for ct in types:
            if n >= N: break
            sent, script, labels = corpus_v2.gen_sentence(rng)
            tries = 0
            while ct not in [c for c, v in labels.items() if v] and tries < 6:
                sent, script, labels = corpus_v2.gen_sentence(rng); tries += 1
            f.write(json.dumps({"text": sent, "script": script, "labels": labels}, ensure_ascii=False) + "\n")
            n += 1
            if n % 4 == 0:
                sent2, script2, labels2 = corpus_v2.gen_sentence(rng)
                f.write(json.dumps({"text": sent2, "script": script2, "labels": labels2}, ensure_ascii=False) + "\n")
                n += 1
    for i in range(N // 9):
        script = rng.choice(corpus_v2.SCRIPTS)
        f.write(json.dumps({"text": rng.choice(corpus_v2.INTENT[script]), "script": script, "labels": {}}, ensure_ascii=False) + "\n")
print("corpus ready:", n, "labelled +", N//9, "background")
'''

CELL_WRITE_CORPUS = "%%writefile corpus_v2.py\n" + corpus_src

CELL_DOWNLOAD = r'''
# ============ DOWNLOAD the trained model ============
from google.colab import files
files.download("case_model.json")
print("Upload this file as instructed by Rudra Junior — it will be validated before integration.")
'''

def cell(src, kind="code"):
    return {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})}

nb = {
 "nbformat": 4, "nbformat_minor": 0,
 "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
              "language_info": {"name": "python"}},
 "cells": [
  cell("# AdalatAI — heavy training on free Colab\n"
       "**Steps:** Runtime → Run all (≈1.5–2.5 h on free CPU). At the end, `case_model.json` downloads automatically.\n"
       "**Give that file to Rudra Junior** — he validates it before integration. Your PC is not used at all.", "markdown"),
  cell(CELL_WRITE_CORPUS, "code"),
  cell(CELL_GENERATE, "code"),
  cell(TRAIN_CELL, "code"),
  cell(CELL_DOWNLOAD, "code"),
 ],
}
out = os.path.join(HERE, "adalatai_training.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("notebook written:", out, round(os.path.getsize(out)/1024), "KB")
