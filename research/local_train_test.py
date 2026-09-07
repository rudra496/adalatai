
# ============ TRAIN (OOM-proof: FeatureHasher + streaming SGD) ============
import json, math, random, re, time
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

random.seed(20260906); np.random.seed(20260906)
t0 = time.time()
N_FEATURES = 2 ** 18          # 262,144 hashed buckets
N_CLASSES = None

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
    return out

# ---- load corpus ----
print("loading corpus…", flush=True)
texts, labels_all = [], []
with open("corpus_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        texts.append(d["text"]); labels_all.append({c for c, v in d["labels"].items() if v})
classes = sorted({c for s in labels_all for c in s})
cls_idx = {c: i for i, c in enumerate(classes)}
print(f"corpus: {len(texts):,} · {len(classes)} classes · {time.time()-t0:.0f}s", flush=True)

# ---- split ----
rng = random.Random(20260906)
idx = list(range(len(texts))); rng.shuffle(idx)
n_test = 30_000
test_idx = idx[:n_test]
train_idx = idx[n_test:n_test + 900_000]
train_texts = [texts[i] for i in train_idx]; train_y = [labels_all[i] for i in train_idx]
test_texts  = [texts[i] for i in test_idx];  test_y  = [labels_all[i] for i in test_idx]

hasher = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)

def vectorize(texts_subset):
    grams = [char_wb_ngrams(t) for t in texts_subset]
    return hasher.transform(grams)

Y_of = lambda labels_rows: np.array([[1 if c in s else 0 for c in classes] for s in labels_rows], dtype=int)

# ---- streaming SGD: 3 epochs over ~45k mini-batches ----
print("training SGD (streaming)…", flush=True)
clfs = {c: SGDClassifier(loss="log_loss", alpha=2e-6, learning_rate="optimal", random_state=7)
        for c in classes}
first = {c: True for c in classes}
order = list(range(len(train_texts)))
for epoch in range(3):
    rng.shuffle(order)
    B = 45_000
    for start in range(0, len(order), B):
        part = order[start:start + B]
        Xb = vectorize([train_texts[i] for i in part])
        for c in classes:
            yb = np.array([1 if c in train_y[i] else 0 for i in part], dtype=int)
            if first[c]:
                clfs[c].partial_fit(Xb, yb, classes=np.array([0, 1])); first[c] = False
            else:
                clfs[c].partial_fit(Xb, yb)
    print(f"  epoch {epoch+1} done · {time.time()-t0:.0f}s", flush=True)

# ---- held-out evaluation (sklearn-exact) ----
print("evaluating…", flush=True)
Xte = vectorize(test_texts)
S = np.column_stack([clfs[c].decision_function(Xte) for c in classes])
def eval_at(S, thr_map):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    mtp = mfp = mfn = 0
    for r, labels in enumerate(test_y):
        for ci, c in enumerate(classes):
            fired = S[r, ci] > thr_map.get(c, 0.0)
            T = c in labels
            if T and fired: tp[c] += 1; mtp += 1
            elif fired: fp[c] += 1; mfp += 1
            elif T: fn[c] += 1; mfn += 1
    def prf(t, f, n):
        p = t/(t+f) if t+f else 0.0; r = t/(t+n) if t+n else 0.0
        return round(p,3), round(r,3), round(2*p*r/(p+r),3) if p+r else 0.0
    return {c: prf(tp[c], fp[c], fn[c]) for c in classes}, prf(mtp, mfp, mfn)
best_t, best_f = 0.0, -1
t = -3.0
while t <= 3.0:
    _, m = eval_at(S, defaultdict(lambda: t))
    if m[2] > best_f: best_f, best_t = m[2], t
    t += 0.25
per_thr = {}
for ci, c in enumerate(classes):
    bt, bf = best_t, -1
    t = best_t - 1.0
    while t <= best_t + 1.0:
        _, m = eval_at(S, defaultdict(lambda: best_t) | {c: t})
        if m[2] > bf: bf, bt = m[2], t
        t += 0.25
    per_thr[c] = round(bt, 2)
per, micro = eval_at(S, per_thr)
print("TEST micro:", micro, flush=True)
worst = sorted(per.items(), key=lambda kv: kv[1][2])[:5]
print("weakest:", [(c, v[2]) for c, v in worst], flush=True)
G1 = micro[2] >= 0.80; G2 = all(v[2] >= 0.65 for v in per.values())
print(f"G1(F1>=0.80)={G1} G2(per-class>=0.65)={G2}", flush=True)

# ---- EXPORT: truncated weights replicable in JS ----
print("exporting…", flush=True)
# per-class coefs from the 30 binary classifiers
ALL_COEFS = np.vstack([clfs[c].coef_ for c in classes])
ALL_INTER = np.array([clfs[c].intercept_[0] for c in classes])
coefs = ALL_COEFS
intercepts = ALL_INTER
export = {"classes": {}, "config": {"n_features": N_FEATURES, "alternate_sign": False,
          "seed": 0, "analyzer": "char_wb(2,3)", "thresholds": per_thr,
          "scoring": "sum(weights[bucket] for unique ngrams)"},
          "metrics": {"micro_precision": micro[0], "micro_recall": micro[1], "micro_f1": micro[2],
                       "per_class_f1": {c: per[c][2] for c in classes},
                       "train_sentences": len(train_texts), "test_sentences": len(test_texts),
                       "total_corpus": len(texts), "trained": "2026-09-06", "seed": 20260906,
                       "corpus": "template-generated trilingual (Bangla/Banglish/English) multi-label, disclosed synthetic"},
          "description": "Streaming SGD (log-loss) over FeatureHashed char_wb(2,3) ngrams, 30 case types, "
                         "trilingual. Suggests case types only; verbatim statutes + human judge decide."}
TOP = 4000
for ci, c in enumerate(classes):
    row = coefs[ci]
    idxs = np.argsort(-np.abs(row))[:TOP]
    export["classes"][c] = {"threshold": per_thr[c], "intercept": round(float(intercepts[ci]), 4),
                             "weights": {str(int(j)): round(float(row[j]), 4) for j in idxs if abs(row[j]) > 0.02}}
json.dump(export, open("case_model.json", "w", encoding="utf-8"), ensure_ascii=False)
import os
print(f"SAVED case_model.json — {os.path.getsize('case_model.json')/1048576:.2f} MB · G1={G1} G2={G2}", flush=True)
