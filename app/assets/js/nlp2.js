// AdalatAI NLP v2 — TF-IDF(char_wb 2-3) + MultinomialNB inference in pure JS.
// Replicates research/train_v2.py's Colab export (case_model.json):
//   score = Σ_ngrams tfidf(ngram) × weight_class(ngram)  →  fire if > class threshold
// tfidf(ngram) = (1 + ln(count)) × idf(ngram), then L2-normalised over the input vector.

let MODEL2 = null;

export async function loadModel2() {
  if (MODEL2) return MODEL2;
  MODEL2 = await (await fetch("data/case_model.json")).json();
  return MODEL2;
}

/** Replicates TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), lowercase=True). */
export function charWbNgrams(text, lo = 2, hi = 3) {
  const words = String(text || "").toLowerCase().replace(/[^\w\u0980-\u09FF]+/g, " ").split(/\s+/).filter(Boolean);
  const out = [];
  for (const w of words) {
    const padded = " " + w + " ";
    const L = padded.length;
    if (L <= lo) { out.push(padded); continue; }
    const nMax = Math.min(hi, L - 1);
    for (let n = lo; n <= nMax; n++)
      for (let i = 0; i + n <= L; i++) out.push(padded.substr(i, n));
    if (L - 1 < lo) out.push(padded.trim());
  }
  return out;
}

/** @returns [{type, score}] — score is the calibrated decision value (higher = stronger) */
export function classifyV2(model, text) {
  const grams = charWbNgrams(text);
  const counts = {};
  for (const g of grams) counts[g] = (counts[g] || 0) + 1;
  const idf = model.idf;
  const thr = model.config.thresholds || {};
  const out = [];
  for (const [cls, obj] of Object.entries(model.classes)) {
    let s = 0, norm = 0;
    for (const [g, cnt] of Object.entries(counts)) {
      const i = idf[g];
      if (i === undefined) continue;
      const tfidf = (1 + Math.log(cnt)) * i;
      norm += tfidf * tfidf;
      const w = obj.weights[g];
      if (w !== undefined) s += tfidf * w;
    }
    norm = Math.sqrt(norm) || 1;
    s /= norm;                                   // L2 — matches TfidfVectorizer norm
    if (s > (obj.threshold ?? thr[cls] ?? 0.5)) out.push({ type: cls, score: Math.round(s * 100) / 100 });
  }
  return out.sort((a, b) => b.score - a.score);
}

export async function classifyV2Loaded(text) {
  return classifyV2(await loadModel2(), text);
}
