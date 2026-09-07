// AdalatAI NLP v3 — inference for the Colab-trained model (case_model.json).
// Pipeline: char_wb(2,3) ngrams -> murmurhash3_32(utf8, seed=0) % 262144 -> binary
// presence -> score(class) = Σ weights[class][bucket]; fire if score > threshold[class].
// Hash parity verified against sklearn.utils.murmurhash3_32 (vitest).

let MODEL2 = null;

export async function loadModel2() {
  if (MODEL2) return MODEL2;
  MODEL2 = await (await fetch("data/case_model.json")).json();
  return MODEL2;
}

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
  }
  return out;
}

// murmurhash3_32 (x86_32) over UTF-8 bytes — byte-exact match to
// sklearn.utils.murmurhash3_32 (which uses the canonical murmur3_x86_32 algorithm)
export function murmurhash3_32(key, seed = 0) {
  const data = new TextEncoder().encode(key);
  const len = data.length;
  const c1 = 0xcc9e2d51, c2 = 0x1b873593;
  let h1 = seed >>> 0;
  const nblocks = Math.floor(len / 4);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let i = 0; i < nblocks; i++) {
    let k1 = view.getUint32(i * 4, true);
    k1 = Math.imul(k1, c1) >>> 0;
    k1 = ((k1 << 15) | (k1 >>> 17)) >>> 0;
    k1 = Math.imul(k1, c2) >>> 0;
    h1 = (h1 ^ k1) >>> 0;
    h1 = ((h1 << 13) | (h1 >>> 19)) >>> 0;
    h1 = (Math.imul(h1, 5) + 0xe6546b64) >>> 0;
  }
  // tail: last (len & 3) bytes, XOR in ascending byte position shifts
  let k1 = 0;
  const tailStart = nblocks * 4;
  const rem = len - tailStart;
  if (rem >= 3) k1 ^= data[tailStart + 2] << 16;
  if (rem >= 2) k1 ^= data[tailStart + 1] << 8;
  if (rem >= 1) {
    k1 ^= data[tailStart];
    k1 = Math.imul(k1, c1) >>> 0;
    k1 = ((k1 << 15) | (k1 >>> 17)) >>> 0;
    k1 = Math.imul(k1, c2) >>> 0;
    h1 = (h1 ^ k1) >>> 0;
  }
  h1 = (h1 ^ len) >>> 0;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b) >>> 0;
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35) >>> 0;
  h1 ^= h1 >>> 16;
  return h1 | 0;   // signed int32 like sklearn
}

export function bucket(ngram, nFeatures) {
  const h = murmurhash3_32(ngram, 0);
  return ((h % nFeatures) + nFeatures) % nFeatures;
}

/** @returns [{type, score}] */
export function classifyV2(model, text) {
  const nFeatures = model.config.n_features;
  const thr = model.config.thresholds || {};
  const MIN_MATCH = model.config.min_match || 2;
  const grams = [...new Set(charWbNgrams(text))];
  const out = [];
  for (const [cls, obj] of Object.entries(model.classes)) {
    let s = 0, cnt = 0;
    for (const g of grams) {
      const b = bucket(g, nFeatures);
      const w = obj.weights[String(b)];
      if (w !== undefined && w > 0) { s += w; cnt++; }
    }
    const mean = cnt >= MIN_MATCH ? s / cnt : 0;
    const t = (model.config.thresholds || {})[cls] ?? 0.5;
    if (mean > t) out.push({ type: cls, score: Math.round(mean * 100) / 100 });
  }
  return out.sort((a, b) => b.score - a.score);
}

export async function classifyV2Loaded(text) {
  return classifyV2(await loadModel2(), text);
}
