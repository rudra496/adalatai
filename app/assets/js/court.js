// AdalatAI core engine — pure functions, testable. Port of research/train_case_models.py.
// RULE: AI suggests; the human judge approves. Every AI output is logged to the audit trail.

let CASE_MODEL = null, STATUTES = null;

export async function loadCaseModel() {
  if (CASE_MODEL) return CASE_MODEL;
  CASE_MODEL = await (await fetch("data/case_model.json")).json();
  return CASE_MODEL;
}
export async function loadStatutes() {
  if (STATUTES) return STATUTES;
  STATUTES = await (await fetch("data/penal_code_full.json")).json();
  return STATUTES;
}

export function tokenize(text) {
  const t = String(text || "").toLowerCase().replace(/[^\w\u0980-\u09FF]+/g, " ");
  const w = t.split(/\s+/).filter(Boolean);
  const grams = w.slice();
  for (let i = 0; i < w.length - 1; i++) grams.push(w[i] + "_" + w[i + 1]);
  return grams;
}

/** Classify a case narrative -> {type: score} above threshold. */
export function classifyCase(model, text, threshold) {
  const thr = threshold ?? (model.config.threshold || 1.0);
  const toks = new Set(tokenize(text));
  const out = {};
  for (const [cls, obj] of Object.entries(model.classes)) {
    let s = 0;
    for (const t of toks) if (obj.weights[t] !== undefined) s += obj.weights[t];
    if (s > thr) out[cls] = Math.round(s * 100) / 100;
  }
  return Object.entries(out).sort((a, b) => b[1] - a[1]);
}

/** BM25 retrieval over the statute corpus (index built once). */
let INDEX = null;
export function buildIndex(statutes) {
  const docs = {};
  for (const s of statutes.sections) docs[s.number] = tokenize(s.title + " " + s.text);
  const df = {};
  for (const toks of Object.values(docs)) for (const t of new Set(toks)) df[t] = (df[t] || 0) + 1;
  const N = Object.keys(docs).length;
  const avgdl = Object.values(docs).reduce((a, d) => a + d.length, 0) / N;
  INDEX = { docs, df, N, avgdl };
  return INDEX;
}
export function bm25Search(query, k = 5, k1 = 1.4, b = 0.75) {
  if (!INDEX) throw new Error("index not built");
  const q = tokenize(query);
  const { docs, df, N, avgdl } = INDEX;
  const scores = [];
  for (const [num, toks] of Object.entries(docs)) {
    const tf = {};
    for (const t of toks) tf[t] = (tf[t] || 0) + 1;
    let sc = 0;
    for (const t of q) {
      if (!tf[t]) continue;
      const idf = Math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1);
      sc += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * toks.length / avgdl));
    }
    if (sc > 0) scores.push([num, sc]);
  }
  return scores.sort((a, b) => b[1] - a[1]).slice(0, k);
}

const WORDNUM = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8,
  nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
  twenty: 20, thirty: 30, forty: 40, fifty: 50 };

/** Deterministic statute engine: parse the punishment range from VERBATIM statute text. */
export function punishmentOf(text) {
  const t = String(text || "").toLowerCase();
  const out = { death: false, life: false, maxYears: null, fine: false, notes: [] };
  if (/shall be punished with death|punished with death/.test(t)) out.death = true;
  if (/imprisonment for life/.test(t)) out.life = true;
  if (/fine/.test(t)) out.fine = true;
  const yrs = [...t.matchAll(/(?:extend to|exceeding)\s+(\w+|\d+)\s+years?/g)];
  for (const m of yrs) {
    const n = /^\d+$/.test(m[1]) ? parseInt(m[1]) : WORDNUM[m[1]];
    if (n && (out.maxYears === null || n > out.maxYears)) out.maxYears = n;
  }
  if (/rigorous imprisonment/.test(t)) out.notes.push("rigorous imprisonment");
  if (!out.death && !out.life && out.maxYears === null && !out.fine) out.notes.push("no standard punishment phrase matched — verify manually");
  return out;
}

/** Full AI pre-analysis. v2 (TF-IDF trained, 30 types, trilingual) first; v1 fallback. */
export async function analyzeCase(text) {
  let v2 = null;
  try { v2 = await (await fetch("data/case_model.json")).json(); } catch {}
  if (v2 && v2.config && v2.config.analyzer) {
    const nlp2 = await import("./nlp2.js");
    const statutes = await loadStatutes();
    if (!INDEX) buildIndex(statutes);
    const byNum = {};
    for (const s of statutes.sections) byNum[s.number] = s;
    let secMap = {};
    try {
      const cs = await (await fetch("data/case_sections.json")).json();
      secMap = cs.mapping || {};
    } catch {}
    const preds = nlp2.classifyV2(v2, text);
    const types = preds.map((p) => [p.type, p.score]);
    const outSecs = [];
    const seen = new Set();
    for (const [ct] of types) {
      for (const s of secMap[ct] || []) {
        if (!seen.has(s.act + s.number)) {
          seen.add(s.act + s.number);
          outSecs.push({ number: s.number, title: s.title, text: s.text, url: s.url,
                         punishment: punishmentOf(s.text), act: s.act });
        }
      }
    }
    for (const [n] of bm25Search(text, 4)) {
      if (!seen.has("PC" + n) && byNum[n]) {
        seen.add("PC" + n);
        outSecs.push({ number: n, title: byNum[n].title, text: byNum[n].text, url: byNum[n].url,
                       punishment: punishmentOf(byNum[n].text), act: "PC" });
      }
    }
    return { types, sections: outSecs.slice(0, 7),
             evidence: evidenceChecklist(types.map(([t]) => t)),
             modelVersion: "case_model-v2 (TF-IDF+NB, " + (v2.metrics?.total_corpus || "1.1M") + " sentences)" };
  }
  // ---------- v1 fallback ----------
  const model = await loadCaseModel();
  const statutes = await loadStatutes();
  if (!INDEX) buildIndex(statutes);
  const byNum = {};
  for (const s of statutes.sections) byNum[s.number] = s;
  const types = classifyCase(model, text);
  const mapped = new Set();
  for (const [ct] of types) for (const n of (model.types[ct] || {}).sections || []) mapped.add(n);
  const bm = bm25Search(text, 6);
  const sections = [...new Set([...mapped, ...bm.map(([n]) => n)])].slice(0, 6)
    .filter((n) => byNum[n])
    .map((n) => {
      const s = byNum[n];
      return { number: n, title: s.title, text: s.text, url: s.url, punishment: punishmentOf(s.text) };
    });
  const evidence = evidenceChecklist(types.map(([t]) => t));
  return { types, sections, evidence, modelVersion: model.model + " · it" + (model.metrics?.iterations || "?") };
}

export function evidenceChecklist(types) {
  const base = ["FIR / complaint copy (এফআইআর)", "Witness statement(s) (সাক্ষীর বয়ান)", "Case number & court name"];
  const extra = {
    theft: ["Stolen-property list (চুরি হওয়া সম্পত্তির তালিকা)", "Ownership proof (মালিকানা দলিল)"],
    robbery: ["Injury/medical report (আহত হলে মেডিকেল রিপোর্ট)", "Weapon description (অস্ত্রের বিবরণ)"],
    murder: ["Post-mortem report (ময়নাতদন্ত রিপোর্ট)", "Death certificate (মৃত্যু সনদ)"],
    hurt: ["Medical report (মেডিকেল রিপোর্ট)"],
    cheating: ["Transaction records (লেনদেনের হিসাব)", "False-promise evidence (প্রতারণার প্রমাণ)"],
    breach_of_trust: ["Entrustment proof (আস্থায় অর্পণের প্রমাণ)", "Accounts/ledger (হিসাবের খাতা)"],
    defamation: ["Screenshot/post copy (পোস্ট/স্ক্রিনশট)"],
    rape: ["Medical examination report (মেডিকেল পরীক্ষা রিপোর্ট)", "Statement under §164 CrPC"],
    kidnapping: ["Missing-person complaint (নিখোঁজ অভিযোগ)", "Age proof of victim (বয়স প্রমাণ)"],
  };
  return [...new Set([...base, ...types.flatMap((t) => extra[t] || [])])];
}

/** Judgment DRAFT assembler — assembled ONLY from verified parts: case facts,
 *  verbatim statute quotes, statutory punishment ranges. The judge edits + signs. */
export function assembleDraft(caseData, analysis) {
  const statuteBlocks = analysis.sections.map((s) => {
    const p = s.punishment;
    const rng = p.death ? "death, or imprisonment for life"
      : p.life ? "imprisonment for life"
      : p.maxYears ? `imprisonment up to ${p.maxYears} year(s)${p.fine ? " and fine" : ""}`
      : p.fine ? "fine" : "(as per statute — verify)";
    return `Section ${s.number} — ${s.title}\nStatute (verbatim): "${s.text}"\nStatutory range: ${rng}\nSource: ${s.url}`;
  }).join("\n\n");
  return {
    header: `IN THE COURT OF [COURT NAME]\nCriminal Case No. ${caseData.caseNo || "____"} of ${new Date().getFullYear()}\nComplainant: ${caseData.complainant || "____"}\nAccused: ${caseData.accused || "____"}`,
    facts: `Brief facts (as narrated): ${caseData.facts}`,
    issues: `Issues for determination:\n1. Whether the acts alleged constitute the offence(s) under the section(s) cited below.\n2. What punishment, if any, within the statutory range, is warranted.`,
    law: `Applicable law (AI-suggested, judge to verify):\n\n${statuteBlocks}`,
    conclusion: `Conclusion (JUDGE TO DECIDE AND EDIT — this system does not decide guilt or sentence):\nThe court finds [GUILTY / NOT GUILTY] and sentences [WITHIN THE STATUTORY RANGE ABOVE, AS DETERMINED BY THE JUDGE].`,
    signature: "____________________\nJUDGE (human) — final approval and signature",
    auditNote: `Draft assembled by AdalatAI (${analysis.modelVersion}) from verbatim statute text (bdlaws.minlaw.gov.bd). Every AI suggestion is logged in the audit trail and requires human-judge approval. This document is a working draft, not a judicial order, until signed by the judge.`,
  };
}
