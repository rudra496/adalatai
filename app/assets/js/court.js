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

// Deterministic keyword→type expansion (auditable rules — no retraining needed for
// common Bangla/English crime phrasings). Complements the trained classifier.
const SYNONYMS = [
  // --- violence against women / eve teasing / sexual offences ---
  [/নারी[\sঀ-৿]*নির্যাতন|নারীর ওপর নির্যাতন|নারীকে মারধর|নারী নির্যাতন/i, ["assault_women", "eve_teasing", "hurt"]],
  [/violence against (a |the )?woman| assaulted (a |the )?woman/i, ["assault_women"]],
  [/ইভ[\s-]*টিজিং|ইভটিজিং|eve[\s-]*teas|অশ্লীল ইশারা|অশ্লীল কথা বলে|অশ্লীল মেসেজ/i, ["eve_teasing"]],
  [/যৌন হয়রানি|অফিসে.*নারী কর্মী|workplace harass/i, ["sexual_harassment_workplace"]],
  [/ধর্ষণ|ধর্ষণ করে|rape/i, ["rape"]],
  [/এসিড নিক্ষেপ|এসিডে পুড়ি|acid (throw|attack|burn)/i, ["acid_violence"]],
  [/যৌতুক|যৌতুকের জন্য|dowry/i, ["dowry"]],
  // --- homicide ---
  [/খুন|হত্যা|গুলি করে মারা|ছুরি মেরে মারা|murder|stabbed to death|shot dead/i, ["murder"]],
  [/অনিচ্ছাকৃত হত্যা|মারপিটে মৃত্যু|culpable homicide/i, ["culpable_homicide"]],
  [/দুর্ঘটনায় মৃত্যু|বেপরোয়া গাড়ি|রাস্তায় নিহত|rash driving|accident.*death/i, ["negligence_death"]],
  // --- physical harm ---
  [/মারধর|মারপিট|জখম|লাঠি দিয়ে|ছুরি মেরে আহত|beaten|injur/i, ["hurt", "assault_women"]],
  [/আক্রমণ|থাপ্পড়|assault|criminal force/i, ["assault_women", "hurt"]],
  // --- property crimes ---
  [/চুরি|চুরি হয়েছে|পকেট কেটে|theft|stole|stolen/i, ["theft"]],
  [/ছিনতাই|ছুরি দেখিয়ে নিয়েছে|snatch|robb/i, ["robbery"]],
  [/গ্যাং|দলবদ্ধ.*লুট|পাঁচ.*জনের|dacoity/i, ["dacoity"]],
  [/ভেঙে ক্ষতি|গাছ কেটে|কাচ ভাঙ|আগুন দিয়ে|mischief|damaged.*property/i, ["mischief"]],
  [/অনধিকার প্রবেশ|জমির ভেতরে|জোর করে ঢুকে|trespass|forcibly entered/i, ["trespass"]],
  // --- fraud/trust ---
  [/প্রতারণা|ভুয়া চাকরি|ভুয়া প্রতিশ্রুতি|cheat|fraud|false promise/i, ["cheating"]],
  [/আত্মসাৎ|আস্থাভাজন|জমা রাখা টাকা|breach of trust|misappropriat/i, ["breach_of_trust"]],
  // --- person crimes ---
  [/অপহরণ|নিয়ে গেছে|গাড়িতে তুলে|kidnap|abduct/i, ["kidnapping"]],
  [/তালাবদ্ধ|কক্ষে বন্দি|বন্দি রেখেছে|wrongful confinement|locked.*inside/i, ["wrongful_confinement"]],
  [/মানহানি|মিথ্যা পোস্ট|সুনাম নষ্ট|defam|false statements/i, ["defamation"]],
  [/হুমকি|ভয় দেখিয়ে|মেরে ফেলার হুমকি|threat|intimidat/i, ["criminal_intimidation"]],
  // --- modern crimes ---
  [/অনলাইন.*প্রতারণা|ফেসবুকে ভুয়া|ভুয়া বিকাশ|অনলাইন শপিং|fake (facebook )?id|online (shopping )?fraud/i, ["cyber_fraud"]],
  [/অনলাইনে মিথ্যা|ডিজিটাল মাধ্যমে মিথ্যা|ইউটিউবে মানহানি|online defam|upload.*defam/i, ["cyber_defamation"]],
  [/হ্যাক|অ্যাকাউন্ট হ্যাক|অনুমতি ছাড়া ঢুকে তথ্য|hack|unauthorised access/i, ["cyber_hacking"]],
  // --- regulated goods ---
  [/ইয়াবা|মাদক|গাঁজা|হিরোইন|yaba|ganja|heroin|narcotic|drug/i, ["narcotics"]],
  [/ঘুষ|দুর্নীতি|টেন্ডারবাজি|brib|corrupt|public funds/i, ["corruption_bribery"]],
  // --- financial/civil ---
  [/চেক.*ডিশনার|চেক.*এনক্যাশ|cheque.*dishon|cheek.*dishon/i, ["cheque_bounce"]],
  [/ঋণ পরিশোধ|ব্যাংকের ঋণ|এনজিওর ঋণ|loan default|ঋণ নিয়ে পালিয়ে/i, ["money_loan_default"]],
  [/ভরণপোষণ দিচ্ছে না|স্ত্রী-সন্তানের খরচ|সন্তানের খরচ|maintenance/i, ["family_maintenance"]],
  [/সীমানা বিরোধ|খালের পানি|প্রতিবেশীর সঙ্গে বিরোধ|boundary dispute|neighbour dispute/i, ["village_dispute"]],
  // --- catch-alls ---
  [/মামলা করতে|মামলা দায়ের|file a case|অভিযোগ লিখুন/i, []],  // intent — no type
];

/** Expand user text with deterministic keyword rules. */
export function expandTypes(text, baseTypes) {
  const found = new Set(baseTypes.map(([t]) => t));
  for (const [rx, types] of SYNONYMS) {
    if (rx.test(text)) types.forEach((t) => found.add(t));
  }
  return [...found].map((t) => {
    const prev = baseTypes.find(([k]) => k === t);
    return [t, prev ? prev[1] : 1.0];
  });
}

/** Direct substring search over ALL 2,083 sections (Bangla acts included). */
let ALL_INDEX = null;
export async function searchAllSections(text, k = 6) {
  if (!ALL_INDEX) {
    ALL_INDEX = await (await fetch("data/all_sections.json")).json();
  }
  const words = String(text || "").toLowerCase()
    .split(/[^\wঀ-৿]+/).filter((w) => w.length >= 4);
  if (!words.length) return [];
  const scored = [];
  for (const s of ALL_INDEX.sections) {
    const title = s.title.toLowerCase();
    const body = s.text.toLowerCase();
    let sc = 0;
    for (const w of words) {
      if (title.includes(w)) sc += 3;
      if (body.includes(w)) sc += 1;
    }
    if (sc > 0) scored.push({ s, sc });
  }
  scored.sort((a, b) => b.sc - a.sc);
  return scored.slice(0, k).map((x) => ({
    number: x.s.number, title: x.s.title, text: x.s.text, url: x.s.url, act: x.s.act,
    punishment: punishmentOf(x.s.text),
  }));
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
    let preds = nlp2.classifyV2(v2, text);
    let types = preds.map((p) => [p.type, p.score]);
    // deterministic keyword expansion (auditable — see SYNONYMS table)
    types = expandTypes(text, types);
    if (!types.length) {
      // nothing detected: fall back to direct full-corpus search only
      const direct = await searchAllSections(text, 5);
      return { types: [], sections: direct,
               evidence: evidenceChecklist([]),
               modelVersion: "direct-search (no classifier match)" };
    }
    const outSecs = [];
    const seen = new Set();
    // Only pull sections from the TOP 2 types (noise reduction)
    const topTypes = types.slice(0, 2).map(([t]) => t);
    for (const ct of topTypes) {
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
    // direct search over ALL acts — only add if fewer than 5 from type mapping
    if (outSecs.length < 5) {
      const direct = await searchAllSections(text, 4);
      for (const dsec of direct) {
        if (!seen.has(dsec.act + dsec.number)) {
          seen.add(dsec.act + dsec.number);
          outSecs.push(dsec);
        }
      }
    }
    return { types, sections: outSecs.slice(0, 8),
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
