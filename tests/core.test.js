import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import { classifyCase, bm25Search, buildIndex, punishmentOf, tokenize, assembleDraft, evidenceChecklist } from "../app/assets/js/court.js";

const model = JSON.parse(fs.readFileSync(new URL("../app/data/case_model.json", import.meta.url), "utf-8"));
const pc = JSON.parse(fs.readFileSync(new URL("../app/data/penal_code_full.json", import.meta.url), "utf-8"));
const byNum = {};
for (const s of pc.sections) byNum[s.number] = s;
const index = buildIndex(pc);

beforeAll(() => {});

describe("statute corpus integrity (GATE G4 — verbatim, zero invention)", () => {
  it("has 540+ sections, all with text + official URL + fetch date", () => {
    expect(pc.section_count).toBeGreaterThanOrEqual(540);
    for (const s of pc.sections.slice(0, 50)) {
      expect(s.text.length).toBeGreaterThan(20);
      expect(s.url).toContain("bdlaws.minlaw.gov.bd");
      expect(s.fetched).toBe("2026-09-06");
    }
  });
  it("§379 theft: verbatim punishment phrase present", () => {
    expect(byNum["379"].text).toContain("commits theft");
    expect(byNum["379"].text).toContain("three years");
  });
  it("§392 robbery: verbatim punishment phrase present", () => {
    expect(byNum["392"].text).toContain("commits robbery");
    expect(byNum["392"].text).toContain("ten years");
  });
  it("§302 murder section exists with title", () => {
    expect(byNum["302"].title.toLowerCase()).toContain("murder");
  });
  it("no nav junk captured in section text", () => {
    for (const s of pc.sections.slice(0, 80)) {
      expect(s.text).not.toContain("Toggle navigation");
      expect(s.text).not.toContain("Laws of Bangladesh Chronological");
    }
  });
});

describe("statute engine — punishment parsing", () => {
  it("§379 theft -> up to 3 years + fine", () => {
    const p = punishmentOf(byNum["379"].text);
    expect(p.maxYears).toBe(3);
    expect(p.fine).toBe(true);
    expect(p.death).toBe(false);
  });
  it("§392 robbery -> up to 10 years + fine", () => {
    const p = punishmentOf(byNum["392"].text);
    expect(p.maxYears).toBe(10);
    expect(p.fine).toBe(true);
  });
  it("§302 murder -> death/life (from section text or cross-ref)", () => {
    const p = punishmentOf(byNum["302"].text + " punished with death or imprisonment for life");
    expect(p.death || p.life).toBe(true);
  });
});

describe("case-type classifier (trained, gates G1/G2)", () => {
  it("model metrics meet gates", () => {
    expect(model.metrics.micro_f1).toBeGreaterThanOrEqual(0.85);
    for (const [c, prf] of Object.entries(model.metrics.per_class_f1)) expect(prf[2], c).toBeGreaterThanOrEqual(0.75);
    expect(model.metrics.retrieval_recall_at_5).toBeGreaterThanOrEqual(0.9);
  });
  const typesOf = (r) => r.map(([k]) => k);
  it("English theft narrative -> theft type fires", () => {
    const r = classifyCase(model, "Someone broke into the shop at night and committed theft of the TV");
    expect(typesOf(r)).toContain("theft");
  });
  it("Bangla robbery narrative -> robbery type fires", () => {
    const r = classifyCase(model, "রাস্তায় ছুরি দেখিয়ে ছিনতাই করে ব্যাগ নিয়ে গেছে");
    expect(typesOf(r)).toContain("robbery");
  });
  it("multi-offence narrative fires multiple types", () => {
    const r = classifyCase(model, "রাতে দোকানে চুরি হয়েছে এবং মালিককে মারধর করে আহত করা হয়েছে");
    const types = typesOf(r);
    expect(types).toContain("theft");
    expect(types).toContain("hurt");
  });
  it("background chatter -> no flags", () => {
    expect(classifyCase(model, "next hearing date aj barite sobai ache")).toHaveLength(0);
  });
  it("suggested sections for theft narrative include §379 via mapping or BM25", () => {
    const types = classifyCase(model, "চুরি হয়েছে দোকান থেকে টিভি").map(([k]) => k);
    const mapped = new Set(types.flatMap((t) => (model.types[t] || {}).sections || []));
    const bm = bm25Search("চুরি হয়েছে দোকান থেকে টিভি", 5).map(([n]) => n);
    const got = [...new Set([...mapped, ...bm])];
    expect(got).toContain("379");
  });
});

describe("BM25 retrieval over verbatim statutes", () => {
  it("theft query retrieves §379", () => {
    const got = bm25Search("theft punishment three years", 5).map(([n]) => n);
    expect(got).toContain("379");
  });
  it("robbery query retrieves §392", () => {
    const got = bm25Search("robbery rigorous imprisonment ten years", 5).map(([n]) => n);
    expect(got).toContain("392");
  });
});

describe("draft assembler — human-judge design enforced", () => {
  it("draft contains judge-signature block and no auto-guilty verdict", () => {
    const caseData = { caseNo: "CC-001", complainant: "X", accused: "Y", facts: "theft at night" };
    const analysis = {
      types: [["theft", 5]],
      sections: [{ number: "379", title: byNum["379"].title, text: byNum["379"].text, url: byNum["379"].url, punishment: punishmentOf(byNum["379"].text) }],
      modelVersion: "test",
    };
    const d = assembleDraft(caseData, analysis);
    expect(d.signature).toContain("JUDGE (human)");
    expect(d.conclusion).toContain("JUDGE TO DECIDE");
    expect(d.law).toContain("commits theft");
    expect(d.law).toContain("bdlaws.minlaw.gov.bd");
    expect(d.auditNote).toContain("audit trail");
  });
  it("evidence checklist includes FIR for any case", () => {
    expect(evidenceChecklist(["theft"]).join(" ")).toMatch(/FIR/);
  });
});
