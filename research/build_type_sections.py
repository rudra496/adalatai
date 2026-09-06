#!/usr/bin/env python3
"""Build case_sections.json — maps each case type to its real statutory sections
(pulled VERBATIM from penal_code_full.json + acts_multi.json), Bangla numerals
normalised for matching. Output feeds the app's statute engine."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def norm_num(n):
    return str(n).translate(BN_DIGITS).strip()

pc = json.load(open(os.path.join(HERE, "..", "app", "data", "penal_code_full.json"), encoding="utf-8"))
multi = json.load(open(os.path.join(HERE, "..", "statutes", "acts_multi.json"), encoding="utf-8"))

by_key = {}
for s in pc["sections"]:
    by_key[("PC", norm_num(s["number"]))] = s
for act_id, act in multi.items():
    for s in act["sections"]:
        by_key[(act_id, norm_num(s["number"]))] = {**s, "act_title": act["title"]}

def pick(act_id, numbers):
    out = []
    for n in numbers:
        s = by_key.get((act_id, norm_num(n)))
        if s:
            out.append({
                "act_id": act_id,
                "act": s.get("act_title", pc["act"] if act_id == "PC" else ""),
                "number": s["number"], "title": s["title"], "text": s["text"], "url": s["url"],
            })
    return out

MAPPING = {
    "theft": pick("PC", ["379", "380"]),
    "robbery": pick("PC", ["390", "392", "394", "397"]),
    "dacoity": pick("PC", ["391", "395", "396"]),
    "murder": pick("PC", ["300", "302", "201", "204"]),
    "culpable_homicide": pick("PC", ["299", "304"]),
    "hurt": pick("PC", ["323", "324", "325", "326"]),
    "assault_women": pick("PC", ["354"]),
    "eve_teasing": pick("PC", ["509"]) + pick("835", ["10"]),
    "rape": pick("PC", ["375", "376"]) + pick("835", ["9"]),
    "kidnapping": pick("PC", ["359", "363", "365", "366"]),
    "wrongful_confinement": pick("PC", ["340", "342", "343"]),
    "cheating": pick("PC", ["415", "417", "420"]),
    "breach_of_trust": pick("PC", ["405", "406", "409"]),
    "mischief": pick("PC", ["425", "426", "427", "435"]),
    "trespass": pick("PC", ["441", "442", "447", "448"]),
    "defamation": pick("PC", ["499", "500"]),
    "criminal_intimidation": pick("PC", ["503", "506"]),
    "sexual_harassment_workplace": pick("PC", ["509"]) + pick("835", ["10"]),
    "dowry": pick("PC", ["498A"]) + pick("607", ["3", "4"]) + pick("1256", ["3", "4"]),
    "cheque_bounce": pick("46", ["138", "141", "142"]),
    "cyber_fraud": pick("PC", ["420"]) + pick("1261", ["24", "30"]),
    "cyber_defamation": pick("PC", ["500"]) + pick("1261", ["25", "29"]),
    "cyber_hacking": pick("1261", ["15", "17", "18"]),
    "narcotics": pick("1276", ["5", "6", "7", "9", "10"]),
    "corruption_bribery": pick("PC", ["161", "163", "165", "409"]) + pick("914", ["27"]),
    "negligence_death": pick("PC", ["304A"]) + pick("654", ["105", "106"]),
    "money_loan_default": pick("901", ["5(1)", "5"]),
    "family_maintenance": pick("682", ["5"]) + pick("835", ["6"]),
    "village_dispute": pick("938", ["8"]),
    "acid_violence": pick("883", ["4"]),
}

out = {
    "source": "Section texts verbatim from bdlaws.minlaw.gov.bd (penal_code_full.json + acts_multi.json), fetched 2026-09-06",
    "mapping": {k: v for k, v in MAPPING.items()},
}
missing = {k: [f"{a}:§{n}" for a, nums in [] for n in nums] for k, v in []}
report = []
for k, v in MAPPING.items():
    if not v:
        report.append((k, "EMPTY"))
json.dump(out, open(os.path.join(HERE, "..", "app", "data", "case_sections.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print("case_sections.json written")
for k, v in MAPPING.items():
    print(f"  {k}: {len(v)} sections")
print("empty mappings:", report)
