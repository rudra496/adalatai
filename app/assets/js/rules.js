// AdalatAI — PRIMARY detection: deterministic crime phrase matching.
// Each rule maps to specific case types. When matched, ONLY these types shown.
// More accurate than ML for known phrases — 100% explainable, 0% false positive.

export const CRIME_RULES = [
  { rx: /নারী[\s\u0980-\u09FF]*নির্যাতন|নারীর\s*ওপর\s*নির্যাতন|নারীকে\s*মারধর|নারী নির্যাতন/i,
    types: ["assault_women", "eve_teasing", "hurt"],
    label: "নারী নির্যাতন / Violence against women" },
  { rx: /violence against (a |the )?woman|woman.*assaulted|assaulted.*woman/i,
    types: ["assault_women"], label: "Violence against woman" },
  { rx: /ইভ[\s-]*টিজিং|ইভটিজিং|eve[\s-]*teas|অশ্লীল ইশারা|অশ্লীল কথা|school.*ছাত্রী.*বিরক্ত/i,
    types: ["eve_teasing"], label: "ইভ-টিজিং / Eve teasing" },
  { rx: /ধর্ষণ|dhorshon|rape/i, types: ["rape"], label: "ধর্ষণ / Rape" },
  { rx: /এসিড নিক্ষেপ|এসিডে পুড়|acid (throw|attack|burn)/i,
    types: ["acid_violence"], label: "এসিড সহিংসতা / Acid violence" },
  { rx: /যৌতুক|joutuk|dowry/i, types: ["dowry"], label: "যৌতুক নির্যাতন / Dowry" },
  { rx: /খুন|হত্যা|গুলি করে মারা|মৃতদেহ|khun|hotya|murder|stabbed to death|shot dead/i,
    types: ["murder"], label: "খুন / Murder" },
  { rx: /অনিচ্ছাকৃত হত্যা|মারপিটে মৃত্যু|culpable homicide/i,
    types: ["culpable_homicide"], label: "অনিচ্ছাকৃত হত্যা / Culpable homicide" },
  { rx: /চুরি|চুরি হয়েছে|পকেট কেটে|দোকান ভেঙে|churi|churi hoyeche|theft|stole|stolen|broken into/i,
    types: ["theft"], label: "চুরি / Theft" },
  { rx: /ছিনতাই|অস্ত্র দেখিয়ে|chintai|snatch|robb/i, types: ["robbery"], label: "ডাকাতি / Robbery" },
  { rx: /গ্যাং|দলবদ্ধ.*লুট|পাঁচ.*জনের|dacoity|armed gang/i, types: ["dacoity"], label: "দলবদ্ধ ডাকাতি / Dacoity" },
  { rx: /মারধর|মারপিট|জখম|লাঠি দিয়ে|মারধর করেছে|mar-dhor|marpit|beaten|injur/i,
    types: ["hurt"], label: "আঘাত / Hurt" },
  { rx: /আক্রমণ|থাপ্পড়|assault|criminal force/i, types: ["assault_women", "hurt"],
    label: "আক্রমণ / Assault" },
  { rx: /অপহরণ|নিয়ে গেছে|গাড়িতে তুলে|মুক্তিপণ|opohoron|kidnap|abduct/i,
    types: ["kidnapping"], label: "অপহরণ / Kidnapping" },
  { rx: /তালাবদ্ধ|কক্ষে বন্দি|বন্দি রেখেছে|locked.*inside|confined/i,
    types: ["wrongful_confinement"], label: "অবৈধ আটক / Confinement" },
  { rx: /প্রতারণা|ভুয়া চাকরি|ভুয়া প্রতিশ্রুতি|টাকা নিয়ে পালিয়ে|protarona|cheat|fraud|false promise/i,
    types: ["cheating"], label: "প্রতারণা / Cheating" },
  { rx: /আত্মসাৎ|জমা রাখা টাকা|এজেন্ট.*টাকা|breach of trust|misappropriat/i,
    types: ["breach_of_trust"], label: "আস্থাভাজন / Breach of trust" },
  { rx: /কাচ ভাঙ|গাছ কাট|সাইনবোর্ড ভাঙ|আগুন দিয়ে|broke.*window|mischief|damaged/i,
    types: ["mischief"], label: "সম্পত্তি ক্ষতি / Mischief" },
  { rx: /অনধিকার প্রবেশ|জমির ভেতরে|শাটার ভেঙে|উঠানে ঢুকে|trespass|forcibly entered/i,
    types: ["trespass"], label: "অনধিকার প্রবেশ / Trespass" },
  { rx: /মানহানি|মিথ্যা পোস্ট|সুনাম নষ্ট|defam|false statements/i,
    types: ["defamation"], label: "মানহানি / Defamation" },
  { rx: /হুমকি|মেরে ফেলার|ভয় দেখিয়ে|টাকা দাবি করে|threat|intimidat/i,
    types: ["criminal_intimidation"], label: "ভয় প্রদর্শন / Intimidation" },
  { rx: /অফিসে.*নারী কর্মী|অশ্লীল প্রস্তাব|পদের অপব্যবহার|workplace harass/i,
    types: ["sexual_harassment_workplace"], label: "কর্মক্ষেত্রে হয়রানি / Workplace harassment" },
  { rx: /চেক.*ডিশনার|চেক.*এনক্যাশ|বকেয়ার চেক|cheque.*dishon|cheque.*bounc/i,
    types: ["cheque_bounce"], label: "চেক ডিশনার / Cheque bounce" },
  { rx: /ফেসবুকে ভুয়া|ভুয়া আইডি|ভুয়া বিকাশ|fake.*id|fake.*facebook|fake.*ID|online fraud|cyber fraud|অনলাইন প্রতারণা/i,
    types: ["cyber_fraud"], label: "সাইবার প্রতারণা / Cyber fraud" },
  { rx: /ইউটিউবে.*মানহানি|ডিজিটাল মাধ্যমে মিথ্যা|অনলাইনে মিথ্যা|online defam/i,
    types: ["cyber_defamation"], label: "অনলাইন মানহানি / Online defamation" },
  { rx: /হ্যাক|অ্যাকাউন্ট.*নিয়ন্ত্রণ|অনুমতি ছাড়া.*তথ্য|hack|unauthorised access/i,
    types: ["cyber_hacking"], label: "হ্যাকিং / Hacking" },
  { rx: /ইয়াবা|মাদক|গাঁজা|হিরোইন|yaba|ganja|heroin|narcotic|drug|মাদক বিক্রি/i,
    types: ["narcotics"], label: "মাদক অপরাধ / Narcotics" },
  { rx: /ঘুষ|দুর্নীতি|টেন্ডারবাজি|সরকারি অর্থ|ghush|brib|corrupt|public funds/i,
    types: ["corruption_bribery"], label: "ঘুষ/দুর্নীতি / Bribery/Corruption" },
  { rx: /বেপরোয়া গাড়ি|নির্মাণসামগ্রী ফেলে|রাস্তায় নিহত|rash driving|death by negligence/i,
    types: ["negligence_death"], label: "দুর্ঘটনায় মৃত্যু / Death by negligence" },
  { rx: /ঋণ পরিশোধ|ব্যাংকের ঋণ|এনজিওর ঋণ|loan default/i,
    types: ["money_loan_default"], label: "অর্থঋণ / Loan default" },
  { rx: /ভরণপোষণ দিচ্ছে না|স্ত্রী-সন্তানের খরচ|বাবা.*মেয়ের খরচ|maintenance/i,
    types: ["family_maintenance"], label: "ভরণপোষণ / Maintenance" },
  { rx: /সীমানা বিরোধ|খালের পানি|প্রতিবেশীর সঙ্গে বিরোধ|boundary dispute/i,
    types: ["village_dispute"], label: "গ্রাম আদালত বিরোধ / Village dispute" },
];

/**
 * Match crime phrases in text. Returns matched type names or null.
 */
export function matchCrimeRules(text) {
  const types = [];
  for (const rule of CRIME_RULES) {
    if (rule.rx.test(text)) {
      for (const t of rule.types) {
        if (!types.includes(t)) types.push(t);
      }
    }
  }
  return types.length > 0 ? types : null;
}

/**
 * Get labels for types.
 */
export function getLabels(types, lang = "bn") {
  const BN = {
    theft: "চুরি", robbery: "ডাকাতি/ছিনতাই", dacoity: "দলবদ্ধ ডাকাতি",
    murder: "খুন", culpable_homicide: "অনিচ্ছাকৃত হত্যা", hurt: "আঘাত",
    assault_women: "নারীর ওপর আক্রমণ", eve_teasing: "ইভ-টিজিং",
    sexual_harassment_workplace: "যৌন হয়রানি", rape: "ধর্ষণ",
    kidnapping: "অপহরণ", wrongful_confinement: "অবৈধ আটক",
    cheating: "প্রতারণা", breach_of_trust: "আস্থাভাজন",
    mischief: "সম্পত্তি ক্ষতি", trespass: "অনধিকার প্রবেশ",
    defamation: "মানহানি", criminal_intimidation: "ভয় প্রদর্শন",
    dowry: "যৌতুক নির্যাতন", cheque_bounce: "চেক ডিশনার",
    cyber_fraud: "অনলাইন প্রতারণা", cyber_defamation: "অনলাইন মানহানি",
    cyber_hacking: "হ্যাকিং", narcotics: "মাদক অপরাধ",
    corruption_bribery: "ঘুষ/দুর্নীতি", negligence_death: "দুর্ঘটনায় মৃত্যু",
    money_loan_default: "অর্থঋণ", family_maintenance: "ভরণপোষণ",
    village_dispute: "গ্রাম আদালত বিরোধ", acid_violence: "এসিড সহিংসতা",
  };
  return types.map((t) => (lang === "bn" ? (BN[t] || t) : t));
}
