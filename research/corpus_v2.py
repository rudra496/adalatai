#!/usr/bin/env python3
"""AdalatAI corpus v2 — 1M+ sentences, 32 case types, 3 scripts (Bangla, Banglish-
romanized, English). Template-generated (disclosed synthetic, fixed seed) with
multi-offence, negation, intent-background and noise naturalization."""
import json, random, os, re, sys

random.seed(20260906)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSONL = os.path.join(HERE, "corpus_v2.jsonl")

# phrase = {"bn":..., "ro":..., "en":...}  (ro = romanized Bangla / Banglish)
T = {}
def add_type(key, bn_label, en_label, sections, phrases, evidence_extra=None):
    T[key] = {"bn": bn_label, "en": en_label, "sections": sections, "phrases": phrases}

add_type("theft", "চুরি", "theft", ["379", "380"], [
 {"bn":"দোকান ভেঙে {item} চুরি হয়েছে","ro":"dokan bhenge {item} churi hoyeche","en":"Someone broke into the shop and committed theft of the {item}"},
 {"bn":"{name}-এর ব্যাগ থেকে মোবাইল চুরি হয়ে গেছে","ro":"{name} er bag theke mobile churi hoye geche","en":"{name}'s mobile phone was stolen from the bag"},
 {"bn":"গরু চুরি হয়েছে গোয়ালঘর থেকে","ro":"goru churi hoyeche gowalghor theke","en":"A cow was stolen from the cowshed"},
 {"bn":"রাস্তায় হেঁটে যাওয়ার সময় পকেট কেটে {item} চুরি করেছে","ro":"rastay hete jawar somoy pocket kete {item} churi koreche","en":"Pickpocketed {name} on the street and stole the {item}"},
 {"bn":"বাড়িতে গাছের ফসল চুরি হয়েছে","ro":"barite gacher foshol churi hoyeche","en":"Crops were stolen from the homestead"},
])
add_type("robbery", "ডাকাতি/ছিনতাই", "robbery", ["390","392","394","397"], [
 {"bn":"রাস্তায় ছুরি দেখিয়ে {name}-এর কাছ থেকে জোর করে চেইন ছিনতাই করেছে","ro":"rastay churi dekhiye {name} er kach theke jore kore chain chintai koreche","en":"Armed with a knife, they snatched {name}'s gold chain on the street — robbery"},
 {"bn":"মোটরসাইকেলে করে এসে মোবাইল ছিনতাই করেছে","ro":"motorcycle e kore ese mobile chintai koreche","en":"They came on a motorcycle and committed robbery of the mobile phone"},
 {"bn":"অস্ত্র দেখিয়ে ভয় দেখিয়ে টাকা-পয়সা ছিনিয়ে নিয়েছে","ro":"ostro dekhiye bhoy dekhiye taka poysa chiniye niyeche","en":"They showed a weapon and robbed the money by intimidation"},
])
add_type("dacoity", "দলবদ্ধ ডাকাতি", "dacoity", ["391","395","396"], [
 {"bn":"পাঁচ-ছয় জনের গ্যাং অস্ত্র নিয়ে বাড়ি লুট করেছে","ro":"pach-chhoy jon er gang ostro niye bari lut koreche","en":"A gang of five-six armed men committed dacoity at the house"},
 {"bn":"দলবদ্ধভাবে ডাকাতি সংঘটিত হয়েছে, সবাই অস্ত্রধারী","ro":"dolboddhabhabe dakoiti shonghotito hoyeche, shobai ostrodhari","en":"Dacoity was committed jointly by five or more armed persons"},
 {"bn":"গ্রামে ঢুকে ১০-১২ জন মিলে একাধিক বাড়ি ডাকাতি করেছে","ro":"grame dhuke 10-12 jon mile ekadhik bari dakoiti koreche","en":"Ten-twelve men entered the village and committed dacoity at several houses"},
])
add_type("murder", "খুন/হত্যা", "murder", ["300","302","201","204"], [
 {"bn":"{name}-কে ছুরি মেরে হত্যা করেছে, মৃতদেহ উদ্ধার হয়েছে","ro":"{name} ke chhuri mere hotya koreche, mrittodo uddhar hoyeche","en":"{name} was stabbed to death — murder; the body was recovered"},
 {"bn":"পিস্তল দিয়ে গুলি করে {name}-এর মৃত্যু ঘটানো হয়েছে","ro":"pistol diye guli kore {name} er mrityu ghatano hoyeche","en":"{name} was shot with a pistol and died — murder"},
 {"bn":"ঝগড়ার জেরে লাঠি দিয়ে পিটিয়ে হত্যা করেছে","ro":"jhograr jere lathi diye pitiye hotya koreche","en":"Beaten to death with sticks after an altercation — murder"},
])
add_type("culpable_homicide", "অনিচ্ছাকৃত হত্যা", "culpable homicide", ["299","304"], [
 {"bn":"মারপিটে {name}-এর মৃত্যু হয়েছে, খুনের উদ্দেশ্য ছিল না","ro":"marpite {name} er mrityu hoyeche, khuner uddhyesho chhilo na","en":"{name} died from the assault — culpable homicide not amounting to murder"},
 {"bn":"বিনা অনুমতিতে ড্রাইভিংয়ে {name} নিহত হয়েছে","ro":"bina onumotiye driving e {name} nihoto hoyeche","en":"{name} was killed by rash driving without intention to kill"},
])
add_type("hurt", "সাধারণ ও মারাত্মক আঘাত", "hurt", ["323","324","325","326"], [
 {"bn":"ছুরি মেরে {name}-কে আহত করেছে","ro":"chhuri mere {name} ke ahoto koreche","en":"They stabbed and injured {name}"},
 {"bn":"লাঠি দিয়ে মারধর করে {name}-এর হাত ভেঙেছে","ro":"lathi diye mar-dhor kore {name} er hath bhengeche","en":"Beaten with sticks, breaking {name}'s arm — grievous hurt"},
 {"bn":"ঝগড়ায় রাস্তার পাশে ফেলে গুরুতর জখম করেছে","ro":"jhogray rastar pashe fele guruto jokhm koreche","en":"Threw {name} down during a quarrel causing grievous hurt"},
])
add_type("assault_women", "নারীর ওপর আক্রমণ", "assault on woman", ["354"], [
 {"bn":"রাস্তায় এক নারীকে জোর করে ধরে আক্রমণ করেছে","ro":"rastay ek nari ke jore kore dhore akromon koreche","en":"Assaulted a woman on the street using criminal force intending to outrage her modesty"},
 {"bn":"নারী বাস যাত্রীকে অশালীনভাবে ছুঁয়ে আক্রমণ করেছে","ro":"nari bash jatrito ke oshalin bhabe chhuye akromon koreche","en":"Assaulted a female bus passenger with intent to outrage her modesty"},
])
add_type("eve_teasing", "ইভ-টিজিং/অশ্লীল তিরস্কার", "eve teasing", ["509"], [
 {"bn":"রাস্তায় স্কুলছাত্রীদের অশ্লীল কথা বলে ইভটিজিং করছে","ro":"rastay school chhatridero oshlil kotha bole ev-tijging korchhe","en":"Passing indecent remarks and eve-teasing schoolgirls on the street"},
 {"bn":"প্রতিদিন কলেজ পথে ছাত্রীকে অশ্লীল ইশারা করে বিরক্ত করে","ro":"protidin college pothe chhatritoke oshlil ishara kore birkoto kore","en":"Daily eve-teasing a college student with indecent gestures on her way to college"},
 {"bn":"মোবাইলে অশ্লীল মেসেজ পাঠিয়ে মেয়েটিকে বিরক্ত করছে","ro":"mobile o oshlil message pathiye meyetike birkoto korchhe","en":"Eve-teasing via indecent mobile messages to the girl"},
])
add_type("rape", "ধর্ষণ", "rape", ["375","376"], [
 {"bn":"{name}-কে জোর করে ধর্ষণ করেছে, মামলা দায়ের হয়েছে","ro":"{name} ke jore kore dhorshon koreche, mamla dayer hoyeche","en":"{name} was raped; a case has been filed"},
 {"bn":"নাবালিকাকে ধর্ষণ করে পালিয়েছে","ro":"nabalika ke dhorshon kore paliyeche","en":"Raped a minor and fled"},
])
add_type("kidnapping", "অপহরণ", "kidnapping", ["359","363","365","366"], [
 {"bn":"স্কুল ফেরত পথে শিশুটিকে অপহরণ করেছে","ro":"school ferot pothe shishutike opohoron koreche","en":"Kidnapped the child on the way home from school"},
 {"bn":"{name}-কে গাড়িতে জোর করে তুলে নিয়ে গেছে","ro":"{name} ke garite jore kore tule niye geche","en":"Abducted {name} by force into a vehicle"},
 {"bn":"মুক্তিপণ দাবি করে {name}-কে বন্দি রেখেছে","ro":"muktipon dabi kore {name} ke bondi rekheche","en":"Kidnapped and held {name} for ransom"},
])
add_type("wrongful_confinement", "অবৈধ আটক", "wrongful confinement", ["340","342","343"], [
 {"bn":"{name}-কে এক কক্ষে তিন দিন তালাবদ্ধ করে রেখেছে","ro":"{name} ke ek kokhhe tin din talaboddho kore rekheche","en":"Locked {name} in a room for three days — wrongful confinement"},
 {"bn":"পরিবারকে ঘরে বন্দি করে রেখেছে","ro":"poribarke ghore bondi kore rekheche","en":"Unlawfully confined the family inside the house"},
])
add_type("cheating", "প্রতারণা", "cheating", ["415","417","420"], [
 {"bn":"ভুয়া চাকরির প্রতিশ্রুতি দিয়ে {amount} টাকা নিয়ে পালিয়েছে","ro":"bhua chakrir protishruti diye {amount} taka niye paliyeche","en":"Cheated {amount} taka with a false job promise and fled"},
 {"bn":"অনলাইনে ভুয়া পণ্য বিক্রির প্রতারণা করেছে","ro":"online bhua ponnyo bikrir protarona koreche","en":"Committed online cheating by selling fake products"},
 {"bn":"ভূমির ভুয়া দলিল বানিয়ে প্রতারণা করেছে","ro":"bhumir bhua dolil baniye protarona koreche","en":"Cheated with a forged land deed"},
])
add_type("breach_of_trust", "আস্থাভাজন", "criminal breach of trust", ["405","406","409"], [
 {"bn":"কোম্পানির টাকা আত্মসাৎ করেছে","ro":"company er taka atomshot koreche","en":"Dishonestly misappropriated company funds — criminal breach of trust"},
 {"bn":"জমা রাখা টাকা ফেরত না দিয়ে আত্মসাৎ করেছে","ro":"joma rakha taka ferot na diye atomshot koreche","en":"Entrusted deposits were dishonestly misappropriated"},
 {"bn":"এজেন্ট হিসেবে পাওয়া টাকা উধাও করেছে","ro":"agent hisebe pawa taka udhao koreche","en":"The agent misappropriated the entrusted money"},
])
add_type("mischief", "সম্পত্তির ক্ষতি", "mischief", ["425","426","427","435"], [
 {"bn":"গাড়ির কাচ ভেঙে ক্ষতি করেছে","ro":"garir kach bhenge khoti koreche","en":"Broke the car window — mischief to property"},
 {"bn":"প্রতিবেশীর গাছ কেটে ফেলেছে","ro":"protibeshir gach kete feleche","en":"Cut down the neighbour's trees — mischief"},
 {"bn":"রাগ করে দোকানে আগুন দিয়েছে","ro":"rag kore dokane agun diyeche","en":"Set fire to the shop intentionally — mischief by fire"},
])
add_type("trespass", "অনধিকার প্রবেশ", "criminal trespass", ["441","442","447","448"], [
 {"bn":"রাতে জমির ভেতরে অনধিকার প্রবেশ করেছে","ro":"rate jamir bhitore onodikar proboesh koreche","en":"Committed criminal trespass into the land at night"},
 {"bn":"দোকানের শাটার ভেঙে ভেতরে ঢুকেছে","ro":"dokaner shatar bhenge bhitore dhukeche","en":"Broke the shutter and entered the shop unlawfully"},
])
add_type("defamation", "মানহানি", "defamation", ["499","500"], [
 {"bn":"ফেসবুকে মিথ্যা পোস্ট করে মানহানি করেছে","ro":"facebook e bhua post kore manhani koreche","en":"Posted false statements on Facebook defaming {name}"},
 {"bn":"গ্রামে মিথ্যা কথা ছড়িয়ে সুনাম নষ্ট করেছে","ro":"grame bhua kotha chhariye sunam nostho koreche","en":"Spread false rumours in the village damaging reputation"},
])
add_type("criminal_intimidation", "ভয় প্রদর্শন", "criminal intimidation", ["503","506"], [
 {"bn":"মেরে ফেলার হুমকি দিয়েছে","ro":"mere felar humki diyeche","en":"Threatened to kill — criminal intimidation"},
 {"bn":"মামলা তুলে নিতে জীবনের ভয় দেখিয়েছে","ro":"mamla tule nite jiboner bhoy dekhiyeche","en":"Intimidated {name} to withdraw the case"},
])
add_type("sexual_harassment_workplace", "কর্মক্ষেত্রে যৌন হয়রানি", "workplace harassment", ["509"], [
 {"bn":"অফিসে নারী কর্মীকে অশ্লীল প্রস্তাব দিয়ে হয়রান করছে","ro":"office e nari kormi ke oshlil prostab diye hoyran korchhe","en":"Sexually harassing a female colleague at work with indecent proposals"},
 {"bn":"বস পদের অপব্যবহার করে কর্মীকে বিরক্ত করছে","ro":"boss poder obbyabohar kore kormike birkoto korchhe","en":"Misusing authority to sexually harass the employee"},
])
add_type("dowry", "যৌতুক", "dowry", ["3","4"], [
 {"bn":"বিয়ের পর যৌতুকের জন্য নির্যাতন করছে","ro":"biyer por joutuker jonno nirjaton korchhe","en":"Torturing the wife for dowry after marriage"},
 {"bn":"{amount} টাকা যৌতুক দাবি করে স্ত্রীকে মারধর করেছে","ro":"{amount} taka joutuk dabi kore strike mar-dhor koreche","en":"Demanded {amount} taka as dowry and assaulted the wife"},
 {"bn":"যৌতুক না পেয়ে স্ত্রীকে তালাক দেওয়ার হুমকি দিচ্ছে","ro":"joutuk na peye strike talak dewar humki dicchhe","en":"Threatening divorce unless dowry is paid"},
])
add_type("cheque_bounce", "চেক ডিশনার", "cheque dishonour", ["138"], [
 {"bn":"মাননীয় চেক এনক্যাশ হয়েছে, টাকা নেই","ro":"manny chek encash hoyeche, taka nei","en":"The cheque was dishonoured for insufficient funds — §138 NI Act"},
 {"bn":"ব্যবসার বকেয়ার চেক ডিশনার হয়েছে","ro":"boshar bokyar chek dishonour hoyeche","en":"A business-payment cheque bounced"},
])
add_type("cyber_fraud", "অনলাইন প্রতারণা", "cyber fraud", ["24"], [
 {"bn":"ফেসবুকে ভুয়া আইডি খুলে টাকা হাতিয়ে নিয়েছে","ro":"facebook e bhua ID khule taka hatiye niyeche","en":"Opened a fake Facebook ID and defrauded money online"},
 {"bn":"মোবাইলে ভুয়া বিকাশ মেসেজ পাঠিয়ে টাকা নিয়েছে","ro":"mobile o bhua bikash message pathiye taka niyeche","en":"Sent fake bKash messages and cheated money"},
 {"bn":"অনলাইন শপিংয়ে টাকা নিয়ে পণ্য পাঠায়নি","ro":"online shopping e taka niye ponnyo pathayni","en":"Online shopping fraud — took payment, never delivered"},
])
add_type("cyber_defamation", "অনলাইন মানহানি", "online defamation", ["25","29"], [
 {"bn":"ডিজিটাল মাধ্যমে ভুয়া তথ্য ছড়িয়ে মানহানি করেছে","ro":"digital madhyome bhua tothyo chhariye manhani koreche","en":"Published false information online defaming {name}"},
 {"bn":"ইউটিউবে মানহানিকর ভিডিও আপলোড করেছে","ro":"youtube o manhanikor video upload koreche","en":"Uploaded defamatory videos on YouTube"},
])
add_type("cyber_hacking", "হ্যাকিং/অবৈধ প্রবেশ", "hacking", ["15","17"], [
 {"bn":"ফেসবুক অ্যাকাউন্ট হ্যাক করে নিয়ন্ত্রণ নিয়েছে","ro":"facebook account hack kore niyontron niyeche","en":"Hacked the Facebook account and took control"},
 {"bn":"কম্পিউটারে অনুমতি ছাড়া ঢুকে তথ্য চুরি করেছে","ro":"computer e onumoti chara dhuke tothyo churi koreche","en":"Unauthorised access to a computer and data theft"},
])
add_type("narcotics", "মাদক বিক্রি/বহন", "narcotics", ["5","6","9"], [
 {"bn":"ইয়াবা বিক্রির সময় ধরা পড়েছে","ro":"yaba bikrir somoy dhora poreche","en":"Caught selling yaba (methamphetamine)"},
 {"bn":"গাঁজা নিয়ে পাহারায় ধরা পড়েছে","ro":"ganza niye paharay dhora poreche","en":"Caught carrying cannabis at the checkpoint"},
 {"bn":"হিরোইনসহ {name} গ্রেপ্তার হয়েছে","ro":"heroin sho {name} grepto hoyeche","en":"{name} was arrested with heroin"},
])
add_type("corruption_bribery", "ঘুষ/দুর্নীতি", "bribery/corruption", ["161","163","165","409"], [
 {"bn":"সরকারি কাজের জন্য ঘুষ দাবি করেছে","ro":"sharkari kajer jonno ghush dabi koreche","en":"A public servant demanded a bribe for official work"},
 {"bn":"টেন্ডারবাজি করে সরকারি অর্থ আত্মসাৎ করেছে","ro":"tenderbaji kore sharkari ortho atomshot koreche","en":"Misappropriated public funds through tender manipulation"},
])
add_type("negligence_death", "দুর্ঘটনায় মৃত্যু", "death by negligence", ["304A"], [
 {"bn":"বেপরোয়া গাড়ি চালিয়ে রাস্তায় {name} নিহত হয়েছে","ro":"beproya gari chaliye rastay {name} nihoto hoyeche","en":"{name} was killed by rash driving — death by negligence"},
 {"bn":"নির্মাণসামগ্রী ফেলে মানুষ মারা গেছে","ro":"nirman-shongi fele manush mara geche","en":"A person died after construction material was dropped carelessly"},
])
add_type("money_loan_default", "অর্থঋণ পরিশোধে ব্যর্থতা", "loan default", ["5(1)"], [
 {"bn":"ব্যাংকের ঋণ পরিশোধ করছে না","ro":"banker rin porishodh korchhe na","en":"Defaulting on the bank loan — Artha Rin Adalat"},
 {"bn":"এনজিওর ঋণ নিয়ে পালিয়ে আছে","ro":"NGO er rin niye paliye achhe","en":"Fled without repaying the NGO loan"},
])
add_type("family_maintenance", "ভরণপোষণ", "maintenance", ["5(1)"], [
 {"bn":"স্ত্রী-সন্তানের ভরণপোষণ দিচ্ছে না","ro":"stri-shantaner bhoronposhon dicchhe na","en":"Not paying maintenance to wife and children — Family Court"},
 {"bn":"বাবা মেয়ের খরচ দিচ্ছেন না","ro":"baba meyer khorch dicchhen na","en":"Father refuses the daughter's maintenance"},
])
add_type("village_dispute", "ছোটখাটো বিরোধ", "village dispute", ["8"], [
 {"bn":"প্রতিবেশীর সঙ্গে সীমানা বিরোধ","ro":"protibeshir shonghe simana birodh","en":"Boundary dispute with the neighbour — Village Court"},
 {"bn":"খালের পানি নিয়ে বিরোধ হয়েছে","ro":"khaler pani niye birodh hoyeche","en":"Dispute over canal water — fit for Village Court"},
])
add_type("acid_violence", "এসিড সহিংসতা", "acid violence", ["4"], [
 {"bn":"প্রত্যাখ্যানে রাগ করে এসিড নিক্ষেপ করেছে","ro":"prottikhan e rag kore acid nikhep koreche","en":"Threw acid after rejection — acid violence"},
 {"bn":"এসিডে পুড়িয়ে দিয়েছে", "ro":"acide puriye diyeche", "en":"Burned the victim with acid"},
])

# fillers / connectors / subjects in 3 scripts
FILL = {
 "bn": ["আজ", "কাল থেকে", "৩ দিন ধরে", "খুব", "গতকাল", "রাতের বেলা", "এলাকায়", "ঘটনাস্থলে", "", "মামলা হয়েছে"],
 "ro": ["aj", "kal theke", "3 din dhore", "khub", "gotokal", "rate", "elar utte", "ghotona sthale", "", "mamla hoyeche"],
 "en": ["today", "since yesterday", "for three days", "badly", "at night", "in the area", "at the spot", "", "an FIR was filed"],
}
NEG = {"bn": ["নেই", "না", "হয়নি"], "ro": ["nei", "na", "hoyeche na"], "en": ["not", "no", "never"]}
SUBJ = {
 "bn": ["আমার", "তার", "আব্দুলের", "শিশুটির", "রোগীর", "মহিলার", "আমাদের এলাকার"],
 "ro": ["amar", "tar", "abduler", "shishu tir", "rogir", "mohilar", "amader elakar"],
 "en": ["My", "His", "Abdul's", "The child's", "The patient's", "The woman's", "In our area"],
}
CONN = {"bn": [" এবং ", "; ", ", এবং ", " কিন্তু "], "ro": [" ar ", "; ", ", and ", " kintu "], "en": [" and ", "; ", ", plus ", " but "]}
INTENT = {
 "bn": ["আমি একটি মামলা করতে চাই", "মামলা দায়ের করার নিয়ম কী", "আদালতের খরচ কত", "আমাকে সাহায্য করুন", "কেস স্ট্যাটাস জানতে চাই", "পরের তারিখ জানতে চাই"],
 "ro": ["ami ekti mamla korte chai", "mamla dayer korar niyom ki", "adalater khorch koto", "amake shahajjo korun", "case status jante chai", "porer tarikh jante chai"],
 "en": ["I want to file a case", "how do I file a case", "what are the court fees", "please help me", "I want case status", "what is the next hearing date"],
}
SLOTS = {
 "item": {"bn": ["টিভি", "কম্পিউটার", "সাইকেল", "সোনার চেইন", "মোবাইল"], "ro": ["TV", "computer", "cycle", "sonar chain", "mobile"], "en": ["TV", "computer", "bicycle", "gold chain", "mobile phone"]},
 "amount": {"bn": ["৫০ হাজার", "২ লাখ", "৫ লাখ"], "ro": ["50 hajar", "2 lakh", "5 lakh"], "en": ["50,000", "200,000", "500,000"]},
}
NAMES = {"bn": ["রফিক", "করিম", "শিমুল", "আব্দুল", "নাসির"], "ro": ["Rafiq", "Karim", "Shimul", "Abdul", "Nasir"], "en": ["Rafiq", "Karim", "Shimul", "Abdul", "Nasir"]}
SCRIPTS = ["bn", "ro", "en"]

def phrase_text(ph, script, rng):
    text = ph[script]
    for slot in ("item", "amount"):
        if "{" + slot + "}" in text:
            text = text.replace("{" + slot + "}", rng.choice(SLOTS[slot][script]))
    text = text.replace("{name}", rng.choice(NAMES[script]))
    return text

def gen_sentence(rng, per_type=1):
    """Generate one sentence; returns (text, script, labels dict)."""
    types = rng.sample(list(T.keys()), k=rng.choice([1, 1, 1, 2, 2, 3]))
    script = rng.choice(SCRIPTS)
    chunks, labels = [], {}
    neg_idx = -1
    negatable = [i for i, ct in enumerate(types) if ct not in ()]
    if negatable and rng.random() < 0.15:
        neg_idx = rng.choice(negatable)
    for i, ct in enumerate(types):
        neg = (i == neg_idx)
        ph = phrase_text(rng.choice(T[ct]["phrases"]), script, rng)
        if neg:
            ph = ph + " " + rng.choice(NEG[script])
        labels[ct] = not neg
        chunks.append(ph)
    subj = rng.choice(SUBJ[script])
    sent = (subj + " " if subj and not subj[0].isupper() else subj + " " if subj else "") + rng.choice(CONN[script]).join(chunks)
    if rng.random() < 0.45:
        sent += " " + rng.choice(FILL[script])
    return re.sub(r"\s+", " ", sent).strip(), script, labels

def build(total=1_020_000, out=OUT_JSONL):
    rng = random.Random(20260906)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        while n < total:
            # type-balanced round-robin
            for ct in T:
                if n >= total: break
                sent, script, labels = gen_sentence(rng)
                # ensure the sentence belongs to a type in labels; force-balance by
                # generating until ct present positive (cheap because k>=1 and we retry)
                tries = 0
                while ct not in [c for c, v in labels.items() if v] and tries < 6:
                    sent, script, labels = gen_sentence(rng)
                    tries += 1
                f.write(json.dumps({"text": sent, "script": script, "labels": labels}, ensure_ascii=False) + "\n")
                n += 1
                if n % 4 == 0:
                    sent2, script2, labels2 = gen_sentence(rng)
                    f.write(json.dumps({"text": sent2, "script": script2, "labels": labels2}, ensure_ascii=False) + "\n")
                    n += 1
    # background intent sentences (10% extra)
    with open(out, "a", encoding="utf-8") as f:
        for i in range(total // 10):
            script = rng.choice(SCRIPTS)
            f.write(json.dumps({"text": rng.choice(INTENT[script]), "script": script, "labels": {}}, ensure_ascii=False) + "\n")
    print(f"corpus written: {n} labelled + {total//10} background -> {out}", flush=True)

if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 1_020_000)
