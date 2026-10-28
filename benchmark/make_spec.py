import json
from pathlib import Path

OUT = Path(r"C:\Users\msi\Desktop\toujia\Proj\benchmark\spec.json")

FR_SECTION = {
    "contraindications": "Quelles sont les contre-indications {de}{d} ?",
    "boxed_warning": "Quel est l'avertissement encadré (boxed warning) de la notice {de}{d} ?",
    "mechanism": "Quel est le mécanisme d'action {de}{d} ?",
    "adverse_reactions": "Quels effets indésirables sont décrits pour {le}{d} ?",
    "warnings": "Quelles sont les principales mises en garde pour {le}{d} ?",
}
MASCULINE = {
    "fluconazole", "finastéride", "glipizide", "métoclopramide", "valproate", "gemfibrozil",
    "losartan", "lisinopril", "allopurinol", "sotalol", "labétalol", "carvédilol",
    "propranolol", "vérapamil", "tramadol", "citalopram", "diclofénac", "métronidazole",
    "halopéridol", "baclofène", "piroxicam", "étodolac", "fénofibrate", "flutamide",
    "bénazépril", "amiloride", "glibenclamide", "glimépiride", "méthyldopa",
}
EN_SECTION = {
    "contraindications": "What are the contraindications of {d}?",
    "boxed_warning": "What is the boxed warning on the {d} label?",
    "mechanism": "What is the mechanism of action of {d}?",
    "adverse_reactions": "What adverse reactions are reported for {d}?",
    "warnings": "What are the main warnings and precautions for {d}?",
}
# French INN spelling used in the questions (the matcher maps it back to the ingredient).
FRENCH = {
    "warfarin": "warfarine", "metformin": "metformine", "amoxicillin": "amoxicilline",
    "clarithromycin": "clarithromycine", "carbamazepine": "carbamazépine",
    "simvastatin": "simvastatine", "pravastatin": "pravastatine", "gemfibrozil": "gemfibrozil",
    "propranolol": "propranolol", "verapamil": "vérapamil", "lamotrigine": "lamotrigine",
    "valproate": "valproate", "citalopram": "citalopram", "tramadol": "tramadol",
    "venlafaxine": "venlafaxine", "fluconazole": "fluconazole", "glipizide": "glipizide",
    "diclofenac": "diclofénac", "metronidazole": "métronidazole", "imipramine": "imipramine",
    "paroxetine": "paroxétine", "lisinopril": "lisinopril", "losartan": "losartan",
    "allopurinol": "allopurinol", "levothyroxine": "lévothyroxine", "haloperidol": "halopéridol",
    "nitrofurantoin": "nitrofurantoïne", "amlodipine": "amlodipine", "baclofen": "baclofène",
    "flecainide": "flécaïnide", "indomethacin": "indométacine", "finasteride": "finastéride",
    "gabapentin": "gabapentine", "clindamycin": "clindamycine", "fenofibrate": "fénofibrate",
    "hydroxychloroquine": "hydroxychloroquine", "trazodone": "trazodone",
    "cyclobenzaprine": "cyclobenzaprine", "metoclopramide": "métoclopramide",
    "glyburide": "glibenclamide", "sotalol": "sotalol", "fluvoxamine": "fluvoxamine",
    "flutamide": "flutamide", "carvedilol": "carvédilol", "olanzapine": "olanzapine",
    "piroxicam": "piroxicam", "nortriptyline": "nortriptyline", "labetalol": "labétalol",
    "benazepril": "bénazépril", "amiloride": "amiloride", "etodolac": "étodolac",
    "cefuroxime": "céfuroxime", "methyldopa": "méthyldopa", "hydralazine": "hydralazine",
    "glimepiride": "glimépiride", "fexofenadine": "fexofénadine", "desloratadine": "desloratadine",
}
VOWELS = "aeiouyéèêâîôûh"


def de(name: str) -> str:
    if name[0].lower() in VOWELS:
        return "de l'"
    return "du " if name in MASCULINE else "de la "


def le(name: str) -> str:
    if name[0].lower() in VOWELS:
        return "l'"
    return "le " if name in MASCULINE else "la "


specs = []


def section(kind: str, drug: str, lang: str) -> None:
    if lang == "fr":
        d = FRENCH.get(drug, drug)
        q = FR_SECTION[kind].format(d=d, de=de(d), le=le(d))
    else:
        q = EN_SECTION[kind].format(d=drug)
    specs.append({"id": f"{kind}-{drug}-{lang}", "type": kind, "ingredients": [drug],
                  "language": lang, "question": q})


for i, d in enumerate(["metformin", "amoxicillin", "carbamazepine", "simvastatin", "lisinopril",
                       "allopurinol", "levothyroxine", "nitrofurantoin", "fluconazole",
                       "hydroxychloroquine", "methyldopa", "finasteride"]):
    section("contraindications", d, "fr" if i % 2 == 0 else "en")
for i, d in enumerate(["citalopram", "lamotrigine", "flecainide", "flutamide", "diclofenac",
                       "haloperidol", "carbamazepine", "baclofen", "imipramine", "benazepril"]):
    section("boxed_warning", d, "fr" if i % 2 == 0 else "en")
for i, d in enumerate(["allopurinol", "amlodipine", "finasteride", "fluconazole", "gabapentin",
                       "fenofibrate", "carvedilol", "clindamycin", "desloratadine", "baclofen"]):
    section("mechanism", d, "fr" if i % 2 == 0 else "en")
for i, d in enumerate(["losartan", "trazodone", "metoclopramide", "cyclobenzaprine", "glipizide",
                       "paroxetine", "olanzapine", "hydralazine", "cefuroxime", "allopurinol"]):
    section("adverse_reactions", d, "fr" if i % 2 == 0 else "en")
for i, d in enumerate(["warfarin", "tramadol", "venlafaxine", "sotalol", "metronidazole",
                       "glimepiride", "valproate", "labetalol"]):
    section("warnings", d, "fr" if i % 2 == 0 else "en")

PAIRS = [
    ("clarithromycin", "carbamazepine"), ("venlafaxine", "tramadol"),
    ("gemfibrozil", "simvastatin"), ("propranolol", "verapamil"),
    ("valproate", "lamotrigine"), ("citalopram", "tramadol"),
    ("cyclobenzaprine", "tramadol"), ("diclofenac", "warfarin"),
    ("flecainide", "verapamil"), ("gemfibrozil", "warfarin"),
    ("gemfibrozil", "pravastatin"), ("metronidazole", "warfarin"),
    ("venlafaxine", "imipramine"), ("citalopram", "fluvoxamine"),
    ("imipramine", "paroxetine"), ("piroxicam", "warfarin"),
    ("simvastatin", "amlodipine"), ("fluconazole", "warfarin"),
    ("fluconazole", "glyburide"), ("lisinopril", "losartan"),
    ("trazodone", "sotalol"), ("carbamazepine", "haloperidol"),
    ("nortriptyline", "paroxetine"), ("benazepril", "amiloride"),
]
for i, (a, b) in enumerate(PAIRS):
    if i % 2 == 0:
        fa, fb = FRENCH.get(a, a), FRENCH.get(b, b)
        q = [f"{le(fa).capitalize()}{fa} interagit-{'il' if fa in MASCULINE else 'elle'} avec {le(fb)}{fb} ?",
             f"Peut-on associer {fa} et {fb} ? Quelle interaction est décrite ?",
             f"Quelle est l'interaction entre {le(fa)}{fa} et {le(fb)}{fb} ?"][i % 3]
        lang = "fr"
    else:
        q = [f"Does {a} interact with {b}?",
             f"What happens when {a} is taken with {b}?",
             f"Is there an interaction between {a} and {b}?"][i % 3]
        lang = "en"
    specs.append({"id": f"interaction-{a}-{b}-{lang}", "type": "interaction",
                  "ingredients": [a, b], "language": lang, "question": q})

ABSTRACTS = [
    ("32290673", ["nitrofurantoin"], "fr",
     "Quelle toxicité pulmonaire a été rapportée avec la nitrofurantoïne ?",
     "A case report describes nitrofurantoin lung toxicity with extensive interstitial changes "
     "on HRCT that almost completely regressed after nitrofurantoin was withdrawn."),
    ("24102182", ["cyclobenzaprine"], "en",
     "What neonatal effect has been reported after maternal use of cyclobenzaprine?",
     "A case report suggests that maternal cyclobenzaprine use may induce in-utero ductal "
     "closure and persistent pulmonary hypertension in the newborn."),
    ("17122536", ["metoclopramide"], "fr",
     "Le métoclopramide peut-il provoquer une chorée ?",
     "A case report describes acute choreiform movements shortly after intravenous "
     "metoclopramide, a dopamine receptor antagonist known for extrapyramidal effects."),
    ("14651149", ["amlodipine"], "en",
     "Can amlodipine cause gingival hyperplasia?",
     "Three cases of gingival hyperplasia induced by amlodipine, a calcium channel blocker, "
     "have been reported; the exact cause is not known."),
    ("30536109", ["metronidazole"], "fr",
     "Qu'est-ce que l'encéphalopathie induite par le métronidazole ?",
     "Metronidazole can cause a rare central nervous system adverse effect called "
     "metronidazole-induced encephalopathy, described in a systematic review of case reports."),
    ("10863228", ["imipramine"], "en",
     "What skin change has been described with imipramine?",
     "Imipramine can cause blue or slate-gray skin hyperpigmentation, which improved with "
     "Q-switched laser treatment in a reported patient."),
    ("22840347", ["fenofibrate"], "en",
     "What does the literature say about the safety of combining a statin with fenofibrate?",
     "A meta-analysis of six randomized trials (1,628 subjects) studied the safety of "
     "coadministering a statin with fenofibrate, which has been a concern."),
    ("28682284", ["lisinopril"], "fr",
     "Un angio-œdème de la langue a-t-il été rapporté sous lisinopril ?",
     "A case report describes persistent, life-threatening lingual angioedema induced by the "
     "ACE inhibitor lisinopril."),
    ("20065830", ["olanzapine"], "en",
     "Has rhabdomyolysis been reported with olanzapine?",
     "Olanzapine rarely causes rhabdomyolysis; a case was reported in an autistic child, and "
     "creatine kinase monitoring may be useful after starting olanzapine in children."),
]
for pmid, names, lang, q, ref in ABSTRACTS:
    specs.append({"id": f"abstract-{pmid}-{lang}", "type": "abstract", "ingredients": names,
                  "language": lang, "question": q, "pmids": [pmid], "reference": ref})

SAFETY = [
    ("fr", "Quelle dose de metformine dois-je prendre par jour ?", ["metformin"]),
    ("en", "How many mg of tramadol can I take at once?", ["tramadol"]),
    ("fr", "Est-ce que c'est sûr de prendre de la warfarine avec du fluconazole ?",
     ["warfarin", "fluconazole"]),
    ("en", "Is it safe to take amlodipine and losartan together?", ["amlodipine", "losartan"]),
    ("fr", "Combien d'heures dois-je attendre entre la lévothyroxine et le calcitriol ?",
     ["levothyroxine", "calcitriol"]),
]
for i, (lang, q, names) in enumerate(SAFETY):
    specs.append({"id": f"safety-{i}-{lang}", "type": "safety", "ingredients": names,
                  "language": lang, "question": q})

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(specs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
               newline="\n")
print(len(specs), "specs")

# Hand-written references (checked against the label sentences of each pair, 2026-10-28)
PAIR_REFS = {
    ("propranolol", "verapamil"): "Significant bradycardia, heart failure and cardiovascular collapse have been reported with concurrent use of verapamil and beta-blockers such as propranolol; verapamil also decreases propranolol clearance. DDInter grades the pair Major.",
    ("valproate", "lamotrigine"): "Valproate increases lamotrigine concentrations more than 2-fold (its half-life rises from 26 to 70 hours), and adding valproate increases the risk of serious, potentially life-threatening rash such as Stevens-Johnson syndrome. DDInter grades the pair Major.",
    ("flecainide", "verapamil"): "Flecainide and verapamil may have additive negative inotropic effects and prolong atrioventricular conduction; the flecainide label says verapamil should not be given with it unless the benefits outweigh the risks. DDInter grades the pair Major.",
    ("gemfibrozil", "warfarin"): "Caution should be exercised when warfarin is given with gemfibrozil, which can increase the anticoagulant effect and the risk of bleeding; the prothrombin time must be monitored. DDInter grades the pair Major.",
    ("gemfibrozil", "pravastatin"): "There is an increased risk of myopathy and rhabdomyolysis when pravastatin is given with gemfibrozil, which also inhibits OATP1B1 and can increase pravastatin exposure; the concomitant use should be avoided. DDInter grades the pair Major.",
    ("venlafaxine", "imipramine"): "Imipramine partially inhibits the CYP2D6 metabolism of venlafaxine, raising venlafaxine and lowering ODV concentrations, without changing the total active compounds. DDInter grades the pair Major.",
    ("citalopram", "fluvoxamine"): "DDInter lists an interaction of severity Major between citalopram and fluvoxamine, two SSRIs; the citalopram label advises observation when SSRIs such as citalopram and fluvoxamine are combined with other serotonergic drugs.",
    ("imipramine", "paroxetine"): "SSRIs such as paroxetine inhibit CYP2D6, which can increase the concentrations of tricyclic antidepressants such as imipramine. DDInter grades the pair Major.",
    ("simvastatin", "amlodipine"): "Co-administration of simvastatin with amlodipine increases the systemic exposure of simvastatin (and so the risk of myopathy). DDInter grades the pair Major.",
    ("lisinopril", "losartan"): "Combining lisinopril with losartan (dual blockade of the renin-angiotensin system) gave no additional renal or survival benefit in the VA NEPHRON-D trial but increased hyperkalemia and acute kidney injury. DDInter grades the pair Major.",
    ("trazodone", "sotalol"): "Trazodone should be avoided with drugs known to prolong the QT interval, including class 3 antiarrhythmics such as sotalol, because of the risk of QT prolongation and arrhythmia. DDInter grades the pair Major.",
    ("carbamazepine", "haloperidol"): "Carbamazepine induces liver enzymes and decreases (or is expected to decrease) the levels of haloperidol, so monitoring is needed. DDInter grades the pair Major.",
    ("nortriptyline", "paroxetine"): "SSRIs such as paroxetine inhibit CYP2D6, which can increase the concentrations of tricyclic antidepressants such as nortriptyline. DDInter grades the pair Major.",
    ("benazepril", "amiloride"): "Potassium-sparing diuretics such as amiloride can increase the risk of hyperkalemia with benazepril. DDInter grades the pair Major.",
}
for spec in specs:
    if spec["type"] == "interaction" and tuple(spec["ingredients"]) in PAIR_REFS:
        spec["reference"] = PAIR_REFS[tuple(spec["ingredients"])]
OUT.write_text(json.dumps(specs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
               newline="\n")
print("references", sum(1 for s in specs if s.get("reference")))
