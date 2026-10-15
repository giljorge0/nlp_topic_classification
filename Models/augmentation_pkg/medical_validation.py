"""Conservative checks. These do NOT prove clinical semantic equivalence."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\b[^\W_]+(?:[’'\-][^\W_]+)*\b", flags=re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?(?!\w)")

# Safety-first fallback when a specialized clinical NLP model is unavailable.
CLINICAL_WORDS = frozenset("""
acute chronic subacute bilateral unilateral left right proximal distal anterior posterior superior inferior
positive negative absent present possible suspected probable unlikely confirmed uncertain severe mild moderate
history prior previous newly new worsening progressive improving recurrent persistent temporary sudden
pain painful painless fever febrile afebrile trauma traumatic fracture broken dislocation sprain strain
bleeding hemorrhage haemorrhage hematoma clot thrombosis embolism embolus stroke infarction ischemia
cancer carcinoma malignancy malignant benign tumor tumour lesion mass cyst abscess infection sepsis
septic inflammation inflammatory pneumonia bronchitis asthma copd hypoxia dyspnea dyspnoea apnea
chest cardiac heart myocardial coronary atrial ventricular tachycardia bradycardia arrhythmia
pulmonary lung airway pleural pneumothorax embolism
abdominal abdomen gastric gastro intestinal bowel colon colorectal rectal rectum appendix appendicitis
liver hepatic gallbladder biliary pancreas pancreatic renal kidney ureter bladder urinary
neurological neurologic brain spinal nerve neuropathy seizure epilepsy migraine headache
orthopedic orthopaedic musculoskeletal knee hip elbow ankle shoulder wrist hand foot
bone joint ligament tendon muscle cartilage meniscus cervical lumbar thoracic
surgery surgical postoperative preoperative post-operative pre-operative biopsy excision removal
laparoscopy laparoscopic endoscopy endoscopic colonoscopy gastroscopy arthroscopy
radiology radiological imaging ultrasound radiograph xray ct mri contrast scan
medication drug dose dosage mg mcg g ml mmol units iv oral intravenous intraoperative
pregnancy pregnant fetal maternal infant neonatal pediatric paediatric adult
admission discharge inpatient outpatient consultation examination assessment follow-up
hypotension hypertension diabetes insulin glucose anemia anaemia hemoglobin haemoglobin
nausea vomiting diarrhea diarrhoea constipation dysphagia odynophagia jaundice
weakness numbness paresthesia paralysis paralysis syncope dizziness vision hearing
denies without none neither no not never non negative cannot can't doesn't did not ruled
""".split())

# Conservative deletion candidates. We deliberately avoid removing 'with', 'without',
# 'and', 'or', 'but', 'after', 'before' and clinically important temporal modifiers.
SAFE_DROPPABLE = frozenset("""
the a an this these those that was were is are be being been had has have
patient patients an individual a person
""".split())

NEGATION_PATTERNS = [
    re.compile(r"\b(?:no|not|never|without|denies|denied|denying|negative|absent|neither|nor|cannot|can't|doesn't|didn't)\b", re.I),
    re.compile(r"\bfree of\b", re.I),
    re.compile(r"\bruled out\b", re.I),
]
LATERALITY_PATTERNS = {
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
    "bilateral": re.compile(r"\bbilateral\b|\bboth (?:sides?|knees?|hips?|legs?|arms?|hands?|feet|shoulders?)\b", re.I),
    "unilateral": re.compile(r"\bunilateral\b", re.I),
}
# Conservative exact-or-equivalent anchors. This cannot replace medical review.
CLINICAL_EQUIVALENTS = {
    "kidney": "renal", "renal": "kidney",
    "liver": "hepatic", "hepatic": "liver",
    "tumor": "tumour", "tumour": "tumor",
    "hemorrhage": "haemorrhage", "haemorrhage": "hemorrhage",
    "dyspnea": "dyspnoea", "dyspnoea": "dyspnea",
    "orthopedic": "orthopaedic", "orthopaedic": "orthopedic",
    "diarrhea": "diarrhoea", "diarrhoea": "diarrhea",
    "anemia": "anaemia", "anaemia": "anemia",
    "hemoglobin": "haemoglobin", "haemoglobin": "hemoglobin",
}


def _normalize_medical_word(word: str) -> str:
    if word in CLINICAL_WORDS:
        return word
    if word.endswith("ies") and word[:-3] + "y" in CLINICAL_WORDS:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2] in CLINICAL_WORDS:
        return word[:-2]
    if word.endswith("s") and word[:-1] in CLINICAL_WORDS:
        return word[:-1]
    return word


# These semantic markers are checked by dedicated, synonym-tolerant rules instead.
ANCHOR_EXCLUSIONS = frozenset("""
no not never without denies denied none neither nor cannot can't doesn't didn't
negative absent non bilateral unilateral left right
""".split())

CERTAINTY_TEMPORAL = frozenset("""
acute chronic possible probable suspected uncertain confirmed prior previous worsening
improving recurrent persistent newly sudden postoperative preoperative
""".split())


def tokens(text: str) -> list[str]:
    return [m.group().casefold() for m in TOKEN_RE.finditer(text)]


def number_counts(text: str) -> Counter[str]:
    return Counter(m.group().replace(",", "") for m in NUMBER_RE.finditer(text))


def negation_count(text: str) -> int:
    return sum(len(p.findall(text)) for p in NEGATION_PATTERNS)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


def validate_paraphrase(original: str, candidate: str, *, min_ratio: float = .5,
                        max_ratio: float = 2.0) -> ValidationResult:
    reasons = []
    old, new = original.strip(), candidate.strip()
    if not new:
        return ValidationResult(False, ("empty_output",))
    if " ".join(old.casefold().split()) == " ".join(new.casefold().split()):
        reasons.append("unchanged_text")
    if new.startswith("```") or new.startswith("<clinical_description>"):
        reasons.append("unexpected_formatting")
    old_tokens = tokens(old)
    new_tokens = tokens(new)
    ratio = len(new_tokens) / max(1, len(old_tokens))
    if not min_ratio <= ratio <= max_ratio:
        reasons.append("length_out_of_range")
    if number_counts(old) != number_counts(new):
        reasons.append("numbers_changed")
    if negation_count(old) != negation_count(new):
        reasons.append("negation_cues_changed")
    for category, pattern in LATERALITY_PATTERNS.items():
        if bool(pattern.search(old)) != bool(pattern.search(new)):
            reasons.append(f"laterality_{category}_changed")
    for marker in CERTAINTY_TEMPORAL:
        if (marker in old_tokens) != (marker in new_tokens):
            reasons.append(f"modifier_{marker}_changed")
    original_concepts = ({_normalize_medical_word(w) for w in old_tokens} & CLINICAL_WORDS) - ANCHOR_EXCLUSIONS
    candidate_concepts = {_normalize_medical_word(w) for w in new_tokens}
    for concept in sorted(original_concepts):
        if concept not in candidate_concepts and CLINICAL_EQUIVALENTS.get(concept) not in candidate_concepts:
            reasons.append(f"clinical_concept_missing:{concept}")
    # Newly introduced clinical concepts can signal a hallucinated condition.
    # Equivalents are tolerated, but all flags still require human inspection.
    candidate_medical_concepts = (candidate_concepts & CLINICAL_WORDS) - ANCHOR_EXCLUSIONS
    original_all = {_normalize_medical_word(w) for w in old_tokens}
    for concept in sorted(candidate_medical_concepts):
        if concept not in original_all and CLINICAL_EQUIVALENTS.get(concept) not in original_all:
            reasons.append(f"clinical_concept_added:{concept}")
    return ValidationResult(not reasons, tuple(reasons))


def validate_dropout(original: str, augmented: str, deleted_words: list[str],
                     protected_words: set[str]) -> ValidationResult:
    reasons = []
    if not augmented.strip() or original == augmented:
        reasons.append("empty_or_unchanged")
    if not deleted_words:
        reasons.append("no_words_deleted")
    if any(w.casefold() in protected_words for w in deleted_words):
        reasons.append("protected_word_deleted")
    if number_counts(original) != number_counts(augmented):
        reasons.append("numbers_changed")
    if negation_count(original) != negation_count(augmented):
        reasons.append("negation_changed")
    return ValidationResult(not reasons, tuple(reasons))
