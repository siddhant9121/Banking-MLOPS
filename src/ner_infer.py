"""Real NER inference for the banking-document API.

Combines the fine-tuned BERT token-classification model (trained in
src/train.py on data from src/ner_data.py) with regex validation for the
structured ID fields. PAN/Aadhaar/passport/DL/voter-ID numbers follow fixed
formats, so regex is used to fill in or correct spans the model misses or
mis-tags -- this matters a lot for OCR'd text where the model only saw
clean synthetic examples during training.
"""
import re
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "ner"

# (entity_type, compiled_pattern) -- checked in this order; first match wins
# for a given character span.
_EXCLUDE_WORDS = {
    "INCOME", "TAX", "DEPARTMENT", "GOVT", "GOVERNMENT", "INDIA", "OFINDIA",
    "FATHER", "FATHERS", "MOTHER", "MOTHERS", "SIGNATURE", "DATE", "BIRTH",
    "CARD", "PERMANENT", "ACCOUNT", "NUMBER", "MALE", "FEMALE", "UNION",
    "PASSPORT", "LICENCE", "ELECTION", "COMMISSION", "IDENTITY", "AUTHORITY",
    "UNIQUE", "IDENTIFICATION", "S/O", "D/O", "W/O", "UTI", "NSDL", "DETAILS",
    "OF", "AND", "NAME", "ELECTOR", "ELECTORS", "RELATION", "RELATIONS",
    "MOBILE", "NO", "PHONE", "ISSUE", "VID", "AADHAAR", "ADDRESS",
}


_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

def clean_dob(dob_str: str) -> str:
    mapping = {
        'C': '0', 'O': '0', 'o': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '5',
        'B': '8',
        'g': '9'
    }
    # Translate mapping
    cleaned = "".join(mapping.get(c, c) for c in dob_str)
    
    # Normalize dividers to spaces first to split
    cleaned_space = re.sub(r'[/-]', ' ', cleaned)
    parts = cleaned_space.split()
    
    if len(parts) == 3:
        day, month, year = parts[0], parts[1], parts[2]
        
        # Convert month if it's text
        month_upper = month.upper()
        if month_upper in _MONTH_MAP:
            month = _MONTH_MAP[month_upper]
            
        # Clean day
        if len(day) == 1:
            day = "0" + day
        elif len(day) > 2:
            day = day[:2]
            
        # Clean month
        if len(month) == 1:
            month = "0" + month
        elif len(month) > 2:
            month = month[:2]
            
        # Clean year. A 3-digit year means OCR dropped one digit -- there's
        # no way to know which one or what it was (e.g. a signature stroke
        # overlapping the date), so this used to blindly append "0" and
        # silently return a fabricated, possibly-wrong date. For a KYC
        # authenticity check, a wrong DOB is worse than a missing one (the
        # latter correctly routes to manual review instead), so a 3-digit
        # year is left as-is and falls through to fail the range check below.
        if len(year) == 2:
            try:
                yr = int(year)
                year = f"{19 if yr > 25 else 20}{yr:02d}"
            except ValueError:
                pass
                
        # Validate values
        try:
            d_val = int(day)
            m_val = int(month)
            y_val = int(year)
            if 1 <= d_val <= 31 and 1 <= m_val <= 12 and 1900 <= y_val <= 2030:
                return f"{day}/{month}/{year}"
        except ValueError:
            pass
            
    return None


def _clean_and_validate_name(name: str) -> str:
    # Restrict to Latin letters: with Hindi OCR enabled, raw_text now
    # contains real Devanagari header text (e.g. "आयकर विभाग"), and
    # Devanagari characters are alphabetic in Unicode with no case, so an
    # unrestricted isalpha()/islower() check let whole Devanagari lines
    # through as bogus "names". Non-Latin letters are dropped here rather
    # than disqualifying the whole line, since OCR occasionally interleaves
    # a stray non-Latin glyph into an otherwise valid Latin name.
    name_clean = "".join(c for c in name.upper() if (c.isalpha() and c.isascii()) or c.isspace())
    words = name_clean.split()
    words = [w for w in words if w not in _EXCLUDE_WORDS and len(w) >= 2]
    # Indian names on ID cards are almost always 2-5 words (first + middle(s)
    # + surname); the old cap of 4 clipped legitimate longer names.
    if len(words) < 2 or len(words) > 5:
        return None
    cleaned_name = " ".join(words)
    if len(cleaned_name) < 5 or len(cleaned_name) > 40:
        return None
    return cleaned_name


def _extract_name_heuristics(text: str, model_hint: str = None) -> str:
    """Scan OCR'd lines for name-shaped text (all-caps, 2-5 words).

    _run_ocr concatenates three independent OCR passes of the same image,
    so a correctly-read name tends to recur (identically) across passes,
    while a misread header or the unlabeled father's-name line on PAN/DL
    cards rarely reproduces itself the same way twice. Candidates are
    therefore ranked by how many times they recur -- with a bonus for
    agreeing with the model's own prediction -- instead of just taking
    whichever plausible-looking line appears first, which easily locks
    onto the wrong line on real (noisy) card photos.
    """
    lines = text.split("\n")
    scores = {}
    first_seen = {}
    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Strip non-letter characters and known field-label words *before*
        # judging casing. Real cards commonly print "Name: AMIT KUMAR" or
        # "Father's Name: RAJESH KUMAR" -- the mixed-case label alone used
        # to push the whole line's lowercase ratio over the threshold and
        # disqualify an otherwise all-caps name value on the same line.
        raw_words = re.findall(r"[A-Za-z]+", line_stripped)
        kept_words = [w for w in raw_words if w.upper() not in _EXCLUDE_WORDS and len(w) >= 2]
        if not kept_words:
            continue

        # Accept either the standard all-caps format used on genuine Indian
        # ID cards, or proper Title Case (each word capitalized) which some
        # scanned/digital documents use -- without this, a title-case name
        # is invisible to the heuristic and a same-case-but-wrong all-caps
        # OCR fragment elsewhere on the page can win by default.
        letters = [c for w in kept_words for c in w]
        lowercase_ratio = sum(1 for c in letters if c.islower()) / len(letters)
        is_mostly_upper = lowercase_ratio <= 0.2
        is_title_case = all(w[0].isupper() and w[1:].islower() for w in kept_words)
        if not (is_mostly_upper or is_title_case):
            continue

        cleaned = _clean_and_validate_name(" ".join(kept_words))
        if not cleaned:
            continue

        scores[cleaned] = scores.get(cleaned, 0) + 1
        first_seen.setdefault(cleaned, idx)

    if not scores:
        return None

    if model_hint:
        hint_cleaned = _clean_and_validate_name(model_hint)
        if hint_cleaned and hint_cleaned in scores:
            scores[hint_cleaned] += 2

    return max(scores, key=lambda c: (scores[c], -first_seen[c]))


# ── Verhoeff checksum (UIDAI Aadhaar check digit) ─────────────────────────────
# Aadhaar numbers are generated with a trailing Verhoeff check digit, so any
# 12-digit string that merely *looks* like an Aadhaar number (a random OCR
# misread, or a made-up placeholder such as "1234 5678 9012") overwhelmingly
# fails this checksum -- a far stronger authenticity signal than format or a
# fixed blacklist alone.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_checksum_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][item]]
    return c == 0


def looks_like_placeholder_digits(digits: str) -> bool:
    """Catches obviously-fake numeric spans: all-repeated digits (e.g.
    '0000000') or an ascending/descending run (e.g. '1234567') -- patterns
    dummy/example IDs commonly use and real government-issued numbers
    essentially never do."""
    if len(digits) < 4:
        return False
    if len(set(digits)) == 1:
        return True
    deltas = [int(digits[i + 1]) - int(digits[i]) for i in range(len(digits) - 1)]
    return all(d == 1 for d in deltas) or all(d == -1 for d in deltas)


# Anchored format checks for entity types whose value can come from the raw
# (unvalidated) model prediction when regex finds nothing -- see run_ner.
_STRICT_FORMATS = {
    "PASSPORT": re.compile(r"^[A-Z][0-9]{7}$"),
    "DL": re.compile(r"^[A-Z]{2}[0-9]{2}\s?[0-9]{11}$"),
    "VOTERID": re.compile(r"^[A-Z]{3}[0-9]{7}$"),
}

_ID_PATTERNS = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PASSPORT", re.compile(r"\b[A-Z][0-9]{7}\b")),
    ("VOTERID", re.compile(r"\b[A-Z]{3}[0-9]{7}\b")),
    ("DL", re.compile(r"\b[A-Z]{2}[0-9]{2}\s?[0-9]{11}\b")),
    ("DOB", re.compile(
        r"\b(\d{2}[/-]\d{2}[/-][\dCOoQIlSsBg]{2,4}|\d{2}\s?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[/-]?\s?[\dCOoQIlSsBg]{2,4})\b",
        re.IGNORECASE,
    )),
    ("AMOUNT", re.compile(r"(?:Rs\.?|INR|₹)\s?[\d,]+(?:\.\d{2})?", re.IGNORECASE)),
]

_FIELD_FOR_TYPE = {
    "PER": "name",
    "PAN": "pan",
    "AADHAAR": "aadhaar",
    "PASSPORT": "passport",
    "DL": "dl_number",
    "VOTERID": "voter_id",
    "DOB": "dob",
}


# ── Fuzzy ID recovery (OCR digit/letter look-alike swaps) ─────────────────────
# Tesseract commonly swaps a digit for a letter that looks like it (5<->S,
# 0<->O, 1<->I, 2<->Z, 8<->B, 6<->G) even on otherwise clean text -- e.g.
# "ABCPS5678F" gets read as "ABCPSS678F". That one swapped character makes
# the strict format regex above miss the whole ID even though it's still
# fully legible. This fallback only runs when the strict regex found *no*
# match for that entity type, and only accepts a candidate if de-confusing
# it produces a string that itself passes the real format validator -- so
# it can recover an ID that's one look-alike swap away from valid, not
# manufacture one out of arbitrary text.
_CONFUSABLE_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
_CONFUSABLE_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

# Position schema per entity type: 'L' = expects a letter, 'D' = expects a digit.
_FUZZY_SCHEMAS = {
    "PAN": "LLLLLDDDDL",
    "AADHAAR": "D" * 12,
    "PASSPORT": "L" + "D" * 7,
    "VOTERID": "LLL" + "D" * 7,
}
_FUZZY_VALIDATORS = {
    "PAN": lambda s: re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", s),
    "AADHAAR": lambda s: re.match(r"^[0-9]{12}$", s),
    "PASSPORT": lambda s: re.match(r"^[A-Z][0-9]{7}$", s),
    "VOTERID": lambda s: re.match(r"^[A-Z]{3}[0-9]{7}$", s),
}


def _denormalize(candidate, schema):
    chars = list(candidate)
    for i, kind in enumerate(schema):
        c = chars[i]
        if kind == 'D' and c in _CONFUSABLE_TO_DIGIT:
            chars[i] = _CONFUSABLE_TO_DIGIT[c]
        elif kind == 'L' and c in _CONFUSABLE_TO_LETTER:
            chars[i] = _CONFUSABLE_TO_LETTER[c]
    return "".join(chars)


def _fuzzy_id_matches(text, entity_type, claimed):
    """Like the strict pass, only accepts a match against characters no
    earlier pattern (strict or fuzzy) has already claimed, so this can't
    reinterpret part of an already-matched Aadhaar/PAN/etc span as a
    different entity type."""
    schema = _FUZZY_SCHEMAS[entity_type]
    loose = re.compile(r"\b[A-Z0-9](?:\s?[A-Z0-9]){%d}\b" % (len(schema) - 1))
    matches = []
    for m in loose.finditer(text):
        if any(claimed[m.start():m.end()]):
            continue
        candidate = re.sub(r"\s", "", m.group())
        if len(candidate) != len(schema):
            continue
        # A plain English word (zero digits) can coincidentally denormalize
        # into something format-valid purely because O/I/S/B/Z/G are common
        # letters -- e.g. "COMMISSION" -> "COMMI5510N", a bogus PAN. Real
        # OCR misreads of an ID number still get most digits right, so
        # requiring at least one already-correct digit rules out reinterpreting
        # ordinary words while still recovering genuine misreads.
        if not any(ch.isdigit() for ch in candidate):
            continue
        normalized = _denormalize(candidate, schema)
        if _FUZZY_VALIDATORS[entity_type](normalized):
            for i in range(m.start(), m.end()):
                claimed[i] = True
            matches.append(normalized)
    return matches


def _regex_matches(text):
    """Return {entity_type: [matched strings]} using deterministic formats.
    Earlier entries in _ID_PATTERNS take priority over later, overlapping ones
    (e.g. a PAN match blocks a looser pattern from also claiming those chars).
    """
    claimed = [False] * len(text)
    results = {}
    for entity_type, pattern in _ID_PATTERNS:
        for m in pattern.finditer(text):
            if any(claimed[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                claimed[i] = True
            results.setdefault(entity_type, []).append(m.group().strip())

    for entity_type in _FUZZY_SCHEMAS:
        if entity_type in results:
            continue
        fuzzy = _fuzzy_id_matches(text, entity_type, claimed)
        if fuzzy:
            results[entity_type] = fuzzy

    return results


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import BertTokenizerFast, BertForTokenClassification

    if not (MODEL_DIR / "config.json").exists():
        logger.warning(
            "No fine-tuned NER model found at %s -- run `python -m src.train` "
            "first. Falling back to regex-only extraction.", MODEL_DIR,
        )
        return None

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return torch, tokenizer, model


def _model_predict(text):
    """Run the fine-tuned model and return {entity_type: [strings]}."""
    loaded = _load_model()
    if loaded is None:
        return {}
    torch, tokenizer, model = loaded

    words = text.split()
    if not words:
        return {}

    encoding = tokenizer(words, is_split_into_words=True, truncation=True,
                          max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"]).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids()

    results = {}
    current_type, current_words = None, []

    def _flush():
        if current_type and current_words:
            results.setdefault(current_type, []).append(" ".join(current_words))

    prev_word_id = None
    for label_id, word_id in zip(pred_ids, word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        prev_word_id = word_id
        label = model.config.id2label[label_id]
        if label.startswith("B-"):
            _flush()
            current_type, current_words = label[2:], [words[word_id]]
        elif label.startswith("I-") and current_type == label[2:]:
            current_words.append(words[word_id])
        else:
            _flush()
            current_type, current_words = None, []
    _flush()
    return results


def run_ner(text: str) -> dict:
    """Extract name + banking ID entities from OCR'd document text.

    Model predictions are used first (they generalize better for names,
    which have no fixed format); regex fills in any structured ID field the
    model missed and overrides mismatched types, since PAN/Aadhaar/etc. have
    unambiguous formats that are more reliable than the classifier here.
    """
    model_results = _model_predict(text)
    regex_results = _regex_matches(text)

    merged = dict(model_results)
    for entity_type, matches in regex_results.items():
        merged[entity_type] = matches  # regex wins for structured IDs

    def first(entity_type):
        values = merged.get(entity_type)
        return values[0] if values else None

    # Validate and standardise PAN
    pan_val = first("PAN")
    if pan_val:
        pan_val = pan_val.upper().strip()
        if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan_val):
            # Fallback to regex matches
            regex_pans = regex_results.get("PAN", [])
            if regex_pans:
                pan_val = regex_pans[0].upper().strip()
            else:
                pan_val = None
    else:
        regex_pans = regex_results.get("PAN", [])
        if regex_pans:
            pan_val = regex_pans[0].upper().strip()

    # Validate and standardise Aadhaar. When OCR (from combining multiple
    # passes) turns up more than one 12-digit candidate, prefer whichever one
    # actually satisfies the Verhoeff check digit -- a real Aadhaar number
    # always does, so this reliably picks the genuine one over stray digit
    # runs the other patterns/passes picked up.
    regex_aadh = regex_results.get("AADHAAR", [])
    if regex_aadh:
        checksum_valid = [a for a in regex_aadh
                           if verhoeff_checksum_valid("".join(c for c in a if c.isdigit()))]
        aadhaar_val = checksum_valid[0] if checksum_valid else regex_aadh[0]
    else:
        aadhaar_val = first("AADHAAR")
        if aadhaar_val and len("".join(c for c in aadhaar_val if c.isdigit())) != 12:
            aadhaar_val = None

    # Resolve date of birth with cleaning
    dob_val = first("DOB")
    if dob_val:
        dob_val = clean_dob(dob_val)

    # Heuristic fallback for name if model fails or returns junk. The
    # heuristic itself takes the model's raw guess as a tie-break hint (see
    # _extract_name_heuristics) so the two signals reinforce each other
    # instead of the heuristic blindly overriding a correct model call.
    model_name_raw = first("PER")
    heuristic_name = _extract_name_heuristics(text, model_hint=model_name_raw)
    cleaned_model_name = _clean_and_validate_name(model_name_raw) if model_name_raw else None

    if heuristic_name:
        name_val = heuristic_name
    elif cleaned_model_name:
        name_val = cleaned_model_name
    else:
        name_val = None

    confidences = []
    entities = {}
    for entity_type, field in _FIELD_FOR_TYPE.items():
        if entity_type == "DOB":
            value = dob_val
        elif entity_type == "PER":
            value = name_val
        elif entity_type == "PAN":
            value = pan_val
        elif entity_type == "AADHAAR":
            value = aadhaar_val
        else:
            # PASSPORT/DL/VOTERID: when regex found nothing, this falls back
            # to the raw model prediction, which -- unlike PAN/Aadhaar above
            # -- was never format-checked, so a bad model tag (e.g. a two-
            # letter fragment misclassified as a DL span) could reach the
            # API response unvalidated. Re-validate against the real format
            # here and discard anything that doesn't match.
            value = first(entity_type)
            if value:
                value = value.upper().strip()
                if not _STRICT_FORMATS[entity_type].match(value):
                    value = None

        entities[field] = value
        if value:
            confidences.append(1.0 if entity_type not in ("PER", "PAN", "AADHAAR", "DOB") else 0.9)

    entities["amounts"] = merged.get("AMOUNT", [])
    if entities["amounts"]:
        confidences.append(0.95)

    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.3
    entities["confidence"] = round(overall_confidence, 4)
    return entities


# ── Content-based document classification ─────────────────────────────────────
# Filenames from real uploads (phone-camera exports, WhatsApp shares) almost
# never contain hints like "pan" or "aadhaar", so classification has to look
# at what's actually printed on the card. Keyword lists mirror the header
# phrasing used in src/ner_data.py's synthetic templates, which in turn
# mirrors real Indian ID card headers.
_DOC_TYPE_KEYWORDS = [
    ("PAN Card",        ("INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "PAN CARD")),
    ("Aadhaar Card",    ("UNIQUE IDENTIFICATION AUTHORITY", "AADHAAR", "UIDAI")),
    ("Passport",        ("REPUBLIC OF INDIA", "PASSPORT")),
    ("Driving Licence", ("DRIVING LICENCE", "DRIVING LICENSE", "TRANSPORT DEPARTMENT")),
    ("Voter ID",        ("ELECTION COMMISSION", "EPIC NO", "IDENTITY CARD")),
    ("Financial Document", ("STATEMENT OF ACCOUNT", "LOAN SANCTION", "SANCTIONED AMOUNT",
                             "MINIMUM PAYMENT", "BANK STATEMENT")),
]

# A correctly-parsed structured ID number is a much stronger signal than
# header keywords (which OCR mangles easily), so it outweighs several
# keyword hits.
_ENTITY_TO_DOC_TYPE = {
    "pan":       "PAN Card",
    "aadhaar":   "Aadhaar Card",
    "passport":  "Passport",
    "dl_number": "Driving Licence",
    "voter_id":  "Voter ID",
}

REQUIRED_FIELD_FOR_DOC_TYPE = {
    "PAN Card":        "pan",
    "Aadhaar Card":    "aadhaar",
    "Passport":        "passport",
    "Driving Licence": "dl_number",
    "Voter ID":        "voter_id",
}


def classify_document_type(raw_text: str, entities: dict) -> tuple:
    """Classify document type from OCR'd text + already-extracted entities.
    Returns (document_type, confidence)."""
    upper = raw_text.upper()

    scores = {}
    for doc_type, keywords in _DOC_TYPE_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in upper)
        if hits:
            scores[doc_type] = scores.get(doc_type, 0) + hits

    for field, doc_type in _ENTITY_TO_DOC_TYPE.items():
        if entities.get(field):
            scores[doc_type] = scores.get(doc_type, 0) + 3

    if entities.get("amounts"):
        scores["Financial Document"] = scores.get("Financial Document", 0) + 1

    if not scores:
        return "Unknown Document", 0.55

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    confidence = min(0.99, 0.75 + 0.06 * best_score)
    return best_type, round(confidence, 2)
