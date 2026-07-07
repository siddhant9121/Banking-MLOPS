"""Synthetic labeled data generator for the banking-document NER model.

Real scanned-ID datasets aren't available for this project, but the entities
we care about (PAN, Aadhaar, passport, driving licence, voter ID, dates,
amounts) all follow fixed, well-documented formats. This module generates
realistic OCR-like sentences from templates with those formats plus random
Indian names, and emits token-level BIO labels so a token-classification
model can be trained without hand-labeled data.
"""
import random

LABELS = ["O"]
for tag in ("PER", "PAN", "AADHAAR", "PASSPORT", "DL", "VOTERID", "DOB", "AMOUNT"):
    LABELS.append(f"B-{tag}")
    LABELS.append(f"I-{tag}")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

FIRST_NAMES = [
    "RAHUL", "PRIYA", "AMIT", "SURESH", "ANITA", "VIJAY", "DEEPA", "SANJAY",
    "KAVITA", "RAJESH", "POOJA", "MANOJ", "SUNITA", "ARJUN", "NEHA", "VIKRAM",
    "SHREYA", "KARAN", "DIVYA", "ROHIT", "MEERA", "ASHOK", "LATA", "GAURAV",
]
LAST_NAMES = [
    "SHARMA", "PATEL", "KUMAR", "SINGH", "GUPTA", "REDDY", "RAO", "NAIR",
    "IYER", "VERMA", "MEHTA", "JOSHI", "DESAI", "AGARWAL", "CHOPRA", "MALHOTRA",
]

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_CONSONANTS = "BCDFGHJKLMNPRSTVW"
_VOWELS = "AEIOU"


def _random_pseudo_word():
    """A random pronounceable-looking word (alternating consonant/vowel
    syllables), used so the model sees names it can't possibly have
    memorized -- real names are far too varied for a small fixed list to
    cover, so the model must learn to recognize a PERSON entity from its
    position and shape (capitalized, alphabetic, next to other name-like
    text) rather than from a closed vocabulary."""
    n_syllables = random.randint(2, 4)
    word = ""
    for _ in range(n_syllables):
        word += random.choice(_CONSONANTS) + random.choice(_VOWELS)
        if random.random() < 0.4:
            word += random.choice(_CONSONANTS)
    return word


def _random_name():
    # Half the time use a real-looking name from the fixed lists, half the
    # time an unseen pseudo-name (or a mix of one real + one pseudo word) --
    # see _random_pseudo_word for why this matters.
    roll = random.random()
    if roll < 0.5:
        first, last = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
    elif roll < 0.75:
        first, last = _random_pseudo_word(), random.choice(LAST_NAMES)
    else:
        first, last = _random_pseudo_word(), _random_pseudo_word()
    return f"{first.upper()} {last.upper()}"


def _random_pan():
    letters1 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    digits = "".join(random.choices("0123456789", k=4))
    letter2 = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"{letters1}{digits}{letter2}"


def _random_aadhaar():
    groups = ["".join(random.choices("0123456789", k=4)) for _ in range(3)]
    return " ".join(groups)


def _random_passport():
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=7))
    return f"{letter}{digits}"


def _random_dl():
    state = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    rto = "".join(random.choices("0123456789", k=2))
    year = str(random.randint(1990, 2023))
    serial = "".join(random.choices("0123456789", k=7))
    return f"{state}{rto} {year}{serial}"


def _random_voterid():
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    digits = "".join(random.choices("0123456789", k=7))
    return f"{letters}{digits}"


def _random_dob(fmt=None):
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1950, 2005)
    fmt = fmt or random.choice(["slash", "dash", "text"])
    if fmt == "slash":
        return f"{day:02d}/{month:02d}/{year}"
    if fmt == "dash":
        return f"{day:02d}-{month:02d}-{year}"
    return f"{day:02d} {MONTHS[month - 1]} {year}"


def _random_amount():
    whole = random.randint(1, 999) * random.choice([100, 1000, 10000])
    formatted = f"{whole:,}"
    prefix = random.choice(["Rs.", "INR", "Rs", "₹"])
    return f"{prefix} {formatted}"


def _tag_span(tokens, labels, text, tag):
    span_tokens = text.split()
    labels.append(f"B-{tag}")
    tokens.append(span_tokens[0])
    for t in span_tokens[1:]:
        tokens.append(t)
        labels.append(f"I-{tag}")


def _tag_plain(tokens, labels, text):
    for t in text.split():
        tokens.append(t)
        labels.append("O")


# Each template is a list of (kind, value_or_None) pieces.
# kind == "text" -> literal filler tokens (label O)
# kind == a tag name -> generated span tagged with that entity type
TEMPLATES = [
    # Real Indian PAN cards print the cardholder's name directly under the
    # header (no "Name:" label), followed by the father's name on its own
    # line (also unlabeled) -- both look identical to the model, so the
    # father's name has to appear as an untagged distractor during training
    # or the model can't learn to prefer the first name-like line.
    lambda: [
        ("text", "INCOME TAX DEPARTMENT GOVT OF INDIA"),
        ("PER", _random_name()),
        ("text", _random_name()),
        ("DOB", _random_dob()),
        ("text", "Permanent Account Number"),
        ("PAN", _random_pan()),
        ("text", "Signature"),
    ],
    lambda: [
        ("text", "INCOME TAX DEPARTMENT GOVT OF INDIA Permanent Account Number Card"),
        ("PAN", _random_pan()),
        ("text", "Name"),
        ("PER", _random_name()),
        ("text", "Date of Birth"),
        ("DOB", _random_dob()),
    ],
    # Real Aadhaar cards likewise print the name directly (no label), with a
    # gender word and address block as unlabeled filler around it.
    lambda: [
        ("text", "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AUTHORITY OF INDIA Aadhaar"),
        ("PER", _random_name()),
        ("text", random.choice(["Male", "Female"])),
        ("text", "Patna Bihar India"),
        ("AADHAAR", _random_aadhaar()),
    ],
    lambda: [
        ("text", "GOVERNMENT OF INDIA UNIQUE IDENTIFICATION AUTHORITY OF INDIA"),
        ("text", "Aadhaar"),
        ("text", "Name"),
        ("PER", _random_name()),
        ("text", "DOB"),
        ("DOB", _random_dob()),
        ("AADHAAR", _random_aadhaar()),
    ],
    lambda: [
        ("text", "REPUBLIC OF INDIA PASSPORT Type P Country Code IND"),
        ("text", "Passport No"),
        ("PASSPORT", _random_passport()),
        ("text", "Surname"),
        ("PER", _random_name()),
        ("text", "Date of Birth"),
        ("DOB", _random_dob("text")),
    ],
    # Passport bio-data page order varies (name printed before the number,
    # with a place-of-birth distractor line) -- give the model a second
    # passport layout so it doesn't only learn the "No -> name" ordering.
    lambda: [
        ("text", "REPUBLIC OF INDIA PASSPORT"),
        ("text", "Given Name"),
        ("PER", _random_name()),
        ("text", "Place of Birth Mumbai Maharashtra"),
        ("text", "Date of Birth"),
        ("DOB", _random_dob("text")),
        ("text", "Passport No"),
        ("PASSPORT", _random_passport()),
    ],
    lambda: [
        ("text", "DRIVING LICENCE UNION OF INDIA"),
        ("text", "DL No"),
        ("DL", _random_dl()),
        ("PER", _random_name()),
        ("text", "S/O"),
        ("text", _random_name()),
        ("text", "DOB"),
        ("DOB", _random_dob("dash")),
    ],
    # DL cards also print with the name first and a blood-group/address
    # filler block instead of "S/O", so the model sees both orderings.
    lambda: [
        ("text", "TRANSPORT DEPARTMENT DRIVING LICENCE"),
        ("PER", _random_name()),
        ("text", "DOB"),
        ("DOB", _random_dob("slash")),
        ("text", "Blood Group O+ Address Pune Maharashtra"),
        ("text", "DL No"),
        ("DL", _random_dl()),
    ],
    lambda: [
        ("text", "ELECTION COMMISSION OF INDIA IDENTITY CARD"),
        ("text", "EPIC No"),
        ("VOTERID", _random_voterid()),
        ("text", "Name"),
        ("PER", _random_name()),
        ("text", "DOB"),
        ("DOB", _random_dob()),
    ],
    # Voter ID cards also print father's/husband's name as an unlabeled
    # distractor line, similar to PAN -- important so the model learns to
    # ignore it here too, not just on PAN templates.
    lambda: [
        ("text", "ELECTION COMMISSION OF INDIA IDENTITY CARD"),
        ("text", "Elector's Name"),
        ("PER", _random_name()),
        ("text", "Relation's Name"),
        ("text", _random_name()),
        ("text", "EPIC No"),
        ("VOTERID", _random_voterid()),
        ("text", "DOB"),
        ("DOB", _random_dob("dash")),
    ],
    lambda: [
        ("text", "STATEMENT OF ACCOUNT"),
        ("text", "Account Holder"),
        ("PER", _random_name()),
        ("text", "Total Amount Due"),
        ("AMOUNT", _random_amount()),
        ("text", "Minimum Payment"),
        ("AMOUNT", _random_amount()),
    ],
    lambda: [
        ("text", "LOAN SANCTION LETTER"),
        ("text", "Applicant Name"),
        ("PER", _random_name()),
        ("text", "Sanctioned Amount"),
        ("AMOUNT", _random_amount()),
        ("text", "PAN"),
        ("PAN", _random_pan()),
    ],
]


# Tesseract's most common misreads are digit/letter look-alikes -- the same
# confusable set src/ner_infer.py's clean_dob() already corrects for dates.
# Injecting these into training text (rather than training on pristine
# strings only) teaches the model to still recognize an entity's shape
# under realistic OCR noise instead of only ever having seen clean text.
_OCR_CONFUSABLES = {
    '0': 'O', 'O': '0', '1': 'I', 'I': '1', 'l': '1',
    '5': 'S', 'S': '5', '8': 'B', 'B': '8', '2': 'Z', 'Z': '2',
    '6': 'G', 'G': '6',
}


def _add_ocr_noise(token, char_error_rate):
    chars = list(token)
    for i, c in enumerate(chars):
        if c in _OCR_CONFUSABLES and random.random() < char_error_rate:
            chars[i] = _OCR_CONFUSABLES[c]
    return "".join(chars)


def _apply_ocr_noise(tokens, char_error_rate):
    return [_add_ocr_noise(t, char_error_rate) for t in tokens]


def _generate_example(noise_prob, char_error_rate):
    template = random.choice(TEMPLATES)()
    tokens, labels = [], []
    for kind, value in template:
        if kind == "text":
            _tag_plain(tokens, labels, value)
        else:
            _tag_span(tokens, labels, value, kind)
    if random.random() < noise_prob:
        tokens = _apply_ocr_noise(tokens, char_error_rate)
    return tokens, labels


def generate_dataset(n_examples, seed=None, noise_prob=0.4, char_error_rate=0.08):
    """Return a list of (tokens, bio_labels) pairs.

    `noise_prob` of examples get light OCR-style character noise injected
    (see _apply_ocr_noise) at `char_error_rate` per confusable character, so
    the trained model doesn't only ever see pristine text -- real card
    photos OCR with a non-trivial per-character error rate, and a model
    trained exclusively on clean synthetic text generalizes poorly to that.
    """
    rng_state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        return [_generate_example(noise_prob, char_error_rate) for _ in range(n_examples)]
    finally:
        if seed is not None:
            random.setstate(rng_state)
