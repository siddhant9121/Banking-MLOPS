import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ner_infer import (
    run_ner,
    classify_document_type,
    verhoeff_checksum_valid,
    looks_like_placeholder_digits,
    REQUIRED_FIELD_FOR_DOC_TYPE,
)


# ── Verhoeff checksum ───────────────────────────────────────────────────────

def test_verhoeff_rejects_wrong_length():
    assert verhoeff_checksum_valid("12345") is False
    assert verhoeff_checksum_valid("1234567890123") is False


def test_verhoeff_rejects_common_placeholder_aadhaar():
    assert verhoeff_checksum_valid("123456789012") is False
    assert verhoeff_checksum_valid("000000000000") is False


def test_verhoeff_accepts_a_valid_checksum_number():
    # Constructed so the Verhoeff check digit (last digit) is correct: the
    # digits 1-11 checksum to a known value, so append that as the 12th
    # digit rather than hand-picking a "real" Aadhaar number.
    from src.ner_infer import _VERHOEFF_D, _VERHOEFF_P

    base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]

    def checksum_digit(digits):
        c = 0
        for i, item in enumerate(reversed(digits + [0])):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][item]]
        # Find the digit d such that appending it makes c end at 0
        for d in range(10):
            c2 = 0
            for i, item in enumerate(reversed(digits + [d])):
                c2 = _VERHOEFF_D[c2][_VERHOEFF_P[i % 8][item]]
            if c2 == 0:
                return d
        raise AssertionError("no valid check digit found")

    check_digit = checksum_digit(base)
    number = "".join(map(str, base)) + str(check_digit)
    assert len(number) == 12
    assert verhoeff_checksum_valid(number) is True


# ── Placeholder digit detection ──────────────────────────────────────────────

def test_looks_like_placeholder_repeated_digits():
    assert looks_like_placeholder_digits("0000000") is True
    assert looks_like_placeholder_digits("1111111") is True


def test_looks_like_placeholder_sequential_digits():
    assert looks_like_placeholder_digits("1234567") is True
    assert looks_like_placeholder_digits("9876543") is True


def test_looks_like_placeholder_rejects_realistic_digits():
    assert looks_like_placeholder_digits("4837261") is False


# ── Content-based document classification ────────────────────────────────────

def test_classify_pan_card_from_content():
    text = "INCOME TAX DEPARTMENT GOVT OF INDIA\nPermanent Account Number\nABCPS1234F"
    doc_type, conf = classify_document_type(text, {"pan": "ABCPS1234F"})
    assert doc_type == "PAN Card"
    assert conf > 0.8


def test_classify_aadhaar_card_from_content():
    text = "GOVERNMENT OF INDIA\nUNIQUE IDENTIFICATION AUTHORITY OF INDIA\n4536 4756 9319"
    doc_type, conf = classify_document_type(text, {"aadhaar": "4536 4756 9319"})
    assert doc_type == "Aadhaar Card"


def test_classify_passport_from_content():
    text = "REPUBLIC OF INDIA PASSPORT\nPassport No\nM1234567"
    doc_type, _ = classify_document_type(text, {"passport": "M1234567"})
    assert doc_type == "Passport"


def test_classify_driving_licence_from_content():
    text = "DRIVING LICENCE UNION OF INDIA\nDL No\nMH12 20110012345"
    doc_type, _ = classify_document_type(text, {"dl_number": "MH12 20110012345"})
    assert doc_type == "Driving Licence"


def test_classify_voter_id_from_content():
    text = "ELECTION COMMISSION OF INDIA IDENTITY CARD\nEPIC No\nABC1234567"
    doc_type, _ = classify_document_type(text, {"voter_id": "ABC1234567"})
    assert doc_type == "Voter ID"


def test_classify_unknown_when_no_signal():
    doc_type, conf = classify_document_type("random unrelated text with no ids", {})
    assert doc_type == "Unknown Document"
    assert conf < 0.7


def test_required_field_mapping_covers_all_id_types():
    for doc_type in ("PAN Card", "Aadhaar Card", "Passport", "Driving Licence", "Voter ID"):
        assert doc_type in REQUIRED_FIELD_FOR_DOC_TYPE


# ── End-to-end NER extraction per card type ──────────────────────────────────

def test_run_ner_extracts_pan_card_fields():
    text = (
        "INCOME TAX DEPARTMENT GOVT OF INDIA\n"
        "RAHUL SHARMA\n"
        "SURESH SHARMA\n"
        "17/07/1985\n"
        "Permanent Account Number\n"
        "ABCPS1234F\n"
        "Signature"
    )
    entities = run_ner(text)
    assert entities["name"] == "RAHUL SHARMA"
    assert entities["pan"] == "ABCPS1234F"
    assert entities["dob"] == "17/07/1985"


def test_run_ner_extracts_aadhaar_card_fields():
    text = (
        "GOVERNMENT OF INDIA\n"
        "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\n"
        "PRIYA PATEL\n"
        "Female\n"
        "Patna Bihar India\n"
        "4536 4756 9319"
    )
    entities = run_ner(text)
    assert entities["name"] == "PRIYA PATEL"
    assert entities["aadhaar"] == "4536 4756 9319"


def test_run_ner_extracts_passport_fields():
    text = (
        "REPUBLIC OF INDIA PASSPORT Type P Country Code IND\n"
        "Passport No\n"
        "M1234567\n"
        "Surname\n"
        "VIKRAM SINGH\n"
        "Date of Birth\n"
        "12 AUG 1990"
    )
    entities = run_ner(text)
    assert entities["name"] == "VIKRAM SINGH"
    assert entities["passport"] == "M1234567"


def test_run_ner_extracts_driving_licence_fields():
    text = (
        "DRIVING LICENCE UNION OF INDIA\n"
        "DL No\n"
        "MH12 20110012345\n"
        "ANITA KUMAR\n"
        "S/O\n"
        "RAMESH KUMAR\n"
        "DOB\n"
        "05-03-1988"
    )
    entities = run_ner(text)
    assert entities["name"] == "ANITA KUMAR"
    assert entities["dl_number"] == "MH12 20110012345"


def test_run_ner_extracts_voter_id_fields():
    text = (
        "ELECTION COMMISSION OF INDIA IDENTITY CARD\n"
        "EPIC No\n"
        "ABC1234567\n"
        "Name\n"
        "DEEPA VERMA\n"
        "DOB\n"
        "21/11/1992"
    )
    entities = run_ner(text)
    assert entities["name"] == "DEEPA VERMA"
    assert entities["voter_id"] == "ABC1234567"


def test_run_ner_recovers_pan_with_digit_letter_confusable():
    # OCR commonly reads '5' as 'S' -- "ABCPS5678F" -> "ABCPSS678F".
    text = "INCOME TAX DEPARTMENT GOVT OF INDIA\nRAHUL SHARMA\nABCPSS678F"
    entities = run_ner(text)
    assert entities["pan"] == "ABCPS5678F"


def test_run_ner_recovers_aadhaar_with_digit_letter_confusable():
    # '0' commonly read as 'O'.
    text = "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\nPRIYA PATEL\n453O 4756 9319"
    entities = run_ner(text)
    assert entities["aadhaar"] == "453047569319"


def test_run_ner_recovers_voter_id_with_digit_letter_confusable():
    text = "ELECTION COMMISSION OF INDIA IDENTITY CARD\nEPIC No\nABC123S567\nDEEPA VERMA"
    entities = run_ner(text)
    assert entities["voter_id"] == "ABC1235567"


def test_fuzzy_id_recovery_does_not_misread_plain_word_as_pan():
    # Regression: "COMMISSION" (from "ELECTION COMMISSION OF INDIA") was
    # being denormalized to "COMMI5510N", which coincidentally passes the
    # PAN format validator, because O/I/S are both common English letters
    # and members of the digit-confusable set.
    text = "ELECTION COMMISSION OF INDIA IDENTITY CARD\nEPIC No\nABC1234567\nDEEPA VERMA"
    entities = run_ner(text)
    assert entities["pan"] is None


def test_fuzzy_id_recovery_does_not_false_positive_on_plain_words():
    # Regression: "GOVERNMENT OF" is 12 alnum characters and was briefly
    # being accepted as a fuzzy Aadhaar match because the validator only
    # checked length, not that the de-confused result was actually numeric.
    text = "GOVERNMENT OF INDIA\nUNIQUE IDENTIFICATION AUTHORITY OF INDIA\nPRIYA PATEL"
    entities = run_ner(text)
    assert entities["aadhaar"] is None


def test_fuzzy_id_recovery_does_not_reclaim_already_matched_span():
    # A strict AADHAAR match should not also be reinterpreted as a fuzzy
    # passport/voter-ID out of the same digits.
    text = "UNIQUE IDENTIFICATION AUTHORITY OF INDIA\nPRIYA PATEL\n4536 4756 9319"
    entities = run_ner(text)
    assert entities["aadhaar"] == "4536 4756 9319"
    assert entities["passport"] is None
    assert entities["voter_id"] is None


def test_run_ner_extracts_name_over_recurring_field_labels():
    # Regression: "MOBILE"/"NO" (from a "Mobile No.: ..." label) weren't in
    # the exclude list, so that reliably-recurring label out-scored the
    # actual name whenever OCR noise merged garbage into one of the three
    # name-line occurrences, dropping its count below the label's.
    text = (
        "GOVERNMENT OF INDIA\n"
        "Siddhant Umesh Singh\n"
        "MALE\n"
        "Mobile No.: 9860547457\n"
        "Siddhant Umesh Singh\n"
        "Mobile No.: 9860547457\n"
        "garbled Siddhant Umesh Singh garbled\n"
        "Mobile No.: 9860547457\n"
    )
    entities = run_ner(text)
    assert entities["name"] == "SIDDHANT UMESH SINGH"


def test_run_ner_extracts_title_case_name_over_recurring_uppercase_junk():
    # Regression: a title-case name ("Cristiano Ronaldo") was invisible to
    # the all-caps-only heuristic, so a recurring but meaningless all-caps
    # OCR fragment ("SITUTS YOUR", from misread header text) won by default
    # even though the real name was clearly legible in every OCR pass.
    text = (
        "GOVERNMENT OF INDIA\n"
        "Cristiano Ronaldo\n"
        "Male\n"
        "SITUTS YOUR\n"
        "SITUTS YOUR\n"
        "Cristiano Ronaldo\n"
        "Cristiano Ronaldo\n"
    )
    entities = run_ner(text)
    assert entities["name"] == "CRISTIANO RONALDO"


def test_run_ner_extracts_name_with_mixed_case_label_prefix():
    # Regression: "Name: AMIT KUMAR" was rejected outright because the
    # mixed-case label "Name:" pushed the whole line's lowercase ratio over
    # the threshold, even though the actual name value is all-caps. Labels
    # must be stripped before casing is judged, not after.
    text = (
        "INCOME TAX DEPARTMENT GOVT OF INDIA\n"
        "PERMANENT ACCOUNT NUMBER CARD\n"
        "Name: AMIT KUMAR\n"
        "Father's Name: RAJESH KUMAR\n"
        "Date of Birth: 15/08/1988\n"
        "Name: AMIT KUMAR\n"
        "Name: AMIT KUMAR\n"
    )
    entities = run_ner(text)
    assert entities["name"] == "AMIT KUMAR"


def test_strict_formats_reject_malformed_ids():
    # Regression: when regex finds no match, run_ner used to fall back to
    # the raw (unvalidated) model prediction for PASSPORT/DL/VOTERID. On
    # messy real OCR text the model tagged a short garbage fragment ("Gf")
    # as a DL span, which reached the API response unvalidated. These
    # formats must reject anything that isn't a real match.
    from src.ner_infer import _STRICT_FORMATS

    assert _STRICT_FORMATS["DL"].match("GF") is None
    assert _STRICT_FORMATS["DL"].match("MH12 20110012345") is not None
    assert _STRICT_FORMATS["PASSPORT"].match("XY") is None
    assert _STRICT_FORMATS["PASSPORT"].match("M1234567") is not None
    assert _STRICT_FORMATS["VOTERID"].match("AB") is None
    assert _STRICT_FORMATS["VOTERID"].match("ABC1234567") is not None


def test_run_ner_ignores_devanagari_header_text_for_name():
    # Regression: with Hindi OCR enabled, raw_text contains real Devanagari
    # header lines (e.g. "आयकर विभाग भारत सरकार"). Devanagari is alphabetic
    # in Unicode but caseless, so an unrestricted isalpha()/islower() check
    # treated a Devanagari line as trivially "all-uppercase" and returned it
    # as a bogus name instead of the real one further down the card.
    text = (
        "आयकर विभाग भारत सरकार\n"
        "INCOME TAX DEPARTMENT GOVT OF INDIA\n"
        "PARTH RAMESH THORAT\n"
        "RAMESH THORAT\n"
        "27/09/2005\n"
        "CPEPT9153G\n"
    )
    entities = run_ner(text)
    assert entities["name"] == "PARTH RAMESH THORAT"


def test_run_ner_name_heuristic_prefers_recurring_candidate_over_first_line():
    # Simulates three concatenated OCR passes: a junk header line that only
    # appears once, and the true name appearing twice (as it would if two of
    # the three OCR passes read it correctly).
    text = (
        "SOME MISREAD HEADER JUNK\n"
        "ROHIT MALHOTRA\n"
        "\n"
        "GARBLED OCR LINE HERE\n"
        "ROHIT MALHOTRA\n"
    )
    entities = run_ner(text)
    assert entities["name"] == "ROHIT MALHOTRA"
