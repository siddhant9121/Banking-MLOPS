import os
import re
import io
import uuid
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy
import pytesseract
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ner_infer import (
    run_ner,
    classify_document_type,
    verhoeff_checksum_valid,
    looks_like_placeholder_digits,
    REQUIRED_FIELD_FOR_DOC_TYPE,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

# pytesseract needs to find the Tesseract-OCR engine binary; it isn't on
# PATH by default on Windows, so point it at the standard install location.
if shutil.which("tesseract") is None:
    _default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _default_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_default_tesseract)

# Real Indian ID cards mix Devanagari (Hindi) header text with the Latin
# text/numbers we actually care about. English-only OCR tries to force
# every Devanagari glyph into an English letter shape, which garbles that
# whole page region and was observed to also throw off segmentation of
# the *English* lines (name, PAN/ID number) elsewhere on the card. A local
# tessdata dir (not the system one, which needs admin rights to modify)
# with both eng + hin trained data fixes this; if hin.traineddata isn't
# present, fall back to English-only rather than failing OCR entirely.
_TESSDATA_DIR = Path(__file__).parent.parent / "tessdata"
if (_TESSDATA_DIR / "hin.traineddata").exists() and (_TESSDATA_DIR / "eng.traineddata").exists():
    OCR_LANG = "eng+hin"
    # Set the env var directly rather than passing --tessdata-dir through
    # pytesseract's config string: that string gets tokenized before
    # reaching tesseract, which mangles a Windows path containing both a
    # space ("proj new") and backslashes.
    os.environ["TESSDATA_PREFIX"] = str(_TESSDATA_DIR)
else:
    OCR_LANG = "eng"

app = FastAPI(
    title="Banking Document Processing API",
    description=(
        "ML-powered OCR + NER pipeline for KYC, loan processing, and compliance."
    ),
    version="1.0.0",
)

# Serve static frontend files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory review queue (swap for DB in production) ────────────────────────
_review_queue: List[dict] = []

CONFIDENCE_THRESHOLD = 0.85
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    name:      Optional[str] = None
    pan:       Optional[str] = None
    aadhaar:   Optional[str] = None
    passport:  Optional[str] = None
    dl_number: Optional[str] = None
    voter_id:  Optional[str] = None
    dob:       Optional[str] = None
    amounts:   List[str] = []


class ExtractionResult(BaseModel):
    document_id:        str
    document_type:      str
    extracted_entities: ExtractedEntities
    confidence_score:   float
    status:             str
    timestamp:          str
    authenticity:       str = "unknown"


class ReviewQueueResponse(BaseModel):
    queue_length: int
    items:        List[dict]


# ── Document classification / OCR / NER helpers ────────────────────────────────

def _classify_document(raw_text: str, entities: dict, filename: str) -> tuple:
    """Classify document type from OCR'd content first -- real uploads
    (phone-camera exports, WhatsApp shares) almost never have "pan" or
    "aadhaar" in the filename, so filename sniffing alone was misclassifying
    most real-world uploads as "Unknown Document". Filename hints are kept
    only as a last-resort fallback for cases with no OCR text at all (e.g.
    unsupported PDFs)."""
    if raw_text.strip():
        return classify_document_type(raw_text, entities)

    name = filename.lower()
    if any(k in name for k in ("kyc", "pan", "aadhaar", "passport", "id", "licence", "license", "voter")):
        return "KYC Document", 0.6
    if any(k in name for k in ("loan", "statement", "financial", "bank")):
        return "Financial Document", 0.6
    return "Unknown Document", 0.5


OCR_TARGET_WIDTH = 1600
OCR_BINARIZE_THRESHOLD = 100  # fallback only -- see _otsu_threshold


def _otsu_threshold(gray_image: Image.Image) -> int:
    """Pick the binarize cutoff from the image's own brightness histogram
    (Otsu's method) instead of a single fixed value. A fixed threshold
    tuned around one card's background (e.g. PAN's plain white) system-
    atically mis-binarizes cards with different backgrounds -- Aadhaar's
    pale blue, Voter ID's tinted pattern, passport security print -- which
    was a major source of OCR (and thus extraction) failures on non-PAN
    cards."""
    histogram = numpy.array(gray_image.histogram(), dtype=numpy.float64)
    total = histogram.sum()
    if total == 0:
        return OCR_BINARIZE_THRESHOLD

    levels = numpy.arange(256)
    sum_all = float(numpy.dot(levels, histogram))

    weight_bg = numpy.cumsum(histogram)
    sum_bg = numpy.cumsum(levels * histogram)
    weight_fg = total - weight_bg

    best_variance, best_threshold = -1.0, OCR_BINARIZE_THRESHOLD
    for t in range(255):
        wb, wf = weight_bg[t], weight_fg[t]
        if wb == 0 or wf == 0:
            continue
        mean_bg = sum_bg[t] / wb
        mean_fg = (sum_all - sum_bg[t]) / wf
        variance_between = wb * wf * (mean_bg - mean_fg) ** 2
        if variance_between > best_variance:
            best_variance, best_threshold = variance_between, t
    return best_threshold


def _crop_to_content(image: Image.Image, threshold: int = 245, padding_frac: float = 0.03) -> Image.Image:
    """Real-world card photos/screenshots are often centered on a large
    uniform white margin (padding). That margin's sheer pixel count can
    dominate Otsu's histogram and drag the binarize threshold up near pure
    white, causing the *entire* card -- including its own light-colored
    background -- to collapse into "foreground", an inverted/unreadable
    result. Cropping to the actual content's bounding box first keeps the
    thresholding statistics representative of the card itself rather than
    the surrounding whitespace."""
    gray = ImageOps.grayscale(image)
    arr = numpy.array(gray)
    mask = arr < threshold
    if not mask.any():
        return image
    ys, xs = numpy.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width, height = image.size
    pad_x = int((x1 - x0) * padding_frac)
    pad_y = int((y1 - y0) * padding_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Real ID-card photos (textured/gradient backgrounds, low-res phone
    exports) OCR very poorly as-is -- tesseract returns nothing at all on an
    untouched photo of a card. Upscaling small images, stretching contrast,
    and binarizing to black-on-white text isolates the dark printed text
    from the background and reliably fixes this."""
    width, height = image.size
    if width < OCR_TARGET_WIDTH:
        # Integer division here used to silently produce scale=1 (a no-op)
        # for any width in ~800-1599px -- a huge share of real phone-camera
        # exports -- because OCR_TARGET_WIDTH // width truncates to 1 well
        # before width actually reaches the target.
        scale = OCR_TARGET_WIDTH / width
        image = image.resize((OCR_TARGET_WIDTH, round(height * scale)), Image.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    threshold = _otsu_threshold(gray)
    binarized = gray.point(lambda p: 255 if p > threshold else 0)

    # Normalize polarity to black-text-on-white, which is what tesseract is
    # tuned for. Otsu only finds a split point; it doesn't know which side
    # is "text" vs "background". On a textured/gradient card background
    # (common on real ID photos) that split can go either way, occasionally
    # leaving the whole background black with white text -- unreadable to
    # tesseract even though the split itself was mathematically fine. Text
    # is always the minority of pixels on a real document, so if black
    # pixels are the majority, the polarity is inverted and needs flipping.
    black_fraction = (numpy.array(binarized) == 0).mean()
    if black_fraction > 0.5:
        binarized = ImageOps.invert(binarized)
    return binarized


def _run_ocr(file_bytes: bytes, filename: str, content_type: str) -> dict:
    if content_type == "application/pdf":
        # PDF -> image rasterization isn't wired up yet (no poppler/pdf2image
        # in this environment); OCR is only implemented for image uploads.
        logger.warning("OCR for PDF uploads (%s) is not yet supported", filename)
        return {"raw_text": "", "confidence": 0.0}

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = _crop_to_content(image)

        # Pass 1: Original binarized OCR (PSM 6, original scale)
        processed_bin = _preprocess_for_ocr(image)
        text_bin = pytesseract.image_to_string(
            processed_bin, lang=OCR_LANG, config="--psm 6")

        # Pass 2: Grayscale OCR (PSM 3, 2x scale for small fonts/names)
        gray = ImageOps.grayscale(image)
        w, h = gray.size
        gray_x2 = gray.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        text_gray_psm3 = pytesseract.image_to_string(
            gray_x2, lang=OCR_LANG, config="--psm 3")

        # Pass 3: Grayscale OCR (PSM 6, 2x scale for layout-agnostic texts/dates)
        text_gray_psm6 = pytesseract.image_to_string(
            gray_x2, lang=OCR_LANG, config="--psm 6")
        
        # Combine all texts (cleanest grayscale text first to prioritize name heuristics)
        raw_text = "\n".join([text_gray_psm3, text_bin, text_gray_psm6])
    except Exception:
        logger.exception("OCR failed for %s", filename)
        return {"raw_text": "", "confidence": 0.0}

    confidence = 0.9 if raw_text.strip() else 0.3
    return {"raw_text": raw_text, "confidence": confidence}


def _run_ner(raw_text: str) -> dict:
    return run_ner(raw_text)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"service": "Banking Document Processing API", "status": "healthy",
            "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/process-document", response_model=ExtractionResult, tags=["Processing"])
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a KYC or financial document and extract entities."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    doc_id     = str(uuid.uuid4())
    file_bytes = await file.read()

    ocr_result           = _run_ocr(file_bytes, file.filename or "", file.content_type)
    ner_result           = _run_ner(ocr_result["raw_text"])
    doc_type,  clf_conf  = _classify_document(ocr_result["raw_text"], ner_result, file.filename or "")

    overall_confidence = round(
        (clf_conf + ocr_result["confidence"] + ner_result["confidence"]) / 3, 4
    )

    # Determine authenticity
    authenticity = "original"
    reasons = []

    pan = ner_result.get("pan")
    if pan:
        if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan):
            authenticity = "fake"
            reasons.append("Invalid PAN format")
        elif pan in ("ABCDE1234F", "BWZPS1234R", "XYZAB1234C") or "1234" in pan or "0000" in pan:
            authenticity = "fake"
            reasons.append("Placeholder/Dummy PAN number detected")

    aadhaar = ner_result.get("aadhaar")
    if aadhaar:
        clean_aadhaar = "".join(c for c in aadhaar if c.isdigit())
        if len(clean_aadhaar) != 12:
            authenticity = "fake"
            reasons.append("Invalid Aadhaar length")
        elif clean_aadhaar in ("123456789012", "000000000000", "123412341234"):
            authenticity = "fake"
            reasons.append("Placeholder/Dummy Aadhaar number detected")
        elif not verhoeff_checksum_valid(clean_aadhaar):
            # Real UIDAI Aadhaar numbers always satisfy this checksum; a
            # correctly-formatted but made-up/mistyped number won't.
            authenticity = "fake"
            reasons.append("Aadhaar number fails checksum validation")

    passport = ner_result.get("passport")
    if passport and looks_like_placeholder_digits(passport[1:]):
        authenticity = "fake"
        reasons.append("Placeholder/Dummy passport number detected")

    dl_number = ner_result.get("dl_number")
    if dl_number and looks_like_placeholder_digits("".join(c for c in dl_number if c.isdigit())):
        authenticity = "fake"
        reasons.append("Placeholder/Dummy driving licence number detected")

    voter_id = ner_result.get("voter_id")
    if voter_id and looks_like_placeholder_digits(voter_id[3:]):
        authenticity = "fake"
        reasons.append("Placeholder/Dummy voter ID number detected")

    name = ner_result.get("name")
    dob = ner_result.get("dob")

    # Every supported ID card type has one defining structured number
    # (PAN/Aadhaar/passport/DL/voter-ID); a document classified as that type
    # but missing its own number, name, or DOB didn't actually yield a
    # trustworthy read and should be routed for manual review rather than
    # marked "original".
    required_field = REQUIRED_FIELD_FOR_DOC_TYPE.get(doc_type)
    if required_field:
        if not ner_result.get(required_field):
            authenticity = "fake"
            reasons.append(f"Expected {required_field.replace('_', ' ')} not found on document")
        if not name:
            authenticity = "fake"
            reasons.append("Missing or invalid cardholder name")
        if not dob:
            authenticity = "fake"
            reasons.append("Missing or invalid Date of Birth")

    if authenticity == "fake":
        status = "manual_review"
        flagged_reason = "Fake ID detected: " + ", ".join(reasons)
        background_tasks.add_task(
            _enqueue_for_review, doc_id, flagged_reason, overall_confidence
        )
    elif overall_confidence >= CONFIDENCE_THRESHOLD:
        status = "automated"
    else:
        status = "manual_review"
        background_tasks.add_task(
            _enqueue_for_review, doc_id, "Low confidence score", overall_confidence
        )

    return ExtractionResult(
        document_id=doc_id,
        document_type=doc_type,
        extracted_entities=ExtractedEntities(
            name=ner_result["name"],
            pan=ner_result["pan"],
            aadhaar=ner_result["aadhaar"],
            passport=ner_result["passport"],
            dl_number=ner_result["dl_number"],
            voter_id=ner_result["voter_id"],
            dob=ner_result["dob"],
            amounts=ner_result["amounts"],
        ),
        confidence_score=overall_confidence,
        status=status,
        timestamp=datetime.now().isoformat(),
        authenticity=authenticity,
    )


@app.get("/review-queue", response_model=ReviewQueueResponse, tags=["Manual Review"])
def get_review_queue():
    return ReviewQueueResponse(queue_length=len(_review_queue), items=_review_queue)


def _remove_from_queue(document_id: str) -> None:
    global _review_queue
    before        = len(_review_queue)
    _review_queue = [i for i in _review_queue if i["document_id"] != document_id]
    if len(_review_queue) == before:
        raise HTTPException(status_code=404, detail="Document not found in review queue")


@app.delete("/review-queue/{document_id}", tags=["Manual Review"])
def resolve_review(document_id: str):
    _remove_from_queue(document_id)
    return {"message": f"Document {document_id} resolved"}


@app.post("/review-queue/{document_id}/reject", tags=["Manual Review"])
def reject_review(document_id: str):
    _remove_from_queue(document_id)
    return {"message": f"Document {document_id} rejected"}


# ── Background task ───────────────────────────────────────────────────────────

def _enqueue_for_review(doc_id: str, reason: str, confidence: float) -> None:
    _review_queue.append({
        "document_id":      doc_id,
        "reason":           reason,
        "confidence_score": confidence,
        "timestamp":        datetime.now().isoformat(),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
