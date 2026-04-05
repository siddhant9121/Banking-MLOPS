import logging

logger = logging.getLogger(__name__)

class VerificationEngine:
    def __init__(self, review_threshold=0.80):
        self.review_threshold = review_threshold
        logger.info(f"Initialized Verification Engine (Threshold: {self.review_threshold})")
        
    def evaluate_authenticity(self, overall_confidence: float, doc_type: str) -> str:
        """
        Determines if a document is likely a fake, suspicious, or verified real
        based on the ML confidence metrics parsing its text payload.
        """
        if overall_confidence < 0.3 or doc_type == "Unknown":
            return "FAKE / UNRECOGNIZED"
        elif overall_confidence < self.review_threshold:
            return "SUSPICIOUS"
        else:
            return "VERIFIED REAL"

    def calculate_routing(self, overall_confidence: float) -> str:
        """
        Calculates backend routing logic (whether human QA needs to step in).
        """
        if overall_confidence < self.review_threshold:
            return "MANUAL_REVIEW"
        return "AUTO_SUCCESS"
