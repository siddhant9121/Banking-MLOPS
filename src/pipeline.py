import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from src.ocr_engine import OCREngine
from src.ner_engine import NEREngine

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    def __init__(self, config_path="configs/model_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        ocr_type = self.config.get('pipeline', {}).get('ocr_engine', 'tesseract')
        ner_type = self.config.get('ner', {}).get('model_name', 'bert-base-uncased')
        self.review_threshold = self.config.get('pipeline', {}).get('manual_review_threshold', 0.80)
        
        # Initialize engines
        self.ocr_engine = OCREngine(engine_type=ocr_type)
        self.ner_engine = NEREngine(model_name=ner_type)
        
        logger.info(f"Pipeline initialized. Manual review threshold: {self.review_threshold}")

    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Orchestrates the entire document processing flow:
        Image -> OCR -> NER -> Routing Output.
        """
        logger.info(f"Processing document: {image_path}")
        
        # Step 1: OCR Extraction
        extracted_text = self.ocr_engine.extract_text(image_path)
        
        # Step 2: NER Entity Extraction
        ner_results = self.ner_engine.extract_entities(extracted_text)
        
        # Step 3: Confidence-Based Routing
        confidence = ner_results.get("overall_confidence", 0.0)
        requires_manual_review = confidence < self.review_threshold
        
        # Prepare Output payload
        output = {
            "document_path": str(image_path),
            "raw_text_extracted": extracted_text,
            "entities": ner_results["entities"],
            "confidence_scores": ner_results["confidence_scores"],
            "overall_confidence": round(confidence, 4),
            "routing": "MANUAL_REVIEW" if requires_manual_review else "AUTO_SUCCESS",
            "threshold_used": self.review_threshold
        }
        
        logger.info(f"Document processed. Routing to: {output['routing']}")
        return output
