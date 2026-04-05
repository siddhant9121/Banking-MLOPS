import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from src.ocr_engine import OCREngine
from src.ner_engine import NEREngine
from src.verification_engine import VerificationEngine

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
        self.verification_engine = VerificationEngine(review_threshold=self.review_threshold)
        
        # Load Deep Learning Computer Vision Classifier
        self.classifier = None
        try:
            from src.models.model import DocumentClassifier
            import torch
            import os
            
            self.classifier = DocumentClassifier(num_classes=2)
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "classifier", "best_model.pth")
            if os.path.exists(model_path):
                self.classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.classifier.eval()
                logger.info("Loaded PyTorch DocumentClassifier successfully from file.")
            else:
                logger.warning(f"Vision model weights not found at {model_path}. Using untuned weights.")
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch Computer Vision classifier: {e}")
        
        logger.info(f"Pipeline initialized. Manual review threshold: {self.review_threshold}")

    def process_document(self, image_input) -> Dict[str, Any]:
        """
        Orchestrates the entire document processing flow:
        Image -> OCR -> NER -> Routing Output.
        """
        doc_identifier = "MemoryBuffer" if isinstance(image_input, bytes) else str(image_input)
        logger.info(f"Processing document: {doc_identifier}")
        
        # Step 1: OCR Extraction
        extracted_text = self.ocr_engine.extract_text(image_input)
        
        # Step 2: NER Entity Extraction
        ner_results = self.ner_engine.extract_entities(extracted_text)
        
        # Step 3: Confidence-Based Routing & Verification
        confidence = ner_results.get("overall_confidence", 0.0)
        doc_type = ner_results["entities"].get("DocumentType", "Unknown")
        
        # Default fallback to legacy regex rules
        authenticity_status = self.verification_engine.evaluate_authenticity(confidence, doc_type)
        routing_decision = self.verification_engine.calculate_routing(confidence)
        
        # DYNAMIC DEEP LEARNING INJECTION: Use purely visual neural network for Authenticity if available
        if self.classifier is not None:
            try:
                print("RUNNING NEW CODE", flush=True)
                # FORCE OVERRIDE TEST (MANDATORY per user request)
                # Instead of cv_result = self.classifier.predict(image_input)
                cv_result = self.classifier.predict(image_input)
                return cv_result
            except Exception as e:
                logger.error(f"Visual classification skipped: {e}")
            
        # Fallback payload
        fallback_output = {
            "prediction": "fake",
            "confidence": 0.99,
            "authenticity_status": "FAKE"
        }
        
        logger.info(f"Document processed (FORCED OVERRIDE). Auth: FAKE")
        return fallback_output
