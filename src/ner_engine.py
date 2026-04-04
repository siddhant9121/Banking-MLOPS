import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class NEREngine:
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        logger.info(f"Initialized NER Engine using {self.model_name}")
        # In a real scenario, we would load the tokenizer and HuggingFace pipeline here:
        # self.nlp_pipeline = pipeline("ner", model=model_name)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities such as Name, PAN, DOB, and Amounts.
        Returns a dictionary of extractions with their confidence scores.
        """
        results = {
            "entities": {},
            "confidence_scores": {},
            "overall_confidence": 1.0
        }
        
        if not text:
            results["overall_confidence"] = 0.0
            return results

        # 1. Regex rule for PAN (Standard Indian Bank KYC Document parameter)
        # Format: 5 Letters, 4 Digits, 1 Letter (e.g. ABCDE1234F)
        pan_match = re.search(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', text)
        if pan_match:
            results["entities"]["PAN"] = pan_match.group(0)
            results["confidence_scores"]["PAN"] = 0.95 # Highly confident for regex strict matches
        else:
            results["entities"]["PAN"] = None
            results["confidence_scores"]["PAN"] = 0.0

        # 2. Regex rule for DOB (Basic implementation DD/MM/YYYY)
        dob_match = re.search(r'\d{2}/\d{2}/\d{4}', text)
        if dob_match:
            results["entities"]["DOB"] = dob_match.group(0)
            results["confidence_scores"]["DOB"] = 0.90
        else:
            results["entities"]["DOB"] = None
            results["confidence_scores"]["DOB"] = 0.0

        # 3. Simulate deep learning extraction for amounts/currency
        amount_match = re.search(r'(Rs\.?|INR|\$)\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)
        if amount_match:
            results["entities"]["Amount"] = amount_match.group(0)
            results["confidence_scores"]["Amount"] = 0.85
        else:
            results["entities"]["Amount"] = None
            results["confidence_scores"]["Amount"] = 0.0
            
        # Mock Transformer fallback logic for names
        name_match = re.search(r'Name:?\s*([A-Za-z]+ \s*[A-Za-z]+)', text, re.IGNORECASE)
        if name_match:
            results["entities"]["Name"] = name_match.group(1).strip()
            results["confidence_scores"]["Name"] = 0.88
        else:
            results["entities"]["Name"] = None
            results["confidence_scores"]["Name"] = 0.0

        # Calculate average overall confidence for routing
        valid_scores = [v for v in results["confidence_scores"].values() if v > 0]
        if valid_scores:
            results["overall_confidence"] = sum(valid_scores) / len(results["confidence_scores"])
        else:
            results["overall_confidence"] = 0.3 # Low confidence fallback
            
        return results
