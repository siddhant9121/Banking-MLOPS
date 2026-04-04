import cv2
import numpy as np
import pytesseract
import logging
import os

logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, engine_type='tesseract'):
        self.engine_type = engine_type
        logger.info(f"Initialized OCR Engine using {self.engine_type}")
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Loads and preprocesses an image for optimal OCR extraction.
        Grayscales and applies binary thresholding.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        # Using simple Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return thresh
        
    def extract_text(self, image_path: str) -> str:
        """
        Extract raw text from physical document.
        """
        try:
            processed_img = self.preprocess_image(image_path)
            
            if self.engine_type == 'tesseract':
                # Assuming tesseract is available in system PATH
                text = pytesseract.image_to_string(processed_img)
            else:
                # Placeholder for paddleOCR or others
                text = "Extracted text: [PaddleOCR is not implemented yet]"
                
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text using {self.engine_type}: {str(e)}")
            return ""
