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
        
    def preprocess_image(self, image_input) -> np.ndarray:
        """
        Loads and preprocesses an image for optimal OCR extraction.
        Applies resize, normalization, color format checks, and thresholding.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = cv2.imread(image_input)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            raise ValueError("image_input must be a file path or bytes")

        if img is None:
            raise ValueError("Failed to decode image data")
            
        # 1. Resize (Standardize scale for better OCR)
        # Assuming typical ID is roughly 1000px wide
        scale = 1000.0 / img.shape[1] 
        dim = (1000, int(img.shape[0] * scale))
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        
        # 2. Color format check (Ensure RGB logic context if needed, though BGR is default cv2)
        # We will natively convert to Gray right away.
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Noise Reduction (Blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. Normalization (0-255 scale optimization)
        norm_img = np.zeros((gray.shape[0], gray.shape[1]))
        normalized = cv2.normalize(blurred, norm_img, 0, 255, cv2.NORM_MINMAX)
        
        # 5. Thresholding
        _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return thresh
        
    def extract_text(self, image_input) -> str:
        """
        Extract raw text from physical document.
        """
        try:
            processed_img = self.preprocess_image(image_input)
            
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
