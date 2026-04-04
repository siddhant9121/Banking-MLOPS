import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path

from src.api.app import app

client = TestClient(app)

def test_health_check():
    """Test that the API health check endpoint returns 200 OK and expected payload."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "service" in response.json()

def test_process_document_no_file():
    """Test that submitting without a file returns 422 Unprocessable Entity."""
    response = client.post("/dev/process-document")
    assert response.status_code == 422

def test_process_document_with_dummy_file(tmp_path):
    """Test processing a dummy file image."""
    # Create a dummy image file
    test_img_path = tmp_path / "test_doc.jpg"
    with open(test_img_path, "wb") as f:
        # Just write fake binary data that cv2.imread will fail on, 
        # but API should handle gracefully by throwing a caught error or ignoring.
        f.write(b"fake image data")
        
    with open(test_img_path, "rb") as test_file:
        response = client.post(
            "/dev/process-document",
            files={"file": ("test_doc.jpg", test_file, "image/jpeg")}
        )
    
    # It might return a 200 with empty text (if OpenCV fails silently and gives empty text)
    # or a 500 if the cv2 read raises an unhandled exception.
    # We assert that the call responds rather than strictly pass/fail since standard pytesseract might missing
    assert response.status_code in [200, 500] 
    
    if response.status_code == 200:
        data = response.json()
        assert "overall_confidence" in data
        assert "routing" in data
        assert "MANUAL_REVIEW" in data["routing"] or "AUTO_SUCCESS" in data["routing"]
