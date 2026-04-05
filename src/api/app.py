import os
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.staticfiles import StaticFiles
import uvicorn
import yaml

from src.pipeline import DocumentProcessingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = "configs/model_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()
app = FastAPI(
    title=config.get('api', {}).get('title', "Banking Document Verification API"),
    description="Automated ML pipeline for bank KYC and document verification."
)

pipeline = DocumentProcessingPipeline()

class ProcessDocumentResponse(BaseModel):
    prediction: str
    confidence: float
    authenticity_status: str

# Mount static files folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def health_check():
    """Health check endpoint that returns the API status."""
    return {
        "status": "healthy",
        "service": app.title
    }

@app.get("/api/health")
def api_health_check():
    return {"status": "healthy", "service": app.title}
@app.get("/dashboard")
def render_dashboard():
    return FileResponse("frontend/index.html")

@app.post("/dev/process-document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload an image or PDF for KYC extraction and routing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    try:
        # Save memory buffer footprint
        file_bytes = await file.read()
            
        logger.info(f"Received file upload: {file.filename} ({len(file_bytes)} bytes)")
        
        # Process the document using ML pipeline natively in memory
        results = pipeline.process_document(file_bytes)
        
        return ProcessDocumentResponse(**results)
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = config.get('api', {}).get('host', "0.0.0.0")
    port = config.get('api', {}).get('port', 8000)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)