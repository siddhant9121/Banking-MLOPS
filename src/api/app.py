import os
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
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
TEMP_DIR = Path(config.get('pipeline', {}).get('temp_upload_dir', '/tmp/banking_uploads'))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files folder
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root_health_check():
    return {"status": "healthy", "service": app.title}

@app.get("/api/health")
def api_health_check():
    return {"status": "healthy", "service": app.title}

@app.post("/dev/process-document")
async def process_document_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload an image or PDF for KYC extraction and routing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    temp_path = TEMP_DIR / file.filename
    
    try:
        # Save uploaded file safely
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Received file upload: {file.filename}")
        
        # Process the document using ML pipeline
        results = pipeline.process_document(str(temp_path))
        
        # Clean up temp file
        temp_path.unlink()
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    host = config.get('api', {}).get('host', "0.0.0.0")
    port = config.get('api', {}).get('port', 8000)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
