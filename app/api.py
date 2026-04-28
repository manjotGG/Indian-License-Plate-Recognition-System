"""
FastAPI service for ALPR system
Provides REST API endpoints for license plate recognition
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from pathlib import Path
import tempfile
import uvicorn
from typing import List, Dict
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.core.lpr_system import ImprovedIndianLPRSystem
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Indian ALPR API",
    description="Automatic License Plate Recognition API for Indian plates",
    version="1.0.0"
)

# Initialize ALPR system
lpr_system = ImprovedIndianLPRSystem(device='cpu', enable_database=True)

@app.post("/api/v1/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    """
    Recognize license plates in uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        JSON response with recognition results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process with ALPR system
        result = lpr_system.process_image_from_array(image, filename=file.filename)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
async def get_plate_history(plate_text: str = None, limit: int = 50):
    """
    Get plate detection history
    
    Args:
        plate_text: Specific plate to search for
        limit: Maximum number of results
    
    Returns:
        List of historical detections
    """
    try:
        if lpr_system.database:
            history = lpr_system.database.get_plate_history(plate_text, limit)
            return {"history": history}
        else:
            return {"history": [], "message": "Database not enabled"}
            
    except Exception as e:
        logger.error(f"History API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/statistics")
async def get_statistics():
    """
    Get system statistics
    
    Returns:
        Database statistics
    """
    try:
        if lpr_system.database:
            stats = lpr_system.database.get_statistics()
            return {"statistics": stats}
        else:
            return {"statistics": {}, "message": "Database not enabled"}
            
    except Exception as e:
        logger.error(f"Statistics API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ALPR API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)