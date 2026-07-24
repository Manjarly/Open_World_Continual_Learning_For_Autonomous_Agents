"""
api/app.py
──────────
FastAPI backend to serve the Open-World Continual Learning model.
Upload an image and receive bounding box predictions with open-set flags.
"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from src.inference import InferenceEngine

# Global inference engine instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights safely on server startup."""
    global engine
    print("Loading OWCL inference engine...")
    try:
        engine = InferenceEngine(
            checkpoint_path="yolov8m.pt",
            num_classes=80,
            uncertainty_threshold=0.6,
            uncertainty_metric="entropy"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down OWCL inference engine...")

app = FastAPI(
    title="OWCL Autonomous Agents API",
    description="Object detection API with Open-Set uncertainty recognition and continual learning support.",
    version="1.1.0",
    lifespan=lifespan
)

# Allow CORS for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "OWCL API is running! Ready for inferences.",
        "model_loaded": engine is not None and engine.detector is not None,
    }


@app.get("/config")
def get_configuration():
    """Return active model configuration and uncertainty thresholds."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load.")
    return engine.get_config()


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Confidence threshold"),
    uncertainty_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Uncertainty threshold"),
    metric: Optional[str] = Query(None, description="Uncertainty metric ('entropy', 'max_softmax', 'energy')"),
):
    """
    Accepts an image file upload.
    Returns JSON detection results with 'is_unknown' flags based on open-set entropy.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load.")
        
    # Read the uploaded file asynchronously
    contents = await file.read()
    
    # Convert file bytes to a numpy array, then to a BGR image (OpenCV format)
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file sent.")
        
    # Process through the unified inference engine
    results = engine.process_image(
        image_bgr,
        conf_threshold=conf_threshold,
        uncertainty_threshold=uncertainty_threshold,
        uncertainty_metric=metric,
    )
    
    unknown_count = sum(1 for d in results if d.get("is_unknown", False))
    
    return {
        "filename": file.filename,
        "count": len(results),
        "unknown_count": unknown_count,
        "detections": results,
        "parameters": {
            "conf_threshold": conf_threshold or engine.conf_threshold,
            "uncertainty_threshold": uncertainty_threshold or engine.uncertainty_flagger.threshold,
            "metric": metric or engine.uncertainty_flagger.metric,
        }
    }


if __name__ == "__main__":
    # Start the dev server when run directly
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

