"""
api/app.py
──────────
FastAPI backend to serve the Open-World Continual Learning model.
Upload an image and receive bounding box predictions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from src.inference import InferenceEngine

app = FastAPI(
    title="OWCL Autonomous Agents API",
    description="Object detection API with Open-Set uncertainty recognition.",
    version="1.0.0"
)

# Allow CORS for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine instance
engine = None

@app.on_event("startup")
def load_model():
    """Load the model weights safely on server startup."""
    global engine
    print("Loading inference engine...")
    try:
        # Loading the base pre-trained model for real-world testing (80 classes)
        engine = InferenceEngine(
            checkpoint_path="yolov8m.pt",
            num_classes=80,
            uncertainty_threshold=0.8
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "OWCL API is running! Ready for inferences."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file upload.
    Returns JSON detection results with 'is_unknown' flags based on entropy.
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
    results = engine.process_image(image_bgr)
    
    return {
        "filename": file.filename,
        "count": len(results),
        "detections": results
    }

if __name__ == "__main__":
    # Start the dev server when run directly
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
