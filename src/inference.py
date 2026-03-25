"""
src/inference.py

Unified inference engine combining YOLOv8 bounding box detection
with entropy-based open-set recognition.
Designed to be consumed directly by the FastAPI backend.
"""

import numpy as np
from typing import Dict, List, Union
from src.models.yolo_detector import YOLODetector
from src.openset.uncertainty import UncertaintyDetector

class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "yolov8m",
        num_classes: int = 4,
        uncertainty_metric: str = "entropy",
        uncertainty_threshold: float = 0.6,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initializes the YOLOv8 detector and Uncertainty scoring module.
        Loads the model weights into memory precisely once.
        """
        # Load object detector
        self.detector = YOLODetector(
            model_size=model_size,
            num_classes=num_classes,
            checkpoint=checkpoint_path
        )
        
        # Load open-set unknown flagger
        self.uncertainty_flagger = UncertaintyDetector(
            metric=uncertainty_metric,
            threshold=uncertainty_threshold,
            num_classes=num_classes
        )
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def process_image(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """
        Runs object detection on a single image and flags unknown objects.
        
        Args:
            image: Image path or numpy HWC array.
            
        Returns:
            List of dictionaries containing bounding boxes, class ID, and uncertainty score.
        """
        # 1. Get raw predictions (requires return_probs=True for open-set analysis)
        raw_detections = self.detector.predict(
            source=image,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            return_probs=True
        )
        
        # 2. Flag high-uncertainty detections as "unknown"
        final_detections = self.uncertainty_flagger.flag_unknowns(raw_detections)
        
        # 3. Clean up the response (remove raw numpy arrays to be JSON serializable)
        cleaned_detections = []
        for det in final_detections:
            if "probs" in det:
                del det["probs"]  # pure probability array is massive and not needed in API
            cleaned_detections.append(det)
            
        return cleaned_detections

# Example usage for testing standalone
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Pass a test image path and checkpoint path as arguments
        img_path = sys.argv[1]
        ckpt = sys.argv[2] if len(sys.argv) > 2 else "runs/detect/runs/continual_ewc/weights/best.pt"
        
        print(f"Loading Inference Engine with checkpoint: {ckpt}")
        engine = InferenceEngine(checkpoint_path=ckpt)
        
        print(f"Processing image: {img_path}")
        results = engine.process_image(img_path)
        
        import json
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python -m src.inference <path/to/image.jpg> [path/to/checkpoint.pt]")
