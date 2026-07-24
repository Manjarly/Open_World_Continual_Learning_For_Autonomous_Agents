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
        checkpoint_path: str = "yolov8m.pt",
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
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

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

    def update_config(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        uncertainty_threshold: Optional[float] = None,
        uncertainty_metric: Optional[str] = None,
    ):
        """Dynamically update threshold settings or metric."""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if uncertainty_threshold is not None:
            self.uncertainty_flagger.threshold = uncertainty_threshold
        if uncertainty_metric is not None and uncertainty_metric in ("entropy", "max_softmax", "energy"):
            self.uncertainty_flagger.metric = uncertainty_metric

    def get_config(self) -> Dict:
        """Return active engine configuration."""
        return {
            "num_classes": self.num_classes,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "uncertainty_threshold": self.uncertainty_flagger.threshold,
            "uncertainty_metric": self.uncertainty_flagger.metric,
        }
        
    def process_image(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: Optional[float] = None,
        uncertainty_threshold: Optional[float] = None,
        uncertainty_metric: Optional[str] = None,
    ) -> List[Dict]:
        """
        Runs object detection on a single image and flags unknown objects.
        
        Args:
            image: Image path or numpy HWC array.
            conf_threshold: Optional override for confidence threshold.
            uncertainty_threshold: Optional override for uncertainty threshold.
            uncertainty_metric: Optional override for uncertainty metric.
            
        Returns:
            List of dictionaries containing bounding boxes, class ID, and uncertainty score.
        """
        conf_t = conf_threshold if conf_threshold is not None else self.conf_threshold
        unc_t = uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_flagger.threshold
        metric = uncertainty_metric if uncertainty_metric is not None else self.uncertainty_flagger.metric

        # Temporarily adapt flagger settings if overridden
        orig_thresh = self.uncertainty_flagger.threshold
        orig_metric = self.uncertainty_flagger.metric
        self.uncertainty_flagger.threshold = unc_t
        self.uncertainty_flagger.metric = metric

        try:
            # 1. Get raw predictions (requires return_probs=True for open-set analysis)
            raw_detections = self.detector.predict(
                source=image,
                conf_threshold=conf_t,
                iou_threshold=self.iou_threshold,
                return_probs=True
            )
            
            # 2. Flag high-uncertainty detections as "unknown"
            final_detections = self.uncertainty_flagger.flag_unknowns(raw_detections)
            
            # 3. Clean up the response (remove raw numpy arrays to be JSON serializable)
            cleaned_detections = []
            for det in final_detections:
                cleaned_det = dict(det)
                if "probs" in cleaned_det:
                    del cleaned_det["probs"]  # pure probability array is massive and not needed in API
                cleaned_detections.append(cleaned_det)
                
            return cleaned_detections
        finally:
            self.uncertainty_flagger.threshold = orig_thresh
            self.uncertainty_flagger.metric = orig_metric

    def process_image_annotated(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: Optional[float] = None,
        uncertainty_threshold: Optional[float] = None,
        uncertainty_metric: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Runs object detection and returns both structured detections and annotated PIL Image.
        """
        from src.utils.visualization import draw_detections

        detections = self.process_image(
            image=image,
            conf_threshold=conf_threshold,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_metric=uncertainty_metric,
        )
        annotated_img = draw_detections(image, detections, class_names=class_names)
        return detections, annotated_img


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

