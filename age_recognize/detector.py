import os
from typing import Union
import numpy as np
import PIL
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .structures import DetectionResult


class YOLODetector:
    """
    YOLO detector for face and person detection.
    Reference: legacy/MiVOLO/mivolo/model/yolo_detector.py
    """
    
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ):
        """
        Initialize YOLO detector.
        
        Args:
            weights: Path to YOLO weights file
            device: Device to run inference on
            half: Use half precision
            verbose: Verbose output
            conf_thresh: Confidence threshold
            iou_thresh: IoU threshold for NMS
        """
        # Fix PyTorch 2.6+ weights_only loading issue for ultralytics models
        # Monkey patch torch.load to use weights_only=False for YOLO loading
        import torch
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            self.yolo = YOLO(weights)
            self.yolo.fuse()
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"
        
        if self.half:
            self.yolo.model = self.yolo.model.half()
        
        self.detector_names = self.yolo.model.names
        
        # Detector parameters
        self.detector_kwargs = {
            "conf": conf_thresh,
            "iou": iou_thresh,
            "half": self.half,
            "verbose": verbose
        }
        
        # Verify that model detects face and person classes
        names = set(self.detector_names.values())
        if "person" not in names or "face" not in names:
            raise ValueError(f"Model must detect 'person' and 'face' classes. Found: {names}")
    
    def predict(self, image: Union[np.ndarray, str, PIL.Image.Image]) -> DetectionResult:
        """
        Predict faces and persons in image.
        
        Args:
            image: Input image (numpy array, file path, or PIL Image)
            
        Returns:
            DetectionResult wrapper containing YOLO results
        """
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return DetectionResult(results)
    
    def track(self, image: Union[np.ndarray, str, PIL.Image.Image]) -> DetectionResult:
        """
        Track faces and persons in image (for video processing).
        
        Args:
            image: Input image
            
        Returns:
            DetectionResult wrapper containing YOLO tracking results
        """
        results: Results = self.yolo.track(image, persist=True, **self.detector_kwargs)[0]
        return DetectionResult(results)