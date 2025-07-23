import os
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import cv2

from .detector import YOLODetector
from .cropper import CropExtractor
from .inference import TransformersInferenceEngine


class EnhancedAgeRecognizer:
    """
    Main orchestrator class that combines detection, cropping, and inference.
    Provides the same API as the legacy implementation.
    """
    
    def __init__(
        self, 
        yolo_weights: str = "models/age/yolov8x_person_face.pt",
        mivolo_model_path: str = "iitolstykh/mivolo_v2",
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize the enhanced age recognizer.
        
        Args:
            yolo_weights: Path to YOLO model weights
            mivolo_model_path: Path to MiVOLO model (local path or HuggingFace hub)
            device: Device to run inference on
            verbose: Enable verbose logging
        """
        self.device = device
        self.verbose = verbose
        
        # Initialize components
        self.detector = YOLODetector(
            weights=yolo_weights,
            device=device,
            verbose=verbose
        )
        self.cropper = CropExtractor()
        self.inference_engine = TransformersInferenceEngine(
            model_path=mivolo_model_path, 
            device=device
        )
        
        if verbose:
            print(f"Enhanced Age Recognizer initialized with device: {device}")
    
    def run_age_recognize(self, img: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Main API function - same interface as legacy implementation.
        
        Args:
            img: PIL Image or numpy array (RGB format for PIL, BGR for numpy)
            
        Returns:
            List of dictionaries with age/gender predictions:
            [{"age": 25.4, "gender": "female", "gender_probability": 0.85}, ...]
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            # Convert RGB to BGR for processing
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Assume numpy array is already in BGR format (OpenCV standard)
            # If you have an RGB numpy array, convert it first with cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_array = img
        
        if self.verbose:
            print(f"Processing image of shape: {img_array.shape}")
        
        # Step 1: Detect faces and persons
        detection_result = self.detector.predict(img_array)
        
        if self.verbose:
            print(f"Detected {detection_result.n_faces} faces and {detection_result.n_persons} persons")
        
        # If no faces or persons detected, return empty list
        if detection_result.n_faces == 0 and detection_result.n_persons == 0:
            if self.verbose:
                print("No faces or persons detected")
            return []
        
        # Associate faces with persons first
        detection_result.associate_faces_with_persons()
        
        # Step 2: Extract crops
        face_crops, body_crops = self.cropper.get_model_inputs(
            img_array, 
            detection_result,
            use_persons=True,
            use_faces=True
        )
        
        if self.verbose:
            print(f"Extracted {len(face_crops)} face crops and {len(body_crops)} body crops")
        
        # If no valid crops extracted, return empty list
        if not face_crops and not body_crops:
            if self.verbose:
                print("No valid crops extracted")
            return []
        
        # Step 3: Run inference
        try:
            results = self.inference_engine.predict_crops(face_crops, body_crops)
            
            if self.verbose:
                print(f"Generated {len(results)} predictions")
                for i, result in enumerate(results):
                    print(f"  Person {i+1}: age={result['age']}, gender={result['gender']} ({result['gender_probability']:.2f})")
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Inference failed: {e}")
            return []
    
    def run_age_recognize_with_details(self, img: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Extended API that returns detailed information including bounding boxes.
        
        Args:
            img: PIL Image or numpy array
            
        Returns:
            Dictionary with detailed results including bounding boxes and crops
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = img
        
        # Step 1: Detect faces and persons
        detection_result = self.detector.predict(img_array)
        
        # Associate faces with persons first
        detection_result.associate_faces_with_persons()
        
        # Step 2: Extract crops and get associations
        face_crops, body_crops = self.cropper.get_model_inputs(
            img_array, 
            detection_result,
            use_persons=True,
            use_faces=True
        )
        
        # Step 3: Run inference
        predictions = []
        if face_crops or body_crops:
            try:
                predictions = self.inference_engine.predict_crops(face_crops, body_crops)
            except Exception as e:
                if self.verbose:
                    print(f"Inference failed: {e}")
        
        # Step 4: Compile detailed results
        detailed_results = {
            "predictions": predictions,
            "detection_info": {
                "n_faces": detection_result.n_faces,
                "n_persons": detection_result.n_persons,
                "face_boxes": [box.cpu().numpy().tolist() for box in detection_result.get_face_boxes()],
                "person_boxes": [box.cpu().numpy().tolist() for box in detection_result.get_person_boxes()],
            },
            "crop_info": {
                "n_face_crops": len([c for c in face_crops if c is not None]),
                "n_body_crops": len([c for c in body_crops if c is not None]),
                "total_pairs": len(face_crops)
            }
        }
        
        return detailed_results


# Global instance for backward compatibility
_global_recognizer = None


def get_global_recognizer(**kwargs) -> EnhancedAgeRecognizer:
    """Get or create global recognizer instance."""
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = EnhancedAgeRecognizer(**kwargs)
    return _global_recognizer