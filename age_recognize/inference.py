import os
import sys
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

# Add the new directory to path to access mivolo_v2 components
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "new"))


class TransformersInferenceEngine:
    """
    Inference engine using the HuggingFace transformers MiVOLO model.
    Reference: new/mivolo_v2/README.md and new/age_recognize.py
    """
    
    def __init__(self, model_path: str = "iitolstykh/mivolo_v2", device: str = "cuda"):
        """
        Initialize the transformers inference engine.
        
        Args:
            model_path: Path to the MiVOLO v2 model directory
            device: Device to run inference on
        """
        self.device = device
        self.model = None
        self.config = None
        self.image_processor = None
        self.model_path = model_path
        
        # Lazy loading - load model when first needed
        self._load_model()
    
    def _load_model(self):
        """Load the model components if not already loaded."""
        if self.model is None:
            # Use model_path directly (supports both local paths and HuggingFace hub)
            model_path = self.model_path
            
            # If it's a relative path, make it absolute
            if not model_path.startswith('/') and '/' not in model_path.split('/')[0]:
                # This is likely a HuggingFace hub model, use as-is
                pass
            else:
                # This is a local path, make it absolute
                current_dir = os.path.dirname(__file__)
                model_path = os.path.join(current_dir, "..", model_path)
                model_path = os.path.abspath(model_path)
            
            # Load model components
            self.config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
    
    def predict_crops(
        self, 
        face_crops: List[Optional[np.ndarray]], 
        body_crops: List[Optional[np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Predict age and gender from face/body crop pairs.
        Reference: new/mivolo_v2/README.md example
        
        Args:
            face_crops: List of face crops (BGR format) or None
            body_crops: List of body crops (BGR format) or None
            
        Returns:
            List of prediction dictionaries with keys: age, gender, gender_probability
        """
        if not face_crops and not body_crops:
            return []
        
        if len(face_crops) != len(body_crops):
            raise ValueError(f"Face crops ({len(face_crops)}) and body crops ({len(body_crops)}) must have same length")
        
        # Ensure model is loaded
        self._load_model()
        
        results = []
        
        # Process crops in batches (for now, process one by one)
        for i in range(len(face_crops)):
            face_crop = face_crops[i]
            body_crop = body_crops[i]
            
            try:
                result = self._predict_single_pair(face_crop, body_crop)
                results.append(result)
            except Exception as e:
                # If individual prediction fails, return default values
                results.append({
                    "age": 25.0,
                    "gender": "unknown",
                    "gender_probability": 0.5
                })
        
        return results
    
    def _predict_single_pair(
        self, 
        face_crop: Optional[np.ndarray], 
        body_crop: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Predict age/gender for a single face-body crop pair.
        
        Args:
            face_crop: Face crop (BGR format) or None
            body_crop: Body crop (BGR format) or None
            
        Returns:
            Dictionary with age, gender, and gender_probability
        """
        # Prepare input crops for processing
        face_list = [face_crop] if face_crop is not None else [None]
        body_list = [body_crop] if body_crop is not None else [None]
        
        # Process crops through image processor
        faces_input = self.image_processor(images=face_list)["pixel_values"]
        body_input = self.image_processor(images=body_list)["pixel_values"]
        
        # Move to device and correct dtype
        faces_input = faces_input.to(dtype=self.model.dtype, device=self.model.device)
        body_input = body_input.to(dtype=self.model.dtype, device=self.model.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(faces_input=faces_input, body_input=body_input)
        
        # Parse results
        age = output.age_output[0].item()
        
        # Get gender prediction
        id2label = self.config.gender_id2label
        gender = id2label[output.gender_class_idx[0].item()]
        gender_prob = output.gender_probs[0].item()
        
        return {
            "age": round(age, 2),
            "gender": gender,
            "gender_probability": round(gender_prob, 2)
        }
    
    def predict_batch(
        self, 
        face_crops: List[Optional[np.ndarray]], 
        body_crops: List[Optional[np.ndarray]],
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Predict age and gender for multiple crop pairs in batches.
        
        Args:
            face_crops: List of face crops (BGR format) or None
            body_crops: List of body crops (BGR format) or None
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if not face_crops and not body_crops:
            return []
        
        results = []
        num_samples = len(face_crops)
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_face_crops = face_crops[i:end_idx]
            batch_body_crops = body_crops[i:end_idx]
            
            batch_results = self._predict_batch_internal(batch_face_crops, batch_body_crops)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(
        self, 
        face_crops: List[Optional[np.ndarray]], 
        body_crops: List[Optional[np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Internal batch prediction method.
        
        Args:
            face_crops: Batch of face crops
            body_crops: Batch of body crops
            
        Returns:
            List of prediction results for the batch
        """
        # Ensure model is loaded
        self._load_model()
        
        # Process crops
        faces_input = self.image_processor(images=face_crops)["pixel_values"]
        body_input = self.image_processor(images=body_crops)["pixel_values"]
        
        # Move to device
        faces_input = faces_input.to(dtype=self.model.dtype, device=self.model.device)
        body_input = body_input.to(dtype=self.model.dtype, device=self.model.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(faces_input=faces_input, body_input=body_input)
        
        # Parse results for all samples in batch
        results = []
        id2label = self.config.gender_id2label
        
        for i in range(len(face_crops)):
            age = output.age_output[i].item()
            gender = id2label[output.gender_class_idx[i].item()]
            gender_prob = output.gender_probs[i].item()
            
            results.append({
                "age": round(age, 2),
                "gender": gender,
                "gender_probability": round(gender_prob, 2)
            })
        
        return results