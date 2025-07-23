import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from .utils import box_iou, assign_faces


class DetectionResult:
    """
    Wrapper for YOLO detection results with helper methods.
    Reference: legacy/MiVOLO/mivolo/structures.py:PersonAndFaceResult
    """
    
    def __init__(self, results: Results):
        """
        Initialize detection result wrapper.
        
        Args:
            results: YOLO detection results
        """
        self.yolo_results = results
        
        # Verify that model detected face and person classes
        names = set(results.names.values())
        if "person" not in names or "face" not in names:
            raise ValueError(f"Model must detect 'person' and 'face' classes. Found: {names}")
        
        # Initialize age/gender storage for compatibility
        n_objects = len(self.yolo_results.boxes) if self.yolo_results.boxes is not None else 0
        self.ages: List[Optional[float]] = [None for _ in range(n_objects)]
        self.genders: List[Optional[str]] = [None for _ in range(n_objects)]
        self.gender_scores: List[Optional[float]] = [None for _ in range(n_objects)]
        
        # Initialize face-person associations (set when associate_faces_with_persons is called)
        self.face_to_person_map: Dict[int, Optional[int]] = {ind: None for ind in self.get_class_indices("face")}
        self.unassigned_persons_inds: List[int] = self.get_class_indices("person")
    
    @property
    def n_objects(self) -> int:
        """Total number of detected objects."""
        return len(self.yolo_results.boxes) if self.yolo_results.boxes is not None else 0
    
    @property
    def n_faces(self) -> int:
        """Number of detected faces."""
        return len(self.get_class_indices("face"))
    
    @property
    def n_persons(self) -> int:
        """Number of detected persons."""
        return len(self.get_class_indices("person"))
    
    def get_class_indices(self, class_name: str) -> List[int]:
        """
        Get indices of detections for a specific class.
        
        Args:
            class_name: Name of the class ("face" or "person")
            
        Returns:
            List of detection indices for the specified class
        """
        if self.yolo_results.boxes is None:
            return []
        
        indices = []
        for idx, detection in enumerate(self.yolo_results.boxes):
            detected_class = self.yolo_results.names[int(detection.cls)]
            if detected_class == class_name:
                indices.append(idx)
        
        return indices
    
    def get_face_boxes(self) -> List[torch.Tensor]:
        """Get bounding boxes for all face detections."""
        face_indices = self.get_class_indices("face")
        return [self.get_bbox_by_index(idx) for idx in face_indices]
    
    def get_person_boxes(self) -> List[torch.Tensor]:
        """Get bounding boxes for all person detections."""
        person_indices = self.get_class_indices("person")
        return [self.get_bbox_by_index(idx) for idx in person_indices]
    
    def get_bbox_by_index(self, index: int, im_h: Optional[int] = None, im_w: Optional[int] = None) -> torch.Tensor:
        """
        Get bounding box coordinates by detection index.
        Reference: PersonAndFaceResult.get_bbox_by_ind
        
        Args:
            index: Detection index
            im_h: Image height for clamping (optional)
            im_w: Image width for clamping (optional)
            
        Returns:
            Bounding box tensor [x1, y1, x2, y2]
        """
        bbox = self.yolo_results.boxes[index].xyxy.squeeze().type(torch.int32)
        
        if im_h is not None and im_w is not None:
            bbox[0] = torch.clamp(bbox[0], min=0, max=im_w - 1)
            bbox[1] = torch.clamp(bbox[1], min=0, max=im_h - 1)
            bbox[2] = torch.clamp(bbox[2], min=0, max=im_w - 1)
            bbox[3] = torch.clamp(bbox[3], min=0, max=im_h - 1)
        
        return bbox
    
    def crop_object(
        self, 
        full_image: np.ndarray, 
        index: int, 
        cut_other_classes: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract crop for a specific detection with overlap handling.
        Reference: PersonAndFaceResult.crop_object
        
        Args:
            full_image: Full input image (BGR format)
            index: Detection index to crop
            cut_other_classes: List of class names to cut out from crop
            
        Returns:
            Cropped image or None if crop is too small/invalid
        """
        IOU_THRESH = 0.000001
        MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
        CROP_ROUND_RATE = 0.3
        MIN_PERSON_SIZE = 50
        
        # Get object bounding box
        obj_bbox = self.get_bbox_by_index(index, *full_image.shape[:2])
        x1, y1, x2, y2 = obj_bbox
        current_class = self.yolo_results.names[int(self.yolo_results.boxes[index].cls)]
        
        # Extract crop
        obj_image = full_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]
        
        # Check minimum size for person crops
        if current_class == "person" and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
            return None
        
        if not cut_other_classes:
            return obj_image
        
        # Calculate IoU with all other detections to handle overlaps
        if self.yolo_results.boxes is not None and len(self.yolo_results.boxes) > 1:
            all_bboxes = [self.get_bbox_by_index(i, *full_image.shape[:2]) for i in range(len(self.yolo_results.boxes))]
            iou_matrix = box_iou(torch.stack([obj_bbox]), torch.stack(all_bboxes)).cpu().numpy()[0]
            
            # Cut out overlapping objects of specified classes
            for other_idx, (detection, iou) in enumerate(zip(self.yolo_results.boxes, iou_matrix)):
                other_class = self.yolo_results.names[int(detection.cls)]
                
                if (index == other_idx or 
                    iou < IOU_THRESH or 
                    other_class not in cut_other_classes):
                    continue
                
                # Get other object's bbox in crop coordinates
                o_x1, o_y1, o_x2, o_y2 = detection.xyxy.squeeze().type(torch.int32)
                
                # Remap to crop coordinates
                o_x1 = max(o_x1 - x1, 0)
                o_y1 = max(o_y1 - y1, 0)
                o_x2 = min(o_x2 - x1, crop_w)
                o_y2 = min(o_y2 - y1, crop_h)
                
                # Apply rounding for non-face objects
                if other_class != "face":
                    if (o_y1 / crop_h) < CROP_ROUND_RATE:
                        o_y1 = 0
                    if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
                        o_y2 = crop_h
                    if (o_x1 / crop_w) < CROP_ROUND_RATE:
                        o_x1 = 0
                    if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
                        o_x2 = crop_w
                
                # Zero out the overlapping region
                obj_image[o_y1:o_y2, o_x1:o_x2] = 0
        
        # Check if enough of the crop remains after cutting
        remain_ratio = np.count_nonzero(obj_image) / (obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2])
        if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
            return None
        
        return obj_image
    
    def get_distance_to_center(self, index: int) -> float:
        """
        Calculate euclidean distance between detection center and image center.
        Reference: PersonAndFaceResult.get_distance_to_center
        
        Args:
            index: Detection index
            
        Returns:
            Distance to image center
        """
        im_h, im_w = self.yolo_results.orig_shape
        x1, y1, x2, y2 = self.get_bbox_by_index(index).cpu().numpy()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = math.dist([center_x, center_y], [im_w / 2, im_h / 2])
        return dist
    
    def set_age(self, index: Optional[int], age: float):
        """Set age prediction for a detection."""
        if index is not None and 0 <= index < len(self.ages):
            self.ages[index] = age
    
    def set_gender(self, index: Optional[int], gender: str, gender_score: float):
        """Set gender prediction for a detection."""
        if index is not None and 0 <= index < len(self.genders):
            self.genders[index] = gender
            self.gender_scores[index] = gender_score
    
    def associate_faces_with_persons(self):
        """
        Associate faces with persons based on IoU overlap.
        Reference: legacy/MiVOLO/mivolo/structures.py:associate_faces_with_persons
        """
        face_bboxes_inds = self.get_class_indices("face")
        person_bboxes_inds = self.get_class_indices("person")
        
        face_bboxes = [self.get_bbox_by_index(ind) for ind in face_bboxes_inds]
        person_bboxes = [self.get_bbox_by_index(ind) for ind in person_bboxes_inds]
        
        self.face_to_person_map = {ind: None for ind in face_bboxes_inds}
        assigned_faces, unassigned_persons_inds = assign_faces(person_bboxes, face_bboxes)
        
        for face_ind, person_ind in enumerate(assigned_faces):
            face_ind = face_bboxes_inds[face_ind]
            person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None
            self.face_to_person_map[face_ind] = person_ind
        
        self.unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds]