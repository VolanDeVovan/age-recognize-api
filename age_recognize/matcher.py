from typing import Dict, List, Optional, Tuple
import torch

from .structures import DetectionResult
from .utils import assign_faces


class FacePersonMatcher:
    """
    Face-person association logic using IoU-based matching.
    Reference: legacy/MiVOLO/mivolo/structures.py:associate_faces_with_persons
    """
    
    def __init__(self):
        """Initialize the face-person matcher."""
        pass
    
    def associate_faces_with_persons(self, detection_result: DetectionResult) -> Tuple[Dict[int, Optional[int]], List[int]]:
        """
        Associate faces with persons based on bounding box overlap.
        
        Args:
            detection_result: Detection results containing faces and persons
            
        Returns:
            Tuple of (face_to_person_map, unassigned_person_indices)
            face_to_person_map: Dict mapping face_index -> person_index (or None)
            unassigned_person_indices: List of person indices without assigned faces
        """
        face_indices = detection_result.get_class_indices("face")
        person_indices = detection_result.get_class_indices("person")
        
        if not face_indices and not person_indices:
            return {}, []
        
        if not face_indices:
            # No faces, all persons are unassigned
            return {}, person_indices
        
        if not person_indices:
            # No persons, all faces are unassigned
            return {face_idx: None for face_idx in face_indices}, []
        
        # Get bounding boxes
        face_boxes = [detection_result.get_bbox_by_index(idx) for idx in face_indices]
        person_boxes = [detection_result.get_bbox_by_index(idx) for idx in person_indices]
        
        # Use assignment algorithm from utils
        assigned_faces, unassigned_person_rel_indices = assign_faces(person_boxes, face_boxes)
        
        # Build face-to-person mapping with actual detection indices
        face_to_person_map = {}
        for face_rel_idx, person_rel_idx in enumerate(assigned_faces):
            face_idx = face_indices[face_rel_idx]
            person_idx = person_indices[person_rel_idx] if person_rel_idx is not None else None
            face_to_person_map[face_idx] = person_idx
        
        # Convert relative person indices to actual detection indices
        unassigned_person_indices = [person_indices[rel_idx] for rel_idx in unassigned_person_rel_indices]
        
        return face_to_person_map, unassigned_person_indices
    
    def get_face_person_pairs(self, detection_result: DetectionResult) -> Tuple[List[Tuple[int, Optional[int]]], List[int]]:
        """
        Get face-person pairs and unassigned persons for crop extraction.
        
        Args:
            detection_result: Detection results
            
        Returns:
            Tuple of (face_person_pairs, unassigned_person_indices)
            face_person_pairs: List of (face_index, person_index) tuples
            unassigned_person_indices: List of person indices without faces
        """
        face_to_person_map, unassigned_person_indices = self.associate_faces_with_persons(detection_result)
        
        # Convert mapping to pairs list
        face_person_pairs = [(face_idx, person_idx) for face_idx, person_idx in face_to_person_map.items()]
        
        return face_person_pairs, unassigned_person_indices