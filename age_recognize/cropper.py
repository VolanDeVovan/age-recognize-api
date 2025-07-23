from typing import Dict, List, Optional, Tuple
import numpy as np

from .structures import DetectionResult


class PersonAndFaceCrops:
    """
    Container for extracted crops.
    Reference: legacy/MiVOLO/mivolo/structures.py:PersonAndFaceCrops
    """
    
    def __init__(self):
        # int: index of person along results
        self.crops_persons: Dict[int, np.ndarray] = {}
        
        # int: index of face along results
        self.crops_faces: Dict[int, np.ndarray] = {}
        
        # int: index of face along results
        self.crops_faces_wo_body: Dict[int, np.ndarray] = {}
        
        # int: index of person along results
        self.crops_persons_wo_face: Dict[int, np.ndarray] = {}
    
    def _add_to_output(
        self, crops: Dict[int, np.ndarray], out_crops: List[np.ndarray], out_crop_inds: List[Optional[int]]
    ):
        inds_to_add = list(crops.keys())
        crops_to_add = list(crops.values())
        out_crops.extend(crops_to_add)
        out_crop_inds.extend(inds_to_add)

    def _get_all_faces(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                faces: faces_with_bodies + faces_without_bodies + [None] * len(crops_persons_wo_face)
            if use_persons and not use_faces
                faces: [None] * n_persons
            if not use_persons and use_faces:
                faces: faces_with_bodies + faces_without_bodies
        """

        def add_none_to_output(faces_inds, faces_crops, num):
            faces_inds.extend([None for _ in range(num)])
            faces_crops.extend([None for _ in range(num)])

        faces_inds: List[Optional[int]] = []
        faces_crops: List[Optional[np.ndarray]] = []

        if not use_faces:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons) + len(self.crops_persons_wo_face))
            return faces_inds, faces_crops

        self._add_to_output(self.crops_faces, faces_crops, faces_inds)
        self._add_to_output(self.crops_faces_wo_body, faces_crops, faces_inds)

        if use_persons:
            add_none_to_output(faces_inds, faces_crops, len(self.crops_persons_wo_face))

        return faces_inds, faces_crops

    def _get_all_bodies(
        self, use_persons: bool, use_faces: bool
    ) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
        """
        Returns
            if use_persons and use_faces
                persons: bodies_with_faces + [None] * len(faces_without_bodies) + bodies_without_faces
            if use_persons and not use_faces
                persons: bodies_with_faces + bodies_without_faces
            if not use_persons and use_faces
                persons: [None] * n_faces
        """

        def add_none_to_output(bodies_inds, bodies_crops, num):
            bodies_inds.extend([None for _ in range(num)])
            bodies_crops.extend([None for _ in range(num)])

        bodies_inds: List[Optional[int]] = []
        bodies_crops: List[Optional[np.ndarray]] = []

        if not use_persons:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces) + len(self.crops_faces_wo_body))
            return bodies_inds, bodies_crops

        self._add_to_output(self.crops_persons, bodies_crops, bodies_inds)
        if use_faces:
            add_none_to_output(bodies_inds, bodies_crops, len(self.crops_faces_wo_body))

        self._add_to_output(self.crops_persons_wo_face, bodies_crops, bodies_inds)

        return bodies_inds, bodies_crops

    def get_faces_with_bodies(self, use_persons: bool, use_faces: bool):
        """
        Return
            faces: faces_with_bodies, faces_without_bodies, [None] * len(crops_persons_wo_face)
            persons: bodies_with_faces, [None] * len(faces_without_bodies), bodies_without_faces
        """

        bodies_inds, bodies_crops = self._get_all_bodies(use_persons, use_faces)
        faces_inds, faces_crops = self._get_all_faces(use_persons, use_faces)

        return (bodies_inds, bodies_crops), (faces_inds, faces_crops)


class CropExtractor:
    """
    Extract face and person crops from detection results.
    Reference: legacy/MiVOLO/mivolo/structures.py:collect_crops
    """
    
    def __init__(self):
        """Initialize crop extractor."""
        pass
    
    def collect_crops(self, image: np.ndarray, detection_result: DetectionResult) -> PersonAndFaceCrops:
        """
        Extract crops from detection results.
        Reference: legacy/MiVOLO/mivolo/structures.py:collect_crops
        
        Args:
            image: Input image (BGR format)
            detection_result: Detection results with face-person associations
            
        Returns:
            PersonAndFaceCrops containing extracted crops
        """
        crops_data = PersonAndFaceCrops()
        
        # Get face-person mapping from the detection result
        # Assumes detection_result has face_to_person_map and unassigned_persons_inds
        if not hasattr(detection_result, 'face_to_person_map'):
            raise ValueError("DetectionResult must have face_to_person_map. Call associate_faces_with_persons() first.")
        
        face_to_person_map = detection_result.face_to_person_map
        unassigned_persons_inds = detection_result.unassigned_persons_inds
        
        # Extract crops for face-person pairs
        for face_ind, person_ind in face_to_person_map.items():
            face_image = detection_result.crop_object(image, face_ind, cut_other_classes=[])
            
            if person_ind is None:
                crops_data.crops_faces_wo_body[face_ind] = face_image
                continue
            
            person_image = detection_result.crop_object(image, person_ind, cut_other_classes=["face", "person"])
            
            crops_data.crops_faces[face_ind] = face_image
            crops_data.crops_persons[person_ind] = person_image
        
        # Extract crops for unassigned persons
        for person_ind in unassigned_persons_inds:
            person_image = detection_result.crop_object(image, person_ind, cut_other_classes=["face", "person"])
            crops_data.crops_persons_wo_face[person_ind] = person_image
        
        return crops_data
    
    def get_model_inputs(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        use_persons: bool = True,
        use_faces: bool = True
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Extract crops and return aligned lists for model input.
        
        Args:
            image: Input image (BGR format)
            detection_result: Detection results
            use_persons: Whether to use person crops
            use_faces: Whether to use face crops
            
        Returns:
            Tuple of (face_crops, body_crops) aligned for model input
        """
        crops_data = self.collect_crops(image, detection_result)
        (bodies_inds, bodies_crops), (faces_inds, faces_crops) = crops_data.get_faces_with_bodies(use_persons, use_faces)
        return faces_crops, bodies_crops