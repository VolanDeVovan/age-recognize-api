import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def box_iou(box1: torch.Tensor, box2: torch.Tensor, over_second: bool = False) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of bounding boxes.
    Reference: legacy/MiVOLO/mivolo/data/misc.py:box_iou
    
    Args:
        box1: Tensor[N, 4] in (x1, y1, x2, y2) format
        box2: Tensor[M, 4] in (x1, y1, x2, y2) format
        over_second: If True, return mean(intersection-over-union, (inter / area2))
    
    Returns:
        iou: Tensor[N, M] containing pairwise IoU values
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # Calculate intersection
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    iou = inter / (area1[:, None] + area2 - inter)
    
    if over_second:
        return (inter / area2 + iou) / 2
    else:
        return iou


def assign_faces(person_bboxes: List[torch.Tensor], face_bboxes: List[torch.Tensor], iou_thresh: float = 0.0001) -> Tuple[List[Optional[int]], List[int]]:
    """
    Assign faces to persons based on IoU overlap.
    Reference: legacy/MiVOLO/mivolo/data/misc.py:assign_faces
    
    Args:
        person_bboxes: List of person bounding boxes
        face_bboxes: List of face bounding boxes
        iou_thresh: IoU threshold for assignment
    
    Returns:
        Tuple of (assigned_faces, unassigned_persons_inds)
        assigned_faces: List where index is face_idx, value is person_idx or None
        unassigned_persons_inds: List of person indices without assigned faces
    """
    assigned_faces = [None] * len(face_bboxes)
    unassigned_persons_inds = list(range(len(person_bboxes)))
    
    if not person_bboxes or not face_bboxes:
        return assigned_faces, unassigned_persons_inds
    
    # Calculate cost matrix - persons x faces (matching legacy parameter order)
    person_tensor = torch.stack(person_bboxes)
    face_tensor = torch.stack(face_bboxes)
    cost_matrix = box_iou(person_tensor, face_tensor, over_second=True).cpu().numpy()
    
    persons_indexes, face_indexes = [], []
    if len(cost_matrix) > 0:
        persons_indexes, face_indexes = linear_sum_assignment(cost_matrix, maximize=True)
    
    matched_persons = set()
    for person_idx, face_idx in zip(persons_indexes, face_indexes):
        ciou = cost_matrix[person_idx][face_idx]
        if ciou > iou_thresh:
            if person_idx in matched_persons:
                # Person can not be assigned twice, in reality this should not happen
                continue
            assigned_faces[face_idx] = person_idx
            matched_persons.add(person_idx)
    
    unassigned_persons_inds = [p_ind for p_ind in range(len(person_bboxes)) if p_ind not in matched_persons]
    
    return assigned_faces, unassigned_persons_inds


def prepare_classification_images(
    img_list: List[Optional[np.ndarray]],
    target_size: int = 224,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    device=None,
) -> torch.Tensor:
    """
    Prepare images for classification model input.
    Reference: legacy/MiVOLO/mivolo/data/misc.py:prepare_classification_images
    
    Args:
        img_list: List of BGR numpy arrays or None
        target_size: Target image size
        mean: Normalization mean
        std: Normalization std
        device: Target device
    
    Returns:
        Batch tensor ready for model input
    """
    prepared_images = []
    
    for img in img_list:
        if img is None:
            # Create zero tensor for missing crops
            img_tensor = torch.zeros((3, target_size, target_size), dtype=torch.float32)
            img_tensor = F.normalize(img_tensor, mean=mean, std=std)
            img_tensor = img_tensor.unsqueeze(0)
            prepared_images.append(img_tensor)
            continue
        
        # Resize with letterbox padding
        img = class_letterbox(img, new_shape=(target_size, target_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)
        
        # Convert to tensor format
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        prepared_images.append(img_tensor)
    
    return torch.cat(prepared_images, dim=0) if prepared_images else None


def class_letterbox(img: np.ndarray, new_shape: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        img: Input BGR image
        new_shape: Target (height, width)
        color: Padding color
    
    Returns:
        Resized and padded image
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img