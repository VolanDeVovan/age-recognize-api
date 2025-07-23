from .recognizer import EnhancedAgeRecognizer

def run_age_recognize(img, yolo_weights=None, mivolo_model_path=None):
    """
    Main API function - same interface as legacy implementation.
    
    Args:
        img: PIL Image or numpy array
        yolo_weights: Optional path to YOLO model weights
        mivolo_model_path: Optional path to MiVOLO model (local path or HuggingFace hub)
        
    Returns:
        List of dictionaries with age/gender predictions:
        [{"age": 25.4, "gender": "female", "gender_probability": 0.85}, ...]
    """
    kwargs = {}
    if yolo_weights is not None:
        kwargs['yolo_weights'] = yolo_weights
    if mivolo_model_path is not None:
        kwargs['mivolo_model_path'] = mivolo_model_path
    
    recognizer = EnhancedAgeRecognizer(**kwargs)
    return recognizer.run_age_recognize(img)