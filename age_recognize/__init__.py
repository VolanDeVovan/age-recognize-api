from .recognizer import EnhancedAgeRecognizer

def run_age_recognize(img):
    """
    Main API function - same interface as legacy implementation.
    
    Args:
        img: PIL Image or numpy array
        
    Returns:
        List of dictionaries with age/gender predictions:
        [{"age": 25.4, "gender": "female", "gender_probability": 0.85}, ...]
    """
    recognizer = EnhancedAgeRecognizer()
    return recognizer.run_age_recognize(img)