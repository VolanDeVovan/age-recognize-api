#!/usr/bin/env python3
"""
Example usage script for the age_recognize package using PIL Image.

This script demonstrates how to use the age recognition API with PIL Images.
"""

import os
from PIL import Image

# Import the age_recognize package
from age_recognize import EnhancedAgeRecognizer


def example_basic_usage():
    """Example using the EnhancedAgeRecognizer class."""
    print("=== Basic Usage Example ===")
    
    # Load an image
    image_path = "mock.jpg"
    
    if os.path.exists(image_path):
        # Using PIL Image with default models
        img = Image.open(image_path)
        recognizer = EnhancedAgeRecognizer()
        results = recognizer.run_age_recognize(img)
        
        print(f"Found {len(results)} people:")
        for i, result in enumerate(results):
            print(f"  Person {i+1}: Age={result['age']:.1f}, "
                  f"Gender={result['gender']} ({result['gender_probability']:.2f})")
    else:
        print(f"Image not found: {image_path}")


def example_custom_model_paths():
    """Example using custom model paths."""
    print("\n=== Custom Model Paths Example ===")
    
    image_path = "mock.jpg"
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        
        # Example 1: Custom YOLO model path only
        print("Using custom YOLO model path:")
        recognizer = EnhancedAgeRecognizer(yolo_weights="models/age/yolov8x_person_face.pt")
        results = recognizer.run_age_recognize(img)
        print(f"  Found {len(results)} people with custom YOLO model")
        
        # Example 2: Custom MiVOLO model path only
        print("Using custom MiVOLO model path:")
        recognizer = EnhancedAgeRecognizer(mivolo_model_path="./models/mivolo_v2_local")
        results = recognizer.run_age_recognize(img)
        print(f"  Found {len(results)} people with custom MiVOLO model")
        
        # Example 3: Both custom model paths
        print("Using both custom model paths:")
        recognizer = EnhancedAgeRecognizer(
            yolo_weights="models/age/yolov8x_person_face.pt",
            mivolo_model_path="iitolstykh/mivolo_v2"
        )
        results = recognizer.run_age_recognize(img)
        print(f"  Found {len(results)} people with both custom models")
        
    else:
        print(f"Image not found: {image_path}")


def example_advanced_usage():
    """Example using the EnhancedAgeRecognizer class directly."""
    print("\n=== Advanced Usage Example ===")
    
    # Initialize recognizer with custom settings and model paths
    recognizer = EnhancedAgeRecognizer(
        yolo_weights="models/age/yolov8x_person_face.pt",
        mivolo_model_path="iitolstykh/mivolo_v2",
        device="cuda",  # or "cpu" if no GPU available
        verbose=True    # Enable detailed logging
    )
    
    image_path = "mock.jpg"
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        
        # Get basic results
        results = recognizer.run_age_recognize(img)
        print(f"\nBasic results: {len(results)} people detected")
        
        # Get detailed results with bounding boxes
        detailed = recognizer.run_age_recognize_with_details(img)
        
        print("\nDetailed results:")
        print(f"  Faces detected: {detailed['detection_info']['n_faces']}")
        print(f"  Persons detected: {detailed['detection_info']['n_persons']}")
        print(f"  Face crops extracted: {detailed['crop_info']['n_face_crops']}")
        print(f"  Body crops extracted: {detailed['crop_info']['n_body_crops']}")
        
        # Print bounding boxes
        if detailed['detection_info']['face_boxes']:
            print("  Face bounding boxes (x1, y1, x2, y2):")
            for i, box in enumerate(detailed['detection_info']['face_boxes']):
                print(f"    Face {i+1}: {box}")
        
        if detailed['detection_info']['person_boxes']:
            print("  Person bounding boxes (x1, y1, x2, y2):")
            for i, box in enumerate(detailed['detection_info']['person_boxes']):
                print(f"    Person {i+1}: {box}")
    else:
        print(f"Image not found: {image_path}")


def example_local_model():
    """Example using local MiVOLO model path."""
    print("\n=== Local Model Usage Example ===")
    
    # Check if local model exists
    local_model_path = "./models/mivolo_v2_local"
    
    if not os.path.exists(local_model_path):
        print(f"Local model not found at '{local_model_path}'")
        print("To download the model locally, run:")
        print(f"  python cache.py --model iitolstykh/mivolo_v2 --save-path {local_model_path}")
        print("Skipping local model example...")
        return
    
    # Initialize recognizer with local model path
    from age_recognize.inference import TransformersInferenceEngine
    
    try:
        recognizer = EnhancedAgeRecognizer(
            device="cuda",
            verbose=True
        )
        
        # Override the inference engine with local model
        recognizer.inference_engine = TransformersInferenceEngine(
            model_path=local_model_path,
            device="cuda"
        )
        
        image_path = "mock.jpg"
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            results = recognizer.run_age_recognize(img)
            
            print(f"Found {len(results)} people with local model:")
            for i, result in enumerate(results):
                print(f"  Person {i+1}: Age={result['age']:.1f}, "
                      f"Gender={result['gender']} ({result['gender_probability']:.2f})")
        else:
            print(f"Image not found: {image_path}")
            
    except Exception as e:
        print(f"Error loading local model: {e}")
        print("Make sure the model was downloaded correctly with cache.py")


def example_huggingface_model():
    """Example using Hugging Face model (default)."""
    print("\n=== Hugging Face Model Usage Example ===")
    
    # Initialize recognizer with HuggingFace hub model (default)
    from age_recognize.inference import TransformersInferenceEngine
    
    recognizer = EnhancedAgeRecognizer(
        device="cuda",
        verbose=True
    )
    
    # Override the inference engine with specific HF model
    recognizer.inference_engine = TransformersInferenceEngine(
        model_path="iitolstykh/mivolo_v2",  # HuggingFace hub model
        device="cuda"
    )
    
    image_path = "mock.jpg"
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        results = recognizer.run_age_recognize(img)
        
        print(f"Found {len(results)} people with HuggingFace model:")
        for i, result in enumerate(results):
            print(f"  Person {i+1}: Age={result['age']:.1f}, "
                  f"Gender={result['gender']} ({result['gender_probability']:.2f})")
    else:
        print(f"Image not found: {image_path}")


def main():
    """Run all examples."""
    print("Age Recognition API Usage Examples - PIL Image")
    print("=" * 50)
    
    example_basic_usage()
    example_custom_model_paths()
    example_advanced_usage()
    example_local_model()
    example_huggingface_model()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()