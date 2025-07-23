#!/usr/bin/env python3
"""
Example usage script for the age_recognize package using PIL Image.

This script demonstrates how to use the age recognition API with PIL Images.
"""

import os
from PIL import Image

# Import the age_recognize package
from age_recognize import run_age_recognize
from age_recognize.recognizer import EnhancedAgeRecognizer


def example_basic_usage():
    """Example using the simple API function."""
    print("=== Basic Usage Example ===")
    
    # Load an image
    image_path = "mock.jpg"
    
    if os.path.exists(image_path):
        # Using PIL Image
        img = Image.open(image_path)
        results = run_age_recognize(img)
        
        print(f"Found {len(results)} people:")
        for i, result in enumerate(results):
            print(f"  Person {i+1}: Age={result['age']:.1f}, "
                  f"Gender={result['gender']} ({result['gender_probability']:.2f})")
    else:
        print(f"Image not found: {image_path}")


def example_advanced_usage():
    """Example using the EnhancedAgeRecognizer class directly."""
    print("\n=== Advanced Usage Example ===")
    
    # Initialize recognizer with custom settings
    recognizer = EnhancedAgeRecognizer(
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


def main():
    """Run all examples."""
    print("Age Recognition API Usage Examples - PIL Image")
    print("=" * 50)
    
    example_basic_usage()
    example_advanced_usage()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()