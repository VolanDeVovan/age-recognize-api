# Age Recognition API

A Python package for age and gender recognition from images using the MiVOLO v2 model with YOLO person/face detection.

This repository is inspired by and uses models from [WildChlamydia/MiVOLO](https://github.com/WildChlamydia/MiVOLO).

## Features

- **Age and gender prediction** from face and body images
- **Person and face detection** using YOLOv8
- **Multiple model loading options**: HuggingFace Hub or local cached models  
- **Batch processing** support
- **Detailed results** with bounding boxes and confidence scores
- **GPU acceleration** support

## Installation

```bash
uv sync
```

## Model Setup

### 1. YOLO Model
Download the YOLO detection model and place it in the correct directory:

1. Download from: https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view
2. Place the downloaded file at: `models/age/yolov8x_person_face.pt`

### 2. MiVOLO Model

### Option 1: Automatic Download (Recommended)
The MiVOLO v2 model will be automatically downloaded from HuggingFace Hub on first use.

### Option 2: Cache MiVOLO Model Locally
For faster loading or offline usage, cache the model locally:

```bash
# Download and cache the model
python cache.py --model iitolstykh/mivolo_v2 --save-path ./models/mivolo_v2_local

# Or run the full example
python cache.py --example
```

## Usage

### Basic Usage

```python
from PIL import Image
from age_recognize import EnhancedAgeRecognizer

# Load image
img = Image.open("your_image.jpg")

# Create recognizer instance and get age and gender predictions
recognizer = EnhancedAgeRecognizer()
results = recognizer.run_age_recognize(img)

for result in results:
    print(f"Age: {result['age']:.1f}, Gender: {result['gender']} ({result['gender_probability']:.2f})")
```

**Output format:**
```python
[
    {
        "age": 25.4,
        "gender": "female", 
        "gender_probability": 0.95
    },
    # ... more people
]
```

### Advanced Usage

```python
from age_recognize.recognizer import EnhancedAgeRecognizer

# Initialize with custom settings
recognizer = EnhancedAgeRecognizer(
    device="cuda",  # or "cpu"
    verbose=True
)

# Get detailed results with bounding boxes
detailed = recognizer.run_age_recognize_with_details(img)

print(f"Detected {detailed['detection_info']['n_faces']} faces")
print(f"Detected {detailed['detection_info']['n_persons']} persons")

# Access bounding boxes
for i, box in enumerate(detailed['detection_info']['face_boxes']):
    print(f"Face {i+1} bounding box: {box}")  # [x1, y1, x2, y2]
```

### Using Local Cached Models

```python
from age_recognize.inference import TransformersInferenceEngine
from age_recognize.recognizer import EnhancedAgeRecognizer

# Initialize recognizer
recognizer = EnhancedAgeRecognizer(device="cuda", verbose=True)

# Use locally cached model (faster loading)
recognizer.inference_engine = TransformersInferenceEngine(
    model_path="./models/mivolo_v2_local",  # Local path
    device="cuda"
)

# Use HuggingFace Hub model (automatic download)
recognizer.inference_engine = TransformersInferenceEngine(
    model_path="iitolstykh/mivolo_v2",  # HF hub identifier
    device="cuda"
)
```

### Batch Processing

```python
# Process multiple image crops efficiently
results = recognizer.inference_engine.predict_batch(
    face_crops=face_crop_list,
    body_crops=body_crop_list,
    batch_size=8
)
```

## Model Caching Utilities

The `cache.py` script provides utilities for downloading and managing models:

```bash
# Download and cache model
python cache.py --model iitolstykh/mivolo_v2 --save-path ./models/mivolo_v2_local

# Verify cached model
python cache.py --verify ./models/mivolo_v2_local

# Run example with caching
python cache.py --example
```

## Examples

Run the complete examples:

```bash
# Run all usage examples
python example_usage.py

# Test with uv
uv run example_usage.py
```

The examples demonstrate:
- Basic age/gender recognition
- Advanced usage with detailed results
- Local model loading
- HuggingFace model usage

## API Reference

### `EnhancedAgeRecognizer`
Main class for age recognition with full control.

**Constructor parameters:**
- `yolo_weights`: Path to YOLO model (default: "models/age/yolov8x_person_face.pt")
- `device`: Device for inference ("cuda" or "cpu")
- `verbose`: Enable detailed logging

**Methods:**
- `run_age_recognize(img)`: Basic recognition
- `run_age_recognize_with_details(img)`: Recognition with bounding boxes

### `TransformersInferenceEngine`
Handles MiVOLO model inference.

**Constructor parameters:**
- `model_path`: HuggingFace model ID or local path (default: "iitolstykh/mivolo_v2")
- `device`: Device for inference

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for best performance
- **CPU**: Works on CPU but significantly slower
- **Memory**: At least 4GB RAM, 8GB+ recommended for GPU usage

## Dependencies

- PyTorch with CUDA support
- Transformers (HuggingFace)
- Ultralytics (YOLOv8)
- PIL/Pillow
- NumPy
- OpenCV

See `pyproject.toml` for complete dependency list.