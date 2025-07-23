# Age Recognition API

A Python package for age and gender recognition from images using deep learning models.

## Installation

```bash
uv sync
```

## Model Setup

Download the required model file and place it in the correct directory:

1. Download from: https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view
2. Place the downloaded file at: `models/age/yolov8x_person_face.pt`

## Usage

### Basic Usage

```python
from PIL import Image
from age_recognize import run_age_recognize

# Load image
img = Image.open("your_image.jpg")

# Get age and gender predictions
results = run_age_recognize(img)

for result in results:
    print(f"Age: {result['age']:.1f}, Gender: {result['gender']}")
```

### Advanced Usage

```python
from age_recognize.recognizer import EnhancedAgeRecognizer

# Initialize with custom settings
recognizer = EnhancedAgeRecognizer(device="cuda", verbose=True)

# Get detailed results with bounding boxes
detailed = recognizer.run_age_recognize_with_details(img)
```

See `example_usage.py` for complete examples.