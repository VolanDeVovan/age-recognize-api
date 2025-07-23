# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `uv sync` - Install dependencies using uv package manager
- `uv run <script>` - Run Python scripts with uv

### Running Examples
- `python example_usage.py` or `uv run example_usage.py` - Run complete usage examples
- `python cache.py --example` - Run model caching example
- `python cache.py --model iitolstykh/mivolo_v2 --save-path ./models/mivolo_v2_local` - Download and cache MiVOLO model locally

### Model Setup
The YOLO detection model must be manually downloaded:
1. Download from: https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view
2. Place at: `models/age/yolov8x_person_face.pt`

## Architecture Overview

### Core Components

**Main API (`age_recognize/__init__.py`)**
- Exports `EnhancedAgeRecognizer` class for creating recognizer instances
- Library usage requires creating an instance of `EnhancedAgeRecognizer`

**Enhanced Age Recognizer (`age_recognize/recognizer.py`)**
- `EnhancedAgeRecognizer` - Main orchestrator class combining detection, cropping, and inference
- Provides both basic recognition and detailed results with bounding boxes
- Supports custom YOLO weights and MiVOLO model paths

**Detection Pipeline:**
1. **YOLODetector** (`detector.py`) - Face and person detection using YOLOv8
2. **CropExtractor** (`cropper.py`) - Extract face/body crops with overlap handling  
3. **TransformersInferenceEngine** (`inference.py`) - MiVOLO v2 model inference using HuggingFace transformers

**Data Structures (`structures.py`)**
- `DetectionResult` - Wrapper for YOLO results with face-person association
- `PersonAndFaceCrops` - Container for extracted image crops

### Model Loading Options

**MiVOLO Model:**
- HuggingFace Hub (default): `"iitolstykh/mivolo_v2"` - Downloaded automatically
- Local cached: Use `cache.py` to download to `./models/mivolo_v2_local`

**YOLO Model:**
- Default path: `"models/age/yolov8x_person_face.pt"`
- Must be manually downloaded and placed

### Key Processing Flow

1. **Detection**: YOLO detects faces and persons in image
2. **Association**: Faces are matched to persons using IoU overlap  
3. **Cropping**: Face and body crops extracted with overlap handling
4. **Inference**: MiVOLO model predicts age/gender from crops
5. **Results**: Combined predictions returned with bounding boxes

### Device Support
- CUDA GPU support (recommended for performance)
- CPU fallback available
- PyTorch with CUDA required for GPU usage

### Dependencies
Uses `uv` package manager with PyTorch from custom index:
- PyTorch 2.7.1+cu128 from pytorch.org/whl/cu128
- MiVOLO from custom git fork
- HuggingFace transformers, Ultralytics YOLOv8, PIL, NumPy, OpenCV