#!/usr/bin/env python3
"""
Model caching utility for downloading and saving MiVOLO models from Hugging Face.

This script allows you to download MiVOLO models from Hugging Face and save them
locally for offline usage or faster loading.
"""

from pathlib import Path
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor


def download_and_save_model(
    model_name: str = "iitolstykh/mivolo_v2",
    save_path: str = "./models/mivolo_v2_local"
) -> bool:
    """
    Download MiVOLO model from Hugging Face and save it locally.
    
    Args:
        model_name: Hugging Face model identifier
        save_path: Local path to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading model '{model_name}' from Hugging Face...")
        
        # Create save directory if it doesn't exist
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Download model components
        print("Loading model configuration...")
        config = AutoConfig.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print("Loading model...")
        model = AutoModelForImageClassification.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        
        print("Loading image processor...")
        image_processor = AutoImageProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Save model components locally
        print(f"Saving model to '{save_path}'...")
        
        # Save config
        config.save_pretrained(save_path)
        
        # Save model
        model.save_pretrained(save_path)
        
        # Save image processor
        image_processor.save_pretrained(save_path)
        
        print(f"Model successfully saved to '{save_path.absolute()}'")
        print("\nYou can now use this model locally by setting:")
        print(f"model_path = '{save_path.absolute()}'")
        
        return True
        
    except Exception as e:
        print(f"Error downloading/saving model: {e}")
        return False


def verify_local_model(model_path: str) -> bool:
    """
    Verify that a local model directory contains all required files.
    
    Args:
        model_path: Path to the local model directory
        
    Returns:
        True if model is valid, False otherwise
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"Model path '{model_path}' does not exist")
        return False
    
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin"  # or model.safetensors
    ]
    
    missing_files = []
    for file in required_files:
        file_path = model_path / file
        safetensors_path = model_path / "model.safetensors"
        
        if file == "pytorch_model.bin" and not file_path.exists():
            if not safetensors_path.exists():
                missing_files.append(f"{file} or model.safetensors")
        elif not file_path.exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Model at '{model_path}' is missing files: {missing_files}")
        return False
    
    print(f"Model at '{model_path}' appears to be valid")
    return True


def example_usage():
    """Example usage of the caching functionality."""
    print("=== MiVOLO Model Caching Example ===")
    
    # Example 1: Download and save default model
    print("\n1. Downloading default MiVOLO v2 model...")
    success = download_and_save_model(
        model_name="iitolstykh/mivolo_v2",
        save_path="./models/mivolo_v2_local"
    )
    
    if success:
        print("✓ Model download completed successfully")
        
        # Verify the saved model
        print("\n2. Verifying saved model...")
        is_valid = verify_local_model("./models/mivolo_v2_local")
        
        if is_valid:
            print("✓ Model verification passed")
            
            # Example usage with the cached model
            print("\n3. Example usage with cached model:")
            print("```python")
            print("from age_recognize.inference import TransformersInferenceEngine")
            print("from age_recognize.recognizer import EnhancedAgeRecognizer")
            print("")
            print("# Use the locally cached model")
            print("recognizer = EnhancedAgeRecognizer(device='cuda', verbose=True)")
            print("recognizer.inference_engine = TransformersInferenceEngine(")
            print("    model_path='./models/mivolo_v2_local',")
            print("    device='cuda'")
            print(")")
            print("```")
        else:
            print("✗ Model verification failed")
    else:
        print("✗ Model download failed")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and cache MiVOLO models")
    parser.add_argument(
        "--model", 
        default="iitolstykh/mivolo_v2",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--save-path", 
        default="./models/mivolo_v2_local",
        help="Local path to save the model"
    )
    parser.add_argument(
        "--verify", 
        type=str,
        help="Verify an existing local model at the given path"
    )
    parser.add_argument(
        "--example", 
        action="store_true",
        help="Run example usage"
    )
    
    args = parser.parse_args()
    
    if args.example:
        example_usage()
    elif args.verify:
        verify_local_model(args.verify)
    else:
        download_and_save_model(
            model_name=args.model,
            save_path=args.save_path
        )


if __name__ == "__main__":
    main()