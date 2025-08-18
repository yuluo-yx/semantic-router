#!/usr/bin/env python3
"""
Script to upload PII classifier model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_pii_model():
    # Model details
    model_path = "./pii_classifier_modernbert-base_presidio_token_model"
    
    # Get username from environment or prompt
    username = input("Enter your Hugging Face username: ")
    repo_name = f"{username}/pii-classifier-modernbert-base-presidio"
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_name, exist_ok=True)
        print(f"Repository {repo_name} created/confirmed")
    except Exception as e:
        print(f"Repository creation failed: {e}")
        return
    
    # Upload model files
    try:
        print(f"Uploading model from {model_path} to {repo_name}...")
        
        # Upload all files in the model directory
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model"
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_pii_model()
