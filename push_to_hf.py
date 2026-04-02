import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
# CHANGE THIS to your actual HF username (e.g. "ashutosh/heart-predictor")
REPO_ID = "YOUR_HF_USERNAME/ann-heart-disease-predictor" 
PROJECT_FOLDER = "/home/ashutosh/Desktop/NO-AI-USE/Deep-learning/ANN Binary Classification"

api = HfApi()

print(f"🚀 Starting deployment of {PROJECT_FOLDER} to {REPO_ID}...")

# 1. Create space if not exists
try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="streamlit")
    print(f"✅ Created/Verified Space: {REPO_ID}")
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Space {REPO_ID} already exists. Proceeding with update.")
    else:
        print(f"❌ Error creating repo: {e}")

# 2. Upload the folder contents
# We ignore notebooks, caches, and large processed datasets to keep the Space clean
api.upload_folder(
    folder_path=PROJECT_FOLDER,
    repo_id=REPO_ID,
    repo_type="space",
    allow_patterns=["*.py", "*.txt", "*.md", "models/*", "data/artifacts/*", "data/raw/*", "charts/*"],
    ignore_patterns=["notebooks/*", "__pycache__/*", "*.pdf", "data/processed/*", ".git/*"],
)

print("-" * 30)
print(f"✅ SUCCESS! Your app is deploying at: https://huggingface.co/spaces/{REPO_ID}")
print("Please allow 3-5 minutes for Hugging Face to build the environment.")
print("-" * 30)
