import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
# User: Ashutosh1975
REPO_ID = "Ashutosh1975/ann-heart-disease-predictor" 
PROJECT_FOLDER = "/home/ashutosh/Desktop/NO-AI-USE/Deep-learning/ANN Binary Classification"

# DO NOT HARDCODE TOKENS. We use the local logged-in session.
api = HfApi()

print(f"🚀 Starting professional Docker-based deployment to {REPO_ID}...")

# 1. Create space if not exists
try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", private=False)
    print(f"✅ Created/Verified Space: {REPO_ID}")
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Space {REPO_ID} already exists. Proceeding with update.")
    else:
        print(f"❌ Error creating repo: {e}")

# 2. Upload the folder contents
# We include Dockerfile and EXCLUDE the push script itself for security
api.upload_folder(
    folder_path=PROJECT_FOLDER,
    repo_id=REPO_ID,
    repo_type="space",
    allow_patterns=["*.py", "*.txt", "*.md", "Dockerfile", "models/*", "data/artifacts/*", "data/raw/*", "charts/*", "data/processed/*.pkl"],
    ignore_patterns=["notebooks/*", "__pycache__/*", "*.pdf", "data/processed/*.json", ".git/*", ".gitattributes", "push_to_hf.py"],
)

print("-" * 30)
print(f"✅ SUCCESS! Your Dockerized app is live at: https://huggingface.co/spaces/{REPO_ID}")
print("Monitor build logs at the URL above. It will take ~5 mins to build the container.")
print("-" * 30)
