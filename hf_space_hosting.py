import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set")
    sys.exit(1)

SPACE_ID = "spac1ngcat/tour-well-pack-app"
DEPLOY_DIR = "deployment"

api = HfApi(token=HF_TOKEN)

"""
# Check/create space
try:
    api.repo_info(repo_id=SPACE_ID, repo_type="space")
    print(f"Space {SPACE_ID} exists. Skipping creation.")
except Exception:
    try:
        api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="streamlit", private=False, exist_ok=True)
        print(f"Space {SPACE_ID} created.")
    except Exception as e:
        print(f"Failed to create space: {e}")
        sys.exit(1)
"""

# Create the HF Streamlit Space manually
# Upload deployment folder
print("Uploading deployment folder...")
try:
    api.upload_folder(
        folder_path=DEPLOY_DIR,
        repo_id=SPACE_ID,
        repo_type="space",
        path_in_repo="",
    )
    print(f"Deployment complete. App: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"Upload failed: {e}")
    sys.exit(1)
