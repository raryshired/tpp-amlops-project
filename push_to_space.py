import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable not set")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
space_id = "spac1ngcat/tour-well-pack-app"
deployment_dir = Path("deployment")

print(f"Creating/updating Hugging Face Space: {space_id}")

try:
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        exist_ok=True,
    )
    print(f"Space '{space_id}' ready")
except Exception as e:
    print(f"Space creation info: {e}")

print("Uploading deployment files...")
files_to_upload = [
    (deployment_dir / "app.py", "app.py"),
    (deployment_dir / "requirements.txt", "requirements.txt"),
    (deployment_dir / "Dockerfile", "Dockerfile"),
]

for local_path, repo_path in files_to_upload:
    if local_path.exists():
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=space_id,
            repo_type="space",
        )
        print(f"  ✓ Uploaded {repo_path}")
    else:
        print(f"  ✗ Warning: {local_path} not found, skipping")

print("\nDeployment complete!")
print(f"App URL: https://huggingface.co/spaces/{space_id}")
