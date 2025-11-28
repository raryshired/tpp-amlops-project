from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import sys
from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set or empty.")
    print("Please set HF_TOKEN before running this script.")
    sys.exit(1)

repo_id = "spac1ngcat/tour-well-pack-ds"
repo_type = "dataset"

base_dir = Path(__file__).resolve().parents[1] / "data"
raw_dir = base_dir / "raw"
processed_dir = base_dir / "processed"
encoders_dir = base_dir / "artifacts" / "encoders"

if not raw_dir.exists():
    print(f"Error: Raw data folder not found at {raw_dir}")
    sys.exit(1)

tourism_csv = raw_dir / "tourism.csv"
if not tourism_csv.exists():
    print(f"Error: tourism.csv not found at {tourism_csv}")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists.")
    force_upload = os.getenv("FORCE_UPLOAD", "false").lower() == "true"
    if not force_upload:
        print("Skipping upload (repo exists). Set FORCE_UPLOAD=true to force upload.")
        sys.exit(0)
except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repository '{repo_id}' created.")

def upload_if_exists(path: Path, path_in_repo: str):
    if path.exists():
        api.upload_folder(
            folder_path=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"Uploaded folder {path} to {repo_id}/{path_in_repo}")
    else:
        print(f"Skipping upload; folder does not exist: {path}")

upload_if_exists(raw_dir, "raw")
upload_if_exists(processed_dir, "processed")
upload_if_exists(encoders_dir, "artifacts/encoders")

print(f"Dataset upload step completed for {repo_id}")
