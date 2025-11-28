import pandas as pd
import os
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import joblib

HF_TOKEN = os.getenv("HF_TOKEN")
run_hf_upload = os.getenv("RUN_HF_UPLOAD", "false").lower() == "true"
run_local = os.getenv("RUN_LOCAL", "false").lower() == "true"

if run_hf_upload and not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set but RUN_HF_UPLOAD=true")
    print("Either set HF_TOKEN or set RUN_HF_UPLOAD=false to run offline")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
ENCODER_DIR = BASE_DIR / "artifacts" / "encoders"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH_HF = "hf://datasets/spac1ngcat/tour-well-pack-ds/raw/tourism.csv"
DATASET_PATH_LOCAL = RAW_DIR / "tourism.csv"

if DATASET_PATH_LOCAL.exists() or run_local:
    if not DATASET_PATH_LOCAL.exists():
        print(f"Error: RUN_LOCAL=true but local dataset not found at {DATASET_PATH_LOCAL}")
        sys.exit(1)
    print(f"Loading dataset from local path: {DATASET_PATH_LOCAL}")
    df = pd.read_csv(DATASET_PATH_LOCAL)
    print("Dataset loaded successfully from local file.")
else:
    print(f"Loading dataset from Hugging Face: {DATASET_PATH_HF}")
    try:
        df = pd.read_csv(DATASET_PATH_HF)
        print("Dataset loaded successfully from Hugging Face.")
    except Exception as e:
        print(f"Error: Failed to load dataset from Hugging Face: {e}")
        print(f"Please ensure either:")
        print(f"  1. Place tourism.csv at: {DATASET_PATH_LOCAL}")
        print(f"  2. Set RUN_LOCAL=true to force local mode")
        print(f"  3. Ensure network access to: {DATASET_PATH_HF}")
        sys.exit(1)

df = df.drop(columns=['Unnamed: 0', 'CustomerID'])

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

categorical_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

joblib.dump(label_encoders, ENCODER_DIR / 'label_encoders.pkl')
print(f"Label encoders saved to {ENCODER_DIR / 'label_encoders.pkl'}")

encoder_mappings = {col: le.classes_.tolist() for col, le in label_encoders.items()}
with open(ENCODER_DIR / 'label_encoders.json', 'w') as f:
    json.dump(encoder_mappings, f, indent=2)
print(f"Label encoder mappings saved to {ENCODER_DIR / 'label_encoders.json'}")

target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Xtrain_path = PROCESSED_DIR / 'Xtrain.csv'
Xtest_path = PROCESSED_DIR / 'Xtest.csv'
ytrain_path = PROCESSED_DIR / 'ytrain.csv'
ytest_path = PROCESSED_DIR / 'ytest.csv'

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Processed datasets saved locally.")
print(f"Train/Test feature files: {Xtrain_path}, {Xtest_path}")
print(f"Train/Test target files: {ytrain_path}, {ytest_path}")

if run_hf_upload:
    api = HfApi(token=HF_TOKEN)
    files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path, ENCODER_DIR / 'label_encoders.pkl', ENCODER_DIR / 'label_encoders.json']

    for file_path in files:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=str(file_path.relative_to(BASE_DIR)),
            repo_id="spac1ngcat/tour-well-pack-ds",
            repo_type="dataset",
        )

    print("Data preparation completed and files uploaded to Hugging Face.")
else:
    print("Skipping Hugging Face upload (RUN_HF_UPLOAD not set). Files saved locally only.")
