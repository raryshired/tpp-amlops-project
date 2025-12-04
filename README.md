# Tourism Wellness Package Prediction - MLOps Pipeline

End-to-end MLOps pipeline for predicting customer purchases using XGBoost with automated CI/CD via GitHub Actions.

## Project Overview
- Automated data registration and preprocessing
- Bayesian hyperparameter optimization
- MLflow experiment tracking
- Deployment to Hugging Face Spaces
- CI/CD with GitHub Actions

## Architecture
```
├── amlopsproj/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── artifacts/encoders/
│   └── model_building/
│       ├── data_register.py
│       ├── prep.py
│       └── train.py
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .github/workflows/
│   └── pipeline.yml
├── hf_space_hosting.py
├── requirements-pipeline.txt
├── tourism.csv
└── README.md
```

## Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-pipeline.txt

# Data registration
python amlopsproj/model_building/data_register.py

# Data prep
python amlopsproj/model_building/prep.py

# Train (start MLflow if desired)
mlflow ui --host 0.0.0.0 --port 5001 &
python amlopsproj/model_building/train.py

# Docker
docker build -t tourism-app -f deployment/Dockerfile deployment/
docker run -p 8501:8501 -e HF_TOKEN=$HF_TOKEN tourism-app
```

## Links
- Dataset: https://huggingface.co/datasets/spac1ngcat/tour-well-pack-ds
- Model: https://huggingface.co/spac1ngcat/tour-well-pack-model
- App: https://huggingface.co/spaces/spac1ngcat/tour-well-pack-app
