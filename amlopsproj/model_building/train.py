import pandas as pd
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from huggingface_hub import HfApi
import xgboost as xgb

HF_TOKEN = os.getenv("HF_TOKEN")
run_hf_upload = os.getenv("RUN_HF_UPLOAD", "false").lower() == "true"
run_local = os.getenv("RUN_LOCAL", "false").lower() == "true"

if run_hf_upload and not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set but RUN_HF_UPLOAD=true")
    print("Either set HF_TOKEN or set RUN_HF_UPLOAD=false to run offline")
    sys.exit(1)

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI: {mlflow_tracking_uri}")

experiment_name = "tourism_package_prediction"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

BASE_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = BASE_DIR / "processed"
ENCODER_DIR = BASE_DIR / "artifacts" / "encoders"
DATASET_PATH_HF = "hf://datasets/spac1ngcat/tour-well-pack-ds/processed/"


def load_data():
    files = [PROCESSED_DIR / "Xtrain.csv", PROCESSED_DIR / "Xtest.csv", PROCESSED_DIR / "ytrain.csv", PROCESSED_DIR / "ytest.csv"]

    if run_local or all(f.exists() for f in files):
        print(f"Loading datasets from local path: {PROCESSED_DIR}")
        try:
            Xtrain = pd.read_csv(files[0])
            Xtest = pd.read_csv(files[1])
            ytrain = pd.read_csv(files[2]).values.ravel()
            ytest = pd.read_csv(files[3]).values.ravel()
            print("Datasets loaded successfully from local files.")
            return Xtrain, Xtest, ytrain, ytest
        except Exception as e:
            if run_local:
                print(f"Error: RUN_LOCAL=true but failed to load local datasets: {e}")
                sys.exit(1)
            print(f"Local load failed, attempting Hugging Face fallback...")

    print(f"Loading datasets from Hugging Face: {DATASET_PATH_HF}")
    try:
        Xtrain = pd.read_csv(DATASET_PATH_HF + "Xtrain.csv")
        Xtest = pd.read_csv(DATASET_PATH_HF + "Xtest.csv")
        ytrain = pd.read_csv(DATASET_PATH_HF + "ytrain.csv").values.ravel()
        ytest = pd.read_csv(DATASET_PATH_HF + "ytest.csv").values.ravel()
        print("Datasets loaded successfully from Hugging Face.")
        return Xtrain, Xtest, ytrain, ytest
    except Exception as e:
        print(f"Error: Failed to load datasets from Hugging Face: {e}")
        sys.exit(1)

Xtrain, Xtest, ytrain, ytest = load_data()

print(f"Dataset shapes:")
print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
print(f"ytrain: {ytrain.shape}, ytest: {ytest.shape}")

class_weight = ytrain[ytrain==0].shape[0] / ytrain[ytrain==1].shape[0]
print(f"Class imbalance ratio: {class_weight:.2f}:1")

search_spaces = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(3, 7),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.8, 1.0),
    'scale_pos_weight': [class_weight]
}

base_model = xgb.XGBClassifier(random_state=42)

bayes_search = BayesSearchCV(
    estimator=base_model,
    search_spaces=search_spaces,
    n_iter=50,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("Starting hyperparameter tuning with BayesSearchCV...")
print(f"Number of iterations: 50 (Bayesian optimization)")

with mlflow.start_run(run_name="xgboost_bayesian_search"):
    bayes_search.fit(Xtrain, ytrain)

    print("Logging cross-validation results to MLflow (no extra refits)...")
    cv_results = bayes_search.cv_results_
    for i, params in enumerate(cv_results['params']):
        with mlflow.start_run(run_name=f"xgboost_config_{i+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "cv_mean_f1": cv_results['mean_test_score'][i],
                "cv_std_f1": cv_results['std_test_score'][i]
            })

    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_

    print(f"Best parameters found by Bayesian optimization:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best cross-validation F1 score: {bayes_search.best_score_:.4f}")

    ytrain_pred_best = best_model.predict(Xtrain)
    ytest_pred_best = best_model.predict(Xtest)

    train_acc_best = accuracy_score(ytrain, ytrain_pred_best)
    train_precision_best = precision_score(ytrain, ytrain_pred_best)
    train_recall_best = recall_score(ytrain, ytrain_pred_best)
    train_f1_best = f1_score(ytrain, ytrain_pred_best)

    test_acc_best = accuracy_score(ytest, ytest_pred_best)
    test_precision_best = precision_score(ytest, ytest_pred_best)
    test_recall_best = recall_score(ytest, ytest_pred_best)
    test_f1_best = f1_score(ytest, ytest_pred_best)

    print("Best Model Performance:")
    print(f"Train - Acc: {train_acc_best:.4f}, Precision: {train_precision_best:.4f}, Recall: {train_recall_best:.4f}, F1: {train_f1_best:.4f}")
    print(f"Test  - Acc: {test_acc_best:.4f}, Precision: {test_precision_best:.4f}, Recall: {test_recall_best:.4f}, F1: {test_f1_best:.4f}")

    print("Classification Report (Test Set):")
    print(classification_report(ytest, ytest_pred_best))

    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "best_train_accuracy": train_acc_best,
        "best_train_precision": train_precision_best,
        "best_train_recall": train_recall_best,
        "best_train_f1_score": train_f1_best,
        "best_test_accuracy": test_acc_best,
        "best_test_precision": test_precision_best,
        "best_test_recall": test_recall_best,
        "best_test_f1_score": test_f1_best,
        "cv_f1_score": bayes_search.best_score_
    })

    mlflow.sklearn.log_model(best_model, "model")

    joblib.dump(best_model, "tour_model.joblib")
    print("Best model saved as tour_model.joblib")

    if run_hf_upload:
        api = HfApi(token=HF_TOKEN)
        model_repo_id = "spac1ngcat/tour-well-pack-model"
        try:
            api.create_repo(repo_id=model_repo_id, repo_type="model", private=False, exist_ok=True)
            print(f"Model repository '{model_repo_id}' ready.")
        except Exception as e:
            print(f"Repository creation info: {e}")
        api.upload_file(
            path_or_fileobj="tour_model.joblib",
            path_in_repo="tour_model.joblib",
            repo_id=model_repo_id,
            repo_type="model"
        )
        print(f"Model uploaded successfully to {model_repo_id}")
    else:
        print("Skipping Hugging Face upload (RUN_HF_UPLOAD not set). Model saved locally only.")

print("Training pipeline completed successfully!")
