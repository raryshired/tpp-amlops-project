import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from pathlib import Path

MODEL_REPO = "spac1ngcat/tour-well-pack-model"
ENCODER_REPO = "spac1ngcat/tour-well-pack-ds"
MODEL_FILENAME = "tour_model.joblib"
ENCODER_PKL = "artifacts/encoders/label_encoders.pkl"
ENCODER_JSON = "artifacts/encoders/label_encoders.json"
LOCAL_ENCODER_DIR = Path(os.getenv("LOCAL_ENCODER_DIR", "/app/encoders"))
ALT_LOCAL_ENCODER_DIR = Path("./amlopsproj/data/artifacts/encoders")

# Get HF_TOKEN from environment for authenticated downloads
HF_TOKEN = os.getenv("HF_TOKEN")

def try_local(path: Path):
    return path if path.exists() else None

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN)
    return joblib.load(model_path)

@st.cache_resource(show_spinner=False)
def load_encoders():
    pkl_local = try_local(LOCAL_ENCODER_DIR / "label_encoders.pkl")
    json_local = try_local(LOCAL_ENCODER_DIR / "label_encoders.json")
    if not pkl_local or not json_local:
        pkl_local = try_local(ALT_LOCAL_ENCODER_DIR / "label_encoders.pkl")
        json_local = try_local(ALT_LOCAL_ENCODER_DIR / "label_encoders.json")
    if pkl_local and json_local:
        encoders = joblib.load(pkl_local)
        with open(json_local, "r") as f:
            mappings = json.load(f)
        st.info(f"Loaded encoders locally from {pkl_local.parent}")
        return encoders, mappings
    enc_path = hf_hub_download(repo_id=ENCODER_REPO, filename=ENCODER_PKL, token=HF_TOKEN)
    mapping_path = hf_hub_download(repo_id=ENCODER_REPO, filename=ENCODER_JSON, token=HF_TOKEN)
    encoders = joblib.load(enc_path)
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
    return encoders, mappings

model = load_model()
if not hasattr(model, "use_label_encoder"):
      setattr(model, "use_label_encoder", False)
if not hasattr(model, "gpu_id"):
      setattr(model, "gpu_id", -1)
encoders, mappings = load_encoders()


def encode_inputs(df: pd.DataFrame) -> pd.DataFrame:
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception as e:
                st.error(f"Encoding error for {col}: {e}")
                raise
    return df

st.title("Tourism Package Purchase Prediction")
st.markdown("Model + encoders loaded from Hugging Face (or local if provided)")

with st.form("prediction_form"):
    Age = st.number_input("Age", min_value=0, max_value=120, value=30)
    TypeofContact = st.selectbox("TypeofContact", mappings.get("TypeofContact", []))
    CityTier = st.selectbox("CityTier", [1, 2, 3])
    DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, value=10)
    Occupation = st.selectbox("Occupation", mappings.get("Occupation", []))
    Gender = st.selectbox("Gender", mappings.get("Gender", []))
    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, value=2)
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, value=2)
    ProductPitched = st.selectbox("ProductPitched", mappings.get("ProductPitched", []))
    PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3)
    MaritalStatus = st.selectbox("MaritalStatus", mappings.get("MaritalStatus", []))
    NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, value=1)
    Passport = st.selectbox("Passport", [0, 1])
    PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
    OwnCar = st.selectbox("OwnCar", [0, 1])
    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, value=0)
    Designation = st.selectbox("Designation", mappings.get("Designation", []))
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, value=20000)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([
        {
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "DurationOfPitch": DurationOfPitch,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "NumberOfFollowups": NumberOfFollowups,
            "ProductPitched": ProductPitched,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": Passport,
            "PitchSatisfactionScore": PitchSatisfactionScore,
            "OwnCar": OwnCar,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome,
        }
    ])

    try:
        encoded_df = encode_inputs(input_df.copy())
        preds = model.predict(encoded_df)
        probs = model.predict_proba(encoded_df)
        st.success(f"Prediction: {'Will Purchase' if preds[0]==1 else 'Will Not Purchase'}")
        st.write({"Prob_No": round(probs[0][0], 4), "Prob_Yes": round(probs[0][1], 4)})
    except Exception as e:
        st.error(f"Prediction failed: {e}")
