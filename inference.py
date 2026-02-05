import os
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from feature_builder import prepare_input_features
import streamlit as st


@st.cache_data
def load_models():
    """
    Downloads models from Hugging Face once and caches them.
    """

    classifier_path = hf_hub_download(
        repo_id="asmithaaa/emi-eligibility-model",
        filename="emi_eligibility_classifier.pkl",
        token=st.secrets["HF_TOKEN"]
    )

    regressor_path = hf_hub_download(
        repo_id="asmithaaa/emi-eligibility-model",
        filename="emi_model.pkl",
        token=st.secrets["HF_TOKEN"]
        
    )

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)

    return classifier, regressor


# Load models once (important for Streamlit performance)
classifier, regressor = load_models()


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------
def predict_emi(raw_input: dict):
    """
    Args:
        raw_input (dict): User input from Streamlit form

    Returns:
        eligibility_label (str): Eligible | High Risk | Not Eligible
        max_emi (float): Predicted maximum EMI
    """

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    input_df = prepare_input_features(raw_input)

    # -------------------------
    # HARD BUSINESS RULES
    # -------------------------
    hard_reject = (
        input_df["credit_score"].iloc[0] < 450
        or input_df["debt_to_income"].iloc[0] > 0.75
        or input_df["expense_to_income"].iloc[0] > 0.85
    )

    # -------------------------
    # ML PREDICTIONS
    # -------------------------
    eligibility_pred = classifier.predict(input_df)[0]
    max_emi = float(regressor.predict(input_df)[0])
    max_emi = max(max_emi, 0.0)  # safety clamp

    # -------------------------
    # FINAL DECISION LOGIC
    # -------------------------
    if hard_reject and eligibility_pred != "Eligible":
        eligibility_label = "Not Eligible"

    elif eligibility_pred in ["Eligible", "ELIGIBLE"]:
        eligibility_label = "Eligible"

    elif eligibility_pred in ["High Risk", "High_Risk", "HIGH_RISK"]:
        eligibility_label = "High Risk"

    else:
        eligibility_label = "Not Eligible"

    return eligibility_label, max_emi
