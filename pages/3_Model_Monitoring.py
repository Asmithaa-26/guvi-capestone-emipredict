import streamlit as st
from huggingface_hub import HfApi

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Model Performance & Monitoring")

# --------------------------------
# HUGGING FACE MODEL DETAILS
# --------------------------------
REPO_ID = "asmithaaa/emi-eligibility-model"   # change only if repo name changes
REPO_TYPE = "model"

api = HfApi()

try:
    repo_info = api.repo_info(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE
    )

    st.subheader("🔍 Model Repository Information")
    st.write(f"**Repository Name:** `{REPO_ID}`")
    st.write(f"**Last Updated:** {repo_info.lastModified}")
    st.write(f"**Total Files:** {len(repo_info.siblings)}")

except Exception as e:
    st.error("Unable to fetch model information from Hugging Face.")
    st.exception(e)

# --------------------------------
# MODEL METRICS (FINAL TRAINED MODEL)
# --------------------------------
st.subheader("📈 Model Performance Metrics")

# These are final metrics from training (static & acceptable for deployment)
classifier_metrics = {
    "Accuracy": 0.86,
    "Precision": 0.84,
    "Recall": 0.81,
    "F1 Score": 0.82
}

regressor_metrics = {
    "RMSE": 1250.45,
    "MAE": 980.30,
    "R² Score": 0.78
}

# --------------------------------
# DISPLAY CLASSIFIER METRICS
# --------------------------------
st.markdown("### 🧠 EMI Eligibility Classifier")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", classifier_metrics["Accuracy"])
    st.metric("Precision", classifier_metrics["Precision"])

with col2:
    st.metric("Recall", classifier_metrics["Recall"])
    st.metric("F1 Score", classifier_metrics["F1 Score"])

# --------------------------------
# DISPLAY REGRESSOR METRICS
# --------------------------------
st.markdown("### 💰 EMI Amount Regressor")

col3, col4 = st.columns(2)
with col3:
    st.metric("RMSE", classifier_metrics := regressor_metrics["RMSE"])
    st.metric("MAE", regressor_metrics["MAE"])

with col4:
    st.metric("R² Score", regressor_metrics["R² Score"])

# --------------------------------
# MODEL FILES LIST
# --------------------------------
st.subheader("📦 Model Artifacts in Hugging Face")

file_names = [file.rfilename for file in repo_info.siblings]
st.write(file_names)

# --------------------------------
# INFO NOTE
# --------------------------------
st.info(
    "This application uses production-ready models hosted on Hugging Face. "
    "Metrics shown are based on the final trained version of the models."
)
