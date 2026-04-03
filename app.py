import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from dashboard_core import get_metrics, get_training_curves, predict_proba, load_model_and_scaler
from path_utils import CHARTS

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Risk Predictor")
st.caption("Raw input -> saved preprocessor (scaler+encoder) -> trained ANN model")

# --- HEALTH CHECK ---
model, preprocessor = load_model_and_scaler()
with st.sidebar:
    st.header("Clinical Inputs")
    if model and preprocessor:
        st.success("✅ System Ready: Model & Preprocessor Loaded")
    else:
        st.error("❌ System Error: Missing Model/Preprocessor Artifacts")
        st.info("Ensure `models/` and `data/artifacts/` folders are populated.")

features = {
    "Age": st.sidebar.slider("Age", 20, 90, 52),
    "Sex": st.sidebar.selectbox("Sex", ["M", "F"]),
    "ChestPainType": st.sidebar.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"]),
    "RestingBP": st.sidebar.slider("Resting BP", 80, 220, 130),
    "Cholesterol": st.sidebar.slider("Cholesterol", 0, 600, 250),
    "FastingBS": st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1]),
    "RestingECG": st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"]),
    "MaxHR": st.sidebar.slider("Max HR", 60, 220, 140),
    "ExerciseAngina": st.sidebar.selectbox("Exercise Angina", ["N", "Y"]),
    "Oldpeak": st.sidebar.slider("Oldpeak", 0.0, 6.5, 1.2, 0.1),
    "ST_Slope": st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"]),
}

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Predict", disabled=(model is None)):
    try:
        prob = predict_proba(features)
        label = "Heart Disease Likely" if prob >= 0.5 else "Low Risk"
        st.session_state.prediction_history.append(prob)

        st.subheader("Live Prediction Result")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Probability", f"{prob:.3f}")
            st.progress(int(prob * 100))
        with c2:
            st.metric("Predicted Class", label)

        hist = np.array(st.session_state.prediction_history, dtype=float)
        st.line_chart(hist)
        st.caption("This line chart is real-time and updates after each Predict click.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

curves_tab, charts_tab, arch_tab = st.tabs(
    ["Training Curves", "Evaluation Charts", "Model Architecture"]
)

with curves_tab:
    curves = get_training_curves()
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(1, len(curves["train_loss"]) + 1)
        axes[0].plot(x, curves["train_loss"], label="Train")
        axes[0].plot(x, curves["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(x, curves["train_acc"], label="Train")
        axes[1].plot(x, curves["val_acc"], label="Val")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        st.pyplot(fig)

with charts_tab:
    cm_path = CHARTS / "confusion_matrix.png"
    roc_path = CHARTS / "roc_curve.png"
    pr_path = CHARTS / "precision_recall_curve.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix")
    if roc_path.exists():
        st.image(str(roc_path), caption="ROC Curve")
    if pr_path.exists():
        st.image(str(pr_path), caption="Precision-Recall Curve")

with arch_tab:
    st.code(
        "Input(15) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.35)\n"
        "          -> Linear(64)  -> BatchNorm1d -> ReLU -> Dropout(0.30)\n"
        "          -> Linear(32)  -> ReLU -> Dropout(0.20)\n"
        "          -> Linear(1)   -> Sigmoid",
        language="text",
    )
    metrics = get_metrics()
    if metrics:
        st.write("Saved test metrics:")
        primary_cols = st.columns(5)
        primary_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for col, key in zip(primary_cols, primary_keys):
            if key in metrics:
                col.metric(key.upper(), f"{metrics[key]:.4f}")

        extra_keys = ["specificity", "npv", "balanced_accuracy", "mcc", "pr_auc", "brier_score"]
        extra = {k: metrics[k] for k in extra_keys if k in metrics}
        if extra:
            st.write("Additional metrics:")
            st.json(extra)

        st.markdown(
            """
            **How to read quickly**
            - Higher is better: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, NPV, Balanced Accuracy, MCC
            - Lower is better: Brier score
            - Precision is key when false positives are costly
            - Recall is key when missing a positive case is risky
            """
        )
        st.info(
            "Simple meaning: Precision = when model says 'disease', how often it is correct. "
            "Use this when false alarms are costly. "
            "Recall = of all real disease cases, how many were caught. "
            "Use this when missing a patient is risky. "
            "For heart screening, recall is usually prioritized first."
        )
        st.markdown("### Metric Meaning (Quick Guide)")
        c1, c2 = st.columns(2)
        with c1:
            st.success(
                "**Precision important**\n\n"
                "If model says YES, it should usually be truly YES.\n\n"
                "Use when false alarms are expensive.\n"
                "Example: model says disease -> costly tests happen.\n"
                "If many are false alarms, time/money is wasted."
            )
            st.info(
                "**Recall important**\n\n"
                "Catch as many real YES cases as possible.\n\n"
                "Use when missing a real case is dangerous.\n"
                "Example: model says no disease but patient actually has disease."
            )
            st.warning(
                "**F1-score**\n\n"
                "Best when you want a balance of Precision and Recall in one number."
            )
        with c2:
            st.success(
                "**Accuracy**\n\n"
                "Overall correct predictions.\n"
                "Good when classes are balanced."
            )
            st.info(
                "**ROC-AUC / PR-AUC**\n\n"
                "How well model ranks high-risk vs low-risk.\n"
                "PR-AUC is very useful when positive class is rare."
            )
            st.warning(
                "**Specificity / NPV / MCC / Balanced Accuracy / Brier**\n\n"
                "Specificity: catches true negatives\n"
                "NPV: reliability of negative prediction\n"
                "MCC/Balanced Accuracy: robust summary under imbalance\n"
                "Brier: probability quality (lower is better)"
            )
