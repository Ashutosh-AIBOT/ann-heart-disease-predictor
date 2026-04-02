---
title: Heart Disease Risk Predictor
emoji: ❤️
colorFrom: pink
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 Heart Disease Risk Predictor (ANN Binary Classification)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A professional-grade deep learning pipeline for classifying heart disease risk based on clinical features. This project uses an Artificial Neural Network (ANN) to provide highly accurate probability scores, helping in early-stage medical screening.

---

## 📋 Project Overview

### 1. The Problem
Heart disease remains a leading cause of mortality worldwide. Early detection is critical but clinical assessments often involve high-dimensional data that is difficult for manual review alone. The goal of this project is to build a **reliable, high-recall binary classifier** that can assist medical professionals in identifying high-risk patients.

### 2. The Dataset
* **Source**: Clinical data containing features like Age, Sex, Chest Pain Type, Resting BP, Cholesterol, and ST Segment characteristics.
* **Format**: Structured Tabular data (`heart.csv`).
* **Imbalance Handling**: The clinical nature of the data requires special attention to the "Heart Disease Likely" class to minimize false negatives.

### 3. The Solution (ANN Architecture)
We implemented a multi-layer **Artificial Neural Network (ANN)** using PyTorch:
* **Input Layer**: 11 clinical features (processed via One-Hot Encoding and Standard Scaling).
* **Hidden Layers**:
  * Dense (128 units) → BatchNorm → ReLU → Dropout (0.35)
  * Dense (64 units) → BatchNorm → ReLU → Dropout (0.30)
  * Dense (32 units) → ReLU → Dropout (0.20)
* **Output Layer**: Sigmoid activation for binary risk probability.
* **Loss Function**: Binary Cross Entropy (BCE) Loss.

---

## 📈 Numerical Results & Performance

The model was evaluated on a held-out test set with the following verified metrics:

| Metric | Value | Significance |
| :--- | :--- | :--- |
| **Accuracy** | **90.2%** | High overall predictive power. |
| **Recall** | **94.1%** | **Critical for Medicine**: Caught 94%+ of positive cases. |
| **Precision** | **88.9%** | High reliability of "At Risk" flags. |
| **ROC-AUC** | **94.8%** | Exceptional separation between safe and at-risk classes. |
| **F1-Score** | **91.4%** | Balanced performance on both classes. |
| **Brier Score**| **0.081** | High-quality probability calibration. |

---

## 🚀 Live Deployment

The project is deployed via **Streamlit Cloud** and **Hugging Face Spaces**.

### Local Host Instructions
1. **Activate Environment**:
   ```bash
   conda activate ml-env
   ```
2. **Run Application**:
   ```bash
   streamlit run app.py
   ```
3. **Access**: Open `http://localhost:8501` in your browser.

---

## 🛠️ GitHub Configuration & Workflow

To maintain and update this repository, use the following standardized commands:

```bash
# 1. Initialize & Track
git init
git add .

# 2. Commit Deployment Ready State
git commit -m "🚀 Production-ready: ANN Heart Disease Predictor with metrics & deployment"

# 3. Remote Configuration
git branch -M main
git remote add origin git@github.com:Ashutosh-AIBOT/ann-heart-disease-predictor.git

# 4. Push to Cloud
git push -u origin main
```

---

## 📂 Project Structure
```text
ANN Binary Classification/
├── app.py                # Main Streamlit Dashboard
├── dashboard_core.py     # Inference Logic & Model Class
├── path_utils.py          # Dynamic Path Management
├── requirements.txt       # Optimized for HF (CPU-only)
├── models/
│   ├── model.pkl          # Trained Weights
│   └── results.json       # Numerical Metrics (Saved)
├── data/
│   ├── artifacts/         # Scalers & Preprocessors
│   └── raw/               # Source Dataset
├── charts/                # Evaluation Plots (ROC, CM, PR)
└── notebooks/             # EDA & Training Pipelines
```

---

**Developed by [Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)**
