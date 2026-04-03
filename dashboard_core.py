from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from path_utils import DATA_ARTIFACTS, MODELS


class HeartANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


_artifact_cache = {}
RAW_FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


def load_model_and_scaler() -> tuple[HeartANN | None, any | None]:
    if _artifact_cache:
        return _artifact_cache.get("model"), _artifact_cache.get("preprocessor")

    try:
        prep_path = DATA_ARTIFACTS / "preprocessor.pkl"
        if not prep_path.exists():
            print(f"❌ Error: {prep_path} missing.")
            return None, None
            
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)

        model_path = MODELS / "model.pkl"
        if not model_path.exists():
            print(f"❌ Error: {model_path} missing.")
            return None, None

        input_dim = len(preprocessor.get_feature_names_out())
        model = HeartANN(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        _artifact_cache["model"] = model
        _artifact_cache["preprocessor"] = preprocessor
        return model, preprocessor
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None, None


def predict_proba(features_dict: dict) -> float:
    missing = [k for k in RAW_FEATURES if k not in features_dict]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    model, preprocessor = load_model_and_scaler()
    frame = pd.DataFrame([{k: features_dict[k] for k in RAW_FEATURES}])
    x = preprocessor.transform(frame).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        prob = model(x_tensor).item()
    return float(prob)


def get_training_curves() -> dict:
    path = DATA_ARTIFACTS / "training_history.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_metrics() -> dict:
    path = MODELS / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
