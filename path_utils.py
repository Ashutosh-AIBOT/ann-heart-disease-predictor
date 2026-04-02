from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_ARTIFACTS = ROOT / "data" / "artifacts"
MODELS = ROOT / "models"
CHARTS = ROOT / "charts"

for folder in (DATA_RAW, DATA_PROCESSED, DATA_ARTIFACTS, MODELS, CHARTS):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # In Docker production (Hugging Face), these folders already exist in the image
        # but the user UID 1000 may not have 'mkdir' permissions on the /app mount.
        pass
