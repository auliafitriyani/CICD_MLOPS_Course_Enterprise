# app/model_loader.py

import os
from functools import lru_cache
from pathlib import Path

import joblib
from google.cloud import storage

LOCAL_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
GCS_MODEL_URI = os.getenv("GCS_MODEL_URI", "")  # contoh: gs://bucket/path/model.pkl


def _download_from_gcs(gcs_uri: str, local_path: Path):
    """Download a model file from Google Cloud Storage into local path."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS_MODEL_URI must start with gs://")

    _, path_part = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path_part.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


@lru_cache(maxsize=1)
def load_model():
    """
    Model loading priority:
    1. If GCS_MODEL_URI is set → download model from GCS to LOCAL_MODEL_PATH.
    2. Otherwise → load model directly from LOCAL_MODEL_PATH (bundled in Docker image).
    """

    if GCS_MODEL_URI:
        print(f"[model_loader] Downloading model from {GCS_MODEL_URI}")
        _download_from_gcs(GCS_MODEL_URI, LOCAL_MODEL_PATH)

    if not LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH}")

    print(f"[model_loader] Loading model from {LOCAL_MODEL_PATH}")
    return joblib.load(LOCAL_MODEL_PATH)
