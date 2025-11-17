# ml/config.py

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 100  # skenario 2: adjust 50 → 100
    model_dir: Path = BASE_DIR / "models"
    model_path: Path = model_dir / "model.pkl"
    metrics_path: Path = model_dir / "metrics.json"


config = TrainingConfig()
