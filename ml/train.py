# ml/train.py

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import config


def train():
    # ensure model directory exists
    config.model_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data.target,
    )

    # initialize model
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1,
    )

    # train model
    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # save model
    joblib.dump(clf, config.model_path)

    # save metrics
    metrics = {
        "accuracy": acc,
        "n_estimators": config.n_estimators,
    }

    with open(config.metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f"Model saved to {config.model_path}")
    print(f"Metrics: {metrics}")

    # simple "gate"
    if acc < 0.9:
        raise ValueError(f"Accuracy too low: {acc:.3f}")


if __name__ == "__main__":
    train()
