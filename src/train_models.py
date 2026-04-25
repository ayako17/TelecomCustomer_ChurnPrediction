from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import ensure_dir, load_yaml_config


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a Logistic Regression model."""
    # TODO: implement Logistic Regression training.
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression(max_iter=1000, random_state=42)
    # model.fit(X_train, y_train)
    # return model
    return None


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a Random Forest model."""
    # TODO: implement Random Forest training.
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(n_estimators=300, random_state=42)
    # model.fit(X_train, y_train)
    # return model
    return None


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train an XGBoost model."""
    # TODO: implement XGBoost training.
    # from xgboost import XGBClassifier
    # model = XGBClassifier(random_state=42, eval_metric="logloss")
    # model.fit(X_train, y_train)
    # return model
    return None


def save_model(model: Any, output_path: str | Path) -> None:
    """Save model object as a joblib file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    joblib.dump(model, output_path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    train_path = project_root / config["paths"]["train_data"]
    model_dir = project_root / config["paths"]["model_dir"]

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data not found: {train_path}. Please run preprocess.py first."
        )

    # TODO: load train data, split X/y, train models, and save trained models.
    _ = pd.read_csv(train_path)
    ensure_dir(model_dir)
    print("Model training script initialized. TODOs are ready for implementation.")


if __name__ == "__main__":
    main()
