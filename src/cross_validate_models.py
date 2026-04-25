from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir, load_yaml_config


def get_model_candidates(random_state: int = 42) -> dict[str, Any]:
    """Build model candidates and skip unavailable dependencies gracefully."""
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    except ModuleNotFoundError as error:
        print(f"xgboost not available, skipping. Details: {error}")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
            verbose=-1,
        )
    except ModuleNotFoundError as error:
        print(f"lightgbm not available, skipping. Details: {error}")

    return models


def run_cross_validation(X: pd.DataFrame, y: pd.Series, model: Any) -> dict[str, float]:
    """Run 5-fold stratified cross validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "f1", "recall"]

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
        "recall_mean": float(np.mean(scores["test_recall"])),
        "recall_std": float(np.std(scores["test_recall"])),
    }


def plot_auc_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Plot CV ROC-AUC mean with std error bars."""
    ensure_dir(output_path.parent)
    ordered = results_df.sort_values("roc_auc_mean", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(
        ordered["model"],
        ordered["roc_auc_mean"],
        yerr=ordered["roc_auc_std"],
        capsize=4,
    )
    plt.ylim(0.5, 1.0)
    plt.ylabel("ROC-AUC (mean ± std)")
    plt.xlabel("Model")
    plt.title("5-Fold Cross Validation ROC-AUC")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    train_path = project_root / config["paths"]["train_data"]
    metrics_dir = project_root / config["paths"]["metrics_dir"]
    figures_dir = project_root / config["paths"]["figure_dir"]
    target_column = config["model"]["target_column"]
    random_state = int(config["project"]["random_state"])

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    print(f"Cross-validation input shape: X={X.shape}, y={y.shape}")
    models = get_model_candidates(random_state=random_state)

    rows = []
    for model_name, model in models.items():
        try:
            result = run_cross_validation(X, y, model)
            rows.append({"model": model_name, **result})
            print(
                f"{model_name}: roc_auc={result['roc_auc_mean']:.4f}±{result['roc_auc_std']:.4f}, "
                f"f1={result['f1_mean']:.4f}±{result['f1_std']:.4f}, "
                f"recall={result['recall_mean']:.4f}±{result['recall_std']:.4f}"
            )
        except Exception as error:
            print(f"{model_name}: cross-validation failed, skipped. Details: {error}")

    if not rows:
        print("No model completed cross-validation. No output generated.")
        return

    results_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    results_path = metrics_dir / "cross_validation_results.csv"
    figure_path = figures_dir / "cross_validation_auc.png"

    ensure_dir(results_path.parent)
    results_df.to_csv(results_path, index=False)
    plot_auc_comparison(results_df, figure_path)

    print(f"Saved cross-validation metrics: {results_path}")
    print(f"Saved cross-validation figure: {figure_path}")


if __name__ == "__main__":
    main()
