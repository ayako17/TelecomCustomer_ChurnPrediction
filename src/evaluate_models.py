from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils import ensure_dir, load_yaml_config


def evaluate_classification_model(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Return common classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan"),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    metrics_dir = project_root / config["paths"]["metrics_dir"]
    output_file = metrics_dir / "model_comparison.csv"

    ensure_dir(metrics_dir)

    # TODO: load test data and trained models, then evaluate each model.
    comparison_df = pd.DataFrame(
        columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    comparison_df.to_csv(output_file, index=False)
    print(f"Model comparison template saved to: {output_file}")


if __name__ == "__main__":
    main()
