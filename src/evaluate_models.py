from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils import ensure_dir, load_pickle, load_yaml_config


def evaluate_classification_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate a binary classification model on the test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }
    return metrics


def get_risk_level(probability: float) -> str:
    """Map churn probability into risk levels."""
    if probability >= 0.7:
        return "高风险"
    if probability >= 0.4:
        return "中风险"
    return "低风险"


def load_available_models(model_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Load available model files from model directory."""
    model_files = {
        "logistic_regression": "logistic_regression.pkl",
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost_model.pkl",
        "lightgbm": "lightgbm_model.pkl",
    }

    loaded_models: dict[str, Any] = {}
    missing_models: list[str] = []

    for model_name, model_file in model_files.items():
        model_path = model_dir / model_file
        if not model_path.exists():
            print(f"{model_name}: 模型文件不存在，已跳过。")
            missing_models.append(model_name)
            continue
        loaded_models[model_name] = load_pickle(model_path)
        print(f"{model_name}: 模型加载成功。")

    return loaded_models, missing_models


def plot_model_comparison(
    comparison_df: pd.DataFrame, output_path: Path, model_order: list[str]
) -> None:
    """Plot F1, ROC-AUC, PR-AUC comparison for all models."""
    ensure_dir(output_path.parent)
    ordered = (
        comparison_df.set_index("model")
        .loc[model_order, ["f1", "roc_auc", "pr_auc"]]
        .reset_index()
    )

    x = np.arange(len(ordered["model"]))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, ordered["f1"], width=width, label="F1")
    plt.bar(x, ordered["roc_auc"], width=width, label="ROC-AUC")
    plt.bar(x + width, ordered["pr_auc"], width=width, label="PR-AUC")
    plt.xticks(x, ordered["model"], rotation=15)
    plt.ylim(0, 1.05)
    plt.title("Model Metric Comparison (F1 / ROC-AUC / PR-AUC)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curves(
    y_test: pd.Series, prediction_cache: dict[str, dict[str, np.ndarray]], output_path: Path
) -> None:
    """Plot ROC curves for all models."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 6))
    for model_name, values in prediction_cache.items():
        fpr, tpr, _ = roc_curve(y_test, values["y_prob"])
        auc_score = roc_auc_score(y_test, values["y_prob"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curves(
    y_test: pd.Series, prediction_cache: dict[str, dict[str, np.ndarray]], output_path: Path
) -> None:
    """Plot Precision-Recall curves for all models."""
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 6))
    for model_name, values in prediction_cache.items():
        precision, recall, _ = precision_recall_curve(y_test, values["y_prob"])
        pr_auc = average_precision_score(y_test, values["y_prob"])
        plt.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix_for_best_model(
    y_test: pd.Series,
    best_model_name: str,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confusion matrix for the best model."""
    ensure_dir(output_path.parent)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - Best Model ({best_model_name})")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    values = np.array([[tn, fp], [fn, tp]])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels[i, j]}\n{values[i, j]}", ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def extract_feature_importance(
    models: dict[str, Any], best_model_name: str
) -> tuple[str | None, np.ndarray | None]:
    """Find a model with available feature importances."""
    candidate_order = [best_model_name, "random_forest", "xgboost", "lightgbm"]

    for model_name in candidate_order:
        model = models.get(model_name)
        if model is None:
            continue

        if hasattr(model, "feature_importances_"):
            return model_name, np.asarray(model.feature_importances_)

        if hasattr(model, "named_steps"):
            steps = getattr(model, "named_steps")
            for _, step in reversed(list(steps.items())):
                if hasattr(step, "feature_importances_"):
                    return model_name, np.asarray(step.feature_importances_)

    return None, None


def plot_feature_importance(
    feature_names: list[str],
    feature_importances: np.ndarray,
    model_name: str,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Plot top-k feature importances."""
    ensure_dir(output_path.parent)
    importance_series = pd.Series(feature_importances, index=feature_names).sort_values(
        ascending=False
    )
    top_features = importance_series.head(top_k).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features.index, top_features.values)
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_threshold_analysis(
    y_test: pd.Series, y_prob: np.ndarray, output_csv: Path, output_figure: Path
) -> None:
    """Evaluate precision/recall/f1 on multiple thresholds and save outputs."""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    threshold_df.to_csv(output_csv, index=False)

    ensure_dir(output_figure.parent)
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    plt.xticks(threshold_df["threshold"])
    plt.ylim(0, 1.05)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Analysis (Best Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_figure, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    test_path = project_root / config["paths"]["test_data"]
    model_dir = project_root / config["paths"]["model_dir"]
    metrics_dir = project_root / config["paths"]["metrics_dir"]
    figure_dir = project_root / config["paths"]["figure_dir"]
    prediction_dir = project_root / config["paths"]["prediction_dir"]
    target_column = config["model"]["target_column"]

    ensure_dir(metrics_dir)
    ensure_dir(figure_dir)
    ensure_dir(prediction_dir)

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    models, missing_models = load_available_models(model_dir)
    if not models:
        print("未发现可评估模型，评估流程结束。")
        return

    comparison_rows = []
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    reports: list[str] = []

    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_classification_model(model, X_test, y_test, threshold=0.5)

        comparison_rows.append({"model": model_name, **metrics})
        prediction_cache[model_name] = {"y_prob": y_prob, "y_pred": y_pred}

        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        reports.append(f"[{model_name}]\n{report}\n")
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)
    model_order = comparison_df["model"].tolist()
    comparison_csv_path = metrics_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    reports_path = metrics_dir / "classification_reports.txt"
    with reports_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(reports))

    plot_model_comparison(comparison_df, figure_dir / "model_comparison.png", model_order)
    plot_roc_curves(y_test, prediction_cache, figure_dir / "roc_curve.png")
    plot_pr_curves(y_test, prediction_cache, figure_dir / "pr_curve.png")

    best_model_name = comparison_df.iloc[0]["model"]
    best_predictions = prediction_cache[best_model_name]
    best_y_prob = best_predictions["y_prob"]
    best_y_pred = best_predictions["y_pred"]

    plot_confusion_matrix_for_best_model(
        y_test,
        best_model_name,
        best_y_pred,
        figure_dir / "confusion_matrix_best_model.png",
    )

    importance_model_name, importances = extract_feature_importance(models, best_model_name)
    feature_importance_path = figure_dir / "feature_importance.png"
    if importances is not None and importance_model_name is not None:
        plot_feature_importance(
            X_test.columns.tolist(),
            importances,
            importance_model_name,
            feature_importance_path,
            top_k=15,
        )
    else:
        print("所有可用模型均不支持特征重要性，已跳过 feature_importance.png。")

    prediction_df = pd.DataFrame(
        {
            "true_label": y_test.values,
            "pred_label": best_y_pred,
            "churn_probability": best_y_prob,
        }
    )
    prediction_df["risk_level"] = prediction_df["churn_probability"].apply(get_risk_level)
    prediction_output_path = prediction_dir / "customer_churn_predictions.csv"
    prediction_df.to_csv(prediction_output_path, index=False)

    threshold_csv = metrics_dir / "threshold_analysis.csv"
    threshold_figure = figure_dir / "threshold_analysis.png"
    save_threshold_analysis(y_test, best_y_prob, threshold_csv, threshold_figure)

    print(f"最佳模型: {best_model_name} (ROC-AUC={comparison_df.iloc[0]['roc_auc']:.4f})")
    print(f"已保存指标文件: {comparison_csv_path}, {reports_path}, {threshold_csv}")
    print(
        "已保存图表文件: "
        f"{figure_dir / 'model_comparison.png'}, "
        f"{figure_dir / 'roc_curve.png'}, "
        f"{figure_dir / 'pr_curve.png'}, "
        f"{figure_dir / 'confusion_matrix_best_model.png'}, "
        f"{feature_importance_path if feature_importance_path.exists() else 'feature_importance.png(跳过)'}, "
        f"{threshold_figure}"
    )
    print(f"已保存预测结果文件: {prediction_output_path}")
    if missing_models:
        print(f"缺失并跳过的模型文件: {', '.join(missing_models)}")


if __name__ == "__main__":
    main()
