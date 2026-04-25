from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir, load_yaml_config, save_pickle


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Pipeline:
    """Train Logistic Regression in a scaling pipeline."""
    model = Pipeline(
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
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    try:
        model.fit(X_train, y_train)
    except OSError as error:
        if getattr(error, "winerror", None) == 5 and n_jobs != 1:
            print("random_forest: 并行训练权限受限，自动回退为 n_jobs=1 重试。")
            return train_random_forest(
                X_train, y_train, random_state=random_state, n_jobs=1
            )
        raise
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Any:
    """Train XGBoost model. Raises ModuleNotFoundError if xgboost is unavailable."""
    from xgboost import XGBClassifier

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = (negative_count / positive_count) if positive_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Any:
    """Train LightGBM model. Raises ModuleNotFoundError if lightgbm is unavailable."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, output_path: str | Path) -> None:
    """Save model object to pickle file."""
    save_pickle(model, output_path)


def evaluate_model_roc_auc(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Compute ROC-AUC on test set using predict_proba."""
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    train_path = project_root / config["paths"]["train_data"]
    test_path = project_root / config["paths"]["test_data"]
    model_dir = project_root / config["paths"]["model_dir"]
    target_column = config["model"]["target_column"]
    random_state = int(config["project"]["random_state"])

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train/Test data not found. Please run preprocess.py first."
        )

    ensure_dir(model_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    print(f"训练集维度: {X_train.shape}")
    print(f"测试集维度: {X_test.shape}")
    print(f"正样本数量: {positive_count}")
    print(f"负样本数量: {negative_count}")

    model_specs = [
        ("logistic_regression", train_logistic_regression, "logistic_regression.pkl"),
        ("random_forest", train_random_forest, "random_forest.pkl"),
        ("xgboost", train_xgboost, "xgboost_model.pkl"),
        ("lightgbm", train_lightgbm, "lightgbm_model.pkl"),
    ]

    trained_models: dict[str, Any] = {}
    roc_auc_scores: dict[str, float] = {}

    for model_name, trainer, model_file in model_specs:
        try:
            model = trainer(X_train, y_train, random_state=random_state)
            model_path = model_dir / model_file
            save_model(model, model_path)
            trained_models[model_name] = model
            roc_auc_scores[model_name] = evaluate_model_roc_auc(model, X_test, y_test)
            print(
                f"{model_name}: 训练成功，ROC-AUC={roc_auc_scores[model_name]:.4f}，"
                f"模型已保存到 {model_path}"
            )
        except ModuleNotFoundError as error:
            print(f"{model_name}: 依赖缺失，已跳过。详情: {error}")
        except Exception as error:  # pragma: no cover - runtime guard
            print(f"{model_name}: 训练失败，已跳过。详情: {error}")

    if not roc_auc_scores:
        print("没有可用模型训练成功，未生成 best_model.pkl。")
        return

    best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
    best_model_auc = roc_auc_scores[best_model_name]
    best_model_path = model_dir / "best_model.pkl"
    save_model(trained_models[best_model_name], best_model_path)

    print(f"最佳模型名称: {best_model_name}")
    print(f"最佳模型 ROC-AUC: {best_model_auc:.4f}")
    print(f"最佳模型已保存到: {best_model_path}")


if __name__ == "__main__":
    main()
