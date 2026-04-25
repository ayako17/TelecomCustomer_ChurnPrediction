from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import build_features
from utils import ensure_dir, load_yaml_config, save_json


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """执行基础清洗并构建业务特征。"""
    cleaned_df = df.copy()

    if "customerID" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["customerID"])

    if "TotalCharges" in cleaned_df.columns:
        cleaned_df["TotalCharges"] = pd.to_numeric(
            cleaned_df["TotalCharges"], errors="coerce"
        )
        total_median = cleaned_df["TotalCharges"].median()
        if pd.isna(total_median):
            total_median = 0.0
        cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].fillna(total_median)

    if "Churn" in cleaned_df.columns:
        churn_raw = cleaned_df["Churn"]
        churn_mapped = (
            churn_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        )
        churn_numeric = pd.to_numeric(churn_raw, errors="coerce")
        cleaned_df["Churn"] = churn_mapped.fillna(churn_numeric).fillna(0).astype(int)

    cleaned_df = build_features(cleaned_df)
    return cleaned_df


def encode_features(
    df: pd.DataFrame,
    target_column: str = "Churn",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """特征编码并返回 X、y 与特征列名。"""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()
    return X_encoded, y, feature_columns


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按分层采样切分训练集和测试集。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    raw_path = project_root / config["paths"]["raw_data"]
    processed_path = project_root / config["paths"]["processed_data"]
    train_path = project_root / config["paths"]["train_data"]
    test_path = project_root / config["paths"]["test_data"]
    feature_columns_path = project_root / config["paths"]["feature_columns"]
    target_column = config["model"]["target_column"]

    if not raw_path.exists():
        print(
            "未找到原始数据文件，请将 Telco-Customer-Churn.csv 放入 data/raw/ 目录。"
        )
        return

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_data(raw_df)
    X, y, feature_columns = encode_features(cleaned_df, target_column=target_column)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["project"]["random_state"],
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train.values
    test_df = X_test.copy()
    test_df[target_column] = y_test.values

    ensure_dir(processed_path.parent)
    ensure_dir(train_path.parent)
    ensure_dir(test_path.parent)
    ensure_dir(feature_columns_path.parent)

    cleaned_df.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    save_json(feature_columns, feature_columns_path)

    target_distribution = y.value_counts(dropna=False).sort_index()
    target_ratio = y.value_counts(normalize=True, dropna=False).sort_index()

    print("原始数据维度:", raw_df.shape)
    print("清洗后数据维度:", cleaned_df.shape)
    print("训练集维度:", train_df.shape)
    print("测试集维度:", test_df.shape)
    print("目标变量分布:")
    for label, count in target_distribution.items():
        ratio = target_ratio.get(label, 0.0)
        print(f"  {label}: {count} ({ratio:.2%})")

    print("输出文件保存路径:")
    print(f"  {processed_path}")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {feature_columns_path}")


if __name__ == "__main__":
    main()
