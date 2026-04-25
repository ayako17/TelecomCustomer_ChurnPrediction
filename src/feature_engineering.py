from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dir, load_yaml_config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """根据业务规则构建特征。"""
    df = df.copy()

    def _get_text_series(column_name: str, default_value: str = "") -> pd.Series:
        if column_name in df.columns:
            return df[column_name]
        return pd.Series(default_value, index=df.index, dtype="object")

    def _get_numeric_series(column_name: str) -> pd.Series:
        if column_name in df.columns:
            return pd.to_numeric(df[column_name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    service_fields = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    service_markers = []
    for col in service_fields:
        if col in df.columns:
            marker = (
                df[col].astype(str).str.strip().str.lower().eq("yes").astype(int)
            )
        else:
            marker = pd.Series(0, index=df.index, dtype="int64")
        service_markers.append(marker)
    df["service_count"] = pd.concat(service_markers, axis=1).sum(axis=1)

    tenure = _get_numeric_series("tenure")
    monthly_charges = _get_numeric_series("MonthlyCharges")
    total_charges = _get_numeric_series("TotalCharges")

    avg_monthly_charge = total_charges.div(tenure.replace(0, np.nan))
    avg_monthly_charge = avg_monthly_charge.where(tenure.ne(0), monthly_charges)
    df["avg_monthly_charge"] = avg_monthly_charge.fillna(monthly_charges)

    df["is_month_to_month"] = (
        _get_text_series("Contract")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("month-to-month")
        .astype(int)
    )

    df["has_internet_service"] = (
        _get_text_series("InternetService")
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("no")
        .astype(int)
    )

    df["is_auto_payment"] = (
        _get_text_series("PaymentMethod")
        .astype(str)
        .str.contains("automatic", case=False, na=False)
        .astype(int)
    )

    monthly_median = monthly_charges.median()
    tenure_median = tenure.median()
    df["high_charge_low_tenure"] = (
        monthly_charges.gt(monthly_median) & tenure.lt(tenure_median)
    ).astype(int)

    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    input_path = project_root / config["paths"]["processed_data"]
    output_path = input_path.with_name("churn_features.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data not found: {input_path}. "
            "Please run preprocess.py first."
        )

    df = pd.read_csv(input_path)
    features_df = build_features(df)

    ensure_dir(output_path.parent)
    features_df.to_csv(output_path, index=False)
    print(f"Feature dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
