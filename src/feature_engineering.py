from pathlib import Path

import pandas as pd

from utils import ensure_dir, load_yaml_config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from raw/cleaned dataframe."""
    features_df = df.copy()

    # TODO: add service_count feature.
    # Example placeholder:
    # service_cols = ["PhoneService", "InternetService", "OnlineSecurity"]
    # features_df["service_count"] = ...

    # TODO: add avg_monthly_charge feature.
    # features_df["avg_monthly_charge"] = ...

    # TODO: add is_month_to_month feature.
    # features_df["is_month_to_month"] = ...

    return features_df


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
