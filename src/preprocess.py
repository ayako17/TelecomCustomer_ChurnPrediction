from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ensure_dir, load_yaml_config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for data cleaning logic."""
    cleaned_df = df.copy()
    # TODO: implement missing value handling, type conversions, and outlier checks.
    return cleaned_df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sets."""
    if target_column in df.columns:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_column],
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    raw_path = project_root / config["paths"]["raw_data"]
    processed_path = project_root / config["paths"]["processed_data"]
    train_path = project_root / config["paths"]["train_data"]
    test_path = project_root / config["paths"]["test_data"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            "Please place Telco-Customer-Churn.csv under data/raw/."
        )

    df = pd.read_csv(raw_path)
    cleaned_df = clean_data(df)
    train_df, test_df = split_data(
        cleaned_df,
        target_column=config["model"]["target_column"],
        test_size=config["model"]["test_size"],
        random_state=config["project"]["random_state"],
    )

    ensure_dir(processed_path.parent)
    cleaned_df.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Preprocessing completed.")
    print(f"Cleaned data saved to: {processed_path}")
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    main()
