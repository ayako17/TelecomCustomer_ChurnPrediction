from pathlib import Path


def check_file_list(project_root: Path, files: list[str], label: str) -> list[str]:
    """Check files and return missing list."""
    missing_files: list[str] = []
    print(f"\n[{label}]")
    for rel_path in files:
        abs_path = project_root / rel_path
        if abs_path.exists():
            print(f"[OK] {rel_path}")
        else:
            print(f"[MISSING] {rel_path}")
            missing_files.append(rel_path)
    return missing_files


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    required_files = [
        # data
        "data/raw/Telco-Customer-Churn.csv",
        "data/processed/churn_cleaned.csv",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "data/processed/feature_columns.json",
        # models
        "models/logistic_regression.pkl",
        "models/random_forest.pkl",
        "models/xgboost_model.pkl",
        "models/lightgbm_model.pkl",
        "models/best_model.pkl",
        # metrics
        "outputs/metrics/model_comparison.csv",
        "outputs/metrics/classification_reports.txt",
        "outputs/metrics/threshold_analysis.csv",
        "outputs/metrics/shap_feature_importance.csv",
        "outputs/metrics/risk_segment_summary.csv",
        "outputs/metrics/cost_sensitive_analysis.csv",
        "outputs/metrics/final_submission_check_report.txt",
        # figures: model/eval
        "outputs/figures/model_comparison.png",
        "outputs/figures/roc_curve.png",
        "outputs/figures/pr_curve.png",
        "outputs/figures/confusion_matrix_best_model.png",
        "outputs/figures/feature_importance.png",
        "outputs/figures/threshold_analysis.png",
        "outputs/figures/cost_sensitive_analysis.png",
        # figures: shap
        "outputs/figures/shap_summary.png",
        "outputs/figures/shap_bar.png",
        "outputs/figures/shap_dependence_top1.png",
        "outputs/figures/shap_dependence_top2.png",
        "outputs/figures/shap_local_explanation.png",
        # figures: eda
        "outputs/figures/eda_churn_distribution.png",
        "outputs/figures/eda_numeric_distribution.png",
        "outputs/figures/eda_contract_churn_rate.png",
        "outputs/figures/eda_payment_churn_rate.png",
        "outputs/figures/eda_internet_churn_rate.png",
        "outputs/figures/eda_correlation_heatmap.png",
        "outputs/figures/preprocess_train_test_churn_distribution.png",
        "outputs/figures/feature_engineering_business_features.png",
        # predictions + strategy
        "outputs/predictions/customer_churn_predictions.csv",
        "outputs/predictions/high_risk_customers.csv",
        "outputs/strategy/retention_strategy.md",
        "outputs/strategy/cost_benefit_strategy.md",
        "outputs/strategy/risk_threshold_rationale.md",
        # notebooks in scope
        "notebooks/01_data_understanding.ipynb",
        "notebooks/02_preprocessing_and_features.ipynb",
        "notebooks/03_model_comparison.ipynb",
        "notebooks/04_shap_and_strategy.ipynb",
        # docs
        "docs/data_dictionary.md",
        "docs/data_quality_report.md",
        "docs/leakage_risk_and_pipeline_improvement.md",
        # scripts
        "src/cost_sensitive_analysis.py",
        "src/final_submission_check.py",
    ]

    optional_files = [
        "outputs/metrics/cross_validation_results.csv",
        "outputs/figures/cross_validation_auc.png",
        "src/cross_validate_models.py",
    ]

    print("Checking project required outputs...")
    missing_required = check_file_list(project_root, required_files, "REQUIRED")
    missing_optional = check_file_list(project_root, optional_files, "OPTIONAL")

    total_required = len(required_files)
    ok_required = total_required - len(missing_required)

    print("\n" + "-" * 70)
    print(f"Required total: {total_required}")
    print(f"Required OK: {ok_required}")
    print(f"Required missing: {len(missing_required)}")
    print(f"Optional missing: {len(missing_optional)}")

    if missing_required:
        print("Overall result: FAILED (required files missing)")
    else:
        print("Overall result: PASSED (all required files exist)")


if __name__ == "__main__":
    main()
