from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    required_files = [
        "data/raw/Telco-Customer-Churn.csv",
        "data/processed/churn_cleaned.csv",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "data/processed/feature_columns.json",
        "models/logistic_regression.pkl",
        "models/random_forest.pkl",
        "models/xgboost_model.pkl",
        "models/lightgbm_model.pkl",
        "models/best_model.pkl",
        "outputs/metrics/model_comparison.csv",
        "outputs/metrics/classification_reports.txt",
        "outputs/metrics/threshold_analysis.csv",
        "outputs/metrics/shap_feature_importance.csv",
        "outputs/metrics/risk_segment_summary.csv",
        "outputs/figures/model_comparison.png",
        "outputs/figures/roc_curve.png",
        "outputs/figures/pr_curve.png",
        "outputs/figures/confusion_matrix_best_model.png",
        "outputs/figures/feature_importance.png",
        "outputs/figures/threshold_analysis.png",
        "outputs/figures/shap_summary.png",
        "outputs/figures/shap_bar.png",
        "outputs/figures/shap_dependence_top1.png",
        "outputs/figures/shap_dependence_top2.png",
        "outputs/figures/shap_local_explanation.png",
        "outputs/predictions/customer_churn_predictions.csv",
        "outputs/predictions/high_risk_customers.csv",
        "outputs/strategy/retention_strategy.md",
    ]

    missing_files: list[str] = []
    print("开始检查项目关键输出文件...")

    for rel_path in required_files:
        abs_path = project_root / rel_path
        if abs_path.exists():
            print(f"[OK] {rel_path}")
        else:
            print(f"[MISSING] {rel_path}")
            missing_files.append(rel_path)

    total = len(required_files)
    missing_count = len(missing_files)
    ok_count = total - missing_count

    print("-" * 60)
    print(f"检查总数: {total}")
    print(f"存在文件: {ok_count}")
    print(f"缺失文件: {missing_count}")

    if missing_files:
        print("总检查结果: 未通过（存在缺失文件）")
    else:
        print("总检查结果: 通过（所有关键文件均存在）")


if __name__ == "__main__":
    main()
