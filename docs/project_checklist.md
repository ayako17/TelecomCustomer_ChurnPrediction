# 项目验收清单

说明：可结合以下命令自动核对关键产物。  
`python src/check_outputs.py`  
`python src/final_submission_check.py`

## 数据文件
- [OK] `data/raw/Telco-Customer-Churn.csv`
- [OK] `data/processed/churn_cleaned.csv`
- [OK] `data/processed/train.csv`
- [OK] `data/processed/test.csv`
- [OK] `data/processed/feature_columns.json`

## 模型文件
- [OK] `models/logistic_regression.pkl`
- [OK] `models/random_forest.pkl`
- [OK] `models/xgboost_model.pkl`
- [OK] `models/lightgbm_model.pkl`
- [OK] `models/best_model.pkl`

## 指标文件
- [OK] `outputs/metrics/model_comparison.csv`
- [OK] `outputs/metrics/classification_reports.txt`
- [OK] `outputs/metrics/threshold_analysis.csv`
- [OK] `outputs/metrics/shap_feature_importance.csv`
- [OK] `outputs/metrics/risk_segment_summary.csv`
- [OK] `outputs/metrics/cost_sensitive_analysis.csv`
- [OK] `outputs/metrics/final_submission_check_report.txt`

## 图表文件
- [OK] `outputs/figures/model_comparison.png`
- [OK] `outputs/figures/roc_curve.png`
- [OK] `outputs/figures/pr_curve.png`
- [OK] `outputs/figures/confusion_matrix_best_model.png`
- [OK] `outputs/figures/feature_importance.png`
- [OK] `outputs/figures/threshold_analysis.png`
- [OK] `outputs/figures/shap_summary.png`
- [OK] `outputs/figures/shap_bar.png`
- [OK] `outputs/figures/shap_dependence_top1.png`
- [OK] `outputs/figures/shap_dependence_top2.png`
- [OK] `outputs/figures/shap_local_explanation.png`
- [OK] `outputs/figures/cost_sensitive_analysis.png`

## 预测与策略文件
- [OK] `outputs/predictions/customer_churn_predictions.csv`
- [OK] `outputs/predictions/high_risk_customers.csv`
- [OK] `outputs/strategy/retention_strategy.md`
- [OK] `outputs/strategy/cost_benefit_strategy.md`
- [OK] `outputs/strategy/risk_threshold_rationale.md`

## 文档文件
- [OK] `docs/run_guide.md`
- [OK] `docs/project_checklist.md`
- [OK] `docs/environment_versions.txt`
- [OK] `docs/data_dictionary.md`
- [OK] `docs/data_quality_report.md`
- [OK] `docs/leakage_risk_and_pipeline_improvement.md`

## 检查脚本
- [OK] `src/check_outputs.py`
- [OK] `src/check_notebooks.py`
- [OK] `src/cost_sensitive_analysis.py`
- [OK] `src/final_submission_check.py`
