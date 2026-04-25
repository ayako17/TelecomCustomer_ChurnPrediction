# 项目验收清单

说明：以下为当前项目关键产物检查项。可配合 `python src/check_outputs.py` 自动核对。

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

## 预测与策略文件
- [OK] `outputs/predictions/customer_churn_predictions.csv`
- [OK] `outputs/predictions/high_risk_customers.csv`
- [OK] `outputs/strategy/retention_strategy.md`
