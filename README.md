# 电信客户流失预测与精准挽留策略分析

## 1. 项目背景
本项目基于 Telco Customer Churn 数据集，围绕电信运营商客户流失问题构建预测模型，识别高风险客户，并结合 SHAP 可解释性与成本敏感分析输出可执行的分层挽留策略。

## 2. 项目目标
- 数据清洗与预处理
- 业务特征工程
- 多模型训练与对比
- 类别不平衡处理
- 模型评估与阈值分析
- SHAP 可解释性分析
- 客户风险分层
- 成本收益驱动策略输出

## 3. 数据说明
- 数据集：Telco Customer Churn
- 来源：Kaggle / IBM Sample Dataset
- 原始路径：`data/raw/Telco-Customer-Churn.csv`
- 样本量：7043
- 原始字段数：21
- 目标变量：`Churn`
- 标签分布：未流失 5174（73.46%），流失 1869（26.54%）

## 4. 技术路线
数据读取  
→ 数据清洗  
→ 特征工程  
→ One-Hot 编码  
→ 分层切分  
→ 多模型训练  
→ 模型评估  
→ 最佳模型选择  
→ SHAP 分析  
→ 风险分层  
→ 成本敏感分析  
→ 挽留策略输出

## 5. 模型对比结果（来自 `outputs/metrics/model_comparison.csv`）

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.7566 | 0.5279 | 0.7834 | 0.6308 | 0.8430 | 0.6556 |
| logistic_regression | 0.7402 | 0.5069 | 0.7834 | 0.6155 | 0.8412 | 0.6306 |
| lightgbm | 0.7551 | 0.5287 | 0.7139 | 0.6075 | 0.8332 | 0.6398 |
| random_forest | 0.7913 | 0.6316 | 0.5134 | 0.5664 | 0.8306 | 0.6283 |

说明：最佳模型为 `xgboost`，在 `ROC-AUC=0.8430`、`PR-AUC=0.6556` 上综合表现最优。

## 6. SHAP 与业务洞察
Top 10 关键特征包括：`is_month_to_month`、`tenure`、`InternetService_Fiber optic`、`Contract_Two year`、`TotalCharges`、`avg_monthly_charge`、`PaymentMethod_Electronic check`、`MonthlyCharges`、`PaperlessBilling_Yes`、`OnlineSecurity_Yes`。  
核心结论：合同类型、在网时长、消费金额与支付方式是流失预测的重要驱动因素。

## 7. 风险分层结果
基于预测概率的运营分层：
- 高风险：`churn_probability >= 0.7`
- 中风险：`0.4 <= churn_probability < 0.7`
- 低风险：`churn_probability < 0.4`

当前高风险客户：326 人（23.14%）。

## 8. 主要输出文件

模型文件：
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost_model.pkl`
- `models/lightgbm_model.pkl`
- `models/best_model.pkl`

指标文件：
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/classification_reports.txt`
- `outputs/metrics/threshold_analysis.csv`
- `outputs/metrics/shap_feature_importance.csv`
- `outputs/metrics/risk_segment_summary.csv`
- `outputs/metrics/cost_sensitive_analysis.csv`
- `outputs/metrics/final_submission_check_report.txt`

图表文件：
- `outputs/figures/model_comparison.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/pr_curve.png`
- `outputs/figures/confusion_matrix_best_model.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/threshold_analysis.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_bar.png`
- `outputs/figures/shap_dependence_top1.png`
- `outputs/figures/shap_dependence_top2.png`
- `outputs/figures/shap_local_explanation.png`
- `outputs/figures/cost_sensitive_analysis.png`

策略文件：
- `outputs/strategy/retention_strategy.md`
- `outputs/strategy/cost_benefit_strategy.md`
- `outputs/strategy/risk_threshold_rationale.md`

文档文件：
- `docs/run_guide.md`
- `docs/project_checklist.md`
- `docs/environment_versions.txt`
- `docs/data_dictionary.md`
- `docs/data_quality_report.md`
- `docs/leakage_risk_and_pipeline_improvement.md`

脚本文件：
- `src/cost_sensitive_analysis.py`
- `src/final_submission_check.py`
- `src/check_outputs.py`
- `src/check_notebooks.py`

## 9. 运行方式
```bash
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
python src/shap_analysis.py
python src/cost_sensitive_analysis.py
python src/final_submission_check.py
python src/check_outputs.py
python src/check_notebooks.py
```

## 10. 项目结论
- XGBoost 是当前最优模型。
- 模型可以有效识别高风险流失客户。
- SHAP 结果证明合同、时长、消费与支付因素对流失影响显著。
- 成本敏感分析支持阈值和预算协同决策，提升运营资源使用效率。
