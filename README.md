# 电信客户流失预测与精准挽留策略分析

## 1. 项目背景
本项目基于 Telco Customer Churn 数据集，围绕电信运营商客户流失问题构建二分类预测模型。项目在完成流失预测的同时，结合 SHAP 可解释性分析识别关键驱动因素，并根据风险分层结果输出差异化客户挽留策略，支持运营部门进行精准干预。

## 2. 项目目标
- 数据清洗与预处理
- 业务特征工程
- 多模型训练与对比
- 类别不平衡处理
- 模型评估
- SHAP 可解释性分析
- 客户风险分层
- 业务挽留策略输出

## 3. 数据说明
- 数据集名称：Telco Customer Churn
- 数据来源：Kaggle / IBM Sample Dataset
- 原始数据路径：`data/raw/Telco-Customer-Churn.csv`
- 样本量：7043
- 原始字段数：21
- 目标变量：`Churn`
- 正负样本分布：
  - 未流失客户：5174，占 73.46%
  - 流失客户：1869，占 26.54%

## 4. 项目目录结构
```text
TelecomCustomer_ChurnPrediction
├── README.md
├── requirements.txt
├── config.yaml
├── data
│   ├── raw
│   │   └── Telco-Customer-Churn.csv
│   └── processed
│       ├── churn_cleaned.csv
│       ├── train.csv
│       ├── test.csv
│       └── feature_columns.json
├── docs
│   ├── run_guide.md
│   ├── project_checklist.md
│   └── environment_versions.txt
├── models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── best_model.pkl
├── notebooks
├── outputs
│   ├── figures
│   ├── metrics
│   ├── predictions
│   └── strategy
└── src
    ├── preprocess.py
    ├── feature_engineering.py
    ├── train_models.py
    ├── evaluate_models.py
    ├── shap_analysis.py
    ├── check_outputs.py
    └── utils.py
```

目录用途说明：
- `data/raw`：原始数据存放目录
- `data/processed`：预处理与特征编码后的训练/测试数据
- `src`：项目核心代码（预处理、训练、评估、SHAP、输出检查）
- `models`：训练完成后的模型文件
- `outputs/metrics`：评估指标与统计结果
- `outputs/figures`：模型评估图与 SHAP 可视化图
- `outputs/predictions`：客户预测与高风险客户清单
- `outputs/strategy`：业务挽留策略建议
- `docs`：运行说明、验收清单、环境版本记录

## 5. 技术路线
数据读取  
→ 数据清洗  
→ 特征工程  
→ One-Hot 编码  
→ 分层训练集/测试集划分  
→ 多模型训练  
→ 模型评估  
→ 最佳模型选择  
→ SHAP 可解释性分析  
→ 风险分层  
→ 挽留策略输出

## 6. 模型对比结果
以下结果来自 `outputs/metrics/model_comparison.csv`：

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---:|---:|---:|---:|---:|---:|
| xgboost | 0.7566 | 0.5279 | 0.7834 | 0.6308 | 0.8430 | 0.6556 |
| logistic_regression | 0.7402 | 0.5069 | 0.7834 | 0.6155 | 0.8412 | 0.6306 |
| lightgbm | 0.7551 | 0.5287 | 0.7139 | 0.6075 | 0.8332 | 0.6398 |
| random_forest | 0.7913 | 0.6316 | 0.5134 | 0.5664 | 0.8306 | 0.6283 |

结论说明：
- 最佳模型为 `xgboost`
- `xgboost` 的 ROC-AUC 为 **0.8430**
- `xgboost` 的 PR-AUC 为 **0.6556**
- `xgboost` 在识别流失客户方面综合表现最好

## 7. SHAP 可解释性结果
根据 `outputs/metrics/shap_feature_importance.csv`，Top 10 特征如下：
- is_month_to_month
- tenure
- InternetService_Fiber optic
- Contract_Two year
- TotalCharges
- avg_monthly_charge
- PaymentMethod_Electronic check
- MonthlyCharges
- PaperlessBilling_Yes
- OnlineSecurity_Yes

业务含义解读：
- `is_month_to_month`：月付短约客户更易流失，契约稳定性不足。
- `tenure`：在网时长越短，客户关系越不稳定。
- `InternetService_Fiber optic`：网络服务类型与价格、体验预期相关，影响流失倾向。
- `Contract_Two year`：长期合约通常能降低流失概率。
- `TotalCharges`：总消费水平反映客户历史贡献和服务使用深度。
- `avg_monthly_charge` / `MonthlyCharges`：月消费越高，价格敏感客户更可能离网。
- `PaymentMethod_Electronic check`：支付方式与续费便利性、用户习惯相关。
- `PaperlessBilling_Yes`：电子账单客户在触达渠道和自助行为上有明显差异。
- `OnlineSecurity_Yes`：增值安全服务有助于提升客户黏性。

## 8. 风险分层结果
数据来自 `outputs/metrics/risk_segment_summary.csv`：

| risk_level | customer_count | ratio_pct |
|---|---:|---:|
| 低风险 | 768 | 54.51% |
| 中风险 | 315 | 22.36% |
| 高风险 | 326 | 23.14% |

已知高风险客户：
- 数量：326
- 占比：23.14%

## 9. 主要输出文件
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

预测与策略文件：
- `outputs/predictions/customer_churn_predictions.csv`
- `outputs/predictions/high_risk_customers.csv`
- `outputs/strategy/retention_strategy.md`

## 10. 运行方式
```bash
python -m pip install -r requirements.txt
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
python src/shap_analysis.py
```

## 11. 项目结论
- XGBoost 是最佳模型。
- 模型能够较好识别高风险流失客户。
- SHAP 解释结果显示合同类型、在网时长、互联网服务、消费金额、支付方式等因素对流失预测影响较大。
- 项目成果可辅助运营部门开展客户挽留、差异化运营与精准营销。
