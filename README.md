# 电信客户流失预测与精准挽留策略分析

## 项目背景
本项目基于 **Telco Customer Churn** 数据集，目标是预测客户是否会流失（`Churn`），并结合可解释性分析输出可落地的挽留策略建议。

## 项目目标
- 完成数据分析与探索
- 完成数据预处理与特征工程
- 完成多模型训练与效果对比
- 完成模型评估
- 完成 SHAP 可解释性分析
- 输出业务策略建议

## 项目目录结构
```text
TelecomCustomer_ChurnPrediction
├── README.md
├── requirements.txt
├── config.yaml
├── data
│   ├── raw
│   ├── processed
│   └── data_source.md
├── notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_and_features.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_shap_and_strategy.ipynb
├── src
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── shap_analysis.py
│   └── utils.py
├── models
└── outputs
    ├── figures
    ├── metrics
    └── predictions
```

## 推荐运行流程
1. 安装依赖
2. 将 `Telco-Customer-Churn.csv` 放入 `data/raw/`
3. 运行数据预处理
4. 运行模型训练
5. 运行模型评估
6. 运行 SHAP 分析

## 说明
当前项目仅初始化目录结构与脚本框架，不包含真实数据集。请先自行下载并放置数据文件后再运行完整流程。
