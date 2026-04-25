# 项目运行说明

## 1. 环境准备
- 推荐 Python 3.10+（当前项目在 Python 3.13 环境下验证）
- 建议在虚拟环境中运行，避免依赖冲突
- 项目根目录：`E:\TelecomCustomer_ChurnPrediction`

## 2. 数据放置方式
- 将原始数据文件放置到：`data/raw/Telco-Customer-Churn.csv`
- 文件名需保持一致，否则预处理脚本会提示缺失

## 3. 依赖安装
```bash
python -m pip install -r requirements.txt
```

## 4. 脚本运行顺序
```bash
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
python src/shap_analysis.py
```

执行说明：
- `preprocess.py`：清洗数据、构造特征、编码并切分训练/测试集
- `train_models.py`：训练多模型并保存最佳模型
- `evaluate_models.py`：输出评估指标、曲线图、阈值分析、预测文件
- `shap_analysis.py`：生成 SHAP 可解释性结果、风险分层与策略建议

## 5. 输出文件说明
- 模型文件：`models/*.pkl`
- 指标文件：`outputs/metrics/*.csv` 与 `classification_reports.txt`
- 图表文件：`outputs/figures/*.png`
- 预测文件：`outputs/predictions/*.csv`
- 策略文件：`outputs/strategy/retention_strategy.md`

## 6. 常见问题说明
- `ModuleNotFoundError`：先执行 `python -m pip install -r requirements.txt`
- `shap` 安装失败：`src/shap_analysis.py` 内置回退方案，可在 xgboost/lightgbm 下继续产出核心文件
- 输出文件缺失：执行 `python src/check_outputs.py` 逐项核对
- 模型文件缺失：确认先完成 `python src/train_models.py` 和 `python src/evaluate_models.py`
