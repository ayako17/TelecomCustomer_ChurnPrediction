# 项目运行说明

## 1. 环境准备
- 建议 Python 3.10+
- 建议使用虚拟环境运行
- 在项目根目录执行所有命令

## 2. 数据放置
- 将原始数据放入：`data/raw/Telco-Customer-Churn.csv`
- 文件名请保持一致

## 3. 安装依赖
```bash
python -m pip install -r requirements.txt
```

## 4. 推荐执行顺序
```bash
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
python src/shap_analysis.py
python src/cost_sensitive_analysis.py
python src/final_submission_check.py
python src/check_outputs.py
python src/check_notebooks.py
```

## 5. 输出文件说明
- 模型文件：`models/*.pkl`
- 指标文件：`outputs/metrics/*.csv` 与 `classification_reports.txt`
- 图表文件：`outputs/figures/*.png`
- 预测文件：`outputs/predictions/*.csv`
- 策略文件：`outputs/strategy/*.md`
- 质量文档：`docs/*.md` 与 `docs/environment_versions.txt`

## 6. 常见问题
- `ModuleNotFoundError`：先执行依赖安装命令
- Notebook 权限问题（Windows）：可设置 `JUPYTER_ALLOW_INSECURE_WRITES=true`
- 输出缺失：运行 `python src/check_outputs.py` 定位缺失项
- 提交前总检：运行 `python src/final_submission_check.py`
