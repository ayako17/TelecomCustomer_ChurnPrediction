# -*- coding: utf-8 -*-
from pathlib import Path

import nbformat as nbf


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )
    nbf.write(nb, path)


def build_01() -> list:
    return [
        nbf.v4.new_markdown_cell(
            "# 数据理解与探索性分析\n\n本 Notebook 展示客户流失数据的基础探索分析（EDA），重点关注流失分布、关键类别特征流失率和数值特征关系。"
        ),
        nbf.v4.new_markdown_cell("## 1. 数据读取与环境准备"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n\n"
            "plt.rcParams['figure.dpi'] = 120\n\n"
            "def find_project_root():\n"
            "    cwd = Path.cwd().resolve()\n"
            "    for p in [cwd, *cwd.parents]:\n"
            "        if (p / 'config.yaml').exists() and (p / 'data').exists():\n"
            "            return p\n"
            "    return cwd\n\n"
            "project_root = find_project_root()\n"
            "raw_path = project_root / 'data' / 'raw' / 'Telco-Customer-Churn.csv'\n"
            "cleaned_path = project_root / 'data' / 'processed' / 'churn_cleaned.csv'\n"
            "fig_dir = project_root / 'outputs' / 'figures'\n"
            "fig_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "raw_df = pd.read_csv(raw_path)\n"
            "cleaned_df = pd.read_csv(cleaned_path)\n"
            "raw_df['TotalCharges_num'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')\n"
            "raw_df['Churn_flag'] = raw_df['Churn'].map({'Yes': 1, 'No': 0})\n\n"
            "print('项目根目录:', project_root)\n"
            "print('原始数据维度:', raw_df.shape)\n"
            "print('清洗后数据维度:', cleaned_df.shape)"
        ),
        nbf.v4.new_markdown_cell("## 2. 原始数据维度、字段类型与缺失值统计"),
        nbf.v4.new_code_cell(
            "dtype_df = raw_df.dtypes.reset_index()\n"
            "dtype_df.columns = ['字段名', '数据类型']\n"
            "missing_df = raw_df.isna().sum().reset_index()\n"
            "missing_df.columns = ['字段名', '缺失数量']\n"
            "missing_df['缺失占比'] = (missing_df['缺失数量'] / len(raw_df)).round(4)\n\n"
            "display(raw_df.head())\n"
            "display(dtype_df)\n"
            "display(missing_df.sort_values('缺失数量', ascending=False))"
        ),
        nbf.v4.new_markdown_cell("## 3. Churn 目标变量分布"),
        nbf.v4.new_code_cell(
            "churn_counts = raw_df['Churn'].value_counts().reindex(['No', 'Yes'])\n"
            "plt.figure(figsize=(6, 4))\n"
            "plt.bar(churn_counts.index, churn_counts.values)\n"
            "plt.title('Churn Distribution')\n"
            "plt.xlabel('Churn')\n"
            "plt.ylabel('Count')\n"
            "for i, v in enumerate(churn_counts.values):\n"
            "    plt.text(i, v, str(v), ha='center', va='bottom')\n"
            "plt.tight_layout()\n"
            "out_path = fig_dir / 'eda_churn_distribution.png'\n"
            "plt.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)"
        ),
        nbf.v4.new_markdown_cell("## 4. tenure、MonthlyCharges、TotalCharges 分布"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "axes[0].hist(raw_df['tenure'].dropna(), bins=30)\n"
            "axes[0].set_title('tenure Distribution')\n"
            "axes[0].set_xlabel('tenure')\n"
            "axes[0].set_ylabel('Count')\n\n"
            "axes[1].hist(raw_df['MonthlyCharges'].dropna(), bins=30)\n"
            "axes[1].set_title('MonthlyCharges Distribution')\n"
            "axes[1].set_xlabel('MonthlyCharges')\n\n"
            "axes[2].hist(raw_df['TotalCharges_num'].dropna(), bins=30)\n"
            "axes[2].set_title('TotalCharges Distribution')\n"
            "axes[2].set_xlabel('TotalCharges')\n\n"
            "fig.tight_layout()\n"
            "out_path = fig_dir / 'eda_numeric_distribution.png'\n"
            "fig.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)"
        ),
        nbf.v4.new_markdown_cell("## 5. Contract 与 Churn 的交叉流失率分析"),
        nbf.v4.new_code_cell(
            "contract_rate = raw_df.groupby('Contract', dropna=False)['Churn_flag'].mean().sort_values(ascending=False)\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.bar(contract_rate.index, contract_rate.values)\n"
            "plt.title('Churn Rate by Contract')\n"
            "plt.xlabel('Contract')\n"
            "plt.ylabel('Churn Rate')\n"
            "plt.ylim(0, 1)\n"
            "plt.xticks(rotation=15)\n"
            "for x, y in zip(contract_rate.index, contract_rate.values):\n"
            "    plt.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout()\n"
            "out_path = fig_dir / 'eda_contract_churn_rate.png'\n"
            "plt.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)\n"
            "display(contract_rate.rename('流失率').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## 6. PaymentMethod 与 Churn 的交叉流失率分析"),
        nbf.v4.new_code_cell(
            "payment_rate = raw_df.groupby('PaymentMethod', dropna=False)['Churn_flag'].mean().sort_values(ascending=False)\n"
            "plt.figure(figsize=(10, 4))\n"
            "plt.bar(payment_rate.index, payment_rate.values)\n"
            "plt.title('Churn Rate by PaymentMethod')\n"
            "plt.xlabel('PaymentMethod')\n"
            "plt.ylabel('Churn Rate')\n"
            "plt.ylim(0, 1)\n"
            "plt.xticks(rotation=20, ha='right')\n"
            "for x, y in zip(payment_rate.index, payment_rate.values):\n"
            "    plt.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout()\n"
            "out_path = fig_dir / 'eda_payment_churn_rate.png'\n"
            "plt.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)\n"
            "display(payment_rate.rename('流失率').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## 7. InternetService 与 Churn 的交叉流失率分析"),
        nbf.v4.new_code_cell(
            "internet_rate = raw_df.groupby('InternetService', dropna=False)['Churn_flag'].mean().sort_values(ascending=False)\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.bar(internet_rate.index, internet_rate.values)\n"
            "plt.title('Churn Rate by InternetService')\n"
            "plt.xlabel('InternetService')\n"
            "plt.ylabel('Churn Rate')\n"
            "plt.ylim(0, 1)\n"
            "for x, y in zip(internet_rate.index, internet_rate.values):\n"
            "    plt.text(x, y, f'{y:.2%}', ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout()\n"
            "out_path = fig_dir / 'eda_internet_churn_rate.png'\n"
            "plt.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)\n"
            "display(internet_rate.rename('流失率').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## 8. 数值特征相关性热图"),
        nbf.v4.new_code_cell(
            "numeric_cols = cleaned_df.select_dtypes(include='number').columns.tolist()\n"
            "corr = cleaned_df[numeric_cols].corr(numeric_only=True)\n\n"
            "plt.figure(figsize=(12, 10))\n"
            "im = plt.imshow(corr, interpolation='nearest', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)\n"
            "plt.colorbar(im, fraction=0.046, pad=0.04)\n"
            "plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)\n"
            "plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)\n"
            "plt.title('Correlation Heatmap (Numeric Features)')\n"
            "plt.tight_layout()\n"
            "out_path = fig_dir / 'eda_correlation_heatmap.png'\n"
            "plt.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)"
        ),
        nbf.v4.new_markdown_cell(
            "## 9. EDA 小结\n"
            "- 数据存在类别不平衡，未流失样本明显多于流失样本。\n"
            "- 合同类型和流失风险关系明显，月付客户流失率更高。\n"
            "- 在网时长（tenure）越短，流失风险越高。\n"
            "- 月费用与总费用共同反映客户的价格敏感和价值感知。\n"
            "- 支付方式与互联网服务类型在流失率上也呈现清晰差异。"
        ),
    ]


def build_02() -> list:
    return [
        nbf.v4.new_markdown_cell(
            "# 数据预处理与特征工程\n\n本 Notebook 用于展示数据清洗、标签转换、特征构造、One-Hot 编码、训练集测试集划分等过程。"
        ),
        nbf.v4.new_markdown_cell("## 1. 读取必要文件"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import display\n\n"
            "plt.rcParams['figure.dpi'] = 120\n\n"
            "def find_project_root():\n"
            "    cwd = Path.cwd().resolve()\n"
            "    for p in [cwd, *cwd.parents]:\n"
            "        if (p / 'config.yaml').exists() and (p / 'data').exists():\n"
            "            return p\n"
            "    return cwd\n\n"
            "project_root = find_project_root()\n"
            "raw_path = project_root / 'data' / 'raw' / 'Telco-Customer-Churn.csv'\n"
            "cleaned_path = project_root / 'data' / 'processed' / 'churn_cleaned.csv'\n"
            "train_path = project_root / 'data' / 'processed' / 'train.csv'\n"
            "test_path = project_root / 'data' / 'processed' / 'test.csv'\n"
            "feature_cols_path = project_root / 'data' / 'processed' / 'feature_columns.json'\n"
            "fig_dir = project_root / 'outputs' / 'figures'\n"
            "fig_dir.mkdir(parents=True, exist_ok=True)\n\n"
            "raw_df = pd.read_csv(raw_path)\n"
            "cleaned_df = pd.read_csv(cleaned_path)\n"
            "train_df = pd.read_csv(train_path)\n"
            "test_df = pd.read_csv(test_path)\n"
            "feature_columns = json.loads(feature_cols_path.read_text(encoding='utf-8'))\n"
            "print('项目根目录:', project_root)"
        ),
        nbf.v4.new_markdown_cell("## 2. 维度与特征数量"),
        nbf.v4.new_code_cell(
            "print('原始数据维度:', raw_df.shape)\n"
            "print('清洗后数据维度:', cleaned_df.shape)\n"
            "print('训练集维度:', train_df.shape)\n"
            "print('测试集维度:', test_df.shape)\n"
            "print('特征数量:', len(feature_columns))"
        ),
        nbf.v4.new_markdown_cell("## 3. TotalCharges 处理逻辑验证"),
        nbf.v4.new_code_cell(
            "print('原始 TotalCharges 的类型:', raw_df['TotalCharges'].dtype)\n"
            "total_num = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')\n"
            "print('转换为数值后的缺失数量:', int(total_num.isna().sum()))\n"
            "print('处理说明: 使用 TotalCharges 中位数填补缺失值。')"
        ),
        nbf.v4.new_markdown_cell("## 4. Churn 标签转换"),
        nbf.v4.new_code_cell(
            "print('标签映射: Yes -> 1, No -> 0')\n"
            "display(cleaned_df['Churn'].value_counts().sort_index().rename('样本数').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## 5. 业务特征工程表"),
        nbf.v4.new_code_cell(
            "feature_logic_df = pd.DataFrame([\n"
            "    {'特征名': 'service_count', '构造逻辑': '统计多个服务字段中 Yes 的数量', '业务含义': '衡量客户服务绑定程度'},\n"
            "    {'特征名': 'avg_monthly_charge', '构造逻辑': 'TotalCharges / tenure，tenure 为 0 时用 MonthlyCharges', '业务含义': '衡量客户平均消费压力'},\n"
            "    {'特征名': 'is_month_to_month', '构造逻辑': 'Contract 是否为 Month-to-month', '业务含义': '衡量合约约束强弱'},\n"
            "    {'特征名': 'has_internet_service', '构造逻辑': 'InternetService 是否不为 No', '业务含义': '衡量是否使用互联网服务'},\n"
            "    {'特征名': 'is_auto_payment', '构造逻辑': 'PaymentMethod 是否包含 automatic', '业务含义': '衡量自动续费和支付便利性'},\n"
            "    {'特征名': 'high_charge_low_tenure', '构造逻辑': 'MonthlyCharges 高于中位数且 tenure 低于中位数', '业务含义': '识别高消费但低黏性的潜在风险客户'},\n"
            "])\n"
            "display(feature_logic_df)"
        ),
        nbf.v4.new_markdown_cell("## 6. 构造特征描述性统计"),
        nbf.v4.new_code_cell(
            "feature_cols = ['service_count', 'avg_monthly_charge', 'is_month_to_month', 'has_internet_service', 'is_auto_payment', 'high_charge_low_tenure']\n"
            "display(cleaned_df[feature_cols].describe(include='all'))\n"
            "for col in ['service_count', 'is_month_to_month', 'has_internet_service', 'is_auto_payment', 'high_charge_low_tenure']:\n"
            "    print(f'\\n{col} 分布:')\n"
            "    display(cleaned_df[col].value_counts(dropna=False).rename('count').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## 7. One-Hot 编码后特征列数量"),
        nbf.v4.new_code_cell(
            "print('特征列数量:', len(feature_columns))\n"
            "print('前 15 个特征名:')\n"
            "for i, col in enumerate(feature_columns[:15], start=1):\n"
            "    print(f'{i:02d}. {col}')"
        ),
        nbf.v4.new_markdown_cell("## 8. 训练集与测试集标签分布"),
        nbf.v4.new_code_cell(
            "train_rate = train_df['Churn'].value_counts(normalize=True).sort_index().rename('train_ratio')\n"
            "test_rate = test_df['Churn'].value_counts(normalize=True).sort_index().rename('test_ratio')\n"
            "dist_df = pd.concat([train_rate, test_rate], axis=1).fillna(0)\n"
            "dist_df.index.name = 'Churn'\n"
            "display(dist_df)\n\n"
            "x = [0, 1]\n"
            "w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar([i - w/2 for i in x], [float(train_rate.get(i, 0)) for i in x], width=w, label='Train')\n"
            "ax.bar([i + w/2 for i in x], [float(test_rate.get(i, 0)) for i in x], width=w, label='Test')\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels(['Churn=0', 'Churn=1'])\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_ylabel('Ratio')\n"
            "ax.set_title('Train/Test Churn Distribution')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "out_path = fig_dir / 'preprocess_train_test_churn_distribution.png'\n"
            "fig.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)"
        ),
        nbf.v4.new_markdown_cell("## 9. 业务构造特征分布图"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
            "items = [\n"
            "    ('service_count', 'service_count'),\n"
            "    ('is_month_to_month', 'is_month_to_month'),\n"
            "    ('has_internet_service', 'has_internet_service'),\n"
            "    ('is_auto_payment', 'is_auto_payment'),\n"
            "    ('high_charge_low_tenure', 'high_charge_low_tenure'),\n"
            "    ('avg_monthly_charge', 'avg_monthly_charge'),\n"
            "]\n"
            "for ax, (col, title) in zip(axes.flatten(), items):\n"
            "    if cleaned_df[col].nunique() <= 10:\n"
            "        vc = cleaned_df[col].value_counts().sort_index()\n"
            "        ax.bar(vc.index.astype(str), vc.values)\n"
            "        ax.set_ylabel('Count')\n"
            "    else:\n"
            "        ax.hist(cleaned_df[col].dropna(), bins=30)\n"
            "        ax.set_ylabel('Count')\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel(col)\n"
            "fig.tight_layout()\n"
            "out_path = fig_dir / 'feature_engineering_business_features.png'\n"
            "fig.savefig(out_path, dpi=150)\n"
            "plt.show()\n"
            "print('已保存图表:', out_path)"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. 结论\n"
            "- 数据清洗解决了 TotalCharges 类型和缺失值问题。\n"
            "- 标签编码将流失预测转化为二分类任务。\n"
            "- 业务特征工程增强了模型对客户黏性、消费压力、合约约束和服务绑定程度的刻画。\n"
            "- 分层划分保持了训练集与测试集类别比例一致，为后续模型评估提供可靠基础。"
        ),
    ]


def build_03() -> list:
    return [
        nbf.v4.new_markdown_cell(
            "# 多模型训练与对比\n\n本 Notebook 仅读取已有评估文件与图表，不重新训练模型。"
        ),
        nbf.v4.new_markdown_cell("## 1. 读取结果文件"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "def find_project_root():\n"
            "    cwd = Path.cwd().resolve()\n"
            "    for p in [cwd, *cwd.parents]:\n"
            "        if (p / 'config.yaml').exists() and (p / 'outputs').exists():\n"
            "            return p\n"
            "    return cwd\n\n"
            "project_root = find_project_root()\n"
            "metrics_dir = project_root / 'outputs' / 'metrics'\n"
            "fig_dir = project_root / 'outputs' / 'figures'\n\n"
            "model_df = pd.read_csv(metrics_dir / 'model_comparison.csv')\n"
            "threshold_df = pd.read_csv(metrics_dir / 'threshold_analysis.csv')\n"
            "report_text = (metrics_dir / 'classification_reports.txt').read_text(encoding='utf-8')\n\n"
            "print('项目根目录:', project_root)\n"
            "print('模型结果行数:', len(model_df))"
        ),
        nbf.v4.new_markdown_cell("## 2. 多模型结果表"),
        nbf.v4.new_code_cell(
            "display(model_df)\n"
            "best = model_df.sort_values('roc_auc', ascending=False).iloc[0]\n"
            "print('按 ROC-AUC 选择的最佳模型:', best['model'])\n"
            "print('ROC-AUC:', round(float(best['roc_auc']), 4))\n"
            "print('PR-AUC:', round(float(best['pr_auc']), 4))"
        ),
        nbf.v4.new_markdown_cell("## 3. 模型指标对比图"),
        nbf.v4.new_code_cell("display(Image(filename=str(fig_dir / 'model_comparison.png')))"),
        nbf.v4.new_markdown_cell("## 4. ROC 曲线图"),
        nbf.v4.new_code_cell("display(Image(filename=str(fig_dir / 'roc_curve.png')))"),
        nbf.v4.new_markdown_cell("## 5. PR 曲线图"),
        nbf.v4.new_code_cell("display(Image(filename=str(fig_dir / 'pr_curve.png')))"),
        nbf.v4.new_markdown_cell("## 6. 最佳模型混淆矩阵"),
        nbf.v4.new_code_cell("display(Image(filename=str(fig_dir / 'confusion_matrix_best_model.png')))"),
        nbf.v4.new_markdown_cell("## 7. 阈值分析结果"),
        nbf.v4.new_code_cell(
            "display(threshold_df)\n"
            "display(Image(filename=str(fig_dir / 'threshold_analysis.png')))"
        ),
        nbf.v4.new_markdown_cell("## 8. 分类报告文本"),
        nbf.v4.new_code_cell("print(report_text[:4000])"),
        nbf.v4.new_markdown_cell(
            "## 9. 为什么选择 XGBoost\n"
            "- 类别不平衡任务中，Accuracy 不能单独作为模型优劣判断标准。\n"
            "- 流失场景更关注正类识别能力，因此要综合看 recall、roc_auc、pr_auc。\n"
            "- XGBoost 在 roc_auc 与 pr_auc 上综合表现最佳，更适合高风险客户筛选。\n"
            "- Random Forest 虽然 Accuracy 较高，但在流失客户识别的综合能力上不如 XGBoost。"
        ),
    ]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    nb_dir = project_root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    write_notebook(nb_dir / "01_data_understanding.ipynb", build_01())
    write_notebook(nb_dir / "02_preprocessing_and_features.ipynb", build_02())
    write_notebook(nb_dir / "03_model_comparison.ipynb", build_03())

    print("已重建 Notebook：")
    print(nb_dir / "01_data_understanding.ipynb")
    print(nb_dir / "02_preprocessing_and_features.ipynb")
    print(nb_dir / "03_model_comparison.ipynb")


if __name__ == "__main__":
    main()
