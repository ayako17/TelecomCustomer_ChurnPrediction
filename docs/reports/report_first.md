# 武汉大学计算机学院

# 本科生课程设计报告

<br>
<br>

# 基于机器学习与 SHAP 可解释性的电信客户流失预测及精准挽留策略研究

<br>
<br>

| 项目 | 内容 |
|---|---|
| 专业名称 | 软件工程 |
| 课程名称 | 商务智能 |
| 指导教师 | 朱卫平 |
| 学生学号 | 【请填写】 |
| 学生姓名 | 【请填写】 |
| 完成时间 | 二〇二六年五月 |

---

# 郑重声明

本人呈交的设计报告，是在指导老师的指导下，独立进行实验工作所取得的成果，所有数据、图片资料真实可靠。尽我所知，除文中已经注明引用的内容外，本设计报告不包含他人享有著作权的内容。对本设计报告做出贡献的其他个人和集体，均已在文中以明确的方式标明。本设计报告的知识产权归属于培养单位。

<br>

本人签名：__________

日期：__________

---

# 目录

1 背景介绍  
&emsp;1.1 研究背景  
&emsp;1.2 选题意义  
&emsp;1.3 本文主要工作  

2 需求分析与商务智能定位  
&emsp;2.1 商务智能分析类型定位  
&emsp;2.2 目标用户与业务痛点  
&emsp;2.3 功能需求与非功能需求  
&emsp;2.4 项目创新点  

3 数据来源与数据理解  
&emsp;3.1 数据集来源  
&emsp;3.2 数据字段说明  
&emsp;3.3 数据质量审计  
&emsp;3.4 目标变量分布  
&emsp;3.5 关键变量与流失关系分析  

4 数据预处理与特征工程  
&emsp;4.1 数据清洗  
&emsp;4.2 标签编码与类别变量处理  
&emsp;4.3 训练集与测试集划分  
&emsp;4.4 业务语义特征工程  
&emsp;4.5 数据处理严谨性说明  

5 解决方案与模型设计  
&emsp;5.1 总体解决方案  
&emsp;5.2 技术平台与开发环境  
&emsp;5.3 模型选择与类别不平衡处理  
&emsp;5.4 评估指标体系  

6 模型训练、评估与结果分析  
&emsp;6.1 实验设置  
&emsp;6.2 多模型结果对比  
&emsp;6.3 最优模型选择  
&emsp;6.4 ROC 曲线与 PR 曲线分析  
&emsp;6.5 混淆矩阵与阈值分析  
&emsp;6.6 交叉验证稳定性分析  

7 SHAP 可解释性分析  
&emsp;7.1 SHAP 方法简介  
&emsp;7.2 全局特征重要性分析  
&emsp;7.3 特征影响方向分析  
&emsp;7.4 局部客户解释  
&emsp;7.5 可解释性结果的业务启示  

8 风险分层、成本敏感分析与挽留策略  
&emsp;8.1 客户风险分层方法  
&emsp;8.2 风险分层结果  
&emsp;8.3 成本敏感分析与商业阈值选择  
&emsp;8.4 SHAP 驱动的差异化挽留策略  
&emsp;8.5 策略执行流程与推广价值  

9 关键代码实现与系统复现说明  
&emsp;9.1 项目目录结构  
&emsp;9.2 核心代码模块  
&emsp;9.3 关键代码片段  
&emsp;9.4 运行流程与输出文件  
&emsp;9.5 最终提交检查  

10 社会价值、总结与讨论  
&emsp;10.1 项目主要结论  
&emsp;10.2 商业价值与社会价值  
&emsp;10.3 推广前景  
&emsp;10.4 局限性  
&emsp;10.5 后续改进方向  

参考文献  
附录  

---

# 摘要

客户流失预测是商务智能在客户关系管理领域中的典型预测性分析任务。电信运营商面对客户获取成本高、客户留存压力大、营销资源有限等问题，需要通过数据驱动方法提前识别高流失风险客户，并制定差异化挽留策略。本文基于公开的 Telco Customer Churn 数据集，围绕电信客户流失问题构建预测性商务智能分析方案，完成数据理解、数据清洗、业务语义特征工程、多模型训练与对比、SHAP 可解释性分析、客户风险分层和成本敏感分析。

实验中，本文对 Logistic Regression、Random Forest、XGBoost 和 LightGBM 四类模型进行比较，综合 Accuracy、Precision、Recall、F1、ROC-AUC 和 PR-AUC 等指标选择最终模型。结果表明，XGBoost 在 ROC-AUC、PR-AUC、Recall 和 F1 等指标上综合表现较优，适合作为本项目的客户流失预测模型。进一步地，本文使用 SHAP 方法解释模型预测结果，识别出合同类型、在网时长、互联网服务类型、费用水平和支付方式等关键影响因素，并将模型解释结果转化为高、中、低风险客户分层和针对性挽留策略。最后，本文引入成本敏感分析，在设定漏判损失、误触达成本和预期挽留收益的基础上比较不同阈值的商业效果，得到推荐业务触达阈值 0.3，从而实现从预测建模到运营决策支持的完整闭环。

**关键词：** 商务智能；客户流失预测；XGBoost；SHAP；成本敏感分析；风险分层

---

# 1 背景介绍

## 1.1 研究背景

随着通信市场逐渐进入存量竞争阶段，电信运营商面临的核心问题已经从单纯获取新客户转向客户留存和客户价值提升。客户流失会直接影响企业收入、客户生命周期价值和市场份额。传统客户管理方式主要依赖事后统计和人工经验，通常只能回答“已经流失了多少客户”，难以及时识别“哪些客户即将流失”。
商务智能技术能够利用企业历史数据发现客户行为规律，并将数据分析结果转化为可执行的业务决策。对于客户流失问题，预测性分析能够根据客户基本信息、服务订购情况、合同类型、费用水平和支付方式等特征，提前预测客户是否存在流失风险。企业可基于预测结果优先触达高风险客户，以降低盲目营销成本，提高挽留效率。

## 1.2 选题意义

本项目选择电信客户流失预测作为商务智能课程设计主题，具有较强的现实应用意义。对企业而言，提前识别高风险客户可以帮助运营部门优化营销资源分配，减少无效触达，提高客户挽留成功率。对客户而言，企业能够基于客户特征提供更匹配的套餐和服务，降低不必要的打扰。对课程实践而言，本项目能够完整体现商务智能中的数据准备、预测建模、模型解释和决策支持过程。
本项目不是简单训练一个分类模型，而是围绕“预测—解释—分层—策略”构建完整分析链路。预测模型用于识别高风险客户，SHAP 用于解释客户流失原因，风险分层用于划分运营资源投入强度，成本敏感分析用于比较不同阈值下的商业收益，从而形成面向电信企业客户运营的商务智能解决方案。

## 1.3 本文主要工作

本文主要完成以下工作：

1. 基于 Telco Customer Churn 数据集构建客户流失预测任务，将 `Churn` 作为目标变量。
2. 完成原始数据读取、缺失值处理、标签编码、One-Hot 编码和分层训练集/测试集划分。
3. 从客户黏性、消费压力、合约约束和支付便利性角度构造业务语义特征。
4. 对 Logistic Regression、Random Forest、XGBoost 和 LightGBM 四类模型进行训练和评估。
5. 使用 ROC-AUC、PR-AUC、Recall、F1 等指标选择最终模型，并通过交叉验证增强结果稳定性说明。
6. 使用 SHAP 方法对模型进行全局和局部解释，识别影响客户流失的重要因素。
7. 将预测概率转化为客户风险分层，并结合成本敏感分析提出差异化客户挽留策略。
8. 构建完整项目工程结构，提供运行说明、依赖说明、数据字典、数据质量报告和最终提交检查脚本。

**图 1-1 电信客户流失预测商务智能分析流程**  
【建议插入自绘流程图：数据采集 → 数据质量审计 → 数据清洗 → 特征工程 → 多模型训练 → 模型评估 → SHAP解释 → 风险分层 → 成本敏感分析 → 挽留策略】

---

# 2 需求分析与商务智能定位

## 2.1 商务智能分析类型定位

商务智能分析通常可以分为描述性分析、预测性分析和规范性分析。描述性分析关注历史数据中已经发生的现象，预测性分析关注未来可能发生的结果，规范性分析进一步关注在预测结果基础上应采取何种行动。

本项目以**预测性分析**为核心，目标是根据客户历史属性和服务使用情况预测客户是否会流失。与此同时，本文在预测结果基础上引入风险分层、SHAP 可解释性分析和成本敏感分析，用于辅助运营部门制定客户挽留策略。因此，本项目的主要分析类型是预测性分析，策略输出部分属于基于预测结果的辅助性规范分析延伸。

## 2.2 目标用户与业务痛点

本项目的目标用户主要包括电信企业的数据运营部门、客户成功部门、营销部门和客户服务部门。

| 用户角色 | 核心痛点 | 项目响应方式 |
|---|---|---|
| 数据运营部门 | 缺少可量化的客户流失风险识别工具 | 输出客户流失概率和风险等级 |
| 营销部门 | 盲目发放优惠导致营销成本高 | 基于风险分层进行精准触达 |
| 客服部门 | 不清楚客户流失原因 | 使用 SHAP 解释客户风险来源 |
| 管理层 | 缺少投入产出评估依据 | 使用成本敏感分析比较不同阈值收益 |

传统客户挽留方式容易采用统一优惠或人工经验判断，缺少对客户风险差异的刻画。本项目通过机器学习模型预测客户流失风险，并结合特征解释和成本收益分析，将客户划分为不同运营优先级，从而提高运营资源利用效率。

## 2.3 功能需求与非功能需求

| 需求类型 | 具体内容 | 项目实现 |
|---|---|---|
| 数据处理需求 | 读取原始数据、清洗缺失值、编码变量 | `preprocess.py`、`feature_engineering.py` |
| 预测需求 | 预测客户是否可能流失 | `train_models.py`、`evaluate_models.py` |
| 解释需求 | 解释客户流失的关键影响因素 | `shap_analysis.py` |
| 策略需求 | 输出风险分层和挽留策略 | `retention_strategy.md`、`risk_threshold_rationale.md` |
| 成本分析需求 | 比较不同阈值下的成本收益 | `cost_sensitive_analysis.py` |
| 复现需求 | 提供依赖说明、运行说明、检查脚本 | `requirements.txt`、`run_guide.md`、`final_submission_check.py` |

非功能需求包括可解释性、可复现性、可扩展性和可展示性。其中，可解释性用于提升业务人员对模型结果的信任；可复现性用于保证代码和结果能够在指定环境中重新运行；可展示性用于支撑课程报告和项目验收。

## 2.4 项目创新点

本文的创新点主要体现在以下方面：

1. **业务语义特征工程。** 项目不是仅对原始字段进行机械编码，而是构造了 `service_count`、`avg_monthly_charge`、`is_month_to_month`、`is_auto_payment`、`high_charge_low_tenure` 等具有业务含义的特征，用于刻画客户服务绑定程度、消费压力、合约约束和支付便利性。
2. **多模型对比与类别不平衡处理。** 项目同时比较线性模型、随机森林和梯度提升模型，并针对客户流失样本占比较低的问题引入 `class_weight` 和 `scale_pos_weight` 等机制。
3. **SHAP 可解释性分析。** 项目使用 SHAP 对模型进行全局和局部解释，识别关键影响因素并解释单个客户的流失风险来源。
4. **风险分层与策略映射。** 项目将预测概率转化为高、中、低风险客户池，并将 SHAP 特征映射为差异化挽留策略。
5. **成本敏感分析。** 项目进一步考虑漏判损失、误触达成本和预期挽留收益，比较不同阈值下的净收益，得到推荐业务触达阈值。

---

# 3 数据来源与数据理解

## 3.1 数据集来源

本项目使用公开数据集 **Telco Customer Churn**。该数据集包含电信客户的基本信息、服务订购情况、合同类型、账单费用、支付方式和是否流失等字段，适用于客户流失预测任务。
| 项目 | 内容 |
|---|---|
| 数据集名称 | Telco Customer Churn |
| 数据来源 | Kaggle / IBM Sample Dataset |
| 原始数据路径 | `data/raw/Telco-Customer-Churn.csv` |
| 原始样本量 | 7043 条客户记录 |
| 原始字段数 | 21 个字段 |
| 目标变量 | `Churn`，表示客户是否流失 |
| 任务类型 | 二分类预测任务 |

## 3.2 数据字段说明

原始数据主要包括客户基本属性、服务订购信息、合同和支付信息、费用信息以及流失标签。字段字典已整理在 `docs/data_dictionary.md` 中。

| 字段类别 | 代表字段 | 业务含义 |
|---|---|---|
| 客户基本信息 | gender、SeniorCitizen、Partner、Dependents | 描述客户画像 |
| 服务订购信息 | PhoneService、InternetService、OnlineSecurity、TechSupport | 描述客户使用的电信服务 |
| 合同与支付信息 | Contract、PaperlessBilling、PaymentMethod | 描述客户合约和付款方式 |
| 费用信息 | MonthlyCharges、TotalCharges | 描述客户消费水平 |
| 目标变量 | Churn | 表示客户是否流失 |

## 3.3 数据质量审计

数据质量审计主要关注字段类型、缺失值、重复值、目标变量分布和训练/测试集分布一致性。项目已生成 `docs/data_quality_report.md`，用于记录样本规模、字段统计、缺失值处理和潜在数据处理风险。

原始数据中需要重点处理的是 `TotalCharges` 字段。该字段在原始数据中表现为字符串格式，部分样本无法直接转换为数值，因此本文将其转换为数值型，并对无法转换产生的缺失值使用中位数进行填补。

## 3.4 目标变量分布

原始数据中未流失客户数量为 5174，占 73.46%；流失客户数量为 1869，占 26.54%。这说明客户流失预测任务存在一定类别不平衡问题。如果仅使用 Accuracy 作为评估指标，模型可能偏向多数类客户，从而掩盖对真实流失客户的识别能力。

**图 3-1 客户流失目标变量分布**  
插入图片：`outputs/figures/eda_churn_distribution.png`

## 3.5 关键变量与流失关系分析

探索性分析显示，合同类型、支付方式、互联网服务类型和费用水平与客户流失存在明显关联。

**图 3-2 数值变量分布图**  
插入图片：`outputs/figures/eda_numeric_distribution.png`

**图 3-3 合同类型与客户流失率关系**  
插入图片：`outputs/figures/eda_contract_churn_rate.png`

合同类型与客户流失具有较强关系。月付合同客户的迁移成本较低，通常更容易产生流失风险；长期合约客户受到合约约束，流失风险相对较低。这一发现为后续构造 `is_month_to_month` 特征提供了业务依据。

**图 3-4 支付方式与客户流失率关系**  
插入图片：`outputs/figures/eda_payment_churn_rate.png`

支付方式能够反映客户续费便利性和稳定性。电子支票支付方式可能增加续费摩擦，因此在后续模型解释中也被识别为重要影响因素。

**图 3-5 互联网服务类型与客户流失率关系**  
插入图片：`outputs/figures/eda_internet_churn_rate.png`

互联网服务类型反映客户使用服务的复杂度和价格敏感性。部分光纤网络客户可能由于费用或服务体验问题表现出更高流失风险。

**图 3-6 数值特征相关性热图**  
插入图片：`outputs/figures/eda_correlation_heatmap.png`

---

# 4 数据预处理与特征工程

## 4.1 数据清洗

数据清洗阶段主要完成字段类型修正、缺失值处理和无效字段删除。`customerID` 字段仅用于客户标识，不直接反映客户流失规律，因此在建模数据中删除。`TotalCharges` 字段被转换为数值型，对转换失败产生的缺失值使用中位数填补。

清洗后数据维度为 `(7043, 26)`。经过 One-Hot 编码和业务特征构造后，训练集维度为 `(5634, 37)`，测试集维度为 `(1409, 37)`，其中包含目标变量 `Churn`。

## 4.2 标签编码与类别变量处理

目标变量 `Churn` 原始取值为 `Yes` 和 `No`。本文将其转换为二分类标签：

| 原始值 | 编码值 | 含义 |
|---|---:|---|
| No | 0 | 客户未流失 |
| Yes | 1 | 客户已流失 |

类别变量使用 One-Hot 编码处理，并使用 `drop_first=True` 避免完全共线的虚拟变量。编码后的特征列被保存至 `data/processed/feature_columns.json`，便于后续模型训练和预测阶段保持特征一致性。

## 4.3 训练集与测试集划分

本文采用 80% 训练集和 20% 测试集划分，并使用 `stratify=y` 保持训练集与测试集中流失客户比例一致。该设计有助于在类别不平衡场景下保证评估结果稳定。

**图 4-1 训练集与测试集客户流失分布对比**  
插入图片：`outputs/figures/preprocess_train_test_churn_distribution.png`

## 4.4 业务语义特征工程

为提升模型对客户行为和业务风险的刻画能力，本文构造了多个具有业务含义的衍生特征。

| 特征名 | 构造逻辑 | 业务含义 |
|---|---|---|
| service_count | 统计多个服务字段中取值为 Yes 的服务数量 | 衡量客户服务绑定深度 |
| avg_monthly_charge | TotalCharges / tenure，tenure 为 0 时使用 MonthlyCharges | 衡量客户平均消费压力 |
| is_month_to_month | Contract 是否为 Month-to-month | 衡量客户合约约束强弱 |
| has_internet_service | InternetService 是否不为 No | 表示客户是否使用互联网服务 |
| is_auto_payment | PaymentMethod 是否包含 automatic | 衡量自动续费便利性 |
| high_charge_low_tenure | MonthlyCharges 高于中位数且 tenure 低于中位数 | 识别高消费但低黏性的潜在风险客户 |

其中，`service_count` 反映客户与运营商之间的服务绑定程度。订购服务越多，客户迁移到其他运营商的成本通常越高。`avg_monthly_charge` 和 `high_charge_low_tenure` 则用于刻画价格压力和新客户稳定性风险。`is_month_to_month` 直接反映合约约束，是后续 SHAP 分析中最重要的特征之一。

**图 4-2 业务特征分布图**  
插入图片：`outputs/figures/feature_engineering_business_features.png`

## 4.5 数据处理严谨性说明

当前课程实验流程中，`TotalCharges` 缺失值填补和 `high_charge_low_tenure` 中位数阈值计算基于全量数据完成。由于这些统计量没有使用目标标签，影响相对有限；但在更严格的生产环境中，应先划分训练集和测试集，再在训练集上拟合缺失值填补、编码规则和业务阈值，并通过 `sklearn Pipeline` 或 `ColumnTransformer` 应用于测试集。项目已在 `docs/leakage_risk_and_pipeline_improvement.md` 中给出改进方案。

---

# 5 解决方案与模型设计

## 5.1 总体解决方案

本文的总体解决方案包括数据治理、特征工程、模型训练、模型评估、模型解释和策略输出六个部分。首先对原始数据进行清洗和编码，然后构造业务语义特征；接着训练多个机器学习模型并进行横向对比；在确定最佳模型后，使用 SHAP 进行可解释性分析；最后根据预测概率进行客户风险分层，并结合成本敏感分析提出客户挽留策略。

## 5.2 技术平台与开发环境

| 类别 | 工具或库 |
|---|---|
| 编程语言 | Python 3.13.7 |
| 数据处理 | pandas、numpy |
| 机器学习 | scikit-learn、XGBoost、LightGBM |
| 可解释性分析 | SHAP |
| 图表绘制 | matplotlib、seaborn |
| 模型保存 | joblib |
| 实验展示 | Jupyter Notebook |
| 配置管理 | config.yaml、pyyaml |

具体版本信息已记录在 `docs/environment_versions.txt`，依赖文件已记录在 `requirements.txt`。

## 5.3 模型选择与类别不平衡处理

本文选择 Logistic Regression、Random Forest、XGBoost 和 LightGBM 四类模型进行对比。

| 模型 | 选择原因 |
|---|---|
| Logistic Regression | 作为线性基线模型，具有较强可解释性 |
| Random Forest | 能捕捉非线性关系，可作为树模型对照 |
| XGBoost | 适合表格数据，具有较强非线性建模能力 |
| LightGBM | 高效梯度提升模型，可作为 XGBoost 的对照模型 |

由于客户流失样本占比低于未流失客户，本文在模型训练中引入类别不平衡处理。Logistic Regression 和 Random Forest 使用 `class_weight="balanced"`，XGBoost 根据训练集中负样本数与正样本数计算 `scale_pos_weight`。

## 5.4 评估指标体系

在客户流失预测任务中，漏判真实流失客户会使企业失去提前挽留机会，因此不能只关注 Accuracy。本文综合使用 Accuracy、Precision、Recall、F1、ROC-AUC 和 PR-AUC 进行评估。

| 指标 | 含义 | 业务解释 |
|---|---|---|
| Accuracy | 整体分类正确率 | 容易受多数类影响 |
| Precision | 预测为流失客户中真实流失的比例 | 衡量触达客户的准确性 |
| Recall | 真实流失客户中被识别出来的比例 | 衡量流失客户覆盖能力 |
| F1 | Precision 和 Recall 的综合 | 平衡误报和漏报 |
| ROC-AUC | 模型整体区分能力 | 反映不同阈值下模型排序能力 |
| PR-AUC | Precision-Recall 曲线下面积 | 更适合类别不平衡任务 |

---

# 6 模型训练、评估与结果分析

## 6.1 实验设置

本文使用训练集进行模型训练，测试集进行最终评估。模型训练完成后保存至 `models/` 目录，评估指标和图表保存至 `outputs/metrics/` 与 `outputs/figures/` 目录。为增强稳定性说明，项目还运行了交叉验证脚本 `cross_validate_models.py`，并生成交叉验证结果文件。

## 6.2 多模型结果对比

模型评估结果如下表所示。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.7566 | 0.5279 | 0.7834 | 0.6308 | 0.8430 | 0.6556 |
| Logistic Regression | 0.7402 | 0.5069 | 0.7834 | 0.6155 | 0.8412 | 0.6306 |
| LightGBM | 0.7551 | 0.5287 | 0.7139 | 0.6075 | 0.8332 | 0.6398 |
| Random Forest | 0.7913 | 0.6316 | 0.5134 | 0.5664 | 0.8306 | 0.6283 |

**图 6-1 多模型指标对比图**  
插入图片：`outputs/figures/model_comparison.png`

## 6.3 最优模型选择

从结果可以看出，Random Forest 的 Accuracy 最高，但其 Recall 仅为 0.5134，说明其漏掉了较多真实流失客户。对于客户流失预警任务，漏判流失客户会导致企业失去提前挽留机会，因此 Recall、F1 和 PR-AUC 比单纯 Accuracy 更符合业务目标。

XGBoost 的 ROC-AUC 为 0.8430，PR-AUC 为 0.6556，F1 为 0.6308，Recall 为 0.7834，在多个关键指标上综合表现最好。因此，本文选择 XGBoost 作为最终客户流失预测模型。

## 6.4 ROC 曲线与 PR 曲线分析

ROC 曲线用于衡量模型在不同阈值下对正负样本的整体区分能力。PR 曲线更关注正类样本识别效果，适合客户流失这种类别不平衡任务。

**图 6-2 ROC 曲线**  
插入图片：`outputs/figures/roc_curve.png`

**图 6-3 PR 曲线**  
插入图片：`outputs/figures/pr_curve.png`

## 6.5 混淆矩阵与阈值分析

混淆矩阵能够展示模型在测试集上的 TP、FP、TN、FN 情况，有助于理解模型的误判类型。阈值分析进一步展示不同分类阈值下 Precision、Recall 和 F1 的变化。

**图 6-4 最佳模型混淆矩阵**  
插入图片：`outputs/figures/confusion_matrix_best_model.png`

**图 6-5 阈值分析图**  
插入图片：`outputs/figures/threshold_analysis.png`

阈值降低时，模型通常能够识别更多真实流失客户，Recall 提升，但误报也会增加；阈值升高时，Precision 可能提升，但会漏掉更多潜在流失客户。该结果为后续成本敏感分析和业务阈值选择提供基础。

## 6.6 交叉验证稳定性分析

为避免单次训练/测试集划分造成偶然性，项目补充了交叉验证稳定性分析。交叉验证结果保存于 `outputs/metrics/cross_validation_results.csv`，对应图表为 `outputs/figures/cross_validation_auc.png`。

**图 6-6 交叉验证 AUC 对比图**  
插入图片：`outputs/figures/cross_validation_auc.png`

【此处根据 `cross_validation_results.csv` 补充各模型的 CV 均值和标准差，并说明 XGBoost 的表现是否稳定。】

---

# 7 SHAP 可解释性分析

## 7.1 SHAP 方法简介

SHAP 是一种基于博弈论思想的模型解释方法，可用于衡量每个特征对模型预测结果的贡献。对于客户流失预测任务，SHAP 不仅能够说明哪些特征重要，还能够解释某个特征如何推动模型将客户判断为高风险或低风险。

## 7.2 全局特征重要性分析

项目使用 XGBoost 最优模型进行 SHAP 分析，并生成特征重要性结果。Top 10 重要特征如下：

| 排名 | 特征 |
|---:|---|
| 1 | is_month_to_month |
| 2 | tenure |
| 3 | InternetService_Fiber optic |
| 4 | Contract_Two year |
| 5 | TotalCharges |
| 6 | avg_monthly_charge |
| 7 | PaymentMethod_Electronic check |
| 8 | MonthlyCharges |
| 9 | PaperlessBilling_Yes |
| 10 | OnlineSecurity_Yes |

**图 7-1 SHAP 特征重要性条形图**  
插入图片：`outputs/figures/shap_bar.png`

## 7.3 特征影响方向分析

SHAP summary 图展示了不同特征取值对预测结果的正负影响。结合 SHAP 结果可以发现，月付合同、较短在网时长、光纤网络服务、较高费用水平和电子支票支付方式等因素对客户流失风险具有较强影响。

**图 7-2 SHAP Summary 图**  
插入图片：`outputs/figures/shap_summary.png`

**图 7-3 Top1 特征依赖图**  
插入图片：`outputs/figures/shap_dependence_top1.png`

**图 7-4 Top2 特征依赖图**  
插入图片：`outputs/figures/shap_dependence_top2.png`

## 7.4 局部客户解释

局部解释用于分析单个客户为何被模型判断为高风险。通过 SHAP waterfall 图，可以观察某一客户的具体风险来源，例如月付合同、较短在网时长、较高月费用或特定支付方式等因素如何共同推高流失概率。

**图 7-5 高风险客户局部 SHAP 解释图**  
插入图片：`outputs/figures/shap_local_explanation.png`

## 7.5 可解释性结果的业务启示

SHAP 结果说明，客户流失不是由单一因素决定，而是由合约约束、客户黏性、服务类型、费用压力和支付便利性等多类因素共同影响。这些解释结果能够帮助运营人员理解模型预测逻辑，并为后续客户挽留策略提供依据。

---

# 8 风险分层、成本敏感分析与挽留策略

## 8.1 客户风险分层方法

本文将模型输出的 `churn_probability` 转化为低风险、中风险和高风险三类客户：

| 风险层级 | 概率区间 | 运营含义 |
|---|---|---|
| 高风险 | churn_probability >= 0.7 | 需要优先人工挽留 |
| 中风险 | 0.4 <= churn_probability < 0.7 | 适合自动化营销触达 |
| 低风险 | churn_probability < 0.4 | 保持常规维护 |

根据项目输出，高风险客户数量为 326，占测试集客户的 23.14%。

## 8.2 风险分层结果

风险分层结果保存于 `outputs/metrics/risk_segment_summary.csv`，高风险客户名单保存于 `outputs/predictions/high_risk_customers.csv`。这些结果可以直接提供给运营部门，用于建立重点挽留客户池。

【此处插入 risk_segment_summary.csv 表格，展示低风险、中风险、高风险客户数量和占比。】

## 8.3 成本敏感分析与商业阈值选择

模型默认分类阈值通常为 0.5，但在客户流失业务中，不同错误类型的成本并不相同。漏判一个真实流失客户会导致企业失去提前挽留机会，而误触达一个非流失客户主要产生营销成本。因此，本文引入成本敏感分析，比较不同阈值下的总成本和净收益。

项目设定如下业务假设：

| 参数 | 含义 | 取值 |
|---|---|---:|
| cost_false_negative | 漏判一个真实流失客户的机会损失 | 500 |
| cost_false_positive | 误触达一个非流失客户的成本 | 50 |
| retention_cost_high | 高风险客户挽留成本 | 120 |
| retention_cost_medium | 中风险客户挽留成本 | 40 |
| expected_revenue_saved | 成功挽留客户的预期收益 | 300 |

成本敏感分析结果表明，当业务触达阈值设置为 0.3 时，净收益最高，`net_benefit = 7780`，`total_estimated_cost = 92120`。因此，本文将 0.3 作为“是否纳入营销触达池”的推荐业务阈值。

**图 8-1 成本敏感分析图**  
插入图片：`outputs/figures/cost_sensitive_analysis.png`

需要注意的是，0.3、0.5 和 0.4/0.7 分别服务于不同目的：0.5 用于模型默认分类评价，0.3 用于成本收益最优的营销触达判断，0.4/0.7 用于运营资源分层。

## 8.4 SHAP 驱动的差异化挽留策略

基于 SHAP 关键特征和客户风险分层结果，本文提出差异化挽留策略。

| SHAP 关键特征 | 典型高风险表现 | 对应挽留策略 | 预期业务目标 |
|---|---|---|---|
| is_month_to_month | 月付合同客户缺少长期绑定 | 推一年期或两年期合约优惠 | 提高合约绑定程度 |
| tenure 低 | 新客户尚未形成稳定黏性 | 新客关怀、服务体验跟踪 | 降低早期流失 |
| InternetService_Fiber optic | 光纤客户对价格和服务质量敏感 | 网络质量保障与资费优化 | 提升服务满意度 |
| MonthlyCharges 高 | 月费用带来价格压力 | 套餐重组或降档推荐 | 降低价格敏感流失 |
| PaymentMethod_Electronic check | 续费摩擦较大 | 引导自动扣款或绑定银行卡 | 提高续费稳定性 |

## 8.5 策略执行流程与推广价值

实际应用中，企业可以每周或每月运行一次客户流失预测模型，将预测概率高于 0.3 的客户纳入营销触达池，再根据 0.4 和 0.7 阈值划分自动化营销与人工重点挽留优先级。对于高风险客户，运营部门可结合 SHAP 解释结果选择具体挽留方式。

该策略可以帮助企业减少盲目营销，提高客户挽留效率，并为运营预算分配提供量化依据。

---

# 9 关键代码实现与系统复现说明

## 9.1 项目目录结构

项目采用工程化目录结构，主要包括数据目录、核心代码目录、Notebook 展示目录、模型文件目录、输出结果目录和说明文档目录。

```text
TelecomCustomer_ChurnPrediction
├── data
│   ├── raw
│   └── processed
├── notebooks
├── src
├── models
├── outputs
│   ├── figures
│   ├── metrics
│   ├── predictions
│   └── strategy
├── docs
├── README.md
├── requirements.txt
└── config.yaml
```

## 9.2 核心代码模块

| 文件 | 功能 |
|---|---|
| `preprocess.py` | 数据清洗、标签转换、数据划分 |
| `feature_engineering.py` | 业务特征构造 |
| `train_models.py` | 多模型训练与模型保存 |
| `evaluate_models.py` | 模型评估、指标和图表输出 |
| `cross_validate_models.py` | 交叉验证稳定性分析 |
| `shap_analysis.py` | SHAP 可解释性分析 |
| `cost_sensitive_analysis.py` | 成本敏感分析 |
| `check_outputs.py` | 关键输出文件检查 |
| `check_notebooks.py` | Notebook 完整性和乱码检查 |
| `final_submission_check.py` | 最终提交一致性检查 |

## 9.3 关键代码片段

### 9.3.1 业务特征工程核心代码

```python
# 示例：业务特征工程核心逻辑
service_columns = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
df["service_count"] = df[service_columns].eq("Yes").sum(axis=1)
df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(df["MonthlyCharges"])
df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
```

该代码将客户服务绑定、平均消费压力和合约约束转化为可建模特征，体现了业务理解在特征工程中的应用。

### 9.3.2 XGBoost 类别不平衡处理

```python
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
```

客户流失样本占比较低，因此 XGBoost 使用 `scale_pos_weight` 增强对流失客户的识别能力。

### 9.3.3 成本敏感分析核心逻辑

```python
total_estimated_cost = FN * cost_false_negative + FP * cost_false_positive
estimated_revenue_saved = TP * expected_revenue_saved
net_benefit = estimated_revenue_saved - total_estimated_cost
```

该逻辑将模型预测结果转化为商业成本和收益，用于选择更符合运营目标的业务阈值。

## 9.4 运行流程与输出文件

完整运行流程如下：

```powershell
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
python src/cross_validate_models.py
python src/shap_analysis.py
python src/cost_sensitive_analysis.py
python src/final_submission_check.py
```

主要输出包括：

| 类型 | 文件 |
|---|---|
| 模型文件 | `models/*.pkl` |
| 模型指标 | `outputs/metrics/model_comparison.csv` |
| SHAP 结果 | `outputs/metrics/shap_feature_importance.csv` |
| 成本分析 | `outputs/metrics/cost_sensitive_analysis.csv` |
| 预测结果 | `outputs/predictions/customer_churn_predictions.csv` |
| 策略文件 | `outputs/strategy/retention_strategy.md` |

## 9.5 最终提交检查

项目提供 `final_submission_check.py` 用于检查提交前文件完整性。该脚本检查数据、模型、指标、图表、Notebook、依赖说明和临时缓存等内容。当前提交前仅需清理 `__pycache__` 和 `.ipynb_checkpoints` 等缓存目录即可。

---

# 10 社会价值、总结与讨论

## 10.1 项目主要结论

本文围绕电信客户流失问题构建了预测性商务智能分析方案。实验结果表明，XGBoost 在 ROC-AUC、PR-AUC、F1 和 Recall 等指标上综合表现最好，适合作为最终客户流失预测模型。SHAP 分析进一步揭示了客户流失的关键因素，包括合同类型、在网时长、互联网服务类型、费用水平和支付方式等。

## 10.2 商业价值与社会价值

从商业价值看，该项目可以帮助电信企业提前识别高风险客户，减少盲目营销，提高客户挽留效率。通过成本敏感分析，企业可以根据不同阈值下的成本收益选择更合理的触达策略，从而提高营销投入产出比。

从客户价值看，企业能够根据客户的真实风险来源提供更匹配的套餐、服务和续约方案，减少无差别营销打扰，提升客户体验。

## 10.3 推广前景

本文提出的“预测—解释—分层—成本分析—策略输出”框架不仅适用于电信客户流失预测，也可以推广到 SaaS 软件续费预测、银行信用卡销卡预测、在线教育续费预测、会员制平台流失预警和电商用户留存分析等场景。

## 10.4 局限性

本项目仍存在以下局限：

1. 数据集为静态历史数据，缺少客户行为时间序列，无法分析客户风险随时间变化的趋势。
2. 成本敏感分析中的成本参数为业务假设，真实应用中需要结合企业历史营销数据校准。
3. 当前策略尚未通过 A/B 测试验证，实际效果仍需在真实业务环境中评估。
4. 当前部分统计量基于全量数据计算，生产环境中应采用 Pipeline 机制进一步避免统计量泄漏。
5. 当前模型主要基于结构化表格数据，未纳入客服文本、投诉记录和用户行为日志等非结构化数据。

## 10.5 后续改进方向

后续可以从以下方向继续优化：

1. 引入客户月度行为序列，构建时序流失预测模型。
2. 使用生存分析方法预测客户可能流失的时间点。
3. 将模型部署为接口服务，接入 CRM 系统，实现周期性客户风险更新。
4. 通过 A/B 测试验证不同挽留策略的实际效果。
5. 结合真实企业营销成本和客户生命周期价值，进一步优化成本敏感分析参数。

---

# 参考文献

[1] Kaggle. Telco Customer Churn Dataset.  
[2] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.  
[3] Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS, 2017.  
[4] Lundberg S M, Lee S I. A Unified Approach to Interpreting Model Predictions. NIPS, 2017.  
[5] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.  
[6] Han J, Kamber M, Pei J. Data Mining: Concepts and Techniques. Morgan Kaufmann, 2011.  

---

# 附录

## 附录 A 项目主要文件清单

| 类型 | 文件 |
|---|---|
| 原始数据 | `data/raw/Telco-Customer-Churn.csv` |
| 处理后数据 | `data/processed/churn_cleaned.csv`、`train.csv`、`test.csv` |
| 模型文件 | `models/logistic_regression.pkl`、`random_forest.pkl`、`xgboost_model.pkl`、`lightgbm_model.pkl`、`best_model.pkl` |
| 模型评估 | `outputs/metrics/model_comparison.csv`、`threshold_analysis.csv`、`cross_validation_results.csv` |
| 可解释性结果 | `outputs/metrics/shap_feature_importance.csv` |
| 成本分析 | `outputs/metrics/cost_sensitive_analysis.csv` |
| 策略文件 | `outputs/strategy/retention_strategy.md`、`cost_benefit_strategy.md`、`risk_threshold_rationale.md` |
| 运行说明 | `docs/run_guide.md` |
| 环境说明 | `docs/environment_versions.txt` |

## 附录 B Notebook 文件说明

| Notebook | 内容 |
|---|---|
| `01_data_understanding.ipynb` | 数据理解与探索性分析 |
| `02_preprocessing_and_features.ipynb` | 数据预处理与特征工程 |
| `03_model_comparison.ipynb` | 多模型训练与对比 |
| `04_shap_and_strategy.ipynb` | SHAP 可解释性分析与业务策略输出 |
