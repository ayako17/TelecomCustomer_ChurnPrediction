# 数据字典（Telco Customer Churn）

说明：
- 数据集：Telco Customer Churn
- 目标变量：`Churn`
- 本表覆盖原始字段与新增业务特征。

| 字段名 | 数据类型 | 业务含义 | 是否参与建模 | 处理方式 |
|---|---|---|---|---|
| customerID | string | 客户唯一标识 | 否 | 预处理阶段删除，不参与训练 |
| gender | category | 性别 | 是 | One-Hot 编码（drop_first） |
| SeniorCitizen | int | 是否老年客户（0/1） | 是 | 保留数值型 |
| Partner | category | 是否有伴侣 | 是 | One-Hot 编码 |
| Dependents | category | 是否有家属 | 是 | One-Hot 编码 |
| tenure | int | 在网时长（月） | 是 | 保留数值型 |
| PhoneService | category | 是否开通电话服务 | 是 | One-Hot 编码 |
| MultipleLines | category | 是否多线路服务 | 是 | One-Hot 编码 |
| InternetService | category | 互联网服务类型 | 是 | One-Hot 编码 |
| OnlineSecurity | category | 是否开通在线安全服务 | 是 | One-Hot 编码 |
| OnlineBackup | category | 是否开通在线备份服务 | 是 | One-Hot 编码 |
| DeviceProtection | category | 是否开通设备保护服务 | 是 | One-Hot 编码 |
| TechSupport | category | 是否开通技术支持服务 | 是 | One-Hot 编码 |
| StreamingTV | category | 是否开通流媒体电视服务 | 是 | One-Hot 编码 |
| StreamingMovies | category | 是否开通流媒体电影服务 | 是 | One-Hot 编码 |
| Contract | category | 合约类型 | 是 | One-Hot 编码 |
| PaperlessBilling | category | 是否电子账单 | 是 | One-Hot 编码 |
| PaymentMethod | category | 支付方式 | 是 | One-Hot 编码 |
| MonthlyCharges | float | 月账单金额 | 是 | 保留数值型 |
| TotalCharges | string/float | 总消费金额 | 是 | 转数值后以中位数填补缺失 |
| Churn | category/int | 是否流失（目标变量） | 是（目标） | Yes/No 转换为 1/0 |
| service_count | int | 多项增值服务中“Yes”数量 | 是 | 新增业务特征，保留数值型 |
| avg_monthly_charge | float | 平均月消费压力指标 | 是 | `TotalCharges / tenure`，tenure=0 时回退 MonthlyCharges |
| is_month_to_month | int | 是否月付短约客户 | 是 | `Contract == Month-to-month` 映射为 1/0 |
| has_internet_service | int | 是否有互联网服务 | 是 | `InternetService != No` 映射为 1/0 |
| is_auto_payment | int | 是否自动支付客户 | 是 | `PaymentMethod` 包含 automatic 映射为 1/0 |
| high_charge_low_tenure | int | 高消费低时长风险标记 | 是 | `MonthlyCharges > 中位数` 且 `tenure < 中位数` 映射为 1/0 |
