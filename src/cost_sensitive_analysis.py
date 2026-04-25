from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import ensure_dir


THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

BUSINESS_ASSUMPTIONS = {
    "cost_false_negative": 500,
    "cost_false_positive": 50,
    "retention_cost_high": 120,
    "retention_cost_medium": 40,
    "expected_revenue_saved": 300,
}


def compute_threshold_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict:
    """Compute confusion-matrix metrics and business cost/benefit under one threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    high_pool = int(((y_pred == 1) & (y_prob >= 0.7)).sum())
    medium_pool = int(((y_pred == 1) & (y_prob >= 0.4) & (y_prob < 0.7)).sum())

    retention_cost = (
        high_pool * BUSINESS_ASSUMPTIONS["retention_cost_high"]
        + medium_pool * BUSINESS_ASSUMPTIONS["retention_cost_medium"]
    )
    estimated_loss_from_fn = fn * BUSINESS_ASSUMPTIONS["cost_false_negative"]
    estimated_cost_from_fp = fp * BUSINESS_ASSUMPTIONS["cost_false_positive"]
    total_estimated_cost = estimated_loss_from_fn + estimated_cost_from_fp + retention_cost
    estimated_revenue_saved = tp * BUSINESS_ASSUMPTIONS["expected_revenue_saved"]
    net_benefit = estimated_revenue_saved - total_estimated_cost

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "estimated_loss_from_fn": estimated_loss_from_fn,
        "estimated_cost_from_fp": estimated_cost_from_fp,
        "retention_intervention_cost": retention_cost,
        "total_estimated_cost": total_estimated_cost,
        "estimated_revenue_saved": estimated_revenue_saved,
        "net_benefit": net_benefit,
    }


def choose_recommended_threshold(result_df: pd.DataFrame) -> float:
    """Choose threshold by max net_benefit, then min total_estimated_cost as tiebreaker."""
    sorted_df = result_df.sort_values(
        by=["net_benefit", "total_estimated_cost"],
        ascending=[False, True],
    )
    return float(sorted_df.iloc[0]["threshold"])


def save_cost_plot(result_df: pd.DataFrame, recommended_threshold: float, output_path: Path) -> None:
    """Plot total cost and net benefit across thresholds."""
    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 5))
    plt.plot(
        result_df["threshold"],
        result_df["total_estimated_cost"],
        marker="o",
        linewidth=2,
        label="Total Estimated Cost",
    )
    plt.plot(
        result_df["threshold"],
        result_df["net_benefit"],
        marker="o",
        linewidth=2,
        label="Net Benefit",
    )

    best_row = result_df[result_df["threshold"] == recommended_threshold].iloc[0]
    plt.scatter(
        [recommended_threshold],
        [best_row["net_benefit"]],
        s=120,
        color="red",
        zorder=3,
        label=f"Recommended Threshold = {recommended_threshold:.1f}",
    )
    plt.annotate(
        f"Recommend {recommended_threshold:.1f}",
        xy=(recommended_threshold, best_row["net_benefit"]),
        xytext=(recommended_threshold + 0.02, best_row["net_benefit"] * 0.95),
        arrowprops={"arrowstyle": "->"},
    )

    plt.title("Cost-Sensitive Analysis by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Amount")
    plt.xticks(result_df["threshold"])
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_cost_benefit_strategy(
    result_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    recommended_threshold: float,
    output_path: Path,
) -> None:
    """Create markdown strategy memo for cost-benefit decision support."""
    ensure_dir(output_path.parent)

    display_df = result_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "total_estimated_cost",
            "estimated_revenue_saved",
            "net_benefit",
        ]
    ].copy()
    display_df = display_df.sort_values("threshold")

    best = result_df[result_df["threshold"] == recommended_threshold].iloc[0]
    threshold_section = dataframe_to_markdown(threshold_df.sort_values("threshold"))
    cost_section = dataframe_to_markdown(display_df)

    content = f"""# 成本收益驱动的阈值策略说明

## 1. 成本参数假设
- cost_false_negative = {BUSINESS_ASSUMPTIONS['cost_false_negative']}
- cost_false_positive = {BUSINESS_ASSUMPTIONS['cost_false_positive']}
- retention_cost_high = {BUSINESS_ASSUMPTIONS['retention_cost_high']}
- retention_cost_medium = {BUSINESS_ASSUMPTIONS['retention_cost_medium']}
- expected_revenue_saved = {BUSINESS_ASSUMPTIONS['expected_revenue_saved']}

## 2. 阈值成本收益对比
{cost_section}

## 3. 推荐阈值
- 推荐阈值：**{recommended_threshold:.1f}**
- 该阈值对应总估计成本：**{int(best['total_estimated_cost'])}**
- 该阈值对应净收益：**{int(best['net_benefit'])}**

## 4. 为什么业务决策不应只追求 Accuracy
- Accuracy 在类别不平衡场景中容易被多数类主导，无法反映流失客户识别价值。
- 运营目标是减少真实流失损失，因此需要综合考虑 FN 损失、FP 干预成本与干预收益。
- 阈值选择应围绕净收益最大化，而不是单一分类指标最大化。

## 5. 如何用于运营预算分配
- 高风险池（概率 >= 0.7）使用人工干预与高成本挽留预算。
- 中风险池（0.4 <= 概率 < 0.7）使用自动化触达与低成本运营动作。
- 依据 `cost_sensitive_analysis.csv` 每周复盘阈值与预算配比，动态调整资源投入。

## 6. 与现有阈值分析对照（classification metrics）
{threshold_section}
"""
    output_path.write_text(content, encoding="utf-8")


def generate_risk_threshold_rationale(
    threshold_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    recommended_threshold: float,
    output_path: Path,
) -> None:
    """Generate rationale document for classification threshold vs risk segmentation thresholds."""
    ensure_dir(output_path.parent)

    merged = threshold_df.merge(
        cost_df[
            [
                "threshold",
                "total_estimated_cost",
                "net_benefit",
            ]
        ],
        on="threshold",
        how="left",
    ).sort_values("threshold")

    table = dataframe_to_markdown(merged)

    content = f"""# 风险阈值依据说明

## 1. 分类阈值与运营分层阈值的区别
- **0.5**：用于二分类判定（流失/不流失）。
- **0.4 / 0.7**：用于运营分层（中风险、高风险）与资源优先级分配。

## 2. 三层风险池定义
- 高风险：`churn_probability >= 0.7`
- 中风险：`0.4 <= churn_probability < 0.7`
- 低风险：`churn_probability < 0.4`

## 3. 运营资源分配逻辑
- 高风险：人工客服、强干预、合约升级。
- 中风险：自动化营销、套餐推荐。
- 低风险：常规维护、满意度跟踪。

## 4. 阈值选择依据
- 推荐分类阈值：**{recommended_threshold:.1f}**（基于成本收益净值而非单一 F1）。
- 选择阈值时同时参考：
  - `threshold_analysis.csv`（precision/recall/f1）
  - `cost_sensitive_analysis.csv`（total_estimated_cost/net_benefit）
- 因此阈值决策并非单纯追求 F1，而是服务于预算利用效率与资源配置效果。

## 5. 量化对照表
{table}
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    test_path = project_root / "data" / "processed" / "test.csv"
    prediction_path = project_root / "outputs" / "predictions" / "customer_churn_predictions.csv"
    threshold_path = project_root / "outputs" / "metrics" / "threshold_analysis.csv"

    cost_output_path = project_root / "outputs" / "metrics" / "cost_sensitive_analysis.csv"
    fig_output_path = project_root / "outputs" / "figures" / "cost_sensitive_analysis.png"
    strategy_output_path = project_root / "outputs" / "strategy" / "cost_benefit_strategy.md"
    rationale_output_path = project_root / "outputs" / "strategy" / "risk_threshold_rationale.md"

    if not test_path.exists() or not prediction_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Required input file missing for cost-sensitive analysis.")

    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(prediction_path)
    threshold_df = pd.read_csv(threshold_path)

    if "Churn" not in test_df.columns:
        raise KeyError("test.csv must contain target column 'Churn'.")
    for col in ["true_label", "churn_probability"]:
        if col not in pred_df.columns:
            raise KeyError(f"Prediction file missing required column: {col}")

    n = min(len(test_df), len(pred_df))
    if len(test_df) != len(pred_df):
        print(
            f"Warning: length mismatch test={len(test_df)} prediction={len(pred_df)}. "
            f"Using first {n} rows for alignment."
        )

    y_true = test_df["Churn"].iloc[:n].astype(int).reset_index(drop=True)
    y_true_pred_file = pred_df["true_label"].iloc[:n].astype(int).reset_index(drop=True)
    y_prob = pred_df["churn_probability"].iloc[:n].astype(float).reset_index(drop=True)

    if not y_true.equals(y_true_pred_file):
        print("Warning: test.csv Churn and prediction true_label are not fully identical.")

    rows = [compute_threshold_metrics(y_true, y_prob, t) for t in THRESHOLDS]
    result_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    # Merge existing threshold analysis as reference fields.
    ref = threshold_df[["threshold", "precision", "recall", "f1"]].rename(
        columns={
            "precision": "ref_precision",
            "recall": "ref_recall",
            "f1": "ref_f1",
        }
    )
    result_df = result_df.merge(ref, on="threshold", how="left")

    recommended_threshold = choose_recommended_threshold(result_df)
    result_df["is_recommended"] = result_df["threshold"].eq(recommended_threshold)

    ensure_dir(cost_output_path.parent)
    result_df.to_csv(cost_output_path, index=False)

    save_cost_plot(result_df, recommended_threshold, fig_output_path)
    generate_cost_benefit_strategy(result_df, threshold_df, recommended_threshold, strategy_output_path)
    generate_risk_threshold_rationale(
        threshold_df,
        result_df,
        recommended_threshold,
        rationale_output_path,
    )

    print("Cost-sensitive analysis completed.")
    print(f"Recommended business threshold: {recommended_threshold:.1f}")
    print(f"Saved metric file: {cost_output_path}")
    print(f"Saved figure file: {fig_output_path}")
    print(f"Saved strategy file: {strategy_output_path}")
    print(f"Saved rationale file: {rationale_output_path}")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without extra dependencies."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    rows: list[str] = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.6f}".rstrip("0").rstrip("."))
            else:
                values.append(str(val))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
