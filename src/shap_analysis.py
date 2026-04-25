from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir, load_pickle, load_yaml_config


def infer_model_name(model: Any) -> str:
    """Infer model name from estimator class/module."""
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    class_name = estimator.__class__.__name__.lower()
    module_name = estimator.__class__.__module__.lower()
    marker = f"{module_name}.{class_name}"

    if "xgb" in marker or "xgboost" in marker:
        return "xgboost"
    if "lgbm" in marker or "lightgbm" in marker:
        return "lightgbm"
    if "randomforest" in marker:
        return "random_forest"
    if "logistic" in marker:
        return "logistic_regression"
    return class_name


def load_best_model(project_root: Path) -> tuple[Any, str]:
    """Load best model and return (model_object, model_name)."""
    models_dir = project_root / "models"
    best_model_path = models_dir / "best_model.pkl"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    model = load_pickle(best_model_path)
    model_name = infer_model_name(model)

    # Prefer explicit xgboost model artifact when best model is xgboost.
    xgb_model_path = models_dir / "xgboost_model.pkl"
    if model_name == "xgboost" and xgb_model_path.exists():
        try:
            model = load_pickle(xgb_model_path)
            model_name = "xgboost"
            print("检测到最佳模型为 xgboost，已优先加载 xgboost_model.pkl 进行 SHAP 分析。")
        except Exception as error:  # pragma: no cover - runtime guard
            print(f"xgboost_model.pkl 加载失败，回退使用 best_model.pkl。详情: {error}")

    # Fallback by model comparison top row if class inference is ambiguous.
    comparison_path = project_root / "outputs" / "metrics" / "model_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        if not comparison_df.empty and "model" in comparison_df.columns:
            top_model = str(comparison_df.iloc[0]["model"])
            top_path_map = {
                "xgboost": models_dir / "xgboost_model.pkl",
                "lightgbm": models_dir / "lightgbm_model.pkl",
                "random_forest": models_dir / "random_forest.pkl",
                "logistic_regression": models_dir / "logistic_regression.pkl",
            }
            top_model_path = top_path_map.get(top_model)
            if top_model_path is not None and top_model_path.exists():
                try:
                    model = load_pickle(top_model_path)
                    model_name = top_model
                except Exception:
                    pass

    print(f"最终用于 SHAP 的模型: {model_name}")
    return model, model_name


def load_test_data(project_root: Path, target_column: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Load test data and split into X_test/y_test."""
    test_path = project_root / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    test_df = pd.read_csv(test_path)
    if target_column not in test_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in test data.")

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return X_test, y_test


def _extract_shap_matrix(shap_values: Any) -> np.ndarray:
    """Convert different SHAP output formats to (n_samples, n_features)."""
    if isinstance(shap_values, list):
        matrix = np.asarray(shap_values[-1])
    elif hasattr(shap_values, "values"):
        matrix = np.asarray(shap_values.values)
    else:
        matrix = np.asarray(shap_values)

    if matrix.ndim == 3:
        # Typical binary-class shape from some explainers: (n, features, classes)
        matrix = matrix[:, :, -1]
    elif matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def _extract_expected_value(expected_value: Any) -> float:
    """Extract scalar base value from explainer expected value."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_value).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(arr[-1])
    return float(expected_value)


def _compute_xgboost_native_shap(model: Any, X_sample: pd.DataFrame) -> dict[str, Any]:
    """Fallback SHAP values via XGBoost pred_contribs without shap package."""
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    dmatrix = xgb.DMatrix(X_sample, feature_names=list(X_sample.columns))
    contrib = booster.predict(dmatrix, pred_contribs=True)

    shap_matrix = np.asarray(contrib[:, :-1], dtype=float)
    expected_values = np.asarray(contrib[:, -1], dtype=float)
    expected_value = float(np.mean(expected_values))

    return {
        "shap_matrix": shap_matrix,
        "expected_value": expected_value,
        "method": "xgboost_pred_contribs",
        "shap_module": None,
        "explainer": None,
        "raw_values": shap_matrix,
    }


def _compute_lightgbm_native_shap(model: Any, X_sample: pd.DataFrame) -> dict[str, Any]:
    """Fallback SHAP values via LightGBM pred_contrib."""
    contrib = model.predict(X_sample, pred_contrib=True)
    contrib = np.asarray(contrib, dtype=float)

    shap_matrix = contrib[:, :-1]
    expected_values = contrib[:, -1]
    expected_value = float(np.mean(expected_values))

    return {
        "shap_matrix": shap_matrix,
        "expected_value": expected_value,
        "method": "lightgbm_pred_contrib",
        "shap_module": None,
        "explainer": None,
        "raw_values": shap_matrix,
    }


def compute_shap_values(model: Any, X_sample: pd.DataFrame) -> dict[str, Any]:
    """
    Compute SHAP values.
    Priority:
    1) shap.TreeExplainer for tree models
    2) shap.Explainer fallback
    3) native model contrib fallback for xgboost/lightgbm when shap is unavailable
    """
    model_name = infer_model_name(model)

    try:
        shap = import_module("shap")
    except ModuleNotFoundError:
        shap = None

    if shap is not None:
        try:
            if model_name in {"xgboost", "lightgbm", "random_forest"}:
                explainer = shap.TreeExplainer(model)
                raw_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(model, X_sample)
                raw_values = explainer(X_sample)

            shap_matrix = _extract_shap_matrix(raw_values)
            expected_value = _extract_expected_value(getattr(explainer, "expected_value", 0.0))
            return {
                "shap_matrix": shap_matrix,
                "expected_value": expected_value,
                "method": "shap",
                "shap_module": shap,
                "explainer": explainer,
                "raw_values": raw_values,
            }
        except Exception as error:  # pragma: no cover - runtime guard
            print(f"shap 计算失败，尝试兼容回退。详情: {error}")

    if model_name == "xgboost":
        print("shap 不可用或不兼容，回退到 xgboost pred_contribs。")
        return _compute_xgboost_native_shap(model, X_sample)

    if model_name == "lightgbm":
        print("shap 不可用或不兼容，回退到 lightgbm pred_contrib。")
        return _compute_lightgbm_native_shap(model, X_sample)

    raise RuntimeError("无法计算 SHAP 值：当前模型与环境不支持可用回退方案。")


def save_shap_summary_plot(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: Path,
    shap_module: Any | None = None,
    max_display: int = 15,
) -> None:
    """Save SHAP summary (beeswarm-like) plot."""
    ensure_dir(output_path.parent)

    if shap_module is not None:
        plt.figure(figsize=(10, 6))
        shap_module.summary_plot(shap_matrix, X_sample, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Manual fallback beeswarm-like plot when shap package is unavailable.
    importances = np.mean(np.abs(shap_matrix), axis=0)
    top_idx = np.argsort(importances)[::-1][:max_display]
    top_features = X_sample.columns[top_idx]
    top_shap = shap_matrix[:, top_idx]
    top_values = X_sample.iloc[:, top_idx].to_numpy()

    plt.figure(figsize=(10, 6))
    rng = np.random.default_rng(42)
    for i in range(len(top_features)):
        y = np.full(top_shap.shape[0], i) + rng.uniform(-0.25, 0.25, size=top_shap.shape[0])
        scatter = plt.scatter(
            top_shap[:, i],
            y,
            c=top_values[:, i],
            cmap="coolwarm",
            s=12,
            alpha=0.7,
        )
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("SHAP value")
    plt.title("SHAP Summary (Fallback Beeswarm)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Feature value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_bar_plot(shap_importance_df: pd.DataFrame, output_path: Path, top_k: int = 15) -> None:
    """Save mean(|SHAP|) bar plot."""
    ensure_dir(output_path.parent)
    top_df = shap_importance_df.head(top_k).copy().iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.xlabel("mean(|SHAP|)")
    plt.title(f"Top {top_k} Feature Importance by SHAP")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_top_feature_dependence_plot(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Save dependence plot for top1 and top2 features if available."""
    ensure_dir(output_dir)
    generated_files: list[str] = []

    top_features = shap_importance_df["feature"].head(2).tolist()
    for idx, feature_name in enumerate(top_features, start=1):
        file_path = output_dir / f"shap_dependence_top{idx}.png"
        try:
            feature_values = X_sample[feature_name].to_numpy()
            shap_values = shap_matrix[:, X_sample.columns.get_loc(feature_name)]

            plt.figure(figsize=(8, 5))
            plt.scatter(feature_values, shap_values, s=14, alpha=0.7)
            plt.xlabel(feature_name)
            plt.ylabel("SHAP value")
            plt.title(f"SHAP Dependence Plot - Top{idx}: {feature_name}")
            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            generated_files.append(file_path.name)
        except Exception as error:  # pragma: no cover - runtime guard
            print(f"Top{idx} dependence plot 生成失败，已跳过。详情: {error}")
    return generated_files


def save_local_explanation_plot(
    model: Any,
    X_test: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_path: Path,
    expected_value: float,
    shap_module: Any | None = None,
) -> None:
    """Save local explanation plot for the highest-risk customer."""
    ensure_dir(output_path.parent)

    if predictions_df.empty:
        raise ValueError("Predictions file is empty, cannot build local explanation.")

    top_pos = int(predictions_df["churn_probability"].astype(float).values.argmax())
    x_local = X_test.iloc[[top_pos]]
    local_result = compute_shap_values(model, x_local)
    local_shap = local_result["shap_matrix"][0]
    base_value = local_result.get("expected_value", expected_value)

    # Try shap waterfall first when shap package exists.
    if shap_module is not None:
        try:
            explanation = shap_module.Explanation(
                values=local_shap,
                base_values=base_value,
                data=x_local.iloc[0].values,
                feature_names=X_test.columns.tolist(),
            )
            plt.figure(figsize=(10, 6))
            shap_module.plots.waterfall(explanation, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return
        except Exception as error:
            print(f"waterfall 图兼容性问题，回退到手工本地解释图。详情: {error}")

    # Manual local explanation fallback.
    local_df = pd.DataFrame(
        {"feature": X_test.columns, "shap_value": local_shap, "feature_value": x_local.iloc[0].values}
    )
    local_df["abs_shap"] = local_df["shap_value"].abs()
    local_top = local_df.sort_values("abs_shap", ascending=False).head(15).iloc[::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in local_top["shap_value"]]

    plt.figure(figsize=(10, 6))
    plt.barh(local_top["feature"], local_top["shap_value"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("SHAP contribution")
    plt.title("Local Explanation for Highest-Risk Customer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def export_shap_importance(shap_matrix: np.ndarray, feature_names: list[str], output_path: Path) -> pd.DataFrame:
    """Export mean absolute SHAP importance table."""
    ensure_dir(output_path.parent)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df.to_csv(output_path, index=False)
    return importance_df


def build_risk_segment_analysis(predictions_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Build and export risk segment summary."""
    ensure_dir(output_path.parent)
    risk_order = ["低风险", "中风险", "高风险"]
    counts = predictions_df["risk_level"].value_counts().reindex(risk_order, fill_value=0)
    total = int(len(predictions_df))
    ratio = (counts / total) if total > 0 else counts * 0

    summary_df = pd.DataFrame(
        {
            "risk_level": counts.index,
            "customer_count": counts.values,
            "ratio": ratio.values,
            "ratio_pct": (ratio * 100).round(2),
        }
    )
    summary_df.to_csv(output_path, index=False)
    return summary_df


def export_high_risk_customers(predictions_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Export high-risk customers from prediction output."""
    ensure_dir(output_path.parent)
    required_cols = ["true_label", "pred_label", "churn_probability", "risk_level"]
    for col in required_cols:
        if col not in predictions_df.columns:
            raise KeyError(f"Column '{col}' missing in predictions file.")

    high_risk_df = predictions_df[predictions_df["risk_level"] == "高风险"][required_cols].copy()
    high_risk_df.to_csv(output_path, index=False)
    return high_risk_df


def _feature_business_meaning(feature_name: str) -> str:
    """Generate business interpretation for a feature."""
    f = feature_name.lower()
    if "tenure" in f:
        return "在网时长偏短通常代表关系尚未稳定，新客户更容易流失。"
    if "monthlycharges" in f or "avg_monthly_charge" in f:
        return "月消费较高可能提升价格敏感度，促使客户寻求替代方案。"
    if "contract" in f or "is_month_to_month" in f:
        return "短期合约客户承诺成本低，更容易在不满意时离网。"
    if "techsupport" in f:
        return "缺少技术支持会降低服务黏性，问题得不到及时解决时流失风险上升。"
    if "onlinesecurity" in f:
        return "未开通安全相关服务的客户通常绑定度较低，易受竞品影响。"
    if "onlinebackup" in f or "deviceprotection" in f:
        return "未使用保障类增值服务，说明客户与平台的长期依赖程度较弱。"
    if "internetservice" in f:
        return "网络服务类型与体验、价格结构相关，会影响留存稳定性。"
    if "paymentmethod" in f or "is_auto_payment" in f:
        return "支付方式反映用户支付习惯与续费摩擦，自动支付通常有助于降低流失。"
    if "service_count" in f:
        return "订购服务数量反映客户黏性，服务覆盖越全面通常越不易流失。"
    if "high_charge_low_tenure" in f:
        return "高消费且低在网时长群体处于敏感期，应优先进行早期挽留干预。"
    return "该特征与流失概率存在显著关联，建议纳入持续监控指标。"


def generate_strategy_report(
    project_root: Path,
    best_model_name: str,
    shap_importance_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate markdown retention strategy report."""
    ensure_dir(output_path.parent)

    comparison_path = project_root / "outputs" / "metrics" / "model_comparison.csv"
    metrics_text = "- 暂无模型指标数据"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        if not comparison_df.empty:
            selected = comparison_df[comparison_df["model"] == best_model_name]
            row = selected.iloc[0] if not selected.empty else comparison_df.iloc[0]
            metrics_text = "\n".join(
                [
                    f"- accuracy: {float(row['accuracy']):.4f}",
                    f"- precision: {float(row['precision']):.4f}",
                    f"- recall: {float(row['recall']):.4f}",
                    f"- f1: {float(row['f1']):.4f}",
                    f"- roc_auc: {float(row['roc_auc']):.4f}",
                    f"- pr_auc: {float(row['pr_auc']):.4f}",
                ]
            )

    top10 = shap_importance_df.head(10).copy()
    top_feature_lines = []
    for _, row in top10.iterrows():
        feature = str(row["feature"])
        value = float(row["mean_abs_shap"])
        meaning = _feature_business_meaning(feature)
        top_feature_lines.append(f"- {feature}（mean_abs_shap={value:.6f}）：{meaning}")
    top_feature_text = "\n".join(top_feature_lines) if top_feature_lines else "- 暂无可用特征"

    risk_lines = []
    for _, row in risk_summary_df.iterrows():
        risk_lines.append(
            f"- {row['risk_level']}：{int(row['customer_count'])} 人，占比 {float(row['ratio_pct']):.2f}%"
        )
    risk_text = "\n".join(risk_lines) if risk_lines else "- 暂无风险分层数据"

    report_content = f"""# 客户流失预测业务策略建议

## 1. 模型概况
- 最佳模型名称：{best_model_name}
- 关键指标：
{metrics_text}

## 2. 关键影响因素
- 根据 SHAP 重要性（Top 10）识别的关键特征如下：
{top_feature_text}

## 3. 风险分层结果
{risk_text}
- 高风险客户代表近期有较高流失概率的重点人群，应纳入优先挽留池并做人工跟进。

## 4. 差异化挽留策略建议
### 高风险客户
- 人工客服优先回访
- 定向优惠券 / 套餐折扣
- 推荐长期合约转化
- 推荐技术支持或增值服务包

### 中风险客户
- 自动化营销触达
- 个性化套餐推荐
- 会员权益提醒
- 服务组合捆绑推荐

### 低风险客户
- 维持常规服务
- 做满意度维护
- 引导续约和交叉销售

## 5. 可落地执行建议
- 每周跑一次流失预测模型
- 导出高风险客户名单给运营部门
- 将 churn_probability >= 0.7 客户纳入重点挽留池
- 将 SHAP Top 特征纳入业务监控看板
"""

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report_content)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    _ = load_yaml_config(project_root / "config.yaml")

    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"
    predictions_dir = project_root / "outputs" / "predictions"
    strategy_dir = project_root / "outputs" / "strategy"
    ensure_dir(figures_dir)
    ensure_dir(metrics_dir)
    ensure_dir(predictions_dir)
    ensure_dir(strategy_dir)

    predictions_path = predictions_dir / "customer_churn_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)

    model, best_model_name = load_best_model(project_root)
    X_test, y_test = load_test_data(project_root, target_column="Churn")
    print(f"测试集维度: X={X_test.shape}, y={y_test.shape}")

    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42) if len(X_test) > sample_size else X_test.copy()
    print(f"用于全局 SHAP 分析的样本数: {len(X_sample)}")

    shap_result = compute_shap_values(model, X_sample)
    shap_matrix = shap_result["shap_matrix"]
    shap_module = shap_result.get("shap_module")
    expected_value = float(shap_result.get("expected_value", 0.0))
    print(f"SHAP 计算方式: {shap_result.get('method')}")

    shap_importance_path = metrics_dir / "shap_feature_importance.csv"
    shap_importance_df = export_shap_importance(shap_matrix, X_sample.columns.tolist(), shap_importance_path)

    save_shap_summary_plot(
        shap_matrix=shap_matrix,
        X_sample=X_sample,
        output_path=figures_dir / "shap_summary.png",
        shap_module=shap_module,
    )
    save_shap_bar_plot(shap_importance_df, figures_dir / "shap_bar.png", top_k=15)
    dependence_files = save_top_feature_dependence_plot(
        shap_matrix, X_sample, shap_importance_df, figures_dir
    )

    save_local_explanation_plot(
        model=model,
        X_test=X_test,
        predictions_df=predictions_df,
        output_path=figures_dir / "shap_local_explanation.png",
        expected_value=expected_value,
        shap_module=shap_module,
    )

    risk_summary_path = metrics_dir / "risk_segment_summary.csv"
    risk_summary_df = build_risk_segment_analysis(predictions_df, risk_summary_path)

    high_risk_output_path = predictions_dir / "high_risk_customers.csv"
    high_risk_df = export_high_risk_customers(predictions_df, high_risk_output_path)

    strategy_output_path = strategy_dir / "retention_strategy.md"
    generate_strategy_report(
        project_root=project_root,
        best_model_name=best_model_name,
        shap_importance_df=shap_importance_df,
        risk_summary_df=risk_summary_df,
        output_path=strategy_output_path,
    )

    print("已生成 SHAP 图表:")
    print(f"  {figures_dir / 'shap_summary.png'}")
    print(f"  {figures_dir / 'shap_bar.png'}")
    print(f"  {figures_dir / 'shap_local_explanation.png'}")
    if dependence_files:
        for file_name in dependence_files:
            print(f"  {figures_dir / file_name}")
    else:
        print("  shap_dependence_top1/top2 未生成（兼容性限制）")

    print("已生成文件:")
    print(f"  {shap_importance_path}")
    print(f"  {risk_summary_path}")
    print(f"  {high_risk_output_path}")
    print(f"  {strategy_output_path}")
    print(f"高风险客户数量: {len(high_risk_df)}")


if __name__ == "__main__":
    main()
