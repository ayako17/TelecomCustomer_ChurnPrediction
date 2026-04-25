from __future__ import annotations

import re
from pathlib import Path

import nbformat

from utils import ensure_dir


def check_file_group(project_root: Path, files: list[str], group_name: str) -> tuple[list[str], list[str]]:
    """Return (report_lines, missing_files) for one file group."""
    lines = [f"[{group_name}]"]
    missing: list[str] = []
    for rel_path in files:
        if (project_root / rel_path).exists():
            lines.append(f"[OK] {rel_path}")
        else:
            lines.append(f"[MISSING] {rel_path}")
            missing.append(rel_path)
    return lines, missing


def notebook_has_q3_or_error(nb_path: Path, q_pattern: re.Pattern[str]) -> tuple[bool, list[str]]:
    """Check notebook source/output for ??? and error outputs."""
    issues: list[str] = []
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as error:
        return True, [f"notebook unreadable: {error}"]

    for idx, cell in enumerate(nb.cells):
        source = cell.get("source", "")
        if q_pattern.search(source):
            issues.append(f"cell#{idx} source contains ???")

        if cell.get("cell_type") == "code":
            for out_idx, output in enumerate(cell.get("outputs", [])):
                if output.get("output_type") == "error":
                    issues.append(f"cell#{idx} output#{out_idx} has error output")
                text_parts: list[str] = []
                if isinstance(output.get("text"), str):
                    text_parts.append(output["text"])
                data = output.get("data")
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            text_parts.append("".join(v for v in value if isinstance(v, str)))
                if q_pattern.search("\n".join(text_parts)):
                    issues.append(f"cell#{idx} output#{out_idx} contains ???")

    return bool(issues), issues


def check_requirements_versioned(requirements_path: Path) -> tuple[bool, list[str]]:
    """Check whether each non-comment requirement line includes a version comparator."""
    pattern = re.compile(r"^[A-Za-z0-9_.-]+\s*(==|>=|<=|~=|>|<)\s*.+$")
    bad_lines: list[str] = []

    if not requirements_path.exists():
        return False, ["requirements.txt not found"]

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not pattern.match(line):
            bad_lines.append(line)

    return len(bad_lines) == 0, bad_lines


def scan_hardcoded_path(project_root: Path, target_pattern: str) -> list[str]:
    """Scan source/docs/config markdown text files for hardcoded absolute path."""
    candidates: list[Path] = []
    for root_rel in ["src", "docs"]:
        root_dir = project_root / root_rel
        if root_dir.exists():
            candidates.extend([p for p in root_dir.rglob("*") if p.is_file()])
    for single_file in ["README.md", "config.yaml", "requirements.txt"]:
        f = project_root / single_file
        if f.exists():
            candidates.append(f)

    hits: list[str] = []
    for file_path in candidates:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if target_pattern in text:
            hits.append(str(file_path.relative_to(project_root)))
    return hits


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report_lines: list[str] = []
    required_missing: list[str] = []
    optional_missing: list[str] = []
    warnings: list[str] = []

    # 1~8: required file completeness
    data_raw_files = ["data/raw/Telco-Customer-Churn.csv"]
    data_processed_files = [
        "data/processed/churn_cleaned.csv",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "data/processed/feature_columns.json",
    ]
    model_files = [
        "models/logistic_regression.pkl",
        "models/random_forest.pkl",
        "models/xgboost_model.pkl",
        "models/lightgbm_model.pkl",
        "models/best_model.pkl",
    ]
    metric_files = [
        "outputs/metrics/model_comparison.csv",
        "outputs/metrics/classification_reports.txt",
        "outputs/metrics/threshold_analysis.csv",
        "outputs/metrics/shap_feature_importance.csv",
        "outputs/metrics/risk_segment_summary.csv",
        "outputs/metrics/cost_sensitive_analysis.csv",
    ]
    figure_files = [
        "outputs/figures/model_comparison.png",
        "outputs/figures/roc_curve.png",
        "outputs/figures/pr_curve.png",
        "outputs/figures/confusion_matrix_best_model.png",
        "outputs/figures/feature_importance.png",
        "outputs/figures/threshold_analysis.png",
        "outputs/figures/shap_summary.png",
        "outputs/figures/shap_bar.png",
        "outputs/figures/shap_dependence_top1.png",
        "outputs/figures/shap_dependence_top2.png",
        "outputs/figures/shap_local_explanation.png",
        "outputs/figures/eda_churn_distribution.png",
        "outputs/figures/eda_numeric_distribution.png",
        "outputs/figures/eda_contract_churn_rate.png",
        "outputs/figures/eda_payment_churn_rate.png",
        "outputs/figures/eda_internet_churn_rate.png",
        "outputs/figures/eda_correlation_heatmap.png",
        "outputs/figures/preprocess_train_test_churn_distribution.png",
        "outputs/figures/feature_engineering_business_features.png",
        "outputs/figures/cost_sensitive_analysis.png",
    ]
    prediction_files = [
        "outputs/predictions/customer_churn_predictions.csv",
        "outputs/predictions/high_risk_customers.csv",
    ]
    strategy_files = [
        "outputs/strategy/retention_strategy.md",
        "outputs/strategy/cost_benefit_strategy.md",
        "outputs/strategy/risk_threshold_rationale.md",
    ]
    notebook_files = [
        "notebooks/01_data_understanding.ipynb",
        "notebooks/02_preprocessing_and_features.ipynb",
        "notebooks/03_model_comparison.ipynb",
        "notebooks/04_shap_and_strategy.ipynb",
    ]
    docs_files = [
        "docs/run_guide.md",
        "docs/project_checklist.md",
        "docs/environment_versions.txt",
        "docs/data_dictionary.md",
        "docs/data_quality_report.md",
        "docs/leakage_risk_and_pipeline_improvement.md",
    ]
    script_files = [
        "src/cost_sensitive_analysis.py",
        "src/final_submission_check.py",
        "src/check_outputs.py",
        "src/check_notebooks.py",
    ]
    optional_files = [
        "outputs/metrics/cross_validation_results.csv",
        "outputs/figures/cross_validation_auc.png",
    ]

    groups = [
        ("DATA_RAW", data_raw_files),
        ("DATA_PROCESSED", data_processed_files),
        ("MODELS", model_files),
        ("OUTPUTS_METRICS", metric_files),
        ("OUTPUTS_FIGURES", figure_files),
        ("OUTPUTS_PREDICTIONS", prediction_files),
        ("OUTPUTS_STRATEGY", strategy_files),
        ("NOTEBOOKS", notebook_files),
        ("DOCS", docs_files),
        ("SCRIPTS", script_files),
    ]

    report_lines.append("Final Submission Consistency Check")
    report_lines.append("=" * 80)
    for group_name, files in groups:
        lines, missing = check_file_group(project_root, files, group_name)
        report_lines.extend(lines)
        required_missing.extend(missing)
        report_lines.append("")

    # optional completeness
    optional_lines, missing_opt = check_file_group(project_root, optional_files, "OPTIONAL")
    report_lines.extend(optional_lines)
    optional_missing.extend(missing_opt)
    report_lines.append("")

    # 9 & 10 notebooks quality
    q_pattern = re.compile(r"\?{3,}")
    report_lines.append("[NOTEBOOK_QUALITY]")
    for rel in notebook_files:
        nb_path = project_root / rel
        bad, issues = notebook_has_q3_or_error(nb_path, q_pattern)
        if not bad:
            report_lines.append(f"[OK] {rel}")
        else:
            report_lines.append(f"[WARNING] {rel}")
            for issue in issues:
                report_lines.append(f"  - {issue}")
            warnings.extend([f"{rel}: {issue}" for issue in issues])
    report_lines.append("")

    # 11 requirements versioning
    ok_version, bad_lines = check_requirements_versioned(project_root / "requirements.txt")
    report_lines.append("[REQUIREMENTS_VERSION]")
    if ok_version:
        report_lines.append("[OK] requirements.txt has version specifiers on all package lines")
    else:
        report_lines.append("[WARNING] requirements.txt has package lines without version specifier")
        for item in bad_lines:
            report_lines.append(f"  - {item}")
        warnings.extend([f"requirements without version: {item}" for item in bad_lines])
    report_lines.append("")

    # 13 __pycache__
    pycache_dirs = [p for p in project_root.rglob("__pycache__") if p.is_dir()]
    report_lines.append("[PYCACHE]")
    if pycache_dirs:
        report_lines.append(f"[WARNING] found __pycache__ dirs: {len(pycache_dirs)}")
        warnings.append(f"__pycache__ count={len(pycache_dirs)}")
    else:
        report_lines.append("[OK] no __pycache__ directory found")
    report_lines.append("")

    # 14 .ipynb_checkpoints
    checkpoint_dirs = [p for p in project_root.rglob(".ipynb_checkpoints") if p.is_dir()]
    report_lines.append("[IPYNB_CHECKPOINTS]")
    if checkpoint_dirs:
        report_lines.append(f"[WARNING] found .ipynb_checkpoints dirs: {len(checkpoint_dirs)}")
        warnings.append(f".ipynb_checkpoints count={len(checkpoint_dirs)}")
    else:
        report_lines.append("[OK] no .ipynb_checkpoints directory found")
    report_lines.append("")

    # 15 temp files
    temp_patterns = ["*.tmp", "*.temp", "*.bak", "~$*"]
    temp_hits: list[Path] = []
    for pattern in temp_patterns:
        temp_hits.extend([p for p in project_root.rglob(pattern) if p.is_file()])
    report_lines.append("[TEMP_FILES]")
    if temp_hits:
        report_lines.append(f"[WARNING] temp-like files found: {len(temp_hits)}")
        for item in temp_hits[:20]:
            report_lines.append(f"  - {item.relative_to(project_root)}")
        warnings.append(f"temp file count={len(temp_hits)}")
    else:
        report_lines.append("[OK] no temp-like files found")
    report_lines.append("")

    # 16 hardcoded absolute path
    hardcoded_hits = scan_hardcoded_path(project_root, "E:\\TelecomCustomer_ChurnPrediction")
    report_lines.append("[HARDCODED_PATH_SCAN]")
    if hardcoded_hits:
        report_lines.append(f"[WARNING] hardcoded absolute path found in {len(hardcoded_hits)} files")
        for item in hardcoded_hits:
            report_lines.append(f"  - {item}")
        warnings.append(f"hardcoded path hits={len(hardcoded_hits)}")
    else:
        report_lines.append("[OK] no hardcoded absolute path found in scanned files")
    report_lines.append("")

    # overall status
    if required_missing:
        status = "FAILED"
    elif warnings or optional_missing:
        status = "WARNING"
    else:
        status = "PASSED"

    report_lines.append("=" * 80)
    report_lines.append(f"required files missing: {len(required_missing)}")
    report_lines.append(f"optional files missing: {len(optional_missing)}")
    report_lines.append(f"warnings: {len(warnings)}")
    report_lines.append(f"overall status: {status}")

    report_path = project_root / "outputs" / "metrics" / "final_submission_check_report.txt"
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"required files missing: {len(required_missing)}")
    print(f"optional files missing: {len(optional_missing)}")
    print(f"warnings: {len(warnings)}")
    print(f"overall status: {status}")
    print(f"report saved to: {report_path}")


if __name__ == "__main__":
    main()
