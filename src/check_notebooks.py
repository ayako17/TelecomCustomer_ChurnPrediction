import re
from pathlib import Path

import nbformat


def output_to_text(output: dict) -> str:
    """Convert notebook output object to searchable text."""
    parts: list[str] = []

    if "text" in output and isinstance(output["text"], str):
        parts.append(output["text"])

    data = output.get("data")
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.append("".join(v for v in value if isinstance(v, str)))

    traceback = output.get("traceback")
    if isinstance(traceback, list):
        parts.append("\n".join(line for line in traceback if isinstance(line, str)))

    return "\n".join(parts)


def check_notebook(nb_path: Path, q_pattern: re.Pattern[str]) -> tuple[str, list[str]]:
    """Return (status, messages) for a notebook."""
    messages: list[str] = []

    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as error:
        return "WARNING", [f"非法 ipynb 或读取失败: {error}"]

    has_markdown = any(cell.get("cell_type") == "markdown" for cell in nb.cells)
    has_code = any(cell.get("cell_type") == "code" for cell in nb.cells)
    has_outputs = any(
        cell.get("cell_type") == "code" and len(cell.get("outputs", [])) > 0
        for cell in nb.cells
    )

    if not has_markdown:
        messages.append("缺少 Markdown 单元")
    if not has_code:
        messages.append("缺少代码单元")
    if not has_outputs:
        messages.append("缺少执行输出")

    # 源码中连续问号
    for idx, cell in enumerate(nb.cells):
        source = cell.get("source", "")
        match = q_pattern.search(source)
        if match:
            snippet = source[max(0, match.start() - 20) : match.end() + 20].replace("\n", " ")
            messages.append(
                f"检测到连续问号(源码): cell#{idx} type={cell.get('cell_type')} 片段={snippet}"
            )

    # 输出中连续问号
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for out_idx, output in enumerate(cell.get("outputs", [])):
            text = output_to_text(output)
            match = q_pattern.search(text)
            if match:
                snippet = text[max(0, match.start() - 20) : match.end() + 20].replace("\n", " ")
                messages.append(
                    f"检测到连续问号(输出): cell#{idx} output#{out_idx} 片段={snippet}"
                )

    status = "OK" if not messages else "WARNING"
    return status, messages


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebooks = [
        project_root / "notebooks" / "01_data_understanding.ipynb",
        project_root / "notebooks" / "02_preprocessing_and_features.ipynb",
        project_root / "notebooks" / "03_model_comparison.ipynb",
        project_root / "notebooks" / "04_shap_and_strategy.ipynb",
    ]

    q_pattern = re.compile(r"\?{3,}")
    warning_count = 0

    print("开始检查 Notebook 展示质量...")
    for nb_path in notebooks:
        if not nb_path.exists():
            print(f"[WARNING] {nb_path} 不存在")
            warning_count += 1
            continue

        status, messages = check_notebook(nb_path, q_pattern)
        print(f"[{status}] {nb_path}")
        if status != "OK":
            warning_count += 1
            for msg in messages:
                print(f"  - {msg}")

    print("-" * 60)
    if warning_count == 0:
        print("总结果: 通过（所有 Notebook 均满足检查要求）")
    else:
        print(f"总结果: 警告（共有 {warning_count} 个 Notebook 需要关注）")


if __name__ == "__main__":
    main()
