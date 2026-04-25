import json
from pathlib import Path
from typing import Any

import joblib

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    yaml = None


def _coerce_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _simple_yaml_load(content: str) -> dict:
    """
    轻量 YAML 解析兜底，仅支持当前项目使用的 key-value 与层级缩进格式。
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in content.splitlines():
        line_no_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_no_comment.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line_no_comment.strip()
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            container: dict[str, Any] = {}
            parent[key] = container
            stack.append((indent, container))
        else:
            parent[key] = _coerce_yaml_scalar(value)

    return root


def load_yaml_config(config_path: str | Path = "config.yaml") -> dict:
    """读取 YAML 配置文件并返回字典。"""
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as file:
        content = file.read()

    if yaml is not None:
        loaded = yaml.safe_load(content)
        return loaded if loaded is not None else {}

    return _simple_yaml_load(content)


def ensure_dir(path: str | Path) -> Path:
    """如果目录不存在则创建，并返回 Path 对象。"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(obj: Any, path: str | Path) -> None:
    """将对象保存为 JSON 文件。"""
    output_file = Path(path)
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件并返回对象。"""
    input_file = Path(path)
    with input_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save object to pickle file via joblib."""
    output_file = Path(path)
    ensure_dir(output_file.parent)
    joblib.dump(obj, output_file)


def load_pickle(path: str | Path) -> Any:
    """Load object from pickle file via joblib."""
    input_file = Path(path)
    return joblib.load(input_file)
