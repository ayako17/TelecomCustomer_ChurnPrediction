import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config file and return a dictionary."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path object."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Any, output_path: str | Path) -> None:
    """Save Python object to JSON file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str | Path) -> Any:
    """Load JSON file and return Python object."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)
