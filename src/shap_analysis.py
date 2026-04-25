from pathlib import Path

import matplotlib.pyplot as plt

from utils import ensure_dir, load_yaml_config


def run_shap_analysis(output_path: str | Path) -> None:
    """
    Placeholder for SHAP analysis.
    Saves a placeholder summary image for workflow verification.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "SHAP summary will be generated here.",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / "config.yaml")

    figure_dir = project_root / config["paths"]["figure_dir"]
    output_file = figure_dir / "shap_summary.png"

    run_shap_analysis(output_file)
    print(f"SHAP summary figure saved to: {output_file}")


if __name__ == "__main__":
    main()
