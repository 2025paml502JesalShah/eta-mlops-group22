from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_params():
    """Load parameters from params.yaml."""
    with open(PROJECT_ROOT / "params.yaml", "r") as file:
        return yaml.safe_load(file)