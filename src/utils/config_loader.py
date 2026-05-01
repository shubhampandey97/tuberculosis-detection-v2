import yaml
import os

def load_config():
    base_path = os.path.dirname(os.path.dirname(__file__))  # points to src/
    config_path = os.path.join(base_path, "config", "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)