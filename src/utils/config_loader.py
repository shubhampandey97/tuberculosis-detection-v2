import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")


def load_config(path=None):
    """Load the project config.

    Order: explicit `path`, then the TB_CONFIG environment variable (e.g. src/config/config_v10_arm2.yaml),
    then src/config/config.yaml (the goal 1-9 setup, unchanged).
    """
    path = path or os.environ.get("TB_CONFIG") or DEFAULT_CONFIG
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("tag", "default")
    return config
