import pathlib

import yaml
from ml_collections import ConfigDict


def get_config(config_string: str = None):
    # Load the yaml config file as a dict and use it to build a ConfigDict
    if config_string is None:
        config_string = "default_config"
    cfg = yaml.safe_load(pathlib.Path(f"cfg/{config_string}.yaml").read_text())
    cfg = ConfigDict(cfg)
    return cfg
