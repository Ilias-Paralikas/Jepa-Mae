import os
import yaml


def save_config(config: dict, folder: str, filename: str = "config.yaml") -> None:
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"[config_manager] Config saved to: {path}")


def load_config(folder: str, filename: str = "config.yaml") -> dict:
    path = os.path.join(folder, filename)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(f"[config_manager] Config loaded from: {path}")
    return cfg
