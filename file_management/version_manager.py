import os


def get_version_folder(base_dir, run_name, load_run=None):
    run_dir = os.path.join(base_dir, run_name)

    if load_run is not None:
        folder = os.path.join(run_dir, f"v{load_run}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Version folder not found: {folder}")
        print(f"[version_manager] Resuming from: {folder}")
        return folder

    os.makedirs(run_dir, exist_ok=True)
    existing = [d for d in os.listdir(run_dir) if d.startswith("v") and d[1:].isdigit()]
    next_v = max((int(d[1:]) for d in existing), default=0) + 1
    folder = os.path.join(run_dir, f"v{next_v}")
    os.makedirs(folder, exist_ok=True)
    print(f"[version_manager] New run folder: {folder}")
    return folder
