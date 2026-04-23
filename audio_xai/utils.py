from pathlib import Path

from audio_xai.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, PROJ_ROOT, REPORTS_DIR


def get_project_paths_status() -> list[tuple[str, Path, bool]]:
    """Return key project paths together with their existence status."""
    project_paths = {
        "project_root": PROJ_ROOT,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "reports": REPORTS_DIR,
        "figures": FIGURES_DIR,
    }
    return [(name, path, path.exists()) for name, path in project_paths.items()]


if __name__ == "__main__":
    for name, path, exists in get_project_paths_status():
        print(f"{name}: {path} - {'Exists' if exists else 'Missing'}")
