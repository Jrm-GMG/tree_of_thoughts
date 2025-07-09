"""Miscellaneous helper functions for benchmark utilities."""

from pathlib import Path


def ensure_directory_exists(path_str: str) -> None:
    """Ensure all directories for the given path exist.

    Args:
        path_str: File or directory path whose parent directories should be
            created if missing.
    """
    path = Path(path_str)
    dir_path = path.parent if path.suffix else path
    if dir_path and not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

