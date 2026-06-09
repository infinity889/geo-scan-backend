from pathlib import Path, PureWindowsPath


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"


def safe_filename(file_name: str) -> str:
    return Path(PureWindowsPath(file_name).name).name


def detect_format(file_name: str) -> str:
    suffix = Path(safe_filename(file_name)).suffix.replace(".", "").upper()
    return suffix or "FILE"
