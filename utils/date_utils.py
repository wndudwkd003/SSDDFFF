# utils/data_utils.py

from datetime import datetime


def get_current_timestamp() -> str:
    """Get the current timestamp as a string in the format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
