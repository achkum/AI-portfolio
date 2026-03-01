# ============================================================
# utils.py  –  Helper Functions
# ============================================================
# Miscellaneous utility functions shared across all modules,
# such as directory creation and safe file reading.
# ============================================================

import os
import pandas as pd
from typing import Optional


def ensure_directory(path: str) -> str:
    """Create a directory (including parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_read_csv(filepath: str,
                  encoding: str = "utf-8-sig",
                  sep: str = ",",
                  **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV file with error handling."""
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath, encoding=encoding, sep=sep, **kwargs)
        return None
    except Exception:
        return None


def safe_read_excel(filepath: str,
                    sheet_name=0,
                    **kwargs) -> Optional[pd.DataFrame]:
    """Read an Excel file with error handling."""
    try:
        if os.path.exists(filepath):
            return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        return None
    except Exception:
        return None
