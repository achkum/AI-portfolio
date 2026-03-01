# ============================================================
# base_loader.py  –  Abstract Base Class for All Data Loaders
# ============================================================

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict, Any


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""

    def __init__(self, filepath: str, config: Dict[str, Any]):
        self._filepath = filepath
        self._config = config
        self._data: Optional[pd.DataFrame] = None
        self._is_loaded: bool = False

    @abstractmethod
    def _read_file(self) -> pd.DataFrame:
        """Read the raw file from disk and return a DataFrame."""
        pass

    @abstractmethod
    def _validate_schema(self, df: pd.DataFrame) -> bool:
        """Return True if *df* contains all required columns."""
        pass

    @abstractmethod
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply source-specific cleaning and transformations."""
        pass

    def load(self) -> 'BaseDataLoader':
        """Template method that defines the loading pipeline."""
        raw_data = self._read_file()

        if not self._validate_schema(raw_data):
            raise ValueError(f"Schema validation failed for {self._filepath}")

        cleaned_data = self._clean_common(raw_data)
        self._data = self._apply_transformations(cleaned_data)
        self._is_loaded = True

        return self

    def _clean_common(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace common missing-value indicators with ``pd.NA``."""
        df = df.replace(['..', '-', 'N/A', ''], pd.NA)
        return df

    @property
    def data(self) -> pd.DataFrame:
        """Read-only access to the loaded data."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._data.copy()

    @property
    def is_loaded(self) -> bool:
        """Whether the data has been successfully loaded."""
        return self._is_loaded

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}(filepath='{self._filepath}', status={status})"
