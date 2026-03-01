# ============================================================
# preprocessor.py  –  Data Cleaning & Imputation
# ============================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataPreprocessor:
    """Cleans and prepares data for modeling."""

    def __init__(self, data: pd.DataFrame):
        self._data = data.copy()

    def fill_missing(self, strategy: str = 'mean', 
                     exclude_cols: List[str] = None) -> 'DataPreprocessor':
        """
        Fill numeric NaN values. 
        IMPORTANT: Skip outcomes where broad imputation hides variance.
        """
        exclude_cols = exclude_cols or []
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in exclude_cols:
                continue
            
            if strategy == 'mean':
                val = self._data[col].mean()
            elif strategy == 'median':
                val = self._data[col].median()
            else:
                val = 0
            
            # For specific outcome columns, don't fill with global median if 
            # we want to see variance between regions.
            # Instead, we fill with 0 or keep NaN to filter out in plots.
            self._data[col] = self._data[col].fillna(val)
        
        return self

    def normalize_features(self, columns: List[str]) -> 'DataPreprocessor':
        """Scale specific numeric columns to [0, 1] range."""
        for col in columns:
            if col in self._data.columns:
                c_min = self._data[col].min()
                c_max = self._data[col].max()
                if c_max > c_min:
                    self._data[col] = (self._data[col] - c_min) / (c_max - c_min)
        return self

    def handle_outliers(self, column: str, 
                        threshold: float = 3.0) -> 'DataPreprocessor':
        """Caps outliers based on Z-score to prevent model bias."""
        if column in self._data.columns:
            mean = self._data[column].mean()
            std = self._data[column].std()
            if std > 0:
                upper = mean + threshold * std
                lower = mean - threshold * std
                self._data[column] = self._data[column].clip(lower, upper)
        return self

    @property
    def data(self) -> pd.DataFrame:
        """Returns the processed DataFrame."""
        return self._data
