# ============================================================
# base_model.py  –  Abstract Base Model Class
# ============================================================

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all predictive models."""

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._is_fitted: bool = False
        self._feature_names: Optional[list] = None
        self._training_history: Dict[str, list] = {}

    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Core fitting logic."""
        pass

    @abstractmethod
    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Core prediction logic."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return a dictionary of hyperparameters."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Public fitting method."""
        self._feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        X_array = np.array(X, dtype=float)
        y_array = np.array(y, dtype=float)

        self._validate_input(X_array, y_array)
        self._fit_implementation(X_array, y_array)
        self._is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_array = np.array(X, dtype=float)
        return self._predict_implementation(X_array)

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """Check that X and y are compatible."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input contains NaN values")

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def feature_names(self) -> Optional[list]:
        """Feature names recorded during fitting."""
        return self._feature_names

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(status={status})"
