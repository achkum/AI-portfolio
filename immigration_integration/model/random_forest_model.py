# ============================================================
# random_forest_model.py  –  Random Forest Wrapper
# ============================================================

import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor as SKRandomForest
from model.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regressor following the BaseModel interface."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 random_state: int = 42,
                 config: Dict[str, Any] = None):
        super().__init__(config)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state

        self._model = SKRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Delegate fitting to scikit-learn."""
        self._model.fit(X, y)

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Delegate prediction to scikit-learn."""
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            'n_estimators': self._n_estimators,
            'max_depth': self._max_depth,
            'random_state': self._random_state,
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        importances = self._model.feature_importances_

        if self._feature_names:
            return dict(zip(self._feature_names, importances))
        return dict(enumerate(importances))
