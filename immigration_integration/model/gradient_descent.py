# ============================================================
# gradient_descent.py  –  Hand-Coded Gradient Descent Regressor
# ============================================================

import numpy as np
from typing import Dict, Any, Optional, Tuple
from model.base_model import BaseModel


class GradientDescentRegressor(BaseModel):
    """Linear regression trained via hand-coded batch gradient descent."""

    def __init__(self,
                 learning_rate: float = 0.01,
                 iterations: int = 1000,
                 tolerance: float = 1e-6,
                 config: Dict[str, Any] = None):
        super().__init__(config)
        self._lr = learning_rate
        self._iterations = iterations
        self._tolerance = tolerance

        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _normalize(self, X: np.ndarray,
                   y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score normalise features and target."""
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        self._X_std[self._X_std == 0] = 1

        self._y_mean = y.mean()
        self._y_std = y.std() if y.std() > 0 else 1

        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        return X_norm, y_norm

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train weights using batch gradient descent."""
        X_norm, y_norm = self._normalize(X, y)
        n_samples, n_features = X_norm.shape
        self._weights = np.zeros(n_features)
        self._bias = 0.0

        self._training_history['cost'] = []
        prev_cost = float('inf')

        for iteration in range(self._iterations):
            y_pred = np.dot(X_norm, self._weights) + self._bias
            cost = np.mean((y_norm - y_pred) ** 2)
            self._training_history['cost'].append(cost)

            if abs(prev_cost - cost) < self._tolerance:
                break
            prev_cost = cost

            error = y_norm - y_pred
            dw = (-2 / n_samples) * np.dot(X_norm.T, error)
            db = (-2 / n_samples) * np.sum(error)

            self._weights -= self._lr * dw
            self._bias -= self._lr * db

    def _predict_implementation(self, X: np.ndarray) -> np.ndarray:
        """Predict on the original (un-normalised) scale."""
        X_norm = (X - self._X_mean) / self._X_std
        y_norm = np.dot(X_norm, self._weights) + self._bias
        return y_norm * self._y_std + self._y_mean

    def get_params(self) -> Dict[str, Any]:
        """Return the model's hyperparameters."""
        return {
            'learning_rate': self._lr,
            'iterations': self._iterations,
            'tolerance': self._tolerance,
        }

    def get_coefficients(self) -> Dict[str, float]:
        """Return model coefficients mapped back to the original scale."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        weights_original = self._weights * (self._y_std / self._X_std)
        bias_original = self._y_mean - np.dot(weights_original, self._X_mean)

        coeffs: Dict[str, float] = {'intercept': float(bias_original)}
        if self._feature_names:
            for name, weight in zip(self._feature_names, weights_original):
                coeffs[name] = float(weight)

        return coeffs

    def get_training_history(self) -> Dict[str, list]:
        """Return a copy of the training history."""
        return self._training_history.copy()
