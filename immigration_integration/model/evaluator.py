# ============================================================
# evaluator.py  –  Model Evaluation Metrics
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model.base_model import BaseModel


class ModelEvaluator:
    """Evaluator that wraps a BaseModel and computes regression metrics."""

    def __init__(self, model: BaseModel):
        self._model = model

    def evaluate(self, X, y_true) -> Dict[str, float]:
        """Compute R², MAE, RMSE, and MAPE."""
        y_pred = self._model.predict(X)
        y_true = np.array(y_true, dtype=float)

        metrics = {
            'r2':   r2_score(y_true, y_pred),
            'mae':  mean_absolute_error(y_true, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100),
        }
        return metrics

    def cross_validate(self, X, y, k_folds: int = 5) -> Dict[str, list]:
        """K-fold cross-validation."""
        from sklearn.model_selection import KFold

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        results: Dict[str, list] = {'r2': [], 'mae': [], 'rmse': []}

        for train_idx, val_idx in kf.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            model_class = type(self._model)
            fold_model = model_class(**self._model.get_params())
            fold_model.fit(pd.DataFrame(X_train), pd.Series(y_train))

            y_pred = fold_model.predict(pd.DataFrame(X_val))

            results['r2'].append(r2_score(y_val, y_pred))
            results['mae'].append(mean_absolute_error(y_val, y_pred))
            results['rmse'].append(float(np.sqrt(mean_squared_error(y_val, y_pred))))

        return results
