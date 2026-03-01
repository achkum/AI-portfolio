# ============================================================
# predictor.py  –  High-Level Prediction Interface
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, Any
from model.base_model import BaseModel
from model.evaluator import ModelEvaluator


class IntegrationPredictor:
    """High-level predictor interface (Facade Pattern)."""

    def __init__(self, model: BaseModel, evaluator: ModelEvaluator = None):
        self._model = model
        self._evaluator = evaluator or ModelEvaluator(model)

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2) -> Dict[str, Any]:
        """Complete training and evaluation pipeline."""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )

        self._model.fit(X_train, y_train)

        train_metrics = self._evaluator.evaluate(X_train, y_train)
        test_metrics  = self._evaluator.evaluate(X_test, y_test)

        results = {
            'train_metrics': train_metrics,
            'test_metrics':  test_metrics,
            'model_params':  self._model.get_params(),
            'n_train':       len(y_train),
            'n_test':        len(y_test),
        }

        return results

    def predict_for_country(self, country_data: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for a specific country's data."""
        return self._model.predict(country_data)
