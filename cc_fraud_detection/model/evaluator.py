import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix

class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_auprc(self, y_true, y_scores):
        """Area Under Precision-Recall Curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

    def calculate_recall_at_fpr(self, y_true, y_scores, fpr_threshold=0.01):
        """Calculate Recall at a specific False Positive Rate."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Find the threshold closest to the target FPR
        idx = np.argmin(np.abs(fpr - fpr_threshold))
        return tpr[idx], thresholds[idx]

    def evaluate_all(self, y_true, model_scores_dict, fpr_thresholds=[0.001, 0.005, 0.01]):
        """Evaluate multiple models and return a metrics summary."""
        results = []
        for model_name, y_scores in model_scores_dict.items():
             auprc = self.calculate_auprc(y_true, y_scores)
             model_results = {'Model': model_name, 'AUPRC': auprc}

             for fpr_t in fpr_thresholds:
                  recall_at_fpr, _ = self.calculate_recall_at_fpr(y_true, y_scores, fpr_t)
                  model_results[f'Recall@{fpr_t*100}% FPR'] = recall_at_fpr

             results.append(model_results)
        return pd.DataFrame(results)
