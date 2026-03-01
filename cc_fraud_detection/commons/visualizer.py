import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from pathlib import Path

class ModelVisualizer:
    def __init__(self, figures_path="figures/"):
        self.figures_path = Path(figures_path)
        self.figures_path.mkdir(parents=True, exist_ok=True)

    def plot_pr_curves(self, y_true, model_scores_dict, filename='pr_curves.png'):
        plt.figure(figsize=(10, 7))
        for model_name, y_scores in model_scores_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            auprc = auc(recall, precision)
            plt.plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.4f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.savefig(self.figures_path / filename)
        plt.close()

    def plot_roc_curves(self, y_true, model_scores_dict, filename='roc_curves.png'):
        plt.figure(figsize=(10, 7))
        for model_name, y_scores in model_scores_dict.items():
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(self.figures_path / filename)
        plt.close()

    def plot_error_distribution(self, common_y_scores, fraud_y_scores, model_name='AE', filename='error_distribution.png'):
        plt.figure(figsize=(10, 6))
        sns.histplot(common_y_scores, bins=50, cumulative=False, color='blue', label='Legitimate', kde=True)
        sns.histplot(fraud_y_scores, bins=50, cumulative=False, color='red', label='Fraud', kde=True)
        plt.title(f'Reconstruction Error Distribution - {model_name}')
        plt.yscale('log')
        plt.legend()
        plt.savefig(self.figures_path / filename)
        plt.close()
