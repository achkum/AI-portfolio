import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy import stats
from pathlib import Path

class ModelVisualizer:
    def __init__(self, figures_path):
        self.figures_path = Path(figures_path)
        self.figures_path.mkdir(parents=True, exist_ok=True)

    def _save(self, filename):
        plt.tight_layout()
        plt.savefig(self.figures_path / filename)
        plt.close()

    def plot_pr_curves(self, y_true, model_scores_dict, filename='pr_curves.png'):
        plt.figure(figsize=(10, 7))
        for name, scores in model_scores_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            plt.plot(recall, precision, label=f'{name} (AUPRC={auc(recall, precision):.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        self._save(filename)

    def plot_roc_curves(self, y_true, model_scores_dict, filename='roc_curves.png'):
        plt.figure(figsize=(10, 7))
        for name, scores in model_scores_dict.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        self._save(filename)

    def plot_error_distribution(self, legit_scores, fraud_scores, model_name='AE', filename='error_dist.png'):
        plt.figure(figsize=(10, 6))
        sns.histplot(legit_scores, bins=50, color='blue', label='Legitimate', kde=True)
        sns.histplot(fraud_scores, bins=50, color='red', label='Fraud', kde=True)
        plt.title(f'Reconstruction Error Distribution - {model_name}')
        plt.yscale('log')
        plt.legend()
        self._save(filename)

    def plot_class_balance(self, df, filename='class_balance.png'):
        plt.figure(figsize=(8, 5))
        counts = df['Class'].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.yscale('log')
        self._save(filename)

    def plot_feature_distributions(self, df, feature='Amount', filename='feature_dist.png'):
        plt.figure(figsize=(10, 6))
        
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], ['steelblue', 'tomato']):
            data = df[df['Class'] == cls][feature]
            sns.kdeplot(data, color=color, fill=True, label=f'{label} (n={len(data):,})', alpha=0.3)
            
        plt.title('Transaction Amount Distribution: Legitimate vs Fraud', fontsize=13)
        plt.xlabel('Amount (EUR)')
        plt.ylabel('Density')
        plt.legend()
        self._save(filename)

    def plot_qq(self, df, filename='qq_plot.png'):
        plt.figure(figsize=(10, 7))
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], ['steelblue', 'tomato']):
            data = df[df['Class'] == cls]['Amount']
            (osm, osr), (slope, intercept, _) = stats.probplot(data, dist='norm')
            plt.scatter(osm, osr, s=5, color=color, alpha=0.4, label=f'{label} (n={len(data):,})')
            plt.plot(osm, slope * np.array(osm) + intercept, color=color, linestyle='--', alpha=0.6)
            
        plt.title('Q-Q Plot: Transaction Amount (Before Scaling)', fontsize=13)
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.legend()
        self._save(filename)

    def plot_scaling_effect(self, train_df, filename='scaling_effect.png'):
        plt.figure(figsize=(10, 6))
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], ['steelblue', 'tomato']):
            data = train_df[train_df['Class'] == cls]['Amount']
            sns.kdeplot(data, color=color, fill=True, label=f'{label}', alpha=0.3)
            plt.axvline(data.mean(), color=color, linestyle='--', linewidth=1)
            
        plt.title('Standardised Transaction Amount Distribution (Post-Scaling)', fontsize=13)
        plt.xlabel('Amount (z-score)')
        plt.ylabel('Density')
        plt.legend()
        self._save(filename)

    def plot_scaled_qq(self, train_df, filename='scaled_qq.png'):
        plt.figure(figsize=(10, 7))
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], ['steelblue', 'tomato']):
            data = train_df[train_df['Class'] == cls]['Amount']
            (osm, osr), (slope, intercept, _) = stats.probplot(data, dist='norm')
            plt.scatter(osm, osr, s=5, color=color, alpha=0.4, label=f'{label}')
            plt.plot(osm, slope * np.array(osm) + intercept, color=color, linestyle='--', alpha=0.6)
            
        plt.title('Q-Q Plot: Standardised Amount', fontsize=13)
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.legend()
        self._save(filename)
