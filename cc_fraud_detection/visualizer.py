import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from pathlib import Path

# --------------------------------------
# Colour-blind-friendly palette
# --------------------------------------
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_RED = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_CYAN = '#56B4E9'
CB_TEAL = '#44AA99'
CB_GRAY = '#999999'

# Class colours used across all EDA plots
CLS_LEGIT = CB_BLUE
CLS_FRAUD = CB_RED


class ModelVisualizer:
    def __init__(self, figures_path):
        self.figures_path = Path(figures_path)
        self.figures_path.mkdir(parents=True, exist_ok=True)
        # Apply global style settings
        sns.set_style('whitegrid')
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
        })

    def _save(self, filename):
        plt.tight_layout()
        plt.savefig(self.figures_path / filename, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # EDA plots
    # ------------------------------------------------------------------
    def plot_class_balance(self, df, filename='class_balance.png'):
        plt.figure(figsize=(7, 5))
        counts = df['Class'].value_counts().sort_index()
        bars = plt.bar(
            ['Legitimate (0)', 'Fraud (1)'], counts.values,
            color=[CLS_LEGIT, CLS_FRAUD], edgecolor='black', linewidth=0.5
        )
        # Annotate counts on bars
        for bar, count in zip(bars, counts.values):
            if count > 1000:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.6,
                         f'n = {count:,}', ha='center', va='center',
                         fontweight='bold', fontsize=12, color='white')
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                         f'n = {count:,}', ha='center', va='bottom',
                         fontweight='bold', fontsize=12, color='black')
        plt.title('Class Distribution (Logarithmic Scale)')
        plt.ylabel('Count')
        plt.yscale('log')
        self._save(filename)

    def plot_feature_distributions(self, df, feature='Amount', filename='feature_dist.png'):
        plt.figure(figsize=(10, 6))
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], [CLS_LEGIT, CLS_FRAUD]):
            data = df[df['Class'] == cls][feature]
            sns.kdeplot(data, color=color, fill=True,
                        label=f'{label} (n={len(data):,})', alpha=0.3)
        plt.title('Transaction Amount Distribution: Legitimate vs Fraud')
        plt.xlabel('Amount (EUR)')
        plt.ylabel('Density')
        plt.legend()
        self._save(filename)

    def plot_feature_boxplots(self, df, features=None, filename='feature_boxplots.png'):
        """Box plots for the most discriminative V-features by class."""
        if features is None:
            features = ['V17', 'V14', 'V12', 'V10']
        n = len(features)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, feat in zip(axes, features):
            data_legit = df[df['Class'] == 0][feat]
            data_fraud = df[df['Class'] == 1][feat]
            bp = ax.boxplot(
                [data_legit, data_fraud],
                tick_labels=['Legit', 'Fraud'],
                patch_artist=True,
                widths=0.5,
                showfliers=False,
            )
            bp['boxes'][0].set_facecolor(CLS_LEGIT)
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor(CLS_FRAUD)
            bp['boxes'][1].set_alpha(0.6)
            ax.set_title(feat, fontweight='bold')
            ax.set_ylabel('Value')
        fig.suptitle('Feature Distributions by Class (Outliers Hidden)', fontsize=13, y=1.02)
        self._save(filename)

    def plot_correlation_heatmap(self, df, features=None, filename='correlation_heatmap.png'):
        """Heatmap of feature correlations (subset for readability)."""
        if features is None:
            features = ['V1', 'V3', 'V7', 'V10', 'V12', 'V14', 'V17', 'Amount']
        corr = df[features].corr()
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5,
            cbar_kws={'label': 'Pearson r'}
        )
        plt.title('Feature Correlation Matrix (Selected Features)')
        self._save(filename)

    def plot_scaling_effect(self, train_df, filename='scaling_effect.png'):
        plt.figure(figsize=(10, 6))
        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], [CLS_LEGIT, CLS_FRAUD]):
            data = train_df[train_df['Class'] == cls]['Amount']
            sns.kdeplot(data, color=color, fill=True, label=f'{label}', alpha=0.3)
            plt.axvline(data.mean(), color=color, linestyle='--', linewidth=1)
        plt.title('Standardised Transaction Amount Distribution (Post-Scaling)')
        plt.xlabel('Amount (z-score)')
        plt.ylabel('Density')
        plt.legend()
        self._save(filename)

    # ------------------------------------------------------------------
    # Model evaluation plots
    # ------------------------------------------------------------------
    def plot_pr_curves(self, y_true, model_scores_dict, filename='pr_curves.png'):
        colors = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_TEAL, CB_GRAY]
        plt.figure(figsize=(10, 7))
        for (name, scores), color in zip(model_scores_dict.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = auc(recall, precision)
            plt.plot(recall, precision, color=color,
                     label=f'{name} (AUPRC={ap:.4f})', linewidth=1.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='upper right', fontsize=9)
        plt.xlim([0, 1.02])
        plt.ylim([0, 1.05])
        self._save(filename)

    def plot_roc_curves(self, y_true, model_scores_dict,
                        fpr_thresholds=(0.001, 0.005, 0.01),
                        filename='roc_curves.png'):
        colors = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_TEAL, CB_GRAY]
        markers = ['o', 's', 'D']
        plt.figure(figsize=(10, 7))
        for (name, scores), color in zip(model_scores_dict.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color,
                     label=f'{name} (AUC={roc_auc:.4f})', linewidth=1.5)
            for fpr_t, marker in zip(fpr_thresholds, markers):
                valid = np.where(fpr <= fpr_t)[0]
                if len(valid) > 0:
                    idx = valid[-1]
                    plt.plot(fpr[idx], tpr[idx], marker=marker, color=color,
                             markersize=7, zorder=5)
        for fpr_t, marker in zip(fpr_thresholds, markers):
            plt.plot([], [], marker=marker, color='gray', linestyle='None',
                     markersize=7, label=f'FPR={fpr_t*100:.1f}%')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves with Operating Points')
        plt.legend(loc='lower right', fontsize=8)
        self._save(filename)

    def plot_error_distribution(self, legit_scores, fraud_scores,
                                model_name='AE', filename='error_dist.png'):
        """Fixed error distribution plot: separate histograms without broken log-KDE."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: linear scale histogram
        upper = np.percentile(np.concatenate([legit_scores, fraud_scores]), 99)
        bins = np.linspace(0, upper, 60)
        ax1.hist(legit_scores, bins=bins, color=CLS_LEGIT, alpha=0.6,
                 label=f'Legitimate (n={len(legit_scores):,})', density=True)
        ax1.hist(fraud_scores, bins=bins, color=CLS_FRAUD, alpha=0.6,
                 label=f'Fraud (n={len(fraud_scores):,})', density=True)
        ax1.set_title(f'Reconstruction Error — {model_name} (Linear)')
        ax1.set_xlabel('Reconstruction Error (MSE)')
        ax1.set_ylabel('Density')
        ax1.legend()

        # Right panel: log scale histogram
        ax2.hist(legit_scores, bins=bins, color=CLS_LEGIT, alpha=0.6,
                 label='Legitimate')
        ax2.hist(fraud_scores, bins=bins, color=CLS_FRAUD, alpha=0.6,
                 label='Fraud')
        ax2.set_yscale('log')
        ax2.set_title(f'Reconstruction Error — {model_name} (Log Scale)')
        ax2.set_xlabel('Reconstruction Error (MSE)')
        ax2.set_ylabel('Count')
        ax2.legend()

        self._save(filename)

    def plot_confusion_matrix(self, y_true, y_scores, threshold,
                              model_name='Model', filename='confusion_matrix.png'):
        """Plot confusion matrix at a given threshold."""
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt=',d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            linewidths=0.5, linecolor='gray'
        )
        plt.title(f'Confusion Matrix — {model_name}\n(threshold = {threshold:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        self._save(filename)

    def plot_training_loss(self, loss_histories, filename='training_loss.png'):
        """Plot training loss curves for AE and VAE."""
        colors = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED]
        plt.figure(figsize=(10, 6))
        for (name, losses), color in zip(loss_histories.items(), colors):
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, color=color, label=name, linewidth=1.5)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save(filename)

    def plot_feature_importance(self, importances, feature_names, top_n=15,
                                filename='feature_importance.png'):
        """Horizontal bar chart of RF feature importances (top N)."""
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(9, 7))
        plt.barh(range(top_n),
                 importances[indices],
                 color=CB_BLUE, edgecolor='black', linewidth=0.5)
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances (Random Forest)')
        self._save(filename)

    def plot_threshold_analysis(self, y_true, model_scores_dict,
                                fpr_range=np.linspace(0.001, 0.05, 50),
                                filename='threshold_analysis.png'):
        """Recall and Precision vs FPR threshold for each model."""
        colors = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_TEAL, CB_GRAY]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for (name, scores), color in zip(model_scores_dict.items(), colors):
            fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
            recalls, precisions = [], []
            for fpr_t in fpr_range:
                valid = np.where(fpr_arr <= fpr_t)[0]
                idx = valid[-1] if len(valid) > 0 else 0
                recalls.append(tpr_arr[idx])
                thr = thresholds[min(idx, len(thresholds) - 1)]
                preds = (scores >= thr).astype(int)
                tp = np.sum((preds == 1) & (y_true == 1))
                fp = np.sum((preds == 1) & (y_true == 0))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            ax1.plot(fpr_range * 100, recalls, color=color, label=name, linewidth=1.5)
            ax2.plot(fpr_range * 100, precisions, color=color, label=name, linewidth=1.5)
        ax1.set_xlabel('FPR Threshold (%)')
        ax1.set_ylabel('Recall (TPR)')
        ax1.set_title('Recall vs FPR Threshold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('FPR Threshold (%)')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs FPR Threshold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        self._save(filename)

    def plot_metrics_heatmap(self, results_df, filename='metrics_heatmap.png'):
        """Heatmap of all models × metrics."""
        metric_cols = [c for c in results_df.columns if c != 'Model']
        data = results_df[metric_cols].values.astype(float)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            data, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=metric_cols,
            yticklabels=results_df['Model'].values,
            linewidths=0.5, linecolor='gray',
            cbar_kws={'label': 'Score'},
        )
        plt.title('Model Performance Comparison')
        self._save(filename)

    def plot_latent_space(self, vae_model, X_test, y_test,
                          n_legit=5000, filename='latent_space.png'):
        """2D t-SNE of VAE latent space, colored by class.
        Subsamples legitimate transactions for speed."""
        from sklearn.manifold import TSNE
        import torch

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        np.random.seed(42)
        if len(legit_idx) > n_legit:
            legit_idx = np.random.choice(legit_idx, n_legit, replace=False)
        subset_idx = np.concatenate([legit_idx, fraud_idx])
        X_sub = X_test[subset_idx]
        y_sub = y_test[subset_idx]

        vae_model.eval()
        with torch.no_grad():
            mu, _ = vae_model.encode(torch.FloatTensor(X_sub))
            latent = mu.numpy()

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = tsne.fit_transform(latent)

        plt.figure(figsize=(10, 8))
        legit_mask = y_sub == 0
        fraud_mask = y_sub == 1
        plt.scatter(coords[legit_mask, 0], coords[legit_mask, 1],
                    c=CLS_LEGIT, alpha=0.3, s=10,
                    label=f'Legitimate (n={legit_mask.sum():,})')
        plt.scatter(coords[fraud_mask, 0], coords[fraud_mask, 1],
                    c=CLS_FRAUD, alpha=0.8, s=30, marker='x',
                    label=f'Fraud (n={fraud_mask.sum():,})')
        plt.title('VAE Latent Space (t-SNE Projection)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        self._save(filename)

