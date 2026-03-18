import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             confusion_matrix, average_precision_score,
                             precision_score)
from sklearn.manifold import TSNE
from pathlib import Path

# --------------------------------------
# Global plot style
# --------------------------------------
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

# --------------------------------------
# Colour-blind-friendly palette
# --------------------------------------
CB_BLUE   = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN  = '#009E73'
CB_RED    = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_CYAN   = '#56B4E9'
CB_TEAL   = '#44AA99'
CB_GRAY   = '#999999'

CLS_LEGIT = CB_BLUE
CLS_FRAUD = CB_RED

PALETTE = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_TEAL, CB_GRAY]


class ModelVisualizer:
    def __init__(self, figures_path):
        self.figures_path = Path(figures_path)
        self.figures_path.mkdir(parents=True, exist_ok=True)

    def _save(self, filename):
        plt.tight_layout()
        plt.savefig(self.figures_path / filename, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')

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
        for bar, count in zip(bars, counts.values):
            if count > 1000:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.6,
                         f'n = {count:,}', ha='center', va='center',
                         fontweight='bold', fontsize=12, color='white')
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 3,
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
        plt.title(f'{feature} Distribution: Legitimate vs Fraud')
        plt.xlabel(feature)
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
            bp = ax.boxplot(
                [df[df['Class'] == cls][feat] for cls in [0, 1]],
                tick_labels=['Legit', 'Fraud'],
                patch_artist=True,
                widths=0.5,
                showfliers=False,
            )
            for box, color in zip(bp['boxes'], [CLS_LEGIT, CLS_FRAUD]):
                box.set_facecolor(color)
                box.set_alpha(0.6)
            ax.set_title(feat, fontweight='bold')
            ax.set_ylabel('Value')
        fig.suptitle('Feature Distributions by Class (Outliers Hidden)', fontsize=13, y=1.02)
        self._save(filename)

    def plot_feature_correlation(self, df, feature_names, filename='feature_correlation.png'):
        """Absolute Pearson correlation of each feature with the fraud label.
        Shows discriminative power without any trained model."""
        corr = df[feature_names + ['Class']].corr()['Class'].drop('Class').abs()
        corr_sorted = corr.sort_values()
        plt.figure(figsize=(9, 7))
        plt.barh(range(len(corr_sorted)), corr_sorted.values,
                 color=CB_BLUE, edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(corr_sorted)), corr_sorted.index)
        plt.xlabel('|Pearson Correlation with Fraud Label|')
        plt.title('Feature Correlation with Fraud')
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

    def plot_amount_distributions(self, eda_df, filename='amount_distributions.png'):
        """log1p-transformed raw EUR amounts reveal the fraud/legit separation."""
        fig, ax = plt.subplots(figsize=(9, 6))

        for cls, label, color in zip([0, 1], ['Legitimate', 'Fraud'], [CLS_LEGIT, CLS_FRAUD]):
            raw = np.log1p(eda_df[eda_df['Class'] == cls]['Amount'])
            sns.kdeplot(raw, ax=ax, color=color, fill=True,
                        label=f'{label} (n={len(raw):,})', alpha=0.3)

        ax.set_title('Amount Distribution')
        ax.set_xlabel('log(1 + Amount)')
        ax.set_ylabel('Density')
        ax.legend()

        self._save(filename)

    # ------------------------------------------------------------------
    # Model evaluation plots
    # ------------------------------------------------------------------
    def plot_pr_curves(self, y_true, model_scores_dict, filename='pr_curves.png'):
        fig, ax = plt.subplots(figsize=(13, 7))
        for (name, scores), color in zip(model_scores_dict.items(), cycle(PALETTE)):
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=color,
                    label=f'{name} (AUPRC={ap:.4f})', linewidth=1.5)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        self._save(filename)

    def plot_roc_curves(self, y_true, model_scores_dict,
                        fpr_thresholds=(0.001, 0.005, 0.01),
                        filename='roc_curves.png'):
        markers = ['o', 's', 'D']
        fig, ax = plt.subplots(figsize=(13, 7))

        curves = []
        for (name, scores), color in zip(model_scores_dict.items(), cycle(PALETTE)):
            fpr, tpr, _ = roc_curve(y_true, scores)
            curves.append((name, fpr, tpr, color))
            ax.plot(fpr, tpr, color=color,
                    label=f'{name} (AUC={auc(fpr, tpr):.4f})', linewidth=1.5)
            for fpr_t, marker in zip(fpr_thresholds, markers):
                valid = np.where(fpr <= fpr_t)[0]
                if len(valid):
                    ax.plot(fpr[valid[-1]], tpr[valid[-1]], marker=marker,
                            color=color, markersize=7, zorder=5)

        for fpr_t, marker in zip(fpr_thresholds, markers):
            ax.plot([], [], marker=marker, color='gray', linestyle='None',
                    markersize=7, label=f'FPR={fpr_t*100:.1f}%')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves with Operating Points')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

        # Zoomed inset — low-FPR operating region
        ax_inset = ax.inset_axes([0.38, 0.08, 0.56, 0.50])
        for name, fpr, tpr, color in curves:
            ax_inset.plot(fpr, tpr, color=color, linewidth=1.2)
            for fpr_t, marker in zip(fpr_thresholds, markers):
                valid = np.where(fpr <= fpr_t)[0]
                if len(valid):
                    ax_inset.plot(fpr[valid[-1]], tpr[valid[-1]], marker=marker,
                                  color=color, markersize=6, zorder=5)
        ax_inset.set_xlim(0, 0.025)
        ax_inset.set_ylim(0.5, 1.0)
        ax_inset.set_xlabel('FPR', fontsize=8)
        ax_inset.set_ylabel('TPR', fontsize=8)
        ax_inset.set_title('Low-FPR Region', fontsize=8)
        ax_inset.tick_params(labelsize=7)
        ax.indicate_inset_zoom(ax_inset, edgecolor='black', alpha=0.4)

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

    def plot_threshold_analysis(self, y_true, model_scores_dict,
                                fpr_range=None,
                                filename='threshold_analysis.png'):
        """Recall and Precision vs FPR threshold for each model."""
        if fpr_range is None:
            fpr_range = np.linspace(0.001, 0.05, 50)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for (name, scores), color in zip(model_scores_dict.items(), cycle(PALETTE)):
            fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
            recalls, precisions = [], []
            for fpr_t in fpr_range:
                valid = np.where(fpr_arr <= fpr_t)[0]
                idx = valid[-1] if len(valid) > 0 else 0
                recalls.append(tpr_arr[idx])
                thr = thresholds[min(idx, len(thresholds) - 1)]
                preds = (scores >= thr).astype(int)
                precisions.append(precision_score(y_true, preds, zero_division=0))
            ax1.plot(fpr_range * 100, recalls, color=color, label=name, linewidth=1.5)
            ax2.plot(fpr_range * 100, precisions, color=color, label=name, linewidth=1.5)
        for ax, ylabel, title in zip(
            [ax1, ax2],
            ['Recall (TPR)', 'Precision'],
            ['Recall vs FPR Threshold', 'Precision vs FPR Threshold'],
        ):
            ax.set_xlabel('FPR Threshold (%)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
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

    def plot_score_correlation(self, model_scores_dict, filename='score_correlation.png'):
        """Pearson correlation between model anomaly scores.
        Reveals redundancy between models and supports ensemble design."""
        scores_df = pd.DataFrame(model_scores_dict)
        plt.figure(figsize=(9, 8))
        sns.heatmap(
            scores_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'label': 'Pearson r'},
        )
        plt.title('Model Score Correlations')
        plt.xticks(rotation=30, ha='right')
        self._save(filename)

    # ------------------------------------------------------------------
    # Autoencoder plots
    # ------------------------------------------------------------------
    def plot_error_distribution(self, legit_scores, fraud_scores,
                                model_name='AE', filename='error_dist.png'):
        """Reconstruction error histograms on linear and log scales."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
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

        ax2.hist(legit_scores, bins=bins, color=CLS_LEGIT, alpha=0.6, label='Legitimate')
        ax2.hist(fraud_scores, bins=bins, color=CLS_FRAUD, alpha=0.6, label='Fraud')
        ax2.set_yscale('log')
        ax2.set_title(f'Reconstruction Error — {model_name} (Log Scale)')
        ax2.set_xlabel('Reconstruction Error (MSE)')
        ax2.set_ylabel('Count')
        ax2.legend()

        self._save(filename)

    def plot_ae_feature_error(self, ae_model, X_test, y_test, feature_names,
                              filename='ae_feature_error.png'):
        """Mean per-feature reconstruction error for legitimate vs fraud transactions.
        Shows which input features drive the anomaly score most."""
        ae_model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test)
            recon = ae_model(X_t).numpy()
        feat_err = (X_test - recon) ** 2
        legit_mean = feat_err[y_test == 0].mean(axis=0)
        fraud_mean = feat_err[y_test == 1].mean(axis=0)

        x = np.arange(len(feature_names))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - 0.2, legit_mean, width=0.4, label='Legitimate',
               color=CLS_LEGIT, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.bar(x + 0.2, fraud_mean, width=0.4, label='Fraud',
               color=CLS_FRAUD, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylabel('Mean Squared Error per Feature (log scale)')
        ax.set_title('Autoencoder: Per-Feature Reconstruction Error by Class')
        ax.legend()
        self._save(filename)

    def plot_latent_space(self, vae_model, X_test, y_test,
                          n_legit=5000, filename='latent_space.png'):
        """2D t-SNE of VAE latent space, colored by class.
        Subsamples legitimate transactions for speed."""

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        rng = np.random.RandomState(42)
        if len(legit_idx) > n_legit:
            legit_idx = rng.choice(legit_idx, n_legit, replace=False)
        subset_idx = np.concatenate([legit_idx, fraud_idx])
        X_sub = X_test[subset_idx]
        y_sub = y_test[subset_idx]

        vae_model.eval()
        with torch.no_grad():
            mu, _ = vae_model.encode(torch.FloatTensor(X_sub))
            latent = mu.detach().numpy()

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
