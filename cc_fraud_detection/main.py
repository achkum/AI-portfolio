import re
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from loader import CreditCardDataLoader
from models import (Autoencoder, VAE, OneClassSVMModel, RandomForestModel,
                    LogisticRegressionModel, XGBoostModel, HybridModel,
                    ModelEvaluator)
from visualizer import ModelVisualizer

ROOT = Path(__file__).parent
FIGURES = ROOT / 'figures'
DATA = ROOT / 'creditcard.csv'


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    print('=' * 62)
    print('   Credit Card Fraud Detection — Evaluation Pipeline')
    print('=' * 62)

    # ------------------------------------------------------------------
    # 1. Data loading and preprocessing
    # ------------------------------------------------------------------
    print('\n[1/9] Loading and preprocessing data...')
    loader = CreditCardDataLoader()
    if not DATA.exists():
        print('  Dataset not found locally — downloading from Kaggle...')
        loader.download_from_kaggle(ROOT)
    train_df, test_df = loader.load_and_split(DATA)

    X_unsupervised = loader.get_unsupervised_train(train_df)
    X_train, y_train = loader.get_supervised_data(train_df)
    X_test, y_test = loader.get_supervised_data(test_df)

    viz = ModelVisualizer(FIGURES)

    # ------------------------------------------------------------------
    # 2. EDA visualisations
    # ------------------------------------------------------------------
    print('\n[2/9] Generating exploratory data analysis (EDA) figures...')
    viz.plot_class_balance(loader.eda_df)
    viz.plot_feature_boxplots(loader.eda_df)
    viz.plot_feature_correlation(loader.eda_df, loader.FEATURES)
    viz.plot_correlation_heatmap(loader.eda_df)
    viz.plot_amount_distributions(loader.eda_df)

    # ------------------------------------------------------------------
    # 3. Train unsupervised models
    # ------------------------------------------------------------------
    train_cfg = dict(learning_rate=0.001, epochs=50, batch_size=512)

    print('\n[3/9] Training unsupervised anomaly detection models...')
    print('  Training Autoencoder (AE)...')
    ae = Autoencoder()
    ae.train_model(X_unsupervised, train_cfg)

    print('  Training Variational Autoencoder (VAE)...')
    vae = VAE()
    vae.train_model(X_unsupervised, train_cfg)

    print('  Training One-Class SVM...')
    ocsvm = OneClassSVMModel(kernel='rbf', gamma='scale', nu=0.01)
    ocsvm.train_model(X_unsupervised)

    # ------------------------------------------------------------------
    # 4. Train supervised models
    # ------------------------------------------------------------------
    print('\n[4/9] Training supervised classification models...')
    print('  Training Random Forest...')
    rf = RandomForestModel(n_estimators=100, max_depth=30, min_samples_leaf=2, random_state=42)
    rf.train_model(X_train, y_train)

    print('  Training Logistic Regression...')
    lr = LogisticRegressionModel(random_state=42)
    lr.train_model(X_train, y_train)

    print('  Training XGBoost...')
    xgb = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.train_model(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Generate anomaly scores on test set
    # ------------------------------------------------------------------
    print('\n[5/9] Generating anomaly scores on the test set...')
    ae_test_scores = ae.predict_anomaly(X_test)
    vae_test_scores = vae.predict_anomaly(X_test)
    ocsvm_test_scores = ocsvm.predict_anomaly(X_test)
    rf_test_scores = rf.predict_anomaly(X_test)
    lr_test_scores = lr.predict_anomaly(X_test)
    xgb_test_scores = xgb.predict_anomaly(X_test)

    # ------------------------------------------------------------------
    # 6. Hybrid stacking: combine all model scores via meta-learner
    # ------------------------------------------------------------------
    print('\n[6/9] Building hybrid stacking ensembles...')
    # Unsupervised train scores (trained on legit-only data)
    ae_train_scores = ae.predict_anomaly(X_train)
    vae_train_scores = vae.predict_anomaly(X_train)
    ocsvm_train_scores = ocsvm.predict_anomaly(X_train)

    # Out-of-fold (OOF) scores for supervised models to avoid data leakage
    print('  Generating out-of-fold scores for supervised models (3-fold CV)...')
    oof_cv = 3
    rf_oof  = cross_val_predict(rf.model,  X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]
    lr_oof  = cross_val_predict(lr.model,  X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]
    xgb_oof = cross_val_predict(xgb.model, X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]
    print('  Out-of-fold scores computed for RF, LR, and XGBoost.')

    # --- Hybrid 1: Meta-learner on unsupervised scores only ---
    print('  Training Hybrid meta-learner (unsupervised scores)...')
    hybrid_unsup = HybridModel(n_estimators=100, max_depth=30, min_samples_leaf=2, random_state=42)
    hybrid_unsup.train_model(X_train, y_train, {
        'AE': ae_train_scores, 'VAE': vae_train_scores,
        'OCSVM': ocsvm_train_scores,
    })
    hybrid_unsup_test_scores = hybrid_unsup.predict_anomaly(X_test, {
        'AE': ae_test_scores, 'VAE': vae_test_scores,
        'OCSVM': ocsvm_test_scores,
    })

    # --- Hybrid 2: Meta-learner on all model scores (full stacking) ---
    print('  Training Hybrid meta-learner (full stacking — all model scores)...')
    hybrid_stack = HybridModel(n_estimators=100, max_depth=30, min_samples_leaf=2, random_state=42)
    hybrid_stack.train_model(X_train, y_train, {
        'AE': ae_train_scores, 'VAE': vae_train_scores,
        'OCSVM': ocsvm_train_scores,
        'RF': rf_oof, 'LR': lr_oof, 'XGB': xgb_oof,
    })
    hybrid_stack_test_scores = hybrid_stack.predict_anomaly(X_test, {
        'AE': ae_test_scores, 'VAE': vae_test_scores,
        'OCSVM': ocsvm_test_scores,
        'RF': rf_test_scores, 'LR': lr_test_scores, 'XGB': xgb_test_scores,
    })

    # ------------------------------------------------------------------
    # 7. Evaluate all models
    # ------------------------------------------------------------------
    print('\n[7/9] Evaluating all models on the held-out test set...')
    model_scores = {
        'Autoencoder': ae_test_scores,
        'VAE': vae_test_scores,
        'One-Class SVM': ocsvm_test_scores,
        'Logistic Regression': lr_test_scores,
        'Random Forest': rf_test_scores,
        'XGBoost': xgb_test_scores,
        'Hybrid (Unsupervised)': hybrid_unsup_test_scores,
        'Hybrid (Stacking)': hybrid_stack_test_scores,
    }

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(
        y_test, model_scores,
        fpr_thresholds=[0.001, 0.005, 0.01]
    )
    print('\n  Evaluation Summary (FPR thresholds: 0.1%, 0.5%, 1.0%):')
    print(results.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Save evaluation figures
    # ------------------------------------------------------------------
    print(f'\n[8/9] Generating and saving evaluation figures to: {FIGURES}')
    viz.plot_pr_curves(y_test, model_scores)
    viz.plot_roc_curves(y_test, model_scores)

    # Error distributions for both autoencoder variants
    viz.plot_error_distribution(
        ae_test_scores[y_test == 0], ae_test_scores[y_test == 1],
        model_name='Autoencoder', filename='error_dist_ae.png'
    )
    viz.plot_error_distribution(
        vae_test_scores[y_test == 0], vae_test_scores[y_test == 1],
        model_name='VAE', filename='error_dist_vae.png'
    )

    # Confusion matrices at 1% FPR for all models
    for model_name, scores in model_scores.items():
        _, thr = evaluator.calculate_recall_at_fpr(y_test, scores, fpr_threshold=0.01)
        safe_name = re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')
        viz.plot_confusion_matrix(
            y_test, scores, threshold=thr,
            model_name=f'{model_name} @ 1% FPR',
            filename=f'confusion_matrix_{safe_name}.png',
        )

    # Threshold analysis (all models)
    viz.plot_threshold_analysis(y_test, model_scores)

    # Metrics heatmap
    viz.plot_metrics_heatmap(results)

    # Latent space visualization
    print('  Generating VAE latent space visualization (t-SNE)...')
    viz.plot_latent_space(vae, X_test, y_test)

    # Score correlation
    viz.plot_score_correlation(model_scores)

    # Autoencoder-specific plots
    viz.plot_ae_feature_error(ae, X_test, y_test, loader.FEATURES)

    # ------------------------------------------------------------------
    # 9. Done
    # ------------------------------------------------------------------
    print('\n[9/9] Pipeline complete.')
    print(f'      All figures saved to: {FIGURES}')
    print('=' * 62)


if __name__ == '__main__':
    main()
