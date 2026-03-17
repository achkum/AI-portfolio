import sklearn
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
    print('--- Fraud Detection Pipeline ---')

    # ------------------------------------------------------------------
    # 1. Data loading and preprocessing
    # ------------------------------------------------------------------
    loader = CreditCardDataLoader()
    try:
        train_df, test_df = loader.load_and_split(DATA)
    except FileNotFoundError:
        print(f'Error: creditcard.csv not found at {DATA}')
        return

    X_unsupervised = loader.get_unsupervised_train(train_df)
    X_train, y_train = loader.get_supervised_data(train_df)
    X_test, y_test = loader.get_supervised_data(test_df)

    viz = ModelVisualizer(FIGURES)

    # ------------------------------------------------------------------
    # 2. EDA visualisations
    # ------------------------------------------------------------------
    print('Generating EDA visualizations...')
    viz.plot_class_balance(loader.eda_df)
    viz.plot_feature_distributions(loader.eda_df, feature='Amount')
    viz.plot_feature_boxplots(loader.eda_df)
    viz.plot_correlation_heatmap(loader.eda_df)

    print('Generating post-standardization visualizations...')
    viz.plot_scaling_effect(train_df)

    # ------------------------------------------------------------------
    # 3. Train unsupervised models
    # ------------------------------------------------------------------
    train_cfg = dict(learning_rate=0.001, epochs=50, batch_size=512)

    print('Training Autoencoder...')
    ae = Autoencoder()
    ae.train_model(X_unsupervised, train_cfg)

    print('Training VAE...')
    vae = VAE()
    vae.train_model(X_unsupervised, train_cfg)

    # Training loss curves
    print('Saving training loss curves...')
    viz.plot_training_loss({
        'Autoencoder (MSE)': ae.loss_history,
        'VAE (ELBO)': vae.loss_history,
    })

    print('Training One-Class SVM...')
    ocsvm = OneClassSVMModel(kernel='rbf', gamma='scale', nu=0.01)
    ocsvm.train_model(X_unsupervised)

    # ------------------------------------------------------------------
    # 4. Train supervised models
    # ------------------------------------------------------------------
    print('Training Random Forest...')
    rf = RandomForestModel(n_estimators=100, max_depth=30, min_samples_leaf=2, random_state=42)
    rf.train_model(X_train, y_train)

    print('Training Logistic Regression...')
    lr = LogisticRegressionModel(random_state=42)
    lr.train_model(X_train, y_train)

    print('Training XGBoost...')
    xgb = XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.train_model(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Generate anomaly scores on test set
    # ------------------------------------------------------------------
    print('Generating anomaly scores...')
    ae_test_scores = ae.predict_anomaly(X_test)
    vae_test_scores = vae.predict_anomaly(X_test)
    ocsvm_test_scores = ocsvm.predict_anomaly(X_test)
    rf_test_scores = rf.predict_anomaly(X_test)
    lr_test_scores = lr.predict_anomaly(X_test)
    xgb_test_scores = xgb.predict_anomaly(X_test)

    # ------------------------------------------------------------------
    # 6. Hybrid stacking: combine all model scores via meta-learner
    # ------------------------------------------------------------------
    # Unsupervised train scores (trained on legit-only data)
    ae_train_scores = ae.predict_anomaly(X_train)
    vae_train_scores = vae.predict_anomaly(X_train)
    ocsvm_train_scores = ocsvm.predict_anomaly(X_train)

    # OOF scores for supervised models only
    print('Generating OOF scores for supervised models...')
    oof_cv = 3
    rf_oof = cross_val_predict(
        RandomForestClassifier(
            n_estimators=100, max_depth=30, min_samples_leaf=2,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]
    lr_oof = cross_val_predict(
        LogisticRegression(
            random_state=42, max_iter=1000,
            class_weight='balanced', solver='lbfgs'
        ),
        X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]
    xgb_oof = cross_val_predict(
        XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric='logloss', n_jobs=1
        ),
        X_train, y_train, cv=oof_cv, method='predict_proba', n_jobs=-1)[:, 1]

    # --- Hybrid 1: Unsupervised-only scores only ---
    print('Training Hybrid (Unsupervised scores only)...')
    hybrid_unsup = HybridModel(n_estimators=100, max_depth=30, min_samples_leaf=2, random_state=42)
    hybrid_unsup.train_model(X_train, y_train, {
        'AE': ae_train_scores, 'VAE': vae_train_scores,
        'OCSVM': ocsvm_train_scores,
    })
    hybrid_unsup_test_scores = hybrid_unsup.predict_anomaly(X_test, {
        'AE': ae_test_scores, 'VAE': vae_test_scores,
        'OCSVM': ocsvm_test_scores,
    })

    # --- Hybrid 2: Full stacking (all model scores) ---
    print('Training Hybrid (Full Stacking)...')
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
    print('Evaluating...')
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
    print('\nEvaluation Summary:')
    print(results.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Save evaluation figures
    # ------------------------------------------------------------------
    print(f'\nSaving figures to {FIGURES}...')
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

    # Confusion matrices at 1% FPR
    _, hybrid_unsup_threshold = evaluator.calculate_recall_at_fpr(
        y_test, hybrid_unsup_test_scores, fpr_threshold=0.01
    )
    viz.plot_confusion_matrix(
        y_test, hybrid_unsup_test_scores, threshold=hybrid_unsup_threshold,
        model_name='Hybrid (Unsup.) @ 1% FPR',
        filename='confusion_matrix_hybrid_unsup.png'
    )

    _, hybrid_stack_threshold = evaluator.calculate_recall_at_fpr(
        y_test, hybrid_stack_test_scores, fpr_threshold=0.01
    )
    viz.plot_confusion_matrix(
        y_test, hybrid_stack_test_scores, threshold=hybrid_stack_threshold,
        model_name='Hybrid (Stacking) @ 1% FPR',
        filename='confusion_matrix_hybrid_stack.png'
    )

    _, rf_threshold = evaluator.calculate_recall_at_fpr(
        y_test, rf_test_scores, fpr_threshold=0.01
    )
    viz.plot_confusion_matrix(
        y_test, rf_test_scores, threshold=rf_threshold,
        model_name='Random Forest @ 1% FPR',
        filename='confusion_matrix_rf.png'
    )

    # Feature importance (RF)
    viz.plot_feature_importance(
        rf.rf.feature_importances_,
        loader.FEATURES,
        top_n=15,
    )

    # Threshold analysis (RF and both Hybrids)
    viz.plot_threshold_analysis(
        y_test,
        {
            'Random Forest': rf_test_scores,
            'Hybrid (Unsup.)': hybrid_unsup_test_scores,
            'Hybrid (Stacking)': hybrid_stack_test_scores,
        },
    )

    # Metrics heatmap
    viz.plot_metrics_heatmap(results)

    # Latent space visualization
    print('Generating latent space visualization...')
    viz.plot_latent_space(vae, X_test, y_test)

    # ------------------------------------------------------------------
    # 9. Done
    # ------------------------------------------------------------------
    print('Done.')


if __name__ == '__main__':
    main()
