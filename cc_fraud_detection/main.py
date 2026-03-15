import sklearn  # import before torch to avoid Windows DLL conflict
from pathlib import Path
import numpy as np

from loader import CreditCardDataLoader
from models import Autoencoder, VAE, RandomForestModel, ModelEvaluator
from visualizer import ModelVisualizer

ROOT = Path(__file__).parent
FIGURES = ROOT / 'figures'
DATA = ROOT / 'creditcard.csv'

def main():
    print('--- Fraud Detection Pipeline ---')

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

    print('Generating EDA visualizations...')
    viz.plot_class_balance(loader.eda_df)
    viz.plot_feature_distributions(loader.eda_df, feature='Amount')
    viz.plot_qq(loader.eda_df)

    print('Generating post-standardization visualizations...')
    viz.plot_scaling_effect(train_df)
    viz.plot_scaled_qq(train_df)

    ae_cfg = dict(input_dim=29, hidden_dims=[16, 8, 16], learning_rate=0.001, epochs=50, batch_size=64)
    vae_cfg = dict(input_dim=29, latent_dim=4, hidden_dims=[16, 8], learning_rate=0.001, epochs=50, batch_size=64)
    rf_cfg = dict(n_estimators=100, random_state=42)

    print('Training Autoencoder...')
    ae = Autoencoder(input_dim=ae_cfg['input_dim'], hidden_dims=ae_cfg['hidden_dims'])
    ae.train_model(X_unsupervised, ae_cfg)

    print('Training VAE...')
    vae = VAE(input_dim=vae_cfg['input_dim'], latent_dim=vae_cfg['latent_dim'], hidden_dims=vae_cfg['hidden_dims'])
    vae.train_model(X_unsupervised, vae_cfg)

    print('Training Random Forest...')
    rf = RandomForestModel(n_estimators=rf_cfg['n_estimators'], random_state=rf_cfg['random_state'])
    rf.train_model(X_train, y_train)

    print('Evaluating...')
    model_scores = {
        'Autoencoder': ae.predict_anomaly(X_test),
        'VAE': vae.predict_anomaly(X_test),
        'Random Forest': rf.predict_anomaly(X_test),
    }

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(y_test, model_scores, fpr_thresholds=[0.001, 0.005, 0.01])
    print('\nEvaluation Summary:')
    print(results.to_string(index=False))

    print(f'\nSaving figures to {FIGURES}...')
    viz.plot_pr_curves(y_test, model_scores)
    viz.plot_roc_curves(y_test, model_scores)
    viz.plot_error_distribution(model_scores['Autoencoder'][y_test == 0], model_scores['Autoencoder'][y_test == 1], model_name='Autoencoder')

    print('Done.')

if __name__ == '__main__':
    main()
