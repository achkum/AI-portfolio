import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from data.loader import CreditCardDataLoader
from model.autoencoder import Autoencoder
from model.vae import VAE
from model.random_forest import RandomForestModel
from model.evaluator import ModelEvaluator
from commons.visualizer import ModelVisualizer

def main():
    print("--- Starting Simplified Fraud Detection Pipeline ---")
    
    # Load Config
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Data Layer
    data_loader = CreditCardDataLoader(raw_path=config['data']['raw_path'])
    try:
        train_df, test_df = data_loader.load_and_split()
    except Exception as e:
        print(f"Error loading data: {e}. Please ensure CSV is in {config['data']['raw_path']}")
        return

    # Prepare datasets
    X_train_unsupervised = data_loader.get_unsupervised_train(train_df)
    X_train_supervised, y_train_supervised = data_loader.get_supervised_data(train_df, oversample=True)
    X_test, y_test = data_loader.get_supervised_data(test_df, oversample=False)

    # 2. Model Training
    # 2.1 Autoencoder
    print("Training Autoencoder...")
    ae_cfg = config['model']['autoencoder']
    ae = Autoencoder(input_dim=ae_cfg['input_dim'], hidden_dims=ae_cfg['hidden_dims'])
    ae.train_model(X_train_unsupervised, ae_cfg)

    # 2.2 VAE
    print("Training VAE...")
    vae_cfg = config['model']['vae']
    vae = VAE(input_dim=vae_cfg['input_dim'], latent_dim=vae_cfg['latent_dim'], hidden_dims=vae_cfg['hidden_dims'])
    vae.train_model(X_train_unsupervised, vae_cfg)

    # 2.3 Random Forest (Supervised)
    print("Training Random Forest...")
    rf_cfg = config['model']['random_forest']
    rf = RandomForestModel(n_estimators=rf_cfg['n_estimators'], random_state=rf_cfg['random_state'])
    rf.train_model(X_train_supervised, y_train_supervised)

    # 3. Evaluation
    print("Evaluating models...")
    ae_scores = ae.predict_anomaly(X_test)
    vae_scores = vae.predict_anomaly(X_test)
    rf_scores = rf.predict_anomaly(X_test)

    model_scores = {
        'Autoencoder': ae_scores,
        'VAE': vae_scores,
        'Random Forest': rf_scores
    }

    evaluator = ModelEvaluator()
    results_df = evaluator.evaluate_all(y_test, model_scores, fpr_thresholds=config['evaluation']['fpr_thresholds'])
    print("\nEvaluation Summary:")
    print(results_df.to_string(index=False))

    # 4. output
    print("Saving figures to output/figures/...")
    visualizer = ModelVisualizer(figures_path=config['output']['figures_path'])
    visualizer.plot_pr_curves(y_test, model_scores)
    visualizer.plot_roc_curves(y_test, model_scores)

    # Error distribution plot for AE
    visualizer.plot_error_distribution(ae_scores[y_test == 0], ae_scores[y_test == 1], model_name='Autoencoder', filename='ae_error_distribution.png')

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
