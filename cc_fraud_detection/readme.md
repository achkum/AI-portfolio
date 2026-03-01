# 💳 Credit Card Fraud Detection: Autoencoder vs VAE

## 🎯 Project Overview
This project implements an anomaly detection framework for credit card fraud under extreme class imbalance (0.172% fraud rate). It compares reconstruction-based (Standard Autoencoder) versus probabilistic (Variational Autoencoder) approaches, benchmarked against supervised baselines.

---

## 📂 Project Structure

```text
fraud_detection/
├── main.py                     # Entry point - runs full pipeline
├── config/
│   └── config.yaml             # Hyperparameters and paths
├── data/
│   ├── creditcard.csv          # Local dataset (download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
│   └── loader.py               # Combined loading, splitting, and scaling
├── model/
│   ├── autoencoder.py          # Standard Autoencoder implementation
│   ├── vae.py                   # Variational Autoencoder implementation
│   ├── random_forest.py        # Supervised baseline
│   ├── optimizer.py            # Hand-coded Gradient Descent
│   └── evaluator.py            # Metrics: AUPRC, Recall@FPR
├── commons/
│   └── visualizer.py           # Matplotlib plots (PR/ROC curves)
├── output/                     # NEW: All generated content
│   ├── figures/                # Saved plots
│   ├── report/                 # Project documentation
├── requirements.txt
└── readme.md
```

---

## 🧩 Component Overview

### 1. Data Layer (`data/`)
- **`credit_card_loader.py`**: Loads the Kaggle dataset, handles the 28 PCA-anonymized features
- **`preprocessor.py`**: StandardScaler normalization, temporal train/test split (Day 1 train, Day 2 test)
- **`sampler.py`**: SMOTE oversampling for supervised baselines

### 2. Model Layer (`model/`)
- **`autoencoder.py`**: Standard AE trained only on legitimate transactions; fraud = high reconstruction error
- **`vae.py`**: VAE with reconstruction loss + KL divergence; anomaly score from latent distribution
- **`gradient_descent.py`**: Hand-coded batch gradient descent for neural network weight updates
- **`random_forest_model.py`**: Supervised baseline with class weighting
- **`evaluator.py`**: Computes AUPRC, Recall@FPR (0.5%, 1%), confusion matrices

### 3. Commons (`commons/`)
- **`visualizer.py`**: PR curves, ROC curves, reconstruction error distributions, latent space plots

---

## 🏗️ Pipeline Summary

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐
│  Load Data  │ -> │  Preprocess  │ -> │  Train Models   │ -> │  Evaluate  │
│  (Kaggle)   │    │  (Scale,     │    │  (AE, VAE, RF)  │    │  (AUPRC,   │
│             │    │   Split)     │    │                 │    │   Recall)  │
└─────────────┘    └──────────────┘    └─────────────────┘    └────────────┘
```

1. **Load**: Read `creditcard.csv` (284,807 transactions, 492 frauds)
2. **Preprocess**: Scale features, split by time (Day 1 → train, Day 2 → test)
3. **Train AE/VAE**: Train only on legitimate transactions (unsupervised)
4. **Train Baselines**: Random Forest with SMOTE on labeled data
5. **Evaluate**: Compare AUPRC, Recall at low FPR, generate visualizations

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transactions** | 284,807 |
| **Frauds** | 492 (0.172%) |
| **Features** | 28 PCA-anonymized + Time + Amount |
| **Split Strategy** | Temporal (Day 1 train, Day 2 test) |

---

## 🚀 Usage

### Installation
```bash
cd fraud_detection
pip install -r requirements.txt
```

### Download Data
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in `datasource/raw/`.

### Run Pipeline
```bash
python main.py
```

---

## 📈 Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| **AUPRC** | Area Under Precision-Recall Curve - robust for imbalanced data |
| **Recall @ 1% FPR** | Fraud detection rate at acceptable false positive level |
| **Precision-Recall Curve** | Visualizes tradeoff across thresholds |

---

## ✅ Course Requirements Coverage

| D7054E Topic | Implementation |
|--------------|----------------|
| **OOP Design** | Abstract base classes (`BaseModel`, `BaseDataLoader`), inheritance |
| **Gradient Descent** | Hand-coded optimizer in `gradient_descent.py` |
| **Matplotlib** | PR curves, ROC curves, error distributions in `visualizer.py` |
| **Evaluation Metrics** | AUPRC, Recall, Precision in `evaluator.py` |
| **Reproducible Notebooks** | `notebooks/` folder |
| **IMRaD Report** | `report/` folder |

---

## 📂 Key Outputs

- **`figures/`**: PR curves, ROC curves, reconstruction error histograms, latent space visualizations
- **`datasource/processed/`**: Scaled train/test sets
- **`report/`**: Final analysis comparing AE vs VAE vs supervised baselines
