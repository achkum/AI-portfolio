# AI Portfolio

Machine learning and data analysis projects by **Achyuth Kumar Baddela**. For my main applied AI projects, see [Applied-AI-Projects](https://github.com/achkum/Applied-AI-Projects).

---

## Projects

### 1. [Credit Card Fraud Detection](./cc_fraud_detection)
**Anomaly detection on highly imbalanced transaction data**
- **Tech Stack**: Python, PyTorch, Scikit-Learn, XGBoost, Matplotlib.
- **Approach**:
    - Unsupervised models trained on legitimate transactions only: Autoencoder, Variational Autoencoder, and One-Class SVM.
    - Supervised baselines: Logistic Regression, Random Forest, and XGBoost.
    - Hybrid models that stack anomaly scores with the original features through a Random Forest meta-learner.
- **Evaluation**: AUPRC, ROC-AUC, and recall at a fixed false positive rate, with ROC/PR curves, confusion matrices, and threshold analysis in [`figures/`](./cc_fraud_detection/figures).
- **Data**: [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (downloaded automatically via the Kaggle CLI).

```bash
cd cc_fraud_detection
python main.py
```

### 2. [Immigration Integration Analysis](./immigration_integration)
**Economic integration of immigrant populations in Sweden by region of birth**
- **Tech Stack**: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
- **Approach**:
    - Merges public data from Statistics Sweden (SCB), the Swedish Migration Agency, and the National Board of Health and Welfare on shared region and year keys.
    - Linear regression implemented from scratch with batch gradient descent, alongside a Random Forest for feature importance.
    - Evaluated with R², MAE, RMSE, and MAPE using five-fold cross-validation.
- **Output**: Regional comparisons, trend lines, and heatmaps in [`figures/`](./immigration_integration/figures).

See the [project README](./immigration_integration/readme.md) for dataset download links and setup.

---

## Contact

**LinkedIn:** [https://www.linkedin.com/in/baddelaachyuth/](https://www.linkedin.com/in/baddelaachyuth/)  
**Location:** Stockholm, Sweden
