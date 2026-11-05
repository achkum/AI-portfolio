# AI Portfolio

Machine learning, deep learning and data analysis projects by **Achyuth Kumar Baddela**. For my main applied AI projects, see [Applied-AI-Projects](https://github.com/achkum/Applied-AI-Projects).

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

## Course Projects

Group projects from the MSc in Applied Artificial Intelligence at Luleå University of Technology.

| Project | Course | Summary | Tech Stack |
|---|---|---|---|
| [Edge-Ready ECG Classification](./ecg_edge_classification) | D7043E Advanced Data Mining | Inter-patient heartbeat classification on MIT-BIH with a 10k-parameter 1D CNN, quantised to INT8 and synthesised for the MAX78002 edge AI chip. Follows all six CRISP-DM stages. | PyTorch, Scikit-Learn, ai8x toolchain |
| [Anchor-Based Object Recognition](./anchor_object_recognition) | R7020E Computer Vision and Image Processing | Finds and localises a specific object in video from three anchor photos, comparing HSV histograms, SIFT and frozen DINOv2 embeddings, with depth cues to reject printed decoys. | OpenCV, DINOv2, PyTorch |
| [SNN Motion Detection](./snn_motion_detection) | D7046E Neural Networks and Learning Machines | Motion-direction classification with spiking neural networks, comparing LIF networks, Reichardt detectors and STDP learning on synthetic data and DVS128 Gesture. | snnTorch, Brian2, PyTorch |
| [Car Detection in Snow](./snow_car_detection) | D7047E Advanced Deep Learning | YOLOv9 car detection on the Nordic Vehicle Dataset with a custom snow augmentation pipeline, domain-shift evaluation and Optuna hyperparameter tuning. | YOLOv9, Ultralytics, Albumentations, Optuna, W&B |

---

## Contact

**LinkedIn:** [https://www.linkedin.com/in/baddelaachyuth/](https://www.linkedin.com/in/baddelaachyuth/)  
**Location:** Stockholm, Sweden
