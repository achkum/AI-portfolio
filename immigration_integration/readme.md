# 🇸🇪 Immigration Integration Analysis: Sweden Study

## 🎯 Project Overview
This analysis investigates the successful economic integration of immigrant populations in Sweden by region of birth. It uses a data-driven approach to identify which factors best predict employment and income levels across different arrivals.

---

### 📊 Data Sources & Download Links
To run the analysis, download the following datasets (or their updated versions) and place them in the `datasource/raw/` subdirectories:

*   **SCB (Statistics Sweden)**:
    - **Employment**: [Labour market status by birth region](https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0210__AM0210D/ArRegArbStatus/)
    - **Income**: [Total income from employment and business](https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__HE__HE0110__HE0110A/SamForvInk3/)
    - **Self-sufficiency**: [Self-sufficiency rates by birth region](https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__HE__HE0000/HE0000Tab04N/)

*   **Migrationsverket (Migration Agency)**:
    - **Asylum/Permit Stats**: [Official Statistics Portal (English)](https://www.migrationsverket.se/English/About-the-Migration-Agency/Statistics.html)

*   **Socialstyrelsen (National Board of Health & Welfare)**:
    - **Economic Assistance**: [Statistics on Economic Assistance (Swedish portal)](https://www.socialstyrelsen.se/statistik-och-data/statistik/alla-statistikamnen/ekonomiskt-bistand/)

---

## 🏗️ What this Project Does
The analysis implements a complete **Automated Data Science Pipeline**:

1.  **Multi-Source Integration**: 
    - Loads **Employment & Income** data from Statistics Sweden (SCB).
    - Loads **Asylum & Residency Permit** statistics from the Swedish Migration Agency (Migrationsverket).
    - Loads **Economic Assistance** (welfare) usage data from the National Board of Health and Welfare (Socialstyrelsen).
2.  **Harmonized Data Merging**: Uses custom region mapping (e.g., *Sverige* → *Sweden*, *Somalia* → *Africa*) to unify disparate datasets on common temporal and regional keys.
3.  **Automated Preprocessing**: Cleans merged data by handling missing values via median imputation and z-score normalization for numerical stability.
4.  **Modeling & Optimization**:
    - **Hand-Coded Linear Regression**: Built from scratch using batch Gradient Descent to model integration predictors.
    - **Random Forest Model**: Used for non-linear relationships and extracting feature importance.
    - **Evaluation**: Computes R², MAE, RMSE, and MAPE metrics across five-fold cross-validation.
5.  **Comparative Visualization**: Generates heatmaps and trend plots comparing integration outcomes for Nordic, EU, Asian, and African groups against native-born Populations.

---

## 🚀 Usage

### 📦 Installation
```bash
# Navigate to implementation directory
cd immigration_integration

# Install required data science stack
pip install -r requirements.txt
```

### 🏃 Running the Analysis
The main script executes the entire workflow, from raw data to visualization:
```bash
python main.py
```

---

## 📂 Key Outputs
- **`figures/`**: Generated bar charts (Regional comparisons), Trend lines (Outcome over time), and Heatmaps (Integration matrix).
- **`datasource/processed/`**: Harmonized and merged datasets ready for further exploration.

## ✅ Model Insights
- **GradientDescentRegressor**: Hand-coded Batch Gradient Descent with multivariate feature support.
- **RandomForestModel**: Provides ranked feature importance scores.
- **ModelEvaluator**: Tracks R², MAE, RMSE, and MAPE with Cross-Validation support.
