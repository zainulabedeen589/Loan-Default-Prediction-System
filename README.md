# 💳 Loan Default Prediction System

> **An end-to-end MLOps pipeline for binary classification of loan default risk, served via an interactive Streamlit web application.**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue?logo=mlflow)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Production--Ready%20v1-green)]()

---

## [Application Link](https://zainul-loan-default-prediction.streamlit.app/)

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline Architecture](#-pipeline-architecture)
- [Feature Engineering](#-feature-engineering)
- [Model Details](#-model-details)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Configuration](#-configuration)
- [Running the Web App](#-running-the-web-app)
- [Documentation](#-documentation)
- [Known Issues & Roadmap](#-known-issues--roadmap)
- [Author](#-author)

---

## 🔍 Overview

The **Loan Default Prediction System** predicts whether a loan applicant is likely to default on their loan. It takes 16 applicant attributes (financial, personal, and loan-specific) as input and outputs:

- **Prediction:** `LOW RISK` (0) or `HIGH RISK` (1)
- **Default Probability:** A percentage score (0–100%)

The system is built with a clean, modular Python pipeline that covers the full ML lifecycle — from raw data ingestion through to a live web interface.

| Property | Value |
|---|---|
| **Problem** | Binary Classification |
| **Target** | `Default` (0 = No Default, 1 = Default) |
| **Model** | LightGBM + GridSearchCV |
| **Interface** | Streamlit Web App |
| **Tracking** | MLflow (local) |

---

## 📁 Project Structure

```
Loan Default Prediction/
│
├── app.py                      # 🌐 Streamlit inference web app
├── main.py                     # 🚀 Training pipeline orchestrator
├── setup.py                    # 📦 Package setup (editable install)
├── requirements.txt            # 📋 Python dependencies
│
├── src/                        # 🧠 Core ML pipeline modules
│   ├── __init__.py
│   ├── data_ingestion.py       # Stage 1: Load & train/test split
│   ├── database_extraction.py  # Optional: MySQL data source
│   ├── data_processing.py      # Stage 2: Clean & outlier handling
│   ├── feature_engineering.py  # Stage 3: FE, encoding, selection
│   ├── model_training.py       # Stage 4: Train, evaluate, log
│   ├── model_slection.py       # Experimental: Multi-model benchmarking
│   ├── paths_config.py         # Centralised file path constants
│   ├── logger.py               # File-based logging setup
│   └── custom_exception.py     # Enriched exception with traceback
│
├── utils/
│   └── helpers.py              # Shared utility functions
│
├── config/
│   └── params.json             # Hyperparameter search grid
│
├── artifact/                   # 📦 All generated pipeline artefacts
│   ├── raw/                    # Source dataset (Loan_default.csv)
│   ├── ingested_data/          # train.csv, test.csv
│   ├── processed_data/         # processed_train.csv
│   ├── engineered_data/        # final_df.csv (model-ready)
│   └── models/                 # trained_model.pkl, encoding_obj.pkl
│
├── logs/                       # 📝 Timestamped training logs
├── mlruns/                     # 📊 MLflow experiment data
├── model/                      # 🗄️  Alternate model storage (legacy)
│
└── Docs/                       # 📚 Architecture documentation
    ├── HLD.md                  # High-Level Design
    └── LLD.md                  # Low-Level Design
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install the project as an editable package
pip install -e .
```

### 2. Add Your Data

Place your raw dataset at:
```
artifact/raw/Loan_default.csv
```

### 3. Run the Training Pipeline

```bash
python main.py
```

This will execute all four stages sequentially:
1. **Data Ingestion** → splits raw CSV to train/test
2. **Data Processing** → cleans and clips outliers  
3. **Feature Engineering** → constructs, encodes, and selects features
4. **Model Training** → tunes LightGBM and logs to MLflow

### 4. Launch the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 5. View MLflow Experiments

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://localhost:5000`

---

## 🔄 Pipeline Architecture

```
Raw CSV
  │
  ▼
DataIngestion ──────── train.csv (80%)
                  └─── test.csv  (20%)
  │
  ▼ (train.csv)
DataProcessor ─── Drop LoanID
              ─── IQR Outlier Clipping
              └── Save: processed_train.csv
  │
  ▼
FeatureEngineer ─── Bin Age (5 groups)
                ─── Label Encode (8 columns)
                ─── Construct 5 new features
                ─── Select Top 12 by Mutual Info
                └── Save: final_df.csv + encoding_obj.pkl
  │
  ▼
ModelTraining ─── GridSearchCV (27 combinations, cv=3)
              ─── Evaluate on internal split
              ─── Log to MLflow
              └── Save: trained_model.pkl
  │
  ▼
app.py (Streamlit) ──── Load trained_model.pkl
                   ──── Load encoding_obj.pkl
                   └──▶ Real-time prediction UI
```

---

## 🛠 Feature Engineering

Five new predictive features are constructed from raw inputs:

| Feature | Formula | Rationale |
|---|---|---|
| `Monthly_Payment` | `LoanAmount × (1 + Rate/100) / Term` | Actual monthly cash outflow |
| `PTI_Ratio` | `Monthly_Payment / (Income / 12)` | Payment-to-Income: key affordability metric |
| `Job_Stability_Index` | `MonthsEmployed / (Age × 12)` | Fraction of adult life spent employed |
| `Debt_per_Line` | `LoanAmount / (NumCreditLines + 1)` | Debt burden per credit line |
| `Young_High_Risk` | `(Age < 30) AND (LoanAmount > median)` | Binary flag for risky young borrowers |

**Age Binning:** `Age` is discretised into 5 ordinal groups:

| Group | Range |
|---|---|
| Child | 0 – 18 |
| Teenager | 18 – 30 |
| Adult | 30 – 45 |
| Senior | 45 – 60 |
| Super Senior | 60 – 100 |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | LightGBM (`LGBMClassifier`) |
| Tuning | GridSearchCV (`cv=3`, `scoring='accuracy'`) |
| Input Features | Top 12 by Mutual Information |
| Serialisation | joblib (`.pkl`) |
| Model Size | ~358 KB |

### Hyperparameter Search Grid (`config/params.json`)

```json
{
  "learning_rate":  [0.01, 0.05, 0.1],
  "n_estimators":   [100, 200, 300],
  "max_depth":      [5, 10, 15]
}
```

### Candidate Models Benchmarked (`model_slection.py`)

Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVC, KNN, Naive Bayes, Decision Tree, **LightGBM** ✅, XGBoost

### Validation Results (Latest Run)

| | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | 35,938 | 135 |
| **Actual: Default** | 4,569 | 214 |

> ⚠️ The high false-negative rate (4,569) indicates class imbalance. See [Known Issues](#-known-issues--roadmap).

---

## 📊 MLflow Experiment Tracking

All training runs are logged automatically. Each run records:

- **Parameters:** full hyperparameter grid + best parameters found
- **Metrics:** accuracy, precision, recall, F1-score  
- **Artefacts:** confusion matrix JSON, serialised model
- **Model:** registered via `mlflow.sklearn.log_model`

```bash
# View all runs in the browser
mlflow ui --backend-store-uri ./mlruns
```

---

## ⚙️ Configuration

### File Paths (`src/paths_config.py`)

| Constant | Path |
|---|---|
| `RAW_DATA_PATH` | `artifact/raw/Loan_default.csv` |
| `TRAIN_DATA_PATH` | `artifact/ingested_data/train.csv` |
| `TEST_DATA_PATH` | `artifact/ingested_data/test.csv` |
| `PROCESSED_DATA_PATH` | `artifact/processed_data/processed_train.csv` |
| `ENGINEERED_DATA_PATH` | `artifact/engineered_data/final_df.csv` |
| `MODEL_SAVE_PATH` | `artifact/models/trained_model.pkl` |
| `ENCODER_SAVE_PATH` | `artifact/models/encoding_obj.pkl` |
| `PARAMS_PATH` | `config/params.json` |

### Hyperparameters (`config/params.json`)

Edit `config/params.json` to change the GridSearchCV search space without touching any Python code.

### Logging

All modules write structured logs to `logs/YYYY-MM-DD_HH-MM-SS.log`. A new log file is created per process start.

---

## 🌐 Running the Web App

The Streamlit app requires trained artefacts to exist:
- `artifact/models/trained_model.pkl`
- `artifact/models/encoding_obj.pkl`

Run the training pipeline first (`python main.py`), then:

```bash
streamlit run app.py
```

**Input fields:**

| Section | Fields |
|---|---|
| 👤 Personal Info | Age, Income, Credit Score |
| 💰 Loan Details | Amount, Interest Rate, Term, DTI Ratio |
| 📊 Other Info | Employment, Education, Marital Status, Mortgage, Dependents, Purpose, Co-Signer |

**Output:** Risk probability percentage + HIGH RISK / LOW RISK classification.

---

## 📚 Documentation

Full architectural documentation is in the `Docs/` directory:

| Document | Description |
|---|---|
| [`Docs/HLD.md`](Docs/HLD.md) | High-Level Design: system architecture, data flow, tech stack justification, bottleneck analysis |
| [`Docs/LLD.md`](Docs/LLD.md) | Low-Level Design: class diagrams, DB schema, API specification, error handling, bug registry |

---

## 🐛 Known Issues & Roadmap

### Critical Issues (Fix Before Production)

- [ ] **Class Imbalance:** Apply SMOTE or class weights; change GridSearchCV `scoring` to `'f1'` or `'roc_auc'`
- [ ] **Holdout Set Unused:** Use `test.csv` for final holdout evaluation, not another internal split
- [ ] **Wrong MLflow Experiment Name:** Change `"Airline_Default_Prediction"` → `"Loan_Default_Prediction"` in `model_training.py`

### Bug Fixes

- [ ] `database_extraction.py` line 49: typo `corsor` → `cursor`
- [ ] `model_slection.py` lines 53–54: wrong target column `'satisfaction'` → `'Default'`
- [ ] `utils/helpers.py`: fix shared `LabelEncoder` reuse bug

### Roadmap

- [ ] Replace GridSearchCV with Optuna (Bayesian hyperparameter optimisation)
- [ ] Add SHAP explainability to web app (per-prediction feature importance)
- [ ] Package inference as a FastAPI REST endpoint
- [ ] Add data drift detection (Evidently AI)
- [ ] CI/CD pipeline (GitHub Actions) for automated retraining
- [ ] Containerise with Docker for portable deployment

---

## 👨‍💻 Author

**Zainul Abedeen**  
📧 zainulpasha589@gmail.com  

---

## 📦 Dependencies

```
pandas          # Tabular data processing
numpy           # Numerical operations
matplotlib      # Plotting
seaborn         # Statistical visualisation
scikit-learn    # ML primitives (encoding, selection, metrics)
xgboost         # XGBoost classifier (benchmarking)
lightgbm        # LightGBM classifier (production model)
imbalanced-learn # SMOTE / class balancing (available, not yet wired)
scipy           # Statistical functions
mlflow          # Experiment tracking & model registry
streamlit       # Web interface
```

---

## Application Preview

![img](./images/loan_pred.png)

*Created by Zainul Abedeen · April 2026*
