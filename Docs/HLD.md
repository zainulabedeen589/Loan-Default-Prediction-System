# High-Level Design (HLD)
# Loan Default Prediction System

> **Version:** 1.0.0  
> **Author:** Zainul Abedeen  
> **Date:** April 2026  
> **Status:** Production-Ready (v1)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Architectural Pattern — Monolith vs Microservices](#3-architectural-pattern)
4. [Modular Breakdown](#4-modular-breakdown)
5. [End-to-End Data Flow](#5-end-to-end-data-flow)
6. [Tech Stack & Justification](#6-tech-stack--justification)
7. [MLOps & Experiment Tracking](#7-mlops--experiment-tracking)
8. [Artifact & State Management](#8-artifact--state-management)
9. [Logging & Observability](#9-logging--observability)
10. [Bottleneck Analysis & Scaling Recommendations](#10-bottleneck-analysis--scaling-recommendations)
11. [Deployment Architecture (Target State)](#11-deployment-architecture-target-state)

---

## 1. Executive Summary

The **Loan Default Prediction System** is a supervised binary-classification ML platform that determines whether a loan applicant is likely to default (`Default=1`) or not (`Default=0`). The product surfaces as an interactive web application powered by **Streamlit** and is backed by a **LightGBM** classifier trained with a structured MLOps pipeline.

| Attribute | Value |
|---|---|
| **Problem Type** | Binary Classification |
| **Target Variable** | `Default` (0 = No Default, 1 = Default) |
| **Primary Model** | LightGBM (LGBMClassifier) with GridSearchCV |
| **Serving Layer** | Streamlit Web App |
| **Experiment Tracking** | MLflow |
| **Persistence** | Local filesystem (CSV → PKL) |

**Confusion Matrix (Validation Set):**

| | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | 35,938 (TN) | 135 (FP) |
| **Actual: Default** | 4,569 (FN) | 214 (TP) |

> **Note:** High FN (4,569) signals a class-imbalance problem — a critical deployment risk. See §10.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOAN DEFAULT PREDICTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌──────────────┐     ┌──────────────────────────────────────────────────┐  │
│   │  Data Source │     │              ML Training Pipeline                │  │
│   │              │     │  (Offline / Batch — orchestrated by main.py)     │  │
│   │  CSV File    │────▶│                                                  │  │
│   │  (or MySQL   │     │  DataIngestion → DataProcessor → FeatureEngineer │  │
│   │   via DB     │     │              → ModelTraining                     │  │
│   │   Extractor) │     │                                                  │  │
│   └──────────────┘     └──────────────────┬───────────────────────────────┘  │
│                                           │                                   │
│                                 Serialised Artifacts                          │
│                          (trained_model.pkl, encoding_obj.pkl)                │
│                                           │                                   │
│                                           ▼                                   │
│                         ┌────────────────────────────┐                        │
│                         │     Streamlit Web App       │                        │
│                         │         (app.py)            │                        │
│                         │                             │                        │
│                         │  User Input → Preprocess →  │                        │
│                         │  Predict → Display Result   │                        │
│                         └────────────────────────────┘                        │
│                                           │                                   │
│                              ┌────────────▼─────────────┐                    │
│                              │   MLflow Tracking Server  │                    │
│                              │   (local: ./mlruns)       │                    │
│                              └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architectural Pattern

### Current Pattern: **Modular Monolith**

The system is a **Modular Monolith** — a single deployable process (`main.py`) that chains together clearly separated, single-responsibility modules. This is the correct architectural choice at the current project scale.

| Criterion | Monolith ✅ | Microservices |
|---|---|---|
| **Team Size** | Solo / Small (appropriate) | Multiple teams |
| **Dataset Size** | ~255K rows (batch-friendly) | Streaming / massive scale |
| **Deployment Complexity** | Low (single process) | High (orchestration needed) |
| **Latency Requirement** | Relaxed (interactive UI) | Sub-millisecond SLAs |
| **Coupling Risk** | Low (clean module boundaries) | — |

> **Recommendation for Industry Scale:** Decompose into a **Training Service** (scheduled batch job) and an **Inference Service** (FastAPI REST API), connected through a model registry (MLflow Model Registry or S3). This allows independent scaling and zero-downtime model updates.

---

## 4. Modular Breakdown

```
Loan Default Prediction/
│
├── main.py                   ← Pipeline Orchestrator (entry point)
│
├── app.py                    ← Serving Layer (Streamlit UI)
│
├── src/                      ← Core Business Logic
│   ├── data_ingestion.py     ← Stage 1: Load & Train/Test Split
│   ├── database_extraction.py← Optional: MySQL source connector
│   ├── data_processing.py    ← Stage 2: Clean, Deduplicate, Clip Outliers
│   ├── feature_engineering.py← Stage 3: Construct + Encode + Select Features
│   ├── model_training.py     ← Stage 4: Train + Tune + Evaluate + Log
│   ├── model_slection.py     ← Experimental: Multi-model Benchmarking
│   ├── paths_config.py       ← Single source of truth for file paths
│   ├── logger.py             ← Structured file-based logging
│   └── custom_exception.py   ← Enriched error reporting with traceback
│
├── utils/
│   └── helpers.py            ← Shared utility functions
│
├── config/
│   └── params.json           ← Externalized hyperparameter search grid
│
├── artifact/                 ← All generated/pipeline data
│   ├── raw/                  ← Source data (Loan_default.csv)
│   ├── ingested_data/        ← train.csv, test.csv
│   ├── processed_data/       ← processed_train.csv
│   ├── engineered_data/      ← final_df.csv (model-ready)
│   └── models/               ← trained_model.pkl, encoding_obj.pkl
│
├── logs/                     ← Timestamped log files (.log)
├── mlruns/                   ← MLflow experiment tracking data
└── Docs/                     ← Architecture documentation
```

---

## 5. End-to-End Data Flow

### 5.1 Training Pipeline (Offline Batch)

```
[Loan_default.csv]  ──(pd.read_csv)──▶  DataIngestion
       │
       │  train_test_split(test_size=0.2, random_state=42)
       │
       ├──▶  artifact/ingested_data/train.csv   (80%)
       └──▶  artifact/ingested_data/test.csv    (20%)  [held-out, unused in pipeline ⚠️]
                          │
                          ▼
                    DataProcessor
                    ├── Drop: LoanID (identifier leak)
                    ├── Outlier Clipping: IQR method on all numeric cols
                    └── Save: artifact/processed_data/processed_train.csv
                          │
                          ▼
                   FeatureEngineer
                    ├── bin_age()          → Age_Group (5 ordinal bins)
                    ├── label_encoding()   → 8 categorical columns encoded
                    ├── feature_construction()
                    │    ├── Monthly_Payment    = LoanAmount*(1+Rate/100)/Term
                    │    ├── PTI_Ratio          = Monthly_Payment / (Income/12)
                    │    ├── Job_Stability_Index= MonthsEmployed / (Age*12)
                    │    ├── Debt_per_Line      = LoanAmount / (NumLines+1)
                    │    └── Young_High_Risk    = (Age<30) & (LoanAmt > median)
                    ├── feature_selection()  → top 12 by Mutual Information
                    └── Save: artifact/engineered_data/final_df.csv
                          │
                          ▼
                    ModelTraining
                    ├── GridSearchCV(LGBMClassifier, cv=3)
                    │    └── grid: {lr, n_estimators, max_depth}
                    ├── Evaluate: Accuracy, Precision, Recall, F1
                    ├── Log to MLflow:
                    │    ├── params (grid + best)
                    │    ├── metrics
                    │    ├── confusion_matrix.json (artifact)
                    │    └── sklearn model
                    └── Save: artifact/models/trained_model.pkl
```

### 5.2 Inference Flow (Real-Time, Streamlit)

```
[User fills Streamlit form]
         │
         ▼
   preprocess() in app.py
    ├── Build single-row DataFrame
    ├── Age binning (pd.cut)
    ├── Label encode via loaded encoders
    ├── Compute 5 engineered features
    └── Drop 'Age', align to model.feature_names_in_
         │
         ▼
   model.predict()        → pred  (0 or 1)
   model.predict_proba()  → prob  (0.0 – 1.0)
         │
         ▼
   Display: Risk Probability % + HIGH RISK / LOW RISK badge
```

---

## 6. Tech Stack & Justification

### 6.1 Data Processing Layer

| Library | Version | Justification |
|---|---|---|
| **pandas** | latest | Industry-standard tabular data manipulation; DataFrame API matches the relational schema of loan data perfectly |
| **numpy** | latest | Vectorised numerical operations (IQR calculation, boolean masking for `Young_High_Risk`) |
| **scipy** | latest | Statistical functions for outlier analysis during EDA |
| **imbalanced-learn** | latest | SMOTE / class-balancing — available but not yet wired into training pipeline ⚠️ |

### 6.2 Machine Learning Layer

| Library | Justification |
|---|---|
| **scikit-learn** | Provides pipeline primitives (train/test split, GridSearchCV, LabelEncoder, mutual_info_classif), ensuring reproducibility |
| **LightGBM (LGBMClassifier)** | **Primary choice.** Leaf-wise tree growth is faster and more accurate than XGBoost on tabular data of this shape (~255K rows, 20 features). Native handling of categorical features reduces preprocessing burden. Low memory footprint (358 KB serialised model) |
| **XGBoost** | Benchmarked in `model_slection.py` but not selected; lacks the speed advantage of LightGBM on this dataset size |
| **RandomForest / GradientBoosting / SVC / etc.** | Benchmarked in `model_slection.py`; ensemble methods were top performers but LightGBM dominated on F1 |

### 6.3 Serving Layer

| Library | Justification |
|---|---|
| **Streamlit** | Enables rapid UI prototyping in pure Python. Zero front-end code required. Ideal for internal tooling and stakeholder demos. **Trade-off:** Not suited for high-concurrency production traffic (single-threaded per session) |
| **joblib** | De-facto standard for serialising scikit-learn/LightGBM models; supports memory-mapped loading for large models |

### 6.4 MLOps Layer

| Tool | Justification |
|---|---|
| **MLflow** | Tracks experiment runs, hyperparameters, metrics, and serialised models. Local file-based backend (`./mlruns`) is zero-infra; upgrading to a remote tracking server requires only changing `MLFLOW_TRACKING_URI` |
| **TensorBoard** | Used in `model_slection.py` (`SummaryWriter`) for visualising multi-model accuracy and confusion matrices during benchmarking |

### 6.5 Potential Bottlenecks

| # | Bottleneck | Severity | Description |
|---|---|---|---|
| 1 | **Class Imbalance** | 🔴 Critical | Confusion matrix shows 4,569 FN vs 214 TP. The model misses 95.5% of actual defaulters. `imbalanced-learn` is installed but unused |
| 2 | **No Test Set Usage** | 🔴 Critical | `test.csv` is created in ingestion but never used for final holdout evaluation; model trains and evaluates on the same ingested train split |
| 3 | **GridSearchCV on CPU** | 🟡 Medium | A 3×3×3 = 27-combination grid with cv=3 = 81 fits. Acceptable now, but will scale poorly as data or grid grows. Switch to `Optuna` (TPE) for smarter search |
| 4 | **Streamlit Scalability** | 🟡 Medium | Streamlit sessions are not thread-safe beyond basic use. Under concurrent users, replace with a **FastAPI** inference endpoint |
| 5 | **Encoder Leakage Risk** | 🟡 Medium | `utils/helpers.py::label_encode` reuses a single `LabelEncoder` instance across all columns — will produce incorrect mappings. `feature_engineering.py` correctly uses per-column encoders |
| 6 | **Typo in Cursor** | 🟠 Low | `database_extraction.py` line 49: `corsor.execute(query)` (typo: `corsor` should be `cursor`) — will throw `NameError` if MySQL path is activated |
| 7 | **Local File Paths** | 🟠 Low | All paths are relative strings from `paths_config.py`. This will break if the working directory is not the project root |

---

## 7. MLOps & Experiment Tracking

```
MLflow Lifecycle per Training Run
──────────────────────────────────
 mlflow.set_experiment("Airline_Default_Prediction")   ← [⚠️ name mismatch]
 mlflow.start_run()
   ├── mlflow.log_params({grid params})
   ├── mlflow.log_params({best params})
   ├── mlflow.log_metric(accuracy, precision, recall, f1_score)
   ├── mlflow.log_artifact("confusion_matrix.json")
   └── mlflow.sklearn.log_model(best_model, "model")
```

> **⚠️ Issue Found:** Experiment name is `"Airline_Default_Prediction"` (copy-paste from a previous project). Should be `"Loan_Default_Prediction"`.

---

## 8. Artifact & State Management

| Path | Content | Produced By | Consumed By |
|---|---|---|---|
| `artifact/raw/Loan_default.csv` | Raw source data | Manual / DB Extractor | `DataIngestion` |
| `artifact/ingested_data/train.csv` | 80% training split | `DataIngestion` | `DataProcessor` |
| `artifact/ingested_data/test.csv` | 20% test split | `DataIngestion` | *(unused in pipeline)* |
| `artifact/processed_data/processed_train.csv` | Cleaned, outlier-clipped data | `DataProcessor` | `FeatureEngineer` |
| `artifact/engineered_data/final_df.csv` | Feature-engineered, encoded data | `FeatureEngineer` | `ModelTraining` |
| `artifact/models/trained_model.pkl` | Serialised LGBMClassifier | `ModelTraining` | `app.py` |
| `artifact/models/encoding_obj.pkl` | Dict of LabelEncoder objects | `FeatureEngineer` | `app.py` |
| `confusion_matrix.json` | Eval metrics | `ModelTraining` | MLflow |
| `mlruns/` | MLflow run database | `ModelTraining` | `mlflow ui` |
| `logs/*.log` | Timestamped step-by-step logs | All modules | DevOps / Debugging |

---

## 9. Logging & Observability

The system uses Python's built-in `logging` module with file-based output. Each module calls `get_logger(__name__)` which returns a named logger configured at `INFO` level.

```
Log Format: [YYYY-MM-DD HH:MM:SS,ms] module_name - LEVEL - message
Log Destination: logs/YYYY-MM-DD_HH-MM-SS.log (new file per process start)
```

**Gaps for production:**
- No log rotation (will grow unbounded)
- No structured/JSON logging (makes log aggregation tools like ELK/Loki harder)
- No `ERROR` / `CRITICAL` level alerting hooks

---

## 10. Bottleneck Analysis & Scaling Recommendations

| Priority | Issue | Recommended Fix |
|---|---|---|
| 🔴 P0 | Class imbalance (95.5% FN rate) | Apply `SMOTE` from `imbalanced-learn` in `FeatureEngineer`; tune decision threshold via PR-AUC |
| 🔴 P0 | Unused holdout test set | Feed `test.csv` into final evaluation in `ModelTraining.run()` |
| 🟡 P1 | GridSearchCV → slow hyper-tuning | Replace with `Optuna` (Bayesian optimisation); parallelize with `n_jobs=-1` |
| 🟡 P1 | Streamlit → not scalable | Package model as a **FastAPI** REST endpoint; deploy on Kubernetes or AWS Lambda |
| 🟡 P1 | Experiment name typo | Fix `"Airline_Default_Prediction"` → `"Loan_Default_Prediction"` in `model_training.py` |
| 🟠 P2 | Relative paths | Use `pathlib.Path(__file__).resolve()` anchored to project root |
| 🟠 P2 | `utils/helpers.py` encoder bug | Fix shared `LabelEncoder` reuse (already correctly fixed in `feature_engineering.py`) |
| 🟠 P2 | `database_extraction.py` typo | Fix `corsor` → `cursor` on line 49 |

---

## 11. Deployment Architecture (Target State)

```
                        ┌──────────────────────┐
                        │   GitHub Actions CI   │
                        │  (trigger on push)    │
                        └──────────┬───────────┘
                                   │
                          Run main.py pipeline
                                   │
                         ┌─────────▼─────────┐
                         │  MLflow Registry   │
                         │  (Remote S3/DB)    │
                         └─────────┬─────────┘
                                   │  "Production" model alias
                          ┌────────▼──────────┐
                          │  FastAPI Service   │
                          │  POST /predict     │
                          │  GET  /health      │
                          └────────┬──────────┘
                                   │
                  ┌────────────────▼──────────────────┐
                  │          Load Balancer             │
                  └────────┬──────────────┬───────────┘
                           │              │
                    ┌──────▼───┐   ┌──────▼───┐
                    │ Replica 1│   │ Replica 2│    (Horizontal Scaling)
                    └──────────┘   └──────────┘
```

---

*Created by Zainul Abedeen · April 2026*

