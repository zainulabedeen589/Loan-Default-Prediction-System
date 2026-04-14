# Low-Level Design (LLD)
# Loan Default Prediction System

> **Version:** 1.0.0  
> **Author:** Zainul Abedeen  
> **Date:** April 2026

---

## Table of Contents

1. [Module / Class Diagrams](#1-module--class-diagrams)
2. [Class Responsibility Catalogue](#2-class-responsibility-catalogue)
3. [Sequence Diagrams](#3-sequence-diagrams)
4. [Dataset Schema & Feature Dictionary](#4-dataset-schema--feature-dictionary)
5. [Engineered Feature Definitions](#5-engineered-feature-definitions)
6. [Feature Selection Results](#6-feature-selection-results)
7. [Encoding Strategy](#7-encoding-strategy)
8. [Model Configuration & Hyperparameter Space](#8-model-configuration--hyperparameter-space)
9. [Artifact Schema](#9-artifact-schema)
10. [API Specification (Target State)](#10-api-specification-target-state)
11. [Error Handling Design](#11-error-handling-design)
12. [Inter-Module Dependency Map](#12-inter-module-dependency-map)
13. [Known Bugs & Code-Level Issues](#13-known-bugs--code-level-issues)

---

## 1. Module / Class Diagrams

### 1.1 `src/data_ingestion.py` — `DataIngestion`

```
┌──────────────────────────────────────────────────────────┐
│                      DataIngestion                        │
├──────────────────────────────────────────────────────────┤
│ - raw_data_path   : str                                   │
│ - ingested_data_dir : str                                 │
├──────────────────────────────────────────────────────────┤
│ + __init__(raw_data_path, ingested_data_dir)              │
│ + create_ingested_data_dir() → None                       │
│ + split_data(train_path, test_path,                       │
│              test_size=0.2, random_state=42) → None       │
└──────────────────────────────────────────────────────────┘
         │ raises
         ▼
  CustomException
```

### 1.2 `src/data_processing.py` — `DataProcessor`

```
┌──────────────────────────────────────────────────────────┐
│                      DataProcessor                        │
├──────────────────────────────────────────────────────────┤
│ - train_path          : str  (= TRAIN_DATA_PATH)          │
│ - processed_data_path : str  (= PROCESSED_DATA_PATH)      │
├──────────────────────────────────────────────────────────┤
│ + __init__()                                              │
│ + load_data() → pd.DataFrame                              │
│ + drop_unnecessary_columns(df, columns) → pd.DataFrame   │
│ + handle_outliers(df, columns) → pd.DataFrame            │
│ + handle_null_values(df, columns) → pd.DataFrame         │
│ + save_data(df) → None                                    │
│ + run() → None           [orchestrates above methods]    │
└──────────────────────────────────────────────────────────┘
```

**Outlier strategy:** IQR clipping  
`lower = Q1 − 1.5×IQR`,  `upper = Q3 + 1.5×IQR`  
Applied to **all numeric columns** except `Default`.

### 1.3 `src/feature_engineering.py` — `FeatureEngineer`

```
┌──────────────────────────────────────────────────────────┐
│                     FeatureEngineer                       │
├──────────────────────────────────────────────────────────┤
│ - data_path     : str  (= PROCESSED_DATA_PATH)           │
│ - encoder_path  : str  (= ENCODER_SAVE_PATH)             │
│ - df            : pd.DataFrame                           │
│ - label_mapping : dict                                   │
├──────────────────────────────────────────────────────────┤
│ + __init__()                                              │
│ + load_data() → None                                     │
│ + bin_age() → None               [creates Age_Group]     │
│ + label_encoding() → None        [fit+transform+save]    │
│ + feature_construction() → None  [5 derived features]   │
│ + feature_selection() → None     [mutual info top 12]    │
│ + save_data() → None                                     │
│ + run() → None           [orchestrates pipeline order]   │
└──────────────────────────────────────────────────────────┘
```

**Pipeline execution order in `run()`:**

```
load_data()
    → bin_age()
        → label_encoding()
            → feature_construction()
                → feature_selection()
                    → save_data()
```

### 1.4 `src/model_training.py` — `ModelTraining`

```
┌──────────────────────────────────────────────────────────────┐
│                       ModelTraining                           │
├──────────────────────────────────────────────────────────────┤
│ - data_path       : str                                       │
│ - params_path     : str                                       │
│ - model_save_path : str                                       │
│ - experiment_name : str  (default: "Airline_Default_Pred...")│
│ - best_model      : LGBMClassifier | None                    │
│ - metrics         : dict | None                              │
│ - tracking_uri    : str  (file://.../mlruns)                 │
├──────────────────────────────────────────────────────────────┤
│ + __init__(data_path, params_path, model_save_path,          │
│            experiment_name)                                   │
│ + load_data() → pd.DataFrame                                 │
│ + split_data(data) → X_train, X_test, y_train, y_test        │
│ + train_model(X_train, y_train, params) → best_params: dict  │
│ + evaluate_model(X_test, y_test) → metrics: dict             │
│ + save_model() → None                                         │
│ + run() → None    [full MLflow-instrumented run]              │
└──────────────────────────────────────────────────────────────┘
```

### 1.5 `src/model_slection.py` — `ModelSelection` *(Experimental)*

```
┌──────────────────────────────────────────────────────────┐
│                     ModelSelection                        │
├──────────────────────────────────────────────────────────┤
│ - data_path  : str                                        │
│ - writer     : SummaryWriter  (TensorBoard)              │
│ - models     : dict[str, Estimator]   (10 classifiers)  │
│ - results    : dict                                      │
├──────────────────────────────────────────────────────────┤
│ + __init__(data_path)                                     │
│ + load_data() → X, y                                     │
│ + split_data(X, y) → X_train, X_test, y_train, y_test    │
│ + train_and_evaluate(...) → None  [fills self.results]   │
│ + log_confusion_matrix(...) → None  [TensorBoard fig]    │
│ + run() → None                                           │
└──────────────────────────────────────────────────────────┘
```

**Candidate models benchmarked:**

| # | Model | Library |
|---|---|---|
| 1 | Logistic Regression | sklearn |
| 2 | Random Forest | sklearn |
| 3 | Gradient Boosting | sklearn |
| 4 | AdaBoost | sklearn |
| 5 | Support Vector Classifier | sklearn |
| 6 | K-Nearest Neighbors | sklearn |
| 7 | Naive Bayes | sklearn |
| 8 | Decision Tree | sklearn |
| 9 | **LightGBM** ✅ (selected) | lightgbm |
| 10 | XGBoost | xgboost |

### 1.6 `src/database_extraction.py` — `MySQLDataExtractor`

```
┌──────────────────────────────────────────────────────────┐
│                  MySQLDataExtractor                       │
├──────────────────────────────────────────────────────────┤
│ - host        : str                                       │
│ - user        : str                                       │
│ - password    : str                                       │
│ - database    : str                                       │
│ - table_name  : str                                       │
│ - connection  : mysql.connector.connection | None        │
├──────────────────────────────────────────────────────────┤
│ + __init__(db_config: dict)                              │
│ + connect() → None                                       │
│ + dissconnect() → None                                   │
│ + extract_to_csv(output_folder="./artifacts") → None     │
└──────────────────────────────────────────────────────────┘
```

### 1.7 Supporting Modules

```
┌───────────────────────────────┐    ┌──────────────────────────────────────┐
│    src/logger.py              │    │    src/custom_exception.py           │
├───────────────────────────────┤    ├──────────────────────────────────────┤
│ get_logger(name) → Logger     │    │ class CustomException(Exception)     │
│                               │    │  - error_message : str               │
│ Writes to:                    │    │  + __init__(msg, error_detail: sys)  │
│  logs/YYYY-MM-DD_HH-MM-SS.log │    │  + get_detailed_error_message()      │
│                               │    │    → "Error in {file}, line {n}: {m}"│
│ Format:                       │    │  + __str__() → error_message         │
│  [timestamp] name - LVL - msg │    └──────────────────────────────────────┘
└───────────────────────────────┘

┌───────────────────────────────┐    ┌──────────────────────────────────────┐
│    src/paths_config.py        │    │    utils/helpers.py                  │
├───────────────────────────────┤    ├──────────────────────────────────────┤
│ ARTIFACTS_DIR                 │    │ label_encode(df, columns)            │
│ RAW_DATA_PATH                 │    │  → (df, label_mappings)              │
│ INGESTED_DATA_DIR             │    │                                      │
│ TRAIN_DATA_PATH               │    │ ⚠️ BUG: Single LabelEncoder instance │
│ TEST_DATA_PATH                │    │ shared across all columns.           │
│ PROCESSED_DIR                 │    │ Classes from col N overwrite col N-1 │
│ PROCESSED_DATA_PATH           │    └──────────────────────────────────────┘
│ ENGINEERED_DIR                │
│ ENGINEERED_DATA_PATH          │
│ PARAMS_PATH                   │
│ MODEL_SAVE_PATH               │
│ ENCODER_SAVE_PATH             │
└───────────────────────────────┘
```

---

## 2. Class Responsibility Catalogue

| Class | Single Responsibility | Output |
|---|---|---|
| `DataIngestion` | Load raw CSV; split into train/test | `train.csv`, `test.csv` |
| `DataProcessor` | Clean data; clip outliers; drop identifiers | `processed_train.csv` |
| `FeatureEngineer` | Construct features; encode; select top 12 | `final_df.csv`, `encoding_obj.pkl` |
| `ModelTraining` | Tune LightGBM; evaluate; log + save | `trained_model.pkl`, MLflow run |
| `ModelSelection` | Benchmark 10 classifiers; visualise via TensorBoard | `tensorboard_logs/` |
| `MySQLDataExtractor` | Connect to MySQL; dump table to CSV | `train.csv` (alternative source) |
| `CustomException` | Enrich exceptions with filename + line number | Formatted error string |

---

## 3. Sequence Diagrams

### 3.1 Training Pipeline

```
main.py          DataIngestion   DataProcessor   FeatureEngineer   ModelTraining   MLflow
  │                  │               │                │                 │             │
  │── __init__() ──▶ │               │                │                 │             │
  │── create_dir() ─▶│               │                │                 │             │
  │── split_data() ─▶│               │                │                 │             │
  │                  │── read_csv ──▶│                │                 │             │
  │                  │── to_csv ────▶│                │                 │             │
  │                  │               │                │                 │             │
  │── run() ─────────────────────────▶               │                 │             │
  │                  │               │── load_data ──▶│                │             │
  │                  │               │── drop_cols ──▶│                │             │
  │                  │               │── clip_IQR ───▶│                │             │
  │                  │               │── save_csv ───▶│                │             │
  │                  │               │                │                 │             │
  │── run() ─────────────────────────────────────────▶                 │             │
  │                  │               │                │── bin_age      │             │
  │                  │               │                │── encode       │             │
  │                  │               │                │── construct_fe │             │
  │                  │               │                │── select_top12 │             │
  │                  │               │                │── save_csv     │             │
  │                  │               │                │── save_pkl ───▶│             │
  │                  │               │                │                 │             │
  │── run() ─────────────────────────────────────────────────────────▶ │             │
  │                  │               │                │                 │─ start_run ▶│
  │                  │               │                │                 │─ GridSearch │
  │                  │               │                │                 │─ evaluate   │
  │                  │               │                │                 │─ log_params▶│
  │                  │               │                │                 │─ log_metrics▶
  │                  │               │                │                 │─ log_model ▶│
  │                  │               │                │                 │─ save_pkl   │
  │◀─────────────────────────────────────────────────────────────────── │             │
```

### 3.2 Inference Sequence (Streamlit)

```
Browser         app.py              joblib              LGBMClassifier
  │                │                   │                      │
  │── form submit ▶│                   │                      │
  │                │── load model ────▶│                      │
  │                │── load encoders ─▶│                      │
  │                │                   │                      │
  │                │── preprocess()    │                      │
  │                │   ├ pd.cut(Age)   │                      │
  │                │   ├ le.transform()│                      │
  │                │   ├ compute 5 FEs │                      │
  │                │   └ reorder cols  │                      │
  │                │                   │                      │
  │                │── model.predict() ─────────────────────▶ │
  │                │── model.predict_proba() ────────────────▶│
  │                │◀──────────────────────────── pred, prob ─│
  │◀── render result│                   │                      │
```

---

## 4. Dataset Schema & Feature Dictionary

**Source:** `Loan_default.csv` (`artifact/raw/`)

| Column | Type | Description | Role |
|---|---|---|---|
| `LoanID` | string | Unique loan identifier | Dropped (identifier leak) |
| `Age` | int | Applicant age in years | Input → Binned → Dropped |
| `Income` | float | Annual income (USD) | Input feature |
| `LoanAmount` | float | Principal loan amount (USD) | Input feature |
| `CreditScore` | int | Credit bureau score (300–850) | Input feature |
| `MonthsEmployed` | int | Months at current employer | Input feature |
| `NumCreditLines` | int | Total open credit lines | Input feature |
| `InterestRate` | float | Annual interest rate (%) | Input feature |
| `LoanTerm` | int | Loan duration in months | Input feature |
| `DTIRatio` | float | Debt-to-Income ratio | Input feature |
| `Education` | string | Highest education level | Categorical → Encoded |
| `EmploymentType` | string | Employment status | Categorical → Encoded |
| `MaritalStatus` | string | Marital status | Categorical → Encoded |
| `HasMortgage` | string | Whether has a mortgage | Categorical → Encoded |
| `HasDependents` | string | Whether has dependents | Categorical → Encoded |
| `LoanPurpose` | string | Reason for loan | Categorical → Encoded |
| `HasCoSigner` | string | Whether has a co-signer | Categorical → Encoded |
| **`Default`** | int | **Target: 1=Default, 0=No Default** | Label |

**Estimated shape:** ~255,000 rows × 18 columns  
**Class ratio:** Heavily imbalanced (Default=1 is a minority class, ~12%)

---

## 5. Engineered Feature Definitions

| Feature | Formula | Business Rationale |
|---|---|---|
| `Age_Group` | `pd.cut(Age, bins=[0,18,30,45,60,100])` → 5 ordinal labels | Age brackets carry different default risk profiles |
| `Monthly_Payment` | `LoanAmount × (1 + InterestRate/100) / LoanTerm` | Actual monthly cash outflow |
| `PTI_Ratio` | `Monthly_Payment / (Income / 12)` | Payment-to-Income ratio — key affordability metric |
| `Job_Stability_Index` | `MonthsEmployed / (Age × 12)` | Fraction of adult working life employed; proxy for stability |
| `Debt_per_Line` | `LoanAmount / (NumCreditLines + 1)` | Average debt burden per credit line; `+1` avoids div/0 |
| `Young_High_Risk` | `(Age < 30) AND (LoanAmount > median(LoanAmount))` → 0/1 | Binary flag for young borrowers with disproportionately large loans |

**Age Binning Labels:**

| Bin | Range | Label |
|---|---|---|
| 1 | 0 – 18 | Child |
| 2 | 18 – 30 | Teenager |
| 3 | 30 – 45 | Adult |
| 4 | 45 – 60 | Senior |
| 5 | 60 – 100 | Super Senior |

---

## 6. Feature Selection Results

**Method:** `sklearn.feature_selection.mutual_info_classif`  
**Strategy:** Top 12 features by mutual information score retained; `Default` column appended.

Columns encoded before selection:

| Column | Encoding |
|---|---|
| `Age_Group` | LabelEncoder |
| `Education` | LabelEncoder |
| `EmploymentType` | LabelEncoder |
| `MaritalStatus` | LabelEncoder |
| `HasMortgage` | LabelEncoder |
| `HasDependents` | LabelEncoder |
| `LoanPurpose` | LabelEncoder |
| `HasCoSigner` | LabelEncoder |

> **Note:** Feature names in `model.feature_names_in_` (LightGBM attribute) define the exact column order required at inference time. The `app.py` uses `df = df[model.feature_names_in_]` to enforce this contract.

---

## 7. Encoding Strategy

### Label Encoding (per-column, independent encoders)

```python
# Correct implementation in feature_engineering.py
encoders = {}
for col in columns_to_encode:
    le = LabelEncoder()          # Fresh encoder per column
    df[col] = le.fit_transform(df[col])
    encoders[col] = le           # Store for inference

joblib.dump(encoders, ENCODER_SAVE_PATH)
# Saves: artifact/models/encoding_obj.pkl
```

### Inference-time decoding

```python
# In app.py
encoders = joblib.load(ENCODER_PATH)       # dict of LabelEncoders
for col, le in encoders.items():
    df[col] = le.transform(df[col])        # Same fitted encoder
```

**Known issue in `utils/helpers.py`:** Shared `le` instance overwrites class mapping with each iteration — this module is **not** used in the current production path but must be fixed before any reuse.

---

## 8. Model Configuration & Hyperparameter Space

### Base Model

```python
lgb.LGBMClassifier()   # Default objective: binary
```

### GridSearchCV Configuration

```json
{
  "learning_rate":  [0.01, 0.05, 0.1],
  "n_estimators":   [100, 200, 300],
  "max_depth":      [5, 10, 15]
}
```

| Parameter | Search Space | Notes |
|---|---|---|
| `learning_rate` | `{0.01, 0.05, 0.1}` | Lower values require more trees |
| `n_estimators` | `{100, 200, 300}` | Total boosting rounds |
| `max_depth` | `{5, 10, 15}` | Controls tree depth; `-1` = unlimited in default LGBM |

**Total combinations:** 3 × 3 × 3 = **27**  
**CV folds:** 3  
**Total model fits:** **81**  
**Scoring:** `accuracy` ← *should be changed to `f1` or `roc_auc` given class imbalance*

### Final Serialised Model

| Property | Value |
|---|---|
| File | `artifact/models/trained_model.pkl` |
| Size | **~358 KB** |
| Format | joblib pickle |
| Interface | `.predict(X)`, `.predict_proba(X)`, `.feature_names_in_` |

### Evaluation Metrics (Latest Run)

| Metric | Value | Notes |
|---|---|---|
| TN | 35,938 | Correctly rejected |
| FP | 135 | Good borrowers flagged as risky |
| FN | **4,569** | **⚠️ Actual defaulters missed** |
| TP | 214 | Correctly caught defaulters |
| Recall (Default class) | ~4.5% | Critically low |

---

## 9. Artifact Schema

### `artifact/models/encoding_obj.pkl`

**Type:** `dict[str, sklearn.preprocessing.LabelEncoder]`

```python
{
  'Age_Group':      LabelEncoder(classes_=['Adult', 'Child', 'Senior', 'Super Senior', 'Teenager']),
  'Education':      LabelEncoder(classes_=['Bachelor', 'High School', 'Master', 'PhD']),
  'EmploymentType': LabelEncoder(classes_=['Full-time', 'Part-time', 'Self-employed', 'Unemployed']),
  'MaritalStatus':  LabelEncoder(classes_=['Divorced', 'Married', 'Single']),
  'HasMortgage':    LabelEncoder(classes_=['No', 'Yes']),
  'HasDependents':  LabelEncoder(classes_=['No', 'Yes']),
  'LoanPurpose':    LabelEncoder(classes_=['Auto', 'Business', 'Education', 'Home', 'Other']),
  'HasCoSigner':    LabelEncoder(classes_=['No', 'Yes'])
}
```

### `artifact/engineered_data/final_df.csv`

Top 12 selected features + `Default` = **13 columns**.  
Features ordered by Mutual Information score descending.  
All columns are numeric after encoding.

### `confusion_matrix.json`

```json
{
  "confusion_matrix": [[35938, 135], [4569, 214]]
}
```

---

## 10. API Specification (Target State)

*The system currently has no REST API. The following specification defines the recommended FastAPI interface for production deployment.*

---

### `POST /api/v1/predict`

**Description:** Accepts a loan applicant's features and returns default risk prediction.

**Request:**

```http
POST /api/v1/predict
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "Age": 34,
  "Income": 75000,
  "LoanAmount": 150000,
  "CreditScore": 620,
  "MonthsEmployed": 48,
  "NumCreditLines": 4,
  "InterestRate": 12.5,
  "LoanTerm": 60,
  "DTIRatio": 0.38,
  "Education": "Bachelor",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "Yes",
  "HasDependents": "No",
  "LoanPurpose": "Home",
  "HasCoSigner": "No"
}
```

**Response `200 OK`:**

```json
{
  "prediction": 0,
  "label": "LOW RISK",
  "default_probability": 0.1342,
  "model_version": "lgbm-v1.2.0",
  "request_id": "a3f8c1d2-b7e4-4a91-9f3c-d2e1b7a8c9f0"
}
```

**Response `422 Unprocessable Entity`:**

```json
{
  "error": "ValidationError",
  "detail": [
    {
      "field": "CreditScore",
      "message": "value must be between 300 and 850"
    }
  ]
}
```

**Response `500 Internal Server Error`:**

```json
{
  "error": "InferenceError",
  "detail": "Model preprocessing failed: unexpected category in Education",
  "request_id": "a3f8c1d2-..."
}
```

---

### `GET /api/v1/health`

**Description:** Service liveness and readiness check.

**Response `200 OK`:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "lgbm-v1.2.0",
  "uptime_seconds": 3842
}
```

---

### `GET /api/v1/model/info`

**Description:** Returns current model metadata.

**Response `200 OK`:**

```json
{
  "model_type": "LGBMClassifier",
  "features": ["LoanAmount", "CreditScore", "Income", "..."],
  "num_features": 12,
  "training_date": "2026-04-14",
  "mlflow_run_id": "abc123def456",
  "metrics": {
    "accuracy": 0.882,
    "precision": 0.876,
    "recall": 0.882,
    "f1_score": 0.873
  }
}
```

---

### `POST /api/v1/predict/batch`

**Description:** Batch prediction for multiple applicants.

**Request:**

```json
{
  "records": [
    { "Age": 34, "Income": 75000, "..." : "..." },
    { "Age": 45, "Income": 120000, "..." : "..." }
  ]
}
```

**Response `200 OK`:**

```json
{
  "predictions": [
    { "index": 0, "prediction": 0, "default_probability": 0.13 },
    { "index": 1, "prediction": 1, "default_probability": 0.72 }
  ],
  "total": 2,
  "processing_time_ms": 38
}
```

---

### HTTP Status Code Reference

| Code | Meaning | When Used |
|---|---|---|
| `200` | OK | Successful prediction or query |
| `400` | Bad Request | Malformed JSON body |
| `401` | Unauthorized | Missing/invalid Bearer token |
| `422` | Unprocessable Entity | Schema validation failure (Pydantic) |
| `500` | Internal Server Error | Model inference crash |
| `503` | Service Unavailable | Model not yet loaded / warming up |

---

## 11. Error Handling Design

### `CustomException` Flow

```
Module method raises Python Exception
         │
         ▼
CustomException.__init__(error_message, sys)
         │
         ├── sys.exc_info() → extracts traceback
         ├── exc_tb.tb_frame.f_code.co_filename → file name
         └── exc_tb.tb_lineno → line number
         │
         ▼
Formatted: "Error in {file}, line {n}: {msg}"
         │
         ▼
logger.error(str(ce))   ← logged to timestamped .log file
```

### `app.py` Error Handling

```python
try:
    df = preprocess()
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
except Exception as e:
    st.error(f"Error: {e}")    # Surfaces raw exception to UI
```

> **Gap:** Production apps should catch specific exceptions, log them server-side, and display sanitised user messages. Raw exception strings may expose internal paths.

---

## 12. Inter-Module Dependency Map

```
main.py
 ├── src.data_ingestion.DataIngestion
 │    ├── src.logger.get_logger
 │    ├── src.custom_exception.CustomException
 │    └── src.paths_config.*
 │
 ├── src.data_processing.DataProcessor
 │    ├── src.logger.get_logger
 │    ├── src.custom_exception.CustomException
 │    └── src.paths_config.*
 │
 ├── src.feature_engineering.FeatureEngineer
 │    ├── src.logger.get_logger
 │    ├── src.custom_exception.CustomException
 │    ├── src.paths_config.*
 │    └── utils.helpers.*       (imported but not called in run())
 │
 └── src.model_training.ModelTraining
      ├── src.logger.get_logger
      ├── src.custom_exception.CustomException
      ├── src.paths_config.*
      └── mlflow, mlflow.sklearn

app.py (independent — runtime serving)
 ├── joblib.load(MODEL_PATH)      → LGBMClassifier
 └── joblib.load(ENCODER_PATH)    → dict[LabelEncoder]
```

---

## 13. Known Bugs & Code-Level Issues

| File | Line | Bug | Impact | Fix |
|---|---|---|---|---|
| `src/database_extraction.py` | 49 | `corsor.execute(query)` — `corsor` is undefined | Runtime `NameError` if MySQL path used | Change to `cursor.execute(query)` |
| `src/model_training.py` | 19 | `experiment_name="Airline_Default_Prediction"` — wrong project name | MLflow experiment logged under wrong name | Change to `"Loan_Default_Prediction"` |
| `src/model_slection.py` | 53–54 | `X = df.drop('satisfaction')`, `y = df['satisfaction']` — wrong column name | `KeyError` at runtime | Change to `'Default'` |
| `utils/helpers.py` | 5–10 | Single `LabelEncoder` reused across all columns | Incorrect label mappings (column N overwrites N-1) | Instantiate `le = LabelEncoder()` inside loop |
| `src/model_training.py` | 59 | `scoring='accuracy'` in GridSearchCV | Optimises for majority class; ignores defaulters | Change to `scoring='f1'` or `'roc_auc'` |
| `main.py` | 19 | `test.csv` generated but never used for evaluation | No true holdout evaluation exists | Pass `test.csv` to `evaluate_model()` in `ModelTraining` |
```

---

*Created by Zainul Abedeen · April 2026*
