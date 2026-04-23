# Diabetes Readmission Prediction – ML Pipeline Project

An end-to-end machine learning project predicting 30-day hospital readmission
in diabetic patients, built as part of my university ML course. The dataset is
sourced from the UCI Machine Learning Repository.

### Problem Statement

Binary classification task: will a diabetic patient be readmitted within 30 days
of discharge? Given the class imbalance (~9% positive class), the pipeline
prioritises recall over accuracy in a clinical setting, since missing a readmission
is costlier than a false alarm.

### Pipeline Overview

| Notebook                                         | Description                                                   | Status         |
| ------------------------------------------------ | ------------------------------------------------------------- | -------------- |
| `01_eda.ipynb`                                   | Exploratory data analysis                                     | ✅ Done        |
| `02_preprocessing_and_feature_engineering.ipynb` | Cleaning, imputation, feature engineering, encoding           | ✅ Done        |
| `03_model_training_and_evaluation.ipynb`         | Model training, comparison, threshold tuning, test evaluation | ✅ Done        |
| `04_model_explainability.ipynb`                  | SHAP-based explainability                                     | 🔄 In progress |

### Key Results

- **Final model:** XGBoost with decision threshold of 0.4
- **Test ROC-AUC:** 0.797 (literature benchmark: 0.642)
- **Test Recall:** 0.807 — correctly flags 8 out of 10 readmissions

### Notable Design Decisions

- Split performed before imputation to prevent data leakage
- Rare categories dropped from training only, val and test sets untouched
- Individual medication columns replaced by aggregate `diabetesMed` flag
- ICD-9 codes mapped to 11 clinical categories to reduce cardinality
- Threshold tuned on validation set before final test evaluation
