# Transformer-Based-Early-Warning-System-for-At-Risk-Student-Identification-in-Higher-Education
### for At-Risk Student Identification in Higher Education

MSc Computing Research Project — Eashkumar Kaki (35057938), Sheffield Hallam University

## Overview
A comparative early-warning system predicting at-risk students using
the OULAD dataset, comparing Logistic Regression, Random Forest,
XGBoost, LSTM and a custom Transformer model, with SHAP/attention
explainability, fairness auditing, and a Streamlit dashboard prototype.

## Dataset
Open University Learning Analytics Dataset (OULAD):
- UCI: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- Kaggle mirror: https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad

## Repository Structure
- `notebook/` — full Kaggle notebook (data pipeline, 5 models, SHAP,
  fairness evaluation, dashboard data generation)
- `dashboard/` — standalone Streamlit dashboard prototype used in UAT
- `outputs/` — generated charts and result tables
- `docs/` — supporting project documents

## Running the notebook
Upload `notebook/OULAD_Transformer_Early_Warning_System.ipynb` to
Kaggle, attach the OULAD dataset, select GPU P100, and run all cells.

## Running the dashboard
cd dashboard
pip install -r requirements.txt
streamlit run streamlit_dashboard.py

## Results Summary
| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| XGBoost | 0.749 | 0.834 | 0.869 |
| Random Forest | 0.748 | 0.838 | 0.868 |
| Logistic Regression | 0.731 | 0.803 | 0.835 |
| LSTM | 0.695 | 0.776 | 0.809 |
| Transformer | 0.668 | 0.749 | 0.790 |

## Ethics
This project received UREC 2 ethical approval from Sheffield Hallam
University (30 July 2026).
