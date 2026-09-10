# Early Warning System — Student Risk Dashboard

Design and Evaluation of a Transformer-Based Early Warning System for
At-Risk Student Identification in Higher Education

Eashkumar Kaki | Student No: 35057938

## What this is

A Streamlit dashboard showing the actual predictions from the trained
XGBoost model (OULAD, weeks 0-4 features): 15 real test-set students,
their predicted risk score, top SHAP-derived contributing factors, and
recommended tutor action. Built for the UAT session with academic
stakeholders.

## Setup (one-time)

Open a terminal in this folder and run:

```
pip install -r requirements.txt
```

## Run

```
streamlit run streamlit_dashboard.py
```

A browser tab opens automatically at `http://localhost:8501`. If it
doesn't open on its own, copy that address into a browser manually.

To stop the app, go back to the terminal and press `Ctrl + C`.

## Files

- `streamlit_dashboard.py` — the dashboard app
- `dashboard_sample_output.csv` — real model predictions, loaded automatically
- `requirements.txt` — the two packages needed (streamlit, pandas)

## Notes for the UAT session

The app runs entirely on this machine, nothing is sent anywhere.
Participants only need to look at the browser tab, click through
students in the left-hand register, and answer the Google Form
questionnaire afterwards.
