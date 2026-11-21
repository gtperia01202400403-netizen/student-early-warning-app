# Student Early-Warning Predictive Web Application (Machine Problem 3)

## Overview
A Streamlit web application that predicts whether a student is **Likely to Pass** or **At Risk of Failing** based on 8 Learning Management System (LMS) behavioral predictors from `augmented_balanced_dataset_m.csv`.

## Repository Contents
- `app.py`: Main Streamlit web application with single-student scoring, explainability comparison against cohort medians, human-in-the-loop governance safeguards, and batch CSV scoring.
- `student_success_model.joblib`: Pre-trained Random Forest classifier (`88.2%` test accuracy, `0.952` ROC-AUC).
- `augmented_balanced_dataset_m.csv`: Balanced dataset (`3,602` records: `1,801` High / `1,801` Low).
- `requirements.txt`: Python package dependencies for cloud deployment.

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Deploy to Streamlit Community Cloud (Free Public URL)
1. Create a new GitHub repository and upload all files in this folder (`app.py`, `requirements.txt`, `student_success_model.joblib`, `augmented_balanced_dataset_m.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, set **Main file path** to `app.py`, and click **Deploy!**.
4. Copy the live `https://...streamlit.app` URL and submit it.
