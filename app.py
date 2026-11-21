import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Page Configuration & Design Principles (Slide Principles)
# 1. Decision-first | 2. Consistent hierarchy | 3. Fast scanning
# 4. Meaningful defaults | 5. Accessible color | 6. Visible freshness
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Early-Warning Predictive App",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling with accessible, colorblind-safe palette (Teal & Amber/Coral)
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border-left: 5px solid #0d9488;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .alert-card {
        background-color: #fff7ed;
        border-left: 5px solid #ea580c;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 0.9rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "augmented_balanced_dataset_m.csv"
MODEL_PATH = "student_success_model.joblib"

FEATURE_INFO = {
    "Submission.Actions": {
        "label": "Submission Actions",
        "help": "Total assignment and coursework submission events in the LMS.",
        "min": 0,
        "max": 100,
        "default": 4,
    },
    "Quiz.Actions": {
        "label": "Quiz Actions",
        "help": "Total quiz attempts, views, and interactions.",
        "min": 0,
        "max": 1000,
        "default": 15,
    },
    "Forum.Actions": {
        "label": "Forum Actions",
        "help": "Posts, replies, and discussions viewed or created in course forums.",
        "min": 0,
        "max": 300,
        "default": 5,
    },
    "Create.Actions": {
        "label": "Create Actions",
        "help": "Total content creation events across LMS activities.",
        "min": 0,
        "max": 200,
        "default": 8,
    },
    "Read.Actions": {
        "label": "Read Actions",
        "help": "Course materials, pages, and resources accessed or read.",
        "min": 0,
        "max": 50000,
        "default": 135,
    },
    "Update.Actions": {
        "label": "Update Actions",
        "help": "Edits or updates made to submissions, profiles, or forum posts.",
        "min": 0,
        "max": 500,
        "default": 3,
    },
    "Delete.Actions": {
        "label": "Delete Actions",
        "help": "Deletion actions recorded in the LMS.",
        "min": 0,
        "max": 20,
        "default": 0,
    },
    "Days.Logged.In": {
        "label": "Days Logged In",
        "help": "Number of distinct days the student logged into the LMS.",
        "min": 0,
        "max": 90,
        "default": 14,
    },
}


@st.cache_resource
def load_or_train_model():
    df = pd.read_csv(DATA_PATH)
    feature_cols = list(FEATURE_INFO.keys())
    X = df[feature_cols]
    y = (df["rating"] == "High").astype(int)  # 1 = Likely to Pass (High), 0 = At-Risk (Low)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "n_records": len(df),
    }
    medians = df.groupby("rating")[feature_cols].median()
    return model, df, metrics, medians, feature_cols


model, df, metrics, medians, feature_cols = load_or_train_model()

# ---------------------------------------------------------
# Sidebar: Student Activity Inputs (Meaningful Defaults)
# ---------------------------------------------------------
st.sidebar.header("Student LMS Activity Inputs")
st.sidebar.caption("Adjust student engagement predictors to generate a real-time prediction.")

preset = st.sidebar.selectbox(
    "Quick Test Presets",
    ["Custom Input (Cohort Median)", "Sample Passing Student", "Sample At-Risk Student"],
)

preset_values = {
    "Custom Input (Cohort Median)": {k: v["default"] for k, v in FEATURE_INFO.items()},
    "Sample Passing Student": {
        "Submission.Actions": 10,
        "Quiz.Actions": 35,
        "Forum.Actions": 12,
        "Create.Actions": 14,
        "Read.Actions": 320,
        "Update.Actions": 6,
        "Delete.Actions": 0,
        "Days.Logged.In": 22,
    },
    "Sample At-Risk Student": {
        "Submission.Actions": 1,
        "Quiz.Actions": 2,
        "Forum.Actions": 0,
        "Create.Actions": 3,
        "Read.Actions": 45,
        "Update.Actions": 1,
        "Delete.Actions": 0,
        "Days.Logged.In": 6,
    },
}

user_inputs = {}
for col in feature_cols:
    meta = FEATURE_INFO[col]
    user_inputs[col] = st.sidebar.number_input(
        label=meta["label"],
        min_value=meta["min"],
        max_value=meta["max"],
        value=int(preset_values[preset][col]),
        step=1,
        help=meta["help"],
    )

# ---------------------------------------------------------
# Main Header & Governance Metadata (Visible Freshness)
# ---------------------------------------------------------
st.title("Student Early-Warning Predictive Application")
st.markdown(
    "Predicts whether a student is **Likely to Pass** or **At Risk of Failing** using LMS behavioral predictors."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Model Algorithm", "Random Forest")
kpi2.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
kpi3.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")
kpi4.metric("Training Cohort", f"{metrics['n_records']:,} records")

st.divider()

# ---------------------------------------------------------
# Decision-First Prediction & Explainability
# ---------------------------------------------------------
input_df = pd.DataFrame([user_inputs])
prob_pass = float(model.predict_proba(input_df)[0][1])
prob_fail = 1.0 - prob_pass
prediction = int(model.predict(input_df)[0])

col_pred, col_exp = st.columns([1.1, 1.3], gap="large")

with col_pred:
    st.subheader("1. Prediction Result")
    if prediction == 1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3 style="color:#0f766e; margin-top:0;">Likely to Pass (High Success Rating)</h3>
                <p style="font-size:1.1rem; margin-bottom:0.25rem;">
                    <strong>Pass Probability:</strong> {prob_pass:.1%} &nbsp;|&nbsp;
                    <strong>Risk Probability:</strong> {prob_fail:.1%}
                </p>
                <p style="margin-bottom:0; color:#334155;">
                    Student LMS engagement aligns with successful cohort patterns. Standard academic monitoring applies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-card">
                <h3 style="color:#c2410c; margin-top:0;">At Risk of Failing (Early-Warning Flag)</h3>
                <p style="font-size:1.1rem; margin-bottom:0.25rem;">
                    <strong>Risk Probability:</strong> {prob_fail:.1%} &nbsp;|&nbsp;
                    <strong>Pass Probability:</strong> {prob_pass:.1%}
                </p>
                <p style="margin-bottom:0; color:#334155;">
                    <strong>Recommended Next Step:</strong> Flag for academic advisor outreach and targeted study support.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(prob_pass, text=f"Predicted Pass Likelihood: {prob_pass:.1%}")

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Human-in-the-Loop Safeguard:</strong> <em>The model prioritizes attention; people decide what support is appropriate.</em>
            Predictions are decision-support signals for advisors and should not be used for automated academic penalties.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_exp:
    st.subheader("2. Predictor Comparison & Explainability")
    comparison_rows = []
    importances = dict(zip(feature_cols, model.feature_importances_))
    for col in feature_cols:
        comparison_rows.append(
            {
                "Predictor": FEATURE_INFO[col]["label"],
                "Student Input": user_inputs[col],
                "Passing Cohort Median": int(medians.loc["High", col]),
                "At-Risk Cohort Median": int(medians.loc["Low", col]),
                "Model Importance": f"{importances[col]:.1%}",
            }
        )
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------
# Cohort Explorer & Batch Scoring Tab
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Feature Importance & Cohort Signals", "Batch CSV Prediction"])

with tab1:
    imp_series = (
        pd.Series(model.feature_importances_, index=[FEATURE_INFO[c]["label"] for c in feature_cols])
        .sort_values(ascending=True)
    )
    st.markdown("**Global Feature Importance (Random Forest)**")
    st.bar_chart(imp_series, horizontal=True, color="#0d9488")

with tab2:
    st.markdown("Upload a CSV containing the 8 LMS activity columns to score multiple students at once.")
    uploaded = st.file_uploader("Upload Student LMS CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            batch_probs = model.predict_proba(batch_df[feature_cols])[:, 1]
            batch_preds = np.where(batch_probs >= 0.5, "Likely Pass", "At-Risk (Flag)")
            result_df = batch_df.copy()
            result_df["Pass_Probability"] = np.round(batch_probs, 4)
            result_df["Prediction"] = batch_preds
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="student_predictions.csv",
                mime="text/csv",
            )
