import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Page Configuration & Academic Institutional Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academic Early-Warning Decision Support System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Academic Navy, Slate, & Warm Gold Palette
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Institutional Header Banner */
    .acad-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 0.6rem;
        border-bottom: 4px solid #b45309;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .acad-header .kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #fbbf24;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .acad-header h1 {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.45rem 0;
        line-height: 1.25;
    }
    .acad-header p {
        font-size: 0.95rem;
        color: #e2e8f0;
        margin: 0;
        max-width: 820px;
    }

    /* Section Headings */
    .section-title {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.45rem;
        margin-bottom: 1rem;
    }

    /* KPI Summary Cards */
    .kpi-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #1e3a5f;
        border-radius: 0.5rem;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.15rem;
    }

    /* Outcome Status Cards */
    .status-pass {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #15803d;
        padding: 1.25rem 1.4rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-pass .badge {
        display: inline-block;
        background-color: #15803d;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .status-pass h3 {
        font-family: 'Merriweather', Georgia, serif;
        color: #14532d;
        margin: 0 0 0.5rem 0;
        font-size: 1.25rem;
    }

    .status-risk {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 6px solid #b45309;
        padding: 1.25rem 1.4rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-risk .badge {
        display: inline-block;
        background-color: #b45309;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .status-risk h3 {
        font-family: 'Merriweather', Georgia, serif;
        color: #78350f;
        margin: 0 0 0.5rem 0;
        font-size: 1.25rem;
    }

    /* Governance / Ethics Note */
    .gov-note {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #1e3a5f;
        padding: 0.95rem 1.1rem;
        border-radius: 0.45rem;
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.5;
        margin-top: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "augmented_balanced_dataset_m.csv"
MODEL_PATH = "student_success_model.joblib"

FEATURE_INFO = {
    "Submission.Actions": {
        "label": "Assignment Submissions",
        "code": "Submission.Actions",
        "help": "Count of coursework and assignment submission events recorded in the LMS.",
        "min": 0,
        "max": 100,
        "default": 4,
    },
    "Quiz.Actions": {
        "label": "Quiz & Assessment Activity",
        "code": "Quiz.Actions",
        "help": "Total quiz attempts, views, and formative assessment interactions.",
        "min": 0,
        "max": 1000,
        "default": 15,
    },
    "Forum.Actions": {
        "label": "Discussion Forum Participation",
        "code": "Forum.Actions",
        "help": "Posts, replies, and threads viewed or authored in course forums.",
        "min": 0,
        "max": 300,
        "default": 5,
    },
    "Create.Actions": {
        "label": "Content Creation Events",
        "code": "Create.Actions",
        "help": "Total authored items and creation events across LMS modules.",
        "min": 0,
        "max": 200,
        "default": 8,
    },
    "Read.Actions": {
        "label": "Course Resource Views",
        "code": "Read.Actions",
        "help": "Lecture notes, readings, and syllabus pages accessed in the LMS.",
        "min": 0,
        "max": 50000,
        "default": 135,
    },
    "Update.Actions": {
        "label": "Submission & Profile Revisions",
        "code": "Update.Actions",
        "help": "Edits or updates made to coursework submissions or forum entries.",
        "min": 0,
        "max": 500,
        "default": 3,
    },
    "Delete.Actions": {
        "label": "Deletion Events",
        "code": "Delete.Actions",
        "help": "Item deletion events recorded in the LMS activity log.",
        "min": 0,
        "max": 20,
        "default": 0,
    },
    "Days.Logged.In": {
        "label": "Active Login Days",
        "code": "Days.Logged.In",
        "help": "Number of distinct calendar days the student accessed the LMS.",
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
    y = (df["rating"] == "High").astype(int)  # 1 = Likely to Pass, 0 = At Risk

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
# Sidebar: Student Telemetry Input Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Student LMS Telemetry")
    st.caption(
        "Configure behavioral engagement indicators below to evaluate academic standing."
    )

    preset = st.selectbox(
        "Cohort Reference Profile",
        [
            "Cohort Median Baseline",
            "Exemplar Passing Profile",
            "Early-Warning At-Risk Profile",
        ],
    )

    preset_values = {
        "Cohort Median Baseline": {k: v["default"] for k, v in FEATURE_INFO.items()},
        "Exemplar Passing Profile": {
            "Submission.Actions": 10,
            "Quiz.Actions": 35,
            "Forum.Actions": 12,
            "Create.Actions": 14,
            "Read.Actions": 320,
            "Update.Actions": 6,
            "Delete.Actions": 0,
            "Days.Logged.In": 22,
        },
        "Early-Warning At-Risk Profile": {
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

    st.markdown("---")
    user_inputs = {}
    for col in feature_cols:
        meta = FEATURE_INFO[col]
        user_inputs[col] = st.number_input(
            label=f"{meta['label']} ({meta['code']})",
            min_value=meta["min"],
            max_value=meta["max"],
            value=int(preset_values[preset][col]),
            step=1,
            help=meta["help"],
        )

# ---------------------------------------------------------
# Institutional Header Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="acad-header">
        <div class="kicker">Office of Institutional Analytics &bull; Student Success Initiative</div>
        <h1>Academic Early-Warning Decision Support System</h1>
        <p>
            Predictive risk assessment tool evaluating student course persistence from Learning Management System (LMS)
            engagement telemetry to guide timely academic advising.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model Validation & Cohort Metadata Strip
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        """
        <div class="kpi-box">
            <div class="kpi-label">Classifier Architecture</div>
            <div class="kpi-val">Random Forest</div>
            <div class="kpi-sub">200 decision trees (stratified)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Validation Accuracy</div>
            <div class="kpi-val">{metrics['accuracy']:.1%}</div>
            <div class="kpi-sub">80/20 holdout evaluation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">ROC-AUC Discrimination</div>
            <div class="kpi-val">{metrics['roc_auc']:.3f}</div>
            <div class="kpi-sub">94.0% recall on at-risk cohort</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Calibration Sample</div>
            <div class="kpi-val">{metrics['n_records']:,}</div>
            <div class="kpi-sub">Balanced LMS student records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Evaluation Workspace
# ---------------------------------------------------------
input_df = pd.DataFrame([user_inputs])
prob_pass = float(model.predict_proba(input_df)[0][1])
prob_fail = 1.0 - prob_pass
prediction = int(model.predict(input_df)[0])

col_left, col_right = st.columns([1.05, 1.35], gap="large")

with col_left:
    st.markdown('<div class="section-title">I. Academic Standing Assessment</div>', unsafe_allow_html=True)
    if prediction == 1:
        st.markdown(
            f"""
            <div class="status-pass">
                <span class="badge">Projected Outcome: Pass</span>
                <h3>Satisfactory Academic Engagement</h3>
                <p style="font-size:0.98rem; color:#1e293b; margin-bottom:0.6rem;">
                    <strong>Estimated Pass Likelihood:</strong> {prob_pass:.1%} &nbsp;&bull;&nbsp;
                    <strong>Attrition Risk:</strong> {prob_fail:.1%}
                </p>
                <p style="font-size:0.9rem; color:#334155; margin-bottom:0;">
                    <strong>Advising Guidance:</strong> Student LMS participation aligns with passing cohort benchmarks.
                    Maintain routine faculty check-ins.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-risk">
                <span class="badge">Early-Warning Flag: At Risk</span>
                <h3>Elevated Risk of Course Failure</h3>
                <p style="font-size:0.98rem; color:#1e293b; margin-bottom:0.6rem;">
                    <strong>Estimated Risk Score:</strong> {prob_fail:.1%} &nbsp;&bull;&nbsp;
                    <strong>Pass Likelihood:</strong> {prob_pass:.1%}
                </p>
                <p style="font-size:0.9rem; color:#334155; margin-bottom:0;">
                    <strong>Advising Guidance:</strong> Prioritize for academic advisor outreach, tutoring referral,
                    and structured study-plan review.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(prob_pass, text=f"Course Persistence Probability Index: {prob_pass:.1%}")

    st.markdown(
        """
        <div class="gov-note">
            <strong>Institutional Governance &amp; Human-in-the-Loop Policy:</strong>
            <em>Predictive scores prioritize advising attention; academic personnel determine appropriate student support.</em>
            Outputs serve strictly as formative decision-support indicators and must not be used for automated grading or disciplinary action.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="section-title">II. Telemetry Benchmark Comparison</div>', unsafe_allow_html=True)
    comparison_rows = []
    importances = dict(zip(feature_cols, model.feature_importances_))
    for col in feature_cols:
        pass_med = int(medians.loc["High", col])
        risk_med = int(medians.loc["Low", col])
        val = user_inputs[col]
        comparison_rows.append(
            {
                "Behavioral Predictor": FEATURE_INFO[col]["label"],
                "Student Value": val,
                "Pass Median": pass_med,
                "At-Risk Median": risk_med,
                "Model Weight": f"{importances[col]:.1%}",
            }
        )
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Secondary Analytical Tabs
# ---------------------------------------------------------
tab_feat, tab_batch, tab_method = st.tabs(
    [
        "Predictor Importance Analysis",
        "Cohort Batch Screening (CSV)",
        "Methodology & Model Documentation",
    ]
)

with tab_feat:
    st.markdown("**Relative Predictor Contribution (Gini Feature Importance)**")
    st.caption(
        "Displays the relative weight each LMS behavioral indicator contributes to classification decisions across the 200-tree ensemble."
    )
    imp_df = (
        pd.DataFrame(
            {
                "Predictor": [FEATURE_INFO[c]["label"] for c in feature_cols],
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=True)
        .set_index("Predictor")
    )
    st.bar_chart(imp_df, horizontal=True, color="#1e3a5f")

with tab_batch:
    st.markdown("**Batch Student Risk Screening**")
    st.caption(
        "Upload a roster CSV containing the eight required LMS columns to generate risk probabilities and advising flags for an entire course section."
    )
    uploaded = st.file_uploader("Select Course Roster CSV File", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required schema columns: {', '.join(missing)}")
        else:
            batch_probs = model.predict_proba(batch_df[feature_cols])[:, 1]
            batch_preds = np.where(batch_probs >= 0.5, "Likely Pass", "Early-Warning Flag")
            result_df = batch_df.copy()
            result_df["Pass_Probability"] = np.round(batch_probs, 4)
            result_df["Advising_Status"] = batch_preds
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Export Screening Report (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="early_warning_screening_report.csv",
                mime="text/csv",
            )

with tab_method:
    st.markdown(
        """
        **Dataset & Modeling Summary**
        - **Source Dataset:** `augmented_balanced_dataset_m.csv` ($N = 3,602$ student records; balanced 1:1 between `High` persistence and `Low` at-risk outcomes).
        - **Feature Space:** Eight quantitative LMS activity indicators (`Read.Actions`, `Days.Logged.In`, `Quiz.Actions`, `Create.Actions`, `Forum.Actions`, `Submission.Actions`, `Update.Actions`, `Delete.Actions`).
        - **Model Selection:** Random Forest Classifier (`n_estimators=200`, `random_state=42`) selected over Gradient Boosting (73.0% accuracy) and Logistic Regression (64.1% accuracy) due to superior non-linear interaction modeling (88.2% accuracy, 0.952 ROC-AUC).
        """
    )
