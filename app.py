import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ---------------------------------------------------------
# Page Configuration (Single Academic Light Theme - No Dark/Light Toggle)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academic Early-Warning Decision Support System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Institutional Academic Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@700&display=swap');

    .stApp {
        background-color: #f4f6f9;
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Dark Navy Institutional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #132238;
        color: #f8fafc;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCaption {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] input {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0f172a !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1260px;
    }

    /* Context Banner */
    .context-banner {
        background-color: #e8eef5;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #1e3a5f;
        border-radius: 0.45rem;
        padding: 0.7rem 1.1rem;
        margin-bottom: 1.1rem;
        font-size: 0.88rem;
        color: #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Institutional Header */
    .acad-header {
        background: linear-gradient(135deg, #132238 0%, #1e3a5f 100%);
        color: #ffffff;
        padding: 1.4rem 1.75rem;
        border-radius: 0.55rem;
        border-bottom: 4px solid #b45309;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    }
    .acad-header .kicker {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #fbbf24;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .acad-header h1 {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0 0 0.35rem 0;
    }
    .acad-header p {
        font-size: 0.9rem;
        color: #e2e8f0;
        margin: 0;
    }

    /* Section Headings */
    .section-title {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 0.4rem;
        margin-bottom: 0.9rem;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #1e3a5f;
        border-radius: 0.5rem;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .kpi-card.alert-top {
        border-top-color: #b45309;
    }
    .kpi-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #475569;
        margin-top: 0.15rem;
    }

    /* Explicit PASS / FAIL Outcome Cards */
    .outcome-pass {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 7px solid #15803d;
        border-radius: 0.55rem;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
    }
    .outcome-fail {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        border-left: 7px solid #b91c1c;
        border-radius: 0.55rem;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
    }
    .outcome-kicker {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin-bottom: 0.2rem;
    }
    .outcome-label-pass {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #15803d;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    .outcome-label-fail {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #b91c1c;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }
    .outcome-mapping {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 0.25rem;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
        margin-bottom: 0.75rem;
    }
    .prob-line {
        font-size: 0.98rem;
        color: #1e293b;
        margin-bottom: 0.35rem;
    }

    /* Governance Box */
    .gov-note {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #1e3a5f;
        padding: 0.9rem 1.05rem;
        border-radius: 0.45rem;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.5;
        margin-top: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "augmented_balanced_dataset_m.csv"

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
def load_and_evaluate(leakage_free: bool = True):
    """
    Loads the MP2 dataset (augmented_balanced_dataset_m.csv) and evaluates
    Random Forest on an untouched 80/20 stratified holdout test set.

    Target mapping:
      rating == 'High' -> 1 (PASS: High Course Persistence / Likely to Pass)
      rating == 'Low'  -> 0 (FAIL: Low Course Persistence / At-Risk of Failing)
    """
    raw_df = pd.read_csv(DATA_PATH)
    feature_cols = list(FEATURE_INFO.keys())

    if leakage_free:
        # Step 1: Deduplicate prior to splitting so synthetic/oversampled duplicates
        # cannot leak across train and test partitions.
        df_clean = raw_df.drop_duplicates().copy()
        X = df_clean[feature_cols]
        y = (df_clean["rating"] == "High").astype(int)

        # Step 2: Split untouched test set first (80/20 stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 3: Apply random oversampling ONLY to the training partition
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        pass_train = train_df[train_df["target"] == 1]
        fail_train = train_df[train_df["target"] == 0]
        fail_upsampled = resample(
            fail_train, replace=True, n_samples=len(pass_train), random_state=42
        )
        balanced_train = pd.concat([pass_train, fail_upsampled])

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(balanced_train[feature_cols], balanced_train["target"])
        eval_df = df_clean
    else:
        # Direct evaluation on the provided CSV file as-is
        X = raw_df[feature_cols]
        y = (raw_df["rating"] == "High").astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        eval_df = raw_df

    # Step 4: Evaluate on untouched holdout test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=["FAIL (Low)", "PASS (High)"], output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(3)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "fail_recall": recall_score(y_test, y_pred, pos_label=0),
        "pass_recall": recall_score(y_test, y_pred, pos_label=1),
        "n_raw": len(raw_df),
        "n_unique": len(raw_df.drop_duplicates()),
        "n_test": len(y_test),
        "cm": cm,
        "report_df": report_df,
    }
    medians = eval_df.groupby("rating")[feature_cols].median()
    return model, raw_df, metrics, medians, feature_cols


# ---------------------------------------------------------
# Sidebar: Academic Navigation & Data/Validation Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Academic Analytics")
    st.caption("Student Success & Early-Warning Portal")
    st.markdown("---")

    nav_section = st.radio(
        "Workspace View",
        [
            "Individual Student Assessment",
            "Cohort Risk Screening (CSV)",
            "Model Validation & Leakage Audit",
            "Advising Interventions & Methodology",
        ],
    )

    st.markdown("---")
    st.markdown("**Data Integrity Mode**")
    validation_mode = st.selectbox(
        "Training & Evaluation Protocol",
        [
            "Leakage-Free Holdout (Deduplicated + Train-Only Oversampling)",
            "Raw MP2 File Split (As-Provided 3,602 Rows)",
        ],
        help="Prevents duplicate rows created during pre-split balancing from leaking into the test partition.",
    )
    use_leakage_free = validation_mode.startswith("Leakage-Free")

model, raw_df, metrics, medians, feature_cols = load_and_evaluate(
    leakage_free=use_leakage_free
)

with st.sidebar:
    st.markdown("---")
    st.markdown("**Student LMS Predictors**")
    preset = st.selectbox(
        "Quick Profile Preset",
        [
            "Cohort Median Baseline",
            "Exemplar PASS Profile",
            "At-Risk FAIL Profile",
        ],
    )

    preset_values = {
        "Cohort Median Baseline": {k: v["default"] for k, v in FEATURE_INFO.items()},
        "Exemplar PASS Profile": {
            "Submission.Actions": 10,
            "Quiz.Actions": 35,
            "Forum.Actions": 12,
            "Create.Actions": 14,
            "Read.Actions": 320,
            "Update.Actions": 6,
            "Delete.Actions": 0,
            "Days.Logged.In": 22,
        },
        "At-Risk FAIL Profile": {
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
        user_inputs[col] = st.number_input(
            label=f"{meta['label']} ({meta['code']})",
            min_value=meta["min"],
            max_value=meta["max"],
            value=int(preset_values[preset][col]),
            step=1,
            help=meta["help"],
        )

# ---------------------------------------------------------
# Top Context Banner & Institutional Header
# ---------------------------------------------------------
mode_label = (
    f"Leakage-Free Protocol Active ({metrics['n_unique']:,} deduplicated records; oversampling applied strictly to training fold)"
    if use_leakage_free
    else f"Raw CSV Split Active ({metrics['n_raw']:,} rows from augmented_balanced_dataset_m.csv)"
)
st.markdown(
    f"""
    <div class="context-banner">
        <div><strong>Dataset &amp; Validation Status:</strong> {mode_label}</div>
        <div><strong>Target Schema:</strong> <code>rating</code> &rarr; <strong>PASS</strong> (High) / <strong>FAIL</strong> (Low)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="acad-header">
        <div class="kicker">Office of Institutional Analytics &bull; Applied Data Science (MP3)</div>
        <h1>Academic Early-Warning Decision Support System</h1>
        <p>
            Predicts binary student course outcomes (<strong>PASS</strong> vs. <strong>FAIL</strong>) from Learning Management System (LMS)
            behavioral predictors to guide timely, advisor-led academic support.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Cohort & Holdout Evaluation Overview Strip
# ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Evaluation Sample</div>
            <div class="kpi-value">{metrics['n_test']:,}</div>
            <div class="kpi-sub">Untouched 20% holdout set</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Holdout Accuracy</div>
            <div class="kpi-value">{metrics['accuracy']:.1%}</div>
            <div class="kpi-sub">200 trees; stratified 80/20 split</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC Score</div>
            <div class="kpi-value">{metrics['roc_auc']:.3f}</div>
            <div class="kpi-sub">Threshold-independent discrimination</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="kpi-card alert-top">
            <div class="kpi-label">FAIL Class Recall</div>
            <div class="kpi-value">{metrics['fail_recall']:.1%}</div>
            <div class="kpi-sub">Sensitivity to at-risk students</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">PASS Class Recall</div>
            <div class="kpi-value">{metrics['pass_recall']:.1%}</div>
            <div class="kpi-sub">Specificity for persisting students</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# VIEW 1: Individual Student Assessment
# ---------------------------------------------------------
if nav_section == "Individual Student Assessment":
    input_df = pd.DataFrame([user_inputs])
    prob_pass = float(model.predict_proba(input_df)[0][1])
    prob_fail = 1.0 - prob_pass
    # Decision rule: P(PASS) >= 0.50 -> PASS, else FAIL
    predicted_class = "PASS" if prob_pass >= 0.50 else "FAIL"

    col_left, col_right = st.columns([1.05, 1.35], gap="large")

    with col_left:
        st.markdown(
            '<div class="section-title">I. Predicted Student Outcome</div>',
            unsafe_allow_html=True,
        )
        if predicted_class == "PASS":
            st.markdown(
                f"""
                <div class="outcome-pass">
                    <div class="outcome-kicker">Predicted Student Outcome</div>
                    <div class="outcome-label-pass">PASS</div>
                    <div class="outcome-mapping">Source Dataset Label: <code>rating = High</code> (High Course Persistence)</div>
                    <div class="prob-line"><strong>Estimated probability of passing:</strong> {prob_pass:.1%}</div>
                    <div class="prob-line"><strong>Estimated probability of failing:</strong> {prob_fail:.1%}</div>
                    <p style="font-size:0.86rem; color:#334155; margin-top:0.65rem; margin-bottom:0;">
                        <strong>Decision Rule:</strong> Classified as <strong>PASS</strong> because P(PASS) = {prob_pass:.1%} &ge; 50.0% threshold.
                        Routine faculty monitoring recommended.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="outcome-fail">
                    <div class="outcome-kicker">Predicted Student Outcome</div>
                    <div class="outcome-label-fail">FAIL</div>
                    <div class="outcome-mapping">Source Dataset Label: <code>rating = Low</code> (Low Persistence / Early-Warning Flag)</div>
                    <div class="prob-line"><strong>Estimated probability of passing:</strong> {prob_pass:.1%}</div>
                    <div class="prob-line"><strong>Estimated probability of failing:</strong> {prob_fail:.1%}</div>
                    <p style="font-size:0.86rem; color:#334155; margin-top:0.65rem; margin-bottom:0;">
                        <strong>Decision Rule:</strong> Classified as <strong>FAIL</strong> because risk score P(FAIL) = {prob_fail:.1%} &gt; 50.0% threshold.
                        Recommended for academic advisor outreach and tutoring support.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(prob_pass, text=f"Estimated Probability of Passing: {prob_pass:.1%}")

        st.markdown(
            """
            <div class="gov-note">
                <strong>Human-in-the-Loop Safeguard:</strong>
                <em>The model prioritizes advising attention; academic personnel decide what support is appropriate.</em>
                Predictions serve strictly as early-warning decision support and must not be used for automated grading or punitive action.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            '<div class="section-title">II. Student Predictors vs. Cohort Benchmarks</div>',
            unsafe_allow_html=True,
        )
        importances = dict(zip(feature_cols, model.feature_importances_))
        rows = []
        for col in feature_cols:
            rows.append(
                {
                    "LMS Predictor": FEATURE_INFO[col]["label"],
                    "Student Input": user_inputs[col],
                    "PASS Median (High)": int(medians.loc["High", col]),
                    "FAIL Median (Low)": int(medians.loc["Low", col]),
                    "Feature Weight": f"{importances[col]:.1%}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("**Global Predictor Importance (Random Forest)**")
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

# ---------------------------------------------------------
# VIEW 2: Cohort Risk Screening (CSV)
# ---------------------------------------------------------
elif nav_section == "Cohort Risk Screening (CSV)":
    st.markdown(
        '<div class="section-title">Cohort Batch Risk Screening</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a student roster CSV containing the eight LMS activity columns to generate **PASS / FAIL** predictions and risk probabilities."
    )
    uploaded = st.file_uploader("Upload Student LMS CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            probs_pass = model.predict_proba(batch_df[feature_cols])[:, 1]
            probs_fail = 1.0 - probs_pass
            outcomes = np.where(probs_pass >= 0.50, "PASS", "FAIL")

            res_df = batch_df.copy()
            res_df["Predicted_Outcome"] = outcomes
            res_df["Prob_PASS"] = np.round(probs_pass, 4)
            res_df["Prob_FAIL"] = np.round(probs_fail, 4)

            s1, s2, s3 = st.columns(3)
            s1.metric("Students Screened", f"{len(res_df):,}")
            s2.metric("Predicted PASS", f"{int((outcomes == 'PASS').sum()):,}")
            s3.metric("Predicted FAIL (At-Risk)", f"{int((outcomes == 'FAIL').sum()):,}")

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "Download Batch Screening Results (CSV)",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="student_pass_fail_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------
# VIEW 3: Model Validation & Leakage Audit
# ---------------------------------------------------------
elif nav_section == "Model Validation & Leakage Audit":
    st.markdown(
        '<div class="section-title">Untouched Holdout Evaluation &amp; Leakage Audit</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        To ensure reported metrics are reliable and free of data leakage, this panel displays evaluation results computed
        exclusively on the **untouched 20% holdout test set** ($n = {metrics['n_test']}$ students) under the active protocol:
        **`{validation_mode}`**.
        """
    )

    col_cm, col_cr = st.columns([1.0, 1.3], gap="large")
    with col_cm:
        st.markdown("**Confusion Matrix (Holdout Test Set)**")
        cm_df = pd.DataFrame(
            metrics["cm"],
            index=["Actual FAIL (Low)", "Actual PASS (High)"],
            columns=["Predicted FAIL", "Predicted PASS"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            "Rows represent true outcomes in the untouched holdout set; columns represent Random Forest predictions at threshold 0.50."
        )

    with col_cr:
        st.markdown("**Classification Report (Precision, Recall, F1-Score)**")
        st.dataframe(metrics["report_df"], use_container_width=True)

    st.info(
        f"**Dataset Audit Note:** `augmented_balanced_dataset_m.csv` contains {metrics['n_raw']:,} total rows "
        f"({metrics['n_unique']:,} unique records and {metrics['n_raw'] - metrics['n_unique']:,} duplicate rows from pre-split balancing). "
        "Selecting the **Leakage-Free Holdout** option in the sidebar deduplicates records before splitting 80/20 and applies "
        "random oversampling strictly inside the training set so no duplicate rows leak into the test evaluation."
    )

# ---------------------------------------------------------
# VIEW 4: Advising Interventions & Methodology
# ---------------------------------------------------------
elif nav_section == "Advising Interventions & Methodology":
    st.markdown(
        '<div class="section-title">Target Variable Mapping &amp; Advising Protocol</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### 1. Alignment Between Dataset Target (`rating`) and Assignment (`PASS` / `FAIL`)
        - **Source Dataset (`augmented_balanced_dataset_m.csv`):** Encodes student course persistence/performance as `rating` with two classes: `High` and `Low`.
        - **Operational Mapping:**
          - **`High` &rarr; `PASS`:** Student LMS engagement matches patterns of successful course completion ($P(\text{PASS}) \ge 0.50$).
          - **`Low` &rarr; `FAIL`:** Student LMS engagement matches patterns of course non-completion / failure risk ($P(\text{FAIL}) = 1 - P(\text{PASS}) > 0.50$).

        ### 2. Tiered Advisor Intervention Framework
        | Risk Tier | Probability Threshold | Recommended Institutional Action |
        | :--- | :--- | :--- |
        | **Tier 1: Satisfactory (`PASS`)** | $P(\text{FAIL}) < 0.50$ | Routine LMS progress tracking; standard faculty feedback. |
        | **Tier 2: Moderate Early Warning (`FAIL`)** | $0.50 \le P(\text{FAIL}) < 0.75$ | Academic advisor check-in email and peer tutoring referral. |
        | **Tier 3: High Early Warning (`FAIL`)** | $P(\text{FAIL}) \ge 0.75$ | Priority 1-on-1 advising conference and structured study-plan contract. |
        """
    )
