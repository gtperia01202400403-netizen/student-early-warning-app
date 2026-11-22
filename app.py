import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ---------------------------------------------------------
# Page Setup: Clean, Simple University Application
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .header-box {
        background-color: #1e3a5f;
        color: #ffffff;
        padding: 1.4rem 1.8rem;
        border-radius: 0.5rem;
        border-bottom: 4px solid #b45309;
        margin-bottom: 1.5rem;
    }
    .header-box h1 {
        color: #ffffff !important;
        margin: 0 0 0.35rem 0;
        font-size: 1.65rem;
    }
    .header-box p {
        color: #e2e8f0 !important;
        margin: 0;
        font-size: 0.95rem;
    }
    .result-pass {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 6px solid #15803d;
        padding: 1.25rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .result-fail {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        border-left: 6px solid #b91c1c;
        padding: 1.25rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .disclaimer {
        background-color: #f1f5f9;
        border-left: 4px solid #1e3a5f;
        padding: 0.85rem 1rem;
        border-radius: 0.4rem;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = "augmented_balanced_dataset_m.csv"

FEATURE_COLS = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]


@st.cache_resource
def train_model():
    raw_df = pd.read_csv(DATA_PATH)
    # Remove duplicate rows before splitting so test metrics have zero data leakage
    df_clean = raw_df.drop_duplicates().copy()

    X = df_clean[FEATURE_COLS]
    # Target mapping: High -> 1 (PASS), Low -> 0 (FAIL)
    y = (df_clean["rating"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance training set only
    train_data = pd.concat([X_train, y_train.rename("target")], axis=1)
    pass_rows = train_data[train_data["target"] == 1]
    fail_rows = train_data[train_data["target"] == 0]
    fail_upsampled = resample(
        fail_rows, replace=True, n_samples=len(pass_rows), random_state=42
    )
    train_balanced = pd.concat([pass_rows, fail_upsampled])

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(train_balanced[FEATURE_COLS], train_balanced["target"])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report_df = (
        pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=["FAIL (rating=Low)", "PASS (rating=High)"],
                output_dict=True,
            )
        )
        .transpose()
        .round(3)
    )
    report_df.loc["accuracy", ["precision", "recall"]] = np.nan
    report_df.loc["accuracy", "support"] = int(len(y_test))

    eval_stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
        "report": report_df,
        "n_unique": len(df_clean),
        "n_test": len(y_test),
    }
    return model, eval_stats


model, eval_stats = train_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <h1>Student Pass / Fail Prediction Application</h1>
        <p>Enter a student's LMS activity counts or upload a class CSV file to predict whether the student is likely to PASS or FAIL.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Simple 3-Tab Layout (Everything in the Main Area)
# ---------------------------------------------------------
tab_single, tab_batch, tab_model = st.tabs(
    [
        "1. Predict Single Student",
        "2. Upload Class CSV (Batch Prediction)",
        "3. Model Info & Validation",
    ]
)

# =========================================================
# TAB 1: Single Student Form (Simple Step-by-Step Inputs)
# =========================================================
with tab_single:
    st.subheader("Enter Student LMS Activity Counts")
    st.caption(
        "Note: Because the MP2 dataset records LMS action counts rather than exam grades, enter the student's activity counts below."
    )

    with st.form("single_student_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Coursework & Assessment Activity**")
            sub_actions = st.number_input(
                "Assignment Submissions (Submission.Actions)",
                min_value=0,
                max_value=200,
                value=4,
                step=1,
                help="How many times the student submitted assignments in the LMS.",
            )
            quiz_actions = st.number_input(
                "Quiz Interactions (Quiz.Actions)",
                min_value=0,
                max_value=1000,
                value=15,
                step=1,
                help="How many quiz views or attempts the student recorded in the LMS.",
            )
            days_logged = st.number_input(
                "Days Logged In (Days.Logged.In)",
                min_value=0,
                max_value=120,
                value=14,
                step=1,
                help="Total distinct days the student logged into the LMS during the term.",
            )
            read_actions = st.number_input(
                "Course Material Views (Read.Actions)",
                min_value=0,
                max_value=50000,
                value=135,
                step=1,
                help="How many times the student opened/read course lessons or pages.",
            )

        with col2:
            st.markdown("**Participation & File Activity**")
            forum_actions = st.number_input(
                "Forum Posts & Views (Forum.Actions)",
                min_value=0,
                max_value=500,
                value=5,
                step=1,
                help="How many forum discussions the student viewed or posted in.",
            )
            create_actions = st.number_input(
                "Items Created (Create.Actions)",
                min_value=0,
                max_value=300,
                value=8,
                step=1,
                help="Total new submissions, posts, or items created in the LMS.",
            )
            update_actions = st.number_input(
                "Items Updated / Edited (Update.Actions)",
                min_value=0,
                max_value=500,
                value=3,
                step=1,
                help="Total edits or updates made to LMS items.",
            )
            delete_actions = st.number_input(
                "Items Deleted (Delete.Actions)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                help="Total items deleted by the student in the LMS.",
            )

        submitted = st.form_submit_button(
            "Predict Student Outcome", type="primary", use_container_width=True
        )

    if submitted:
        student_row = pd.DataFrame(
            [
                {
                    "Submission.Actions": sub_actions,
                    "Quiz.Actions": quiz_actions,
                    "Forum.Actions": forum_actions,
                    "Create.Actions": create_actions,
                    "Read.Actions": read_actions,
                    "Update.Actions": update_actions,
                    "Delete.Actions": delete_actions,
                    "Days.Logged.In": days_logged,
                }
            ]
        )

        prob_pass = float(model.predict_proba(student_row)[0][1])
        prob_fail = 1.0 - prob_pass
        outcome = "PASS" if prob_pass >= 0.50 else "FAIL"

        if outcome == "PASS":
            st.markdown(
                f"""
                <div class="result-pass">
                    <div style="font-size:0.85rem; font-weight:700; color:#475569;">PREDICTED STUDENT OUTCOME</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#15803d; margin:0.2rem 0;">PASS</div>
                    <div style="font-size:1rem; margin-bottom:0.3rem;">
                        <strong>Estimated probability of passing:</strong> {prob_pass:.1%} &nbsp;|&nbsp;
                        <strong>Estimated probability of failing:</strong> {prob_fail:.1%}
                    </div>
                    <div style="font-size:0.88rem; color:#334155;">
                        <strong>Explanation:</strong> Mapped from dataset label <code>rating = High</code> (High Course Persistence).
                        Classified as <strong>PASS</strong> because P(PASS) &ge; 50.0%.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-fail">
                    <div style="font-size:0.85rem; font-weight:700; color:#475569;">PREDICTED STUDENT OUTCOME</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#b91c1c; margin:0.2rem 0;">FAIL</div>
                    <div style="font-size:1rem; margin-bottom:0.3rem;">
                        <strong>Estimated probability of passing:</strong> {prob_pass:.1%} &nbsp;|&nbsp;
                        <strong>Estimated probability of failing:</strong> {prob_fail:.1%}
                    </div>
                    <div style="font-size:0.88rem; color:#334155;">
                        <strong>Explanation:</strong> Mapped from dataset label <code>rating = Low</code> (At-Risk / Low Persistence).
                        Classified as <strong>FAIL</strong> because P(FAIL) &gt; 50.0%. Recommended for advisor outreach.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="disclaimer">
                <strong>Advisor Note:</strong> This prediction is intended to guide early academic support and advising,
                not to replace instructor grading or automatically penalize students.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# TAB 2: Batch CSV Upload (For Multiple Students)
# =========================================================
with tab_batch:
    st.subheader("Predict Outcomes for a Whole Class (CSV Upload)")
    st.markdown(
        """
        Upload a CSV file containing your students' LMS activity counts. The file must include these **8 exact column names**:
        `Submission.Actions`, `Quiz.Actions`, `Forum.Actions`, `Create.Actions`, `Read.Actions`, `Update.Actions`, `Delete.Actions`, `Days.Logged.In`.
        """
    )

    sample_template = pd.DataFrame(
        [
            {
                "Submission.Actions": 10,
                "Quiz.Actions": 35,
                "Forum.Actions": 12,
                "Create.Actions": 14,
                "Read.Actions": 320,
                "Update.Actions": 6,
                "Delete.Actions": 0,
                "Days.Logged.In": 22,
            },
            {
                "Submission.Actions": 1,
                "Quiz.Actions": 2,
                "Forum.Actions": 0,
                "Create.Actions": 3,
                "Read.Actions": 45,
                "Update.Actions": 1,
                "Delete.Actions": 0,
                "Days.Logged.In": 6,
            },
        ]
    )
    st.download_button(
        "Download Sample CSV Template",
        data=sample_template.to_csv(index=False).encode("utf-8"),
        file_name="sample_student_lms_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in FEATURE_COLS if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Your file is missing these required columns: {', '.join(missing_cols)}")
        else:
            probs_pass = model.predict_proba(batch_df[FEATURE_COLS])[:, 1]
            probs_fail = 1.0 - probs_pass
            outcomes = np.where(probs_pass >= 0.50, "PASS", "FAIL")

            out_df = batch_df.copy()
            out_df["Predicted_Outcome"] = outcomes
            out_df["Pass_Probability"] = np.round(probs_pass, 4)
            out_df["Fail_Probability"] = np.round(probs_fail, 4)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Students", len(out_df))
            c2.metric("Predicted PASS", int((outcomes == "PASS").sum()))
            c3.metric("Predicted FAIL", int((outcomes == "FAIL").sum()))

            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="student_pass_fail_results.csv",
                mime="text/csv",
            )

# =========================================================
# TAB 3: Model Info & Validation (Holdout Metrics)
# =========================================================
with tab_model:
    st.subheader("Model Methodology & Holdout Validation")
    st.markdown(
        f"""
        - **Dataset:** `augmented_balanced_dataset_m.csv` ({eval_stats['n_unique']:,} unique student records after deduplication).
        - **Target Variable Mapping:** `rating = High` is mapped to **PASS** (`1`), and `rating = Low` is mapped to **FAIL** (`0`).
        - **Algorithm & Split:** Random Forest Classifier (200 trees; stratified 80/20 train-test split, with oversampling applied strictly to the training set to prevent data leakage).
        """
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Holdout Accuracy", f"{eval_stats['accuracy']:.1%}")
    m2.metric("ROC-AUC Score", f"{eval_stats['roc_auc']:.3f}")
    m3.metric("Untouched Test Samples", f"{eval_stats['n_test']:,} students")

    col_cm, col_cr = st.columns(2)
    with col_cm:
        st.markdown("**Confusion Matrix (Untouched Test Set)**")
        cm_df = pd.DataFrame(
            eval_stats["cm"],
            index=["Actual FAIL (Low)", "Actual PASS (High)"],
            columns=["Predicted FAIL", "Predicted PASS"],
        )
        st.dataframe(cm_df, use_container_width=True)

    with col_cr:
        st.markdown("**Classification Report**")
        st.dataframe(eval_stats["report"], use_container_width=True)
