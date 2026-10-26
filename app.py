"""
app.py
------
Streamlit frontend for the Explainable AI-Based Healthcare Risk Prediction
and Decision Support System.

Run with:
    streamlit run app.py

Folder layout expected:
    app.py
    utils.py
    .streamlit/config.toml   (theme colors)
    data/
        diabetes.csv   (optional - real UCI Pima Indians CSV, columns must
                         match utils.DATASETS['diabetes']['features'] + 'Outcome')
        heart.csv      (optional - real UCI Cleveland CSV, columns must
                         match utils.DATASETS['heart']['features'] + 'target')

If the CSVs above are not present, the app trains on a synthetic dataset
that mimics the real one's structure, purely so the frontend is runnable
end-to-end during development. Swap in the real CSVs before final submission.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

import utils

st.set_page_config(
    page_title="Healthcare Risk Prediction",
    page_icon="favicon.png",
    layout="wide",
)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}
    /* Hide the header bar's own content, but keep it present so the
       sidebar's collapse/expand arrow (which lives inside it) still works. */
    header[data-testid="stHeader"] {background: transparent; height: 2.5rem;}
    header[data-testid="stHeader"] * {visibility: visible;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}

    .page-header {
        padding-bottom: 1rem; margin-bottom: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .page-header h1 {margin: 0 0 0.15rem 0; font-size: 1.5rem; color: #0F172A; font-weight: 700;}
    .page-header p {margin: 0; color: #64748B; font-size: 0.92rem;}

    .stat-strip {
        display: flex; gap: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap;
    }
    .stat-card {
        flex: 1; min-width: 150px; padding: 0.85rem 1rem; border-radius: 8px;
        background: #F8FAFC; border: 1px solid #E2E8F0;
    }
    .stat-card .label {
        font-size: 0.72rem; color: #64748B; text-transform: uppercase;
        letter-spacing: 0.04em; font-weight: 600; margin-bottom: 0.15rem;
    }
    .stat-card .value {font-size: 1.3rem; color: #0F172A; font-weight: 700;}

    .risk-card {
        padding: 1.4rem 1.6rem; border-radius: 8px; margin-top: 0.5rem;
        border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .risk-card.high {background: #FEF2F2; border-color: #FCA5A5;}
    .risk-card.moderate {background: #FFFBEB; border-color: #FCD34D;}
    .risk-card.low  {background: #F0FDF4; border-color: #86EFAC;}
    .risk-card h2 {margin: 0 0 0.2rem 0; font-size: 1.2rem;}
    .risk-card.high h2 {color: #B91C1C;}
    .risk-card.moderate h2 {color: #B45309;}
    .risk-card.low h2 {color: #15803D;}
    .risk-card p {margin: 0; color: #475569; font-size: 0.9rem;}

    .section-label {
        font-weight: 600; color: #334155; font-size: 0.78rem;
        letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.3rem;
    }

    div[data-testid="stMetric"] {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px;
        padding: 0.7rem 0.9rem;
    }
    .stTabs [data-baseweb="tab"] {font-weight: 600;}

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 6px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .avatar-circle {
        width: 36px; height: 36px; border-radius: 50%;
        background: #2563EB; color: white; font-weight: 600;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.95rem;
    }
    div[data-testid="stPopover"] button {
        border-radius: 50%; width: 42px; height: 42px; padding: 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loading / training so we don't retrain on every interaction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training model...")
def get_trained_model(dataset_key: str):
    df, is_synthetic = utils.load_dataset(dataset_key)
    model, X_train, comparison, feature_cols, best_name = utils.train_model(df, dataset_key)
    return model, X_train, comparison, feature_cols, best_name, is_synthetic, len(df)


def risk_gauge(proba: float):
    """A Plotly gauge chart for the predicted risk probability, banded to
    match utils.get_risk_band's Low / Moderate / High thresholds."""
    pct = proba * 100
    bar_color = "#16A34A" if pct < 33 else "#D97706" if pct < 66 else "#DC2626"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#F1F5F9",
            "steps": [
                {"range": [0, 33], "color": "#DCFCE7"},
                {"range": [33, 66], "color": "#FEF3C7"},
                {"range": [66, 100], "color": "#FEE2E2"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=10))
    return fig


def sparkline(values, color="#2563EB"):
    """A minimal, axis-free trend line for quick at-a-glance reading in tight spaces."""
    fig = go.Figure(go.Scatter(
        y=values, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
    ))
    fig.update_layout(
        height=60, margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# Grouping fields into clinically meaningful sections (rather than one long
# flat list) makes the form easier to scan; this mirrors how real lab report
# forms and clinical intake sheets are organized.
FEATURE_GROUPS = {
    "diabetes": [
        ("Demographics", ["Age", "Pregnancies"]),
        ("Body Metrics", ["BMI", "SkinThickness"]),
        ("Lab Values", ["Glucose", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]),
    ],
    "heart": [
        ("Demographics", ["age", "sex"]),
        ("Vitals & Labs", ["trestbps", "chol", "thalach"]),
        ("Symptoms & History", ["cp", "fbs", "exang", "restecg"]),
        ("Exercise Test Results", ["oldpeak", "slope", "ca", "thal"]),
    ],
    "breast_cancer": [
        ("Size Measurements", ["mean_radius", "mean_perimeter", "mean_area"]),
        ("Texture & Shape", ["mean_texture", "mean_smoothness", "mean_compactness"]),
        ("Concavity Measurements", ["mean_concavity", "mean_concave_points"]),
    ],
    "liver": [
        ("Demographics", ["Age", "Gender"]),
        ("Liver Function Tests", ["Total_Bilirubin", "Albumin", "Albumin_and_Globulin_Ratio"]),
        ("Liver Enzymes", ["Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase"]),
    ],
    "kidney": [
        ("Demographics & Vitals", ["age", "bp", "htn"]),
        ("Blood Tests", ["bgr", "bu", "sc", "sod", "hemo"]),
    ],
}


# ---------------------------------------------------------------------------
# Authentication gate: user must log in or sign up before seeing the app
# ---------------------------------------------------------------------------

def show_login_screen():
    st.markdown("""
    <div class="page-header">
        <h1>Healthcare Risk Prediction</h1>
        <p>Log in to save your predictions and review them over time.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Log In")
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
        if login_submitted:
            if utils.verify_user(login_user.strip(), login_pass):
                st.session_state["username"] = login_user.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with right:
        st.subheader("Create an Account")
        with st.form("signup_form"):
            new_user = st.text_input("Choose a username", key="signup_username")
            new_pass = st.text_input("Choose a password", type="password", key="signup_password")
            new_pass2 = st.text_input("Confirm password", type="password", key="signup_password2")
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)
        if signup_submitted:
            if not new_user.strip() or not new_pass:
                st.error("Please fill in a username and password.")
            elif new_pass != new_pass2:
                st.error("Passwords don't match.")
            elif utils.create_user(new_user.strip(), new_pass):
                st.success("Account created. You can log in now on the left.")
            else:
                st.error("That username is already taken.")


if "username" not in st.session_state:
    show_login_screen()
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: account info + dataset choice
# ---------------------------------------------------------------------------

st.sidebar.markdown(f"**{st.session_state['username']}**")
if st.sidebar.button("Log out", use_container_width=True):
    del st.session_state["username"]
    st.rerun()
st.sidebar.markdown("---")

st.sidebar.subheader("Risk Prediction")
st.sidebar.caption("AI-assisted decision support")
st.sidebar.markdown("---")

dataset_key = st.sidebar.radio(
    "Choose a condition to assess",
    options=list(utils.DATASETS.keys()),
    format_func=lambda k: utils.DATASETS[k]["display_name"],
)

config = utils.DATASETS[dataset_key]
model, X_train, comparison, feature_cols, best_model_name, is_synthetic, dataset_size = get_trained_model(dataset_key)

if is_synthetic:
    st.sidebar.warning(
        "No real dataset file found. Training on synthetic demo data. "
        f"Add `data/{config['csv_filename']}` to use real patient records."
    )
elif config["csv_filename"] is None:
    st.sidebar.success("Using a built-in, real clinical dataset (no setup needed).")
else:
    st.sidebar.success(f"Using real dataset: `data/{config['csv_filename']}`")

st.sidebar.caption(f"Active model: **{best_model_name}**")

st.sidebar.markdown("---")
st.sidebar.caption(
    "This tool provides AI-assisted risk estimation and is not a medical "
    "diagnosis. Always consult a qualified healthcare professional."
)

# ---------------------------------------------------------------------------
# Page header with account panel
# ---------------------------------------------------------------------------

header_col, profile_col = st.columns([6, 1])

with header_col:
    st.markdown(f"""
    <div class="page-header">
        <h1>Healthcare Risk Prediction</h1>
        <p>{config['display_name']}</p>
    </div>
    """, unsafe_allow_html=True)

with profile_col:
    initial = st.session_state["username"][0].upper()
    with st.popover(initial, use_container_width=True):
        user_info = utils.get_user_info(st.session_state["username"])
        st.markdown(f"**{st.session_state['username']}**")
        if user_info["created_at"]:
            st.caption(f"Member since {user_info['created_at'][:10]}")
        st.metric("Total predictions logged", user_info["total_predictions"])

        st.markdown("---")
        st.caption("Recent progress for this condition")
        progress_preview = utils.get_progress(st.session_state["username"], dataset_key)
        if len(progress_preview) >= 2:
            latest = progress_preview.iloc[-1]["risk_probability"]
            previous = progress_preview.iloc[-2]["risk_probability"]
            change = (latest - previous) * 100
            st.metric("Latest risk", f"{latest * 100:.1f}%", delta=f"{change:+.1f} pts", delta_color="inverse")
            spark_values = (progress_preview["risk_probability"].tail(8) * 100).tolist()
            spark_color = "#16A34A" if change <= 0 else "#DC2626"
            st.plotly_chart(
                sparkline(spark_values, color=spark_color),
                use_container_width=True, config={"displayModeBar": False},
            )
        else:
            st.caption("Log at least 2 predictions to see a trend here.")
        st.caption("Full charts and history are in the tabs below.")

        st.markdown("---")
        if st.button("Log out", use_container_width=True, key="popover_logout"):
            del st.session_state["username"]
            st.rerun()

# ---------------------------------------------------------------------------
# Stats strip: real, derived numbers only (no placeholder/fake figures)
# ---------------------------------------------------------------------------
user_stats = utils.get_user_info(st.session_state["username"])
best_f1 = comparison[best_model_name]["f1"]

st.markdown(f"""
<div class="stat-strip">
    <div class="stat-card">
        <div class="label">Training Samples</div>
        <div class="value">{dataset_size:,}</div>
    </div>
    <div class="stat-card">
        <div class="label">Best Model</div>
        <div class="value" style="font-size: 1.05rem;">{best_model_name}</div>
    </div>
    <div class="stat-card">
        <div class="label">Model F1-score</div>
        <div class="value">{best_f1:.2f}</div>
    </div>
    <div class="stat-card">
        <div class="label">Your Predictions</div>
        <div class="value">{user_stats['total_predictions']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_performance, tab_progress, tab_history = st.tabs(
    ["Predict", "Model Performance", "My Progress", "Prediction History"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    input_values = {}

    # ---- Optional: upload a report image and auto-fill fields from it ----
    st.markdown('<div class="section-label">Optional: Auto-fill from a Report</div>', unsafe_allow_html=True)
    uploaded_report = st.file_uploader(
        "Upload a photo or scan of a lab report (JPG/PNG)", type=["png", "jpg", "jpeg"],
    )

    extracted_values = {}
    if uploaded_report is not None:
        image_bytes = uploaded_report.getvalue()

        @st.cache_data(show_spinner="Reading report...")
        def _run_ocr(img_bytes, ds_key):
            text = utils.extract_text_from_image(img_bytes)
            fields = utils.extract_fields_from_text(text, ds_key)
            return text, fields

        try:
            ocr_text, extracted_values = _run_ocr(image_bytes, dataset_key)

            if extracted_values:
                st.success(
                    f"Found {len(extracted_values)} value(s) in the report: "
                    + ", ".join(extracted_values.keys())
                    + ". These have pre-filled the matching fields below. Please check them before predicting."
                )
            else:
                st.warning(
                    "Couldn't confidently read any known fields from this image. "
                    "You can still enter values manually below."
                )
            with st.expander("See raw text extracted from the image"):
                st.text(ocr_text if ocr_text.strip() else "(no text detected)")

        except ModuleNotFoundError:
            st.error(
                "OCR requires the `pytesseract` and `Pillow` packages "
                "(`pip install pytesseract pillow`), plus the Tesseract-OCR "
                "engine installed separately. See README.md for the link."
            )
        except Exception as e:
            if "tesseract is not installed" in str(e).lower():
                st.error(
                    "Tesseract-OCR engine not found on this system. It's a separate "
                    "install from the Python package. See README.md for the Windows link."
                )
            else:
                st.error(f"Couldn't read that image: {e}")

    st.caption(
        "Note: only common numeric lab values (like Glucose, BMI, Blood "
        "Pressure, Cholesterol) can realistically be auto-detected from a "
        "report. Clinical/derived fields still need manual entry."
    )

    # The Pima Indians Diabetes dataset was collected only on women, so
    # "Pregnancies" is one of its risk factors. To support both genders in
    # the UI, ask for sex first and only show Pregnancies for female
    # patients; for male patients it's fixed at 0 (not applicable).
    is_male_patient = False
    if dataset_key == "diabetes":
        sex_choice = st.radio("Sex", options=["Female", "Male"], horizontal=True)
        is_male_patient = sex_choice == "Male"
        st.caption(
            "This dataset's original data was collected only on women, so "
            "'Number of Pregnancies' only applies to female patients."
        )

    st.write("")
    st.markdown('<div class="section-label">Patient Health Parameters</div>', unsafe_allow_html=True)

    visible_col_names = {
        f[0] for f in config["features"]
        if not (dataset_key == "diabetes" and f[0] == "Pregnancies" and is_male_patient)
    }
    feature_lookup = {f[0]: f for f in config["features"]}

    for group_name, group_cols in FEATURE_GROUPS[dataset_key]:
        group_cols = [c for c in group_cols if c in visible_col_names]
        if not group_cols:
            continue

        st.markdown(f"**{group_name}**")
        with st.container(border=True):
            cols = st.columns(2)
            for i, col_name in enumerate(group_cols):
                _, label, widget, kwargs = feature_lookup[col_name]
                target_col = cols[i % 2]
                with target_col:
                    if widget == "slider":
                        slider_kwargs = dict(kwargs)
                        if col_name in extracted_values:
                            # Clip the OCR-read value into the slider's valid range
                            # so a misread digit can't produce an out-of-range default.
                            clipped = max(slider_kwargs["min_value"], min(slider_kwargs["max_value"], extracted_values[col_name]))
                            slider_kwargs["value"] = type(slider_kwargs["value"])(clipped)
                            label = f"{label} (from report)"
                        input_values[col_name] = st.slider(label, **slider_kwargs)
                    elif widget == "select":
                        options = kwargs["options"]  # list of (display, value)
                        choice = st.selectbox(label, options=[o[0] for o in options])
                        input_values[col_name] = dict(options)[choice]
        st.write("")

    if dataset_key == "diabetes" and is_male_patient:
        input_values["Pregnancies"] = 0

    st.write("")
    predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
        instance_df = pd.DataFrame([input_values])[feature_cols]

        proba = model.predict_proba(instance_df)[0, 1]
        band, band_color, risk_label = utils.get_risk_band(proba, config["condition_name"])

        utils.log_prediction(st.session_state["username"], dataset_key, input_values, proba, risk_label)

        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        result_col1, result_col2 = st.columns([1, 1.3])

        with result_col1:
            st.plotly_chart(risk_gauge(proba), use_container_width=True, config={"displayModeBar": False})
            st.caption("Green: low risk (under 33%). Amber: moderate (33 to 66%). Red: high (above 66%).")

        with result_col2:
            css_class = band.lower()
            st.markdown(f"""
            <div class="risk-card {css_class}">
                <h2>{risk_label}</h2>
                <p>Predicted risk probability: <b>{proba * 100:.1f}%</b></p>
                <p style="margin-top:0.5rem;">This score reflects patterns learned from
                historical data and is intended to support, not replace, clinical judgement.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-label">Why did the model predict this?</div>', unsafe_allow_html=True)
        st.caption(
            "Powered by SHAP. Bars pushing risk **up** are shown in red, "
            "bars pushing risk **down** are shown in green."
        )

        try:
            explanation = utils.explain_instance(model, X_train, instance_df)

            fig, ax = plt.subplots(figsize=(7, 0.45 * len(explanation) + 1))
            colors = ["#DC2626" if v > 0 else "#16A34A" for v in explanation["shap_value"]]
            ax.barh(explanation["feature"], explanation["shap_value"], color=colors)
            ax.axvline(0, color="#1E293B", linewidth=0.8)
            ax.set_xlabel("Contribution to predicted risk (SHAP value)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

            with st.expander("See raw feature values and contributions"):
                st.dataframe(explanation, use_container_width=True)

            st.write("")
            report_text = utils.build_report_text(
                username=st.session_state["username"],
                dataset_name=config["display_name"],
                model_name=best_model_name,
                inputs_dict=input_values,
                risk_probability=proba,
                risk_label=risk_label,
                explanation_df=explanation,
            )
            st.download_button(
                "Download this result as a report",
                data=report_text,
                file_name=f"risk_report_{dataset_key}_{st.session_state['username']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except ModuleNotFoundError:
            st.info(
                "Install the `shap` package (`pip install shap`) to see the "
                "feature-level explanation for this prediction."
            )

# ---------------------------------------------------------------------------
# Tab 2: Model performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.markdown('<div class="section-label">Algorithm Comparison</div>', unsafe_allow_html=True)
    st.caption("Multiple algorithms are trained and compared on a held-out 20% test split; the best one by F1-score is used for predictions.")

    comparison_df = pd.DataFrame(comparison).T
    comparison_df = comparison_df.rename(columns={
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1-score", "auc": "AUC-ROC",
    }).round(3)
    comparison_df.insert(0, "Selected", ["Best" if name == best_model_name else "" for name in comparison_df.index])

    def _highlight_best_row(row):
        is_best = row.name == best_model_name
        return ["background-color: #EFF6FF; font-weight: 600;" if is_best else "" for _ in row]

    styled_comparison = comparison_df.style.apply(_highlight_best_row, axis=1)
    st.dataframe(styled_comparison, use_container_width=True)
    st.success(f"**{best_model_name}** was automatically selected as the best-performing model for this dataset.")

    st.write("")
    metric_cols = st.columns(5)
    metric_labels = [
        ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
        ("F1-score", "f1"), ("AUC-ROC", "auc"),
    ]
    for c, (label, key) in zip(metric_cols, metric_labels):
        value = comparison[best_model_name][key]
        c.metric(label, f"{value:.2f}" if value == value else "N/A")  # NaN check

    st.write("")
    st.markdown('<div class="section-label">Global Feature Importance</div>', unsafe_allow_html=True)
    st.caption(f"Which features matter most to the selected model ({best_model_name}) overall.")

    importances = utils.get_feature_importance(model, feature_cols).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 0.4 * len(importances) + 1))
    ax2.barh(importances.index, importances.values, color="#2563EB")
    ax2.set_xlabel("Importance")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)

# ---------------------------------------------------------------------------
# Tab 3: My Progress. Compares latest reading to the previous one, plus trend
# ---------------------------------------------------------------------------
with tab_progress:
    st.markdown('<div class="section-label">Your Risk Over Time</div>', unsafe_allow_html=True)
    st.caption(f"Tracking your predictions for: {config['display_name']}")

    progress_df = utils.get_progress(st.session_state["username"], dataset_key)

    if len(progress_df) < 2:
        st.info(
            "Make at least **2 predictions** for this condition to see whether "
            "your risk is improving or worsening over time. "
            f"You currently have {len(progress_df)} recorded prediction(s)."
        )
    else:
        latest = progress_df.iloc[-1]
        previous = progress_df.iloc[-2]
        change = (latest["risk_probability"] - previous["risk_probability"]) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Previous Reading", f"{previous['risk_probability'] * 100:.1f}%")
        col2.metric(
            "Latest Reading", f"{latest['risk_probability'] * 100:.1f}%",
            delta=f"{change:+.1f} pts", delta_color="inverse",
        )
        with col3:
            if change < -1:
                st.success("Your risk has **improved** since your last check.")
            elif change > 1:
                st.error("Your risk has **worsened** since your last check.")
            else:
                st.info("Your risk is about the **same** as your last check.")

        st.write("")
        st.markdown('<div class="section-label">Full Trend</div>', unsafe_allow_html=True)

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=pd.to_datetime(progress_df["timestamp"]),
            y=progress_df["risk_probability"] * 100,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8),
        ))
        trend_fig.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Predicted Risk (%)", xaxis_title="Date",
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("See all recorded predictions"):
            st.dataframe(progress_df.iloc[::-1], use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: History (across all conditions, with a Condition column)
# ---------------------------------------------------------------------------
with tab_history:
    st.markdown('<div class="section-label">Prediction History</div>', unsafe_allow_html=True)

    show_all = st.toggle("Show all conditions", value=True)
    history_df = utils.get_history(
        st.session_state["username"],
        dataset_key=None if show_all else dataset_key,
    )
    if history_df.empty:
        st.info("No predictions logged yet. Make a prediction in the Predict tab.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)
