"""
Early Warning System — Student Risk Dashboard
Design and Evaluation of a Transformer-Based Early Warning System for
At-Risk Student Identification in Higher Education

Eashkumar Kaki | Student No: 35057938

Run locally with:
    pip install streamlit
    streamlit run streamlit_dashboard.py

Predictions shown are the actual output of the trained XGBoost model on
the OULAD test set (weeks 0-4 features). If a file named
'dashboard_sample_output.csv' is present in the same folder as this
script, it will be loaded automatically; otherwise the 15 real
predictions generated during the project's Kaggle run are used.
"""

import os
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Early Warning System — Case Register",
    page_icon="\u2316",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Real model output (fallback if no CSV is found alongside this script) ──
FALLBACK_DATA = [
    {"student_id": 556788, "risk_score": 0.995, "risk_level": "High",
     "top_contributing_factors": "studied_credits, active_days, early_avg_score",
     "recommended_action": "Immediate tutor outreach recommended"},
    {"student_id": 2321492, "risk_score": 0.968, "risk_level": "High",
     "top_contributing_factors": "early_avg_score, week_4_clicks, total_clicks",
     "recommended_action": "Immediate tutor outreach recommended"},
    {"student_id": 605939, "risk_score": 0.656, "risk_level": "Medium",
     "top_contributing_factors": "studied_credits, total_clicks, week_4_clicks",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 628722, "risk_score": 0.538, "risk_level": "Medium",
     "top_contributing_factors": "studied_credits, week_4_clicks, early_avg_score",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 533091, "risk_score": 0.528, "risk_level": "Medium",
     "top_contributing_factors": "active_days, total_clicks, gender_enc",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 307291, "risk_score": 0.454, "risk_level": "Medium",
     "top_contributing_factors": "early_num_submitted, early_avg_score, active_days",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 286527, "risk_score": 0.352, "risk_level": "Medium",
     "top_contributing_factors": "early_avg_score, week_4_clicks, early_num_submitted",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 680231, "risk_score": 0.332, "risk_level": "Medium",
     "top_contributing_factors": "week_4_clicks, highest_education_enc, week_2_clicks",
     "recommended_action": "Monitor engagement over next 2 weeks"},
    {"student_id": 685705, "risk_score": 0.287, "risk_level": "Low",
     "top_contributing_factors": "early_avg_score, week_4_clicks, early_num_submitted",
     "recommended_action": "No action needed"},
    {"student_id": 595769, "risk_score": 0.282, "risk_level": "Low",
     "top_contributing_factors": "week_4_clicks, active_days, studied_credits",
     "recommended_action": "No action needed"},
    {"student_id": 540224, "risk_score": 0.229, "risk_level": "Low",
     "top_contributing_factors": "week_4_clicks, highest_education_enc, early_avg_score",
     "recommended_action": "No action needed"},
    {"student_id": 588233, "risk_score": 0.192, "risk_level": "Low",
     "top_contributing_factors": "studied_credits, early_avg_score, early_num_submitted",
     "recommended_action": "No action needed"},
    {"student_id": 626379, "risk_score": 0.173, "risk_level": "Low",
     "top_contributing_factors": "studied_credits, early_avg_score, early_num_submitted",
     "recommended_action": "No action needed"},
    {"student_id": 682306, "risk_score": 0.168, "risk_level": "Low",
     "top_contributing_factors": "week_4_clicks, active_days, total_clicks",
     "recommended_action": "No action needed"},
    {"student_id": 528351, "risk_score": 0.067, "risk_level": "Low",
     "top_contributing_factors": "early_avg_score, week_4_clicks, active_days",
     "recommended_action": "No action needed"},
]

FACTOR_LABELS = {
    "studied_credits": "Studied credits",
    "active_days": "Active days, early weeks",
    "early_avg_score": "Early assessment score",
    "week_4_clicks": "Week 4 VLE clicks",
    "week_2_clicks": "Week 2 VLE clicks",
    "total_clicks": "Total early clicks",
    "early_num_submitted": "Assessments submitted early",
    "gender_enc": "Gender",
    "highest_education_enc": "Highest prior education",
}


@st.cache_data
def load_data():
    local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_sample_output.csv")
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
    else:
        df = pd.DataFrame(FALLBACK_DATA)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df


df = load_data()

# ── Case-file visual system: serif headers + monospace data, muted risk tones ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --paper: #F1EFE9;
    --paper-raised: #FBFAF7;
    --ink: #1C1E24;
    --ink-soft: #565A66;
    --line: #C9C4B4;
    --line-soft: #DEDACD;
    --high: #7A2222;
    --high-tint: #F2E4E1;
    --medium: #7A5A12;
    --medium-tint: #F1E9D6;
    --low: #2E4F35;
    --low-tint: #E4EBE2;
}

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--paper); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1180px; }

.masthead {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 14px;
    margin-bottom: 28px;
}
.masthead .kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-soft);
}
.masthead h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 30px;
    color: var(--ink);
    margin: 4px 0 6px 0;
}
.masthead .sub {
    font-size: 13px;
    color: var(--ink-soft);
    font-family: 'IBM Plex Mono', monospace;
}

/* Register list (left column) */
.register-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin-bottom: 10px;
}

div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
    width: 100%;
    text-align: left;
    background: var(--paper-raised);
    border: 1px solid var(--line-soft);
    border-radius: 2px;
    padding: 9px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--ink);
    margin-bottom: 3px;
    transition: border-color 0.12s ease;
}
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button:hover {
    border-color: var(--ink);
    color: var(--ink);
}
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button:focus:not(:active) {
    border-color: var(--ink);
    box-shadow: none;
}

/* Case report (right column) */
.case-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--line);
    padding-bottom: 12px;
    margin-bottom: 20px;
}
.case-header .case-id {
    font-family: 'Source Serif 4', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--ink);
}
.case-header .case-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
}
.risk-stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 5px 12px;
    border: 1.5px solid;
    border-radius: 2px;
}
.risk-stamp.High { color: var(--high); border-color: var(--high); }
.risk-stamp.Medium { color: var(--medium); border-color: var(--medium); }
.risk-stamp.Low { color: var(--low); border-color: var(--low); }

.panel {
    background: var(--paper-raised);
    border: 1px solid var(--line-soft);
    border-radius: 2px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.panel .panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 14px;
}

/* Tick-mark risk gauge (signature element) */
.gauge-score {
    font-family: 'Source Serif 4', serif;
    font-size: 40px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
    margin-bottom: 14px;
}
.gauge-score .unit { font-size: 18px; font-weight: 400; color: var(--ink-soft); }

.gauge-track {
    position: relative;
    height: 34px;
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    margin-bottom: 6px;
}
.gauge-fill {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: repeating-linear-gradient(
        90deg, var(--fill-color) 0px, var(--fill-color) 2px,
        transparent 2px, transparent 5px
    );
}
.gauge-needle {
    position: absolute;
    top: -6px;
    bottom: -6px;
    width: 2px;
    background: var(--ink);
}
.gauge-needle::after {
    content: "";
    position: absolute;
    top: -5px;
    left: -4px;
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid var(--ink);
}
.gauge-ticks {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--ink-soft);
    margin-top: 4px;
}

/* Factor ledger */
.factor-line {
    display: flex;
    align-items: baseline;
    padding: 7px 0;
    border-bottom: 1px dotted var(--line);
    font-size: 13.5px;
    color: var(--ink);
}
.factor-line:last-child { border-bottom: none; }
.factor-line .rank {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink-soft);
    width: 22px;
    flex-shrink: 0;
}

/* Directive box */
.directive {
    border-left: 3px solid var(--tone-color);
    background: var(--tone-tint);
    padding: 14px 18px;
    font-size: 14.5px;
}
.directive .d-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--tone-color);
    margin-bottom: 5px;
}
.directive .d-text {
    font-weight: 600;
    color: var(--ink);
}

.empty-state {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink-soft);
    font-size: 13px;
    padding: 60px 20px;
    text-align: center;
    border: 1px dashed var(--line);
}

.filter-row div[data-testid="stRadio"] > div { flex-direction: row; gap: 4px; }
.filter-row div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
.filter-row div[data-testid="stRadio"] label span,
.filter-row div[data-testid="stRadio"] label p,
.filter-row div[data-testid="stRadio"] label div {
    color: var(--ink) !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Masthead ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="kicker">Early Warning System &middot; Research Prototype</div>
    <h1>Student Risk Case Register</h1>
    <div class="sub">Design and Evaluation of a Transformer-Based Early Warning System for At-Risk Student Identification &middot; XGBoost predictions, OULAD weeks 0&ndash;4</div>
</div>
""", unsafe_allow_html=True)

if "selected_id" not in st.session_state:
    st.session_state.selected_id = int(df.iloc[0]["student_id"])
if "level_filter" not in st.session_state:
    st.session_state.level_filter = "All"

col_list, col_detail = st.columns([1, 2], gap="large")

with col_list:
    st.markdown(f'<div class="register-label">Register &middot; {len(df)} records</div>', unsafe_allow_html=True)

    st.markdown('<div class="filter-row">', unsafe_allow_html=True)
    level_filter = st.radio(
        "Filter", ["All", "High", "Medium", "Low"],
        index=["All", "High", "Medium", "Low"].index(st.session_state.level_filter),
        horizontal=True, label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state.level_filter = level_filter

    filtered = df if level_filter == "All" else df[df["risk_level"] == level_filter]

    for _, row in filtered.iterrows():
        sid = int(row["student_id"])
        label = f"{row['risk_level'][0]}  \u00b7  {sid}  \u00b7  {row['risk_score']*100:.0f}%"
        if st.button(label, key=f"btn_{sid}", use_container_width=True):
            st.session_state.selected_id = sid

with col_detail:
    sel = df[df["student_id"] == st.session_state.selected_id]
    if sel.empty:
        st.markdown('<div class="empty-state">Select a record from the register</div>', unsafe_allow_html=True)
    else:
        row = sel.iloc[0]
        level = row["risk_level"]
        score_pct = row["risk_score"] * 100
        tone = {"High": "#7A2222", "Medium": "#7A5A12", "Low": "#2E4F35"}[level]
        tint = {"High": "#F2E4E1", "Medium": "#F1E9D6", "Low": "#E4EBE2"}[level]

        st.markdown(f"""
        <div class="case-header">
            <div>
                <div class="case-label">Case File</div>
                <div class="case-id">Student {int(row['student_id'])}</div>
            </div>
            <div class="risk-stamp {level}">{level} Risk</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="panel" style="--fill-color: {tone};">
            <div class="panel-label">Predicted Risk Score</div>
            <div class="gauge-score">{score_pct:.1f}<span class="unit">%</span></div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width: {score_pct}%;"></div>
                <div class="gauge-needle" style="left: {score_pct}%;"></div>
            </div>
            <div class="gauge-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
        </div>
        """, unsafe_allow_html=True)

        factors = [f.strip() for f in row["top_contributing_factors"].split(",")]
        factor_html = "".join(
            f'<div class="factor-line"><span class="rank">{i+1:02d}</span>{FACTOR_LABELS.get(f, f)}</div>'
            for i, f in enumerate(factors)
        )
        st.markdown(f"""
        <div class="panel">
            <div class="panel-label">Contributing Factors, Ranked by SHAP Value</div>
            {factor_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="directive" style="--tone-color: {tone}; --tone-tint: {tint};">
            <div class="d-label">Recommended Tutor Action</div>
            <div class="d-text">{row['recommended_action']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-soft); margin-top: 20px; border-top: 1px solid var(--line); padding-top: 12px;">
            Model: XGBoost, trained on OULAD (weeks 0&ndash;4 engagement and demographic features).
            Factors ranked by SHAP value for this individual prediction. Research prototype;
            no automated decisions are made from these outputs.
        </div>
        """, unsafe_allow_html=True)
