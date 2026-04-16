“””
app.py — NurtureNet Community Health Worker Tool
“””

import streamlit as st
import anthropic
import json
import os

st.set_page_config(
page_title=“NurtureNet”,
page_icon=“🌿”,
layout=“centered”
)

# ── Custom CSS ────────────────────────────────────────────────────

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #faf7f4;
}

.hero {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 60%, #3d7a52 100%);
    border-radius: 20px;
    padding: 40px 36px 32px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -40px;
    right: -40px;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -60px;
    left: -30px;
    width: 250px;
    height: 250px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin: 0 0 16px 0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 0.75rem;
    padding: 5px 12px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7c6e;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #e8e2da;
}

.sdoh-card {
    background: white;
    border-radius: 14px;
    padding: 16px 20px;
    border: 1px solid #e8e2da;
    margin: 16px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sdoh-label {
    font-size: 0.8rem;
    color: #6b7c6e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: white;
    border-radius: 16px;
    padding: 28px 28px;
    margin: 20px 0;
    text-align: center;
}

.risk-moderate {
    background: linear-gradient(135deg, #78350f, #92400e);
    color: white;
    border-radius: 16px;
    padding: 28px 28px;
    margin: 20px 0;
    text-align: center;
}

.risk-low {
    background: linear-gradient(135deg, #14532d, #166534);
    color: white;
    border-radius: 16px;
    padding: 28px 28px;
    margin: 20px 0;
    text-align: center;
}

.risk-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.risk-sub {
    font-size: 0.8rem;
    opacity: 0.75;
    margin: 4px 0 0 0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.output-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e8e2da;
    margin: 12px 0;
}

.output-card-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7c6e;
    margin: 0 0 8px 0;
}

.output-card-text {
    font-size: 0.95rem;
    color: #1a2e1e;
    line-height: 1.6;
    margin: 0;
}

.patient-script {
    background: #f0f7f2;
    border-left: 3px solid #2d5a3d;
    border-radius: 0 12px 12px 0;
    padding: 18px 20px;
    font-size: 1rem;
    color: #1a3a2a;
    line-height: 1.7;
    font-style: italic;
    margin: 12px 0;
}

.violation-card {
    background: #fff5f5;
    border: 1px solid #fca5a5;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #7f1d1d;
}

.equity-card {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #78350f;
}

.handoff-card {
    background: #f8faff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 16px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #1e3a5f;
    font-family: 'DM Sans', monospace;
    line-height: 1.6;
}

.footer {
    text-align: center;
    font-size: 0.72rem;
    color: #9ba89d;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e8e2da;
    letter-spacing: 0.3px;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"] label {
    font-size: 0.85rem !important;
    color: #374840 !important;
    font-weight: 500 !important;
}

div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a2a, #2d5a3d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    margin-top: 8px !important;
    cursor: pointer !important;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #2d5a3d, #3d7a52) !important;
    transform: translateY(-1px) !important;
}
</style>

“””, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────

st.markdown(”””

<div class="hero">
    <p class="hero-subtitle">Vanderbilt DS 5690 · Spring 2026</p>
    <h1 class="hero-title">🌿 NurtureNet</h1>
    <p style="color:rgba(255,255,255,0.75); font-size:0.95rem; margin:0 0 20px 0; line-height:1.5;">
        AI triage for community health workers conducting<br>between-visit maternal monitoring in Tennessee.
    </p>
    <span class="hero-badge">Advisory only · Not for direct patient use · Not FDA cleared</span>
</div>
""", unsafe_allow_html=True)

# ── Patient vitals ────────────────────────────────────────────────

st.markdown(’<p class="section-label">Patient Vitals</p>’, unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
sbp = st.number_input(“Systolic BP (mmHg)”, 70, 220, 120)
dbp = st.number_input(“Diastolic BP (mmHg)”, 40, 140, 80)
ga  = st.number_input(“Gestational age (weeks)”, 1, 42, 28)
with col2:
hr        = st.number_input(“Heart rate (bpm)”, 40, 140, 76)
prev_pe   = st.checkbox(“Prior preeclampsia”)
multiples = st.checkbox(“Multiple gestation”)

# ── Symptoms ──────────────────────────────────────────────────────

st.markdown(’<p class="section-label">Symptoms</p>’, unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
headache   = st.checkbox(“Severe headache”)
vision     = st.checkbox(“Visual disturbances”)
epigastric = st.checkbox(“Epigastric / upper right pain”)
with c2:
edema   = st.checkbox(“Facial or hand swelling”)
nausea  = st.checkbox(“Sudden nausea / vomiting”)
no_symp = st.checkbox(“No symptoms”)

# ── Social context ────────────────────────────────────────────────

st.markdown(’<p class="section-label">Social Context</p>’, unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
race = st.selectbox(“Race/ethnicity”, [
“Non-Hispanic White”,
“Non-Hispanic Black”,
“Hispanic or Latina”,
“Asian or Pacific Islander”,
“American Indian or Alaska Native”,
“Other / prefer not to say”
])
insurance = st.selectbox(“Insurance”, [
“Private insurance”,
“Medicaid / CHIP”,
“Uninsured”
])
with col4:
food_insecure = st.checkbox(“Food insecure”)
rural         = st.checkbox(“Rural area”)
housing       = st.checkbox(“Housing instability”)
late_care     = st.checkbox(“Late or no prenatal care”)

# ── SDOH score ────────────────────────────────────────────────────

sdoh_burden = (
int(food_insecure) * 2 +
int(housing) * 2 +
int(late_care) * 3 +
int(rural) * 1 +
(3 if insurance == “Uninsured” else 1 if insurance == “Medicaid / CHIP” else 0) +
(2 if race == “Non-Hispanic Black” else 0)
)

burden_level = “Low” if sdoh_burden <= 3 else “Moderate” if sdoh_burden <= 6 else “High” if sdoh_burden <= 9 else “Critical”
burden_color_hex = “#16a34a” if sdoh_burden <= 3 else “#d97706” if sdoh_burden <= 6 else “#dc2626”

st.markdown(f”””

<div class="sdoh-card">
    <span class="sdoh-label">SDOH Burden Score</span>
    <span style="font-size:1.4rem; font-weight:700; color:{burden_color_hex}; font-family:'DM Serif Display',serif;">
        {sdoh_burden}<span style="font-size:0.9rem; color:#9ba89d;">/14</span>
        &nbsp;<span style="font-size:0.75rem; font-weight:500; color:{burden_color_hex};">{burden_level}</span>
    </span>
</div>
""", unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────

if st.button(“Run NurtureNet Assessment”):

```
symptoms = []
if headache:    symptoms.append("severe headache")
if vision:      symptoms.append("visual disturbances")
if epigastric:  symptoms.append("epigastric pain")
if edema:       symptoms.append("facial or hand swelling")
if nausea:      symptoms.append("sudden nausea or vomiting")
if multiples:   symptoms.append("multiple gestation")
if not symptoms or no_symp:
    symptoms = ["none reported"]

try:
    with open("skill/SKILL.md") as f:
        skill = f.read()
except:
    skill = ""

prompt = f"""You are the NurtureNet constitutional AI reviewer for community health workers.
```

PATIENT:

- Gestational age: {ga} weeks
- Blood pressure: {sbp}/{dbp} mmHg
- Heart rate: {hr} bpm
- Symptoms: {’, ’.join(symptoms)}
- Prior preeclampsia: {prev_pe}
- Race/ethnicity: {race}
- Insurance: {insurance}
- Food insecure: {food_insecure}
- Rural: {rural}
- Housing instability: {housing}
- Late or no prenatal care: {late_care}
- SDOH burden score: {sdoh_burden}/14

NURTURENET CHW SKILL:
{skill[:2000]}

CONSTITUTIONAL PRINCIPLES:

1. NEVER assess low risk when BP >= 140/90 mmHg
1. Prior preeclampsia + ANY BP elevation = minimum moderate risk, always
1. SDOH burden >= 7 must be explicitly flagged
1. Non-Hispanic Black patients: heightened vigilance at ALL thresholds (3.15x mortality)
1. Confidence < 0.7 on moderate/high = escalate
1. Every output must give the CHW a specific action with a timeline
1. Never discourage the patient from seeking care

Apply ACOG criteria and all constitutional principles. Return ONLY valid JSON:
{{
“risk_level”: “low” | “moderate” | “high”,
“principles_violated”: [1, 2, 3],
“equity_flag”: true | false,
“equity_note”: “note or null”,
“chw_action”: “specific action with timeline”,
“what_to_say”: “exact plain-language script for CHW to read to patient”,
“clinician_handoff”: “clinical summary if escalating, null if low risk”
}}”””

```
with st.spinner("Running assessment..."):
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result = json.loads(raw[start:end])

        risk = result.get("risk_level", "unknown").upper()
        violations = result.get("principles_violated", [])
        equity = result.get("equity_flag", False)

        risk_class = {"HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}.get(risk, "risk-low")
        risk_sub = {"HIGH": "Immediate action required", "MODERATE": "Close monitoring needed", "LOW": "Routine follow-up"}.get(risk, "")

        st.markdown(f"""
```

<div class="{risk_class}">
    <p class="risk-label">{risk} RISK</p>
    <p class="risk-sub">{risk_sub}</p>
</div>
""", unsafe_allow_html=True)

```
        if violations:
            st.markdown(f"""
```

<div class="violation-card">
    ⚠️ <strong>Constitutional violations caught:</strong> principles {violations}
</div>
""", unsafe_allow_html=True)

```
        if equity and result.get("equity_note"):
            st.markdown(f"""
```

<div class="equity-card">
    🏳 <strong>Equity flag:</strong> {result['equity_note']}
</div>
""", unsafe_allow_html=True)

```
        if result.get("chw_action"):
            st.markdown(f"""
```

<div class="output-card">
    <p class="output-card-label">CHW Action</p>
    <p class="output-card-text">{result['chw_action']}</p>
</div>
""", unsafe_allow_html=True)

```
        if result.get("what_to_say"):
            st.markdown('<p class="section-label">What to say to the patient</p>', unsafe_allow_html=True)
            st.markdown(f"""
```

<div class="patient-script">
    "{result['what_to_say']}"
</div>
""", unsafe_allow_html=True)

```
        if result.get("clinician_handoff"):
            st.markdown('<p class="section-label">Clinician Handoff Note</p>', unsafe_allow_html=True)
            st.markdown(f"""
```

<div class="handoff-card">
    {result['clinician_handoff']}
</div>
""", unsafe_allow_html=True)

```
    except Exception as e:
        st.error(f"Assessment error: {e}")
```

# ── Footer ────────────────────────────────────────────────────────

st.markdown(”””

<div class="footer">
    NurtureNet · DS 5690 Vanderbilt University 2026 · Mary Morkos<br>
    Not FDA cleared · Not a substitute for clinical judgment · All outputs advisory only
</div>
""", unsafe_allow_html=True)
