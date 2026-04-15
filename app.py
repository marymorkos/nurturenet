"""
app.py — NurtureNet 2.0 Community Health Worker Tool
=====================================================
Streamlit app for CHWs conducting between-visit maternal check-ins.

Usage:
    streamlit run app.py

Requires:
    - Ollama running with phi4-mini pulled
    - ANTHROPIC_API_KEY set in environment
"""

import streamlit as st
import anthropic
import requests
import json
import os

st.set_page_config(
    page_title="NurtureNet CHW",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 NurtureNet")
st.caption("Community Health Worker Maternal Triage Tool | DS 5690 Vanderbilt 2026")
st.info("For community health workers conducting between-visit check-ins. "
        "Not for direct patient use. All outputs are advisory only.")

# ── Patient vitals ────────────────────────────────────────────────
st.subheader("Patient vitals")
col1, col2 = st.columns(2)
with col1:
    sbp = st.number_input("Systolic BP (mmHg)", 70, 220, 120)
    dbp = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
    ga  = st.number_input("Gestational age (weeks)", 1, 42, 28)
with col2:
    hr       = st.number_input("Heart rate (bpm)", 40, 140, 76)
    prev_pe  = st.checkbox("Prior preeclampsia")
    multiples = st.checkbox("Multiple gestation (twins/triplets)")

st.subheader("Symptoms — check all that apply")
c1, c2 = st.columns(2)
with c1:
    headache   = st.checkbox("Severe headache")
    vision     = st.checkbox("Visual disturbances")
    epigastric = st.checkbox("Epigastric / upper right pain")
with c2:
    edema    = st.checkbox("Facial or hand swelling")
    nausea   = st.checkbox("Sudden nausea / vomiting")
    no_symp  = st.checkbox("No symptoms")

# ── SDOH ─────────────────────────────────────────────────────────
st.subheader("Social context")
col3, col4 = st.columns(2)
with col3:
    race = st.selectbox("Race/ethnicity", [
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Hispanic or Latina",
        "Asian or Pacific Islander",
        "American Indian or Alaska Native",
        "Other / prefer not to say"
    ])
    insurance = st.selectbox("Insurance", [
        "Private insurance",
        "Medicaid / CHIP",
        "Uninsured"
    ])
with col4:
    food_insecure = st.checkbox("Food insecure")
    rural         = st.checkbox("Rural area")
    housing       = st.checkbox("Housing instability")
    late_care     = st.checkbox("Late or no prenatal care")

# ── SDOH burden score ─────────────────────────────────────────────
sdoh_burden = (
    int(food_insecure) * 2 +
    int(housing) * 2 +
    int(late_care) * 3 +
    int(rural) * 1 +
    (3 if insurance == "Uninsured" else 1 if insurance == "Medicaid / CHIP" else 0) +
    (2 if race == "Non-Hispanic Black" else 0)
)

burden_color = (
    "green" if sdoh_burden <= 3
    else "orange" if sdoh_burden <= 6
    else "red"
)
st.markdown(f"**SDOH burden score:** :{burden_color}[{sdoh_burden}/14]")

# ── Run ───────────────────────────────────────────────────────────
if st.button("Run NurtureNet assessment", type="primary"):
    symptoms = []
    if headache:    symptoms.append("severe headache")
    if vision:      symptoms.append("visual disturbances")
    if epigastric:  symptoms.append("epigastric pain")
    if edema:       symptoms.append("facial/hand edema")
    if nausea:      symptoms.append("sudden nausea/vomiting")
    if multiples:   symptoms.append("multiple gestation")
    if not symptoms or no_symp:
        symptoms = ["none"]

    vignette = {
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "gestational_age_weeks": ga,
        "heart_rate": hr,
        "symptoms": symptoms,
        "prior_preeclampsia": prev_pe,
        "sdoh": {
            "race_ethnicity": race,
            "insurance": insurance,
            "food_insecure": food_insecure,
            "rural": rural,
            "housing_instability": housing,
            "late_prenatal_care": late_care,
            "sdoh_burden": sdoh_burden
        }
    }

    # ── Layer 1: Phi-4-mini ───────────────────────────────────────
    with st.spinner("Layer 1: Phi-4-mini local triage..."):
        local_prompt = f"""You are a clinical triage AI supporting a community health worker.
Reason through this patient step by step.

VITALS: BP {sbp}/{dbp} mmHg | HR {hr} bpm | GA {ga} weeks
SYMPTOMS: {', '.join(symptoms)}
PRIOR PREECLAMPSIA: {prev_pe}
SOCIAL CONTEXT: Race: {race} | Insurance: {insurance} | Food insecure: {food_insecure}
  Rural: {rural} | Housing instability: {housing} | Late prenatal care: {late_care}
  SDOH burden: {sdoh_burden}/14

ACOG: BP >= 140/90 = hypertension | BP >= 160/110 = severe

Step 1: Evaluate BP
Step 2: Evaluate symptoms
Step 3: Consider history
Step 4: Assess SDOH burden
Step 5: Synthesize risk

Return JSON only:
{{"risk_level": "low"|"moderate"|"high", "confidence": 0.0-1.0,
  "reasoning": "brief explanation", "key_factors": ["f1","f2","f3"]}}"""

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi4-mini", "prompt": local_prompt, "stream": False},
                timeout=120
            )
            raw_local = r.json()["response"]
            start = raw_local.rfind("{")
            end = raw_local.rfind("}") + 1
            local_out = json.loads(raw_local[start:end])
        except Exception as e:
            local_out = {"risk_level": "unknown", "confidence": 0,
                         "reasoning": str(e), "key_factors": []}
            raw_local = str(e)

    local_risk = local_out.get("risk_level", "unknown")
    local_conf = local_out.get("confidence", 0)

    st.subheader("Layer 1 — Phi-4-mini (local, no internet)")
    color1 = {"low": "green", "moderate": "orange", "high": "red"}.get(local_risk, "gray")
    st.markdown(f"**:{color1}[{local_risk.upper()}]** — confidence {local_conf:.0%}")
    with st.expander("Local model reasoning"):
        st.write(local_out.get("reasoning", ""))
        st.write("Key factors:", local_out.get("key_factors", []))

    # ── Layer 2: Claude constitutional review ─────────────────────
    escalate = (
        local_risk in ["moderate", "high"] or
        local_conf < 0.7 or
        sdoh_burden >= 7 or
        sbp >= 140
    )

    if escalate:
        with st.spinner("Layer 2: Claude constitutional review..."):
            try:
                with open("skill/SKILL.md") as f:
                    skill = f.read()

                review_prompt = f"""You are the NurtureNet constitutional review layer for community health workers.

LOCAL MODEL OUTPUT:
{json.dumps(local_out, indent=2)}

LOCAL MODEL REASONING:
{raw_local[:2000]}

PATIENT:
{json.dumps(vignette, indent=2)}

NURTURENET CHW SKILL:
{skill[:2000]}

CONSTITUTIONAL PRINCIPLES:
1. NEVER assess low risk when BP >= 140/90
2. Prior preeclampsia + ANY BP elevation = minimum moderate
3. SDOH burden >= 7 must be explicitly flagged
4. Non-Hispanic Black: heightened vigilance at ALL thresholds (3.15x mortality)
5. Confidence < 0.7 on moderate/high = escalate

Review the local assessment. Correct if needed. Return ONLY valid JSON:
{{
  "final_risk_level": "low"|"moderate"|"high",
  "risk_changed": true|false,
  "change_direction": "upgraded"|"downgraded"|"unchanged",
  "principles_violated": [1,2,3],
  "equity_flag": true|false,
  "equity_note": "note or null",
  "chw_action": "specific action with timeline",
  "what_to_say": "exact script CHW reads to patient in plain language",
  "clinician_handoff": "clinical summary if escalating, null if low risk"
}}"""

                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY")
                )
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": review_prompt}]
                )
                raw_review = response.content[0].text.strip()
                raw_review = raw_review.replace("```json","").replace("```","").strip()
                review = json.loads(raw_review)
            except Exception as e:
                review = {
                    "final_risk_level": local_risk,
                    "risk_changed": False,
                    "error": str(e)
                }

        st.subheader("Layer 2 — Claude constitutional review")

        final = review.get("final_risk_level", local_risk)
        changed = review.get("risk_changed", False)
        direction = review.get("change_direction", "unchanged")
        violations = review.get("principles_violated", [])
        equity = review.get("equity_flag", False)

        color2 = {"low": "green", "moderate": "orange", "high": "red"}.get(final, "gray")
        change_note = f" — ⬆️ UPGRADED from {local_risk}" if (changed and direction == "upgraded") else ""
        st.markdown(f"**:{color2}[{final.upper()}]**{change_note}")

        if violations:
            st.error(f"Constitutional violations caught: principles {violations}")

        if equity:
            st.warning(f"**Equity flag:** {review.get('equity_note', '')}")

        st.info(f"**CHW action:** {review.get('chw_action', '')}")

        if review.get("what_to_say"):
            st.subheader("What to say to the patient")
            st.success(f'"{review["what_to_say"]}"')

        if review.get("clinician_handoff"):
            st.subheader("Clinician handoff note")
            st.code(review["clinician_handoff"])

    else:
        st.success("Local model confident — no cloud escalation needed. "
                   "Patient data stayed on device.")

st.divider()
st.caption("NurtureNet 2.0 | DS 5690 Vanderbilt | Not FDA cleared | "
           "Not a substitute for clinical judgment")
