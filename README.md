# NurtureNet
### Dual-Architecture Agentic Maternal Triage for Community Health Workers

**Vanderbilt University — DS 5690: Gen AI Models in Theory & Practice, Spring 2026**
**Mary Morkos** | MS Data Science | mary.morkos@vanderbilt.edu

---

## Live Demo

**[nurturenet.streamlit.app](https://nurturenet.streamlit.app)**

The deployed application runs the NurtureNet CHW Skill — Claude with the full clinical protocol, SDOH Composite Burden Index, and constitutional safety principles. The complete two-layer pipeline with local Phi-4-mini runs via evaluate.py.

---

## Research Question

**Can a dual-architecture agentic system — combining a quantized small language model running locally with a cloud-based constitutional AI reviewer — improve equitable preeclampsia risk detection for community health workers serving rural women between clinic visits?**

---

## The Problem

Over 2.3 million US women live in maternity care deserts where the nearest obstetric provider is more than 38 minutes away (March of Dimes, 2024). Preeclampsia affects approximately 8% of all US deliveries and is present in nearly one in three maternal deaths during delivery hospitalization (Ford et al., 2022; Fink et al., 2023). Current ACOG screening detects only 61.5% of preterm preeclampsia cases (Guerby et al., 2024), and 84% of pregnancy-related deaths are preventable (Trost et al., 2022).

In 2024, non-Hispanic Black women died from maternal causes at a rate of 44.8 per 100,000 live births compared to 14.2 for non-Hispanic White women — a ratio of 3.15 to 1 (Hoyert, 2026). In Tennessee specifically, Black women are three times more likely to die from pregnancy-related causes than White women, and TennCare patients die at three times the rate of those with private insurance (Tennessee Lookout, 2025). I was born and raised in Nashville, Tennessee. This project is personal.

The core problem is architectural. Prenatal care is organized around the clinic visit as the unit of detection — but preeclampsia develops and kills between visits. Community health workers (CHWs) are the only people who actually reach rural women in that gap. Zero published AI systems have been built specifically for CHWs conducting between-visit maternal health monitoring.

NurtureNet is built for them.

---

## Who This Tool Serves

Community health workers — not physicians, not patients directly. CHWs conduct home visits, check in on patients between clinic appointments, and serve as the bridge between rural pregnant women and the healthcare system. They need one thing: a clear answer about whether a patient is okay, or whether she needs to get to a doctor right now. NurtureNet gives them that answer plus the exact words to say to the patient.

---

## Architecture
CHW enters patient vitals + SDOH during home visit
|
v
Layer 1: Phi-4-mini (local, no internet)
3.8B parameters, 4-bit quantized via Ollama
Chain-of-thought ACOG clinical reasoning
Outputs: risk level + confidence + CHW action
|
|
|                     |
Low risk +            Moderate/High OR
high confidence       confidence < 0.7 OR
(stay on device)      SDOH burden >= 7 OR
BP >= 140/90
|
v
Layer 2: Claude + NurtureNet CHW Skill
Constitutional review of Layer 1 reasoning
SDOH Composite Burden Index
Equity flag for patients of color
CHW action with specific timeline
Plain-language patient script
Clinician handoff document

**Why two layers?** Phi-4-mini runs on any phone, requires no internet, and protects patient privacy. Claude handles high-stakes reasoning where constitutional safety principles and equity analysis matter most. The architecture matches computational resources to clinical stakes.

> **Deployment note:** The live Streamlit app at [nurturenet.streamlit.app](https://nurturenet.streamlit.app) runs the NurtureNet CHW Skill via Claude only — Phi-4-mini requires a local Ollama server and runs via `evaluate.py`. The full two-layer pipeline is the research contribution; the deployed app is the CHW interface.

**Why Phi-4-mini?** Scaling laws (Kaplan et al., 2020) show a large pretrained model quantized for edge deployment outperforms a smaller model trained from scratch. Phi-4-mini at Q4_K_M fits in 2.4GB and runs at 20-35 tok/s on a MacBook Air M1 — the same hardware profile as a mid-range Android phone. HealthSLM-Bench (Wang et al., 2025) found Phi-family models among the top performers for mobile health monitoring at 16x faster inference than large models.

---

## Course Connections

| Concept | Implementation |
|---|---|
| DInference (Algorithm 14, Phuong & Hutter 2022) | Phi-4-mini local triage — decoder generates risk assessment token by token |
| DInference (Algorithm 14, Phuong & Hutter 2022) | Claude constitutional review — same algorithm, cloud scale, equity reasoning |
| Scaling laws (Kaplan et al. 2020) | Quantized large model beats small model from scratch |
| Constitutional AI (Anthropic 2022) | 7 explicit clinical safety principles checked every output |
| Chain-of-thought prompting | Phi-4-mini forced to reason step by step before concluding |
| Claude Skills | NurtureNet CHW Skill encodes complete clinical protocol in SKILL.md |

---

## NurtureNet SDOH Composite Burden Index

Original contribution. Weights grounded in published epidemiological effect sizes:

```python
sdoh_burden = (
    food_insecure          * 2 +  # Blumenshine et al. 2010 — OR 2.0 gestational diabetes
    housing_instability    * 2 +  # AHRQ 2020 — cortisol elevation raises BP
    late_prenatal_care     * 3 +  # Trost et al. 2022 — strongest missed detection predictor
    rural                  * 1 +  # Ford et al. 2022 — limited MFM specialist access
    (insurance == 'Uninsured')       * 3 +  # KFF 2023 — 3x late diagnosis risk
    (insurance == 'Medicaid / CHIP') * 1 +
    (race == 'Non-Hispanic Black')   * 2    # Hoyert 2026 — 3.15x mortality ratio
)
```

Score: 0-3 Low | 4-6 Moderate | 7-9 High | 10+ Critical

---

## Constitutional Principles

1. NEVER assess low risk when BP >= 140/90 mmHg
2. Prior preeclampsia + ANY BP elevation = minimum moderate risk, always
3. SDOH burden >= 7 must be explicitly flagged in CHW guidance
4. Non-Hispanic Black patients: heightened vigilance at ALL thresholds (3.15x mortality, Hoyert 2026)
5. Confidence < 0.7 on moderate/high assessment = escalate, never reassure
6. Every output must give the CHW a specific action with a timeline
7. Never use language that could discourage the patient from seeking care

---

## Evaluation Results

20 patient vignettes, ACOG-grounded ground truth. 8 high risk, 6 moderate, 6 low.
Demographics: 6 Non-Hispanic Black, 4 Hispanic, 6 White, 2 Asian, 2 AIAN.

| Model | Overall | Non-Hispanic Black | Hispanic | White | AIAN |
|---|---|---|---|---|---|
| Phi-4-mini (no SDOH) | 100% | 100% | 100% | 100% | 100% |
| Phi-4-mini (w/ SDOH) | 88% | **80%** | 100% | 100% | 100% |
| After constitutional review | **100%** | **100%** | 100% | 100% | 100% |

**Finding 1:** Adding SDOH context to the local model hurt recall for Black patients (80% vs 100%). Constitutional review restored 100% recall.

**Finding 2:** 7 constitutional violations across 20 cases — every single violation was in a patient of color. Not one violation in a White patient with private insurance.

**Finding 3:** 3 risk assessments upgraded by constitutional review — all patients of color with elevated SDOH burden.

**Finding 4:** 15 equity flags raised that Phi-4-mini never generated alone.

### Constitutional violations by case

| Case | Race/Ethnicity | Principles Violated |
|---|---|---|
| 4 | Non-Hispanic Black | 3, 4 |
| 7 | American Indian or Alaska Native | 3, 4 |
| 9 | Hispanic or Latina | 3 |
| 12 | Non-Hispanic Black | 1, 3, 4, 6 |
| 13 | Hispanic or Latina | 3 |
| 14 | Non-Hispanic Black | 3, 4 |
| 19 | Non-Hispanic Black | 4 |

---

## Example Output — Case 1

Patient: 32 weeks, BP 158/102, severe headache, visual disturbances, prior preeclampsia.
Non-Hispanic Black, uninsured, rural, food insecure, housing instability, no prenatal care. SDOH burden 11/14.

**Phi-4-mini:** HIGH, 90% confidence

**After constitutional review:**
- Equity flag: Non-Hispanic Black patient with critical SDOH burden (11/14) — heightened vigilance required per 3.15x mortality risk
- CHW action: Call 911 immediately. Do not delay. Stay with patient until EMS arrives.
- What to say: "Your blood pressure is very high and you have warning signs that need immediate medical attention. I'm calling 911 right now to get you to the hospital safely. This is about keeping you and your baby healthy — the doctors need to see you today."

---

## Limitations

- 20 vignettes is a proof-of-concept evaluation set, not a statistically powered clinical validation study
- Phi-4-mini can hallucinate — the constitutional review layer mitigates but does not eliminate this risk
- Ground truth labels are ACOG-based but assigned by the researcher, not clinicians
- No real CHW user testing
- Not validated on real patient data — all vignettes are synthetic
- No IRB approval
- Not FDA cleared

---

## Setup

```bash
brew install ollama
ollama serve
ollama pull phi4-mini
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python evaluate.py --case 1
python evaluate.py
streamlit run app.py
```

---

```
NurtureNet/
├── evaluate.py              # Two-layer evaluation harness
├── app.py                   # Streamlit CHW interface
├── data/
│   ├── vignettes.json                # 20 vignettes with ground truth
│   └── vignettes_tennessee.json      # 115 Tennessee vignettes, 23 ethnic groups
├── skill/
│   └── SKILL.md                      # NurtureNet CHW Skill
├── results/
├── requirements.txt
└── README.md
```

---

## Roadmap

1. **WhatsApp interface** — CHW texts patient info, NurtureNet responds in 30 seconds. Twilio WhatsApp API. In development.
2. **Tennessee evaluation** — 115 cases across 23 ethnic groups including Nashville refugee communities (Somali, Kurdish, Burmese/Karen, Congolese, Guatemalan), Appalachian rural families, Cherokee Nation members in East Tennessee.
3. **QLoRA fine-tuning** — adapt Phi-4-mini on clinical vignettes using MLX
4. **VUMC retrospective validation** — Vanderbilt CTSA IRB pathway
5. **CMS UDS SDOH integration** — real measured SDOH data from FQHCs

---

## References

**Course papers:**
- Phuong & Hutter (2022). Formal Algorithms for Transformers. arXiv:2207.09238
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361

**Maternal mortality:**
- Hoyert (2026). Maternal Mortality Rates in the United States, 2024. doi:10.15620/cdc/174651
- Trost et al. (2022). Pregnancy-Related Deaths: Data from MMRCs. CDC.
- Petersen et al. (2019). Racial/Ethnic Disparities in Pregnancy-Related Deaths. MMWR 68(35).
- Ramos et al. (2021). Racial and Ethnic Disparities in Maternal Mortality. AJPH 111(9).
- Ford et al. (2022). Hypertensive Disorders in Pregnancy. MMWR 71(17).
- Fink et al. (2023). Trends in Maternal Mortality. JAMA Network Open 6(6).
- Guerby et al. (2024). PREDICTION Study. Hypertension 81(7).
- March of Dimes (2024). Maternity Care Deserts Report.

**Edge AI:**
- Wang et al. (2025). HealthSLM-Bench. arXiv:2509.07260

**Responsible AI:**
- Obermeyer et al. (2019). Dissecting Racial Bias. Science 366(6464).
- Anthropic (2022). Constitutional AI: Harmlessness from AI Feedback.

---

*NurtureNet is a research prototype for DS 5690 at Vanderbilt University. Not FDA cleared. Not a substitute for clinical judgment. All outputs advisory only.*

*Author: Mary Morkos, MS Data Science, Vanderbilt University, May 2026*
