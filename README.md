# NurtureNet 2.0
### Dual-Architecture Agentic Maternal Triage for Community Health Workers

**Vanderbilt University — DS 5690: Gen AI Models in Theory & Practice, Spring 2026**
**Mary Morkos** | MS Data Science (Health Informatics) | mary.morkos@vanderbilt.edu

---

## Research Question

**Can a dual-architecture agentic system — combining a quantized small language model running locally with a cloud-based constitutional AI reviewer — improve equitable preeclampsia risk detection for community health workers serving rural women between clinic visits?**

---

## The Problem

Over 2.3 million US women live in maternity care deserts where the nearest obstetric provider is more than 38 minutes away (March of Dimes, 2024). Preeclampsia affects approximately 8% of all US deliveries and is present in nearly one in three maternal deaths during delivery hospitalization (Ford et al., 2022; Fink et al., 2023). Current ACOG screening detects only 61.5% of preterm preeclampsia cases (Guerby et al., 2024), and 84% of pregnancy-related deaths are preventable (Trost et al., 2022).

The mortality gap is not distributed equally. In 2024, non-Hispanic Black women died from maternal causes at a rate of 44.8 per 100,000 live births compared to 14.2 for non-Hispanic White women — a ratio of 3.15 to 1 (Hoyert, 2026). Preeclampsia and eclampsia mortality rates are approximately five times higher for Black women than White women (Ramos et al., 2021).

**The core problem is architectural.** Prenatal care is organized around the clinic visit as the unit of detection — but preeclampsia develops and kills between visits. Community health workers (CHWs) are the only people who actually reach rural women in that gap. They live in the community. They make home visits. They have smartphones. Zero published AI systems have been built specifically for CHWs conducting between-visit maternal health monitoring.

NurtureNet 2.0 is built for them.

---

## Who This Tool Serves

**Community health workers** — not physicians, not patients directly. CHWs are trained community members who conduct home visits, check in on patients between clinic appointments, and serve as the bridge between rural pregnant women and the healthcare system. They need one thing: a clear answer about whether a patient is okay, or whether she needs to get to a doctor right now. NurtureNet gives them that answer plus the exact words to say to the patient.

---

## Architecture

```
CHW enters patient vitals + SDOH during home visit
                    |
                    v
      Layer 1: Phi-4-mini (local, no internet)
      3.8B parameters, 4-bit quantized via Ollama
      Chain-of-thought ACOG clinical reasoning
      Outputs: risk level + confidence + CHW action
                    |
          __________|__________
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
```

**Why two layers?**

Phi-4-mini runs on any phone, requires no internet, and protects patient privacy — data never leaves the device for low-risk cases. Claude handles the high-stakes reasoning where constitutional safety principles, equity analysis, and clinical nuance matter most. The architecture matches computational resources to clinical stakes.

**Why Phi-4-mini specifically?**

Three reasons grounded in course content:

1. **Scaling laws (Kaplan et al., 2020):** Loss scales as a power law with model size. A large pretrained model quantized for edge deployment outperforms a smaller model trained from scratch on limited data. Phi-4-mini was pretrained by Microsoft on 5 trillion tokens — we get that knowledge for free and add clinical reasoning on top.

2. **HealthSLM-Bench (Wang et al., 2025):** The only benchmark specifically evaluating small language models for mobile health monitoring found Phi-family models among the top performers, achieving comparable accuracy to models 17x larger at 16x faster inference on mobile hardware.

3. **Deployment reality:** Phi-4-mini at Q4_K_M quantization fits in 2.4GB and runs at 20-35 tokens per second on a MacBook Air M1 — the same hardware profile as a mid-range Android phone. A CHW in a rural county has a phone, not a server. Phi-4-mini runs on that phone.

---

## Course Connections

| Concept | Implementation |
|---|---|
| DInference (Algorithm 14, Phuong & Hutter 2022) | Phi-4-mini local triage — decoder generates risk assessment token by token |
| DInference (Algorithm 14, Phuong & Hutter 2022) | Claude constitutional review — same algorithm, cloud scale, equity reasoning |
| Scaling laws (Kaplan et al. 2020) | Quantized large model beats small model from scratch; early stopping; small head |
| Constitutional AI (Anthropic 2022) | 7 explicit clinical safety principles checked against every output |
| Chain-of-thought prompting | Phi-4-mini forced to reason step by step before concluding |
| Claude Skills | NurtureNet CHW Skill encodes complete clinical protocol in SKILL.md |
| Two-model comparison | Same DInference algorithm at two scales — first empirical comparison for maternal health equity |

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

Score interpretation: 0-3 Low | 4-6 Moderate | 7-9 High | 10+ Critical

---

## Constitutional Principles

Seven explicit safety rules the Claude review layer checks against every output:

1. NEVER assess low risk when BP >= 140/90 mmHg
2. Prior preeclampsia + ANY BP elevation = minimum moderate risk, always
3. SDOH burden >= 7 must be explicitly flagged in CHW guidance
4. Non-Hispanic Black patients: heightened vigilance at ALL thresholds (3.15x mortality, Hoyert 2026)
5. Confidence < 0.7 on moderate/high assessment = escalate, never reassure
6. Every output must give the CHW a specific action with a timeline
7. Never use language that could discourage the patient from seeking care

---

## Evaluation Results

**20 patient vignettes** with ACOG-grounded ground truth labels.
Distribution: 8 high risk, 6 moderate risk, 6 low risk.
Demographics: 6 Non-Hispanic Black, 4 Hispanic, 6 White, 2 Asian, 2 AIAN.

### High-risk recall

| Model | Overall | Non-Hispanic Black | Hispanic | White | AIAN |
|---|---|---|---|---|---|
| Phi-4-mini (no SDOH) | 100% | 100% | 100% | 100% | 100% |
| Phi-4-mini (w/ SDOH) | 88% | **80%** | 100% | 100% | 100% |
| After constitutional review | **100%** | **100%** | 100% | 100% | 100% |

### Key findings

**Finding 1: Adding SDOH context to the local model hurt recall for Black patients.**
Phi-4-mini with SDOH performed worse on Black high-risk patients (80%) than without SDOH context (100%). Adding social information confused the local model on 1 of 5 Black high-risk cases. The constitutional review caught and corrected this, restoring 100% recall. This is a concrete demonstration of why a safety review layer is necessary — the local model cannot reliably integrate equity-relevant social context.

**Finding 2: Constitutional violations were concentrated in patients of color.**
7 of 20 cases triggered constitutional violations. Every single violation was in a patient of color — Non-Hispanic Black, Hispanic, or American Indian/Alaska Native. Not one violation occurred in a White patient with private insurance. This is not random. It reflects Phi-4-mini's consistent underweighting of social context and equity considerations even when that context was explicitly provided. Principles 3 and 4 (SDOH burden flagging and heightened vigilance for patients of color) were the most frequently violated.

**Finding 3: The review layer upgraded 3 cases that the local model underestimated.**
3 patients received higher risk assessments after constitutional review than the local model assigned. All 3 were patients of color with elevated SDOH burden. These are the cases where the dual-architecture system changed the clinical outcome — a CHW acting on the local model's assessment alone would have underresponded.

**Finding 4: 15 equity flags were raised that the local model did not generate.**
The constitutional review raised equity flags in 15 of 20 cases — providing CHWs with specific equity-aware guidance that Phi-4-mini alone never produced.

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

**Patient:** 32 weeks, BP 158/102, severe headache, visual disturbances, prior preeclampsia.
Non-Hispanic Black, uninsured, rural, food insecure, housing instability, no prenatal care. SDOH burden 11/14.

**Phi-4-mini:** HIGH, 90% confidence

**After constitutional review:**
- Risk: HIGH (unchanged)
- Equity flag: Non-Hispanic Black patient with critical SDOH burden (11/14) including uninsured status and rural location — heightened vigilance required per 3.15x mortality risk
- CHW action: Call 911 immediately for emergency transport. Do not delay for patient to arrange own transport. Stay with patient until EMS arrives. Call receiving hospital to report incoming preeclampsia with severe features.
- What to say: "Your blood pressure is very high and you have warning signs that need immediate medical attention. I'm calling 911 right now to get you to the hospital safely. This is about keeping you and your baby healthy — the doctors need to see you today."

---

## Limitations

- 20 vignettes is a proof-of-concept evaluation set, not a statistically powered clinical validation study
- Phi-4-mini can hallucinate — the constitutional review layer mitigates but does not eliminate this risk
- Ground truth labels are ACOG-based but assigned by the researcher, not clinicians
- No real CHW user testing — the interface has not been evaluated for usability or comprehension
- Not validated on real patient data — all vignettes are synthetic
- No IRB approval — real patient data requires institutional review board oversight
- Not FDA cleared

---

## Setup

```bash
# 1. Install Ollama and pull Phi-4-mini
brew install ollama
ollama serve
ollama pull phi4-mini

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# 4. Run the demo case (Case 1 — high-risk Black patient, SDOH burden 11)
python evaluate.py --case 1

# 5. Run full evaluation (all 20 vignettes, ~15 minutes)
python evaluate.py

# 6. Run local model only (no API key needed)
python evaluate.py --no-review

# 7. Launch CHW Streamlit app
streamlit run app.py
```

---

## Repository Structure

```
nurturenet2/
├── evaluate.py              # Main evaluation harness — run this
├── app.py                   # Streamlit CHW interface
├── data/
│   └── vignettes.json       # 20 patient vignettes with ground truth
├── skill/
│   └── SKILL.md             # NurtureNet CHW Skill (Claude Skills format)
├── results/                 # Evaluation outputs (JSON)
├── requirements.txt
└── README.md
```

---

## Roadmap

1. **WhatsApp interface** — CHW texts patient info in plain language, NurtureNet responds in 30 seconds. Works on any phone, no app install, 2G cellular. Twilio WhatsApp API. In development.
2. **Tennessee-specific vignette set** — 115 cases across 23 ethnic groups grounded in Tennessee demographics, including Nashville refugee communities (Somali, Kurdish, Burmese/Karen, Congolese, Guatemalan), Appalachian rural White families, and Cherokee Nation members in East Tennessee. See `data/vignettes_tennessee.json`.
3. **QLoRA fine-tuning** — adapt Phi-4-mini on clinical vignettes using MLX on MacBook Air
4. **VUMC retrospective validation** — Vanderbilt CTSA IRB pathway for real EHR data
5. **CMS UDS SDOH integration** — real measured SDOH data already collected at Federally Qualified Health Centers

---

## References

**Course papers:**
- Phuong, M. & Hutter, M. (2022). Formal Algorithms for Transformers. arXiv:2207.09238
- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361

**Maternal mortality and disparity:**
- Hoyert, D.L. (2026). Maternal Mortality Rates in the United States, 2024. doi:10.15620/cdc/174651
- Trost, S.L. et al. (2022). Pregnancy-Related Deaths: Data from MMRCs in 36 US States. CDC.
- Petersen, E.E. et al. (2019). Racial/Ethnic Disparities in Pregnancy-Related Deaths. MMWR 68(35). doi:10.15585/mmwr.mm6835a3
- Ramos, I.G. et al. (2021). Racial and Ethnic Disparities in Maternal Mortality. AJPH 111(9). doi:10.2105/AJPH.2021.306375
- Ford, N.D. et al. (2022). Hypertensive Disorders in Pregnancy and Mortality. MMWR 71(17). doi:10.15585/mmwr.mm7117a1
- Fink, D.A. et al. (2023). Trends in Maternal Mortality. JAMA Network Open 6(6). doi:10.1001/jamanetworkopen.2023.17641

**Preeclampsia detection:**
- Guerby, P. et al. (2024). PREDICTION Study. Hypertension 81(7). doi:10.1161/HYPERTENSIONAHA.123.22584
- March of Dimes (2024). Maternity Care Deserts Report.

**Edge AI and small models:**
- Wang, Y. et al. (2025). HealthSLM-Bench. arXiv:2509.07260
- Microsoft (2025). Phi-4-mini technical report. azure.microsoft.com

**Responsible AI:**
- Obermeyer, Z. et al. (2019). Dissecting Racial Bias in an Algorithm. Science 366(6464). doi:10.1126/science.aax2342
- Anthropic (2022). Constitutional AI: Harmlessness from AI Feedback.

---

*NurtureNet 2.0 is a research prototype developed for DS 5690 at Vanderbilt University.
Not FDA cleared. Not a substitute for clinical judgment or physician oversight.
All outputs are advisory only.*

*Author: Mary Morkos, MS Data Science (Health Informatics), Vanderbilt University, May 2026*
