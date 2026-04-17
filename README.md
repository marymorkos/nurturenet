# NurtureNet
### Dual-Architecture Agentic Maternal Triage for Community Health Workers

**Vanderbilt University — DS 5690: Gen AI Models in Theory & Practice, Spring 2026**
**Mary Morkos** | MS Data Science | mary.morkos@vanderbilt.edu

---

## Live Demo

**[nurturenet.streamlit.app](https://nurturenet.streamlit.app)**

The deployed application runs the NurtureNet CHW Skill — Claude with the full clinical protocol, Care Access Index, and constitutional safety principles. The complete two-layer pipeline with local Phi-4-mini runs via `evaluate.py`.

> **Deployment note:** The live Streamlit app runs the NurtureNet CHW Skill via Claude only. Phi-4-mini requires a local Ollama server and runs via `evaluate.py`. The full two-layer pipeline is the research contribution; the deployed app is the CHW interface.

---

## Research Question

**Can a dual-architecture agentic system — combining a quantized small language model running locally with a cloud-based constitutional AI reviewer — improve equitable preeclampsia risk detection for community health workers serving rural women between clinic visits?**

---

## The Problem

Over 2.3 million US women of reproductive age live in maternity care deserts — counties with no hospital or birth center offering obstetric care and no obstetric clinicians (March of Dimes, 2024). Preeclampsia complicates an estimated 2–8% of pregnancies globally (ACOG Practice Bulletin 222, 2020), and hypertensive disorders of pregnancy are documented in approximately one in three (31.6%) maternal deaths during delivery hospitalization (Ford et al., 2022). Current first-trimester ACOG screening detects only 61.5% of preterm preeclampsia cases (Guerby et al., 2024), and 84% of pregnancy-related deaths are preventable (Trost et al., 2022).

The mortality gap is not distributed equally. In 2024, non-Hispanic Black women died from maternal causes at a rate of 44.8 per 100,000 live births compared to 14.2 for non-Hispanic White women — a ratio of 3.15 to 1 (Hoyert, 2026). In Tennessee specifically, Black women are 2 to 2.5 times more likely to die from pregnancy-related causes than White women, and TennCare patients die at nearly three times the rate of those with private insurance (Tennessee Department of Health Maternal Mortality Review, as reported by Tennessee Lookout, 2025). I was born and raised in Nashville, Tennessee. This project is personal.

The core problem is architectural. Prenatal care is organized around the clinic visit as the unit of detection — but preeclampsia develops and kills between visits. Community health workers (CHWs) are the only people who actually reach rural women in that gap. Zero published AI systems have been built specifically for CHWs conducting between-visit maternal health monitoring.

NurtureNet is built for them.

---

## Who This Tool Serves

Maternal care starts with the relationship next door. NurtureNet is built for the people already showing up: community health workers, faith community nurses, doulas, neighborhood navigators, and volunteers at religious congregations who are actively present in the lives of pregnant women in their communities. These are the people closest to the mothers who need the most support. This tool equips them to act.

They need one thing: a clear answer about whether a patient is okay, or whether she needs to get to a doctor right now. NurtureNet gives them that answer plus the exact words to say to the patient.

---

## Architecture

```
CHW enters patient vitals + social context during home visit
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
    (stay on device)      Care Access score >= 7 OR
                          BP >= 140/90
                               |
                               v
              Layer 2: Claude + NurtureNet CHW Skill
              Constitutional review of Layer 1 reasoning
              Care Access Index
              Equity flag for patients of color
              CHW action with specific timeline
              Plain-language patient script
              Clinician handoff document
```

**Why two layers?** Phi-4-mini runs on any phone, requires no internet, and protects patient privacy. Claude handles high-stakes reasoning where constitutional safety principles and equity analysis matter most. The architecture matches computational resources to clinical stakes.

**Why Phi-4-mini?** Scaling laws (Kaplan et al., 2020) show a large pretrained model quantized for edge deployment outperforms a smaller model trained from scratch. Phi-4-mini at Q4_K_M fits in 2.4GB and runs at 20-35 tok/s on a MacBook Air M1 — the same hardware profile as a mid-range Android phone. HealthSLM-Bench (Wang et al., 2025) benchmarked Phi-family models for on-device healthcare monitoring and found that fine-tuned small language models can match or surpass much larger models across healthcare tasks while delivering substantially lower memory usage and inference latency.

---

## Course Connections

| Concept | Implementation |
|---|---|
| DInference (Algorithm 14, Phuong & Hutter 2022) | Phi-4-mini local triage: decoder generates risk assessment token by token |
| DInference (Algorithm 14, Phuong & Hutter 2022) | Claude constitutional review: same algorithm, cloud scale, equity reasoning |
| Scaling laws (Kaplan et al. 2020) | Quantized large model beats small model trained from scratch |
| Constitutional AI (Anthropic 2022) | 7 explicit clinical safety principles checked against every output |
| Chain-of-thought prompting | Phi-4-mini forced to reason step by step before concluding |
| Claude Skills | NurtureNet CHW Skill encodes complete clinical protocol in SKILL.md |

---

## NurtureNet Care Access Index

Original contribution. Weights grounded in published epidemiological effect sizes:

```python
care_access_score = (
    food_insecure          * 2 +  # Laraia et al. 2010 — marginal food security OR 2.76 for GDM
    housing_instability    * 2 +  # Leifheit et al. 2020 — housing insecurity and adverse birth outcomes
    late_prenatal_care     * 3 +  # Trost et al. 2022 — late/no prenatal care top contributing factor
    rural                  * 1 +  # Ford et al. 2022 — higher HDP prevalence in rural counties (15.5%)
    (insurance == 'Uninsured')       * 3 +  # KFF 2024 — uninsured women delay or forgo prenatal care
    (insurance == 'Medicaid / CHIP') * 1 +
    (race == 'Non-Hispanic Black')   * 2    # Hoyert 2026 — 3.15x mortality ratio
)
```

Score: 0-3 Low | 4-6 Moderate | 7-9 High | 10+ Critical

---

## Constitutional Principles

Seven explicit safety rules Claude checks against every output:

1. NEVER assess low risk when BP >= 140/90 mmHg
2. Prior preeclampsia + ANY BP elevation = minimum moderate risk, always
3. Care Access score >= 7 must be explicitly flagged in CHW guidance
4. Non-Hispanic Black patients: heightened vigilance at ALL thresholds (3.15x mortality, Hoyert 2026)
5. Confidence < 0.7 on moderate/high assessment: escalate, never reassure
6. Every output must give the CHW a specific action with a timeline
7. Never use language that could discourage the patient from seeking care

---

## Evaluation Results

20 patient vignettes with ACOG-grounded ground truth labels.
Distribution: 8 high risk, 6 moderate risk, 6 low risk.
Demographics: 6 Non-Hispanic Black, 4 Hispanic, 6 White, 2 Asian, 2 AIAN.

| Model | Overall | Non-Hispanic Black | Hispanic | White | AIAN |
|---|---|---|---|---|---|
| Phi-4-mini (no SDOH) | 100% | 100% | 100% | 100% | 100% |
| Phi-4-mini (w/ SDOH) | 88% | **80%** | 100% | 100% | 100% |
| After constitutional review | **100%** | **100%** | 100% | 100% | 100% |

**Finding 1:** Adding SDOH context to the local model hurt recall for Black patients (80% vs 100%). Constitutional review restored 100% recall.

**Finding 2:** 7 constitutional violations across 20 cases. Every single violation was in a patient of color. Not one violation occurred in a White patient with private insurance.

**Finding 3:** 3 risk assessments upgraded by constitutional review. All 3 were patients of color with elevated care access burden.

**Finding 4:** 15 equity flags raised that Phi-4-mini never generated alone.

### Safety checks by case

| Case | Race/Ethnicity | Safety Checks Flagged |
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
Non-Hispanic Black, uninsured, rural, food insecure, housing instability, no prenatal care. Care Access score 11/14.

**Phi-4-mini:** HIGH, 90% confidence

**After constitutional review:**
- Equity note: Non-Hispanic Black patient with critical care access burden (11/14) including uninsured status and rural location. Heightened vigilance required per 3.15x mortality risk.
- What to do next: Call 911 immediately for emergency transport. Do not delay. Stay with patient until EMS arrives.
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
# To use the deployed CHW interface:
# nurturenet.streamlit.app — no setup needed

# To run the full two-layer research evaluation locally:
brew install ollama
ollama serve
ollama pull phi4-mini
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python evaluate.py --case 1   # demo — Case 1, high-risk Black patient, Care Access 11/14
python evaluate.py            # full evaluation, all 20 vignettes
streamlit run app.py          # local CHW interface
```

---

## Repository Structure

```
NurtureNet/
├── evaluate.py                       # Two-layer evaluation harness
├── app.py                            # Streamlit CHW interface
├── MANIFEST.md                       # Project manifest
├── data/
│   ├── vignettes.json                # 20 vignettes with ground truth
│   └── vignettes_tennessee.json      # 120 Tennessee vignettes, 24 ethnic groups
├── skill/
│   └── SKILL.md                      # NurtureNet CHW Skill
├── .streamlit/
│   └── config.toml                   # Light theme configuration
├── results/
├── requirements.txt
└── README.md
```

---

## Roadmap

### Near-term (0-3 months)

**WhatsApp interface** — CHW texts patient info in plain language, NurtureNet responds in 30 seconds. Works on any phone, no app install, 2G cellular. Twilio WhatsApp API. In active development.

**Bluetooth blood pressure integration** — FDA-cleared Bluetooth cuffs (iHealth Track, Omron) connect directly to the CHW phone. BP readings populate automatically at the moment of measurement. No manual entry, no transcription error, timestamped sensor data. One integration away from deployment.

**Tennessee evaluation** — Full evaluation across all 120 vignettes spanning 24 ethnic groups, including Nashville refugee communities (Somali, Kurdish, Burmese/Karen, Congolese, Guatemalan, Yemeni, Iraqi), Appalachian rural White families, and Cherokee Nation members in East Tennessee.

**QLoRA fine-tuning** — Adapt Phi-4-mini on clinical vignettes using MLX on MacBook Air M-series. 30-60 minutes of local training. Re-evaluate equity metrics before and after.

### Medium-term (3-6 months)

**Nashville community partnerships** — Direct outreach to CHW programs already serving Tennessee communities:
- Matthew Walker Comprehensive Health Center (North Nashville, FQHC)
- Siloam Family Health (immigrant and refugee communities)
- Conexión Américas (Latino community, Nolensville Road corridor)
- Somali Community Center of Tennessee
- Nashville General Hospital community health team
- Tennessee Justice Center (TennCare advocacy)

The entry point is not marketing — it is relationship. One CHW who finds NurtureNet useful tells five others.

**VUMC retrospective validation** — Vanderbilt CTSA IRB pathway for retrospective cohort study. Apply NurtureNet Care Access Index to de-identified VUMC obstetric patients and measure whether high scores predict missed or delayed preeclampsia detection.

**CMS UDS SDOH integration** — Federally Qualified Health Centers already collect standardized SDOH screening data under CMS Uniform Data System mandates. Integrate real measured social context data.

### Long-term (6-18 months)

**Continuous monitoring** — Patient keeps a Bluetooth cuff at home. Readings push to NurtureNet automatically between visits. If BP crosses 140/90, NurtureNet alerts the CHW immediately without anyone initiating anything. This is the product that saves lives between appointments.

**Meharry Medical College collaboration** — Meharry has a 150-year history in Black health equity and is two miles from Vanderbilt. A Vanderbilt/Meharry partnership combines research infrastructure with deep community roots.

**AMIA 2027 submission** — Submit findings to the American Medical Informatics Association annual symposium. The equity finding is publishable. Target: student paper competition.

**NIH SBIR Phase 1** — Up to $314,000 for clinical validation and deployment infrastructure.

**Gates Foundation Grand Challenges** — The Gates maternal health portfolio funds community-based AI tools for low-resource settings. The CHW model built for rural Tennessee translates directly to global maternal health contexts.

---

## References

**Course papers:**
- Phuong, M. & Hutter, M. (2022). Formal Algorithms for Transformers. arXiv:2207.09238
- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361

**Maternal mortality and preeclampsia:**
- Hoyert, D.L. (2026). Maternal Mortality Rates in the United States, 2024. NCHS Health E-Stats No. 113. doi:10.15620/cdc/174651
- Trost, S.L., Beauregard, J., Njie, F., et al. (2022). Pregnancy-Related Deaths: Data from Maternal Mortality Review Committees in 36 US States, 2017-2019. CDC.
- Petersen, E.E., Davis, N.L., Goodman, D., et al. (2019). Racial/Ethnic Disparities in Pregnancy-Related Deaths. MMWR 68(35):762-765. doi:10.15585/mmwr.mm6835a3
- MacDorman, M.F., Thoma, M., Declercq, E., & Howell, E.A. (2021). Racial and Ethnic Disparities in Maternal Mortality in the United States Using Enhanced Vital Records, 2016-2017. AJPH 111(9):1673-1681. doi:10.2105/AJPH.2021.306375
- Ford, N.D., Cox, S., Ko, J.Y., et al. (2022). Hypertensive Disorders in Pregnancy and Mortality at Delivery Hospitalization. MMWR 71(17):585-591. doi:10.15585/mmwr.mm7117a1
- Fink, D.A., Kilday, D., Cao, Z., et al. (2023). Trends in Maternal Mortality and Severe Maternal Morbidity During Delivery-Related Hospitalizations. JAMA Netw Open 6(6):e2317641. doi:10.1001/jamanetworkopen.2023.17641
- Guerby, P., Audibert, F., Johnson, J.-A., et al. (2024). Prospective Validation of First-Trimester Screening for Preterm Preeclampsia in Nulliparous Women (PREDICTION Study). Hypertension 81(7):1574-1582. doi:10.1161/HYPERTENSIONAHA.123.22584
- American College of Obstetricians and Gynecologists (2020). Gestational Hypertension and Preeclampsia. ACOG Practice Bulletin No. 222. Obstet Gynecol 135:e237-e260.
- March of Dimes (2024). Nowhere to Go: Maternity Care Deserts Across the US.
- Tennessee Lookout (2025). TennCare's maternal death rates are 3x those of private insurance. tennesseelookout.com/2025/01/21

**Care Access Index sources:**
- Laraia, B.A., Siega-Riz, A.M., & Gundersen, C. (2010). Household Food Insecurity Is Associated with Self-Reported Pregravid Weight Status, Gestational Weight Gain, and Pregnancy Complications. J Am Diet Assoc 110(5):692-701.
- Leifheit, K.M., Schwartz, G.L., Pollack, C.E., et al. (2020). Severe Housing Insecurity During Pregnancy: Associations with Adverse Birth and Infant Outcomes. Int J Environ Res Public Health 17(22):8659.
- Kaiser Family Foundation (2024). Health Coverage by Race and Ethnicity, 2010-2022.

**Edge AI:**
- Wang, X., Dang, T., Zhang, X., Kostakos, V., Witbrock, M., & Jia, H. (2025). HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring. arXiv:2509.07260

**Responsible AI:**
- Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations. Science 366(6464):447-453.
- Bai, Y., Kadavath, S., Kundu, S., et al. / Anthropic (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073

---

*NurtureNet is a research prototype for DS 5690 at Vanderbilt University. Not FDA cleared. Not a substitute for clinical judgment. All outputs advisory only.*

*Author: Mary Morkos, MS Data Science, Vanderbilt University, May 2026*
