"""
evaluate.py — NurtureNet 2.0 CHW evaluation harness
=====================================================
Runs patient vignettes through:
  Layer 1: Phi-4-mini local (via Ollama) — chain-of-thought triage
  Layer 2: Claude constitutional review — equity-aware correction

Measures:
  - High-risk recall by race (Layer 1 alone vs after review)
  - Constitutional violations caught
  - Risk upgrades/downgrades by the review layer
  - SDOH sensitivity (with vs without social context)

Usage:
    python evaluate.py                    # full evaluation
    python evaluate.py --case 1           # single case demo
    python evaluate.py --no-review        # local model only
"""

import json
import requests
import anthropic
import os
import argparse
from datetime import datetime


client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
OLLAMA_URL = "http://localhost:11434/api/generate"


# ── Prompt builders ───────────────────────────────────────────────

def build_chw_prompt(v, include_sdoh=False):
    symptoms = ", ".join(v["symptoms"]) if v["symptoms"] else "none reported"
    prompt = f"""You are a clinical triage AI supporting a community health worker during a home visit.
Reason through this patient case step by step before giving your assessment.

PATIENT VITALS:
- Gestational age: {v['gestational_age_weeks']} weeks
- Blood pressure: {v['systolic_bp']}/{v['diastolic_bp']} mmHg
- Heart rate: {v['heart_rate']} bpm
- Symptoms reported: {symptoms}
- Prior preeclampsia: {v['prior_preeclampsia']}
"""
    if include_sdoh and "sdoh" in v:
        s = v["sdoh"]
        prompt += f"""
SOCIAL CONTEXT (collected by CHW):
- Race/ethnicity: {s['race_ethnicity']}
- Insurance: {s['insurance']}
- Food insecure: {s['food_insecure']}
- Rural location: {s['rural']}
- Housing instability: {s['housing_instability']}
- Late or no prenatal care: {s['late_prenatal_care']}
- SDOH burden score: {s['sdoh_burden']}/14
"""

    prompt += """
ACOG THRESHOLDS:
- BP >= 140/90 = hypertension (monitor)
- BP >= 160/110 = severe range (escalate now)
- Severe features: severe headache, visual changes, epigastric pain, facial edema

Reason through this case step by step:

Step 1: Evaluate blood pressure against ACOG thresholds
Step 2: Identify any severe features in symptoms
Step 3: Consider prior obstetric history
Step 4: If social context provided, assess how SDOH amplifies clinical risk
Step 5: Synthesize into overall risk level with confidence

After your reasoning, provide your assessment as JSON:
{
  "risk_level": "low" | "moderate" | "high",
  "confidence": 0.0-1.0,
  "reasoning_summary": "one sentence explaining your assessment",
  "key_factors": ["factor1", "factor2", "factor3"],
  "chw_action": "specific next action for the community health worker",
  "constitutional_concerns": []
}"""
    return prompt


# ── Layer 1: Local model ──────────────────────────────────────────

def run_phi4(prompt, verbose=False):
    if verbose:
        print("    [Phi-4-mini reasoning...]")
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "phi4-mini", "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f'{{"risk_level": "unknown", "confidence": 0.0, "error": "{str(e)}"}}'


def parse_json_from_text(raw):
    try:
        start = raw.rfind("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except:
        pass
    return {
        "risk_level": "unknown",
        "confidence": 0.0,
        "reasoning_summary": "parse error",
        "key_factors": [],
        "chw_action": "Unable to parse — CHW should use clinical judgment",
        "constitutional_concerns": []
    }


# ── Layer 2: Claude constitutional review ─────────────────────────

SKILL_CONTENT = open("skill/SKILL.md").read()

CONSTITUTIONAL_PRINCIPLES = """
NURTURENET CONSTITUTIONAL PRINCIPLES FOR CHW TRIAGE:
1. NEVER assess as low risk when BP >= 140/90 mmHg
2. Prior preeclampsia + ANY BP elevation = minimum moderate risk, always
3. SDOH burden >= 7 must be explicitly flagged in CHW guidance
4. Non-Hispanic Black patients: heightened vigilance at ALL thresholds (3.15x mortality, Hoyert 2026)
5. Confidence < 0.7 on moderate/high assessment = escalate, never reassure
6. Every output must give the CHW a specific action with a timeline
7. Never use language that could discourage the patient from seeking care
"""


def run_constitutional_review(vignette, local_output, local_reasoning, verbose=False):
    if verbose:
        print("    [Claude constitutional review...]")

    with open("skill/SKILL.md") as f:
        skill = f.read()

    prompt = f"""You are the NurtureNet constitutional review layer for community health workers.

A local triage model assessed this patient and produced the following:

LOCAL MODEL ASSESSMENT:
{json.dumps(local_output, indent=2)}

LOCAL MODEL FULL REASONING:
{local_reasoning[:3000]}

PATIENT DATA:
{json.dumps(vignette, indent=2)}

NURTURENET CHW SKILL:
{skill[:2000]}

{CONSTITUTIONAL_PRINCIPLES}

Review the local model assessment against each constitutional principle.
Identify any violations. Correct the assessment if needed.
Generate CHW-appropriate output.

Return ONLY valid JSON — no markdown, no preamble:
{{
  "final_risk_level": "low" | "moderate" | "high",
  "risk_changed": true | false,
  "change_direction": "upgraded" | "downgraded" | "unchanged",
  "principles_violated": [1, 2, 3],
  "violations_explained": ["principle 1 violated because...", "..."],
  "sdoh_burden_score": 0,
  "sdoh_burden_level": "Low" | "Moderate" | "High" | "Critical",
  "equity_flag": true | false,
  "equity_note": "note if flagged, null otherwise",
  "chw_action": "specific action with timeline",
  "what_to_say": "exact plain-language script CHW reads to patient",
  "clinician_handoff": "2-3 sentence clinical summary if escalating"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Find outermost JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
        else:
            raise ValueError(f"No JSON object found in response: {raw[:200]}")

        # Ensure required fields exist
        result.setdefault("final_risk_level", local_output.get("risk_level", "unknown"))
        result.setdefault("risk_changed", False)
        result.setdefault("change_direction", "unchanged")
        result.setdefault("principles_violated", [])
        result.setdefault("equity_flag", False)
        result.setdefault("equity_note", None)
        result.setdefault("chw_action", "Follow up with care team within 24 hours.")
        result.setdefault("what_to_say", "Thank you for letting me check in on you today.")
        return result

    except Exception as e:
        print(f"    [Constitutional review error: {e}]")
        # Run a simpler fallback prompt
        try:
            fallback = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[{"role": "user", "content": f"""Patient: BP {vignette['systolic_bp']}/{vignette['diastolic_bp']}, 
race: {vignette.get('sdoh',{}).get('race_ethnicity','unknown')}, 
SDOH burden: {vignette.get('sdoh',{}).get('sdoh_burden',0)}/14,
prior preeclampsia: {vignette.get('prior_preeclampsia', False)},
local model said: {local_output.get('risk_level','unknown')}.

Apply ACOG criteria and equity principles. Return JSON only:
{{"final_risk_level": "low|moderate|high", "risk_changed": false, "change_direction": "unchanged", 
"principles_violated": [], "equity_flag": false, "equity_note": null,
"chw_action": "action here", "what_to_say": "script here"}}"""}]
            )
            raw2 = fallback.content[0].text.strip().replace("```json","").replace("```","").strip()
            start = raw2.find("{"); end = raw2.rfind("}") + 1
            return json.loads(raw2[start:end])
        except Exception as e2:
            return {
                "error": str(e2),
                "final_risk_level": local_output.get("risk_level", "unknown"),
                "risk_changed": False,
                "change_direction": "unchanged",
                "principles_violated": [],
                "equity_flag": False,
                "equity_note": None,
                "chw_action": "Unable to complete review — use clinical judgment.",
                "what_to_say": "I want to make sure you are doing well. Please contact your care team."
            }


# ── Single case runner ────────────────────────────────────────────

def run_case(v, include_sdoh=True, run_review=True, verbose=True):
    case_id = v["id"]
    race = v.get("sdoh", {}).get("race_ethnicity", "Not recorded")
    burden = v.get("sdoh", {}).get("sdoh_burden", 0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Case {case_id} | GA {v['gestational_age_weeks']}w | "
              f"BP {v['systolic_bp']}/{v['diastolic_bp']}")
        print(f"  Race: {race} | SDOH burden: {burden}/14")
        print(f"  Ground truth: {v['ground_truth'].upper()}")
        print(f"{'='*60}")

    # Layer 1 — no SDOH
    prompt_no_sdoh = build_chw_prompt(v, include_sdoh=False)
    raw_no_sdoh = run_phi4(prompt_no_sdoh, verbose=verbose)
    out_no_sdoh = parse_json_from_text(raw_no_sdoh)

    # Layer 1 — with SDOH
    prompt_sdoh = build_chw_prompt(v, include_sdoh=True)
    raw_sdoh = run_phi4(prompt_sdoh, verbose=verbose)
    out_sdoh = parse_json_from_text(raw_sdoh)

    if verbose:
        print(f"\n  Phi-4-mini (no SDOH): {out_no_sdoh['risk_level'].upper()} "
              f"[confidence: {out_no_sdoh.get('confidence', 0):.0%}]")
        print(f"  Phi-4-mini (w/ SDOH): {out_sdoh['risk_level'].upper()} "
              f"[confidence: {out_sdoh.get('confidence', 0):.0%}]")

    review = None
    if run_review:
        review = run_constitutional_review(v, out_sdoh, raw_sdoh, verbose=verbose)

        if verbose:
            final = review.get("final_risk_level", "unknown").upper()
            changed = review.get("risk_changed", False)
            direction = review.get("change_direction", "unchanged")
            violations = review.get("principles_violated", [])
            equity = review.get("equity_flag", False)

            print(f"  After review:         {final} "
                  f"[{'CHANGED — ' + direction if changed else 'unchanged'}]")
            if violations:
                print(f"  Constitutional violations: principles {violations}")
            if equity:
                print(f"  EQUITY FLAG: {review.get('equity_note', '')}")
            print(f"\n  CHW action: {review.get('chw_action', '')}")
            print(f"\n  What to say: \"{review.get('what_to_say', '')}\"")

    return {
        "case_id": case_id,
        "ground_truth": v["ground_truth"],
        "race_ethnicity": race,
        "sdoh_burden": burden,
        "local_no_sdoh": out_no_sdoh["risk_level"],
        "local_confidence_no_sdoh": out_no_sdoh.get("confidence", 0),
        "local_with_sdoh": out_sdoh["risk_level"],
        "local_confidence_sdoh": out_sdoh.get("confidence", 0),
        "after_review": review.get("final_risk_level") if review else None,
        "risk_changed": review.get("risk_changed") if review else None,
        "change_direction": review.get("change_direction") if review else None,
        "principles_violated": review.get("principles_violated", []) if review else [],
        "equity_flag": review.get("equity_flag") if review else None,
        "chw_action": review.get("chw_action") if review else None,
    }


# ── Metrics ───────────────────────────────────────────────────────

def compute_metrics(results):
    print("\n" + "="*60)
    print("  NURTURENET 2.0 — EVALUATION RESULTS")
    print("="*60)

    high_cases = [r for r in results if r["ground_truth"] == "high"]

    def recall(field, cases):
        if not cases:
            return 0
        return sum(1 for r in cases if r.get(field) == "high") / len(cases)

    print(f"\nOverall high-risk recall (n={len(high_cases)} high-risk cases):")
    print(f"  Phi-4-mini — no SDOH : {recall('local_no_sdoh', high_cases):.0%}")
    print(f"  Phi-4-mini — w/ SDOH : {recall('local_with_sdoh', high_cases):.0%}")
    print(f"  After constitutional  : {recall('after_review', high_cases):.0%}")

    races = [
        "Non-Hispanic Black",
        "Hispanic or Latina",
        "Non-Hispanic White",
        "Asian or Pacific Islander",
        "American Indian or Alaska Native"
    ]

    print(f"\nHigh-risk recall by race/ethnicity:")
    for race in races:
        race_high = [r for r in high_cases if r["race_ethnicity"] == race]
        if not race_high:
            continue
        r1 = recall("local_no_sdoh", race_high)
        r2 = recall("local_with_sdoh", race_high)
        r3 = recall("after_review", race_high)
        gap = r3 - r1
        print(f"  {race:<35} "
              f"no_sdoh={r1:.0%}  "
              f"w_sdoh={r2:.0%}  "
              f"reviewed={r3:.0%}  "
              f"gap={gap:+.0%}  (n={len(race_high)})")

    violations_cases = [r for r in results if r.get("principles_violated")]
    print(f"\nConstitutional violations caught: "
          f"{len(violations_cases)}/{len(results)} cases")
    for r in violations_cases:
        print(f"  Case {r['case_id']:>2} "
              f"({r['race_ethnicity']:<35}): "
              f"principles {r['principles_violated']}")

    changed = [r for r in results if r.get("risk_changed")]
    upgraded = [r for r in changed if r.get("change_direction") == "upgraded"]
    equity_flagged = [r for r in results if r.get("equity_flag")]
    print(f"\nRisk assessments changed by review: {len(changed)}/{len(results)}")
    print(f"  Upgraded (safer for patient): {len(upgraded)}")
    print(f"Equity flags raised: {len(equity_flagged)}")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NurtureNet 2.0 CHW Evaluation")
    parser.add_argument("--case", type=int, help="Run single case by ID")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip constitutional review (local model only)")
    args = parser.parse_args()

    with open("data/vignettes.json") as f:
        vignettes = json.load(f)

    if args.case:
        v = next((v for v in vignettes if v["id"] == args.case), None)
        if not v:
            print(f"Case {args.case} not found")
            return
        run_case(v, run_review=not args.no_review, verbose=True)
        return

    print("\nNurtureNet 2.0 — Full Evaluation")
    print(f"Running {len(vignettes)} cases...")
    print("(This will take a few minutes — each case runs through "
          "Phi-4-mini then Claude)\n")

    results = []
    for v in vignettes:
        result = run_case(v, run_review=not args.no_review, verbose=True)
        results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/evaluation_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    compute_metrics(results)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
