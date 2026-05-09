import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv
from modules.normalize import normalize
from modules.inference_engine import run_inference

load_dotenv()


def apply_overrides(result: dict, overrides: dict) -> dict:
    updated = result.copy()
    if overrides:
        for k, v in overrides.items():
            updated[k] = v
        updated["code_status"] = "manual_review"
    return updated


client  = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_ID = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Terminology maps
# ---------------------------------------------------------------------------

LOINC_VITALS = {
    "systolic_bp":      "8480-6",
    "diastolic_bp":     "8462-4",
    "temperature_f":    "8310-5",
    "temperature_c":    "8310-5",
    "pulse":            "8867-4",
    "spo2":             "59408-5",
    "respiratory_rate": "9279-1",
    "weight_kg":        "29463-7",
    "blood_glucose":    "2339-0",
}

SNOMED_CONDITIONS = {
    "cough":                      "49727002",
    "productive cough":           "28743005",
    "shortness of breath":        "267036007",
    "difficulty breathing":       "267036007",
    "wheezing":                   "56018004",
    "copd":                       "13645005",
    "copd exacerbation":          "195951007",
    "pneumonia":                  "233604007",
    "tuberculosis":               "56717001",
    "tb":                         "56717001",
    "fever":                      "386661006",
    "high fever":                 "386661006",
    "malaria":                    "61462000",
    "typhoid":                    "4834000",
    "dengue":                     "38362002",
    "cholera":                    "63650001",
    "chest pain":                 "29857009",
    "hypertension":               "38341003",
    "palpitations":               "80313002",
    "headache":                   "25064002",
    "altered consciousness":      "419284004",
    "seizure":                    "91175000",
    "dizziness":                  "404640003",
    "diarrhea":                   "62315008",
    "vomiting":                   "422400008",
    "abdominal pain":             "21522001",
    "dehydration":                "34095006",
    "malnutrition":               "76113001",
    "severe acute malnutrition":  "76113001",
    "moderate acute malnutrition":"76113001",
    "anemia":                     "271737000",
    "pregnancy complication":     "609496007",
    "wound infection":            "76844004",
    "laceration":                 "262531003",
    "rash":                       "271807003",
    "petechiae":                  "423902002",
    "jaundice":                   "18165001",
    "pallor":                     "267093001",
    "hyperglycemia":              "80394007",
    "hypoglycemia":               "302866003",
    "hypoxia":                    "389086002",
    "severe hypoxia":             "389086002",
    "joint pain":                 "57676002",
    "back pain":                  "161891005",
    "bacterial infection":        "87628006",
    "hiv reactive":               "86406008",
    "diabetes mellitus type 2":   "44054006",
    "diabetes mellitus type 1":   "46635009",
    "asthma":                     "195967001",
    "heart failure":              "84114007",
    "bilateral pitting edema":    "60046008",
    "facial edema":               "217372002",
    "ankle edema":                "248491001",
    "intestinal parasites":       "47826004",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a medical triage assistant for community health workers (CHWs) in low- and middle-income countries.
Input may be in any language or a mix of languages including Hindi, Swahili, Tagalog, Amharic, Vietnamese, Arabic, and English.
Medical terms often appear in English even inside non-English sentences.

Output MUST be a valid JSON object with EXACTLY these fields:

{
  "symptoms": ["list of symptoms explicitly stated or clearly described — in English"],
  "vitals": {
    "systolic_bp": null or number,
    "diastolic_bp": null or number,
    "temperature_f": null or number,
    "temperature_c": null or number,
    "pulse": null or number,
    "spo2": null or number,
    "respiratory_rate": null or number,
    "weight_kg": null or number,
    "blood_glucose": null or number
  },
  "duration": null or string describing symptom duration in English,
  "severity_score": number between 0.0 and 1.0,
  "referral_flag": true or false,
  "evidence_cited": "the exact phrase or value from the input that determined severity",
  "language_detected": "primary language detected",
  "language_notes": null or "non-English terms translated",
  "code_status": "auto"
}

Severity rules:
RED FLAG → severity_score >= 0.80, referral_flag true:
  Systolic BP > 160, temperature > 103F / 39.4C, SpO2 < 90%, chest pain,
  breathing difficulty, altered consciousness, seizure, severe dehydration,
  symptoms > 3 weeks with no improvement, suspected TB, suspected cholera

MODERATE → severity_score 0.40–0.79, referral_flag false (unless duration rule fires):
  Fever 38–39.4C, persistent cough < 3 weeks, moderate pain,
  diarrhea without severe dehydration, wound without systemic infection

LOW → severity_score < 0.40, referral_flag false:
  Mild self-limiting symptoms, minor wounds, no red flag vitals

Temperature: if Celsius, convert to Fahrenheit (F = C × 9/5 + 32). Store both.
Only extract vitals explicitly stated. Never guess missing values.

Few-shot examples:

Input: "Mera sir dukh raha hai aur BP 160/100 hai, 2 din se"
Output: {"symptoms": ["headache"], "vitals": {"systolic_bp": 160, "diastolic_bp": 100, "temperature_f": null, "temperature_c": null, "pulse": null, "spo2": null, "respiratory_rate": null, "weight_kg": null, "blood_glucose": null}, "duration": "2 days", "severity_score": 0.85, "referral_flag": true, "evidence_cited": "BP 160/100 exceeds red flag threshold", "language_detected": "hindi", "language_notes": "sir dukh raha hai = headache", "code_status": "auto"}

Input: "Amina, miaka 28. Homa kali, joto 39C. Kikohozi wiki mbili."
Output: {"symptoms": ["fever", "cough"], "vitals": {"systolic_bp": null, "diastolic_bp": null, "temperature_f": 102.2, "temperature_c": 39.0, "pulse": null, "spo2": null, "respiratory_rate": null, "weight_kg": null, "blood_glucose": null}, "duration": "2 weeks", "severity_score": 0.55, "referral_flag": false, "evidence_cited": "Fever 39C, cough 2 weeks — moderate, no red flag yet", "language_detected": "swahili", "language_notes": "homa kali = high fever, joto = temperature, kikohozi = cough, wiki mbili = two weeks", "code_status": "auto"}

Input: "Maria, 34 taon. Ubo ng tatlong linggo, may plema."
Output: {"symptoms": ["productive cough"], "vitals": {"systolic_bp": null, "diastolic_bp": null, "temperature_f": null, "temperature_c": null, "pulse": null, "spo2": null, "respiratory_rate": null, "weight_kg": null, "blood_glucose": null}, "duration": "3 weeks", "severity_score": 0.82, "referral_flag": true, "evidence_cited": "Productive cough for 3 weeks — TB screening required per duration rule", "language_detected": "tagalog", "language_notes": "ubo = cough, tatlong linggo = three weeks, may plema = with phlegm", "code_status": "auto"}

Input: "Patient has fever 102 and cough for 3 days"
Output: {"symptoms": ["fever", "cough"], "vitals": {"systolic_bp": null, "diastolic_bp": null, "temperature_f": 102, "temperature_c": null, "pulse": null, "spo2": null, "respiratory_rate": null, "weight_kg": null, "blood_glucose": null}, "duration": "3 days", "severity_score": 0.45, "referral_flag": false, "evidence_cited": "Fever 102F, no red flag vitals", "language_detected": "english", "language_notes": null, "code_status": "auto"}

Input: "فاطمة، 30 سنة. حمى شديدة 40 درجة، سعال مستمر أسبوعين."
Output: {"symptoms": ["fever", "cough"], "vitals": {"systolic_bp": null, "diastolic_bp": null, "temperature_f": 104.0, "temperature_c": 40.0, "pulse": null, "spo2": null, "respiratory_rate": null, "weight_kg": null, "blood_glucose": null}, "duration": "2 weeks", "severity_score": 0.88, "referral_flag": true, "evidence_cited": "Fever 40C / 104F exceeds red flag threshold", "language_detected": "arabic", "language_notes": "حمى شديدة = high fever, سعال = cough, أسبوعين = two weeks", "code_status": "auto"}"""

# ---------------------------------------------------------------------------
# History upgrade — deterministic, auditable
# ---------------------------------------------------------------------------

UPGRADE_CONDITIONS = [
    ("copd",         ["cough", "shortness of breath", "difficulty breathing", "wheezing"]),
    ("tuberculosis", ["cough", "productive cough"]),
    ("tb",           ["cough", "productive cough"]),
    ("hypertension", ["headache", "dizziness", "chest pain"]),
    ("malaria",      ["fever", "high fever"]),
    ("anemia",       ["shortness of breath", "dizziness", "fatigue"]),
    ("diabetes",     ["wound infection", "dehydration"]),
    ("asthma",       ["cough", "wheezing", "shortness of breath"]),
]

UPGRADE_BUMPS = {
    "copd":         0.35,
    "tuberculosis": 0.35,
    "tb":           0.35,
    "hypertension": 0.25,
    "malaria":      0.20,
    "anemia":       0.20,
    "diabetes":     0.25,
    "asthma":       0.25,
}


def _apply_history_upgrades(result: dict, conditions_lower: list) -> dict:
    symptoms_lower   = [s.lower() for s in result.get("symptoms", [])]
    upgrades_applied = []
    total_bump       = 0.0

    for condition_key, trigger_symptoms in UPGRADE_CONDITIONS:
        if any(condition_key in c for c in conditions_lower):
            if any(s in symptoms_lower for s in trigger_symptoms):
                upgrades_applied.append(condition_key)
                total_bump += UPGRADE_BUMPS.get(condition_key, 0.20)

    if upgrades_applied:
        old_score = result.get("severity_score", 0.0)
        new_score = min(round(old_score + total_bump, 2), 1.0)
        result["severity_score"]          = new_score
        result["referral_flag"]           = True
        result["evidence_cited"]          = (
            f"History upgrade: {', '.join(u.upper() for u in upgrades_applied)} "
            f"matched current symptoms. Score {old_score} → {new_score} "
            f"(+{round(total_bump, 2)})."
        )
        result["history_upgrades_applied"] = upgrades_applied
        result["severity_bump_applied"]    = round(total_bump, 2)

    return result


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

async def triage_extractor(
    raw_text: str,
    patient_id: str,
    patient_history: dict = None,
    chw_id: str = None
) -> dict:
    history_context = ""
    if patient_history:
        conditions = patient_history.get("conditions", [])
        if conditions:
            history_context = f"\nPatient history from FHIR: {', '.join(conditions)}."

    try:
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Extract from: \"{raw_text}\"{history_context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)

        if patient_history:
            conditions_lower = [c.lower() for c in patient_history.get("conditions", [])]
            if conditions_lower:
                result = _apply_history_upgrades(result, conditions_lower)

        vitals = result.get("vitals", {}) or {}
        result["loinc_map"] = {
            LOINC_VITALS[k]: v
            for k, v in vitals.items()
            if v is not None and k in LOINC_VITALS
        }
        result["snomed_map"] = {
            s.lower(): SNOMED_CONDITIONS[s.lower()]
            for s in result.get("symptoms", [])
            if s.lower() in SNOMED_CONDITIONS
        }

        result["patient_id"] = patient_id
        if chw_id:
            result["chw_id"] = chw_id

        result = normalize(result)
        result = run_inference(result)
        result = apply_overrides(result, None)

        return result

    except json.JSONDecodeError as e:
        return {
            "error":      "JSON parse failed — LLM returned malformed output",
            "details":    str(e),
            "patient_id": patient_id,
            "chw_id":     chw_id,
            "code_status":"manual_review"
        }
    except Exception as e:
        return {
            "error":      "Triage extraction failed",
            "details":    str(e),
            "patient_id": patient_id,
            "chw_id":     chw_id,
            "code_status":"manual_review"
        }