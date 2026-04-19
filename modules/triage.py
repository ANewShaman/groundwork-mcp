import os
import json
from groq import Groq
from dotenv import load_dotenv

# 1. Load the variables IMMEDIATELY
load_dotenv()

# 2. Now initialize the client (it will find the key now)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_ID = "llama-3.3-70b-versatile"

LOINC_VITALS = {
    "systolic_bp":      "8480-6",
    "diastolic_bp":     "8462-4",
    "temperature_f":    "8310-5",
    "pulse":            "8867-4"
}

SNOMED_DEMO = {
    "copd": "13645005",
    "copd exacerbation": "195951007"
}

SYSTEM_PROMPT = """You are a medical triage assistant for community health workers in India.
Input is mixed Hindi-English (Hinglish) clinical notes. Output MUST be a valid JSON object.

Severity rules:
- Red flags (Score > 0.8): BP > 160, Temp > 103F, chest pain, breathing difficulty.
- Moderate (0.4 - 0.7): fever, persistent cough.
- Low (< 0.4): mild symptoms."""

async def triage_extractor(raw_text: str, patient_id: str, patient_history: dict = None) -> dict:
    history_context = ""
    if patient_history:
        conditions = patient_history.get("conditions", [])
        if conditions:
            history_context = f"\nPatient history: {', '.join(conditions)}."

    try:
        # Groq specific call
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract from: \"{raw_text}\"{history_context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        # Groq uses .message.content, not .parsed
        result = json.loads(response.choices[0].message.content)

        # Add LOINC codes
        vitals = result.get("vitals", {}) or {}
        result["loinc_map"] = {
            LOINC_VITALS[k]: v
            for k, v in vitals.items()
            if v is not None and k in LOINC_VITALS
        }

        # Add SNOMED mapping
        result["snomed_map"] = {
            s.lower(): SNOMED_DEMO[s.lower()] 
            for s in result.get("symptoms", []) 
            if s.lower() in SNOMED_DEMO
        }

        result["patient_id"] = patient_id
        return result

    except Exception as e:
        return {
            "error": "Triage extraction failed",
            "details": str(e),
            "patient_id": patient_id,
            "code_status": "manual_review"
        }