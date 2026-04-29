import os
import json
import base64
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Import existing mappings from the triage module
try:
    from modules.triage import LOINC_VITALS, SNOMED_CONDITIONS
except ImportError:
    # Fallback/Mock for standalone testing
    LOINC_VITALS = {}
    SNOMED_CONDITIONS = {}

# ---------------------------------------------------------------------------
# Vision system prompt
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are a clinical image interpreter for community health workers in low-resource settings.
Analyze the image and extract any visible clinical information.

You MUST output a valid JSON object with EXACTLY these fields:

{
  "symptoms": ["list of symptoms or findings in English"],
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
  "duration": null,
  "severity_score": number between 0.0 and 1.0,
  "referral_flag": true or false,
  "evidence_cited": "what you saw in the image that determined severity",
  "language_detected": "image",
  "language_notes": null or "description of what type of image this is",
  "code_status": "auto",
  "image_type": "one of: malaria_rdt | thermometer | wound | health_record | lab_result | glucose_meter | unknown"
}

Image interpretation rules:

MALARIA RDT STRIP:
- One line at C only → negative → symptoms: [], severity_score 0.2, referral_flag false
- Two lines (C and T) → positive → symptoms: ["malaria"], severity_score 0.85, referral_flag true
- No lines → invalid → severity_score 0.5, referral_flag true, evidence_cited: "Invalid RDT — retest required"

THERMOMETER:
- Read displayed value precisely. Store in both temperature_f and temperature_c.
- >39.4C or >103F → referral_flag true, severity_score >= 0.80
- 38.0-39.4C → severity_score 0.50, referral_flag false

WOUND / SKIN:
- Redness spreading beyond wound, yellow/green discharge, red streaks → symptoms: ["wound infection"], severity_score >= 0.75, referral_flag true
- Clean minor laceration → symptoms: ["laceration"], severity_score 0.25, referral_flag false

GLUCOSE METER:
- Read displayed value. Store in blood_glucose field.
- >200 mg/dL or >11.1 mmol/L → symptoms: ["hyperglycemia"], severity_score 0.75, referral_flag true
- <70 mg/dL or <3.9 mmol/L → symptoms: ["hypoglycemia"], severity_score 0.85, referral_flag true

GENERAL RULES:
- If image is unclear or unrelated to clinical data → severity_score 0.0, referral_flag false, image_type: "unknown"
- Never diagnose. Only describe what is visually present.
- Never infer values not visible in the image."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_image_as_base64(url: str) -> tuple[str, str]:
    """Fetch an image from a URL, return (base64_string, mime_type)."""
    headers = {"User-Agent": "GroundWork-MCP/1.0 (community health research)"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
        r = await http.get(url, headers=headers)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    b64 = base64.b64encode(r.content).decode()
    return b64, mime

def _image_block(
    image_base64: str = None,
    image_mime: str = None,
    image_url: str = None
) -> dict:
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}
    }

def _add_terminology_maps(result: dict) -> dict:
    """Safely map extracted findings to clinical standards."""
    vitals = result.get("vitals", {}) or {}
    
    # Map LOINC codes only if the key exists and value is present
    result["loinc_map"] = {
        LOINC_VITALS[k]: v
        for k, v in vitals.items()
        if v is not None and k in LOINC_VITALS
    }
    
    # Map SNOMED codes for symptoms found in the image
    symptoms = result.get("symptoms", []) or []
    result["snomed_map"] = {
        s.lower(): SNOMED_CONDITIONS[s.lower()]
        for s in symptoms
        if s.lower() in SNOMED_CONDITIONS
    }
    
    return result

# ---------------------------------------------------------------------------
# Main Vision Triage Function
# ---------------------------------------------------------------------------

async def analyze_clinical_image(
    patient_id: str,
    chw_id: str = None,
    context_hint: str = None,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    manual_history: str = None
) -> dict:
    """
    Analyze clinical image and return structured triage data.
    Output is schema-aligned with text triage for downstream processing.
    """
    if not image_base64 and not image_url:
        return {
            "error": "Provide either image_base64 or image_url",
            "patient_id": patient_id,
            "code_status": "manual_review"
        }

    user_text = "Analyze this clinical image and extract all visible health data."

    if manual_history:
        user_text += f" IMPORTANT PATIENT HISTORY: {manual_history}."
    if context_hint:
        user_text += f" This appears to be: {context_hint}."

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        _image_block(image_base64, image_mime, image_url),
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1024
        )

        # Parse and enrich with terminology mappings
        result = json.loads(response.choices[0].message.content)
        result = _add_terminology_maps(result)
        
        # Meta-data for downstream pipeline
        result["patient_id"] = patient_id
        result["input_type"] = "image"
        if chw_id:
            result["chw_id"] = chw_id

        return result

    except json.JSONDecodeError as e:
        return {
            "error": "Vision model returned malformed JSON",
            "details": str(e),
            "patient_id": patient_id,
            "code_status": "manual_review"
        }
    except Exception as e:
        return {
            "error": "Vision triage failed",
            "details": str(e),
            "patient_id": patient_id,
            "code_status": "manual_review"
        }