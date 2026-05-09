import os
import json
import base64
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

try:
    from modules.triage import LOINC_VITALS, SNOMED_CONDITIONS
except ImportError:
    LOINC_VITALS = {}
    SNOMED_CONDITIONS = {}

# ---------------------------------------------------------------------------
# VISION SYSTEM PROMPT — Llama-4, FHIR-compliant, CHW-optimised
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are a clinical image observer for Community Health Workers in LMICs.

Your job is to describe what is visually present in the image.
DO NOT diagnose diseases.
DO NOT infer conditions from medications or test results.

═══════════════════════════════════════════
OUTPUT SCHEMA — return this exact structure
═══════════════════════════════════════════
{
  "image_type": "thermometer | glucometer | pulse_oximeter | malaria_rdt | hiv_rdt | pregnancy_rdt | wound | rash | edema | health_record | lab_report | prescription | unknown",
  "symptoms": ["explicitly visible findings only — use terms from safe list below"],
  "vitals": {
    "systolic_bp":      null or number,
    "diastolic_bp":     null or number,
    "temperature_f":    null or number,
    "temperature_c":    null or number,
    "pulse":            null or number,
    "spo2":             null or number,
    "respiratory_rate": null or number,
    "weight_kg":        null or number,
    "blood_glucose":    null or number
  },
  "medications": [],
  "rdt_result": null or "positive | negative | invalid | reactive | non-reactive",
  "duration": null or string,
  "severity_score": 0.0,
  "referral_flag": false,
  "evidence_cited": "exact description of visible findings",
  "language_detected": "image",
  "language_notes": null or string,
  "ocr_text": null or "full verbatim text visible in the image",
  "code_status": "auto | manual_review"
}

═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
- If image is blurry, non-clinical, or ambiguous: image_type = "unknown", all numeric fields = null, code_status = "manual_review".
- NEVER infer or estimate values not clearly visible.
- NEVER add symptoms based on drug names or test results — only record observable findings.
- severity_score = 0.0 and referral_flag = false at all times; downstream engine handles prioritization.
- symptoms[] must contain ONLY terms from the safe list below.

Safe symptom terms list (only use these):
fever, high_fever, hypothermia, hypoxia, severe_hypoxia,
hyperglycemia, hypoglycemia, wound, visible_wound, visible_laceration,
rash, petechial_rash, vesicular_rash, jaundice, pallor,
bilateral_pitting_edema, facial_edema, ankle_edema,
pregnancy (only when visible test strip is clearly positive).

═══════════════════════════════════════════
READING INSTRUMENTS
═══════════════════════════════════════════
THERMOMETER:
- Read displayed number only.
- If Celsius: store in temperature_c and convert to temperature_f (F = C × 9/5 + 32).
- If Fahrenheit: store in temperature_f and convert to temperature_c.

GLUCOMETER:
- Read displayed value and unit (mg/dL or mmol/L).
- If mmol/L, convert to mg/dL (× 18.0182) and store in blood_glucose.

PULSE OXIMETER:
- Read SpO2 (%) into spo2 and pulse rate (bpm) into pulse.

═══════════════════════════════════════════
RDT STRIP READING
═══════════════════════════════════════════
- Read only line presence; do NOT infer disease.
- C line only → "negative" or "non-reactive"
- C + T lines → "positive" or "reactive"
- No C line / T only → "invalid"
- Faint T line: treat as present; describe in evidence_cited.

═══════════════════════════════════════════
PHYSICAL FINDINGS
═══════════════════════════════════════════
Record only what is directly visible:
- wounds: note presence, describe appearance in evidence_cited
- rash: note distribution and type in evidence_cited
- edema: note location and laterality in evidence_cited
- jaundice: note if skin or sclera appear yellow

Do NOT use terms like "infection", "pneumonia", "sepsis", "disease", or any condition‑name.

═══════════════════════════════════════════
OCR — HEALTH RECORDS / PRESCRIPTIONS
═══════════════════════════════════════════
- Transcribe ALL visible text into ocr_text verbatim.
- Populate medications[] with drug names and doses exactly as written.
- Do NOT add symptoms from drug names.
- Do NOT infer diagnoses or conditions from text. Only extract visible findings and complaints."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_image_as_base64(url: str) -> tuple[str, str]:
    """
    Fetch a remote image and return (base64_string, mime_type).
    Groq's vision API only accepts data: URIs — it cannot fetch external URLs.
    This must be called whenever image_url is provided.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
        r = await http.get(url, headers=headers)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return base64.b64encode(r.content).decode(), mime


def _image_block(image_base64: str, image_mime: str) -> dict:
    """
    Build a Groq-compatible image content block from base64 data.
    Always uses data: URI — never passes raw URLs to the model.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}
    }


def _add_terminology_maps(result: dict) -> dict:
    vitals = result.get("vitals", {}) or {}
    result["loinc_map"] = {
        LOINC_VITALS[k]: v
        for k, v in vitals.items()
        if v is not None and k in LOINC_VITALS
    }
    symptoms = result.get("symptoms", []) or []
    result["snomed_map"] = {
        s.lower(): SNOMED_CONDITIONS[s.lower()]
        for s in symptoms
        if s.lower() in SNOMED_CONDITIONS
    }
    return result

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def analyze_clinical_image(
    patient_id: str,
    chw_id: str = None,
    context_hint: str = None,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    manual_history: str = None,
    overrides: dict = None,
    patient_history: dict = None
) -> dict:
    from modules.normalize import normalize
    from modules.inference_engine import run_inference
    from modules.triage import apply_overrides

    if not image_base64 and not image_url:
        return {
            "error": "Provide either image_base64 or image_url",
            "patient_id": patient_id,
            "code_status": "manual_review"
        }

    if image_url and not image_base64:
        try:
            image_base64, image_mime = await fetch_image_as_base64(image_url)
        except Exception as e:
            return {
                "error": f"Failed to fetch image from URL: {str(e)}",
                "patient_id": patient_id,
                "image_url": image_url,
                "code_status": "manual_review"
            }

    user_text = "Describe all visible findings in this image."
    if context_hint:
        user_text += f" Context: {context_hint}."

    try:
        response = await client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        _image_block(image_base64, image_mime),
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1500
        )

        result = json.loads(response.choices[0].message.content)

        result = _add_terminology_maps(result)
        result["patient_id"] = patient_id
        result["input_type"] = "image"
        if chw_id:
            result["chw_id"] = chw_id

        result = normalize(result)
        result = run_inference(result)
        result = apply_overrides(result, overrides)

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