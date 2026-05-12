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
    from modules.triage import LOINC_VITALS
except ImportError:
    LOINC_VITALS = {}

# ---------------------------------------------------------------------------
# VISION SYSTEM PROMPT — Llama-4, FHIR-compliant, CHW-optimised
# ---------------------------------------------------------------------------
VISION_SYSTEM_PROMPT = """You are a clinical image observer for Community Health Workers in LMICs.

Your job is to extract every visible clinical value from the image.
DO NOT diagnose diseases.
DO NOT infer conditions from medications or test results.

===========================================
OUTPUT SCHEMA - return this exact structure
===========================================
{
  "image_type": "thermometer | glucometer | pulse_oximeter | malaria_rdt | hiv_rdt | pregnancy_rdt | wound | rash | edema | health_record | lab_report | prescription | unknown",
  "symptoms": ["explicitly visible findings only -- use terms from safe list below"],
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
  "code_status": "auto"
}

===========================================
CODE STATUS - read this carefully
===========================================
code_status = "auto"          -- all extracted values are clearly readable.
code_status = "manual_review" -- you extracted values but are less than fully
                                confident due to image quality, angle, or
                                partial obstruction. A CHW will verify.

CRITICAL: code_status = "manual_review" means EXTRACT AND FLAG -- not null and skip.
- ALWAYS populate every field you can read, even partially.
- Only set a field to null if that specific value is genuinely not visible.
- There is NO situation where all numeric fields are null AND code_status is
  "manual_review" unless the image contains zero medical content at all
  (e.g. a blank wall, random object, or completely unrelated photo).
- Product photos, white-background device images, and low-resolution images
  of medical instruments MUST still be read. Attempt extraction regardless
  of whether the image looks "clinical".

===========================================
READING INSTRUMENTS
===========================================
THERMOMETER:
- Read the displayed number. If Celsius: populate temperature_c and convert
  to temperature_f (F = C x 9/5 + 32). If Fahrenheit: populate temperature_f
  and convert to temperature_c.
- If the unit symbol is unclear, use context (values >50 are Fahrenheit).

GLUCOMETER:
- Read the displayed value and unit (mg/dL or mmol/L).
- If mmol/L, convert to mg/dL (x 18.0182) and store in blood_glucose.

PULSE OXIMETER:
- The larger number is SpO2 (%) -> spo2. The smaller number is pulse (bpm) -> pulse.

===========================================
RDT STRIP READING
===========================================
MANDATORY: If image_type is any *_rdt value, rdt_result MUST be set. Never null for RDT images.
- C line only           -> "negative" or "non-reactive"
- C + T lines           -> "positive" or "reactive"
- No C line / T only    -> "invalid"
- Faint T line: treat as present, set rdt_result accordingly, note in evidence_cited.
- If lines are ambiguous: set rdt_result to your best reading, set code_status to "manual_review".

===========================================
PHYSICAL FINDINGS
===========================================
Record only what is directly visible:
- wounds, rash, edema, jaundice: describe in evidence_cited.
- Do NOT use terms like "infection", "pneumonia", "sepsis", or any diagnosis.

Safe symptom terms (symptoms[] only):
fever, high_fever, hypothermia, hypoxia, severe_hypoxia,
hyperglycemia, hypoglycemia, wound, visible_wound, visible_laceration,
rash, petechial_rash, vesicular_rash, jaundice, pallor,
bilateral_pitting_edema, facial_edema, ankle_edema,
pregnancy (only when a positive test strip is clearly visible).

===========================================
OCR - HEALTH RECORDS / PRESCRIPTIONS
===========================================
- Transcribe ALL visible text into ocr_text verbatim.
- Populate medications[] with drug names and doses exactly as written.
- Do NOT add symptoms from drug names.
- Do NOT infer diagnoses from text."""
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
    # snomed_map intentionally excluded — resolved async in analyze_clinical_image
    vitals = result.get("vitals", {}) or {}
    result["loinc_map"] = {
        LOINC_VITALS[k]: v
        for k, v in vitals.items()
        if v is not None and k in LOINC_VITALS
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
    from modules.terminology import build_snomed_map

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

    user_text = (
        "Extract all visible clinical values from this image and return a single JSON object "
        "exactly matching the schema in the system prompt. "
        "Do not include any text before or after the JSON object."
    )
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
            # Do NOT use response_format={"type": "json_object"} with Llama-4 Scout
            # on Groq — the constrained decoder corrupts output when the system prompt
            # contains multi-byte Unicode (box-drawing chars). The model follows
            # explicit JSON instructions in the user turn reliably without it.
            temperature=0.0,
            max_tokens=4096
        )

        raw_content = response.choices[0].message.content.strip()
        # Strip markdown fences if present — model occasionally wraps JSON
        # in ```json ... ``` when response_format is not enforced.
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```", 2)[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
            raw_content = raw_content.rsplit("```", 1)[0].strip()
        result = json.loads(raw_content)

        result = _add_terminology_maps(result)
        result["snomed_map"] = await build_snomed_map(result.get("symptoms", []) or [])
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