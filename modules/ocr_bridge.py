import os
import json
import base64
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client       = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Stage 1 — pure OCR, no interpretation
# ---------------------------------------------------------------------------

OCR_SYSTEM_PROMPT = """You are an OCR engine. Your only job is to transcribe text from the image.

Rules:
- Transcribe ALL visible text exactly as written, including abbreviations, symbols, and numbers.
- Preserve line breaks using \\n.
- Do not interpret, translate, or summarise anything.
- Do not add any text that is not in the image.
- If handwriting is unclear, write your best guess followed by [unclear].
- Output ONLY a JSON object with one field: { "ocr_text": "..." }"""

# ---------------------------------------------------------------------------
# Stage 2 — extract structured fields from OCR text, no inference
# ---------------------------------------------------------------------------

CLINICAL_INTERPRETATION_PROMPT = """You are a clinical text extractor for community health workers in LMICs.
You receive raw OCR text from a health document (prescription, lab report, health record, or referral slip).

Extract only what is explicitly present in the text. Do not infer. Do not interpret.

Output MUST be a valid JSON object with EXACTLY these fields:
{
  "document_type": "prescription | lab_report | health_record | referral | unknown",
  "medications": [
    {
      "name": "drug name in full English as written",
      "abbreviation": "original abbreviation if present, else null",
      "dose": null or string,
      "quantity": null or number,
      "frequency": null or string
    }
  ],
  "symptoms": ["complaints or findings explicitly stated in the document text only"],
  "observations": [
    {
      "name": "test or measurement name as written",
      "value": "raw value as written",
      "unit": null or string
    }
  ],
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
  "rdt_result": null or "positive | negative | invalid | reactive | non-reactive",
  "severity_score": 0.0,
  "referral_flag": false,
  "evidence_cited": "verbatim values or terms copied from the document",
  "language_detected": "english | tagalog | hindi | swahili | arabic | mixed | unknown",
  "language_notes": null or string,
  "code_status": "auto | manual_review"
}

Rules:
- symptoms[] contains only complaints explicitly stated in the document.
- Do not add symptoms from drug names.
- severity_score must be 0.0. referral_flag must be false.
- If nothing useful is found: document_type "unknown", code_status "manual_review".
"""

# ---------------------------------------------------------------------------
# Image fetch helper
# ---------------------------------------------------------------------------

async def _fetch_image_as_base64(url: str) -> tuple[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
        r = await http.get(url, headers=headers)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return base64.b64encode(r.content).decode(), mime

# ---------------------------------------------------------------------------
# Stage 1: OCR
# ---------------------------------------------------------------------------

async def extract_ocr_text(image_base64: str, image_mime: str = "image/jpeg") -> str:
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}
    }
    response = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    image_block,
                    {"type": "text", "text": "Transcribe all text from this image exactly as written. Return only a JSON object: {\"ocr_text\": \"...\"}"}
                ]
            }
        ],
        # No response_format here — same Groq/Llama-4 constrained decoder issue.
        temperature=0.0,
        max_tokens=2048
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    raw = json.loads(raw)
    return raw.get("ocr_text", "")

# ---------------------------------------------------------------------------
# Stage 2: structured extraction
# ---------------------------------------------------------------------------

async def interpret_clinical_text(ocr_text: str) -> dict:
    response = await client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": CLINICAL_INTERPRETATION_PROMPT},
            {"role": "user",   "content": f"Extract from this health document text:\n\n{ocr_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1500
    )
    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def triage_from_document_image(
    patient_id: str,
    chw_id: str = None,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    patient_history: dict = None
) -> dict:
    from modules.normalize import normalize
    from modules.inference_engine import run_inference
    from modules.triage import apply_overrides, LOINC_VITALS, _apply_history_upgrades
    from modules.terminology import build_snomed_map

    if not image_base64 and not image_url:
        return {"error": "Provide either image_base64 or image_url", "patient_id": patient_id, "code_status": "manual_review"}

    if image_url and not image_base64:
        try:
            image_base64, image_mime = await _fetch_image_as_base64(image_url)
        except Exception as e:
            return {"error": f"Failed to fetch image from URL: {str(e)}", "patient_id": patient_id, "image_url": image_url, "code_status": "manual_review"}

    try:
        ocr_text = await extract_ocr_text(image_base64, image_mime)

        if not ocr_text.strip():
            return {"error": "OCR returned empty text — image may be unreadable", "patient_id": patient_id, "ocr_text": "", "code_status": "manual_review"}

        result = await interpret_clinical_text(ocr_text)

        if patient_history:
            conditions_lower = [c.lower() for c in patient_history.get("conditions", [])]
            if conditions_lower:
                result = _apply_history_upgrades(result, conditions_lower)

        vitals = result.get("vitals", {}) or {}
        # Prefer temperature_c when both are present — same LOINC code (8310-5),
        # and fhir_ops always writes Cel/°C units. Storing °F under that code
        # would be a silent unit mismatch in the FHIR Observation.
        if vitals.get("temperature_c") is not None:
            vitals = {k: v for k, v in vitals.items() if k != "temperature_f"}
        result["loinc_map"] = {
            LOINC_VITALS[k]: v
            for k, v in vitals.items()
            if v is not None and k in LOINC_VITALS
        }
        result["snomed_map"] = await build_snomed_map(result.get("symptoms", []))

        result["ocr_text"]   = ocr_text
        result["patient_id"] = patient_id
        result["input_type"] = "document_image"
        if chw_id:
            result["chw_id"] = chw_id

        result = normalize(result)
        result = run_inference(result)
        result = apply_overrides(result, None)

        return result

    except json.JSONDecodeError as e:
        return {"error": "Clinical interpretation returned malformed JSON", "details": str(e), "patient_id": patient_id, "code_status": "manual_review"}
    except Exception as e:
        return {"error": "Document triage failed", "details": str(e), "patient_id": patient_id, "code_status": "manual_review"}