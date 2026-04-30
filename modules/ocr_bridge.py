import os
import json
import base64
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Stage 1 prompt — OCR only, no interpretation
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
# Stage 2 prompt — clinical interpretation of OCR text
# ---------------------------------------------------------------------------

CLINICAL_INTERPRETATION_PROMPT = """You are a clinical text interpreter for community health workers in LMICs.
You will receive raw OCR text extracted from a health document (prescription, lab report, health record, or referral slip).

Your job is to interpret the clinical meaning and return structured JSON.

Output MUST be a valid JSON object with EXACTLY these fields:
{
  "document_type": "prescription | lab_report | health_record | referral | unknown",
  "medications": [
    {
      "name": "drug name in full English",
      "abbreviation": "original abbreviation if any e.g. FeSO4",
      "dose": null or "e.g. 100mg",
      "quantity": null or number,
      "frequency": null or "e.g. once daily",
      "clinical_inference": "what this drug implies e.g. FeSO4 = iron supplement = anemia treatment"
    }
  ],
  "symptoms": ["normalised English symptom terms inferred from medications or explicit mentions"],
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
  "severity_score": number 0.0-1.0,
  "referral_flag": true or false,
  "evidence_cited": "what drove the severity assessment",
  "language_detected": "english | tagalog | hindi | swahili | arabic | mixed | unknown",
  "language_notes": null or "any translated terms",
  "code_status": "auto | manual_review"
}

Drug inference rules — apply these mappings:
  FeSO4 / Ferrous Sulfate / Iron tablet → symptoms: ["anemia"], clinical_inference: "iron supplementation indicates anemia"
  Ascorbic Acid / Vitamin C → clinical_inference: "vitamin C adjunct, often co-prescribed with iron"
  Metformin / Glucophage → symptoms: ["diabetes mellitus type 2"]
  Insulin → symptoms: ["diabetes mellitus type 1"], severity_score bump to min 0.60
  Salbutamol / Ventolin → symptoms: ["asthma"], severity_score bump to min 0.55
  Amoxicillin / Augmentin → symptoms: ["bacterial infection"]
  ORS / Oral Rehydration → symptoms: ["dehydration"]
  Paracetamol / Acetaminophen → symptoms: ["fever"] if no other context
  Albendazole / Mebendazole → symptoms: ["intestinal parasites"]
  Cotrimoxazole / Septrin → symptoms: ["bacterial infection"]
  Folic Acid → symptoms: ["anemia"] if co-prescribed with iron, else "pregnancy supplementation"
  Nifedipine / Amlodipine / Atenolol → symptoms: ["hypertension"], severity_score bump to min 0.55
  Digoxin → symptoms: ["heart failure"], severity_score bump to min 0.70
  ARV / Antiretroviral → symptoms: ["HIV"], severity_score 0.70, referral_flag true

Severity rules:
  Any medication implying HIV, heart failure, insulin-dependent diabetes, or active TB → referral_flag true, severity_score >= 0.70
  Iron + Folic Acid combination (maternal) → severity_score 0.30, referral_flag false
  Single antibiotic prescription → severity_score 0.40
  ORS only → severity_score 0.35
  Vitamins/supplements only → severity_score 0.15

If no medications and no vitals found → severity_score 0.0, document_type: "unknown", code_status: "manual_review"
"""

# ---------------------------------------------------------------------------
# Image fetch helper (shared with vision_triage pattern)
# ---------------------------------------------------------------------------

async def _fetch_image_as_base64(url: str) -> tuple[str, str]:
    """
    Fetch a remote image and return (base64_string, mime_type).
    Groq's vision API only accepts data: URIs — never raw external URLs.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
        r = await http.get(url, headers=headers)
        r.raise_for_status()
    mime = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return base64.b64encode(r.content).decode(), mime

# ---------------------------------------------------------------------------
# Stage 1: extract raw OCR text from image
# ---------------------------------------------------------------------------

async def extract_ocr_text(
    image_base64: str,
    image_mime: str = "image/jpeg",
) -> str:
    """
    Stage 1: Call vision model as a pure OCR engine.
    Accepts base64 only — callers must resolve URLs before calling.
    Returns raw transcribed text string.
    """
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
                    {"type": "text", "text": "Transcribe all text from this image exactly as written."}
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1024
    )

    raw = json.loads(response.choices[0].message.content)
    return raw.get("ocr_text", "")


# ---------------------------------------------------------------------------
# Stage 2: interpret OCR text clinically
# ---------------------------------------------------------------------------

async def interpret_clinical_text(ocr_text: str) -> dict:
    """
    Stage 2: Pass OCR text to text model for clinical interpretation.
    Uses the faster text model — no vision needed here.
    """
    response = await client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": CLINICAL_INTERPRETATION_PROMPT},
            {"role": "user", "content": f"Interpret this health document text:\n\n{ocr_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1500
    )

    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Main: two-stage pipeline
# ---------------------------------------------------------------------------

async def triage_from_document_image(
    patient_id: str,
    chw_id: str = None,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    patient_history: dict = None
) -> dict:
    """
    Two-stage OCR + clinical interpretation pipeline for health documents.

    Stage 1: Vision model extracts raw text (pure OCR, no interpretation).
    Stage 2: Text model interprets clinical meaning, infers symptoms from drugs.

    Works on: prescriptions, lab reports, health records, referral slips.
    Output schema is compatible with triage_extractor for downstream FHIR building.

    Args:
        patient_id:      patient identifier
        chw_id:          CHW identifier (optional)
        image_base64:    base64-encoded image
        image_mime:      image MIME type
        image_url:       direct image URL — fetched and converted to base64 here
        patient_history: FHIR patient history dict for severity cross-reference
    """
    if not image_base64 and not image_url:
        return {
            "error": "Provide either image_base64 or image_url",
            "patient_id": patient_id,
            "code_status": "manual_review"
        }

    # FIX: Groq cannot fetch external URLs — resolve to base64 before Stage 1
    if image_url and not image_base64:
        try:
            image_base64, image_mime = await _fetch_image_as_base64(image_url)
        except Exception as e:
            return {
                "error": f"Failed to fetch image from URL: {str(e)}",
                "patient_id": patient_id,
                "image_url": image_url,
                "code_status": "manual_review"
            }

    try:
        # Stage 1: OCR (base64 only from here)
        ocr_text = await extract_ocr_text(image_base64, image_mime)

        if not ocr_text.strip():
            return {
                "error": "OCR returned empty text — image may be unreadable",
                "patient_id": patient_id,
                "ocr_text": "",
                "code_status": "manual_review"
            }

        # Stage 2: Clinical interpretation
        result = await interpret_clinical_text(ocr_text)

        # Apply patient history severity upgrades if provided
        if patient_history:
            conditions_lower = [
                c.lower() for c in patient_history.get("conditions", [])
            ]
            symptoms_lower = [s.lower() for s in result.get("symptoms", [])]

            upgrades = []
            upgrade_rules = [
                ("copd",         ["anemia", "fever", "bacterial infection"]),
                ("tuberculosis", ["bacterial infection"]),
                ("diabetes",     ["anemia", "bacterial infection"]),
                ("hypertension", ["hypertension"]),
            ]
            for condition_key, triggers in upgrade_rules:
                if any(condition_key in c for c in conditions_lower):
                    if any(t in symptoms_lower for t in triggers):
                        upgrades.append(condition_key)

            if upgrades:
                old_score = result.get("severity_score", 0)
                result["severity_score"] = min(round(max(old_score + 0.25, 0.70), 2), 1.0)
                result["referral_flag"] = True
                result["evidence_cited"] = (
                    f"History upgrade: {', '.join(upgrades).upper()} + current medications. "
                    + result.get("evidence_cited", "")
                )

        # Add LOINC/SNOMED maps using existing triage maps
        try:
            from modules.triage import LOINC_VITALS, SNOMED_CONDITIONS
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
        except ImportError:
            result["loinc_map"] = {}
            result["snomed_map"] = {}

        # Metadata
        result["ocr_text"] = ocr_text
        result["patient_id"] = patient_id
        result["input_type"] = "document_image"
        if chw_id:
            result["chw_id"] = chw_id

        return result

    except json.JSONDecodeError as e:
        return {
            "error": "Clinical interpretation returned malformed JSON",
            "details": str(e),
            "patient_id": patient_id,
            "code_status": "manual_review"
        }
    except Exception as e:
        return {
            "error": "Document triage failed",
            "details": str(e),
            "patient_id": patient_id,
            "code_status": "manual_review"
        }