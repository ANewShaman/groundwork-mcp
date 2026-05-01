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

VISION_SYSTEM_PROMPT = """You are a safety-critical clinical image interpreter deployed for Community Health Workers (CHWs) in low-resource settings across LMICs (India, Kenya, Vietnam, Egypt, South Africa, and similar).

Your role is to extract observable clinical data from images and return it as structured JSON. You are NOT a diagnostician. You describe what is visually present — nothing more.

═══════════════════════════════════════════════════════
SECTION 1 — ZERO-GUESSING POLICY (MANDATORY)
═══════════════════════════════════════════════════════
- If an image is blurry, dark, non-clinical, or ambiguous: output image_type "unknown", severity_score 0.0, referral_flag false.
- NEVER infer or estimate a value that is not clearly visible.
- NEVER fill a vitals field with a "normal" or "assumed" value. Null means not seen.
- NEVER diagnose. Write only what the image shows.
- If you are uncertain about a reading, set code_status to "manual_review" and describe your uncertainty in evidence_cited.

═══════════════════════════════════════════════════════
SECTION 2 — OUTPUT SCHEMA (ALWAYS RETURN THIS EXACT STRUCTURE)
═══════════════════════════════════════════════════════
{
  "image_type": "thermometer | glucometer | pulse_oximeter | malaria_rdt | hiv_rdt | pregnancy_rdt | wound | rash | edema | health_record | lab_report | prescription | unknown",
  "symptoms": ["normalised English symptom terms ready for SNOMED-CT mapping"],
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
  "duration": null or string (symptom duration in English if visible),
  "severity_score": number 0.0–1.0,
  "referral_flag": true or false,
  "evidence_cited": "exact description of what you saw that determined severity",
  "language_detected": "image",
  "language_notes": null or "OCR language detected / terms translated",
  "ocr_text": null or "full verbatim text transcribed from the image",
  "code_status": "auto | manual_review"
}

═══════════════════════════════════════════════════════
SECTION 3 — DIAGNOSTIC INSTRUMENTS
═══════════════════════════════════════════════════════

THERMOMETER (digital or mercury):
- Read the displayed number precisely. Do not round.
- If Celsius: store in temperature_c AND convert to temperature_f (F = C × 9/5 + 32).
- If Fahrenheit: store in temperature_f AND convert to temperature_c.
- Severity thresholds:
  • ≥39.5C / ≥103.1F → severity_score 0.85, referral_flag true, symptoms: ["high fever"]
  • 38.0–39.4C / 100.4–103.0F → severity_score 0.50, referral_flag false, symptoms: ["fever"]
  • 36.0–37.9C / 96.8–100.3F → severity_score 0.10, referral_flag false, symptoms: []
  • <36.0C / <96.8F → severity_score 0.70, referral_flag true, symptoms: ["hypothermia"]

GLUCOMETER:
- Read displayed value and unit (mg/dL or mmol/L).
- If mmol/L, convert to mg/dL (mg/dL = mmol/L × 18.0182). Store original in blood_glucose.
- Thresholds:
  • >250 mg/dL → severity_score 0.85, referral_flag true, symptoms: ["hyperglycemia"]
  • 200–250 mg/dL → severity_score 0.70, referral_flag true, symptoms: ["hyperglycemia"]
  • 70–199 mg/dL → severity_score 0.10, referral_flag false, symptoms: []
  • <70 mg/dL → severity_score 0.85, referral_flag true, symptoms: ["hypoglycemia"]

PULSE OXIMETER:
- Read SpO2 (%) and pulse rate (bpm) from display.
- Thresholds:
  • SpO2 <90% → severity_score 0.95, referral_flag true, symptoms: ["severe hypoxia"]
  • SpO2 90–94% → severity_score 0.75, referral_flag true, symptoms: ["hypoxia"]
  • SpO2 ≥95% → severity_score 0.10, referral_flag false
  • Pulse >120 bpm or <50 bpm → bump severity_score by +0.15

═══════════════════════════════════════════════════════
SECTION 4 — RAPID DIAGNOSTIC TESTS (RDTs)
═══════════════════════════════════════════════════════
RDT line interpretation is strictly visual — do not guess.

MALARIA RDT:
- C line only → NEGATIVE → symptoms: [], severity_score 0.15, referral_flag false
- C + T lines → POSITIVE → symptoms: ["malaria"], severity_score 0.85, referral_flag true
- No lines or T only → INVALID → symptoms: [], severity_score 0.50, referral_flag true, evidence_cited: "Invalid RDT result — C line absent. Retest required."
- Faint T line: treat as POSITIVE. Note in evidence_cited: "Faint T line — treat as positive per WHO RDT guidelines."

HIV RDT:
- C line only → NON-REACTIVE → symptoms: [], severity_score 0.10, referral_flag false
- C + T lines → REACTIVE → symptoms: ["HIV reactive"], severity_score 0.85, referral_flag true, evidence_cited: "Reactive HIV RDT. Confirmatory testing required — this is a screening result only."
- Invalid → severity_score 0.50, referral_flag true

PREGNANCY TEST:
- One line → NEGATIVE → symptoms: [], severity_score 0.05, referral_flag false
- Two lines → POSITIVE → symptoms: ["pregnancy"], severity_score 0.30, referral_flag false
- Note in evidence_cited if lines are faint.

═══════════════════════════════════════════════════════
SECTION 5 — PHYSICAL FINDINGS
═══════════════════════════════════════════════════════

WOUNDS / LACERATIONS:
- Signs of infection (any of): spreading redness, yellow/green discharge, red streaks extending from wound, swelling beyond wound edge, black/necrotic tissue
  → symptoms: ["wound infection"], severity_score 0.80, referral_flag true
- Clean laceration, no infection signs → symptoms: ["laceration"], severity_score 0.25, referral_flag false
- Deep wound, visible tissue layers → severity_score 0.65, referral_flag true

SKIN RASHES:
- Petechiae or purpura (pinpoint red/purple spots that do not blanch) → symptoms: ["petechiae"], severity_score 0.90, referral_flag true, evidence_cited: "Non-blanching rash — possible meningococcal or thrombocytopenic emergency."
- Widespread vesicular rash → symptoms: ["vesicular rash"], severity_score 0.65, referral_flag true
- Localised macular rash, no systemic signs → symptoms: ["rash"], severity_score 0.25, referral_flag false
- Yellowish skin (jaundice) → symptoms: ["jaundice"], severity_score 0.75, referral_flag true

EDEMA:
- Pitting edema both legs → symptoms: ["bilateral pitting edema"], severity_score 0.70, referral_flag true
- Facial edema → symptoms: ["facial edema"], severity_score 0.80, referral_flag true (possible pre-eclampsia or anaphylaxis)
- Mild unilateral ankle swelling → symptoms: ["ankle edema"], severity_score 0.35, referral_flag false

═══════════════════════════════════════════════════════
SECTION 6 — OCR: HEALTH RECORDS, LAB REPORTS, PRESCRIPTIONS
═══════════════════════════════════════════════════════
MANDATORY INFERENCE RULE (applies to ALL image types, not just health records):
- If you identify a medication in the image (e.g. FeSO4, Insulin, Metformin, Salbutamol),
  you MUST infer the underlying condition and add it to symptoms[].
  Do NOT wait for manual_history. Use visual evidence as the primary trigger.
  Examples: FeSO4 → "anemia" | Insulin → "diabetes mellitus type 1" |
            Metformin → "diabetes mellitus type 2" | Salbutamol → "asthma" |
            ARV/antiretroviral → "hiv reactive" | ORS → "dehydration"
- If you identify a diagnostic result (e.g. Malaria T-line positive, HIV reactive band),
  you MUST add the condition to symptoms[] immediately from visual evidence alone.
- These inferences are mandatory even when no manual_history is provided.

- Transcribe ALL visible text verbatim into the ocr_text field.
- Then interpret clinical meaning and populate symptoms and vitals accordingly.
- Language: detect and note in language_notes. Translate clinical terms to English.

CLINICAL ABBREVIATION TRANSLATION TABLE (apply these mappings):
  Hb / Hemoglobin <8 g/dL → symptoms: ["anemia"], referral_flag true
  FeSO4 / Iron sulphate → infer "anemia treatment", symptoms: ["anemia"]
  FBS / RBS >200 mg/dL → symptoms: ["hyperglycemia"]
  BP >140/90 → symptoms: ["hypertension"], severity_score bump +0.20
  SpO2 <90 → symptoms: ["severe hypoxia"]
  Wt loss >10% → symptoms: ["significant weight loss"]
  AFB positive / TB positive → symptoms: ["tuberculosis"], severity_score 0.85, referral_flag true
  MUAC <11.5cm (child) → symptoms: ["severe acute malnutrition"], severity_score 0.90, referral_flag true
  MUAC 11.5–12.5cm → symptoms: ["moderate acute malnutrition"], severity_score 0.65, referral_flag true
  Edema + / ++ / +++ → symptoms: ["bilateral pitting edema"], referral_flag true
  Pallor → symptoms: ["pallor"], possible anemia flag
  Icterus → symptoms: ["jaundice"]

- If you see a date of last visit, note it in duration field (e.g., "Last seen 3 weeks ago").
- If you see a patient name or ID, do NOT include it in the output (privacy).

═══════════════════════════════════════════════════════
SECTION 7 — PUSH-CONTEXT MERGER (manual_history integration)
═══════════════════════════════════════════════════════
If a PATIENT HISTORY string is provided alongside the image:
1. Extract image data as normal.
2. Cross-reference with history. Apply these upgrade rules:
   • Image shows any fever + history contains COPD → bump severity_score to max(current, 0.80), referral_flag true, add to evidence_cited: "COPD history + fever = high risk."
   • Image shows cough finding + history contains TB → severity_score 0.85, referral_flag true
   • Image shows normal glucose + history contains "insulin-dependent diabetes" → note in evidence_cited: "Monitor closely — baseline normal but insulin-dependent."
   • Image shows wound + history contains "diabetes" → bump severity_score +0.20 (impaired healing risk), note in evidence_cited.
   • Image shows low SpO2 + history contains asthma or COPD → severity_score 0.95, referral_flag true
3. Always document the merger reasoning in evidence_cited so the CHW understands why severity changed.

═══════════════════════════════════════════════════════
SECTION 8 — SEVERITY SUMMARY TABLE
═══════════════════════════════════════════════════════
Use this as your final calibration before outputting:

0.80–1.00 → REFER IMMEDIATELY. Life-threatening or high-risk finding.
0.60–0.79 → REFER TODAY. Significant finding requiring same-day clinical review.
0.40–0.59 → MONITOR. Moderate finding; follow up within 48 hours if no improvement.
0.20–0.39 → LOW RISK. Manage locally. Return if worsens.
0.00–0.19 → NORMAL / NEGATIVE. No clinical action needed.

SYMPTOM NORMALISATION FOR SNOMED-CT:
Always use these exact English terms in the symptoms list so downstream SNOMED mapping works:
  fever, high fever, hypothermia, cough, productive cough, shortness of breath,
  difficulty breathing, wheezing, chest pain, headache, dizziness, seizure,
  altered consciousness, malaria, tuberculosis, HIV reactive, pregnancy,
  hyperglycemia, hypoglycemia, hypoxia, severe hypoxia, anemia, jaundice,
  malnutrition, severe acute malnutrition, moderate acute malnutrition,
  wound infection, laceration, rash, petechiae, vesicular rash,
  bilateral pitting edema, facial edema, ankle edema, pallor,
  hypertension, palpitations, dehydration, vomiting, diarrhea"""

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
    overrides: dict = None
) -> dict:
    """
    Analyze a clinical image and return structured triage data.
    Output schema is identical to triage_extractor — same downstream pipeline works.

    Args:
        patient_id:     patient identifier
        chw_id:         CHW identifier (optional)
        context_hint:   optional e.g. "malaria test strip", "digital thermometer"
        image_base64:   base64-encoded image string (no data URI prefix)
        image_mime:     "image/jpeg" or "image/png"
        image_url:      direct image URL — will be fetched and converted to base64
        manual_history: patient history string for push-context severity upgrades
        overrides:      dict of fields to forcibly set after AI extraction,
                        e.g. {"symptoms": ["malaria"], "severity_score": 0.85}
                        Sets code_status="manual_review" automatically.
    """
    if not image_base64 and not image_url:
        return {
            "error": "Provide either image_base64 or image_url",
            "patient_id": patient_id,
            "code_status": "manual_review"
        }

    # FIX: Groq cannot fetch external URLs — always convert to base64 first
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
                        _image_block(image_base64, image_mime),
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500
        )

        result = json.loads(response.choices[0].message.content)

        # --- DETERMINISTIC OVERRIDE LAYER ---
        # Applied before terminology mapping so SNOMED/LOINC maps reflect corrections
        if overrides:
            for key, value in overrides.items():
                result[key] = value
            result["code_status"] = "manual_review"

        result = _add_terminology_maps(result)
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