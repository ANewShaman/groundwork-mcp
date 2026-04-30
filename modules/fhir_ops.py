import httpx
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

FALLBACK_FHIR_URL = "https://r4.smarthealthit.org"

# ---------------------------------------------------------------------------
# RxNorm codes for common LMIC medications — no database needed
# ---------------------------------------------------------------------------

RXNORM_MAP = {
    "ferrous sulfate":       "4482",
    "feso4":                 "4482",
    "iron tablet":           "4482",
    "ascorbic acid":         "1049502",
    "vitamin c":             "1049502",
    "paracetamol":           "161",
    "acetaminophen":         "161",
    "amoxicillin":           "723",
    "metformin":             "6809",
    "salbutamol":            "435",
    "albuterol":             "435",
    "cotrimoxazole":         "10829",
    "ors":                   "9863",
    "oral rehydration":      "9863",
    "albendazole":           "16681",
    "folic acid":            "4511",
    "nifedipine":            "7417",
    "amlodipine":            "17767",
    "atenolol":              "1202",
    "digoxin":               "3407",
    "insulin":               "5856",
}

def _get_rxnorm_code(drug_name: str) -> str | None:
    key = drug_name.lower().strip()
    return RXNORM_MAP.get(key) or RXNORM_MAP.get(key.split()[0])

# ---------------------------------------------------------------------------
# Patient context
# ---------------------------------------------------------------------------

async def get_patient_context(
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """Pull patient Condition resources from FHIR server."""
    base_url = (fhir_server_url or FALLBACK_FHIR_URL).rstrip("/")
    headers = {"Accept": "application/fhir+json"}
    if fhir_token:
        headers["Authorization"] = f"Bearer {fhir_token}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/Condition",
                params={
                    "patient": patient_id,
                    "clinical-status": "active",
                    "_count": "20"
                },
                headers=headers
            )
            response.raise_for_status()
            bundle = response.json()

        conditions = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Condition":
                continue
            code_block = resource.get("code", {})
            codings = code_block.get("coding", [])
            display = next(
                (c.get("display") for c in codings if c.get("display")),
                code_block.get("text")
            )
            if display:
                conditions.append(display)

        return {
            "patient_id":      patient_id,
            "conditions":      conditions,
            "condition_count": len(conditions),
            "source":          "fhir_live" if fhir_server_url else "fhir_sandbox",
            "fhir_server":     base_url
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "conditions": [],
            "condition_count": 0,
            "source": "fhir_live" if fhir_server_url else "fhir_sandbox",
            "fhir_server": base_url,
            "error": str(e)
        }

# ---------------------------------------------------------------------------
# Triage FHIR bundle (Condition + Observation + RiskAssessment)
# ---------------------------------------------------------------------------

async def fhir_bundle_builder(triage_json: dict, patient_id: str) -> dict:
    """
    Build a FHIR R4 Transaction Bundle from triage output.
    Generates: Condition (SNOMED), Observation (LOINC), RiskAssessment.
    For prescription documents, use medication_bundle_builder instead.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    bundle_entries = []

    # 1. Conditions from SNOMED map
    for display, code in triage_json.get("snomed_map", {}).items():
        bundle_entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}],
                    "text": display
                },
                "recordedDate": timestamp
            },
            "request": {"method": "POST", "url": "Condition"}
        })

    # 2. Observations from LOINC map
    LOINC_UNITS = {
        "8480-6": ("mm[Hg]", "mmHg"),
        "8462-4": ("mm[Hg]", "mmHg"),
        "8310-5": ("Cel",    "°C"),
        "8867-4": ("/min",   "bpm"),
        "59408-5":("%",      "%"),
        "9279-1": ("/min",   "/min"),
        "29463-7":("kg",     "kg"),
        "2339-0": ("mg/dL",  "mg/dL"),
    }
    for loinc_code, value in triage_json.get("loinc_map", {}).items():
        ucum, display_unit = LOINC_UNITS.get(loinc_code, ("1", ""))
        bundle_entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"coding": [{"system": "http://loinc.org", "code": loinc_code}]},
                "effectiveDateTime": timestamp,
                "valueQuantity": {
                    "value":  value,
                    "unit":   display_unit,
                    "system": "http://unitsofmeasure.org",
                    "code":   ucum
                }
            },
            "request": {"method": "POST", "url": "Observation"}
        })

    # 3. RiskAssessment
    bundle_entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": timestamp,
            "prediction": [{
                "probabilityDecimal": triage_json.get("severity_score", 0),
                "qualitativeRisk":    {"text": "Referral" if triage_json.get("referral_flag") else "Non-Urgent"},
                "rationale":          triage_json.get("evidence_cited", "")
            }]
        },
        "request": {"method": "POST", "url": "RiskAssessment"}
    })

    return {"resourceType": "Bundle", "type": "transaction", "entry": bundle_entries}

# ---------------------------------------------------------------------------
# Prescription / document FHIR bundle (MedicationRequest + Condition)
# ---------------------------------------------------------------------------

async def medication_bundle_builder(ocr_result: dict, patient_id: str) -> dict:
    """
    Build a FHIR R4 Transaction Bundle from OCR bridge output.
    Generates: MedicationRequest per drug, Condition per inferred symptom,
    and a RiskAssessment for triage severity.

    Call this instead of fhir_bundle_builder when input_type == "document_image".
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    bundle_entries = []

    # 1. MedicationRequest per medication
    for med in ocr_result.get("medications", []):
        drug_name = med.get("name", "Unknown medication")
        rxnorm_code = _get_rxnorm_code(med.get("abbreviation", "") or drug_name)

        medication_coding = [{
            "system":  "http://www.nlm.nih.gov/research/umls/rxnorm",
            "code":    rxnorm_code,
            "display": drug_name
        }] if rxnorm_code else [{"display": drug_name}]

        dosage = []
        if med.get("frequency") or med.get("dose"):
            dosage_text = " ".join(filter(None, [med.get("dose"), med.get("frequency")]))
            dosage = [{"text": dosage_text}]

        dispense = {}
        if med.get("quantity"):
            dispense = {
                "quantity": {
                    "value":  med["quantity"],
                    "unit":   "tablet",
                    "system": "http://terminology.hl7.org/CodeSystem/v3-orderableDrugForm",
                    "code":   "TAB"
                }
            }

        entry = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType":     "MedicationRequest",
                "status":           "active",
                "intent":           "order",
                "subject":          {"reference": f"Patient/{patient_id}"},
                "authoredOn":       timestamp,
                "medicationCodeableConcept": {
                    "coding": medication_coding,
                    "text":   drug_name
                },
                "meta": {
                    "tag": [{"system": "http://groundwork.health/tags", "code": "CHW_ORIGINATED"}]
                }
            },
            "request": {"method": "POST", "url": "MedicationRequest"}
        }

        if dosage:
            entry["resource"]["dosageInstruction"] = dosage
        if dispense:
            entry["resource"]["dispenseRequest"] = dispense
        if med.get("clinical_inference"):
            entry["resource"]["note"] = [{"text": med["clinical_inference"]}]

        bundle_entries.append(entry)

    # 2. Conditions from SNOMED map (inferred from drugs)
    snomed_map = ocr_result.get("snomed_map", {})
    for display, code in snomed_map.items():
        bundle_entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
                },
                "subject":      {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}],
                    "text":   display
                },
                "recordedDate": timestamp,
                "note": [{"text": "Inferred from prescription — CHW document scan"}]
            },
            "request": {"method": "POST", "url": "Condition"}
        })

    # 3. RiskAssessment
    bundle_entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType":      "RiskAssessment",
            "status":            "final",
            "subject":           {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": timestamp,
            "prediction": [{
                "probabilityDecimal": ocr_result.get("severity_score", 0),
                "qualitativeRisk":    {"text": "Referral" if ocr_result.get("referral_flag") else "Non-Urgent"},
                "rationale":          ocr_result.get("evidence_cited", "")
            }]
        },
        "request": {"method": "POST", "url": "RiskAssessment"}
    })

    return {
        "resourceType": "Bundle",
        "type":         "transaction",
        "entry":        bundle_entries,
        "meta": {
            "tag": [{"system": "http://groundwork.health/tags", "code": "PRESCRIPTION_OCR"}]
        }
    }