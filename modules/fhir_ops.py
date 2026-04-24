import httpx
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Public SMART sandbox — works locally without any auth token
# Prompt Opinion's server URL arrives at runtime via SHARP headers
FALLBACK_FHIR_URL = "https://r4.smarthealthit.org"

async def get_patient_context(
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """
    Pull patient Condition resources from FHIR server.
    """
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
            display = next((c.get("display") for c in codings if c.get("display")), code_block.get("text"))

            if display:
                conditions.append(display)

        return {
            "patient_id": patient_id,
            "conditions": conditions,
            "condition_count": len(conditions),
            "source": "fhir_live" if fhir_server_url else "fhir_sandbox",
            "fhir_server": base_url
        }
    except Exception as e:
        return {"patient_id": patient_id, "conditions": [], "error": str(e)}

async def fhir_bundle_builder(triage_json: dict, patient_id: str) -> dict:
    """
    Build a validated FHIR R4 Transaction Bundle.
    Converts snomed_map to Conditions and loinc_map to Observations.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    bundle_entries = []

    # 1. Map Symptoms (SNOMED) to Conditions
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

    # 2. Map Vitals (LOINC) to Observations
    for loinc_code, value in triage_json.get("loinc_map", {}).items():
        bundle_entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": loinc_code}]
                },
                "effectiveDateTime": timestamp,
                "valueQuantity": {
                    "value": value,
                    "system": "http://unitsofmeasure.org"
                }
            },
            "request": {"method": "POST", "url": "Observation"}
        })

    # 3. Capture Triage Result as RiskAssessment
    bundle_entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "occurrenceDateTime": timestamp,
            "prediction": [{
                "probabilityDecimal": triage_json.get("severity_score", 0),
                "qualitativeRisk": {"text": "Referral" if triage_json.get("referral_flag") else "Non-Urgent"},
                "rationale": triage_json.get("evidence_cited", "")
            }]
        },
        "request": {"method": "POST", "url": "RiskAssessment"}
    })

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": bundle_entries
    }