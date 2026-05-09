from modules.fhir_ops import get_patient_context

async def resolve_patient_context(
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    if not fhir_server_url:
        return {
            "patient_id": patient_id,
            "conditions": [],
            "condition_count": 0,
            "source": "none"
        }

    try:
        result = await get_patient_context(patient_id, fhir_server_url, fhir_token)

        if result.get("error"):
            result["source"] = "fhir_error"
        else:
            result["source"] = "fhir"

        return result

    except Exception as e:
        return {
            "patient_id": patient_id,
            "conditions": [],
            "condition_count": 0,
            "source": "fhir_error",
            "error": str(e)
        }