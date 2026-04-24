from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

from modules.triage import triage_extractor
from modules.fhir_ops import get_patient_context, fhir_bundle_builder

mcp = FastMCP(
    name="groundwork",
    instructions="GroundWork: Community health triage and FHIR bundle generation for CHWs."
)


@mcp.tool()
async def extract_triage(raw_text: str, patient_id: str) -> dict:
    """
    Extract triage data from Hinglish/multilingual clinical text.
    Returns severity score, referral flag, LOINC-mapped vitals.
    Use extract_triage_with_history for full COPD-aware pipeline.
    """
    return await triage_extractor(raw_text, patient_id)


@mcp.tool()
async def get_patient_history(
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """
    Pull active Condition resources for a patient from the FHIR server.
    Accepts SHARP context headers: X-FHIR-Server-URL and X-FHIR-Access-Token.
    Returns list of conditions for triage cross-referencing.
    """
    return await get_patient_context(patient_id, fhir_server_url, fhir_token)


@mcp.tool()
async def extract_triage_with_history(
    raw_text: str, 
    patient_id: str, 
    chw_id: str = None, # Added
    fhir_server_url: str = None, 
    fhir_token: str = None
) -> dict:
    patient_history = await get_patient_context(patient_id, fhir_server_url, fhir_token)
    # Pass chw_id here
    result = await triage_extractor(raw_text, patient_id, patient_history, chw_id=chw_id)
    
    result["patient_history_used"] = patient_history.get("conditions", [])
    return result

@mcp.tool()
async def build_fhir_bundle(triage_output: dict, patient_id: str) -> dict:
    """
    Build a validated FHIR R4 Transaction Bundle from triage output.
    Only call when referral_flag is true. Implemented in Phase 4.
    """
    return await fhir_bundle_builder(triage_output, patient_id)


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)