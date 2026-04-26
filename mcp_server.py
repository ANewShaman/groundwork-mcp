import asyncio
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

from modules.triage import triage_extractor
from modules.fhir_ops import get_patient_context, fhir_bundle_builder
from modules.action_dispatcher import dispatch_bundle
from modules.sync_queue import init_db, sync_worker, queue_status
from modules.vision_triage import analyze_clinical_image as _analyze_image

mcp = FastMCP(
    name="groundwork",
    instructions=(
        "GroundWork: Multilingual CHW clinical note triage, FHIR bundle generation, "
        "image analysis, and low-bandwidth sync for community health workers in LMICs."
    )
)

# ---------------------------------------------------------------------------
# Startup: initialise SQLite queue + launch background sync worker
# ---------------------------------------------------------------------------

@mcp.on_startup()
async def add_fhir_extension():
    # 1. Maintain your original startup logic
    init_db()
    asyncio.create_task(sync_worker(interval_seconds=30))

    # 2. Inject PO capabilities
    mcp._server.settings.capabilities.extensions = {
        "ai.promptopinion/fhir-context": {
            "scopes": [
                {"name": "patient/Patient.rs", "required": True},
                {"name": "patient/Condition.rs", "required": False}
            ]
        }
    }
# ---------------------------------------------------------------------------
# Phase 3 tools (unchanged)
# ---------------------------------------------------------------------------

@mcp.tool()
async def extract_triage(raw_text: str, patient_id: str) -> dict:
    """
    Extract triage data from multilingual CHW clinical text.
    Returns severity score, referral flag, LOINC-mapped vitals, SNOMED symptoms.
    For FHIR history cross-referencing, use extract_triage_with_history instead.
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
    Accepts SHARP context: fhir_server_url (X-FHIR-Server-URL) and fhir_token (X-FHIR-Access-Token).
    Returns condition list for triage cross-referencing.
    """
    return await get_patient_context(patient_id, fhir_server_url, fhir_token)


@mcp.tool()
async def extract_triage_with_history(
    raw_text: str,
    patient_id: str,
    chw_id: str = None,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """
    PRIMARY TRIAGE TOOL: Full pipeline with FHIR history cross-reference.
    Step 1 — Pull patient conditions from FHIR server.
    Step 2 — Triage the clinical note with history injected.
    Step 3 — Apply deterministic severity upgrades (COPD+cough, TB+cough, etc).
    Accepts SHARP context: fhir_server_url and fhir_token.
    """
    patient_history = await get_patient_context(patient_id, fhir_server_url, fhir_token)
    result = await triage_extractor(raw_text, patient_id, patient_history, chw_id=chw_id)
    result["patient_history_used"] = patient_history.get("conditions", [])
    result["fhir_source"] = patient_history.get("source", "none")
    return result


@mcp.tool()
async def build_fhir_bundle(triage_output: dict, patient_id: str) -> dict:
    """
    Build a validated FHIR R4 Transaction Bundle from triage output.
    Generates Condition resources (SNOMED), Observation resources (LOINC), and RiskAssessment.
    Only call when referral_flag is true.
    """
    return await fhir_bundle_builder(triage_output, patient_id)

# ---------------------------------------------------------------------------
# Phase 5: dispatch with idempotency + retries
# ---------------------------------------------------------------------------

@mcp.tool()
async def dispatch_fhir_bundle(
    triage_output: dict,
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None,
    chw_id: str = None
) -> dict:
    """
    Build and dispatch a FHIR bundle in one step.
    Includes idempotency key (prevents duplicates on retry) and tenacity retries
    (exponential backoff on network failure). Falls back to local sync queue if
    all retries fail — returns status=queued instead of error.
    Only call when triage_output contains referral_flag=true.
    """
    bundle = await fhir_bundle_builder(triage_output, patient_id)
    return await dispatch_bundle(bundle, patient_id, fhir_server_url, fhir_token, chw_id)

# ---------------------------------------------------------------------------
# Phase 6: sync queue status
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_sync_queue() -> dict:
    """
    Return current sync queue status: pending, synced, failed counts.
    Use during demo to show offline queue draining when connection restores.
    The background worker retries pending bundles every 30 seconds automatically.
    """
    return queue_status()

# ---------------------------------------------------------------------------
# Phase 7: image triage
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_clinical_image(
    image_base64: str,
    image_mime: str,
    patient_id: str,
    chw_id: str = None,
    context_hint: str = None
) -> dict:
    """
    Vision triage: analyze a photo taken by a CHW and extract structured clinical data.
    Supports: malaria RDT strips, thermometers, wounds, handwritten health records, lab results.
    Output schema is identical to extract_triage — same downstream pipeline (build_fhir_bundle,
    dispatch_fhir_bundle) works unchanged.

    Args:
        image_base64: base64-encoded image (no data URI prefix)
        image_mime:   "image/jpeg" or "image/png"
        patient_id:   patient identifier
        chw_id:       CHW identifier (optional)
        context_hint: optional hint e.g. "malaria test strip", "thermometer reading"
    """
    return await _analyze_image(image_base64, image_mime, patient_id, chw_id, context_hint)


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)