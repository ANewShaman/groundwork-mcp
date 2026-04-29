import asyncio
import os
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
# Startup
# ---------------------------------------------------------------------------

@mcp.on_startup()
async def startup():
    init_db()
    asyncio.create_task(sync_worker(interval_seconds=30))
    print("[GroundWork] Started. Sync worker running.")

    # Attempt FHIR extension injection — log result either way so you know if it worked
    try:
        raw = mcp._mcp_server
        orig = raw.handle_initialize

        async def patched(params):
            result = await orig(params)
            try:
                if result.capabilities.experimental is None:
                    result.capabilities.experimental = {}
                result.capabilities.experimental["ai.promptopinion/fhir-context"] = {
                    "scopes": [
                        {"name": "patient/Patient.rs",    "required": True},
                        {"name": "patient/Condition.rs",  "required": False},
                        {"name": "patient/Observation.rs","required": False}
                    ]
                }
                print("[GroundWork] FHIR context extension injected.")
            except Exception as inner:
                print(f"[GroundWork] Extension inject failed inside handler: {inner}")
            return result

        raw.handle_initialize = patched
        print("[GroundWork] Initialize handler patched.")
    except Exception as e:
        print(f"[GroundWork] Could not patch initialize: {e}")
        print("[GroundWork] FHIR Context Ext will show No on PO — ask PO Discord for correct hook.")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def extract_triage(raw_text: str, patient_id: str) -> dict:
    """
    Extract triage data from multilingual CHW clinical text.
    Returns severity score, referral flag, LOINC-mapped vitals, SNOMED symptoms.
    Use extract_triage_with_history for FHIR history cross-referencing.
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
    SHARP context: fhir_server_url = X-FHIR-Server-URL, fhir_token = X-FHIR-Access-Token.
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
    Step 2 — Triage the multilingual clinical note with history injected.
    Step 3 — Apply deterministic severity upgrades (COPD+cough, TB+cough, etc).
    SHARP context: fhir_server_url = X-FHIR-Server-URL, fhir_token = X-FHIR-Access-Token,
    patient_id = X-Patient-ID.
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
    Generates Condition (SNOMED), Observation (LOINC), and RiskAssessment resources.
    Only call when referral_flag is true.
    """
    return await fhir_bundle_builder(triage_output, patient_id)


@mcp.tool()
async def dispatch_fhir_bundle(
    triage_output: dict,
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None,
    chw_id: str = None
) -> dict:
    """
    Build and dispatch a FHIR bundle in one step with idempotency and retries.
    Idempotency key prevents duplicate records on bad-connection retries.
    Falls back to local SQLite sync queue if all retries fail — returns status=queued.
    Only call when triage_output contains referral_flag=true.
    SHARP context: fhir_server_url = X-FHIR-Server-URL, fhir_token = X-FHIR-Access-Token.
    """
    bundle = await fhir_bundle_builder(triage_output, patient_id)
    return await dispatch_bundle(bundle, patient_id, fhir_server_url, fhir_token, chw_id)


@mcp.tool()
async def check_sync_queue() -> dict:
    """
    Return sync queue status: pending, synced, failed counts.
    Background worker retries pending bundles every 30 seconds automatically.
    """
    return queue_status()


@mcp.tool()
async def analyze_clinical_image(
    patient_id: str,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    chw_id: str = None,
    context_hint: str = None,
    manual_history: str = None
) -> dict:
    """
    Vision triage: analyze a CHW photo and extract structured clinical data.
    Supports: malaria RDT strips, thermometers, wounds, handwritten records, lab results.
    Output schema identical to extract_triage — same downstream pipeline works unchanged.
    Pass image_base64 + image_mime for encoded images, or image_url for direct URLs.
    """
    return await _analyze_image(
        patient_id=patient_id,
        chw_id=chw_id,
        context_hint=context_hint,
        image_base64=image_base64,
        image_mime=image_mime,
        image_url=image_url,
        manual_history=manual_history
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="sse", host="0.0.0.0", port=port)