import os
import threading
from fastmcp import FastMCP
from dotenv import load_dotenv
from modules.fhir_context import resolve_patient_context

load_dotenv()

from modules.triage import triage_extractor
from modules.fhir_ops import (
    get_patient_context,
    fhir_bundle_builder,
    medication_bundle_builder
)
from modules.action_dispatcher import dispatch_bundle
from modules.sync_queue import init_db, sync_worker, queue_status
from modules.vision_triage import analyze_clinical_image as _analyze_image
from modules.ocr_bridge import triage_from_document_image


# ---------------------------------------------------------------------------
# Sync worker — runs in a daemon thread, no lifespan needed
# ---------------------------------------------------------------------------

def _start_sync_worker():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(sync_worker(interval_seconds=30))
    except Exception:
        pass
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# MCP server — no lifespan parameter
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="groundwork",
    instructions=(
        "GroundWork: Multilingual CHW clinical note triage, FHIR bundle generation, "
        "image analysis, prescription OCR, and low-bandwidth sync for community health "
        "workers in LMICs. Supports Hindi, Swahili, Tagalog, Amharic, Vietnamese, "
        "Arabic, English."
    )
)

# ---------------------------------------------------------------------------
# Text triage
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
    Pull active Condition resources from FHIR server via SHARP context.
    SHARP headers: fhir_server_url = X-FHIR-Server-URL,
    fhir_token = X-FHIR-Access-Token.
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
    PRIMARY TRIAGE TOOL. Full pipeline with FHIR history cross-reference.
    Step 1 — Pull patient conditions from FHIR (only if fhir_server_url present).
    Step 2 — Triage multilingual note with history injected.
    Step 3 — Deterministic severity upgrades fire for COPD, TB, hypertension, etc.
    SHARP: fhir_server_url = X-FHIR-Server-URL, fhir_token = X-FHIR-Access-Token.
    """
    patient_history = await resolve_patient_context(
        patient_id, fhir_server_url, fhir_token
    )
    result = await triage_extractor(
        raw_text, patient_id, patient_history, chw_id=chw_id
    )
    result["fhir_context_used"] = patient_history.get("source") in ["fhir", "fhir_error"]
    result["fhir_source"] = patient_history.get("source", "none")
    result["patient_history_used"] = patient_history.get("conditions", [])
    return result


# ---------------------------------------------------------------------------
# FHIR bundle tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def build_fhir_bundle(triage_output: dict, patient_id: str) -> dict:
    """
    Build a validated FHIR R4 Transaction Bundle from text triage output.
    Generates Condition (SNOMED), Observation (LOINC), RiskAssessment.
    Only call when referral_flag is true.
    """
    return await fhir_bundle_builder(triage_output, patient_id)


@mcp.tool()
async def build_medication_bundle(ocr_result: dict, patient_id: str) -> dict:
    """
    Build a FHIR R4 Transaction Bundle from prescription OCR output.
    Generates MedicationRequest (RxNorm), Condition (SNOMED), RiskAssessment.
    Call after triage_document_image.
    """
    return await medication_bundle_builder(ocr_result, patient_id)


# ---------------------------------------------------------------------------
# Dispatch tools
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
    Build and dispatch a triage FHIR bundle in one step.
    Idempotency key prevents duplicates. Exponential backoff on failure.
    Falls back to SQLite sync queue if all retries fail — returns status=queued.
    Only call when triage_output contains referral_flag=true.
    """
    bundle = await fhir_bundle_builder(triage_output, patient_id)
    return await dispatch_bundle(
        bundle, patient_id, fhir_server_url, fhir_token, chw_id
    )


@mcp.tool()
async def dispatch_medication_bundle(
    ocr_result: dict,
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None,
    chw_id: str = None
) -> dict:
    """
    Build and dispatch a prescription FHIR bundle in one step.
    Same idempotency and retry guarantees as dispatch_fhir_bundle.
    """
    bundle = await medication_bundle_builder(ocr_result, patient_id)
    return await dispatch_bundle(
        bundle, patient_id, fhir_server_url, fhir_token, chw_id
    )


# ---------------------------------------------------------------------------
# Sync queue
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_sync_queue() -> dict:
    """
    Return sync queue status: pending, synced, failed counts.
    Background worker retries pending bundles every 30 seconds automatically.
    """
    return queue_status()


# ---------------------------------------------------------------------------
# Vision triage
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_clinical_image(
    patient_id: str,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    chw_id: str = None,
    context_hint: str = None,
    manual_history: str = None,
    overrides: dict = None,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """
    Vision triage: analyze a CHW photo for structured clinical data.
    Best for: malaria RDT strips, thermometers, wounds, glucose meters,
    pulse oximeters.
    Output schema matches extract_triage — use build_fhir_bundle downstream.
    For prescriptions and handwritten records use triage_document_image.
    """
    patient_history = await resolve_patient_context(
        patient_id, fhir_server_url, fhir_token
    )

    result = await _analyze_image(
        patient_id=patient_id,
        chw_id=chw_id,
        context_hint=context_hint,
        image_base64=image_base64,
        image_mime=image_mime,
        image_url=image_url,
        manual_history=manual_history,
        overrides=overrides,
        patient_history=patient_history if patient_history.get("source") != "none" else None
    )
    result["fhir_context_used"] = patient_history.get("source") in ["fhir", "fhir_error"]
    result["fhir_source"] = patient_history.get("source", "none")
    result["patient_history_used"] = patient_history.get("conditions", [])
    return result

# ---------------------------------------------------------------------------
# OCR bridge
# ---------------------------------------------------------------------------
@mcp.tool()
async def triage_document_image(
    patient_id: str,
    image_base64: str = None,
    image_mime: str = "image/jpeg",
    image_url: str = None,
    chw_id: str = None,
    fhir_server_url: str = None,
    fhir_token: str = None
) -> dict:
    """
    Two-stage OCR + clinical interpretation for health documents.
    Stage 1: Vision model transcribes all text from image (pure OCR).
    Stage 2: Text model interprets clinical meaning — infers symptoms
    from drug names: FeSO4 → anemia, Metformin → diabetes, ARVs → HIV.
    Best for: prescriptions, lab reports, referral slips, health records.
    Pass result to build_medication_bundle or dispatch_medication_bundle.
    """
    patient_history = await resolve_patient_context(
        patient_id, fhir_server_url, fhir_token
    )

    result = await triage_from_document_image(
        patient_id=patient_id,
        chw_id=chw_id,
        image_base64=image_base64,
        image_mime=image_mime,
        image_url=image_url,
        patient_history=patient_history if patient_history.get("source") != "none" else None
    )
    result["fhir_context_used"] = patient_history.get("source") in ["fhir", "fhir_error"]
    result["fhir_source"] = patient_history.get("source", "none")
    result["patient_history_used"] = patient_history.get("conditions", [])
    return result

# ---------------------------------------------------------------------------
# Startup + run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("[GroundWork] GROQ_API_KEY not set — check your .env file.")

    init_db()
    print("[GroundWork] SQLite queue initialised.")

    # Patch handle_initialize to inject FHIR context capability.
    # Must happen before mcp.run() — the handshake fires on first client connect.
    try:
        raw         = mcp._mcp_server
        orig_handle = raw.handle_initialize

        async def _patched_initialize(params):
            result = await orig_handle(params)
            try:
                if not hasattr(result.capabilities, "extensions") or result.capabilities.extensions is None:
                     result.capabilities.extensions = {}
                result.capabilities.extensions["ai.promptopinion/fhir-context"] = {
                    "scopes": [{"name": "patient/Patient.rs", "required": True}]
                }
                
                print("[GroundWork] FHIR context capability injected.")
            except Exception as inner:
                print(f"[GroundWork] FHIR context inject failed (non-fatal): {inner}")
            return result

        raw.handle_initialize = _patched_initialize
        print("[GroundWork] handle_initialize patched.")
    except AttributeError as e:
        print(f"[GroundWork] Could not patch handle_initialize (non-fatal): {e}")

    worker_thread = threading.Thread(
        target=_start_sync_worker,
        daemon=True,
        name="groundwork-sync-worker"
    )
    worker_thread.start()
    print("[GroundWork] Sync worker running in background thread.")

    port = int(os.environ.get("PORT", 8000))
    print(f"[GroundWork] Starting on port {port}.")
    mcp.run(transport="sse", host="0.0.0.0", port=port)