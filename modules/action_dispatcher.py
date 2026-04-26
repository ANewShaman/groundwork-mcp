import hashlib
import time
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)
from dotenv import load_dotenv

load_dotenv()

from modules.sync_queue import enqueue, QueueFull

# ---------------------------------------------------------------------------
# Idempotency key generation
# ---------------------------------------------------------------------------

def _make_idempotency_key(patient_id: str, chw_id: str = None) -> str:
    """
    Generate a deterministic idempotency key.
    Time-bucketed to 5-minute windows — prevents duplicate records on retry
    while allowing a genuine new submission after 5 minutes.
    """
    bucket = round(time.time() / 300)
    key_source = f"{patient_id}:{chw_id or 'unknown'}:{bucket}"
    return hashlib.sha256(key_source.encode()).hexdigest()[:32]

# ---------------------------------------------------------------------------
# Retry-wrapped POST — only retries on network/timeout errors, not 4xx
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True
)
async def _post_bundle(
    client: httpx.AsyncClient,
    url: str,
    bundle: dict,
    headers: dict
) -> httpx.Response:
    response = await client.post(url, json=bundle, headers=headers, timeout=12.0)
    response.raise_for_status()
    return response

# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

async def dispatch_bundle(
    fhir_bundle: dict,
    patient_id: str,
    fhir_server_url: str = None,
    fhir_token: str = None,
    chw_id: str = None
) -> dict:
    """
    Dispatch a FHIR Transaction Bundle to the target server.

    On success:  returns status=dispatched
    On failure:  enqueues to SQLite sync queue, returns status=queued
    On 4xx:      returns status=rejected (bad data, not retried)
    """
    from modules.fhir_ops import FALLBACK_FHIR_URL

    base_url = (fhir_server_url or FALLBACK_FHIR_URL).rstrip("/")
    idempotency_key = _make_idempotency_key(patient_id, chw_id)

    headers = {
        "Content-Type":    "application/fhir+json",
        "Accept":          "application/fhir+json",
        "Idempotency-Key": idempotency_key,
        "X-CHW-ID":        chw_id or "unknown",
        "X-Source":        "groundwork-mcp"
    }
    if fhir_token:
        headers["Authorization"] = f"Bearer {fhir_token}"

    retries_used = 0

    try:
        async with httpx.AsyncClient() as client:
            response = await _post_bundle(client, f"{base_url}/", fhir_bundle, headers)

        return {
            "status":              "dispatched",
            "idempotency_key":     idempotency_key,
            "fhir_response_status": response.status_code,
            "patient_id":          patient_id,
            "chw_id":              chw_id,
            "fhir_server":         base_url,
            "retries_used":        retries_used
        }

    except httpx.HTTPStatusError as e:
        # 4xx = bad data, do not retry or queue
        if 400 <= e.response.status_code < 500:
            return {
                "status":          "rejected",
                "idempotency_key": idempotency_key,
                "reason":          f"HTTP {e.response.status_code} — bad request, not retried",
                "patient_id":      patient_id,
                "chw_id":          chw_id
            }
        # 5xx — fall through to queue
        reason = f"HTTP {e.response.status_code} after retries"

    except (httpx.TimeoutException, httpx.NetworkError, RetryError) as e:
        reason = f"Network failure after 3 retries: {type(e).__name__}"

    except Exception as e:
        reason = f"Unexpected error: {str(e)}"

    # --- Fallback: enqueue for later sync ---
    try:
        enqueue(
            bundle=fhir_bundle,
            patient_id=patient_id,
            idempotency_key=idempotency_key,
            fhir_server_url=base_url,
            fhir_token=fhir_token,
            chw_id=chw_id
        )
        queue_status = "queued"
    except QueueFull:
        queue_status = "queue_full"

    return {
        "status":          queue_status,
        "idempotency_key": idempotency_key,
        "reason":          reason,
        "patient_id":      patient_id,
        "chw_id":          chw_id,
        "message":         "Bundle saved locally. Will sync when connection restores." if queue_status == "queued"
                           else "Queue full. Manual review required."
    }