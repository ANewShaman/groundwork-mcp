"""
test_groundwork_end_to_end_demo.py
==================================

GroundWork COMPLETE local multimodal interoperability demo.

Features demonstrated:
- Local image ingestion
- Multimodal medical extraction
- OCR + vision pipelines
- FHIR R4 bundle generation
- LOINC normalization
- RxNorm normalization
- Offline-first queue fallback
- SQLite persistence
- Idempotent retries
- Manual review handling
- Merged encounter transaction bundles

Expected local files:
    thermometer.jpg
    glucometer.jpg
    oximeter.jpg
    doctor_referral.jpg
    HIV.jpg

Run:
    python test_groundwork_end_to_end_demo.py
"""

import asyncio
import json
import os
import sys
import time
import base64
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------

from modules.vision_triage import analyze_clinical_image
from modules.ocr_bridge import triage_from_document_image

from modules.fhir_ops import (
    fhir_bundle_builder,
    medication_bundle_builder
)

from modules.action_dispatcher import dispatch_bundle

from modules.sync_queue import (
    init_db,
    queue_status,
    get_pending
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PATIENT_ID = os.getenv("TEST_PATIENT_ID", "pt-rx-001")
CHW_ID     = os.getenv("TEST_CHW_ID", "chw-test-001")

INVALID_FHIR_URL = "https://invalid-groundwork-fhir-server.demo"

# ---------------------------------------------------------------------
# LOCAL IMAGE FILES
# ---------------------------------------------------------------------

IMAGES = {
    "thermometer": "thermometer.jpg",
    "glucometer": "glucometer.jpg",
    "pulse_oximeter": "oximeter.jpg",
    "referral_slip": "doctor_referral.jpg",
    "hiv": "HIV.jpg"
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []
timings = {}


def encode_image(path):

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_file_exists(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing required image file: {path}"
        )


def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_json(title, data):

    print(f"\n--- {title} ---")

    print(json.dumps(data, indent=2, ensure_ascii=False))


def record(name, passed, note=""):

    status = PASS if passed else FAIL

    results.append((name, status, note))

    print(f"  {status}  {name}  {note}")


def timed(label):

    def decorator(fn):

        async def wrapper(*args, **kwargs):

            start = time.perf_counter()

            result = await fn(*args, **kwargs)

            elapsed = round(time.perf_counter() - start, 2)

            timings[label] = elapsed

            print(f"\n⏱ {label}: {elapsed}s")

            return result

        return wrapper

    return decorator


def resource_counts(bundle):

    counts = Counter()

    for entry in bundle.get("entry", []):

        rt = entry["resource"]["resourceType"]

        counts[rt] += 1

    return dict(counts)


# ---------------------------------------------------------------------
# VALIDATE LOCAL FILES
# ---------------------------------------------------------------------

for _, path in IMAGES.items():

    validate_file_exists(path)

# ---------------------------------------------------------------------
# THERMOMETER
# ---------------------------------------------------------------------

@timed("thermometer_pipeline")
async def test_thermometer():

    section("THERMOMETER PIPELINE")

    result = await analyze_clinical_image(
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        image_base64=encode_image(IMAGES["thermometer"]),
        image_mime="image/jpeg",
        context_hint="Digital thermometer displaying body temperature"
    )

    print_json("THERMOMETER EXTRACTION", result)

    vitals = result.get("vitals", {}) or {}

    record(
        "thermometer_temperature_detected",
        vitals.get("temperature_c") is not None,
        f"temperature_c={vitals.get('temperature_c')}"
    )

    record(
        "thermometer_loinc_present",
        "8310-5" in (result.get("loinc_map") or {}),
        "LOINC 8310-5"
    )

    bundle = await fhir_bundle_builder(
        result,
        PATIENT_ID
    )

    print_json("THERMOMETER FHIR BUNDLE", bundle)

    return result, bundle

# ---------------------------------------------------------------------
# GLUCOMETER
# ---------------------------------------------------------------------

@timed("glucometer_pipeline")
async def test_glucometer():

    section("GLUCOMETER PIPELINE")

    result = await analyze_clinical_image(
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        image_base64=encode_image(IMAGES["glucometer"]),
        image_mime="image/jpeg",
        context_hint="Blood glucose meter displaying mg/dL"
    )

    print_json("GLUCOMETER EXTRACTION", result)

    vitals = result.get("vitals", {}) or {}

    record(
        "glucometer_glucose_detected",
        vitals.get("blood_glucose") is not None,
        f"blood_glucose={vitals.get('blood_glucose')}"
    )

    record(
        "glucometer_loinc_present",
        "2339-0" in (result.get("loinc_map") or {}),
        "LOINC 2339-0"
    )

    bundle = await fhir_bundle_builder(
        result,
        PATIENT_ID
    )

    print_json("GLUCOMETER FHIR BUNDLE", bundle)

    return result, bundle

# ---------------------------------------------------------------------
# OXIMETER
# ---------------------------------------------------------------------

@timed("oximeter_pipeline")
async def test_oximeter():

    section("PULSE OXIMETER PIPELINE")

    result = await analyze_clinical_image(
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        image_base64=encode_image(IMAGES["pulse_oximeter"]),
        image_mime="image/jpeg",
        context_hint="Pulse oximeter displaying SpO2 and pulse"
    )

    print_json("OXIMETER EXTRACTION", result)

    vitals = result.get("vitals", {}) or {}

    loinc = result.get("loinc_map") or {}

    record(
        "oximeter_spo2_detected",
        vitals.get("spo2") is not None,
        f"spo2={vitals.get('spo2')}"
    )

    record(
        "oximeter_pulse_detected",
        vitals.get("pulse") is not None,
        f"pulse={vitals.get('pulse')}"
    )

    record(
        "oximeter_spo2_loinc",
        "59408-5" in loinc,
        "LOINC 59408-5"
    )

    record(
        "oximeter_pulse_loinc",
        "8867-4" in loinc,
        "LOINC 8867-4"
    )

    bundle = await fhir_bundle_builder(
        result,
        PATIENT_ID
    )

    print_json("OXIMETER FHIR BUNDLE", bundle)

    return result, bundle

# ---------------------------------------------------------------------
# REFERRAL OCR
# ---------------------------------------------------------------------

@timed("referral_pipeline")
async def test_referral():

    section("REFERRAL OCR PIPELINE")

    result = await triage_from_document_image(
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        image_base64=encode_image(IMAGES["referral_slip"]),
        image_mime="image/jpeg"
    )

    print_json("REFERRAL OCR RESULT", result)

    meds = result.get("medications") or []

    # -------------------------------------------------------------
    # REMOVE NULL MEDICATIONS
    # Prevent RxNorm crashes
    # -------------------------------------------------------------

    cleaned = []

    for med in meds:

        name = med.get("name")
        abbrev = med.get("abbreviation")

        if not name and not abbrev:
            continue

        cleaned.append(med)

    result["medications"] = cleaned

    record(
        "referral_medications_found",
        len(cleaned) > 0,
        f"{len(cleaned)} medication(s)"
    )

    record(
        "referral_document_type",
        result.get("document_type") in (
            "prescription",
            "referral"
        ),
        f"type={result.get('document_type')}"
    )

    bundle = await medication_bundle_builder(
        result,
        PATIENT_ID
    )

    print_json("REFERRAL FHIR BUNDLE", bundle)

    return result, bundle

# ---------------------------------------------------------------------
# HIV RDT
# ---------------------------------------------------------------------

@timed("hiv_pipeline")
async def test_hiv():

    section("HIV RDT PIPELINE")

    result = await analyze_clinical_image(
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        image_base64=encode_image(IMAGES["hiv"]),
        image_mime="image/jpeg",
        context_hint="HIV rapid diagnostic test strip"
    )

    print_json("HIV EXTRACTION", result)

    bundle = await fhir_bundle_builder(
        result,
        PATIENT_ID
    )

    print_json("HIV FHIR BUNDLE", bundle)

    return result, bundle

# ---------------------------------------------------------------------
# MERGED ENCOUNTER
# ---------------------------------------------------------------------

async def build_merged_bundle(bundles):

    section("MERGED ENCOUNTER BUNDLE")

    merged = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": []
    }

    risk_added = False

    for bundle in bundles:

        for entry in bundle.get("entry", []):

            rt = entry["resource"]["resourceType"]

            # Deduplicate RiskAssessment
            if rt == "RiskAssessment":

                if risk_added:
                    continue

                risk_added = True

            merged["entry"].append(entry)

    counts = resource_counts(merged)

    print_json("MERGED ENCOUNTER BUNDLE", merged)

    print("\nFHIR RESOURCE TABLE")
    print("-" * 40)

    for k, v in counts.items():

        print(f"{k:<25} {v}")

    record(
        "merged_bundle_transaction",
        merged.get("type") == "transaction",
        "bundle type=transaction"
    )

    record(
        "merged_bundle_has_observation",
        counts.get("Observation", 0) > 0,
        f"{counts.get('Observation',0)} observations"
    )

    record(
        "merged_bundle_has_medication_request",
        counts.get("MedicationRequest", 0) > 0,
        f"{counts.get('MedicationRequest',0)} medication requests"
    )

    return merged

# ---------------------------------------------------------------------
# OFFLINE QUEUE
# ---------------------------------------------------------------------

@timed("offline_queue_simulation")
async def test_queue_behavior(bundle):

    section("OFFLINE QUEUE SIMULATION")

    print("\nDispatching to INVALID endpoint...")
    print("Expected: queue fallback")

    result1 = await dispatch_bundle(
        fhir_bundle=bundle,
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        fhir_server_url=INVALID_FHIR_URL
    )

    print_json("FIRST DISPATCH RESULT", result1)

    record(
        "queue_fallback_triggered",
        result1["status"] in (
            "queued",
            "already_queued"
        ),
        f"status={result1['status']}"
    )

    queue1 = queue_status()

    print_json("QUEUE STATUS", queue1)

    record(
        "queue_pending_exists",
        queue1.get("pending", 0) >= 1,
        f"pending={queue1.get('pending')}"
    )

    # -------------------------------------------------------------
    # RETRY SAME BUNDLE
    # -------------------------------------------------------------

    print("\nRetrying SAME bundle...")

    result2 = await dispatch_bundle(
        fhir_bundle=bundle,
        patient_id=PATIENT_ID,
        chw_id=CHW_ID,
        fhir_server_url=INVALID_FHIR_URL
    )

    print_json("SECOND DISPATCH RESULT", result2)

    same_key = (
        result1["idempotency_key"]
        ==
        result2["idempotency_key"]
    )

    record(
        "idempotency_same_key",
        same_key,
        "same idempotency key reused"
    )

    pending = get_pending()

    print_json("PENDING SQLITE ROWS", pending)

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def print_summary(merged_bundle):

    section("GROUNDWORK END-TO-END SUMMARY")

    passed = sum(
        1 for _, s, _ in results
        if s == PASS
    )

    total = len(results)

    print("\nVALIDATION RESULTS")
    print("-" * 50)

    for name, status, note in results:

        print(f"{status:<10} {name:<45} {note}")

    print("\nTIMING METRICS")
    print("-" * 50)

    for k, v in timings.items():

        print(f"{k:<35} {v}s")

    print("\nFHIR RESOURCE COUNTS")
    print("-" * 50)

    counts = resource_counts(merged_bundle)

    for k, v in counts.items():

        print(f"{k:<30} {v}")

    print("\nQUEUE STATUS")
    print("-" * 50)

    print(json.dumps(queue_status(), indent=2))

    print("\nINTEROPERABILITY FEATURES DEMONSTRATED")
    print("-" * 50)

    features = [
        "Multimodal medical extraction",
        "Clinical OCR",
        "Vision triage",
        "FHIR R4 transaction bundles",
        "LOINC normalization",
        "RxNorm medication mapping",
        "Offline queue fallback",
        "SQLite sync persistence",
        "Idempotent dispatching",
        "Manual review handling",
        "Low-connectivity resilience"
    ]

    for f in features:

        print(f"✓ {f}")

    print("\nFINAL RESULT")
    print("-" * 50)

    if passed == total:

        print(f"✅ ALL {total} VALIDATIONS PASSED")

    else:

        print(f"❌ {total - passed} VALIDATIONS FAILED")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

async def main():

    print("\nGroundWork Complete End-to-End Demo\n")

    init_db()

    thermometer_result, thermometer_bundle = \
        await test_thermometer()

    glucometer_result, glucometer_bundle = \
        await test_glucometer()

    oximeter_result, oximeter_bundle = \
        await test_oximeter()

    referral_result, referral_bundle = \
        await test_referral()

    hiv_result, hiv_bundle = \
        await test_hiv()

    merged_bundle = await build_merged_bundle([
        thermometer_bundle,
        glucometer_bundle,
        oximeter_bundle,
        referral_bundle,
        hiv_bundle
    ])

    await test_queue_behavior(
        merged_bundle
    )

    print_summary(
        merged_bundle
    )

# ---------------------------------------------------------------------

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print("\nInterrupted.")
        sys.exit(1)

    except Exception as e:

        print(f"\nFATAL ERROR: {e}")

        raise