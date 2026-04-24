import asyncio
import json
from modules.triage import triage_extractor
from modules.fhir_ops import get_patient_context

async def run_tests():
    print("=" * 50)
    print("TEST 1: Control (no history)")
    print("=" * 50)
    r = await triage_extractor("Patient has cough for 2 weeks", "pt-1")
    print(f"Score:    {r.get('severity_score')}")
    print(f"Referral: {r.get('referral_flag')}")
    print(f"Evidence: {r.get('evidence_cited')}")

    print()
    print("=" * 50)
    print("TEST 2: COPD Upgrade (the demo)")
    print("=" * 50)
    history = {
        "conditions": ["Chronic obstructive pulmonary disease"],
        "patient_id": "pt-1"
    }
    r = await triage_extractor("Patient has cough for 2 weeks", "pt-1", history)
    print(f"Score:    {r.get('severity_score')}")
    print(f"Referral: {r.get('referral_flag')}")
    print(f"Evidence: {r.get('evidence_cited')}")
    print(f"History:  {r.get('patient_history_used', history['conditions'])}")

    print()
    print("=" * 50)
    print("TEST 3: FHIR sandbox pull (no auth needed)")
    print("=" * 50)
    r = await get_patient_context("87a339d0-8cae-418e-89c7-8651e6aab3c6")
    print(json.dumps(r, indent=2))


asyncio.run(run_tests())