MEDICATION_MAP = {
    # antianemic
    "ferrous sulfate": "antianemic",
    "iron tablet":     "antianemic",
    "folic acid":      "antianemic",
    # antidiabetic
    "metformin":       "antidiabetic_oral",
    "insulin":         "antidiabetic_injectable",
    # antihypertensive
    "nifedipine":      "antihypertensive",
    "amlodipine":      "antihypertensive",
    "atenolol":        "antihypertensive",
    # cardiac
    "digoxin":         "cardiac_glycoside",
    # antibiotic
    "amoxicillin":     "antibiotic",
    "cotrimoxazole":   "antibiotic",
    # antiretroviral
    "arv":             "antiretroviral",
    "antiretroviral":  "antiretroviral",
    # bronchodilator
    "salbutamol":      "bronchodilator",
    # analgesic/antipyretic
    "paracetamol":     "analgesic_antipyretic",
    # rehydration
    "ors":             "rehydration_therapy",
    "oral rehydration salts": "rehydration_therapy",
    # anthelmintic
    "albendazole":     "anthelmintic",
    "mebendazole":     "anthelmintic",
    # vitamin/supplement
    "ascorbic acid":   "vitamin_supplement",
}
# Aliases removed (feso4, glucophage, ventolin, albuterol, augmentin,
# septrin, acetaminophen) — normalize.py canonicalizes these before
# run_inference() is called, so they were unreachable dead code.

TEST_MAP = {
    "rdt":         "rapid_diagnostic_test",
    "malaria rdt": "rapid_diagnostic_test",
    "hiv rdt":     "rapid_diagnostic_test",
    "rbs":         "blood_glucose_test",
    "fbs":         "fasting_blood_glucose_test",
    "hb":          "hemoglobin_test",
    "hemoglobin":  "hemoglobin_test",
    "muac":        "mid_upper_arm_circumference",
    "afb":         "acid_fast_bacilli_smear",
    "spo2":        "pulse_oximetry",
    "bp":          "blood_pressure_measurement",
}


def run_inference(result: dict) -> dict:
    updated = result.copy()

    # medication_categories + medications_normalized
    categories = []
    normalized_names = []
    for med in updated.get("medications", []):
        abbrev = (med.get("abbreviation") or "").lower().strip()
        name   = (med.get("name") or "").lower().strip()
        cat    = MEDICATION_MAP.get(abbrev) or MEDICATION_MAP.get(name)
        if cat and cat not in categories:
            categories.append(cat)
        canonical = name or abbrev
        if canonical and canonical not in normalized_names:
            normalized_names.append(canonical)

    if categories:
        updated["medication_categories"] = categories
    if normalized_names:
        updated["medications_normalized"] = normalized_names

    # observation_tags — explicit observations only
    obs_tags = []
    for obs in updated.get("observations", []):
        key = (obs.get("name") or "").lower().strip()
        tag = TEST_MAP.get(key)
        if tag and tag not in obs_tags:
            obs_tags.append(tag)

    rdt_raw = (updated.get("rdt_result") or "").lower().strip()
    if rdt_raw:
        tag = f"rdt_result:{rdt_raw}"
        if tag not in obs_tags:
            obs_tags.append(tag)

    if obs_tags:
        updated["observation_tags"] = obs_tags

    # workflow_flags
    flags = []
    if len(updated.get("medications", [])) > 1:
        flags.append("multiple_medications")
    if rdt_raw in ("positive", "reactive", "invalid"):
        flags.append("abnormal_test_present")
    vitals = updated.get("vitals") or {}
    if any(v is not None for v in vitals.values()):
        flags.append("vitals_recorded")
    if updated.get("observations"):
        flags.append("observations_present")

    if flags:
        updated["workflow_flags"] = flags

    return updated