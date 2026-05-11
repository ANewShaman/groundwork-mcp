MEDICATION_NORMALIZATION = {
    "feso4":         "ferrous sulfate",
    "fe so4":        "ferrous sulfate",
    "glucophage":    "metformin",
    "ventolin":      "salbutamol",
    "albuterol":     "salbutamol",
    "augmentin":     "amoxicillin",
    "septrin":       "cotrimoxazole",
    "acetaminophen": "paracetamol",
}

_SCALAR_STRING_FIELDS = (
    "language_detected",
    "document_type",
    "image_type",
    "rdt_result",
    "code_status",
)


def _s(v) -> str:
    return v.strip().lower() if isinstance(v, str) else v


def _normalize_med_name(name: str) -> str:
    key = name.strip().lower()
    return MEDICATION_NORMALIZATION.get(key, key)


def normalize(result: dict) -> dict:
    updated = result.copy()

    for field in _SCALAR_STRING_FIELDS:
        if isinstance(updated.get(field), str):
            updated[field] = _s(updated[field])

    if isinstance(updated.get("symptoms"), list):
        updated["symptoms"] = [_s(s) for s in updated["symptoms"]]

    if isinstance(updated.get("medications"), list):
        meds = []
        for med in updated["medications"]:
            m = med.copy()
            if isinstance(m.get("name"), str):
                m["name"] = _normalize_med_name(m["name"])
            if isinstance(m.get("abbreviation"), str):
                m["abbreviation"] = m["abbreviation"].strip().lower()
            meds.append(m)
        updated["medications"] = meds

    return updated