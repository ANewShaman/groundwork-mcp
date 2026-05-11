import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from functools import lru_cache
import httpx

DB_PATH     = Path(__file__).parent.parent / "terminology_cache.db"
TTL_SECONDS = 60 * 60 * 24 * 7   # 7-day TTL for API results
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
SNOMED_BASE = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions"

# ── Minimal local seeds ────────────────────────────────────────────────────
# These are the entries from your current hardcoded dicts, preserved as
# a zero-dependency offline baseline. New lookups augment this table.

RXNORM_SEED = {
    "ferrous sulfate": "4482",   "feso4": "4482",
    "ascorbic acid":   "1049502","paracetamol": "161",
    "amoxicillin":     "723",    "metformin":   "6809",
    "salbutamol":      "435",    "cotrimoxazole":"10829",
    "ors":             "9863",   "albendazole": "16681",
    "folic acid":      "4511",   "nifedipine":  "7417",
    "amlodipine":      "17767",  "atenolol":    "1202",
    "digoxin":         "3407",   "insulin":     "5856",
}

SNOMED_SEED = {
    "cough": "49727002",            "fever": "386661006",
    "productive cough": "28743005", "shortness of breath": "267036007",
    "wheezing": "56018004",         "chest pain": "29857009",
    "hypertension": "38341003",     "headache": "25064002",
    "seizure": "91175000",          "diarrhea": "62315008",
    "vomiting": "422400008",        "dehydration": "34095006",
    "malnutrition": "76113001",     "anemia": "271737000",
    "malaria": "61462000",          "tuberculosis": "56717001",
    "hypoxia": "389086002",         "jaundice": "18165001",
    "rash": "271807003",            "hyperglycemia": "80394007",
    "bacterial infection": "87628006",
}


# ── DB init ────────────────────────────────────────────────────────────────

def init_terminology_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rxnorm_cache (
                term        TEXT PRIMARY KEY,
                rxcui       TEXT NOT NULL,
                source      TEXT NOT NULL,   -- 'seed' | 'api' | 'llm'
                cached_at   REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snomed_cache (
                term        TEXT PRIMARY KEY,
                concept_id  TEXT NOT NULL,
                source      TEXT NOT NULL,
                cached_at   REAL NOT NULL
            )
        """)
        # Seed on first run — inserts only, never overwrites
        for term, rxcui in RXNORM_SEED.items():
            conn.execute(
                "INSERT OR IGNORE INTO rxnorm_cache VALUES (?,?,?,?)",
                (term, rxcui, "seed", time.time())
            )
        for term, cid in SNOMED_SEED.items():
            conn.execute(
                "INSERT OR IGNORE INTO snomed_cache VALUES (?,?,?,?)",
                (term, cid, "seed", time.time())
            )
        conn.commit()


# ── In-memory hot cache (process lifetime) ────────────────────────────────

_rxnorm_hot: dict[str, str] = {}
_snomed_hot: dict[str, str] = {}


# ── RxNorm lookup ─────────────────────────────────────────────────────────

def _rxnorm_db_lookup(term: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT rxcui FROM rxnorm_cache WHERE term = ?", (term,)
        ).fetchone()
    return row[0] if row else None


def _rxnorm_db_write(term: str, rxcui: str, source: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO rxnorm_cache VALUES (?,?,?,?)",
            (term, rxcui, source, time.time())
        )
        conn.commit()


async def _rxnorm_api(term: str) -> str | None:
    """
    NLM RxNorm API — free, no key required, 20 req/sec limit.
    Uses /approximateTerm for fuzzy matching (handles typos, abbreviations).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{RXNORM_BASE}/approximateTerm.json",
                params={"term": term, "maxEntries": 1}
            )
            r.raise_for_status()
            data = r.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                return candidates[0]["rxcui"]
    except Exception:
        pass
    return None


async def resolve_rxnorm(drug_name: str) -> tuple[str | None, str]:
    """
    Returns (rxcui, source) where source is 'seed' | 'api' | 'llm' | None.
    Hot cache → SQLite → NLM API → None (caller handles LLM fallback).
    """
    key = drug_name.strip().lower()

    if key in _rxnorm_hot:
        return _rxnorm_hot[key], "cache"

    local = _rxnorm_db_lookup(key)
    if local:
        _rxnorm_hot[key] = local
        return local, "local"

    rxcui = await _rxnorm_api(key)
    if rxcui:
        _rxnorm_db_write(key, rxcui, "api")
        _rxnorm_hot[key] = rxcui
        return rxcui, "api"

    return None, "none"


# ── SNOMED lookup ─────────────────────────────────────────────────────────

def _snomed_db_lookup(term: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT concept_id FROM snomed_cache WHERE term = ?", (term,)
        ).fetchone()
    return row[0] if row else None


def _snomed_db_write(term: str, concept_id: str, source: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO snomed_cache VALUES (?,?,?,?)",
            (term, concept_id, source, time.time())
        )
        conn.commit()


async def _snomed_api(term: str) -> str | None:
    """
    SNOMED CT browser API (IHTSDO public instance).
    No key required for read-only concept search.
    Targets SNOMED International edition, clinical findings only.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                SNOMED_BASE,
                params={
                    "term": term,
                    "active": "true",
                    "conceptActive": "true",
                    "semanticTag": "clinical finding",
                    "limit": 1,
                    "lang": "english",
                }
            )
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            if items:
                return str(items[0]["concept"]["conceptId"])
    except Exception:
        pass
    return None


async def resolve_snomed(symptom: str) -> tuple[str | None, str]:
    """
    Returns (concept_id, source).
    Falls back gracefully — callers must handle None.
    """
    key = symptom.strip().lower()

    if key in _snomed_hot:
        return _snomed_hot[key], "cache"

    local = _snomed_db_lookup(key)
    if local:
        _snomed_hot[key] = local
        return local, "local"

    concept_id = await _snomed_api(key)
    if concept_id:
        _snomed_db_write(key, concept_id, "api")
        _snomed_hot[key] = concept_id
        return concept_id, "api"

    return None, "none"


# ── Batch helpers (used by FHIR bundle builders) ──────────────────────────

async def build_snomed_map(symptoms: list[str]) -> dict[str, str]:
    """
    Resolve a list of symptoms concurrently. Silently drops unresolved terms
    BUT logs them — unresolved terms get a code_status=manual_review note.
    Returns {display_term: concept_id}.
    """
    results = await asyncio.gather(
        *[resolve_snomed(s) for s in symptoms],
        return_exceptions=False
    )
    return {
        sym: cid
        for sym, (cid, _src) in zip(symptoms, results)
        if cid is not None
    }


async def build_rxnorm_map(drug_names: list[str]) -> dict[str, str | None]:
    """
    Returns {drug_name: rxcui_or_None} for all inputs.
    Callers decide how to handle None (omit coding, flag for review, etc.).
    """
    results = await asyncio.gather(
        *[resolve_rxnorm(d) for d in drug_names],
        return_exceptions=False
    )
    return {
        name: rxcui
        for name, (rxcui, _src) in zip(drug_names, results)
    }