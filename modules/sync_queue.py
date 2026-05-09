import sqlite3
import json
import asyncio
import httpx
from datetime import datetime, timezone
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent.parent / "groundwork_queue.db"
MAX_QUEUE_SIZE = 500
MAX_ATTEMPTS = 5


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create queue table if it doesn't exist. Call once at startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key    TEXT UNIQUE NOT NULL,
                patient_id         TEXT NOT NULL,
                chw_id             TEXT,
                fhir_bundle        TEXT NOT NULL,
                fhir_server_url    TEXT,
                fhir_token         TEXT,
                status             TEXT DEFAULT 'pending',
                attempts           INTEGER DEFAULT 0,
                created_at         TEXT NOT NULL,
                last_attempted_at  TEXT
            )
        """)
        # Migrate: safely add any columns absent from older schema versions.
        # ALTER TABLE ADD COLUMN is idempotent-safe via the existence check.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(sync_queue)")}
        migrations = {
            "chw_id":            "TEXT",
            "fhir_server_url":   "TEXT",
            "fhir_token":        "TEXT",
            "attempts":          "INTEGER DEFAULT 0",
            "last_attempted_at": "TEXT",
        }
        for col, typedef in migrations.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE sync_queue ADD COLUMN {col} {typedef}")
        conn.commit()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class QueueFull(Exception):
    pass

# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def enqueue(
    bundle: dict,
    patient_id: str,
    idempotency_key: str,
    fhir_server_url: str = None,
    fhir_token: str = None,
    chw_id: str = None
) -> bool:
    """
    Add a bundle to the sync queue.
    Raises QueueFull if queue exceeds MAX_QUEUE_SIZE.
    Returns True on insert, False if idempotency_key already exists.
    """
    with _get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM sync_queue WHERE status = 'pending'"
        ).fetchone()[0]

        if count >= MAX_QUEUE_SIZE:
            raise QueueFull(f"Sync queue at capacity ({MAX_QUEUE_SIZE} pending items)")

        try:
            conn.execute("""
                INSERT INTO sync_queue
                    (idempotency_key, patient_id, chw_id, fhir_bundle,
                     fhir_server_url, fhir_token, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (
                idempotency_key,
                patient_id,
                chw_id,
                json.dumps(bundle),
                fhir_server_url,
                fhir_token,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate idempotency_key — already queued
            return False


def get_pending(limit: int = 10) -> list[dict]:
    """Return up to `limit` pending rows, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM sync_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def mark_synced(idempotency_key: str):
    with _get_conn() as conn:
        conn.execute("""
            UPDATE sync_queue
            SET status = 'synced', last_attempted_at = ?
            WHERE idempotency_key = ?
        """, (datetime.now(timezone.utc).isoformat(), idempotency_key))
        conn.commit()


# modules/sync_queue.py

def mark_failed(idempotency_key: str, attempts: int):
    """
    Mark as failed ONLY when attempts >= MAX_ATTEMPTS.
    Otherwise keep as pending.
    Raises RuntimeError if no matching row is found.
    """
    new_status = "failed" if attempts >= MAX_ATTEMPTS else "pending"

    with _get_conn() as conn:
        cur = conn.execute("""
            UPDATE sync_queue
            SET status = ?, attempts = ?, last_attempted_at = ?
            WHERE idempotency_key = ?
        """, (
            new_status,
            attempts,
            datetime.now(timezone.utc).isoformat(),
            idempotency_key
        ))
        conn.commit()

        if cur.rowcount == 0:
            raise RuntimeError(
                f"mark_failed: no row found for idempotency_key={idempotency_key!r}"
            )


def queue_status() -> dict:
    """Return counts by status — exposed as MCP tool."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM sync_queue
            GROUP BY status
        """).fetchall()
    counts = {r["status"]: r["count"] for r in rows}
    return {
        "pending": counts.get("pending", 0),
        "synced":  counts.get("synced", 0),
        "failed":  counts.get("failed", 0),
        "total":   sum(counts.values())
    }


def get_failed(limit: int = 20) -> list[dict]:
    """Return failed rows for manual review."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT id, idempotency_key, patient_id, chw_id,
                   attempts, created_at, last_attempted_at
            FROM sync_queue
            WHERE status = 'failed'
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Retry-wrapped POST (mirrors action_dispatcher — no circular import)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True
)
async def _worker_post(url: str, bundle: dict, headers: dict) -> int:
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.post(url, json=bundle, headers=headers)
        response.raise_for_status()
        return response.status_code

# ---------------------------------------------------------------------------
# Background sync worker
# ---------------------------------------------------------------------------

async def sync_worker(interval_seconds: int = 30):
    """
    Poll the queue every `interval_seconds`.
    Retry pending bundles. Mark synced or increment failure count.
    Runs as an asyncio background task — start with asyncio.create_task(sync_worker()).
    """
    init_db()

    while True:
        pending = get_pending(limit=10)

        for row in pending:
            key = row["idempotency_key"]
            bundle = json.loads(row["fhir_bundle"])
            base_url = (row["fhir_server_url"] or "https://r4.smarthealthit.org").rstrip("/")

            headers = {
                "Content-Type":    "application/fhir+json",
                "Accept":          "application/fhir+json",
                "Idempotency-Key": key,
                "X-CHW-ID":        row["chw_id"] or "unknown",
                "X-Source":        "groundwork-sync-worker"
            }
            if row["fhir_token"]:
                headers["Authorization"] = f"Bearer {row['fhir_token']}"

            try:
                await _worker_post(f"{base_url}/", bundle, headers)
                mark_synced(key)

            except Exception:
                attempts = row["attempts"]
                attempts += 1
                mark_failed(key, attempts)

        await asyncio.sleep(interval_seconds)