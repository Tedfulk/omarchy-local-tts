"""
SQLite command history for Omarchy Local TTS.

Stores the last 30 days of TTS commands so history persists across restarts.
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path.home() / ".config" / "omarchy-local-tts" / "history.db"

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    """Create a connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the command_history table if it doesn't exist and run cleanup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = _connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    voice_id TEXT NOT NULL,
                    voice_name TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()
    cleanup_old_entries()


def log_command(
    text: str,
    voice_id: str,
    voice_name: str,
    char_count: int,
    chunk_count: int,
) -> None:
    """Insert a command history row."""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO command_history (text, voice_id, voice_name, char_count, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (text, voice_id, voice_name, char_count, chunk_count),
            )
            conn.commit()
        finally:
            conn.close()


def get_history(
    days: int = 30,
    voice_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Query command history with optional filters.

    Returns rows newest-first.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    query = "SELECT * FROM command_history WHERE created_at >= ?"
    params: list = [cutoff]

    if voice_id:
        query += " AND voice_id = ?"
        params.append(voice_id)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def get_last_command() -> dict | None:
    """Return the most recent command history entry, or None if empty."""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM command_history ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def cleanup_old_entries(days: int = 30) -> int:
    """Delete entries older than the given number of days. Returns rows deleted."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _lock:
        conn = _connect()
        try:
            cursor = conn.execute(
                "DELETE FROM command_history WHERE created_at < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
