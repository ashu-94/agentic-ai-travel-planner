"""Daily run cap for the public demo.

Keeps a per-day tally in Postgres so a public tester can't drain the Groq
free tier. Self-contained: opens its own short connection using DATABASE_URL,
so it is unaffected by how the checkpointer manages its connection.
"""

import datetime
import psycopg

from config import DATABASE_URL

# How many fresh plans may be started per day.
DAILY_LIMIT = 15


def _today() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _ensure_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS demo_usage (
            day   TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def usage_today() -> int:
    """Return how many plans have been started today. 0 if unlimited (no DB)."""
    if not DATABASE_URL:
        return 0
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            _ensure_table(conn)
            row = conn.execute(
                "SELECT count FROM demo_usage WHERE day = %s", (_today(),)
            ).fetchone()
            return row[0] if row else 0
    except Exception as e:
        # Never let a counter hiccup block the app — fail open.
        print(f"[usage_limit] read failed, allowing run: {e!r}")
        return 0


def limit_reached() -> bool:
    if not DATABASE_URL:
        
        return False
    return usage_today() >= DAILY_LIMIT


def record_run() -> None:
    """Add one to today's tally. Called only when a fresh plan is started."""
    if not DATABASE_URL:
        return
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            _ensure_table(conn)
            conn.execute(
                """
                INSERT INTO demo_usage (day, count) VALUES (%s, 1)
                ON CONFLICT (day) DO UPDATE SET count = demo_usage.count + 1
                """,
                (_today(),),
            )
    except Exception as e:
        print(f"[usage_limit] record failed: {e!r}")