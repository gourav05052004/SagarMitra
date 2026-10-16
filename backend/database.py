"""
database.py — SQLite metadata store for the Maritime Object Detection pipeline.

Schema
------
detection_runs   : one row per uploaded image / inference call
detections       : one row per detected object (linked to detection_runs)

This module is intentionally dependency-free (stdlib only) so it works
inside the Docker container without extra packages.
"""

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

# ── Location of the database file ─────────────────────────────────────────────
# In Docker the Dockerfile sets DB_DIR=/data (host bind-mounted at /data).
# Locally it falls back to the directory containing this file.
_db_dir = os.environ.get("DB_DIR", str(Path(__file__).resolve().parent))
DB_PATH = Path(_db_dir) / "detections.db"

# ── DDL ────────────────────────────────────────────────────────────────────────
_CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS detection_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL UNIQUE,          -- UUID for the image
    timestamp           TEXT    NOT NULL,                 -- ISO-8601 UTC
    filename            TEXT    NOT NULL,                 -- S3 key / filename
    image_url           TEXT    NOT NULL,                 -- public S3 URL
    image_width         INTEGER,
    image_height        INTEGER,
    total_detections    INTEGER NOT NULL DEFAULT 0,
    swimmer_count       INTEGER NOT NULL DEFAULT 0,
    boat_count          INTEGER NOT NULL DEFAULT 0,
    jetski_count        INTEGER NOT NULL DEFAULT 0,
    lsa_count           INTEGER NOT NULL DEFAULT 0,       -- life saving appliance
    buoy_count          INTEGER NOT NULL DEFAULT 0,
    avg_confidence      REAL,                             -- mean score of all dets
    device              TEXT,                             -- 'cpu' or 'cuda:0'
    inference_time_s    REAL,                             -- wall-clock seconds
    cbnet_checkpoint    TEXT,
    yolo_checkpoint     TEXT
);
"""

_CREATE_DETECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,                         -- FK -> detection_runs.run_id
    category_id INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    score       REAL    NOT NULL,
    bbox_x      REAL    NOT NULL,
    bbox_y      REAL    NOT NULL,
    bbox_w      REAL    NOT NULL,
    bbox_h      REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES detection_runs(run_id)
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON detection_runs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_detections_run  ON detections(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_detections_cat  ON detections(category_id);",
]

# ── Category map (matches pipeline_config.json) ────────────────────────────────
CATEGORY_NAMES = {
    1: "swimmer",
    2: "boat",
    3: "jetski",
    4: "life saving appliance",
    5: "buoy",
}

COUNT_COLUMNS = {
    1: "swimmer_count",
    2: "boat_count",
    3: "jetski_count",
    4: "lsa_count",
    5: "buoy_count",
}


# ── Connection helper ──────────────────────────────────────────────────────────
@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safe for concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ─────────────────────────────────────────────────────────────
def init_db():
    """Create tables and indexes if they don't exist yet."""
    with _get_conn() as conn:
        conn.execute(_CREATE_RUNS_TABLE)
        conn.execute(_CREATE_DETECTIONS_TABLE)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
    print(f"[DB] SQLite metadata store ready -> {DB_PATH}")


# ── Write helpers ──────────────────────────────────────────────────────────────
def save_run(
    *,
    run_id: str,
    timestamp: str,
    filename: str,
    image_url: str,
    image_width: int,
    image_height: int,
    detections: list,
    device: str,
    inference_time_s: float,
    cbnet_checkpoint: str = "",
    yolo_checkpoint: str = "",
):
    """
    Persist one detection run and all its individual detections.

    Parameters
    ----------
    detections : list of dicts from pipeline_engine.predict()
        Each dict has keys: category_id, bbox [x,y,w,h], score
    """
    counts = {col: 0 for col in COUNT_COLUMNS.values()}
    scores = []

    for det in detections:
        cat = int(det.get("category_id", 0))
        col = COUNT_COLUMNS.get(cat)
        if col:
            counts[col] += 1
        scores.append(float(det.get("score", 0.0)))

    avg_conf = (sum(scores) / len(scores)) if scores else None

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO detection_runs (
                run_id, timestamp, filename, image_url,
                image_width, image_height,
                total_detections,
                swimmer_count, boat_count, jetski_count, lsa_count, buoy_count,
                avg_confidence,
                device, inference_time_s,
                cbnet_checkpoint, yolo_checkpoint
            ) VALUES (
                :run_id, :timestamp, :filename, :image_url,
                :image_width, :image_height,
                :total_detections,
                :swimmer_count, :boat_count, :jetski_count, :lsa_count, :buoy_count,
                :avg_confidence,
                :device, :inference_time_s,
                :cbnet_checkpoint, :yolo_checkpoint
            )
            """,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "filename": filename,
                "image_url": image_url,
                "image_width": image_width,
                "image_height": image_height,
                "total_detections": len(detections),
                **counts,
                "avg_confidence": avg_conf,
                "device": device,
                "inference_time_s": inference_time_s,
                "cbnet_checkpoint": cbnet_checkpoint,
                "yolo_checkpoint": yolo_checkpoint,
            },
        )

        rows = []
        for det in detections:
            cat = int(det.get("category_id", 0))
            bbox = det.get("bbox", [0, 0, 0, 0])
            rows.append(
                (
                    run_id,
                    cat,
                    CATEGORY_NAMES.get(cat, "unknown"),
                    float(det.get("score", 0.0)),
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )
            )

        conn.executemany(
            """
            INSERT INTO detections
                (run_id, category_id, category, score, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


# ── Read helpers ───────────────────────────────────────────────────────────────
def get_records(limit: int = 50, offset: int = 0) -> list:
    """Return paginated detection-run summaries (newest first)."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM detection_runs
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_total_runs() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM detection_runs").fetchone()[0]


def get_stats() -> dict:
    """Aggregate statistics across all runs — useful for research tables."""
    with _get_conn() as conn:
        run_row = conn.execute(
            """
            SELECT
                COUNT(*)                    AS total_runs,
                SUM(total_detections)       AS total_objects,
                SUM(swimmer_count)          AS total_swimmers,
                SUM(boat_count)             AS total_boats,
                SUM(jetski_count)           AS total_jetskis,
                SUM(lsa_count)              AS total_lsas,
                SUM(buoy_count)             AS total_buoys,
                AVG(avg_confidence)         AS overall_avg_confidence,
                AVG(inference_time_s)       AS avg_inference_time_s,
                MIN(timestamp)              AS earliest_run,
                MAX(timestamp)              AS latest_run
            FROM detection_runs
            """
        ).fetchone()

        per_class = conn.execute(
            """
            SELECT category, COUNT(*) AS count, AVG(score) AS avg_score
            FROM detections
            GROUP BY category
            ORDER BY count DESC
            """
        ).fetchall()

    return {
        "summary": dict(run_row),
        "per_class": [dict(r) for r in per_class],
    }


def get_all_runs_for_csv() -> list:
    """Return every run row for CSV export."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detection_runs ORDER BY timestamp ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_detections_for_run(run_id: str) -> list:
    """Return individual object detections for a specific run."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE run_id = ? ORDER BY score DESC",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]
