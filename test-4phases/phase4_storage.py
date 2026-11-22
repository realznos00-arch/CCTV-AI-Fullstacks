"""
phase4_storage.py — Phase 4: Storage Leg
=========================================
Event Intelligence Pipeline — Storage Side

Responsibility:
    Consume DerivedEvents from Phase 3's DerivationLog.
    Write to durable storage: SQLite index + JSONL audit log (dual write).
    Provide a read API for post-run query, export, and replay.
    Maintain storage health metrics.

This module owns ONLY:
    - Consuming DerivedEvents from DerivationLog (read-only from Phase 3)
    - Writing to SQLite (indexed query surface) + JSONL (full-fidelity audit)
    - Providing StorageReader: query by track, time range, tag, score
    - Storage health: write latency, queue depth, flush confirmation

This module does NOT own:
    - Any detection, inference, filtering, or derivation   (Phases 1–3)
    - Any alert routing, notification delivery             (downstream consumer)
    - Any re-evaluation, re-scoring, re-tagging            (Phase 3's job)
    - Any write to the Blackboard                          (Phase 1 contract)
    - Any mutation of DerivedEvent after receipt           (immutable by contract)

CQRS contract:
    Phase 4 reads from Phase 3's DerivationLog (append-only, no write path).
    Phase 4 writes to its own SQLite + JSONL stores only.
    Phase 4 never writes back to any upstream phase.

STRUCTURAL RISKS ADDRESSED IN THIS VERSION:

    RISK 1 — SEMANTIC LEAKAGE
        Phase 4 is pure infrastructure. It stores and retrieves.
        StorageReader query methods are named by structural axes:
            by_track, by_time_range, by_tag, by_score_floor, by_delta_reason.
        No query method is named by behavioral intent.
        No field is added to the stored record beyond what DerivedEvent.to_dict()
        produces. Phase 4 does not enrich, re-label, or re-score events.

    RISK 2 — OVER-COMPRESSION
        source_distribution is stored as a JSON blob — never flattened to
        top_label + top_conf columns only.
        source_confidence is stored as a JSON blob — all four dimensions preserved.
        derivation_tags is stored as a JSON array — full tag set, not just primary_tag.
        The SQLite schema has indexed columns for fast query (track_id, timestamp_sec,
        primary_tag, derivation_score) but the full to_dict() blob is always written
        alongside them. The index is a query convenience — not the record of truth.

    RISK 3 — TEMPORAL UNDERDEFINITION
        Three temporal fields are stored as separate columns, never merged:
            timestamp_sec          — video time of source candidate
            derived_at_sec         — wall-clock time of Phase 3 derivation
            stored_at_sec          — wall-clock time of Phase 4 write (new)
        stored_at_sec is Phase 4's own timestamp — it does not overwrite the
        upstream timestamps. All three are preserved independently.
        dwell_sec_at_derivation is stored as a column for range queries.

    RISK 4 — MUTABLE SHARED MEMORY
        Phase 4 reads from DerivationLog.snapshot() — a list copy, not the live list.
        StorageWriter has its own _lock — no shared lock with any upstream phase.
        SQLite WAL mode is enabled: writes do not block reads.
        JSONL is append-only — no seek, no overwrite, no in-place mutation.
        StorageReader is read-only: it opens a separate connection, no write methods.
        DerivedEvent objects are never mutated — stored by to_dict() serialisation.

    RISK 5 — TAXONOMY RIGIDITY
        derivation_tags stored as JSON array — not mapped to an enum column.
        primary_tag stored as nullable text — no foreign key to a fixed tag table.
        delta_reason stored as text — not an integer code.
        Adding a new tag or delta_reason requires no schema migration.
        The schema has a single json_blob column (full to_dict() output) as the
        canonical record. Indexed columns are redundant projections for query speed.

Schema (SQLite):
    CREATE TABLE derived_events (
        event_id              TEXT PRIMARY KEY,
        candidate_id          TEXT NOT NULL,
        track_id              INTEGER NOT NULL,
        camera_id             TEXT NOT NULL,
        timestamp_sec         REAL NOT NULL,    -- video time
        derived_at_sec        REAL NOT NULL,    -- phase3 wall clock
        stored_at_sec         REAL NOT NULL,    -- phase4 write time (new)
        primary_tag           TEXT,             -- nullable
        derivation_score      REAL NOT NULL,
        dwell_sec             REAL NOT NULL,
        delta_reason          TEXT NOT NULL,
        identity_stability    REAL NOT NULL,
        composite_score       REAL NOT NULL,
        derivation_tags_json  TEXT NOT NULL,    -- JSON array — full tag set
        json_blob             TEXT NOT NULL     -- full DerivedEvent.to_dict()
    )

Usage:
    from phase3_derivation import DerivationLeg, DerivationConfig, DerivationLog
    from phase4_storage import StorageLeg, StorageConfig, StorageReader

    log    = DerivationLog()   # or leg3.derivation_log via StorageLeg
    leg4   = StorageLeg(StorageConfig(db_path="events.db", jsonl_path="events.jsonl"), log)

    leg4.start()
    # ... run phase 1-3 ...
    leg4.stop()
    leg4.join()

    reader = StorageReader("events.db")
    events = reader.by_track(track_id=3)
    events = reader.by_time_range(t_start=2.0, t_end=8.0)
    events = reader.by_tag("ZONE_CONTACT")
    events = reader.by_score_floor(0.7)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False

if TYPE_CHECKING:
    from phase3_derivation import DerivedEvent, DerivationLog

log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StorageConfig:
    db_path:            str   = "phase4_events.db"
    jsonl_path:         str   = "phase4_events.jsonl"
    poll_interval_sec:  float = 0.1     # how often to check DerivationLog for new events
    flush_every:        int   = 50      # commit SQLite every N writes (also flush on stop)
    wal_mode:           bool  = True    # SQLite WAL — writes don't block reads
    # RISK 3: stored_at_sec is always written by Phase 4.
    # It is separate from timestamp_sec and derived_at_sec — never overwritten.


# =============================================================================
# SCHEMA
# =============================================================================

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS derived_events (
    event_id              TEXT PRIMARY KEY,
    candidate_id          TEXT NOT NULL,
    track_id              INTEGER NOT NULL,
    camera_id             TEXT NOT NULL,
    timestamp_sec         REAL NOT NULL,
    derived_at_sec        REAL NOT NULL,
    stored_at_sec         REAL NOT NULL,
    primary_tag           TEXT,
    derivation_score      REAL NOT NULL,
    dwell_sec             REAL NOT NULL,
    delta_reason          TEXT NOT NULL,
    identity_stability    REAL NOT NULL,
    composite_score       REAL NOT NULL,
    derivation_tags_json  TEXT NOT NULL,
    json_blob             TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_track_id        ON derived_events(track_id)",
    "CREATE INDEX IF NOT EXISTS idx_timestamp_sec   ON derived_events(timestamp_sec)",
    "CREATE INDEX IF NOT EXISTS idx_primary_tag     ON derived_events(primary_tag)",
    "CREATE INDEX IF NOT EXISTS idx_derivation_score ON derived_events(derivation_score)",
    "CREATE INDEX IF NOT EXISTS idx_delta_reason    ON derived_events(delta_reason)",
    "CREATE INDEX IF NOT EXISTS idx_stored_at_sec   ON derived_events(stored_at_sec)",
]

_INSERT = """
INSERT OR IGNORE INTO derived_events (
    event_id, candidate_id, track_id, camera_id,
    timestamp_sec, derived_at_sec, stored_at_sec,
    primary_tag, derivation_score, dwell_sec,
    delta_reason, identity_stability, composite_score,
    derivation_tags_json, json_blob
) VALUES (
    :event_id, :candidate_id, :track_id, :camera_id,
    :timestamp_sec, :derived_at_sec, :stored_at_sec,
    :primary_tag, :derivation_score, :dwell_sec,
    :delta_reason, :identity_stability, :composite_score,
    :derivation_tags_json, :json_blob
)
"""


# =============================================================================
# STORAGE WRITER — write path
# =============================================================================

class StorageWriter:
    """
    Write path: SQLite + JSONL dual write.

    RISK 4:
        self._lock guards _conn and _jsonl_file.
        StorageWriter is Phase 4-internal — not shared with upstream phases.
        DerivedEvent.to_dict() is called inside the lock; the resulting dict
        is the record written to both stores. The original DerivedEvent is
        never mutated.

    RISK 2:
        json_blob column stores the full to_dict() output.
        source_distribution, source_confidence, derivation_tags are preserved
        in full inside json_blob. Indexed columns (primary_tag, derivation_score,
        etc.) are projections for query speed only — not the record of truth.

    RISK 3:
        stored_at_sec is set by StorageWriter at write time (time.monotonic_ns
        converted to seconds). It is a Phase 4-owned timestamp, separate from
        timestamp_sec (video) and derived_at_sec (phase 3 wall clock).
    """

    def __init__(self, cfg: StorageConfig) -> None:
        self._cfg  = cfg
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._jsonl_file = None
        self._write_count = 0
        self._total_written = 0
        self._total_latency_ms = 0.0

    def open(self) -> None:
        with self._lock:
            self._conn = sqlite3.connect(
                self._cfg.db_path,
                check_same_thread = False,  # guarded by self._lock
            )
            if self._cfg.wal_mode:
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_CREATE_TABLE)
            for idx in _CREATE_INDEXES:
                self._conn.execute(idx)
            self._conn.commit()
            self._jsonl_file = open(self._cfg.jsonl_path, "a", encoding="utf-8")
        log.info(
            "StorageWriter open: db=%s  jsonl=%s  wal=%s",
            self._cfg.db_path, self._cfg.jsonl_path, self._cfg.wal_mode,
        )

    def write(self, event: "DerivedEvent") -> None:
        """
        Write one DerivedEvent to both stores.
        Commits SQLite every flush_every writes; always commits on close().
        JSONL is flushed on every write (append-only, no buffered data loss).
        """
        t_start = time.monotonic()
        stored_at = time.time()

        blob = event.to_dict()

        row = {
            "event_id":             blob["event_id"],
            "candidate_id":         blob["candidate_id"],
            "track_id":             blob["track_id"],
            "camera_id":            blob["camera_id"],
            "timestamp_sec":        blob["timestamp_sec"],
            "derived_at_sec":       blob["derived_at_sec"],
            "stored_at_sec":        stored_at,           # Phase 4 timestamp (RISK 3)
            "primary_tag":          blob["primary_tag"],
            "derivation_score":     blob["derivation_score"],
            "dwell_sec":            blob["dwell_sec_at_derivation"],
            "delta_reason":         blob["delta_reason"],
            "identity_stability":   blob["identity_stability"],
            "composite_score":      blob["composite_score"],
            # Full tag set as JSON array (RISK 2, RISK 5)
            "derivation_tags_json": json.dumps(sorted(blob["derivation_tags"])),
            # Full record — canonical source of truth (RISK 2)
            "json_blob":            json.dumps({**blob, "stored_at_sec": stored_at}),
        }

        with self._lock:
            self._conn.execute(_INSERT, row)
            self._write_count += 1
            if self._write_count >= self._cfg.flush_every:
                self._conn.commit()
                self._write_count = 0

            # JSONL: one line per event, always flushed (RISK 4 — no data loss on crash)
            self._jsonl_file.write(json.dumps({**blob, "stored_at_sec": stored_at}) + "\n")
            self._jsonl_file.flush()

        latency = (time.monotonic() - t_start) * 1000.0
        self._total_written += 1
        self._total_latency_ms += latency

        log.debug(
            "StorageWriter wrote eid=%s  track=%d  t=%.3fs  latency=%.2fms",
            blob["event_id"][:8], blob["track_id"], blob["timestamp_sec"], latency,
        )

    def flush(self) -> None:
        """Force SQLite commit and JSONL flush."""
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._write_count = 0
            if self._jsonl_file:
                self._jsonl_file.flush()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._conn.close()
                self._conn = None
            if self._jsonl_file:
                self._jsonl_file.flush()
                self._jsonl_file.close()
                self._jsonl_file = None
        log.info(
            "StorageWriter closed. total_written=%d  avg_latency=%.2fms",
            self._total_written,
            self._total_latency_ms / max(1, self._total_written),
        )

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / max(1, self._total_written)


# =============================================================================
# STORAGE READER — read-only query API
# =============================================================================

class StorageReader:
    """
    Read-only query API over the SQLite store.

    RISK 4:
        StorageReader opens its own connection — never shares the writer's
        connection. WAL mode allows concurrent reads with no write blocking.
        All methods return list[dict] — plain dicts, not SQLite row objects.
        json_blob is always deserialized and returned — full record, never
        a projected subset.

    RISK 1:
        Query methods are named by structural axes only:
            by_track, by_time_range, by_tag, by_score_floor, by_delta_reason.
        No method named by behavioral intent (no "by_threat_level", etc.).

    RISK 2:
        All returned records include the full json_blob deserialized.
        source_distribution, source_confidence, derivation_tags are always present.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _rows_to_dicts(self, rows) -> list[dict]:
        """Deserialize json_blob for each row — full record, not projected columns."""
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["json_blob"]))
            except (json.JSONDecodeError, KeyError):
                result.append(dict(row))
        return result

    def by_track(self, track_id: int) -> list[dict]:
        """All events for a specific track, ordered by video timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT json_blob FROM derived_events WHERE track_id=? ORDER BY timestamp_sec",
                (track_id,),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def by_time_range(self, t_start: float, t_end: float) -> list[dict]:
        """All events within a video time range [t_start, t_end], ordered by timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT json_blob FROM derived_events "
                "WHERE timestamp_sec >= ? AND timestamp_sec <= ? ORDER BY timestamp_sec",
                (t_start, t_end),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def by_tag(self, tag: str) -> list[dict]:
        """
        All events where derivation_tags_json contains the given tag string.
        Uses JSON array membership check via SQLite json_each().
        Falls back to LIKE if json_each is unavailable (older SQLite).
        """
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT e.json_blob FROM derived_events e "
                    "JOIN json_each(e.derivation_tags_json) t ON t.value=? "
                    "ORDER BY e.timestamp_sec",
                    (tag,),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback: LIKE search (less precise but works on older SQLite)
                rows = conn.execute(
                    "SELECT json_blob FROM derived_events "
                    "WHERE derivation_tags_json LIKE ? ORDER BY timestamp_sec",
                    (f'%"{tag}"%',),
                ).fetchall()
        return self._rows_to_dicts(rows)

    def by_score_floor(self, min_score: float) -> list[dict]:
        """All events with derivation_score >= min_score, ordered by score descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT json_blob FROM derived_events "
                "WHERE derivation_score >= ? ORDER BY derivation_score DESC",
                (min_score,),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def by_delta_reason(self, reason: str) -> list[dict]:
        """All events with a specific delta_reason, ordered by video timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT json_blob FROM derived_events "
                "WHERE delta_reason=? ORDER BY timestamp_sec",
                (reason,),
            ).fetchall()
        return self._rows_to_dicts(rows)

    def all_events(self, order_by: str = "timestamp_sec") -> list[dict]:
        """All events. order_by must be a column name (validated against schema)."""
        allowed = {
            "timestamp_sec", "derived_at_sec", "stored_at_sec",
            "derivation_score", "track_id", "dwell_sec",
        }
        if order_by not in allowed:
            raise ValueError(f"order_by must be one of {allowed}, got {order_by!r}")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT json_blob FROM derived_events ORDER BY {order_by}"
            ).fetchall()
        return self._rows_to_dicts(rows)

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM derived_events").fetchone()[0]

    def track_ids(self) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT track_id FROM derived_events ORDER BY track_id"
            ).fetchall()
        return [r[0] for r in rows]

    def tag_counts(self) -> dict[str, int]:
        """Count of events per primary_tag (nullable — null counted separately)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT primary_tag, COUNT(*) FROM derived_events GROUP BY primary_tag"
            ).fetchall()
        return {(r[0] or "__no_primary_tag__"): r[1] for r in rows}

    def score_histogram(self, bins: int = 10) -> list[dict]:
        """
        Histogram of derivation_score in `bins` equal-width buckets.
        Returns list of {bin_floor, bin_ceil, count}.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT derivation_score FROM derived_events"
            ).fetchall()
        if not rows:
            return []
        scores = [r[0] for r in rows]
        lo, hi = min(scores), max(scores)
        if lo == hi:
            return [{"bin_floor": lo, "bin_ceil": hi, "count": len(scores)}]
        width = (hi - lo) / bins
        counts = [0] * bins
        for s in scores:
            b = min(bins - 1, int((s - lo) / width))
            counts[b] += 1
        return [
            {"bin_floor": round(lo + i*width, 4), "bin_ceil": round(lo + (i+1)*width, 4), "count": counts[i]}
            for i in range(bins)
        ]


# =============================================================================
# STORAGE STATS — observability only
# =============================================================================

@dataclass
class StorageStats:
    total_written:    int   = 0
    total_skipped:    int   = 0   # events already seen (cursor-based polling)
    cycles_run:       int   = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        # BUG FIX: was referencing self._total_latency_ms (private, non-existent).
        # StorageStats is a dataclass — field is self.total_latency_ms (public).
        return self.total_latency_ms / max(1, self.total_written)

    def to_dict(self) -> dict:
        return {
            "total_written":   self.total_written,
            "total_skipped":   self.total_skipped,
            "cycles_run":      self.cycles_run,
            "avg_latency_ms":  round(self.avg_latency_ms, 2),
        }


# =============================================================================
# STORAGE LEG — background thread
# =============================================================================

class StorageLeg:
    """
    Phase 4 background thread.

    Polls DerivationLog for new DerivedEvents via cursor-based polling.
    Passes each event to StorageWriter (SQLite + JSONL dual write).

    CQRS:
        Reads from DerivationLog (Phase 3-owned, append-only).
        Writes to its own SQLite + JSONL stores.
        Never writes to Blackboard, FilterBlackboardExtension, or DerivationLog.

    Thread model:
        One background thread runs _run().
        StorageWriter is Phase 4-internal — not shared with other phases.
        DerivationLog.snapshot() returns a list copy — no shared reference.

    Lifecycle:
        start() → opens StorageWriter, starts background thread.
        stop()  → signals thread to exit, flushes remaining events.
        join()  → waits for thread to finish, closes StorageWriter.
    """

    def __init__(
        self,
        cfg:         StorageConfig,
        deriv_log:   "DerivationLog",
    ) -> None:
        self._cfg      = cfg
        self._log      = deriv_log
        self._writer   = StorageWriter(cfg)
        self._stats    = StorageStats()
        self._stop     = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor   = 0   # index into DerivationLog — events before cursor already written

    def start(self) -> None:
        self._writer.open()
        self._thread = threading.Thread(
            target = self._run,
            name   = "phase4-storage",
            daemon = True,
        )
        self._thread.start()
        log.info(
            "Phase 4 Storage started. db=%s  jsonl=%s  poll=%.3fs  flush_every=%d",
            self._cfg.db_path, self._cfg.jsonl_path,
            self._cfg.poll_interval_sec, self._cfg.flush_every,
        )

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 15.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._writer.close()
        log.info(
            "Phase 4 complete. written=%d  skipped=%d  cycles=%d",
            self._stats.total_written,
            self._stats.total_skipped,
            self._stats.cycles_run,
        )

    def _run(self) -> None:
        log.info("Phase 4 storage loop running.")
        pbar = _tqdm(
            desc="Phase 4 — Storing",
            unit="events",
            dynamic_ncols=True,
            colour="blue",
            leave=True,
        ) if _TQDM_OK else None
        try:
            while not self._stop.is_set():
                try:
                    self._drain(pbar)
                except Exception as e:
                    log.exception("Phase 4 storage error: %s", e)
                self._stop.wait(timeout=self._cfg.poll_interval_sec)
        finally:
            if pbar is not None:
                pbar.close()

        # Final drain after stop signal
        try:
            self._drain(None)
            self._writer.flush()
        except Exception as e:
            log.exception("Phase 4 final drain error: %s", e)

    def _drain(self, pbar=None) -> None:
        """
        Cursor-based polling — only processes events not yet written.

        RISK 4: DerivationLog.snapshot() returns a list copy. self._cursor
        is Phase 4-internal state — never shared with Phase 3.
        Events before self._cursor are already written — never re-processed.
        """
        snapshot = self._log.snapshot()
        new_events = snapshot[self._cursor:]
        if not new_events:
            return

        t_batch = time.monotonic()
        for event in new_events:
            self._writer.write(event)
            self._stats.total_written += 1
        self._stats.total_latency_ms += (time.monotonic() - t_batch) * 1000.0

        self._cursor += len(new_events)
        self._stats.cycles_run += 1

        if pbar is not None:
            pbar.update(len(new_events))
            pbar.set_postfix({
                "total": self._stats.total_written,
                "lat_ms": f"{self._stats.avg_latency_ms:.1f}",
                "cursor": self._cursor,
            }, refresh=False)

        if self._stats.cycles_run % 20 == 0:
            log.info("Storage stats: %s  cursor=%d", self._stats.to_dict(), self._cursor)

    @property
    def reader(self) -> StorageReader:
        """Read-only query interface. Usable during or after the run."""
        return StorageReader(self._cfg.db_path)

    @property
    def stats(self) -> StorageStats:
        return self._stats


# =============================================================================
# Standalone test — full Phase 1 + 2 + 3 + 4 pipeline
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 1+2+3+4 pipeline test")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--model",        default="")
    parser.add_argument("--mode",         type=int, default=2, choices=[1, 2])
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--zones",        default=None)
    parser.add_argument("--skip",         type=int, default=1)
    parser.add_argument("--end",          type=float, default=0.0)
    parser.add_argument("--cooldown",     type=float, default=3.0)
    parser.add_argument("--score",        type=float, default=0.40)
    parser.add_argument("--stability",    type=float, default=0.30)
    parser.add_argument("--require-zone", action="store_true")
    parser.add_argument("--permissive",   action="store_true")
    parser.add_argument("--dwell",        type=float, default=10.0)
    parser.add_argument("--peak",         type=float, default=0.60)
    parser.add_argument("--db",           default="phase4_events.db")
    parser.add_argument("--jsonl",        default="phase4_events.jsonl")
    args = parser.parse_args()

    import json as _json
    from phase1_observation import (
        ObservationConfig, ObservationLeg, Blackboard, resolve_action_model_path,
    )
    from phase2_filter import FilterLeg, FilterConfig
    from phase3_derivation import DerivationLeg, DerivationConfig

    zones_raw = []
    if args.zones:
        with open(args.zones, encoding="utf-8") as f:
            zones_raw = _json.load(f)

    model_path = resolve_action_model_path(args.mode, args.model)

    cfg1 = ObservationConfig(
        input_path  = args.input, model_path = model_path,
        vision_mode = args.mode,  device     = args.device,
        zones       = zones_raw,  end_sec    = args.end, frame_skip = args.skip,
    )
    cfg2 = FilterConfig(
        min_composite_score    = args.score,    min_identity_stability = args.stability,
        cooldown_sec           = args.cooldown, require_zone_contact   = args.require_zone,
        permissive_mode        = args.permissive,
    )
    cfg3 = DerivationConfig(dwell_threshold_sec=args.dwell, peak_conf_threshold=args.peak)
    cfg4 = StorageConfig(db_path=args.db, jsonl_path=args.jsonl)

    bb   = Blackboard(cfg1.build_temporal_contract())
    leg2 = FilterLeg(cfg2, bb)
    leg3 = DerivationLeg(cfg3, leg2.bb_ext)
    leg4 = StorageLeg(cfg4, leg3.log)
    leg1 = ObservationLeg(cfg1, bb)

    print(f"\n  Phase 1+2+3+4 pipeline")
    print(f"  Input   : {args.input}")
    print(f"  Model   : {model_path}")
    print(f"  Mode    : {args.mode}")
    print(f"  DB      : {args.db}")
    print(f"  JSONL   : {args.jsonl}")
    print()

    leg4.start()
    leg3.start()
    leg2.start()
    leg1.run()
    leg2.stop(); leg2.join()
    leg3.stop(); leg3.join()
    leg4.stop(); leg4.join()

    reader = leg4.reader
    n = reader.count()
    tracks = reader.track_ids()
    tags = reader.tag_counts()

    print(f"  Stored events : {n}")
    print(f"  Tracks        : {tracks}")
    print(f"  Tag counts    : {tags}")
    print(f"  Score hist    : {reader.score_histogram(bins=5)}")
    print()
