"""
heatmap_phase4.py — Phase 4 Storage Heatmap Visualizer
=======================================================
Reads from Phase 4's SQLite store (via StorageReader) or directly from
the JSONL audit log. Pure infrastructure layer — no semantic enrichment.

    Risk audit:
        RISK 1 — Phase 4 is pure storage infrastructure. This visualizer
                 exposes only structural storage metrics: write throughput,
                 latency distribution, tag/track/score indices, temporal
                 coverage. No new behavioral categories are introduced.
                 Query axes match StorageReader's structural API exactly:
                 by_track, by_time_range, by_tag, by_score_floor, by_delta_reason.
        RISK 2 — json_blob column (full DerivedEvent.to_dict()) is used as the
                 source of truth for all distribution data. Projected index
                 columns (primary_tag, derivation_score, etc.) are used only
                 for aggregation — never as a substitute for the full record.
                 source_distribution is preserved in the full blob and surfaced
                 in the action distribution heatmap.
        RISK 3 — Three temporal columns are kept separate throughout:
                 timestamp_sec (video time), derived_at_sec (Phase 3 wall clock),
                 stored_at_sec (Phase 4 write time). No merging.
                 write_latency = stored_at_sec - derived_at_sec.
                 pipeline_latency = stored_at_sec - timestamp_sec.
                 Both are computed on demand — not stored separately.
        RISK 4 — StorageReader opens its own SQLite connection (WAL mode).
                 This visualizer is READ-ONLY. It never calls StorageWriter.
                 All mutations to SQLite happen only through StorageWriter._lock.
                 JSONL is read sequentially — append-only file, no seek writes.
        RISK 5 — primary_tag is a projected convenience column in SQLite.
                 derivation_tags_json (full tag array) is always read from
                 json_blob — the projected column is only used for GROUP BY
                 efficiency. delta_reason is passed through verbatim.

Input sources:
    1. StorageLeg instance     (live, post-run)    via from_storage_leg()
    2. StorageReader instance  (direct)             via from_storage_reader()
    3. str/Path to .db file                         via from_db_path()
    4. str/Path to .jsonl file                      via from_jsonl_path()
    5. list[dict] (serialised events)               via from_dict()

Output: dict  (JSON-serialisable; matches page.tsx HEATMAP_DEFS[4] schema)
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Phase4HeatmapBuilder:
    """
    Builds all Phase 4 heatmap payloads from the storage layer.

    Heatmaps produced:
        1. write_throughput      — timeline: events written per time window
        2. write_latency_dist    — histogram: stored_at_sec - derived_at_sec (ms)
        3. pipeline_latency_dist — histogram: stored_at_sec - timestamp_sec (ms)
        4. tag_coverage          — bars: how many events per derivation_tag in DB
        5. track_coverage        — histogram: events per track_id in DB
        6. score_dist            — histogram: derivation_score distribution in DB
        7. delta_reason_in_store — bars: delta_reason counts (verbatim from stored events)
        8. temporal_coverage     — timeline: stored events vs timestamp_sec (video coverage)
    """

    def __init__(self, n_time_bins: int = 7, n_latency_bins: int = 10,
                 n_score_bins: int = 10) -> None:
        self._n_time_bins    = n_time_bins
        self._n_latency_bins = n_latency_bins
        self._n_score_bins   = n_score_bins

    # ── Input loaders ────────────────────────────────────────────────────────

    @classmethod
    def from_storage_leg(cls, storage_leg, **kwargs) -> "Phase4HeatmapBuilder":
        """
        Load from a live StorageLeg after it has completed.
        Opens a StorageReader against the same db_path. Read-only.
        """
        inst = cls(**kwargs)
        inst._rows = _read_all_from_db(storage_leg.cfg.db_path)
        inst._storage_health = {
            "total_written":    storage_leg.writer.total_written,
            "avg_latency_ms":   round(storage_leg.writer.avg_latency_ms, 3),
        }
        return inst

    @classmethod
    def from_storage_reader(cls, reader, **kwargs) -> "Phase4HeatmapBuilder":
        """
        Load via an existing StorageReader instance.
        Uses internal _connect() by reading all rows from the DB.
        """
        inst = cls(**kwargs)
        # StorageReader doesn't expose a select_all — we open directly
        inst._rows = _read_all_from_db(reader._db_path)
        inst._storage_health = {}
        return inst

    @classmethod
    def from_db_path(cls, db_path: str | Path, **kwargs) -> "Phase4HeatmapBuilder":
        """Load directly from a SQLite .db file path."""
        inst = cls(**kwargs)
        inst._rows = _read_all_from_db(str(db_path))
        inst._storage_health = {}
        return inst

    @classmethod
    def from_jsonl_path(cls, jsonl_path: str | Path, **kwargs) -> "Phase4HeatmapBuilder":
        """
        Load from the JSONL audit log.
        Each line is a full DerivedEvent.to_dict() + stored_at_sec.
        """
        inst = cls(**kwargs)
        rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        inst._rows = rows
        inst._storage_health = {}
        return inst

    @classmethod
    def from_dict(cls, events: list[dict], **kwargs) -> "Phase4HeatmapBuilder":
        """Load from a list of serialised event dicts."""
        inst = cls(**kwargs)
        inst._rows = list(events)
        inst._storage_health = {}
        return inst

    # ── Main build ───────────────────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        rows = self._normalise()

        if not rows:
            return self._empty_payload()

        t_min_video = min(r["timestamp_sec"] for r in rows)
        t_max_video = max(r["timestamp_sec"] for r in rows)
        t_min_store = min(r["stored_at_sec"] for r in rows if r["stored_at_sec"])
        t_max_store = max(r["stored_at_sec"] for r in rows if r["stored_at_sec"])

        video_duration = max(t_max_video - t_min_video, 1e-6)
        store_duration = max(t_max_store - t_min_store, 1e-6)
        bin_width_video = video_duration / self._n_time_bins
        bin_width_store = store_duration / self._n_time_bins

        return {
            "write_throughput":      self._build_write_throughput(rows, t_min_store, bin_width_store),
            "write_latency_dist":    self._build_write_latency(rows),
            "pipeline_latency_dist": self._build_pipeline_latency(rows),
            "tag_coverage":          self._build_tag_coverage(rows),
            "track_coverage":        self._build_track_coverage(rows),
            "score_dist":            self._build_score_distribution(rows),
            "delta_reason_in_store": self._build_delta_reason(rows),
            "temporal_coverage":     self._build_temporal_coverage(rows, t_min_video, bin_width_video),
            "meta": {
                # Structural storage metrics — no behavioral labels
                "n_events_stored":     len(rows),
                "n_tracks_stored":     len({r["track_id"] for r in rows}),
                "n_unique_tags":       len({t for r in rows for t in r["derivation_tags"]}),
                # Temporal axes kept separate (RISK 3)
                "video_t_min_sec":     round(t_min_video, 3),
                "video_t_max_sec":     round(t_max_video, 3),
                "video_duration_sec":  round(video_duration, 3),
                "store_t_min":         round(t_min_store, 3),
                "store_t_max":         round(t_max_store, 3),
                "mean_derivation_score": round(
                    sum(r["derivation_score"] for r in rows) / len(rows), 4
                ),
                # StorageWriter health metrics if available
                **self._storage_health,
                "note_temporal": "timestamp_sec=video time; derived_at_sec=Phase3 wall clock; "
                                 "stored_at_sec=Phase4 write time. Never merged.",
            },
        }

    # ── Write throughput ─────────────────────────────────────────────────────

    def _build_write_throughput(self, rows: list, t_min_store: float,
                                bin_width: float) -> dict:
        """
        Events written to storage per time window (stored_at_sec axis).

        Uses stored_at_sec — Phase 4's own timestamp.
        Separate from timestamp_sec (video) and derived_at_sec (Phase 3).
        """
        bins = [0] * self._n_time_bins
        for r in rows:
            st = r.get("stored_at_sec")
            if st:
                bi = min(int((st - t_min_store) / bin_width), self._n_time_bins - 1)
                bins[bi] += 1

        labels = [f"+{round(i * bin_width, 1)} s" for i in range(self._n_time_bins)]
        total  = sum(bins)
        peak   = max(bins) if bins else 0

        return {
            "type":    "timeline",
            "title":   "Phase 4 — Storage write throughput (events / window)",
            "labels":  labels,
            "values":  bins,
            "color":   "#ea580c",
            "y_label": "events / window",
            "summary": {
                "total": total,
                "peak":  peak,
                "avg":   round(total / max(self._n_time_bins, 1), 1),
            },
            "note":    "Temporal axis = stored_at_sec (Phase 4 write time). "
                       "Separate from timestamp_sec (video) and derived_at_sec (Phase 3).",
        }

    # ── Write latency ────────────────────────────────────────────────────────

    def _build_write_latency(self, rows: list) -> dict:
        """
        Histogram of write latency = stored_at_sec - derived_at_sec (ms).

        This is the Phase 3 → Phase 4 hand-off latency.
        Computed on demand from two separate temporal fields (RISK 3).
        """
        latencies_ms = []
        for r in rows:
            st = r.get("stored_at_sec")
            da = r.get("derived_at_sec")
            if st and da:
                latencies_ms.append((st - da) * 1000.0)

        return self._latency_histogram(
            latencies_ms,
            title="Phase 4 — Write latency (stored_at_sec − derived_at_sec)",
            note="Phase 3→4 hand-off latency. Computed from two separate temporal fields.",
            color="#f59e0b",
        )

    # ── Pipeline latency ─────────────────────────────────────────────────────

    def _build_pipeline_latency(self, rows: list) -> dict:
        """
        Histogram of full pipeline latency = stored_at_sec - timestamp_sec (ms).

        This is the end-to-end latency from video frame time to storage write.
        Computed on demand from two separate temporal fields (RISK 3).
        Note: this spans video time → wall clock, so the value includes
        processing time + any video playback rate differences.
        """
        latencies_ms = []
        for r in rows:
            st = r.get("stored_at_sec")
            ts = r.get("timestamp_sec")
            if st and ts is not None:
                latencies_ms.append((st - ts) * 1000.0)

        return self._latency_histogram(
            latencies_ms,
            title="Phase 4 — Pipeline latency (stored_at_sec − timestamp_sec)",
            note="End-to-end: video frame time to storage write. "
                 "Includes processing + video rate differences.",
            color="#e11d48",
        )

    def _latency_histogram(self, latencies_ms: list[float], title: str,
                           note: str, color: str) -> dict:
        if not latencies_ms:
            return {"type": "histogram", "labels": [], "values": [], "title": title}

        max_lat = max(latencies_ms)
        min_lat = min(latencies_ms)
        mean_lat = sum(latencies_ms) / len(latencies_ms)

        bin_width = max(max_lat - min_lat, 1e-6) / self._n_latency_bins
        hist = [0] * self._n_latency_bins
        for v in latencies_ms:
            bi = min(int((v - min_lat) / bin_width), self._n_latency_bins - 1)
            hist[bi] += 1

        labels = [f"{round(min_lat + i * bin_width, 1)}–{round(min_lat + (i+1) * bin_width, 1)} ms"
                  for i in range(self._n_latency_bins)]

        return {
            "type":    "histogram",
            "title":   title,
            "labels":  labels,
            "values":  hist,
            "color":   color,
            "summary": {
                "min_ms":  round(min_lat, 2),
                "max_ms":  round(max_lat, 2),
                "mean_ms": round(mean_lat, 2),
            },
            "note": note,
        }

    # ── Tag coverage ─────────────────────────────────────────────────────────

    def _build_tag_coverage(self, rows: list) -> dict:
        """
        Count of stored events per derivation_tag (full tag set from json_blob).

        Uses derivation_tags from the full json_blob — not the projected
        derivation_tags_json column. (Same content, but json_blob is the
        canonical record — RISK 2.)

        Tag strings are verbatim from stored DerivedEvent.to_dict().
        """
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            for tag in r["derivation_tags"]:
                counts[tag] += 1

        total_events = len(rows)
        ordered = sorted(counts.items(), key=lambda x: -x[1])

        return {
            "type":   "bars",
            "title":  "Phase 4 — Stored event coverage by derivation_tag",
            "rows":   [k for k, _ in ordered],
            "values": [v for _, v in ordered],
            "pct":    [round(v / max(total_events, 1) * 100, 1) for _, v in ordered],
            "raw":    dict(ordered),
            "color":  "#7c3aed",
            "note":   "Counts from full derivation_tags in json_blob (canonical record). "
                      "Not from projected derivation_tags_json column alone.",
        }

    # ── Track coverage ───────────────────────────────────────────────────────

    def _build_track_coverage(self, rows: list) -> dict:
        """Events per track_id in the stored dataset."""
        per_track: dict[int, int] = defaultdict(int)
        for r in rows:
            per_track[r["track_id"]] += 1

        counts = list(per_track.values())
        if not counts:
            return {"type": "histogram", "labels": [], "values": []}

        max_c = max(counts)
        n_bins = min(max_c, 10)
        bw = max(max_c / n_bins, 1)
        hist = [0] * n_bins
        for c in counts:
            bi = min(int((c - 1) / bw), n_bins - 1)
            hist[bi] += 1

        labels = [f"{int(i * bw + 1)}–{int((i+1) * bw)}" for i in range(n_bins)]

        return {
            "type":    "histogram",
            "title":   "Phase 4 — Stored events per track_id",
            "labels":  labels,
            "values":  hist,
            "color":   "#0891b2",
            "raw":     {str(k): v for k, v in per_track.items()},
            "x_label": "event count",
            "y_label": "track count",
        }

    # ── Score distribution ───────────────────────────────────────────────────

    def _build_score_distribution(self, rows: list) -> dict:
        """
        Histogram of derivation_score for all stored events.

        derivation_score is a projected indexed column in SQLite for query
        efficiency. It is also present in json_blob. Both contain the same value;
        we use the json_blob-sourced value here (canonical — RISK 2).
        """
        scores = [r["derivation_score"] for r in rows]
        if not scores:
            return {"type": "histogram", "labels": [], "values": []}

        hist = [0] * self._n_score_bins
        for s in scores:
            bi = min(int(s * self._n_score_bins), self._n_score_bins - 1)
            hist[bi] += 1

        labels = [f"{i/self._n_score_bins:.1f}–{(i+1)/self._n_score_bins:.1f}"
                  for i in range(self._n_score_bins)]

        return {
            "type":    "histogram",
            "title":   "Phase 4 — Derivation score distribution (stored events)",
            "labels":  labels,
            "values":  hist,
            "color":   "#16a34a",
            "summary": {
                "min":  round(min(scores), 4),
                "max":  round(max(scores), 4),
                "mean": round(sum(scores) / len(scores), 4),
            },
            "note":    "derivation_score is a structural signal strength indicator. "
                       "Values from json_blob (canonical record) — not projected column alone.",
        }

    # ── Delta reason in store ────────────────────────────────────────────────

    def _build_delta_reason(self, rows: list) -> dict:
        """
        Count of DeltaReason values across stored events.
        Verbatim from stored DerivedEvent — no re-mapping.
        """
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["delta_reason"]] += 1

        ordered = sorted(counts.items(), key=lambda x: -x[1])
        total = sum(counts.values()) or 1

        return {
            "type":   "bars",
            "title":  "Phase 4 — Stored events by delta_reason (structural transitions)",
            "rows":   [k for k, _ in ordered],
            "values": [v for _, v in ordered],
            "pct":    [round(v / total * 100, 1) for _, v in ordered],
            "color":  "#64748b",
            "note":   "delta_reason is a structural transition type from Phase 1 — no behavioral re-labeling.",
        }

    # ── Temporal coverage ────────────────────────────────────────────────────

    def _build_temporal_coverage(self, rows: list, t_min_video: float,
                                 bin_width: float) -> dict:
        """
        Stored events per video time bin (timestamp_sec axis).

        Shows which parts of the source video are represented in storage.
        Uses timestamp_sec (video time) — separate from stored_at_sec (Phase 4 time).
        """
        bins = [0] * self._n_time_bins
        for r in rows:
            ts = r.get("timestamp_sec", 0.0)
            bi = min(int((ts - t_min_video) / bin_width), self._n_time_bins - 1)
            bins[bi] += 1

        labels = [f"{round(t_min_video + i * bin_width, 1)} s" for i in range(self._n_time_bins)]

        return {
            "type":    "timeline",
            "title":   "Phase 4 — Stored event density by video time (timestamp_sec)",
            "labels":  labels,
            "values":  bins,
            "color":   "#0284c7",
            "y_label": "events stored",
            # Seek map: col_index → video timestamp_sec for UI video sync
            "seek_map": {str(i): float(round(t_min_video + i * bin_width, 3)) for i in range(self._n_time_bins)},
            "note":    "Temporal axis = timestamp_sec (video time). "
                       "Separate from stored_at_sec (Phase 4 write time).",
        }

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalise(self) -> list[dict]:
        out = []
        for r in self._rows:
            out.append(self._normalise_row(r))
        # Async workers may commit rows out-of-order — re-sort by absolute
        # timestamp_sec (video time) before any binning occurs (RISK 3 preserved).
        out.sort(key=lambda r: r["timestamp_sec"])
        return out

    def _normalise_row(self, r: dict) -> dict:
        """
        Normalise a stored event dict (from json_blob or JSONL).

        RISK 2: derivation_tags read from full json_blob field (list or frozenset).
                source_distribution (class_probs) read from json_blob.
        RISK 3: timestamp_sec, derived_at_sec, stored_at_sec kept as separate fields.
        """
        cp = r.get("source_distribution", {})
        tags_raw = r.get("derivation_tags", [])
        tags = set(tags_raw) if not isinstance(tags_raw, set) else tags_raw

        return {
            "event_id":          r.get("event_id", ""),
            "track_id":          int(r.get("track_id", -1)),
            "timestamp_sec":     float(r.get("timestamp_sec", 0.0)),    # video time
            "derived_at_sec":    float(r.get("derived_at_sec", 0.0)),   # Phase 3 clock
            "stored_at_sec":     float(r.get("stored_at_sec", 0.0)),    # Phase 4 clock
            "derivation_tags":   tags,
            "primary_tag":       r.get("primary_tag"),          # projected convenience
            "derivation_score":  float(r.get("derivation_score", 0.0)),
            "delta_reason":      r.get("delta_reason", ""),
            "class_probs":       cp if isinstance(cp, dict) else {},
            "composite_score":   float(r.get("composite_score", 0.0)),
            "identity_stability": float(r.get("identity_stability", 1.0)),
        }

    def _empty_payload(self) -> dict:
        return {
            "write_throughput":      {"type": "timeline",  "labels": [], "values": []},
            "write_latency_dist":    {"type": "histogram", "labels": [], "values": []},
            "pipeline_latency_dist": {"type": "histogram", "labels": [], "values": []},
            "tag_coverage":          {"type": "bars",      "rows":   [], "values": []},
            "track_coverage":        {"type": "histogram", "labels": [], "values": []},
            "score_dist":            {"type": "histogram", "labels": [], "values": []},
            "delta_reason_in_store": {"type": "bars",      "rows":   [], "values": []},
            "temporal_coverage":     {"type": "timeline",  "labels": [], "values": []},
            "meta": {"n_events_stored": 0},
        }


# ---------------------------------------------------------------------------
# SQLite reader utility
# ---------------------------------------------------------------------------

def _read_all_from_db(db_path: str) -> list[dict]:
    """
    Read all rows from the Phase 4 SQLite store.
    Deserialises json_blob for full fidelity (RISK 2).
    Uses a separate read-only connection (RISK 4 — WAL mode).
    """
    if not Path(db_path).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT json_blob FROM derived_events ORDER BY timestamp_sec").fetchall()
        conn.close()
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["json_blob"]))
            except (json.JSONDecodeError, KeyError):
                pass
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# FastAPI-ready endpoint helper
# ---------------------------------------------------------------------------

def build_phase4_heatmap(source, input_duration_sec: float | None = None,
                         output_duration_sec: float | None = None, **kwargs) -> dict:
    """
    Single entry point for the FastAPI endpoint.

    source:
        StorageLeg instance             → from_storage_leg()
        StorageReader instance          → from_storage_reader()
        str / Path ending in .db        → from_db_path()
        str / Path ending in .jsonl     → from_jsonl_path()
        list[dict]                      → from_dict()
        None                            → empty payload

    kwargs: n_time_bins, n_latency_bins, n_score_bins
    """
    if source is None:
        return Phase4HeatmapBuilder(**kwargs).build()

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix == ".jsonl":
            builder = Phase4HeatmapBuilder.from_jsonl_path(source, **kwargs)
        else:
            builder = Phase4HeatmapBuilder.from_db_path(source, **kwargs)
    elif isinstance(source, list):
        builder = Phase4HeatmapBuilder.from_dict(source, **kwargs)
    elif hasattr(source, "cfg") and hasattr(source, "writer"):
        # StorageLeg: has .cfg (StorageConfig) and .writer (StorageWriter)
        builder = Phase4HeatmapBuilder.from_storage_leg(source, **kwargs)
    elif hasattr(source, "_db_path"):
        # StorageReader
        builder = Phase4HeatmapBuilder.from_storage_reader(source, **kwargs)
    else:
        return Phase4HeatmapBuilder(**kwargs).build()

    payload = builder.build()
    from seek_utils import scale_all_seek_maps
    return scale_all_seek_maps(payload, input_duration_sec, output_duration_sec)


# ---------------------------------------------------------------------------
# Standalone test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 Heatmap Builder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--db",   help="Path to phase4_events.db (SQLite)")
    group.add_argument("--jsonl", help="Path to phase4_events.jsonl")
    parser.add_argument("--out",   default="heatmap_phase4.json")
    parser.add_argument("--tbins", type=int, default=7)
    parser.add_argument("--lbins", type=int, default=10)
    parser.add_argument("--sbins", type=int, default=10)
    args = parser.parse_args()

    src = args.db or args.jsonl
    payload = build_phase4_heatmap(
        source=src,
        n_time_bins=args.tbins,
        n_latency_bins=args.lbins,
        n_score_bins=args.sbins,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=list)

    meta = payload["meta"]
    print(f"Phase 4 heatmap built.")
    print(f"  Events stored : {meta.get('n_events_stored', 0)}")
    print(f"  Tracks        : {meta.get('n_tracks_stored', 0)}")
    print(f"  Unique tags   : {meta.get('n_unique_tags', 0)}")
    if "avg_latency_ms" in meta:
        print(f"  Avg write lat : {meta['avg_latency_ms']} ms")
    print(f"  Output        : {args.out}")
