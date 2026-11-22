"""
heatmap_phase3.py — Phase 3 Derivation Heatmap Visualizer
==========================================================
Reads from Phase 3's DerivationLog and DerivedEvent records.
Phase 3 is the FIRST phase permitted to assign operational meaning — and
this visualizer mirrors that permission boundary exactly.

    Risk audit:
        RISK 1 — Phase 3 owns derivation_tags (configured operator strings).
                 This visualizer exposes tag distributions, tag co-occurrence,
                 and primary_tag frequency — exactly what Phase 3 is permitted to
                 output. It does NOT add new behavioral labels beyond what appears
                 in derivation_tags. Tag strings come verbatim from DerivedEvent.
        RISK 2 — source_distribution (full ActionDistribution) is read for the
                 action probability heatmap. primary_tag and derivation_score are
                 treated as convenience fields — the full distribution is the
                 canonical record.
        RISK 3 — Three temporal fields are kept separate throughout:
                 timestamp_sec (video time), derived_at_sec (Phase 3 wall clock),
                 derivation_latency_ms (per-event cost). dwell_sec_at_derivation
                 is the point-in-time snapshot from DerivationHistory — not a
                 live accumulator.
        RISK 4 — This module is READ-ONLY. It reads from DerivationLog.snapshot()
                 (a list copy). Never writes to DerivationLog, DerivationHistory,
                 or any upstream structure.
        RISK 5 — derivation_tags are stored as frozenset in DerivedEvent —
                 a full set, not an ordered priority list. primary_tag is a
                 convenience projection. This visualizer shows both the full
                 tag set distribution and the primary_tag frequency separately,
                 making the distinction explicit in output metadata.

Input sources:
    1. DerivationLeg instance      (live, post-run)     via from_derivation_leg()
    2. DerivationLog instance      (direct)              via from_derivation_log()
    3. list[DerivedEvent]          (direct)              via from_events()
    4. list[dict] (serialised)                          via from_dict()

Output: dict  (JSON-serialisable; matches page.tsx HEATMAP_DEFS[3] schema)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Phase3HeatmapBuilder:
    """
    Builds all Phase 3 heatmap payloads from DerivedEvent records.

    Heatmaps produced:
        1. action_prob_heatmap   — grid: action class × event (full class_probs)
        2. tag_frequency         — bars: derivation_tag occurrence count (full tag set)
        3. primary_tag_dist      — bars: primary_tag frequency (convenience field)
        4. score_over_time       — timeline: derivation_score per time bin
        5. dwell_distribution    — histogram: dwell_sec_at_derivation
        6. tag_cooccurrence      — matrix: how often tag pairs appear together
        7. track_event_count     — histogram: events per track_id
    """

    def __init__(self, n_time_bins: int = 8, n_dwell_bins: int = 10,
                 max_events_in_grid: int = 10) -> None:
        self._n_time_bins      = n_time_bins
        self._n_dwell_bins     = n_dwell_bins
        self._max_events_grid  = max_events_in_grid   # cap columns in action grid

    # ── Input loaders ────────────────────────────────────────────────────────

    @classmethod
    def from_derivation_leg(cls, leg, **kwargs) -> "Phase3HeatmapBuilder":
        """
        Load from a live DerivationLeg after it has completed.
        Uses leg.derivation_log.snapshot() — a list copy. Read-only.
        """
        inst = cls(**kwargs)
        inst._events = leg.derivation_log.snapshot()
        inst._raw_mode = False
        return inst

    @classmethod
    def from_derivation_log(cls, derivation_log, **kwargs) -> "Phase3HeatmapBuilder":
        """Load from a DerivationLog instance directly."""
        inst = cls(**kwargs)
        inst._events = derivation_log.snapshot()
        inst._raw_mode = False
        return inst

    @classmethod
    def from_events(cls, events: list, **kwargs) -> "Phase3HeatmapBuilder":
        """Load from a list of live DerivedEvent objects."""
        inst = cls(**kwargs)
        inst._events = list(events)
        inst._raw_mode = False
        return inst

    @classmethod
    def from_dict(cls, events_dicts: list[dict], **kwargs) -> "Phase3HeatmapBuilder":
        """
        Load from a list of DerivedEvent.to_dict() dicts.
        Used when events are loaded from JSONL or a serialised audit file.
        """
        inst = cls(**kwargs)
        inst._events = events_dicts
        inst._raw_mode = True
        return inst

    # ── Main build ───────────────────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        rows = self._normalise()

        if not rows:
            return self._empty_payload()

        t_min = min(r["timestamp_sec"] for r in rows)
        t_max = max(r["timestamp_sec"] for r in rows)
        duration = max(t_max - t_min, 1e-6)
        bin_width = duration / self._n_time_bins

        return {
            "action_prob_heatmap":  self._build_action_prob_heatmap(rows),
            "tag_frequency":        self._build_tag_frequency(rows),
            "primary_tag_dist":     self._build_primary_tag_dist(rows),
            "score_over_time":      self._build_score_timeline(rows, t_min, bin_width),
            "dwell_distribution":   self._build_dwell_distribution(rows),
            "tag_cooccurrence":     self._build_tag_cooccurrence(rows),
            "track_event_count":    self._build_track_event_count(rows),
            "meta": {
                "n_events":           len(rows),
                "n_tracks":           len({r["track_id"] for r in rows}),
                "n_unique_tags":      len({t for r in rows for t in r["derivation_tags"]}),
                "n_primary_tags":     len({r["primary_tag"] for r in rows if r["primary_tag"]}),
                "t_min_sec":          round(t_min, 3),
                "t_max_sec":          round(t_max, 3),
                "duration_sec":       round(duration, 3),
                "mean_score":         round(sum(r["derivation_score"] for r in rows) / len(rows), 4),
                "mean_dwell_sec":     round(sum(r["dwell_sec"] for r in rows) / len(rows), 3),
                "mean_latency_ms":    round(sum(r["derivation_latency_ms"] for r in rows) / len(rows), 2),
                # Temporal fields remain separate — RISK 3
                "note_temporal":      "timestamp_sec=video time; derived_at_sec=Phase3 wall clock; "
                                      "derivation_latency_ms=per-event cost. Not merged.",
            },
        }

    # ── Action probability heatmap ───────────────────────────────────────────

    def _build_action_prob_heatmap(self, rows: list) -> dict:
        """
        Grid: action class × event index.
        Cell = probability mass for that class in that event.

        RISK 2 compliance:
            Uses source_distribution.class_probs (FULL vector) — NOT primary_tag.
            Each event column shows the complete probability distribution.
            primary_tag is shown only in column metadata as a convenience label.

        Events are ordered by timestamp_sec. Columns are capped at max_events_grid
        to keep the grid readable — a sampling note is included in output.
        """
        # Sort events by video time
        sorted_rows = sorted(rows, key=lambda r: r["timestamp_sec"])

        # Collect all action classes
        all_classes: list[str] = []
        seen: set[str] = set()
        for r in sorted_rows:
            for cls in r.get("class_probs", {}).keys():
                if cls not in seen:
                    all_classes.append(cls)
                    seen.add(cls)
        # Sort classes by mean probability descending for readability
        class_means = {
            cls: sum(r.get("class_probs", {}).get(cls, 0.0) for r in sorted_rows) / len(sorted_rows)
            for cls in all_classes
        }
        all_classes = sorted(all_classes, key=lambda c: -class_means[c])

        # Sample events if too many
        n_events = len(sorted_rows)
        sampled = sorted_rows
        sampled_note = None
        if n_events > self._max_events_grid:
            step = n_events / self._max_events_grid
            sampled = [sorted_rows[int(i * step)] for i in range(self._max_events_grid)]
            sampled_note = f"Showing {self._max_events_grid} of {n_events} events (uniform sample by timestamp)"

        col_labels = [f"E{i+1}" for i in range(len(sampled))]
        # Column metadata: timestamp + primary_tag (convenience) + event_id prefix
        col_meta = [
            {
                "label":       f"E{i+1}",
                "timestamp_sec": round(r["timestamp_sec"], 2),
                "primary_tag": r["primary_tag"],    # convenience field — not authority
                "event_id":    r["event_id"][:8],
                "track_id":    r["track_id"],
            }
            for i, r in enumerate(sampled)
        ]

        # Build values matrix: rows=classes, cols=events
        values = []
        for cls in all_classes:
            row_vals = [round(r.get("class_probs", {}).get(cls, 0.0), 4) for r in sampled]
            values.append(row_vals)

        return {
            "type":      "grid",
            "title":     "Phase 3 — Action probability heatmap (source_distribution × event)",
            "rows":      all_classes,
            "cols":      col_labels,
            "col_meta":  col_meta,
            "values":    values,
            "legend":    ["Low prob", "High prob"],
            "ramp":      ["#fef9c3", "#fde047", "#f59e0b", "#b45309", "#78350f"],
            # Seek map: col_index → timestamp_sec (video time)
            "seek_map":  {str(i): float(round(r["timestamp_sec"], 3)) for i, r in enumerate(sampled)},
            "note":      "Uses full source_distribution.class_probs — NOT primary_tag. "
                         "primary_tag in col_meta is a convenience label only.",
            "sampled_note": sampled_note,
        }

    # ── Tag frequency ────────────────────────────────────────────────────────

    def _build_tag_frequency(self, rows: list) -> dict:
        """
        Count of each derivation_tag across all events.

        derivation_tags is the full tag set (frozenset) — not just primary_tag.
        This counts how many events carry each tag, including events where
        the tag is present but not primary.

        Tag strings are verbatim from DerivedEvent — no re-labeling.
        """
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            for tag in r["derivation_tags"]:
                counts[tag] += 1

        total_events = len(rows)
        ordered = sorted(counts.items(), key=lambda x: -x[1])

        return {
            "type":   "bars",
            "title":  "Phase 3 — Tag occurrence frequency (full derivation_tags set)",
            "rows":   [k for k, _ in ordered],
            "values": [v for _, v in ordered],
            "pct":    [round(v / total_events * 100, 1) for _, v in ordered],
            "raw":    dict(ordered),
            "color":  "#b45309",
            "note":   "Counts events carrying each tag (from full frozenset). "
                      "A single event can contribute to multiple tag counts. "
                      "Separate from primary_tag frequency below.",
        }

    # ── Primary tag distribution ─────────────────────────────────────────────

    def _build_primary_tag_dist(self, rows: list) -> dict:
        """
        Frequency of primary_tag across events.

        primary_tag is a CONVENIENCE field — the highest-priority matched tag
        determined by DerivationConfig.tag_priority order.
        It is explicitly marked as such in the output.

        Events with primary_tag=None are counted separately.
        """
        counts: dict[str, int] = defaultdict(int)
        n_none = 0
        for r in rows:
            pt = r.get("primary_tag")
            if pt:
                counts[pt] += 1
            else:
                n_none += 1

        ordered = sorted(counts.items(), key=lambda x: -x[1])
        if n_none:
            ordered.append(("__no_primary_tag__", n_none))

        total = len(rows)

        return {
            "type":   "bars",
            "title":  "Phase 3 — Primary tag distribution (convenience field)",
            "rows":   [k for k, _ in ordered],
            "values": [v for _, v in ordered],
            "pct":    [round(v / total * 100, 1) for _, v in ordered],
            "color":  "#ea580c",
            "note":   "primary_tag is a derived convenience field (highest-priority tag "
                      "per DerivationConfig.tag_priority). It is NOT the authoritative record. "
                      "Full derivation_tags set is shown separately above.",
        }

    # ── Score over time ──────────────────────────────────────────────────────

    def _build_score_timeline(self, rows: list, t_min: float, bin_width: float) -> dict:
        """
        Mean derivation_score per time bin.

        derivation_score = composite_score * (1 - normalised_entropy_weight).
        It is a structural signal strength indicator — not a behavioral verdict.
        """
        bins_scores: list[list[float]] = [[] for _ in range(self._n_time_bins)]
        for r in rows:
            bi = min(int((r["timestamp_sec"] - t_min) / bin_width), self._n_time_bins - 1)
            bins_scores[bi].append(r["derivation_score"])

        mean_scores = [
            round(sum(b) / len(b), 4) if b else 0.0
            for b in bins_scores
        ]
        counts = [len(b) for b in bins_scores]
        labels = [f"{round(t_min + i * bin_width, 1)} s" for i in range(self._n_time_bins)]

        return {
            "type":    "timeline",
            "title":   "Phase 3 — Mean derivation_score over time",
            "labels":  labels,
            "values":  mean_scores,
            "counts":  counts,
            "color":   "#ea580c",
            "y_label": "derivation_score [0..1]",
            # Seek map uses timestamp_sec (video time) — RISK 3
            "seek_map": {str(i): float(round(t_min + i * bin_width, 3)) for i in range(self._n_time_bins)},
            "note":    "derivation_score = composite_score × (1 − entropy_weight). "
                       "Structural signal indicator — not a behavioral verdict.",
        }

    # ── Dwell distribution ───────────────────────────────────────────────────

    def _build_dwell_distribution(self, rows: list) -> dict:
        """
        Histogram of dwell_sec_at_derivation.

        dwell_sec is a point-in-time snapshot from DerivationHistory at the moment
        of derivation. It is NOT a live accumulator — it captures dwell_sec
        (last_seen_sec - first_seen_sec) at derivation time exactly.
        """
        dwell_values = [r["dwell_sec"] for r in rows]
        max_dwell = max(dwell_values) if dwell_values else 1.0
        bin_width = max_dwell / self._n_dwell_bins if max_dwell > 0 else 1.0

        hist = [0] * self._n_dwell_bins
        for d in dwell_values:
            bi = min(int(d / bin_width), self._n_dwell_bins - 1)
            hist[bi] += 1

        labels = [f"{round(i * bin_width, 1)}–{round((i+1) * bin_width, 1)} s"
                  for i in range(self._n_dwell_bins)]

        return {
            "type":    "histogram",
            "title":   "Phase 3 — Dwell duration at derivation time",
            "labels":  labels,
            "values":  hist,
            "color":   "#7c3aed",
            "x_label": "dwell_sec_at_derivation",
            "y_label": "event count",
            "note":    "dwell_sec_at_derivation is a point-in-time snapshot "
                       "from DerivationHistory — not a running accumulator.",
        }

    # ── Tag co-occurrence matrix ─────────────────────────────────────────────

    def _build_tag_cooccurrence(self, rows: list) -> dict:
        """
        Symmetric matrix: how many events carry BOTH tag_i AND tag_j.

        This is a structural relationship between derivation rules — not a
        behavioral compound category. The matrix is normalised by event count.
        """
        all_tags = sorted({t for r in rows for t in r["derivation_tags"]})
        n = len(all_tags)
        tag_idx = {t: i for i, t in enumerate(all_tags)}

        co: list[list[int]] = [[0] * n for _ in range(n)]
        for r in rows:
            tags = sorted(r["derivation_tags"])
            for ti in tags:
                for tj in tags:
                    if ti in tag_idx and tj in tag_idx:
                        co[tag_idx[ti]][tag_idx[tj]] += 1

        total = len(rows) or 1
        norm_co = [[round(co[i][j] / total, 4) for j in range(n)] for i in range(n)]

        return {
            "type":   "grid",
            "title":  "Phase 3 — Tag co-occurrence matrix",
            "rows":   all_tags,
            "cols":   all_tags,
            "values": norm_co,
            "raw":    co,
            "legend": ["Rare co-occurrence", "Frequent co-occurrence"],
            "ramp":   ["#fff7ed", "#fed7aa", "#f97316", "#c2410c", "#7c2d12"],
            "note":   "Cell (i, j) = fraction of events carrying both tag_i and tag_j. "
                      "Diagonal = single-tag frequency. Structural — not behavioral.",
        }

    # ── Track event count ────────────────────────────────────────────────────

    def _build_track_event_count(self, rows: list) -> dict:
        """
        Histogram of DerivedEvent count per track_id.

        track_id is a continuity hypothesis identifier from Phase 1.
        High event count per track means the track generated many
        structural transitions — not that it is behaviorally significant.
        """
        per_track: dict[int, int] = defaultdict(int)
        for r in rows:
            per_track[r["track_id"]] += 1

        counts = list(per_track.values())
        if not counts:
            return {"type": "histogram", "labels": [], "values": []}

        max_count = max(counts)
        n_bins = min(max_count, 10)
        bin_width = max(max_count / n_bins, 1)

        hist = [0] * n_bins
        for c in counts:
            bi = min(int((c - 1) / bin_width), n_bins - 1)
            hist[bi] += 1

        labels = [f"{int(i * bin_width + 1)}–{int((i+1) * bin_width)}" for i in range(n_bins)]

        return {
            "type":    "histogram",
            "title":   "Phase 3 — Events per track",
            "labels":  labels,
            "values":  hist,
            "color":   "#0891b2",
            "raw_per_track": {str(k): v for k, v in per_track.items()},
            "x_label": "derived events",
            "y_label": "track count",
            "note":    "track_id is a continuity hypothesis (Phase 1). "
                       "High event count = many structural transitions observed.",
        }

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalise(self) -> list[dict]:
        raw_mode = getattr(self, "_raw_mode", False)
        out = []
        for e in self._events:
            if raw_mode or isinstance(e, dict):
                out.append(self._normalise_raw(e))
            else:
                out.append(self._normalise_event(e))
        # Async workers may commit events out-of-order — re-sort by absolute
        # timestamp_sec before any binning occurs (RISK 3 preserved).
        out.sort(key=lambda r: r["timestamp_sec"])
        return out

    def _normalise_event(self, e) -> dict:
        """Extract from a live DerivedEvent object."""
        return {
            "event_id":            e.event_id,
            "track_id":            int(e.track_id),
            "timestamp_sec":       float(e.timestamp_sec),
            "derived_at_sec":      float(e.derived_at_sec),          # RISK 3 — separate
            "derivation_latency_ms": float(e.derivation_latency_ms), # RISK 3 — separate
            "derivation_tags":     set(e.derivation_tags),
            "primary_tag":         e.primary_tag,          # convenience field
            "derivation_score":    float(e.derivation_score),
            "dwell_sec":           float(e.dwell_sec_at_derivation),  # point-in-time snapshot
            "class_probs":         dict(e.source_distribution.class_probs),  # FULL vector
            "composite_score":     float(e.composite_score),
            "identity_stability":  float(e.identity_stability),
            "delta_reason":        e.delta_reason,
        }

    def _normalise_raw(self, e: dict) -> dict:
        """Extract from a DerivedEvent.to_dict() dict."""
        cp = e.get("source_distribution", {})
        return {
            "event_id":            e.get("event_id", ""),
            "track_id":            int(e.get("track_id", -1)),
            "timestamp_sec":       float(e.get("timestamp_sec", 0.0)),
            "derived_at_sec":      float(e.get("derived_at_sec", 0.0)),
            "derivation_latency_ms": float(e.get("derivation_latency_ms", 0.0)),
            "derivation_tags":     set(e.get("derivation_tags", [])),
            "primary_tag":         e.get("primary_tag"),
            "derivation_score":    float(e.get("derivation_score", 0.0)),
            "dwell_sec":           float(e.get("dwell_sec_at_derivation", 0.0)),
            "class_probs":         cp if isinstance(cp, dict) else {},
            "composite_score":     float(e.get("composite_score", 0.0)),
            "identity_stability":  float(e.get("identity_stability", 1.0)),
            "delta_reason":        e.get("delta_reason", ""),
        }

    def _empty_payload(self) -> dict:
        return {
            "action_prob_heatmap":  {"type": "grid",      "rows": [], "cols": [], "values": []},
            "tag_frequency":        {"type": "bars",       "rows": [], "values": []},
            "primary_tag_dist":     {"type": "bars",       "rows": [], "values": []},
            "score_over_time":      {"type": "timeline",   "labels": [], "values": []},
            "dwell_distribution":   {"type": "histogram",  "labels": [], "values": []},
            "tag_cooccurrence":     {"type": "grid",       "rows": [], "cols": [], "values": []},
            "track_event_count":    {"type": "histogram",  "labels": [], "values": []},
            "meta": {"n_events": 0},
        }


# ---------------------------------------------------------------------------
# FastAPI-ready endpoint helper
# ---------------------------------------------------------------------------

def build_phase3_heatmap(source, input_duration_sec: float | None = None,
                         output_duration_sec: float | None = None, **kwargs) -> dict:
    """
    Single entry point for the FastAPI endpoint.

    source:
        DerivationLeg instance          → from_derivation_leg()
        DerivationLog instance          → from_derivation_log()
        list[DerivedEvent]              → from_events()
        list[dict] (serialised)         → from_dict()
        None                            → empty payload

    kwargs:
        n_time_bins (int, default 8)
        n_dwell_bins (int, default 10)
        max_events_in_grid (int, default 10)
    """
    if source is None:
        return Phase3HeatmapBuilder(**kwargs).build()

    if isinstance(source, list):
        if source and isinstance(source[0], dict):
            builder = Phase3HeatmapBuilder.from_dict(source, **kwargs)
        else:
            builder = Phase3HeatmapBuilder.from_events(source, **kwargs)
    else:
        # Try derivation_log attribute (DerivationLeg) then direct log
        if hasattr(source, "derivation_log"):
            builder = Phase3HeatmapBuilder.from_derivation_leg(source, **kwargs)
        else:
            builder = Phase3HeatmapBuilder.from_derivation_log(source, **kwargs)

    payload = builder.build()
    from seek_utils import scale_all_seek_maps
    return scale_all_seek_maps(payload, input_duration_sec, output_duration_sec)


# ---------------------------------------------------------------------------
# Standalone test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 3 Heatmap Builder")
    parser.add_argument("--input",  required=True,
                        help="Path to a JSONL or JSON file of DerivedEvent.to_dict() records")
    parser.add_argument("--out",    default="heatmap_phase3.json")
    parser.add_argument("--format", choices=["jsonl", "json"], default="json")
    parser.add_argument("--tbins",  type=int, default=8)
    parser.add_argument("--dbins",  type=int, default=10)
    parser.add_argument("--grid",   type=int, default=10,
                        help="Max events shown in action probability grid")
    args = parser.parse_args()

    events_raw = []
    if args.format == "jsonl":
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events_raw.append(json.loads(line))
    else:
        with open(args.input, encoding="utf-8") as f:
            events_raw = json.load(f)

    payload = build_phase3_heatmap(
        source=events_raw,
        n_time_bins=args.tbins,
        n_dwell_bins=args.dbins,
        max_events_in_grid=args.grid,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=list)

    meta = payload["meta"]
    print(f"Phase 3 heatmap built.")
    print(f"  Events      : {meta.get('n_events', 0)}")
    print(f"  Tracks      : {meta.get('n_tracks', 0)}")
    print(f"  Unique tags : {meta.get('n_unique_tags', 0)}")
    print(f"  Mean score  : {meta.get('mean_score', 0):.4f}")
    print(f"  Mean dwell  : {meta.get('mean_dwell_sec', 0):.2f} s")
    print(f"  Output      : {args.out}")
