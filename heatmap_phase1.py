"""
heatmap_phase1.py — Phase 1 Observation Heatmap Visualizer
===========================================================
Reads from Phase 1's output structures WITHOUT semantic leakage:

    Risk audit:
        RISK 1 — No behavioral labels. All axes are structural:
                 zone_id (spatial geometry), timestamp_sec (temporal position),
                 delta_reason (structural transition type), track_id (continuity id).
        RISK 2 — ActionDistribution is consumed in FULL. The heatmap reads
                 class_probs (full vector) and entropy — never top_label alone.
                 Axes that use top_label mark it as a *derived convenience axis*
                 and always offer the full distribution via hover/detail data.
        RISK 3 — All temporal axes use absolute timestamp_sec from the candidate.
                 No relative durations are inferred. Bin boundaries are returned
                 as absolute seconds, not reconstructed offsets.
        RISK 4 — This module is READ-ONLY. It reads from Blackboard._candidate_log
                 (a list snapshot) or a pre-serialised phase1_candidates.json.
                 It never writes to Blackboard, never mutates Candidate objects.
        RISK 5 — DeltaReason values are passed through verbatim from the enum.
                 No new behavioral taxonomy is introduced.

Input sources (in priority order):
    1. Blackboard instance  (live, post-run)          via from_blackboard()
    2. phase1_candidates.json  (serialised output)    via from_json_file()
    3. list[Candidate]  (direct)                      via from_candidates()

Output: dict  (serialisation-ready; keys match page.tsx HEATMAP_DEFS[1] schema)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal types — structural only, no behavioral vocabulary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ZoneBin:
    """One cell in the zone × time grid. Structural counting only."""
    zone_id:     str
    bin_index:   int    # temporal bin index
    bin_start_sec: float
    bin_end_sec:   float
    observation_count: int          # raw count of candidates in this cell
    mean_entropy:      float        # mean ActionDistribution entropy in this cell
    mean_composite:    float        # mean composite_score in this cell
    delta_reason_counts: dict       # {DeltaReason.value: count}
    # Full distribution aggregate — never compressed to a label
    mean_class_probs:  dict[str, float]  # mean prob per action class across candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Phase1HeatmapBuilder:
    """
    Builds all Phase 1 heatmap data from candidates.

    Does NOT inspect top_label for any heatmap axis.
    All action-related axes operate on the full class_probs dict or on entropy.

    Heatmaps produced:
        1. zone_time_density   — zone × time grid (observation count)
        2. delta_reason_counts — bar chart of structural transition types
        3. entropy_over_time   — mean distribution entropy per time bin
        4. track_stability     — identity_stability histogram per track
        5. class_prob_means    — mean probability per action class (full vector)
    """

    def __init__(self, n_time_bins: int = 7) -> None:
        self._n_bins = n_time_bins

    # ── Input loaders ────────────────────────────────────────────────────────

    @classmethod
    def from_blackboard(cls, bb, n_time_bins: int = 7) -> "Phase1HeatmapBuilder":
        """
        Load from a live Blackboard after Phase 1 has completed.
        Reads _candidate_log as a snapshot — no mutation.
        """
        inst = cls(n_time_bins)
        inst._candidates = list(bb._candidate_log)
        return inst

    @classmethod
    def from_json_file(cls, path: str | Path, n_time_bins: int = 7) -> "Phase1HeatmapBuilder":
        """
        Load from phase1_candidates.json (written by __main__ in phase1_observation.py).
        Reconstructs the minimal fields needed for heatmap computation.
        Does NOT reconstruct ActionDistribution objects — operates on the serialised dicts.
        """
        inst = cls(n_time_bins)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        inst._raw_candidates = data.get("candidates", [])
        inst._meta = data.get("meta", {})
        inst._candidates = None   # raw path: use _raw_candidates
        return inst

    @classmethod
    def from_candidates(cls, candidates: list, n_time_bins: int = 7) -> "Phase1HeatmapBuilder":
        """Load from a list of live Candidate objects."""
        inst = cls(n_time_bins)
        inst._candidates = list(candidates)
        return inst

    # ── Main build ───────────────────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        """
        Build all heatmap payloads.

        Returns a dict with keys:
            zone_time_density   → HEATMAP_DEFS[1] grid schema
            delta_reason_counts → bars schema
            entropy_over_time   → timeline schema
            track_stability     → histogram schema
            class_prob_means    → full distribution summary
            meta                → structural metadata (no semantic labels)
        """
        rows = self._normalise()

        if not rows:
            return self._empty_payload()

        t_min = min(r["timestamp_sec"] for r in rows)
        t_max = max(r["timestamp_sec"] for r in rows)
        duration = max(t_max - t_min, 1e-6)
        bin_width = duration / self._n_bins

        # ── 1. Zone × time density grid ──────────────────────────────────────
        zone_time = self._build_zone_time(rows, t_min, bin_width)

        # ── 2. Delta reason distribution (structural transitions only) ────────
        delta_counts = self._build_delta_reason_counts(rows)

        # ── 3. Entropy over time (distribution uncertainty profile) ───────────
        entropy_timeline = self._build_entropy_timeline(rows, t_min, bin_width)

        # ── 4. Track identity stability distribution ───────────────────────────
        stability_hist = self._build_stability_histogram(rows)

        # ── 5. Mean class probability vector (full — never top_label only) ─────
        class_prob_means = self._build_class_prob_means(rows)

        return {
            "zone_time_density":   zone_time,
            "delta_reason_counts": delta_counts,
            "entropy_over_time":   entropy_timeline,
            "track_stability":     stability_hist,
            "class_prob_means":    class_prob_means,
            "meta": {
                # Structural counts — no behavioral labels
                "n_candidates":    len(rows),
                "n_tracks":        len({r["track_id"] for r in rows}),
                "n_zones":         len({z for r in rows for z in r["zone_contacts"]}),
                "t_min_sec":       round(t_min, 3),
                "t_max_sec":       round(t_max, 3),
                "duration_sec":    round(duration, 3),
                "n_time_bins":     self._n_bins,
                "bin_width_sec":   round(bin_width, 3),
            },
        }

    # ── Zone × time density ─────────────────────────────────────────────────

    def _build_zone_time(self, rows, t_min, bin_width) -> dict:
        """
        Grid: rows = zone_ids, cols = time bins.
        Cell value = normalised observation count [0..1].

        Normalisation is per-grid-max so relative density is preserved.
        No zone_type semantics are used — zone_id is the only axis.
        """
        # Collect all zone_ids (may include candidates with no zone contact)
        all_zones = sorted({z for r in rows for z in (r["zone_contacts"] or ["__no_zone__"])})
        bins = [round(t_min + i * bin_width, 2) for i in range(self._n_bins)]

        # count[zone][bin_idx] = candidate count
        count: dict[str, list[int]] = {z: [0] * self._n_bins for z in all_zones}
        entropy_acc: dict[str, list[float]] = {z: [0.0] * self._n_bins for z in all_zones}
        n_acc: dict[str, list[int]] = {z: [0] * self._n_bins for z in all_zones}

        for r in rows:
            bi = min(int((r["timestamp_sec"] - t_min) / bin_width), self._n_bins - 1)
            zones = r["zone_contacts"] or ["__no_zone__"]
            for z in zones:
                if z in count:
                    count[z][bi] += 1
                    entropy_acc[z][bi] += r.get("entropy", 0.0)
                    n_acc[z][bi] += 1

        # Normalise to [0..1] by grid maximum
        grid_max = max((v for row in count.values() for v in row), default=1)
        norm_values = [
            [round(count[z][bi] / max(grid_max, 1), 4) for bi in range(self._n_bins)]
            for z in all_zones
        ]

        # Mean entropy per cell (structural — not behavioral)
        entropy_values = [
            [
                round(entropy_acc[z][bi] / max(n_acc[z][bi], 1), 4)
                for bi in range(self._n_bins)
            ]
            for z in all_zones
        ]

        col_labels = [f"{round(t_min + i * bin_width, 1)} s" for i in range(self._n_bins)]

        return {
            "type":         "grid",
            "title":        "Phase 1 — Observation density (zone × time)",
            "rows":         all_zones,
            "cols":         col_labels,
            "values":       norm_values,        # [0..1] normalised count
            "raw_counts":   [[count[z][bi] for bi in range(self._n_bins)] for z in all_zones],
            "entropy_grid": entropy_values,     # mean entropy — measures distribution uncertainty
            "legend":       ["Low density", "High density"],
            "ramp":         ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"],
            # Seek map: col_index → timestamp_sec for video synchronisation
            "seek_map":     {str(i): float(round(t_min + i * bin_width, 3)) for i in range(self._n_bins)},
        }

    # ── Delta reason counts ──────────────────────────────────────────────────

    def _build_delta_reason_counts(self, rows) -> dict:
        """
        Bar chart of DeltaReason structural transitions.

        DeltaReason values are purely structural (ZONE_ENTRY, SPATIAL_DELTA…).
        No behavioral vocabulary is introduced here.
        """
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["delta_reason"]] += 1

        total = sum(counts.values()) or 1
        ordered = sorted(counts.items(), key=lambda x: -x[1])

        return {
            "type":   "bars",
            "title":  "Phase 1 — Structural transition counts (DeltaReason)",
            "rows":   [k for k, _ in ordered],
            "values": [round(v / total * 100, 1) for _, v in ordered],
            "raw":    {k: v for k, v in ordered},
            "color":  "#2563eb",
            "note":   "DeltaReason describes structural state transitions only — no behavioral categories",
        }

    # ── Entropy over time ────────────────────────────────────────────────────

    def _build_entropy_timeline(self, rows, t_min, bin_width) -> dict:
        """
        Mean ActionDistribution entropy per time bin.

        High entropy = behavioral uncertainty (distribution spread across classes).
        Low entropy = concentrated distribution — one class dominates.

        This measures uncertainty in the distribution — NOT a behavioral verdict.
        The full class_probs dict is the primary record; entropy is a derived summary.
        """
        bins_entropy: list[list[float]] = [[] for _ in range(self._n_bins)]
        for r in rows:
            bi = min(int((r["timestamp_sec"] - t_min) / bin_width), self._n_bins - 1)
            bins_entropy[bi].append(r.get("entropy", 0.0))

        mean_entropy = [
            round(sum(e) / len(e), 4) if e else 0.0
            for e in bins_entropy
        ]
        labels = [f"{round(t_min + i * bin_width, 1)} s" for i in range(self._n_bins)]

        return {
            "type":   "timeline",
            "title":  "Phase 1 — Mean distribution entropy over time",
            "labels": labels,
            "values": mean_entropy,
            "color":  "#7c3aed",
            "y_label": "entropy (nats)",
            "note":   "High entropy = behavioral uncertainty. Full class_probs is primary record.",
            "seek_map": {str(i): float(round(t_min + i * bin_width, 3)) for i in range(self._n_bins)},
        }

    # ── Track stability histogram ────────────────────────────────────────────

    def _build_stability_histogram(self, rows) -> dict:
        """
        Distribution of identity_stability values across all candidates.

        identity_stability [0..1] is the mean OKS to the previous frame.
        This is a CONTINUITY HYPOTHESIS — not a confirmed identity.
        The histogram shows the stability profile of all continuity hypotheses.
        """
        # 10 bins from 0.0 to 1.0
        n_bins = 10
        hist = [0] * n_bins
        for r in rows:
            s = max(0.0, min(1.0, r.get("identity_stability", 1.0)))
            bi = min(int(s * n_bins), n_bins - 1)
            hist[bi] += 1

        labels = [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)]
        total = sum(hist) or 1

        return {
            "type":    "histogram",
            "title":   "Phase 1 — Identity stability distribution",
            "labels":  labels,
            "values":  hist,
            "pct":     [round(v / total * 100, 1) for v in hist],
            "color":   "#0891b2",
            "x_label": "identity_stability [0..1]",
            "y_label": "candidate count",
            "note":    "Tracks are continuity hypotheses, not confirmed identities. "
                       "Low stability = possible track jump.",
        }

    # ── Class probability means ──────────────────────────────────────────────

    def _build_class_prob_means(self, rows) -> dict:
        """
        Mean probability per action class across all candidates.

        Operates on the FULL class_probs dict — not on top_label.
        This preserves the entire distribution rather than the dominant class.

        RISK 2 compliance:
            top_label is NEVER used here. The aggregation sums class_probs[class]
            across all candidates and divides by count. The result is a full
            mean distribution vector, not a frequency count of top labels.
        """
        accum: dict[str, float] = defaultdict(float)
        n = 0
        for r in rows:
            cp = r.get("class_probs", {})
            if cp:
                for cls, prob in cp.items():
                    accum[cls] += prob
                n += 1

        if n == 0:
            return {"type": "bars", "title": "Phase 1 — Mean action class probabilities",
                    "rows": [], "values": [], "color": "#2563eb"}

        mean_probs = {cls: round(v / n, 4) for cls, v in accum.items()}
        ordered = sorted(mean_probs.items(), key=lambda x: -x[1])

        return {
            "type":   "bars",
            "title":  "Phase 1 — Mean action class probabilities (full distribution)",
            "rows":   [k for k, _ in ordered],
            "values": [round(v * 100, 2) for _, v in ordered],
            "raw":    mean_probs,
            "color":  "#0284c7",
            "note":   "Aggregated from full class_probs vectors — NOT from top_label counts. "
                      "Preserves the full distribution across all candidates.",
        }

    # ── Normalisation layer ──────────────────────────────────────────────────

    def _normalise(self) -> list[dict]:
        """
        Produce a uniform list of dicts from either live Candidates or raw JSON.
        Extracts only the fields needed for heatmap computation.

        Fields extracted:
            timestamp_sec, track_id, zone_contacts, delta_reason,
            identity_stability, composite_score,
            entropy (derived from class_probs — NOT top_label),
            class_probs (FULL probability dict)

        RISK 2: entropy is computed here from class_probs.
                top_label is deliberately not extracted at all in this path.
        """
        if hasattr(self, "_candidates") and self._candidates is not None:
            rows = [self._normalise_candidate(c) for c in self._candidates]
        elif hasattr(self, "_raw_candidates"):
            rows = [self._normalise_raw(r) for r in self._raw_candidates]
        else:
            return []
        # Async workers may commit candidates out-of-order — re-sort by
        # absolute timestamp_sec before any binning occurs (RISK 3 preserved:
        # values remain absolute timestamp_sec, only ordering changes).
        rows.sort(key=lambda r: r["timestamp_sec"])
        return rows

    def _normalise_candidate(self, c) -> dict:
        """Extract from a live Candidate object. Reads class_probs directly."""
        cp = dict(c.action_distribution.class_probs)
        entropy = float(c.action_distribution.entropy)
        return {
            "timestamp_sec":      float(c.timestamp_sec),
            "track_id":           int(c.track_id),
            "zone_contacts":      list(c.zone_contacts),
            "delta_reason":       c.delta_reason.value,
            "identity_stability": float(c.identity_stability),
            "composite_score":    float(c.composite_score),
            "class_probs":        cp,
            "entropy":            entropy,
        }

    def _normalise_raw(self, r: dict) -> dict:
        """
        Extract from a serialised candidate dict (from phase1_candidates.json).

        RISK 2: entropy is recomputed from the serialised class_probs vector.
                top_label is never read.
        """
        cp = r.get("action_distribution", {})
        # Recompute entropy from the full distribution — not from any stored label
        entropy = _entropy_from_probs(cp)
        return {
            "timestamp_sec":      float(r.get("timestamp_sec", 0.0)),
            "track_id":           int(r.get("track_id", -1)),
            "zone_contacts":      list(r.get("zone_contacts", [])),
            "delta_reason":       r.get("delta_reason", "UNKNOWN"),
            "identity_stability": float(r.get("identity_stability", 1.0)),
            "composite_score":    float(r.get("composite_score", 0.0)),
            "class_probs":        cp,
            "entropy":            entropy,
        }

    def _empty_payload(self) -> dict:
        return {
            "zone_time_density":   {"type": "grid", "rows": [], "cols": [], "values": []},
            "delta_reason_counts": {"type": "bars", "rows": [], "values": []},
            "entropy_over_time":   {"type": "timeline", "labels": [], "values": []},
            "track_stability":     {"type": "histogram", "labels": [], "values": []},
            "class_prob_means":    {"type": "bars", "rows": [], "values": []},
            "meta": {"n_candidates": 0},
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _entropy_from_probs(class_probs: dict) -> float:
    """Shannon entropy from a probability dict. Matches ActionDistribution.entropy."""
    import math
    vals = [v for v in class_probs.values() if isinstance(v, (int, float)) and v > 1e-10]
    if not vals:
        return 0.0
    return float(-sum(p * math.log(p + 1e-10) for p in vals))


# ---------------------------------------------------------------------------
# FastAPI-ready endpoint helper
# ---------------------------------------------------------------------------

def build_phase1_heatmap(
    source: "Blackboard | str | Path | list | None" = None,
    n_time_bins: int = 7,
    input_duration_sec: float | None = None,
    output_duration_sec: float | None = None,
) -> dict:
    """
    Single entry point for the FastAPI endpoint.

    source:
        Blackboard instance     → from_blackboard()
        str / Path              → from_json_file()
        list                    → from_candidates()
        None                    → returns empty payload with schema

    Returns the full heatmap payload dict, JSON-serialisable.
    """
    if source is None:
        return Phase1HeatmapBuilder(n_time_bins).build()

    if isinstance(source, (str, Path)):
        builder = Phase1HeatmapBuilder.from_json_file(source, n_time_bins)
    elif isinstance(source, list):
        builder = Phase1HeatmapBuilder.from_candidates(source, n_time_bins)
    else:
        # Assume Blackboard instance
        builder = Phase1HeatmapBuilder.from_blackboard(source, n_time_bins)

    payload = builder.build()
    from seek_utils import scale_all_seek_maps
    return scale_all_seek_maps(payload, input_duration_sec, output_duration_sec)


# ---------------------------------------------------------------------------
# Standalone test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 1 Heatmap Builder")
    parser.add_argument("--input", required=True,
                        help="Path to phase1_candidates.json")
    parser.add_argument("--bins",  type=int, default=7,
                        help="Number of time bins (default: 7)")
    parser.add_argument("--out",   default="heatmap_phase1.json",
                        help="Output JSON path")
    args = parser.parse_args()

    payload = build_phase1_heatmap(source=args.input, n_time_bins=args.bins)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    meta = payload["meta"]
    print(f"Phase 1 heatmap built.")
    print(f"  Candidates  : {meta.get('n_candidates', 0)}")
    print(f"  Tracks      : {meta.get('n_tracks', 0)}")
    print(f"  Zones       : {meta.get('n_zones', 0)}")
    print(f"  Duration    : {meta.get('duration_sec', 0):.1f} s")
    print(f"  Output      : {args.out}")
