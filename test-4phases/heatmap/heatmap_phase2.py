"""
heatmap_phase2.py — Phase 2 Filter Heatmap Visualizer
======================================================
Reads from Phase 2's FilterBlackboardExtension, FilterStats, and FilterDecision
audit log WITHOUT adding semantic leakage.

    Risk audit:
        RISK 1 — All axes are structural filter properties:
                 SuppressComponent names (LOW_COMPOSITE_SCORE, POOR_POSE_QUALITY…),
                 FilterVerdict (PROMOTED / SUPPRESSED), cooldown state, zone contacts.
                 No behavioral vocabulary (no "threat", "alert", "suspicious") added.
        RISK 2 — FilterDecision.action_distribution (full class_probs) is read
                 for the action distribution heatmap axis. action_top is used ONLY
                 as a cooldown key label — never as an action verdict.
                 composite_score is read as a filter signal, not a behavioral score.
        RISK 3 — processed_at_sec (wall-clock monotonic) and timestamp_sec (video time)
                 are kept separate. No merging. Temporal bins use timestamp_sec for
                 video-synchronised axes and processed_at_sec for latency metrics.
        RISK 4 — This module is READ-ONLY. It calls get_decisions() which returns
                 a list copy (inside FilterBlackboardExtension._lock).
                 It never writes to Blackboard, FilterBlackboardExtension, or
                 FilterEngine state. FilterStats.to_dict() is called for a snapshot.
        RISK 5 — SuppressComponent values are passed through verbatim.
                 No re-mapping or enrichment of reason taxonomy.

Input sources:
    1. FilterBlackboardExtension + FilterStats   (live, post-run)   via from_filter_leg()
    2. list[FilterDecision]  (direct)                               via from_decisions()
    3. dict  (pre-serialised FilterStats.to_dict() + decision list) via from_dict()

Output: dict  (JSON-serialisable; matches page.tsx HEATMAP_DEFS[2] schema)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Phase2HeatmapBuilder:
    """
    Builds all Phase 2 heatmap payloads from FilterDecision audit records.

    Heatmaps produced:
        1. rule_pass_rates          — bar chart: pass-rate % per filter rule
        2. suppress_reason_breakdown— bar chart: count per SuppressComponent
        3. composite_score_dist     — histogram: composite_score distribution
                                      across promoted vs suppressed candidates
        4. action_dist_heatmap      — grid: action class × promote/suppress outcome
                                      uses full class_probs, NOT top_label
        5. zone_filter_rates        — bar chart: promotion rate per zone
        6. latency_timeline         — timeline: filter decision latency over time
    """

    def __init__(self, n_score_bins: int = 10, n_time_bins: int = 7) -> None:
        self._n_score_bins = n_score_bins
        self._n_time_bins  = n_time_bins

    # ── Input loaders ────────────────────────────────────────────────────────

    @classmethod
    def from_filter_leg(cls, filter_leg, n_score_bins: int = 10,
                        n_time_bins: int = 7) -> "Phase2HeatmapBuilder":
        """
        Load from a live FilterLeg after it has run.
        Reads bb_ext.get_decisions() — a locked list copy. Read-only.
        """
        inst = cls(n_score_bins, n_time_bins)
        inst._decisions = filter_leg.bb_ext.get_decisions()
        inst._stats_dict = filter_leg.stats.to_dict()
        return inst

    @classmethod
    def from_decisions(cls, decisions: list, stats_dict: dict | None = None,
                       n_score_bins: int = 10, n_time_bins: int = 7) -> "Phase2HeatmapBuilder":
        """Load from a list of live FilterDecision objects."""
        inst = cls(n_score_bins, n_time_bins)
        inst._decisions  = list(decisions)
        inst._stats_dict = stats_dict or {}
        return inst

    @classmethod
    def from_dict(cls, decisions_dicts: list[dict], stats_dict: dict | None = None,
                  n_score_bins: int = 10, n_time_bins: int = 7) -> "Phase2HeatmapBuilder":
        """
        Load from a list of FilterDecision.to_dict() outputs.
        Used when decisions are loaded from a serialised audit log.
        """
        inst = cls(n_score_bins, n_time_bins)
        inst._decisions  = decisions_dicts   # raw dicts — handled in _normalise()
        inst._stats_dict = stats_dict or {}
        inst._raw_mode   = True
        return inst

    # ── Main build ───────────────────────────────────────────────────────────

    def build(self) -> dict[str, Any]:
        rows = self._normalise()

        if not rows:
            return self._empty_payload()

        promoted   = [r for r in rows if r["verdict"] == "PROMOTED"]
        suppressed = [r for r in rows if r["verdict"] == "SUPPRESSED"]

        return {
            "rule_pass_rates":          self._build_rule_pass_rates(rows, promoted, suppressed),
            "suppress_reason_breakdown":self._build_suppress_reasons(suppressed),
            "composite_score_dist":     self._build_score_distribution(promoted, suppressed),
            "action_dist_heatmap":      self._build_action_dist_heatmap(rows),
            "zone_filter_rates":        self._build_zone_filter_rates(rows),
            "latency_timeline":         self._build_latency_timeline(rows),
            "meta": {
                "total_evaluated":  len(rows),
                "total_promoted":   len(promoted),
                "total_suppressed": len(suppressed),
                "promotion_rate":   round(len(promoted) / max(len(rows), 1), 4),
                "n_score_bins":     self._n_score_bins,
                "n_time_bins":      self._n_time_bins,
                # from FilterStats if available
                "cycles_run":       self._stats_dict.get("cycles_run", None),
            },
        }

    # ── Rule pass rates ──────────────────────────────────────────────────────

    def _build_rule_pass_rates(self, rows, promoted, suppressed) -> dict:
        """
        Pass-rate % for each of the 6 filter rules.

        Pass-rate = 1 - (candidates where this rule fired / total evaluated).
        A rule "fires" when the corresponding SuppressComponent appears in
        suppress_reasons (regardless of whether the final verdict was SUPPRESSED —
        in permissive mode, soft-rule failures appear even on PROMOTED decisions).

        Mapping SuppressComponent → rule label uses verbatim enum values.
        No behavioral re-labeling.
        """
        total = len(rows) or 1
        # Count how many candidates had each SuppressComponent in suppress_reasons
        rule_fire_count: dict[str, int] = defaultdict(int)
        for r in rows:
            for reason in r["suppress_reasons"]:
                rule_fire_count[reason] += 1

        # Canonical rule order from phase2_filter.py — structural names only
        RULE_ORDER = [
            "LOW_COMPOSITE_SCORE",
            "POOR_POSE_QUALITY",
            "UNSTABLE_TRACKING",
            "NO_ZONE_CONTACT",
            "COOLDOWN_ACTIVE",
            "DUPLICATE_TRACK_ZONE",
        ]

        pass_rates = []
        for rule in RULE_ORDER:
            fired = rule_fire_count.get(rule, 0)
            pass_rates.append(round((1 - fired / total) * 100, 1))

        return {
            "type":   "bars",
            "title":  "Phase 2 — Filter rule pass rates",
            "rows":   RULE_ORDER,
            "values": pass_rates,
            "raw_fire_counts": {r: rule_fire_count.get(r, 0) for r in RULE_ORDER},
            "total_evaluated": total,
            "color":  "#16a34a",
            "note":   "Pass rate = fraction of candidates where this rule did not fire. "
                      "Includes soft-rule fires on PROMOTED candidates in permissive mode.",
        }

    # ── Suppress reason breakdown ────────────────────────────────────────────

    def _build_suppress_reasons(self, suppressed: list) -> dict:
        """
        Count each SuppressComponent across all SUPPRESSED decisions.

        Uses only the SUPPRESSED subset so counts reflect actual suppressions,
        not permissive-mode soft fires on promoted candidates.
        (Matches FilterStats.record() behaviour — RISK 5 compliance.)
        """
        counts: dict[str, int] = defaultdict(int)
        for r in suppressed:
            for reason in r["suppress_reasons"]:
                counts[reason] += 1

        total = sum(counts.values()) or 1
        ordered = sorted(counts.items(), key=lambda x: -x[1])

        return {
            "type":   "bars",
            "title":  "Phase 2 — Suppression reasons (SUPPRESSED candidates only)",
            "rows":   [k for k, _ in ordered],
            "values": [v for _, v in ordered],
            "pct":    [round(v / total * 100, 1) for _, v in ordered],
            "raw":    dict(ordered),
            "color":  "#dc2626",
            "note":   "Counts are from SUPPRESSED decisions only. "
                      "Soft-rule fires on PROMOTED candidates are excluded (see audit log).",
        }

    # ── Composite score distribution ─────────────────────────────────────────

    def _build_score_distribution(self, promoted: list, suppressed: list) -> dict:
        """
        Histogram of composite_score for promoted vs suppressed candidates.

        composite_score is a derived convenience field from ConfidenceProfile.composite().
        It is NOT the primary record — ConfidenceProfile is.
        The histogram shows filter behaviour relative to this structural signal.
        """
        bins = self._n_score_bins
        prom_hist  = [0] * bins
        supp_hist  = [0] * bins

        for r in promoted:
            bi = min(int(r["composite_score"] * bins), bins - 1)
            prom_hist[bi] += 1
        for r in suppressed:
            bi = min(int(r["composite_score"] * bins), bins - 1)
            supp_hist[bi] += 1

        labels = [f"{i/bins:.1f}–{(i+1)/bins:.1f}" for i in range(bins)]

        return {
            "type":            "histogram_dual",
            "title":           "Phase 2 — Composite score distribution (promoted vs suppressed)",
            "labels":          labels,
            "promoted_values": prom_hist,
            "suppressed_values": supp_hist,
            "x_label":         "composite_score [0..1]",
            "y_label":         "candidate count",
            "color_promoted":  "#16a34a",
            "color_suppressed": "#dc2626",
            "note":            "composite_score is a derived convenience field. "
                               "ConfidenceProfile (4-dim) is the primary record.",
        }

    # ── Action distribution heatmap ──────────────────────────────────────────

    def _build_action_dist_heatmap(self, rows: list) -> dict:
        """
        Grid: action class × verdict (PROMOTED / SUPPRESSED).
        Cell value = mean probability mass for that class within that verdict group.

        RISK 2 compliance:
            Uses class_probs (full probability vector) — NOT top_label.
            A candidate may have significant probability mass on multiple classes.
            Aggregating by top_label would destroy that information.
            This heatmap shows the full mean distribution split by verdict.
        """
        verdicts = ["PROMOTED", "SUPPRESSED"]
        # Collect all known classes
        all_classes: set[str] = set()
        for r in rows:
            all_classes.update(r.get("class_probs", {}).keys())
        all_classes = sorted(all_classes)

        if not all_classes:
            return {"type": "grid", "rows": [], "cols": verdicts, "values": []}

        # Accumulate probability mass per (class, verdict)
        accum: dict[str, dict[str, float]] = {
            v: {cls: 0.0 for cls in all_classes} for v in verdicts
        }
        count: dict[str, int] = {v: 0 for v in verdicts}

        for r in rows:
            v = r["verdict"]
            if v not in accum:
                continue
            cp = r.get("class_probs", {})
            for cls in all_classes:
                accum[v][cls] += cp.get(cls, 0.0)
            count[v] += 1

        # Normalise to mean probability
        mean_probs: dict[str, dict[str, float]] = {}
        for v in verdicts:
            n = max(count[v], 1)
            mean_probs[v] = {cls: round(accum[v][cls] / n, 4) for cls in all_classes}

        # Build values matrix: rows=classes, cols=verdicts
        values = [
            [mean_probs[v].get(cls, 0.0) for v in verdicts]
            for cls in all_classes
        ]

        return {
            "type":   "grid",
            "title":  "Phase 2 — Mean action probability by filter verdict",
            "rows":   all_classes,
            "cols":   verdicts,
            "values": values,
            "legend": ["Low prob", "High prob"],
            "ramp":   ["#f0fdf4", "#86efac", "#22c55e", "#15803d", "#14532d"],
            "note":   "Uses full class_probs vectors — NOT top_label. "
                      "Each cell = mean probability mass for that action class in that verdict group.",
        }

    # ── Zone filter rates ────────────────────────────────────────────────────

    def _build_zone_filter_rates(self, rows: list) -> dict:
        """
        Promotion rate per zone contact.

        zone_contacts is a structural spatial axis (zone_id strings).
        No zone_type semantics are used — only zone_id.
        Candidates with no zone contacts are grouped under '__no_zone__'.
        """
        zone_total:    dict[str, int] = defaultdict(int)
        zone_promoted: dict[str, int] = defaultdict(int)

        for r in rows:
            zones = r.get("zone_contacts", []) or ["__no_zone__"]
            for z in zones:
                zone_total[z] += 1
                if r["verdict"] == "PROMOTED":
                    zone_promoted[z] += 1

        all_zones = sorted(zone_total.keys())
        prom_rates = [
            round(zone_promoted[z] / max(zone_total[z], 1) * 100, 1)
            for z in all_zones
        ]

        return {
            "type":   "bars",
            "title":  "Phase 2 — Promotion rate per zone",
            "rows":   all_zones,
            "values": prom_rates,
            "raw_totals":   {z: zone_total[z] for z in all_zones},
            "raw_promoted": {z: zone_promoted[z] for z in all_zones},
            "color":  "#0284c7",
            "note":   "Zone axis uses zone_id strings only. No zone_type semantics applied.",
        }

    # ── Decision latency timeline ────────────────────────────────────────────

    def _build_latency_timeline(self, rows: list) -> dict:
        """
        Mean processing latency (processed_at_sec relative to first decision)
        per time bin. Shows filter throughput over the run.

        processed_at_sec is wall-clock monotonic time of each FilterDecision.
        It is separate from timestamp_sec (video time) — not merged or overwritten.
        """
        if not rows:
            return {"type": "timeline", "labels": [], "values": [], "color": "#64748b"}

        # Use processed_at_sec for the temporal axis (wall-clock latency profile)
        proc_times = [r.get("processed_at_sec", 0.0) for r in rows if r.get("processed_at_sec")]
        if not proc_times:
            return {"type": "timeline", "labels": [], "values": [], "color": "#64748b"}

        t_min = min(proc_times)
        t_max = max(proc_times)
        duration = max(t_max - t_min, 1e-6)
        bin_width = duration / self._n_time_bins

        bins_count: list[int] = [0] * self._n_time_bins
        for r in rows:
            pt = r.get("processed_at_sec", 0.0)
            if pt:
                bi = min(int((pt - t_min) / bin_width), self._n_time_bins - 1)
                bins_count[bi] += 1

        labels = [f"+{round(t_min - t_min + i * bin_width, 1)} s" for i in range(self._n_time_bins)]

        return {
            "type":    "timeline",
            "title":   "Phase 2 — Filter throughput (decisions per time window)",
            "labels":  labels,
            "values":  bins_count,
            "color":   "#64748b",
            "y_label": "decisions / window",
            "note":    "Uses processed_at_sec (wall-clock monotonic). "
                       "Separate from timestamp_sec (video time).",
        }

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalise(self) -> list[dict]:
        """
        Produce uniform dicts from either live FilterDecision objects or
        serialised FilterDecision.to_dict() dicts.

        RISK 2: class_probs is extracted from action_distribution (full vector).
                action_top is extracted only for the cooldown-key label axis —
                it is explicitly marked as a derived field in output.
        """
        raw_mode = getattr(self, "_raw_mode", False)
        out = []
        for d in self._decisions:
            if raw_mode or isinstance(d, dict):
                out.append(self._normalise_raw(d))
            else:
                out.append(self._normalise_decision(d))
        return out

    def _normalise_decision(self, d) -> dict:
        """Extract from a live FilterDecision object."""
        return {
            "verdict":           d.verdict.value,
            "suppress_reasons":  [r.value for r in d.suppress_reasons],
            "composite_score":   float(d.composite_score),
            "n_visible_kpts":    int(d.n_visible_kpts),
            "identity_stability": float(d.identity_stability),
            "zone_contacts":     list(d.zone_contacts),
            "timestamp_sec":     float(d.timestamp_sec),
            "processed_at_sec":  float(d.processed_at_sec),
            "action_top":        d.action_top,           # derived convenience — used only for cooldown key display
            "class_probs":       dict(d.action_distribution.class_probs),  # FULL vector
            "action_entropy":    float(d.action_entropy),
        }

    def _normalise_raw(self, d: dict) -> dict:
        """Extract from a FilterDecision.to_dict() dict."""
        cp = d.get("action_distribution", {})
        return {
            "verdict":           d.get("verdict", "UNKNOWN"),
            "suppress_reasons":  d.get("suppress_reasons", []),
            "composite_score":   float(d.get("composite_score", 0.0)),
            "n_visible_kpts":    int(d.get("n_visible_kpts", 0)),
            "identity_stability": float(d.get("identity_stability", 1.0)),
            "zone_contacts":     d.get("zone_contacts", []),
            "timestamp_sec":     float(d.get("timestamp_sec", 0.0)),
            "processed_at_sec":  float(d.get("processed_at_sec", 0.0)),
            "action_top":        d.get("action_top", ""),   # derived convenience only
            "class_probs":       cp if isinstance(cp, dict) else {},
            "action_entropy":    float(d.get("action_entropy", 0.0)),
        }

    def _empty_payload(self) -> dict:
        return {
            "rule_pass_rates":           {"type": "bars", "rows": [], "values": []},
            "suppress_reason_breakdown": {"type": "bars", "rows": [], "values": []},
            "composite_score_dist":      {"type": "histogram_dual", "labels": [], "promoted_values": [], "suppressed_values": []},
            "action_dist_heatmap":       {"type": "grid", "rows": [], "cols": [], "values": []},
            "zone_filter_rates":         {"type": "bars", "rows": [], "values": []},
            "latency_timeline":          {"type": "timeline", "labels": [], "values": []},
            "meta": {"total_evaluated": 0, "total_promoted": 0, "total_suppressed": 0},
        }


# ---------------------------------------------------------------------------
# FastAPI-ready endpoint helper
# ---------------------------------------------------------------------------

def build_phase2_heatmap(
    source,
    stats_dict: dict | None = None,
    n_score_bins: int = 10,
    n_time_bins:  int = 7,
) -> dict:
    """
    Single entry point for the FastAPI endpoint.

    source:
        FilterLeg instance              → from_filter_leg()
        list[FilterDecision]            → from_decisions()
        list[dict] (serialised)         → from_dict()
        None                            → empty payload

    Returns JSON-serialisable heatmap payload.
    """
    if source is None:
        return Phase2HeatmapBuilder(n_score_bins, n_time_bins).build()

    if isinstance(source, list):
        if source and isinstance(source[0], dict):
            builder = Phase2HeatmapBuilder.from_dict(source, stats_dict, n_score_bins, n_time_bins)
        else:
            builder = Phase2HeatmapBuilder.from_decisions(source, stats_dict, n_score_bins, n_time_bins)
    else:
        # Assume FilterLeg instance
        builder = Phase2HeatmapBuilder.from_filter_leg(source, n_score_bins, n_time_bins)

    return builder.build()


# ---------------------------------------------------------------------------
# Standalone test / CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Phase 2 Heatmap Builder")
    parser.add_argument("--input",  required=True,
                        help="Path to a JSON file containing a list of FilterDecision.to_dict() records")
    parser.add_argument("--out",    default="heatmap_phase2.json")
    parser.add_argument("--bins",   type=int, default=10)
    parser.add_argument("--tbins",  type=int, default=7)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        decisions_raw = json.load(f)

    payload = build_phase2_heatmap(
        source=decisions_raw,
        n_score_bins=args.bins,
        n_time_bins=args.tbins,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    meta = payload["meta"]
    print(f"Phase 2 heatmap built.")
    print(f"  Evaluated  : {meta['total_evaluated']}")
    print(f"  Promoted   : {meta['total_promoted']}")
    print(f"  Suppressed : {meta['total_suppressed']}")
    print(f"  Prom. rate : {meta['promotion_rate']:.1%}")
    print(f"  Output     : {args.out}")
