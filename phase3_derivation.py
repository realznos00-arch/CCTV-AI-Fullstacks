"""
phase3_derivation.py — Phase 3: Derivation Leg
===============================================
Event Intelligence Pipeline — Derivation Side

Responsibility:
    Consume CONFIRMED candidates from FilterBlackboardExtension.drain_active().
    Apply a composable derivation rule chain.
    Produce DerivedEvent records — structured, fully-auditable outputs.
    Write DerivedEvents to the DerivationLog (append-only, never mutated).
    Optionally emit notifications through a pluggable OutputChannel.

This module owns ONLY:
    - Consuming CONFIRMED candidates from FilterBlackboardExtension.drain_active()
    - Running DerivationRule chain in defined order
    - Producing DerivedEvent from each candidate
    - Appending to DerivationLog
    - Emitting through OutputChannel (pluggable, default: structured log)
    - Maintaining per-track DerivationHistory for temporal context

This module does NOT own:
    - Any detection or pose inference                    (Phase 1)
    - Any filter decisions or promotion logic            (Phase 2)
    - Any storage, archiving, or heatmaps               (Phase 4)
    - Any alert routing or notification delivery         (downstream consumer)

CQRS contract:
    Phase 3 reads CONFIRMED candidates from Phase 2's write surface.
    Phase 3 writes DerivedEvents to its own DerivationLog only.
    Phase 3 never writes back to the Blackboard.
    Phase 3 never modifies Candidate state.

STRUCTURAL RISKS ADDRESSED IN THIS VERSION:

    RISK 1 — SEMANTIC LEAKAGE
        Phase 3 is the FIRST phase permitted to assign operational meaning.
        All prior phases (1, 2) are epistemically neutral.
        Phase 3 composes meaning from distributions — it does NOT inherit labels.

        DerivationRule is a pure function: (Candidate, DerivationContext) → str | None
        Rules return a derivation_tag — a free-form operational string.
        Tags are defined in DerivationConfig.rule_tags, not hardcoded in the engine.
        The engine does not know what tags mean — that is the consumer's concern.

        DerivedEvent carries:
            derivation_tags   — frozenset of matched tags (structural output)
            primary_tag       — the highest-priority matched tag (convenience, not authority)
            source_distribution — the full ActionDistribution from the Candidate
            source_confidence   — the full ConfidenceProfile from the Candidate
        DerivedEvent does NOT carry:
            A threat verdict. A behavioral label. An alert level. An anomaly score.
        Those are consumer responsibilities, not Phase 3 responsibilities.

    RISK 2 — OVER-COMPRESSION
        DerivedEvent.source_distribution carries the FULL ActionDistribution.
        DerivedEvent.source_confidence carries the FULL ConfidenceProfile.
        derivation_score is a convenience float — not a compression of the distribution.
        It is computed from composite_score * distribution entropy weight.
        primary_tag is derived on demand from tag priority order — never stored as
        primary truth independent of the tag set.

        DerivationHistory per track stores the last N DerivedEvents in full.
        Temporal context is available for rules that need it (e.g. dwell time).
        History is never compressed into a single "last state" scalar.

    RISK 3 — TEMPORAL UNDERDEFINITION
        DerivedEvent carries:
            timestamp_sec       — video time of the source candidate
            derived_at_sec      — wall-clock time of derivation (monotonic)
            derivation_latency_ms — wall-clock derivation latency
        DerivationHistory tracks first_seen_sec and last_seen_sec per track.
        dwell_sec is derived on demand from history — not stored as a running float
        that silently accumulates without knowing when it was last updated.
        All temporal state is absolute monotonic, not relative.

    RISK 4 — MUTABLE SHARED MEMORY
        Phase 3 does NOT write back to the Blackboard or to Candidate objects.
        DerivationLog is append-only — no mutation after append.
        DerivationHistory is Phase 3-internal state, not shared with other phases.
        DerivationContext is frozen=True — rules are pure functions.
        All mutations to _history and _log happen under self._lock.

    RISK 5 — TAXONOMY RIGIDITY
        DerivationRule tags are configured strings, not hardcoded enum values.
        Rules evaluate structural properties of distributions, not frozen labels.
        Rules compose: a candidate can match multiple tags simultaneously.
        primary_tag is chosen by priority order in config — never by the rule itself.
        Adding a new operational category requires adding a config entry + rule,
        not modifying the engine or any enum.

Usage:
    from phase1_observation import Blackboard
    from phase2_filter import FilterLeg, FilterBlackboardExtension, FilterConfig
    from phase3_derivation import DerivationLeg, DerivationConfig

    bb     = Blackboard(cfg1.build_temporal_contract())
    bb_ext = FilterBlackboardExtension(bb)
    leg3   = DerivationLeg(DerivationConfig(), bb_ext)

    leg3.start()    # background thread
    # ... run phase1 + phase2 ...
    leg3.stop()
    leg3.join()

    events = leg3.derivation_log
"""

from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False

if TYPE_CHECKING:
    from phase1_observation import (
        ActionDistribution, ConfidenceProfile, ZoneDefinition,
        Candidate, CandidateState,
    )
    from phase2_filter import FilterBlackboardExtension, FilterDecision

log = logging.getLogger(__name__)


# =============================================================================
# DOMAIN — Derivation-specific types
# =============================================================================

@dataclass(frozen=True)
class DerivationContext:
    """
    Immutable context passed to each derivation rule.
    Built once per candidate evaluation.

    frozen=True (RISK 4):
        Rules are pure functions. A mutable context would allow a rule to
        inadvertently modify shared engine state. frozen=True makes that
        a TypeError at the point of mutation, not a silent data race.

    history and zone_registry are read-only references — rules must never
    mutate them. All mutations happen in DerivationEngine under self._lock.
    """
    history:        dict        # track_id → DerivationHistory [read-only in rules]
    zone_registry:  dict        # zone_id → ZoneDefinition     [read-only in rules]
    wall_clock_now: float       # monotonic seconds
    cfg:            "DerivationConfig"


@dataclass
class DerivationHistory:
    """
    Per-track derivation history.

    Phase 3-internal only — never written to the Blackboard.

    Stores last N DerivedEvents in full (RISK 2, RISK 3):
        - Full events, not compressed state scalars.
        - dwell_sec is derived from (last_seen_sec - first_seen_sec) on demand.
        - event_count is a counter only — the full event list is always available.

    history_window: configurable max length. Older events are dropped to bound
    memory. The full DerivationLog is the authoritative unbounded record.
    """
    track_id:       int
    first_seen_sec: float
    last_seen_sec:  float
    event_count:    int                                     = 0
    recent_events:  collections.deque = field(             # bounded window
        default_factory=lambda: collections.deque(maxlen=20)
    )

    @property
    def dwell_sec(self) -> float:
        """Derived on demand — not stored as a running accumulator."""
        return max(0.0, self.last_seen_sec - self.first_seen_sec)


@dataclass(frozen=True)
class DerivedEvent:
    """
    The output unit of Phase 3.

    Immutable. Appended to DerivationLog. Never mutated after creation.

    RISK 1 (semantic leakage):
        derivation_tags is a frozenset of matched tag strings.
        Tags are configured by the operator — not hardcoded in the engine.
        DerivedEvent does NOT carry a threat verdict, alert level, or anomaly score.
        Those are consumer responsibilities.

    RISK 2 (over-compression):
        source_distribution carries the FULL ActionDistribution from the Candidate.
        source_confidence carries the FULL ConfidenceProfile from the Candidate.
        primary_tag is a convenience field — it is always derivable from
        derivation_tags and the priority order in config.
        derivation_score is a structural signal strength indicator, not a
        compression of the full distribution.

    RISK 3 (temporal underdefinition):
        timestamp_sec is the source candidate's video time.
        derived_at_sec is the wall-clock time of derivation (monotonic).
        derivation_latency_ms is the wall-clock cost of this derivation step.
        dwell_sec_at_derivation is the track's dwell duration at the moment
        of derivation, captured from DerivationHistory. It is a point-in-time
        snapshot — not a live accumulator.
    """
    event_id:                 str
    candidate_id:             str
    track_id:                 int
    camera_id:                str
    timestamp_sec:            float       # video time of source candidate
    derived_at_sec:           float       # wall-clock time of derivation
    derivation_latency_ms:    float       # wall-clock cost of this derivation
    # ── Derivation outputs ────────────────────────────────────────────────────
    derivation_tags:          frozenset[str]  # all matched tags — primary output
    primary_tag:              str | None      # highest-priority tag (convenience)
    derivation_score:         float           # signal strength [0..1], not a verdict
    dwell_sec_at_derivation:  float           # track dwell at point of derivation
    zone_contacts:            tuple[str, ...]
    delta_reason:             str             # DeltaReason.value from source candidate
    # ── Full source distributions — never compressed (RISK 2) ─────────────────
    source_distribution:      "ActionDistribution"   # full prob vector
    source_confidence:        "ConfidenceProfile"    # full 4-dim profile
    # ── Structural signal context ─────────────────────────────────────────────
    composite_score:          float               # from source Candidate
    identity_stability:       float               # from source Candidate

    def to_dict(self) -> dict:
        return {
            "event_id":                self.event_id,
            "candidate_id":            self.candidate_id,
            "track_id":                self.track_id,
            "camera_id":               self.camera_id,
            "timestamp_sec":           round(self.timestamp_sec, 3),
            "derived_at_sec":          round(self.derived_at_sec, 3),
            "derivation_latency_ms":   round(self.derivation_latency_ms, 2),
            "derivation_tags":         sorted(self.derivation_tags),
            "primary_tag":             self.primary_tag,
            "derivation_score":        round(self.derivation_score, 4),
            "dwell_sec_at_derivation": round(self.dwell_sec_at_derivation, 3),
            "zone_contacts":           list(self.zone_contacts),
            "delta_reason":            self.delta_reason,
            # Full distributions preserved (RISK 2)
            "source_distribution":     {k: round(v, 4)
                                        for k, v in self.source_distribution.class_probs.items()},
            "source_top3":             [(a, round(c, 4))
                                        for a, c in self.source_distribution.top_n(3)],
            "source_entropy":          round(self.source_distribution.entropy, 4),
            "source_confidence":       self.source_confidence.as_dict(),
            "composite_score":         round(self.composite_score, 4),
            "identity_stability":      round(self.identity_stability, 4),
        }


# =============================================================================
# DERIVATION RULES
# Pure functions: (Candidate, DerivationContext) → str | None
# None = rule does not fire. A string = the tag this rule contributes.
# All rules run — no short-circuit — so the full tag set is always complete.
# Rules evaluate structural properties ONLY: distributions, geometry, dwell.
# Rules do NOT assign behavioral meaning — that is the consumer's job.
# =============================================================================

def _rule_distribution_peak(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if top class probability exceeds the configured peak threshold.

    Evaluates: is there a dominant class in the distribution?
    Does NOT interpret what that class means operationally.
    The tag returned is looked up from cfg.peak_tag — not hardcoded.
    """
    dist = c.action_distribution
    if not dist.class_probs:
        return None
    if dist.top_conf >= ctx.cfg.peak_conf_threshold:
        return ctx.cfg.peak_tag
    return None


def _rule_distribution_entropy(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if distribution entropy is below the low-entropy threshold.

    Low entropy means the distribution is concentrated — high certainty.
    High entropy means mass is spread across classes — behavioral ambiguity.

    This rule fires on LOW entropy (high certainty), which is structurally
    significant regardless of which class is dominant.
    """
    dist = c.action_distribution
    if not dist.class_probs:
        return None
    if dist.entropy <= ctx.cfg.low_entropy_threshold:
        return ctx.cfg.low_entropy_tag
    return None


def _rule_zone_contact(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if the candidate intersects at least one registered zone.

    Evaluates spatial contact against the zone registry.
    Does NOT interpret zone_type or assign operational meaning to the contact.
    """
    if not ctx.zone_registry:
        return None
    known = set(ctx.zone_registry.keys())
    if any(z in known for z in c.zone_contacts):
        return ctx.cfg.zone_contact_tag
    return None


def _rule_dwell(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if the track's dwell duration exceeds the configured threshold.

    dwell_sec is derived from DerivationHistory on demand (RISK 3).
    No running accumulator — the history record is the source of truth.
    First-seen tracks have dwell_sec = 0.0 and will not fire this rule.
    """
    hist = ctx.history.get(c.track_id)
    if hist is None:
        return None
    if hist.dwell_sec >= ctx.cfg.dwell_threshold_sec:
        return ctx.cfg.dwell_tag
    return None


def _rule_tracking_stability(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if identity_stability is below the degraded-tracking floor.

    A low-stability track may represent multiple people. Tagging it
    allows consumers to apply additional scrutiny without Phase 3
    making a binary verdict on the track's identity.
    """
    if c.identity_stability < ctx.cfg.degraded_stability_floor:
        return ctx.cfg.degraded_stability_tag
    return None


def _rule_composite_strength(
    c: "Candidate", ctx: DerivationContext
) -> str | None:
    """
    Fire if composite_score exceeds the high-confidence threshold.

    Structural signal: this candidate has strong multi-dimensional support.
    Does not assign operational meaning — tags it for consumer use.
    """
    if c.composite_score >= ctx.cfg.high_confidence_threshold:
        return ctx.cfg.high_confidence_tag
    return None


# Default rule chain — order defines the check sequence but all rules always run.
_DEFAULT_RULES: list[Callable] = [
    _rule_distribution_peak,
    _rule_distribution_entropy,
    _rule_zone_contact,
    _rule_dwell,
    _rule_tracking_stability,
    _rule_composite_strength,
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DerivationConfig:
    # ── Rule: distribution peak ───────────────────────────────────────────────
    peak_conf_threshold:    float = 0.60   # top_conf must exceed this to fire
    peak_tag:               str   = "DISTRIBUTION_PEAK"

    # ── Rule: distribution entropy ────────────────────────────────────────────
    low_entropy_threshold:  float = 0.80   # entropy below this fires
    low_entropy_tag:        str   = "LOW_ENTROPY"

    # ── Rule: zone contact ────────────────────────────────────────────────────
    zone_contact_tag:       str   = "ZONE_CONTACT"

    # ── Rule: dwell ───────────────────────────────────────────────────────────
    dwell_threshold_sec:    float = 10.0   # track must have been seen this long
    dwell_tag:              str   = "DWELL_THRESHOLD_MET"

    # ── Rule: tracking stability ──────────────────────────────────────────────
    degraded_stability_floor: float = 0.40  # below this → degraded stability tag
    degraded_stability_tag:   str   = "DEGRADED_STABILITY"

    # ── Rule: composite strength ──────────────────────────────────────────────
    high_confidence_threshold: float = 0.75
    high_confidence_tag:       str   = "HIGH_CONFIDENCE"

    # ── Tag priority (index 0 = highest) — determines primary_tag ─────────────
    # Tags not in this list are still included in derivation_tags but will not
    # be selected as primary_tag.
    tag_priority: list[str] = field(default_factory=lambda: [
        "HIGH_CONFIDENCE",
        "DISTRIBUTION_PEAK",
        "LOW_ENTROPY",
        "ZONE_CONTACT",
        "DWELL_THRESHOLD_MET",
        "DEGRADED_STABILITY",
    ])

    # ── History ───────────────────────────────────────────────────────────────
    history_window:         int   = 20     # max recent events per track in memory

    # ── Polling ───────────────────────────────────────────────────────────────
    poll_interval_sec:      float = 0.05

    # ── derivation_score weighting ────────────────────────────────────────────
    # derivation_score = composite_score * (1 - normalised_entropy_weight)
    # This is a structural signal indicator, not a compression of the distribution.
    entropy_weight:         float = 0.25   # how much entropy suppresses score


# =============================================================================
# DERIVATION ENGINE
# =============================================================================

class DerivationEngine:
    """
    Stateful derivation engine.

    Maintains per-track DerivationHistory.
    Applies the full rule chain to each CONFIRMED candidate.
    Produces DerivedEvent records.
    Never writes to the Blackboard or mutates Candidates.

    Thread safety: all history mutations under self._lock.
    DerivationContext is frozen — rules are pure functions.
    """

    def __init__(
        self,
        cfg: DerivationConfig,
        zone_registry: dict,
        rules: list[Callable] | None = None,
    ) -> None:
        self._cfg           = cfg
        self._zone_registry = zone_registry
        self._rules         = rules if rules is not None else list(_DEFAULT_RULES)
        self._history:      dict[int, DerivationHistory] = {}
        self._lock          = threading.Lock()

    def derive(self, c: "Candidate") -> DerivedEvent:
        """
        Run all rules against the candidate.
        Build a DerivedEvent. Update DerivationHistory.
        Return the DerivedEvent.

        RISK 4: history is updated under self._lock.
        DerivationContext is built with a snapshot of history references —
        rules receive read-only access and cannot acquire self._lock.
        """
        t_start = time.monotonic()

        # ── Build context — snapshot under lock, rules run outside lock ─────────
        # RISK 4: self._history is a live dict mutated by concurrent derive() calls.
        # Passing the live reference means rules see a dict that can change while
        # they execute (lock is released before rules run).
        # Fix: shallow-copy the dict under the lock. _update_history runs first
        # so dwell_rule sees this candidate's history entry in the snapshot.
        with self._lock:
            self._update_history(c, t_start)
            history_snapshot = dict(self._history)  # stable snapshot, not live ref

        ctx = DerivationContext(
            history        = history_snapshot,  # safe outside lock
            zone_registry  = self._zone_registry,
            wall_clock_now = t_start,
            cfg            = self._cfg,
        )

        # ── Run all rules — no short-circuit ─────────────────────────────────
        fired_tags: set[str] = set()
        for rule in self._rules:
            tag = rule(c, ctx)
            if tag is not None:
                fired_tags.add(tag)

        # ── Derive primary_tag from priority order ────────────────────────────
        primary_tag: str | None = None
        for t in self._cfg.tag_priority:
            if t in fired_tags:
                primary_tag = t
                break

        # ── Derive derivation_score ───────────────────────────────────────────
        dist     = c.action_distribution
        n        = len(dist.class_probs)
        max_ent  = float(__import__("math").log(max(n, 1)))
        norm_ent = (dist.entropy / max_ent) if max_ent > 1e-8 else 0.0
        derivation_score = float(
            c.composite_score * (1.0 - self._cfg.entropy_weight * norm_ent)
        )

        # ── Dwell at point of derivation ──────────────────────────────────────
        with self._lock:
            hist = self._history.get(c.track_id)
            dwell = hist.dwell_sec if hist else 0.0

        t_end = time.monotonic()

        event = DerivedEvent(
            event_id                 = str(uuid.uuid4()),
            candidate_id             = c.candidate_id,
            track_id                 = c.track_id,
            camera_id                = c.camera_id,
            timestamp_sec            = c.timestamp_sec,
            derived_at_sec           = t_end,
            derivation_latency_ms    = (t_end - t_start) * 1000.0,
            derivation_tags          = frozenset(fired_tags),
            primary_tag              = primary_tag,
            derivation_score         = max(0.0, min(1.0, derivation_score)),
            dwell_sec_at_derivation  = dwell,
            zone_contacts            = tuple(c.zone_contacts),
            delta_reason             = c.delta_reason.value,
            source_distribution      = c.action_distribution,   # full — not compressed
            source_confidence        = c.confidence,            # full — not compressed
            composite_score          = c.composite_score,
            identity_stability       = c.identity_stability,
        )

        # ── Append event to history ───────────────────────────────────────────
        with self._lock:
            if c.track_id in self._history:
                self._history[c.track_id].recent_events.append(event)

        log.debug(
            "Derivation  eid=%s  track=%d  tags=%s  primary=%s  score=%.3f  "
            "dwell=%.1fs  top=%s(%.2f)",
            event.event_id[:8], event.track_id,
            sorted(event.derivation_tags), event.primary_tag,
            event.derivation_score, event.dwell_sec_at_derivation,
            dist.top_label, dist.top_conf,
        )

        return event

    def _update_history(self, c: "Candidate", now: float) -> None:
        """
        Create or update DerivationHistory for a track.
        Called under self._lock.
        """
        tid = c.track_id
        if tid not in self._history:
            self._history[tid] = DerivationHistory(
                track_id       = tid,
                first_seen_sec = c.timestamp_sec,
                last_seen_sec  = c.timestamp_sec,
                event_count    = 0,
                recent_events  = collections.deque(maxlen=self._cfg.history_window),
            )
        hist = self._history[tid]
        hist.last_seen_sec = max(hist.last_seen_sec, c.timestamp_sec)
        hist.event_count  += 1

    @property
    def history(self) -> dict[int, DerivationHistory]:
        """Read-only snapshot of current history state."""
        with self._lock:
            return dict(self._history)


# =============================================================================
# DERIVATION LOG — append-only, never mutated
# =============================================================================

class DerivationLog:
    """
    Append-only log of all DerivedEvents.

    RISK 4: all appends and reads under self._lock.
    Events are never mutated after append.
    snapshot() returns a copy — callers cannot modify the internal list.
    """

    def __init__(self) -> None:
        self._lock:   threading.Lock      = threading.Lock()
        self._events: list[DerivedEvent]  = []

    def append(self, event: DerivedEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[DerivedEvent]:
        """Returns a copy — safe to iterate without holding the lock."""
        with self._lock:
            return list(self._events)

    def snapshot_for_track(self, track_id: int) -> list[DerivedEvent]:
        with self._lock:
            return [e for e in self._events if e.track_id == track_id]

    def count(self) -> int:
        with self._lock:
            return len(self._events)


# =============================================================================
# OUTPUT CHANNEL — pluggable, default: structured log
# =============================================================================

class OutputChannel:
    """
    Pluggable output interface.

    Phase 3 calls channel.emit(event) after every DerivedEvent.
    The default implementation logs structured output.
    Replace with a custom channel for webhooks, queues, or storage writes.

    emit() is called outside any lock — channel implementations must be
    thread-safe if they share state.
    """

    def emit(self, event: DerivedEvent) -> None:
        log.info(
            "DerivedEvent  eid=%s  track=%d  camera=%s  t=%.3fs  "
            "tags=%s  primary=%s  score=%.3f  dwell=%.1fs  "
            "top=%s(%.2f)  composite=%.3f  stability=%.3f  zones=%s",
            event.event_id[:8],
            event.track_id,
            event.camera_id,
            event.timestamp_sec,
            sorted(event.derivation_tags),
            event.primary_tag,
            event.derivation_score,
            event.dwell_sec_at_derivation,
            event.source_distribution.top_label,
            event.source_distribution.top_conf,
            event.composite_score,
            event.identity_stability,
            list(event.zone_contacts),
        )


# =============================================================================
# DERIVATION STATS — observability, not a verdict
# =============================================================================

@dataclass
class DerivationStats:
    total_derived:   int   = 0
    total_no_tags:   int   = 0    # candidates that matched no rules
    tag_counts:      dict  = field(default_factory=dict)
    cycles_run:      int   = 0
    total_latency_ms: float = 0.0

    def record(self, event: DerivedEvent) -> None:
        self.total_derived += 1
        self.total_latency_ms += event.derivation_latency_ms
        if not event.derivation_tags:
            self.total_no_tags += 1
        for tag in event.derivation_tags:
            self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1

    @property
    def avg_latency_ms(self) -> float:
        if self.total_derived == 0:
            return 0.0
        return self.total_latency_ms / self.total_derived

    def to_dict(self) -> dict:
        return {
            "total_derived":    self.total_derived,
            "total_no_tags":    self.total_no_tags,
            "tag_counts":       dict(self.tag_counts),
            "cycles_run":       self.cycles_run,
            "avg_latency_ms":   round(self.avg_latency_ms, 2),
        }


# =============================================================================
# DERIVATION LEG — background thread, consumes from FilterBlackboardExtension
# =============================================================================

class DerivationLeg:
    """
    Phase 3 background thread.

    Polls FilterBlackboardExtension.drain_active() for CONFIRMED candidates.
    Passes each through DerivationEngine.
    Appends results to DerivationLog.
    Emits through OutputChannel.

    Thread model:
        One background thread runs _run().
        DerivationEngine is not shared with other threads.
        DerivationLog is thread-safe (its own lock).
        OutputChannel.emit() is called outside any lock.

    Lifecycle:
        start() → background thread begins polling.
        stop()  → signals thread to exit after current batch.
        join()  → waits for thread to finish, logs final stats.
    """

    def __init__(
        self,
        cfg:     DerivationConfig,
        bb_ext:  "FilterBlackboardExtension",
        channel: OutputChannel | None = None,
        rules:   list[Callable] | None = None,
    ) -> None:
        self._cfg     = cfg
        self._bb_ext  = bb_ext
        self._channel = channel or OutputChannel()
        self._log     = DerivationLog()
        self._stats   = DerivationStats()
        self._stop    = threading.Event()
        self._thread: threading.Thread | None = None

        # Engine is Phase 3-internal — not shared with other threads
        self._engine = DerivationEngine(
            cfg           = cfg,
            zone_registry = bb_ext._bb.zone_registry,
            rules         = rules,
        )

    def start(self) -> None:
        self._thread = threading.Thread(
            target = self._run,
            name   = "phase3-derivation",
            daemon = True,
        )
        self._thread.start()
        log.info(
            "Phase 3 Derivation started. "
            "peak_threshold=%.2f  entropy_threshold=%.2f  dwell_threshold=%.1fs  "
            "poll=%.3fs",
            self._cfg.peak_conf_threshold,
            self._cfg.low_entropy_threshold,
            self._cfg.dwell_threshold_sec,
            self._cfg.poll_interval_sec,
        )

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info(
            "Phase 3 complete. derived=%d  no_tags=%d  cycles=%d  avg_latency=%.2fms",
            self._stats.total_derived,
            self._stats.total_no_tags,
            self._stats.cycles_run,
            self._stats.avg_latency_ms,
        )

    def _run(self) -> None:
        log.info("Phase 3 derivation loop running.")
        pbar = _tqdm(
            desc="Phase 3 — Deriving",
            unit="events",
            dynamic_ncols=True,
            colour="magenta",
            leave=True,
        ) if _TQDM_OK else None
        try:
            while not self._stop.is_set():
                try:
                    batch = self._bb_ext.drain_active()
                    if batch:
                        for c in batch:
                            event = self._engine.derive(c)
                            self._log.append(event)
                            self._channel.emit(event)
                            self._stats.record(event)
                        self._stats.cycles_run += 1
                        if pbar is not None:
                            pbar.update(len(batch))
                            pbar.set_postfix({
                                "total": self._stats.total_derived,
                                "no_tag": self._stats.total_no_tags,
                                "lat_ms": f"{self._stats.avg_latency_ms:.1f}",
                            }, refresh=False)
                        if self._stats.cycles_run % 20 == 0:
                            log.info(
                                "Derivation stats: %s",
                                self._stats.to_dict(),
                            )
                except Exception as e:
                    log.exception("Phase 3 derivation error: %s", e)

                self._stop.wait(timeout=self._cfg.poll_interval_sec)
        finally:
            if pbar is not None:
                pbar.close()

        # Final drain
        try:
            final_batch = self._bb_ext.drain_active()
            if final_batch:
                for c in final_batch:
                    event = self._engine.derive(c)
                    self._log.append(event)
                    self._channel.emit(event)
                    self._stats.record(event)
                log.info("Phase 3 final drain: %d candidates processed.", len(final_batch))
        except Exception as e:
            log.exception("Phase 3 final drain error: %s", e)

    @property
    def derivation_log(self) -> list[DerivedEvent]:
        """Full audit log snapshot — safe to read at any time."""
        return self._log.snapshot()

    @property
    def log(self) -> "DerivationLog":
        """Live DerivationLog reference — for Phase 4 cursor-based polling.
        Phase 4 must treat this as read-only (snapshot() calls only).
        """
        return self._log

    @property
    def stats(self) -> DerivationStats:
        return self._stats


def _format_action_head(dist: "ActionDistribution", threshold: float = 0.45, top_n: int = 3) -> str:
    """Format the distribution head for summary output.

    Low-confidence priors are rendered as uncertain to avoid misleading
    the user into thinking a weak top label is a strong prediction.
    """
    if not dist.class_probs:
        return "unknown"
    if dist.top_conf >= threshold:
        return f"{dist.top_label}({dist.top_conf:.2f})"

    top_k = dist.top_n(top_n)
    if not top_k:
        return "unknown"
    return "uncertain[" + ", ".join(f"{a}({c:.2f})" for a, c in top_k) + "]"


def _is_meaningful_event(e: DerivedEvent) -> bool:
    if e.primary_tag in {"HIGH_CONFIDENCE", "DISTRIBUTION_PEAK", "LOW_ENTROPY", "DWELL_THRESHOLD_MET"}:
        return True
    if e.derivation_score >= 0.40 and e.source_distribution.top_conf >= 0.50:
        return True
    return False


def print_derived_summary(events: list[DerivedEvent], limit: int = 5) -> None:
    """Print a count and a sample of the most meaningful derived events."""
    print(f"  Derived events: {len(events)}")
    if not events:
        return

    meaningful = [e for e in events if _is_meaningful_event(e)]
    if meaningful:
        sample = meaningful[:limit]
        skipped = len(events) - len(meaningful)
        print(f"  First {min(limit, len(sample))} meaningful derived events:")
    else:
        sample = events[:limit]
        skipped = 0
        print(f"  First {min(limit, len(sample))} derived events:")

    for e in sample:
        print(
            f"    track={e.track_id}  tags={sorted(e.derivation_tags)}  "
            f"primary={e.primary_tag}  score={e.derivation_score:.3f}  "
            f"dwell={e.dwell_sec_at_derivation:.1f}s  "
            f"top={_format_action_head(e.source_distribution)}"
        )
    if skipped:
        print(f"    ({skipped} low-meaning events omitted from sample)")


# =============================================================================
# Standalone test — runs Phase 1 + 2 + 3 pipeline
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    from collections import Counter

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 1 + 2 + 3 pipeline test")
    parser.add_argument("--input",        required=True,          help="Video file")
    parser.add_argument("--model",        default="",             help="LSTM model path")
    parser.add_argument("--mode",         type=int, default=2,    choices=[1, 2])
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--zones",        default=None,           help="JSON zones file")
    parser.add_argument("--skip",         type=int, default=1)
    parser.add_argument("--end",          type=float, default=0.0)
    parser.add_argument("--cooldown",     type=float, default=3.0)
    parser.add_argument("--score",        type=float, default=0.40)
    parser.add_argument("--stability",    type=float, default=0.30)
    parser.add_argument("--require-zone", action="store_true")
    parser.add_argument("--permissive",   action="store_true")
    parser.add_argument("--dwell",        type=float, default=10.0)
    parser.add_argument("--peak",         type=float, default=0.60)
    parser.add_argument("--out",          default="phase3_derivation_results.json")
    args = parser.parse_args()

    from phase1_observation import (
        ObservationConfig, ObservationLeg, Blackboard, resolve_action_model_path,
    )
    from phase2_filter import FilterLeg, FilterConfig

    zones_raw = []
    if args.zones:
        with open(args.zones, encoding="utf-8") as f:
            zones_raw = json.load(f)

    model_path = resolve_action_model_path(args.mode, args.model)

    cfg1 = ObservationConfig(
        input_path  = args.input,
        model_path  = model_path,
        vision_mode = args.mode,
        device      = args.device,
        zones       = zones_raw,
        end_sec     = args.end,
        frame_skip  = args.skip,
    )
    cfg2 = FilterConfig(
        min_composite_score    = args.score,
        min_identity_stability = args.stability,
        cooldown_sec           = args.cooldown,
        require_zone_contact   = args.require_zone,
        permissive_mode        = args.permissive,
    )
    cfg3 = DerivationConfig(
        dwell_threshold_sec  = args.dwell,
        peak_conf_threshold  = args.peak,
    )

    bb     = Blackboard(cfg1.build_temporal_contract())
    leg2   = FilterLeg(cfg2, bb)
    leg3   = DerivationLeg(cfg3, leg2.bb_ext)
    leg1   = ObservationLeg(cfg1, bb)

    print(f"\n  Phase 1 + 2 + 3 pipeline test")
    print(f"  Input      : {args.input}")
    print(f"  Model      : {model_path}")
    print(f"  Mode       : {args.mode}  Zones: {len(zones_raw)}  Skip: {args.skip}")
    print(f"  Filter     : score={args.score}  stability={args.stability}  "
          f"cooldown={args.cooldown}s  require_zone={args.require_zone}")
    print(f"  Derivation : peak={args.peak}  dwell={args.dwell}s")
    print(f"  Permissive : {args.permissive}")
    print()

    # Start Phase 3, then Phase 2, then Phase 1 (blocking)
    leg3.start()
    leg2.start()
    leg1.run()
    leg2.stop()
    leg2.join()
    leg3.stop()
    leg3.join()

    events   = leg3.derivation_log
    p2_stats = leg2.stats.to_dict()
    p3_stats = leg3.stats.to_dict()
    action_counts = Counter(e.source_distribution.top_label for e in events)
    action_probs_mean: dict[str, float] = {}
    if events:
        accum: dict[str, float] = {}
        for e in events:
            for k, v in e.source_distribution.class_probs.items():
                accum[k] = accum.get(k, 0.0) + float(v)
        n = float(len(events))
        action_probs_mean = {k: round(v / n, 4) for k, v in accum.items()}

    print(f"  Phase 1 candidates emitted : {bb.count_candidates()}")
    print(f"  Phase 2 evaluated          : {p2_stats['total_evaluated']}")
    print(f"  Phase 2 promoted           : {p2_stats['total_promoted']}")
    print(f"  Phase 2 suppressed         : {p2_stats['total_suppressed']}")
    print(f"  Phase 3 derived            : {p3_stats['total_derived']}")
    print(f"  Phase 3 no-tag events      : {p3_stats['total_no_tags']}")
    print(f"  Phase 3 avg latency        : {p3_stats['avg_latency_ms']:.2f}ms")
    print(f"  Derived action counts      : {dict(action_counts)}")
    print()
    print("  Derivation tag counts:")
    for tag, count in sorted(p3_stats["tag_counts"].items(), key=lambda x: -x[1]):
        print(f"    {tag:<30} {count}")
    print()

    out_data = {
        "meta": {
            "input":        args.input,
            "model_path":   model_path,
            "vision_mode":  args.mode,
            "frame_skip":   args.skip,
            "filter_config": {
                "min_composite_score":    args.score,
                "min_identity_stability": args.stability,
                "cooldown_sec":           args.cooldown,
                "require_zone_contact":   args.require_zone,
                "permissive_mode":        args.permissive,
            },
            "derivation_config": {
                "peak_conf_threshold":  args.peak,
                "dwell_threshold_sec":  args.dwell,
            },
            "derived_action_counts": action_counts,
            "derived_action_probs_mean": action_probs_mean,
        },
        "phase2_stats":    p2_stats,
        "phase3_stats":    p3_stats,
        "derived_events":  [e.to_dict() for e in
                            sorted(events, key=lambda x: x.timestamp_sec)],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"  Output: {args.out}")
    print()

    if events:
        print_derived_summary(sorted(events, key=lambda x: x.timestamp_sec), limit=5)
    print()
