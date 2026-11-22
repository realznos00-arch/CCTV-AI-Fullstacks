"""
phase2_filter.py — Phase 2: Filter Leg
=======================================
Event Intelligence Pipeline — CQRS Read Side

Responsibility:
    Read EMERGING candidates from the Blackboard.
    Apply a deterministic, ordered rule chain.
    Promote surviving candidates to ACTIVE.
    Suppress failing candidates with a structured drop record.
    Write promotion and suppression decisions back to the Blackboard.

This module owns ONLY:
    - Consuming EMERGING candidates from Blackboard.drain_emerging()
    - Running FilterRule chain in defined order
    - Writing FilterDecision (PROMOTED / SUPPRESSED) back to Blackboard
    - Maintaining cooldown state per zone+action pair
    - Maintaining active candidate registry for duplicate detection
    - Emitting a drop log with structured reason vectors

This module does NOT own:
    - Any inference                   (Phase 1)
    - Any event derivation            (Phase 3)
    - Any alert emission              (Phase 3)
    - Any context flags or scoring    (Phase 3)
    - Any storage or archiving        (Phase 4)
    - Heatmaps                        (Phase 4)

CQRS read-side contract:
    Phase 2 reads candidates and the zone_registry from the Blackboard.
    Phase 2 writes FilterDecisions and updated candidate states.
    Phase 2 never writes new candidates. It only promotes or suppresses.

Design principles carried from Phase 1:
    - No scalar compression: SuppressReason is a structured vector, not a string
    - No semantic leakage: filter rules evaluate structural properties only
      (confidence, geometry, cooldown, duplicate) — not threat/behavioral meaning
    - Full auditability: every suppressed candidate has a complete FilterDecision
    - Downstream reinterpretability: SUPPRESSED candidates are never deleted,
      only state-tagged. Phase 3 may query suppression history.

Filter rule execution order (deterministic):
    Rule 1 — CompositeScoreRule      : composite_score >= floor
    Rule 2 — PoseQualityRule         : n_visible_kpts >= min_kpts AND pose_valid
    Rule 3 — TrackingStabilityRule   : identity_stability >= floor OR first observation
    Rule 4 — ZoneContactRule         : must intersect at least one registered zone (if required)
    Rule 5 — CooldownRule            : no active cooldown for this (zone_id, action_top) pair
    Rule 6 — DuplicateRule           : no ACTIVE candidate for same track_id in same zone set

    Each rule either PASSES or emits a SuppressReason component.
    All failing reasons are collected — not short-circuited —
    so the full suppression reason vector is always complete.

Structural risks addressed in v2 (audit against Phase 1 risk framework):

    RISK 1 — SEMANTIC LEAKAGE
        SuppressComponent names describe structural properties only:
            LOW_COMPOSITE_SCORE, POOR_POSE_QUALITY, UNSTABLE_TRACKING,
            NO_ZONE_CONTACT, COOLDOWN_ACTIVE, DUPLICATE_TRACK_ZONE.
        No behavioral vocabulary anywhere in Phase 2.
        FilterDecision carries what was measured, not what it means.

    RISK 2 — OVER-COMPRESSION  [FIXED in v2]
        FilterDecision now carries full ActionDistribution (class_probs vector)
        and full ConfidenceProfile (4-dim + validity flags).
        action_top / action_top_conf / action_entropy are derived convenience
        fields for logging — never the primary record.
        Phase 3 reads from action_distribution, not from action_top.

    RISK 3 — TEMPORAL UNDERDEFINITION
        FilterDecision.processed_at_sec records wall-clock time of each decision.
        CooldownEntry.expires_at_sec is absolute monotonic — not a duration.
        ActiveCandidateEntry.expires_at is absolute monotonic — not a duration.
        All temporal state is absolute, not relative. Relative durations are
        config values only (cooldown_sec, active_window_sec).

    RISK 4 — MUTABLE SHARED MEMORY  [FIXED in v2]
        FilterContext is now frozen=True — rules cannot mutate shared engine state.
        promote() and suppress() use dataclasses.replace() — no in-place Candidate
        mutation. The Candidate reference held by Phase 1 is never modified.
        FilterBlackboardExtension holds its own _active / _suppressed dicts.
        All dict mutations happen inside self._lock.
        _maybe_prune() now updates self._last_prune inside the lock —
        not after releasing it.

    RISK 5 — TAXONOMY RIGIDITY
        FilterDecision stores action_distribution (full) not action_top (frozen label).
        suppress_reasons is a frozenset — no ordered priority implied.
        Permissive mode separates hard/soft failures without changing the reason set.
        Phase 3 can reinterpret any FilterDecision from its full distributions.

    ADDITIONAL FIX — drain_active() re-delivery bug
        drain_active() previously returned the full _active dict on every call.
        Phase 3 would receive the same ACTIVE candidates on every poll cycle.
        Fixed: _consumed_ids tracks what has been delivered. drain_active()
        returns only candidates Phase 3 has not yet received.

Structural risks addressed in v3 (full audit against Phase 1 risk framework):

    RISK 1 + RISK 2 — FROZEN LABEL IN ActiveCandidateEntry  [FIXED in v3]
        ActiveCandidateEntry previously stored action_top: str — a scalar label
        extracted from the distribution at promotion time. This is compression.
        Duplicate detection only requires track_id + zone_set (spatial+identity).
        The action distribution does not determine whether a spatial event is
        already being tracked. action_top removed from ActiveCandidateEntry.

    RISK 2 — ABSENT DISTRIBUTION COOLDOWN COLLISION  [FIXED in v3]
        absent()/uninformed() distributions have top_label = 'unknown'.
        The previous cooldown key f"{zone_id}:unknown" was shared across all
        tracks with absent distributions in the same zone — one track's cooldown
        blocked all others. Fixed: cooldown registration is guarded by
        min_action_conf_for_cooldown. Below this threshold, no cooldown is
        registered for that promotion. The full distribution is always preserved
        in FilterDecision regardless.

    RISK 4 — blackboard._lock private access  [DOCUMENTED in v3]
        FilterBlackboardExtension accesses blackboard._lock by private name.
        This is a deliberate coupling, now made explicit with a contract comment
        in __init__. Full decoupling requires Phase 1 to expose a public
        acquire() context manager — noted for Phase 1 future iteration.

    RISK 5 — __no_zone__ magic string  [FIXED in v3]
        The implicit sentinel "__no_zone__" used as a fallback zone in cooldown
        keys has been replaced with _NO_ZONE_SENTINEL, a named module-level
        constant. Not buried in method bodies.

    STATS INTEGRITY — FilterStats reason inflation  [FIXED in v3]
        In permissive mode, FilterStats.record() was counting suppress_reasons
        for PROMOTED candidates (soft failures overridden by permissive mode).
        reason_counts was inflated with reasons that did not cause suppression.
        Fixed: reason_counts is only updated when verdict == SUPPRESSED.
        Soft failures on promoted candidates remain in the full FilterDecision
        audit log for separate analysis.

Usage:
    from phase1_observation import Blackboard
    from phase2_filter import FilterLeg, FilterConfig

    cfg_f = FilterConfig()
    leg2  = FilterLeg(cfg_f, blackboard)
    leg2.start()    # background thread
    # ...
    leg2.stop()
    leg2.join()

    # Promotion results
    promoted   = blackboard.get_active_candidates()
    drop_log   = leg2.drop_log
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False

if TYPE_CHECKING:
    # Import phase1 types for typing only — avoids circular at runtime
    from phase1_observation import (
        Blackboard, Candidate, CandidateState,
        ActionDistribution, ConfidenceProfile, ZoneDefinition,
    )

log = logging.getLogger(__name__)

# Sentinel used as the zone component of a cooldown key when a candidate has
# no zone contacts. Explicit constant — not an implicit magic string.
# Must not collide with any real zone_id. Prefixed with __ to signal sentinel.
_NO_ZONE_SENTINEL = "__no_zone__"


# =============================================================================
# DOMAIN — Filter-specific types
# =============================================================================

class FilterVerdict(str, Enum):
    PROMOTED   = "PROMOTED"    # candidate advances to ACTIVE state
    SUPPRESSED = "SUPPRESSED"  # candidate killed, reason recorded


class SuppressComponent(str, Enum):
    """
    Atomic suppression reasons. Collected as a set — never a single string.

    Keeping them as a set (not a priority-ordered single reason) matters:
    a candidate may fail multiple rules simultaneously.
    Phase 3 or offline analysis can inspect the full reason vector.
    """
    LOW_COMPOSITE_SCORE   = "LOW_COMPOSITE_SCORE"
    POOR_POSE_QUALITY     = "POOR_POSE_QUALITY"
    UNSTABLE_TRACKING     = "UNSTABLE_TRACKING"
    NO_ZONE_CONTACT       = "NO_ZONE_CONTACT"
    COOLDOWN_ACTIVE       = "COOLDOWN_ACTIVE"
    DUPLICATE_TRACK_ZONE  = "DUPLICATE_TRACK_ZONE"


@dataclass(frozen=True)
class FilterDecision:
    """
    Immutable record of a filter decision.
    Appended to FilterLeg.drop_log regardless of verdict.
    PROMOTED decisions have suppress_reasons = frozenset().

    Designed for full auditability:
        - every candidate that entered the filter has a FilterDecision
        - suppressed candidates are never silently discarded
        - the full reason vector is preserved for reanalysis

    RISK 2 (over-compression):
        action_distribution carries the FULL probability vector from the candidate.
        action_top / action_top_conf / action_entropy are derived convenience fields
        for logging and serialisation only — they are NEVER the primary record.
        Phase 3 must read action_distribution, not action_top, for any decision-making.
        FilterDecision stores what Phase 2 saw, not a summary of what Phase 2 saw.

    confidence carries the FULL ConfidenceProfile (all four dimensions + validity flags).
    composite_score is a derived convenience field, not the authority.
    """
    decision_id:         str
    candidate_id:        str
    track_id:            int
    camera_id:           str
    timestamp_sec:       float
    verdict:             FilterVerdict
    suppress_reasons:    frozenset[SuppressComponent]  # empty if PROMOTED
    # ── Full distributions — primary record (RISK 2) ──────────────────────────
    action_distribution: "ActionDistribution"   # full prob vector — NOT compressed
    confidence:          "ConfidenceProfile"    # full 4-dim profile — NOT a scalar
    # ── Derived convenience fields — for logging / serialisation only ──────────
    composite_score:     float
    n_visible_kpts:      int
    identity_stability:  float
    zone_contacts:       tuple[str, ...]
    action_top:          str           # derived from action_distribution.top_label
    action_top_conf:     float         # derived from action_distribution.top_conf
    action_entropy:      float         # derived from action_distribution.entropy
    cooldown_key:        str | None    # which cooldown blocked (if any)
    duplicate_of:        str | None    # candidate_id of the active duplicate (if any)
    processed_at_sec:    float         # wall-clock time of this decision

    def to_dict(self) -> dict:
        return {
            "decision_id":        self.decision_id,
            "candidate_id":       self.candidate_id,
            "track_id":           self.track_id,
            "camera_id":          self.camera_id,
            "timestamp_sec":      round(self.timestamp_sec, 3),
            "verdict":            self.verdict.value,
            "suppress_reasons":   sorted(r.value for r in self.suppress_reasons),
            # Full distributions preserved
            "action_distribution": {k: round(v, 4)
                                    for k, v in self.action_distribution.class_probs.items()},
            "action_top3":        [(a, round(c, 4))
                                   for a, c in self.action_distribution.top_n(3)],
            "confidence":         self.confidence.as_dict(),
            # Derived convenience fields
            "composite_score":    round(self.composite_score, 4),
            "n_visible_kpts":     self.n_visible_kpts,
            "identity_stability": round(self.identity_stability, 4),
            "zone_contacts":      list(self.zone_contacts),
            "action_top":         self.action_top,
            "action_top_conf":    round(self.action_top_conf, 4),
            "action_entropy":     round(self.action_entropy, 4),
            "cooldown_key":       self.cooldown_key,
            "duplicate_of":       self.duplicate_of,
            "processed_at_sec":   round(self.processed_at_sec, 3),
        }


@dataclass
class CooldownEntry:
    """
    Tracks cooldown state for a (zone_id, action_top) pair.
    Cooldown prevents the same signal from re-entering as ACTIVE
    within the cooldown window after a prior promotion.
    """
    key:            str       # f"{zone_id}:{action_top}"
    expires_at_sec: float     # wall-clock expiry (monotonic)
    promoted_from:  str       # candidate_id that started this cooldown


@dataclass
class ActiveCandidateEntry:
    """
    Tracks a currently ACTIVE candidate for duplicate detection.
    An ACTIVE candidate expires when a configurable active_window_sec elapses.

    RISK2 — action_top removed:
        The previous version stored action_top: str — a frozen label extracted
        from the distribution at promotion time. This is scalar compression.
        Duplicate detection only requires track identity + spatial context
        (track_id + zone_set). The action distribution is irrelevant to the
        question "is this the same spatial event already being tracked?"
        action_top has been removed entirely from this struct.

    RISK1 — no behavioral vocabulary:
        This struct describes continuity state (track + zone), not behavior.
        It answers "is there already an active observation for this track here?"
        not "is this action already being handled?"
    """
    candidate_id:   str
    track_id:       int
    zone_set:       frozenset[str]
    activated_at:   float     # wall-clock time of promotion
    expires_at:     float     # wall-clock time this entry becomes stale


# =============================================================================
# FILTER RULES
# Each rule is a pure function: (Candidate, FilterContext) → SuppressComponent | None
# None means PASS. A SuppressComponent means this rule wants to suppress.
# All rules run regardless — no short-circuit — so the full reason vector is built.
# =============================================================================

@dataclass(frozen=True)
class FilterContext:
    """
    Immutable context passed to each rule.
    Built once per candidate evaluation cycle.
    Contains all external state rules may need.

    frozen=True is mandatory (RISK 4):
        Rules are pure functions. A mutable context would allow a rule
        to inadvertently modify shared engine state. frozen=True makes
        that a TypeError at the point of mutation, not a silent data race.

    The dicts (zone_registry, cooldown_registry, active_registry) are
    references to the engine live dicts — they are NOT copies.
    Rules must treat them as read-only and must NEVER mutate them.
    Mutations happen only through FilterEngine._on_promote() under lock.
    """
    zone_registry:       dict       # zone_id → ZoneDefinition  [read-only in rules]
    cooldown_registry:   dict       # key → CooldownEntry       [read-only in rules]
    active_registry:     dict       # candidate_id → ActiveCandidateEntry [read-only]
    wall_clock_now:      float      # monotonic seconds
    cfg:                 FilterConfig


def _rule_composite_score(c: "Candidate", ctx: FilterContext) -> SuppressComponent | None:
    """
    Rule 1 — Composite score floor.

    composite_score is the weighted, conditionally-normalised signal
    from Phase 1's ConfidenceProfile. Low composite = poor signal quality.
    A candidate that fails this gate is structurally too uncertain to promote.
    """
    if c.composite_score < ctx.cfg.min_composite_score:
        return SuppressComponent.LOW_COMPOSITE_SCORE
    return None


def _rule_pose_quality(c: "Candidate", ctx: FilterContext) -> SuppressComponent | None:
    """
    Rule 2 — Pose quality gate.

    Two conditions, either failing triggers suppression:
        a) fewer than min_visible_keypoints keypoints above kpt_conf_min
        b) confidence.pose_valid is explicitly False (set by Phase 1's ConfidenceModel)

    This gate is separate from composite_score because:
        A high composite_score can still carry a degraded skeleton if
        detection_conf and tracking_oks are strong.
        The LSTM window is unreliable when the skeleton is incomplete.
        We want explicit, transparent pose-quality gating — not hidden in weights.
    """
    n_vis = sum(1 for v in c.kp_confidence if v >= ctx.cfg.kpt_conf_min)
    pose_ok = (
        n_vis >= ctx.cfg.min_visible_keypoints
        and c.confidence.pose_valid
    )
    if not pose_ok:
        return SuppressComponent.POOR_POSE_QUALITY
    return None


def _rule_tracking_stability(c: "Candidate", ctx: FilterContext) -> SuppressComponent | None:
    """
    Rule 3 — Tracking stability gate.

    identity_stability is the mean OKS to the previous frame from Phase 1.
    Low stability means the track may have jumped to a different person.

    Exception: NEW_TRACK deltas are exempt.
    A first observation has no prior frame to compute OKS against.
    Requiring stability on NEW_TRACK would systematically suppress every new person.

    Configurable floor: unstable_stability_floor.
    Below this, the continuity hypothesis is too weak to trust.
    """
    from phase1_observation import DeltaReason
    if c.delta_reason == DeltaReason.NEW_TRACK:
        return None  # exempt
    if c.identity_stability < ctx.cfg.min_identity_stability:
        return SuppressComponent.UNSTABLE_TRACKING
    return None


def _rule_zone_contact(c: "Candidate", ctx: FilterContext) -> SuppressComponent | None:
    """
    Rule 4 — Zone contact requirement.

    If require_zone_contact=True, a candidate must intersect at least one
    registered zone. Candidates outside all zones carry no spatial context
    and are structurally unanchored for downstream event derivation.

    If require_zone_contact=False, this rule always passes.
    Use False for open-area monitoring without zone config.

    Note: zone_contacts is set by Phase 1 from the zone_registry.
    If the zone_registry is empty, all candidates will have no zone_contacts
    and this gate will suppress everything. Configure zones before enabling.
    """
    if not ctx.cfg.require_zone_contact:
        return None
    if not ctx.zone_registry:
        return None  # no zones registered — open-area mode, pass all
    if not c.zone_contacts:
        return SuppressComponent.NO_ZONE_CONTACT
    # Verify at least one contact is a known registered zone
    known = set(ctx.zone_registry.keys())
    if not any(z in known for z in c.zone_contacts):
        return SuppressComponent.NO_ZONE_CONTACT
    return None


def _rule_cooldown(c: "Candidate", ctx: FilterContext) -> tuple[SuppressComponent | None, str | None]:
    """
    Rule 5 — Cooldown gate.

    After a candidate is promoted, a cooldown is registered for the
    (zone_id, action_top) pair. Subsequent candidates matching the same
    zone+action pair within the cooldown window are suppressed.

    This prevents the same ongoing action in the same zone from flooding
    the active candidate registry with near-duplicate promotions.

    Cooldown keys are checked for ALL zone contacts × action_top.
    If any key is under cooldown, the candidate is suppressed.
    cooldown_key in FilterDecision records which key blocked.

    Returns (SuppressComponent | None, blocking_key | None)
    """
    action_top = c.action_distribution.top_label
    now        = ctx.wall_clock_now

    for zone_id in (c.zone_contacts or [_NO_ZONE_SENTINEL]):
        key = f"{zone_id}:{action_top}"
        entry = ctx.cooldown_registry.get(key)
        if entry is not None and entry.expires_at_sec > now:
            return SuppressComponent.COOLDOWN_ACTIVE, key

    return None, None


def _rule_duplicate(c: "Candidate", ctx: FilterContext) -> tuple[SuppressComponent | None, str | None]:
    """
    Rule 6 — Duplicate detection.

    A candidate is a duplicate if there is already an ACTIVE (non-expired)
    candidate for the SAME track_id AND overlapping zone set.

    Zone overlap is defined as: intersection of zone sets is non-empty.
    (not equality — a track that spans two zones that share one zone
    with an existing active candidate counts as a duplicate)

    duplicate_of in FilterDecision records the conflicting candidate_id.

    Returns (SuppressComponent | None, duplicate_candidate_id | None)
    """
    now        = ctx.wall_clock_now
    c_zones    = frozenset(c.zone_contacts)
    c_track    = c.track_id

    for active in ctx.active_registry.values():
        if active.expires_at <= now:
            continue  # stale — will be pruned next cycle
        if active.track_id != c_track:
            continue
        # Same track — check zone overlap
        if c_zones & active.zone_set:
            return SuppressComponent.DUPLICATE_TRACK_ZONE, active.candidate_id

    return None, None


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FilterConfig:
    # ── Rule 1: Composite score ────────────────────────────────────────────────
    min_composite_score:    float = 0.40

    # ── Rule 2: Pose quality ──────────────────────────────────────────────────
    min_visible_keypoints:  int   = 10
    kpt_conf_min:           float = 0.25

    # ── Rule 3: Tracking stability ────────────────────────────────────────────
    min_identity_stability: float = 0.30   # OKS floor; NEW_TRACK exempt

    # ── Rule 4: Zone contact ──────────────────────────────────────────────────
    require_zone_contact:   bool  = False  # True = candidates must be in a zone

    # ── Rule 5: Cooldown ──────────────────────────────────────────────────────
    cooldown_sec:                   float = 3.0    # per (zone, action_top) cooldown window
    min_action_conf_for_cooldown:   float = 0.40
    # Minimum action_top_conf required to register a cooldown.
    # Below this, the action label is too uncertain to use as a cooldown discriminator.
    # Specifically protects against absent()/uninformed() distributions where
    # top_label='unknown' would create cross-track cooldown interference.

    # ── Rule 6: Duplicate ─────────────────────────────────────────────────────
    active_window_sec:      float = 10.0   # active candidate TTL for dup detection

    # ── Polling ───────────────────────────────────────────────────────────────
    poll_interval_sec:      float = 0.05   # how often to drain blackboard
    prune_interval_sec:     float = 5.0    # how often to expire stale entries

    # ── Permissive mode ───────────────────────────────────────────────────────
    # In permissive mode, candidates that fail ONLY soft rules
    # (UNSTABLE_TRACKING, COOLDOWN_ACTIVE, DUPLICATE_TRACK_ZONE)
    # are PROMOTED with a degraded flag rather than suppressed.
    # Use for initial calibration / zone tuning.
    permissive_mode:        bool  = False

    # ── Soft rules (only relevant when permissive_mode=True) ──────────────────
    # These reasons are treated as warnings, not hard failures, in permissive mode.
    soft_rules: frozenset = field(default_factory=lambda: frozenset({
        SuppressComponent.UNSTABLE_TRACKING,
        SuppressComponent.COOLDOWN_ACTIVE,
        SuppressComponent.DUPLICATE_TRACK_ZONE,
    }))


# =============================================================================
# BLACKBOARD EXTENSION — Phase 2 write contract
# =============================================================================

class FilterBlackboardExtension:
    """
    Phase 2's write surface on the Blackboard.

    Phase 1's Blackboard has:
        write_candidate()   → EMERGING state
        drain_emerging()    → consumed by Phase 2

    Phase 2 needs to add:
        promote_candidate() → EMERGING → ACTIVE
        suppress_candidate() → EMERGING → SUPPRESSED
        get_active_candidates() → list of ACTIVE candidates for Phase 3

    These methods are injected onto the Blackboard instance at runtime
    (monkey-patching avoided — we wrap the blackboard in an adapter).
    The Blackboard object itself is not modified.

    Thread safety: same RLock as the Blackboard's own lock.
    """

    def __init__(self, blackboard: "Blackboard") -> None:
        self._bb   = blackboard
        # We use the same RLock as Phase 1's Blackboard so that our state
        # transitions (promote/suppress) are atomic with Blackboard writes.
        # Accessing _lock by name is a deliberate cross-module coupling —
        # noted here so it is visible, not buried in method bodies.
        # Future: Phase 1 Blackboard should expose a public acquire() context
        # manager so this reference can be replaced without name coupling.
        self._lock = blackboard._lock

        # Phase 2-owned state, separate from Blackboard internals
        self._active:       dict[str, "Candidate"] = {}   # candidate_id → Candidate (ACTIVE)
        self._suppressed:   dict[str, "Candidate"] = {}   # candidate_id → Candidate (SUPPRESSED)
        self._decisions:    list[FilterDecision]   = []   # full audit log
        self._consumed_ids: set[str]               = set() # delivered to Phase 3; never re-deliver

    # ── Phase 2 write API ─────────────────────────────────────────────────────

    def promote(self, c: "Candidate", decision: FilterDecision) -> None:
        """
        Promote candidate to CONFIRMED state.

        RISK 4 (mutable shared memory):
            All state transitions go through Blackboard.transition_candidate_state().
            That method holds the Blackboard's own lock and produces a new Candidate
            via dataclasses.replace() — no in-place mutation, no lock bypass.
            The original Candidate reference held by Phase 1 is never modified.
        """
        from phase1_observation import CandidateState
        promoted = self._bb.transition_candidate_state(c, CandidateState.CONFIRMED)
        with self._lock:
            self._active[promoted.candidate_id] = promoted
            self._decisions.append(decision)
        log.debug(
            "Filter PROMOTE  cid=%s  track=%d  score=%.3f  zones=%s  action=%s(%.2f)",
            promoted.candidate_id[:8], promoted.track_id, promoted.composite_score,
            promoted.zone_contacts, promoted.action_distribution.top_label,
            promoted.action_distribution.top_conf,
        )

    def suppress(self, c: "Candidate", decision: FilterDecision) -> None:
        """
        Suppress candidate. State → SUPPRESSED.
        Candidate is never deleted — suppression is always auditable.

        Same transition_candidate_state() discipline as promote() — no in-place
        mutation, no direct Blackboard field access.
        """
        from phase1_observation import CandidateState
        suppressed = self._bb.transition_candidate_state(c, CandidateState.SUPPRESSED)
        with self._lock:
            self._suppressed[suppressed.candidate_id] = suppressed
            self._decisions.append(decision)
        log.debug(
            "Filter SUPPRESS cid=%s  track=%d  reasons=%s",
            suppressed.candidate_id[:8], suppressed.track_id,
            sorted(r.value for r in decision.suppress_reasons),
        )

    # ── Phase 3 read API ──────────────────────────────────────────────────────

    def drain_active(self) -> list["Candidate"]:
        """
        Phase 3 consumes CONFIRMED candidates.
        Returns only candidates Phase 3 has NOT yet received (new since last drain).
        Does NOT change candidate state — Phase 3 owns the lifecycle from
        CONFIRMED → SUPPRESSED / EXPIRED.

        Consumed tracking:
            _consumed_ids tracks candidate_ids already returned to Phase 3.
            A candidate is delivered exactly once through this method.
            Phase 3 must explicitly mark candidates SUPPRESSED or EXPIRED
            when resolved — it cannot rely on drain_active to re-deliver.

        This prevents re-delivery: Phase 3 would otherwise receive the same
        CONFIRMED candidates on every poll because _active is never cleared.
        """
        with self._lock:
            new_batch = [
                c for cid, c in self._active.items()
                if cid not in self._consumed_ids
            ]
            for c in new_batch:
                self._consumed_ids.add(c.candidate_id)
            return new_batch

    def get_active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def get_suppressed_count(self) -> int:
        with self._lock:
            return len(self._suppressed)

    def get_decisions(self) -> list[FilterDecision]:
        with self._lock:
            return list(self._decisions)

    def get_decisions_for_track(self, track_id: int) -> list[FilterDecision]:
        with self._lock:
            return [d for d in self._decisions if d.track_id == track_id]


# =============================================================================
# FILTER ENGINE — rule chain execution
# =============================================================================

class FilterEngine:
    """
    Stateful filter engine. Holds cooldown and active-candidate registries.
    Applies the full six-rule chain to each candidate.
    Emits FilterDecisions.

    All state is internal to this engine.
    The engine writes decisions through the FilterBlackboardExtension.
    """

    def __init__(self, cfg: FilterConfig, bb_ext: FilterBlackboardExtension) -> None:
        self._cfg       = cfg
        self._bb_ext    = bb_ext
        self._cooldowns: dict[str, CooldownEntry]        = {}
        self._actives:   dict[str, ActiveCandidateEntry] = {}
        self._lock       = threading.Lock()
        self._last_prune = time.monotonic()

    def evaluate(self, c: "Candidate") -> FilterDecision:
        """
        Run all six rules against the candidate.
        Build a FilterDecision. Execute promotion or suppression.
        Register cooldown and active entry if promoted.
        Return the FilterDecision for the drop log.
        """
        now = time.monotonic()
        self._maybe_prune(now)

        ctx = FilterContext(
            zone_registry     = self._bb_ext._bb.zone_registry,
            cooldown_registry = self._cooldowns,
            active_registry   = self._actives,
            wall_clock_now    = now,
            cfg               = self._cfg,
        )

        suppress_reasons: set[SuppressComponent] = set()
        cooldown_key:  str | None = None
        duplicate_of:  str | None = None

        # ── Run all rules, collect all failures ──────────────────────────────
        r1 = _rule_composite_score(c, ctx)
        if r1: suppress_reasons.add(r1)

        r2 = _rule_pose_quality(c, ctx)
        if r2: suppress_reasons.add(r2)

        r3 = _rule_tracking_stability(c, ctx)
        if r3: suppress_reasons.add(r3)

        r4 = _rule_zone_contact(c, ctx)
        if r4: suppress_reasons.add(r4)

        r5_comp, r5_key = _rule_cooldown(c, ctx)
        if r5_comp:
            suppress_reasons.add(r5_comp)
            cooldown_key = r5_key

        r6_comp, r6_dup = _rule_duplicate(c, ctx)
        if r6_comp:
            suppress_reasons.add(r6_comp)
            duplicate_of = r6_dup

        # ── Permissive mode: demote soft failures to warnings ─────────────────
        hard_failures = suppress_reasons - self._cfg.soft_rules
        if self._cfg.permissive_mode:
            effective_failures = hard_failures
        else:
            effective_failures = suppress_reasons

        verdict = FilterVerdict.PROMOTED if not effective_failures else FilterVerdict.SUPPRESSED

        decision = FilterDecision(
            decision_id          = str(uuid.uuid4()),
            candidate_id         = c.candidate_id,
            track_id             = c.track_id,
            camera_id            = c.camera_id,
            timestamp_sec        = c.timestamp_sec,
            verdict              = verdict,
            suppress_reasons     = frozenset(suppress_reasons),  # full set, including soft
            # Full distributions — primary record (RISK 2)
            action_distribution  = c.action_distribution,
            confidence           = c.confidence,
            # Derived convenience fields
            composite_score      = c.composite_score,
            n_visible_kpts       = sum(1 for v in c.kp_confidence if v >= self._cfg.kpt_conf_min),
            identity_stability   = c.identity_stability,
            zone_contacts        = tuple(c.zone_contacts),
            action_top           = c.action_distribution.top_label,
            action_top_conf      = c.action_distribution.top_conf,
            action_entropy       = c.action_distribution.entropy,
            cooldown_key         = cooldown_key,
            duplicate_of         = duplicate_of,
            processed_at_sec     = now,
        )

        if verdict == FilterVerdict.PROMOTED:
            self._on_promote(c, now)
            self._bb_ext.promote(c, decision)
        else:
            self._bb_ext.suppress(c, decision)

        return decision

    def _on_promote(self, c: "Candidate", now: float) -> None:
        """
        Register cooldown + active entry after a successful promotion.

        Cooldown registration guard (RISK2 / absent-distribution fix):
            Cooldown key is zone × action_top_label. If the candidate carries
            an absent() or uninformed() distribution (action_top_conf below
            min_action_conf_for_cooldown), we do NOT register a cooldown.
            Reason: absent().top_label == 'unknown' — all absent-distribution
            candidates in the same zone would share the key 'zone_id:unknown',
            meaning the first promoted absent-dist candidate would block all
            subsequent absent-dist candidates in that zone for cooldown_sec.
            That is a cross-track interference from a meaningless label.

        Active entry (RISK2 fix):
            action_top removed from ActiveCandidateEntry.
            Duplicate detection is purely spatial+identity (track_id + zone_set).
            The action distribution does not determine whether a spatial event
            is already being tracked.
        """
        action_top      = c.action_distribution.top_label
        action_top_conf = c.action_distribution.top_conf
        with self._lock:
            # Only register cooldown when action classification is confident
            if action_top_conf >= self._cfg.min_action_conf_for_cooldown:
                for zone_id in (c.zone_contacts or [_NO_ZONE_SENTINEL]):
                    key = f"{zone_id}:{action_top}"
                    self._cooldowns[key] = CooldownEntry(
                        key            = key,
                        expires_at_sec = now + self._cfg.cooldown_sec,
                        promoted_from  = c.candidate_id,
                    )
            else:
                log.debug(
                    "Cooldown skipped for cid=%s: action_top_conf=%.3f < %.3f "
                    "(absent/uninformed distribution — no cooldown registered)",
                    c.candidate_id[:8], action_top_conf,
                    self._cfg.min_action_conf_for_cooldown,
                )
            # Active entry: track_id + zone_set only — no frozen label
            self._actives[c.candidate_id] = ActiveCandidateEntry(
                candidate_id = c.candidate_id,
                track_id     = c.track_id,
                zone_set     = frozenset(c.zone_contacts),
                activated_at = now,
                expires_at   = now + self._cfg.active_window_sec,
            )

    def _maybe_prune(self, now: float) -> None:
        """
        Expire stale cooldowns and active entries periodically.

        Thread safety:
            self._last_prune is read outside the lock for the early-exit check —
            this is a deliberate benign race. Two threads may both pass the check
            and both run a prune in the same interval. That is safe: double-pruning
            is idempotent. The alternative (locking the early-exit check too) would
            serialise every evaluate() call on the prune lock unnecessarily.

            self._last_prune is updated INSIDE the lock after the prune work,
            so the update is atomic with respect to the dict mutations.
        """
        if now - self._last_prune < self._cfg.prune_interval_sec:
            return
        with self._lock:
            dead_cd = [k for k, v in self._cooldowns.items() if v.expires_at_sec <= now]
            for k in dead_cd:
                del self._cooldowns[k]
            dead_ac = [k for k, v in self._actives.items() if v.expires_at <= now]
            for k in dead_ac:
                del self._actives[k]
            self._last_prune = now   # updated inside lock — atomic with dict mutations
        if dead_cd or dead_ac:
            log.debug("Filter prune: expired %d cooldowns, %d active entries",
                      len(dead_cd), len(dead_ac))

    @property
    def cooldown_count(self) -> int:
        return len(self._cooldowns)

    @property
    def active_entry_count(self) -> int:
        return len(self._actives)


# =============================================================================
# FILTER STATISTICS — observability without side effects
# =============================================================================

@dataclass
class FilterStats:
    """
    Running statistics for the filter leg.
    Read-only. Updated on every decision cycle.
    """
    total_evaluated:   int = 0
    total_promoted:    int = 0
    total_suppressed:  int = 0
    reason_counts:     dict = field(default_factory=dict)
    promotion_rate:    float = 0.0
    cycles_run:        int = 0
    last_cycle_sec:    float = 0.0

    def record(self, decisions: list[FilterDecision]) -> None:
        self.total_evaluated  += len(decisions)
        for d in decisions:
            if d.verdict == FilterVerdict.PROMOTED:
                self.total_promoted += 1
                # Do NOT count suppress_reasons here even if non-empty.
                # In permissive mode, soft-rule failures appear in suppress_reasons
                # but the candidate was still promoted. Counting them as suppression
                # reasons would inflate reason_counts and misrepresent filter behaviour.
                # Soft failures on promoted candidates are available in the full
                # FilterDecision audit log for separate analysis if needed.
            else:
                self.total_suppressed += 1
                for r in d.suppress_reasons:
                    self.reason_counts[r.value] = self.reason_counts.get(r.value, 0) + 1
        self.promotion_rate = (
            self.total_promoted / max(self.total_evaluated, 1)
        )
        self.cycles_run     += 1
        self.last_cycle_sec  = time.monotonic()

    def to_dict(self) -> dict:
        return {
            "total_evaluated":  self.total_evaluated,
            "total_promoted":   self.total_promoted,
            "total_suppressed": self.total_suppressed,
            "promotion_rate":   round(self.promotion_rate, 4),
            "cycles_run":       self.cycles_run,
            "suppress_reasons": self.reason_counts,
        }


# =============================================================================
# FILTER LEG — background thread entry point
# =============================================================================

class FilterLeg:
    """
    Phase 2 entry point.

    Runs as a background daemon thread.
    Polls Blackboard.drain_emerging() at poll_interval_sec.
    Applies FilterEngine to each candidate batch.
    Writes promotion/suppression decisions through FilterBlackboardExtension.

    Usage:
        from phase1_observation import Blackboard
        from phase2_filter import FilterLeg, FilterConfig

        bb     = Blackboard(contract)
        cfg_f  = FilterConfig()
        leg2   = FilterLeg(cfg_f, bb)
        leg2.start()

        # ... Phase 1 runs and fills blackboard ...

        leg2.stop()
        leg2.join()

        # Inspect results
        print(leg2.stats.to_dict())
        print(f"Promoted:   {leg2.bb_ext.get_active_count()}")
        print(f"Suppressed: {leg2.bb_ext.get_suppressed_count()}")
        for d in leg2.drop_log:
            if d.verdict == FilterVerdict.SUPPRESSED:
                print(d.to_dict())
    """

    def __init__(self, cfg: FilterConfig, blackboard: "Blackboard") -> None:
        self._cfg    = cfg
        self._bb     = blackboard
        self.bb_ext  = FilterBlackboardExtension(blackboard)
        self._engine = FilterEngine(cfg, self.bb_ext)
        self._stats  = FilterStats()
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background filter thread."""
        self._thread = threading.Thread(
            target  = self._run,
            daemon  = True,
            name    = "Ph2-Filter",
        )
        self._thread.start()
        log.info(
            "Phase 2 Filter started. "
            "min_score=%.2f  min_kpts=%d  stability_floor=%.2f  "
            "cooldown=%.1fs  active_ttl=%.1fs  require_zone=%s  permissive=%s",
            self._cfg.min_composite_score,
            self._cfg.min_visible_keypoints,
            self._cfg.min_identity_stability,
            self._cfg.cooldown_sec,
            self._cfg.active_window_sec,
            self._cfg.require_zone_contact,
            self._cfg.permissive_mode,
        )

    def stop(self) -> None:
        """Signal the filter thread to stop after current batch."""
        self._stop.set()

    def join(self, timeout: float = 10.0) -> None:
        """Wait for the filter thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info(
            "Phase 2 complete. evaluated=%d  promoted=%d  suppressed=%d  "
            "rate=%.3f  cycles=%d",
            self._stats.total_evaluated,
            self._stats.total_promoted,
            self._stats.total_suppressed,
            self._stats.promotion_rate,
            self._stats.cycles_run,
        )

    def _run(self) -> None:
        """Main filter loop."""
        log.info("Phase 2 filter loop running.")
        pbar = _tqdm(
            desc="Phase 2 — Filtering",
            unit="cands",
            dynamic_ncols=True,
            colour="yellow",
            leave=True,
        ) if _TQDM_OK else None
        try:
            while not self._stop.is_set():
                try:
                    batch = self._bb.drain_emerging()
                    if batch:
                        decisions = [self._engine.evaluate(c) for c in batch]
                        self._stats.record(decisions)
                        if pbar is not None:
                            promoted = sum(1 for d in decisions if d.verdict == FilterVerdict.PROMOTED)
                            pbar.update(len(decisions))
                            pbar.set_postfix({
                                "pass": self._stats.total_promoted,
                                "drop": self._stats.total_suppressed,
                                "rate": f"{self._stats.promotion_rate:.0%}",
                            }, refresh=False)
                        if self._stats.cycles_run % 20 == 0:
                            log.info(
                                "Filter stats: %s  cooldowns=%d  actives=%d",
                                self._stats.to_dict(),
                                self._engine.cooldown_count,
                                self._engine.active_entry_count,
                            )
                except Exception as e:
                    log.exception("Phase 2 filter error: %s", e)

                self._stop.wait(timeout=self._cfg.poll_interval_sec)
        finally:
            if pbar is not None:
                pbar.close()

        # Drain any remaining candidates before exit
        try:
            final_batch = self._bb.drain_emerging()
            if final_batch:
                decisions = [self._engine.evaluate(c) for c in final_batch]
                self._stats.record(decisions)
                log.info("Phase 2 final drain: %d candidates processed.", len(final_batch))
        except Exception as e:
            log.exception("Phase 2 final drain error: %s", e)

    @property
    def stats(self) -> FilterStats:
        return self._stats

    @property
    def drop_log(self) -> list[FilterDecision]:
        """Full audit log. All decisions, promoted and suppressed."""
        return self.bb_ext.get_decisions()


def print_promoted_summary(promoted: list[FilterDecision], limit: int = 5) -> None:
    """Print a summary count plus a filtered sample of promoted candidates."""
    print(f"  Promoted candidates: {len(promoted)}")
    if not promoted:
        return

    # Skip high-entropy candidates that are effectively uniform/ambiguous.
    # These are rarely useful in a quick summary because the distribution
    # carries little meaningful signal.
    meaningful = [
        d for d in promoted
        if d.action_distribution.entropy <= 1.0
    ]
    if meaningful:
        sample = meaningful[:limit]
        skipped = len(promoted) - len(meaningful)
        print(f"  First {min(limit, len(sample))} meaningful promoted candidates:")
    else:
        sample = promoted[:limit]
        skipped = 0
        print(f"  First {min(limit, len(sample))} promoted candidates:")

    for d in sample:
        top3 = d.action_distribution.top_n(3)
        print(
            f"    track={d.track_id}  top3={top3}  "
            f"entropy={d.action_entropy:.2f}  stability={d.identity_stability:.3f}"
        )
    if skipped:
        print(f"    ({skipped} high-entropy promotions omitted from sample)")


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    import logging
    from collections import Counter

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 2 — Filter leg test")
    parser.add_argument("--input",      required=True,          help="Video file")
    parser.add_argument("--model",      default="",             help="LSTM model path")
    parser.add_argument("--mode",       type=int, default=2,    choices=[1, 2])
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--zones",      default=None,           help="JSON zones file")
    parser.add_argument("--skip",       type=int, default=1)
    parser.add_argument("--end",        type=float, default=0.0)
    parser.add_argument("--cooldown",   type=float, default=3.0)
    parser.add_argument("--score",      type=float, default=0.40, help="min composite score")
    parser.add_argument("--stability",  type=float, default=0.30)
    parser.add_argument("--require-zone", action="store_true")
    parser.add_argument("--permissive",   action="store_true")
    parser.add_argument("--out",        default="phase2_filter_results.json")
    args = parser.parse_args()

    # ── Import Phase 1 ────────────────────────────────────────────────────────
    from phase1_observation import (
        ObservationConfig, ObservationLeg, Blackboard, resolve_action_model_path,
    )

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

    bb   = Blackboard(cfg1.build_temporal_contract())
    leg2 = FilterLeg(cfg2, bb)
    leg1 = ObservationLeg(cfg1, bb)

    print(f"\n  Phase 1 + 2 pipeline test")
    print(f"  Input      : {args.input}")
    print(f"  Model      : {model_path}")
    print(f"  Mode       : {args.mode}  Zones: {len(zones_raw)}  Skip: {args.skip}")
    print(f"  Filter     : score={args.score}  stability={args.stability}  "
          f"cooldown={args.cooldown}s  require_zone={args.require_zone}")
    print(f"  Permissive : {args.permissive}")
    print()

    # Start filter first so it's ready when Phase 1 begins emitting
    leg2.start()
    leg1.run()          # blocking — Phase 2 runs concurrently
    leg2.stop()
    leg2.join()

    stats   = leg2.stats.to_dict()
    decided = leg2.drop_log
    promoted = [d for d in decided if d.verdict == FilterVerdict.PROMOTED]

    action_counts = Counter(d.action_top for d in promoted)
    action_probs_mean: dict[str, float] = {}
    if promoted:
        accum: dict[str, float] = {}
        for d in promoted:
            for k, v in d.action_distribution.class_probs.items():
                accum[k] = accum.get(k, 0.0) + float(v)
        n = float(len(promoted))
        action_probs_mean = {k: round(v / n, 4) for k, v in accum.items()}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  Phase 1 candidates emitted : {bb.count_candidates()}")
    print(f"  Phase 2 evaluated          : {stats['total_evaluated']}")
    print(f"  Promoted (ACTIVE)          : {stats['total_promoted']}")
    print(f"  Suppressed                 : {stats['total_suppressed']}")
    print(f"  Promotion rate             : {stats['promotion_rate']:.1%}")
    print(f"  Promoted action counts     : {dict(action_counts)}")
    print()
    print("  Suppress reasons:")
    for reason, count in sorted(stats["suppress_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:<30} {count}")
    print()

    # ── Write output ──────────────────────────────────────────────────────────
    out_data = {
        "meta": {
            "input":          args.input,
            "model_path":     model_path,
            "vision_mode":    args.mode,
            "frame_skip":     args.skip,
            "filter_config":  {
                "min_composite_score":    args.score,
                "min_identity_stability": args.stability,
                "cooldown_sec":           args.cooldown,
                "require_zone_contact":   args.require_zone,
                "permissive_mode":        args.permissive,
            },
            "promoted_action_counts": action_counts,
            "promoted_action_probs_mean": action_probs_mean,
        },
        "stats":   stats,
        "decisions": [d.to_dict() for d in sorted(decided, key=lambda x: x.timestamp_sec)],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"  Output: {args.out}")
    print()

    # ── Suppression breakdown (sample) ────────────────────────────────────────
    suppressed = [d for d in decided if d.verdict == FilterVerdict.SUPPRESSED]
    if suppressed:
        print(f"  First 5 suppressed candidates:")
        for d in suppressed[:5]:
            print(f"    track={d.track_id}  score={d.composite_score:.3f}  "
                  f"reasons={sorted(r.value for r in d.suppress_reasons)}  "
                  f"zones={list(d.zone_contacts)}")
    print()

    promoted = [d for d in decided if d.verdict == FilterVerdict.PROMOTED]
    if promoted:
        print_promoted_summary(promoted, limit=5)
    print()
