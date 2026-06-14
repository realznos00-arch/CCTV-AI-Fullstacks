"""
phase1_observation.py — Phase 1: Observation Leg (v3)
======================================================
Event Intelligence Pipeline — Speed Layer

STRUCTURAL RISKS ADDRESSED IN THIS VERSION:

    RISK 1 — SEMANTIC LEAKAGE
        Observation layer is epistemically neutral.
        It ONLY perceives, localises, classifies motion, maintains continuity.
        Every method name, field name, and enum value has been audited.
        No "suspicious", "novel", "important", "triggered" — anywhere.
        DeltaDetector answers: "what changed?"
        CandidateEmitter answers: "does this warrant memory?"
        These are explicitly separate classes.

    RISK 2 — OVER-COMPRESSION
        Confidence is never a scalar.
        ActionDistribution preserves the full probability vector over all classes.
        ConfidenceProfile carries four independent dimensions with invalidation flags.
        composite() is conditional and normalised — not additive.
        Observation stores action_distribution (full), not just top label.
        Candidate stores action_distribution (full), not just top label.
        Top label is derived on demand — never stored as primary truth.

    RISK 3 — TEMPORAL UNDERDEFINITION
        TemporalContract defines the system's observable universe:
            - frame_skip: fixed. changing it changes what behaviors exist.
            - observation_window_sec: maximum age of a valid observation.
            - temporal_pulse_sec: maximum silence before a forced delta.
            - infer_every: LSTM schedule in frames — explicit, not emergent.
        Every Observation carries ingestion_time_sec (wall clock)
        and processing_latency_ms for observability.
        ObservationStream respects observation_window_sec on reads.

    RISK 4 — MUTABLE SHARED MEMORY RISK
        Blackboard contains ONLY:
            - zone_registry (static config)
            - active_track_summaries (continuity state, not conclusions)
            - candidate_queue (unresolved, awaiting Phase 2)
        Blackboard does NOT contain:
            - observations (those live in ObservationStream)
            - action labels (those live in ActionDistribution inside observations)
            - semantic conclusions of any kind
        active_track_summaries stores ContinuityState — a humility struct
        that explicitly names the track as a "continuity hypothesis", not identity.
        Blackboard write methods are strictly typed. No free-form dict writes.

    RISK 5 — TAXONOMY RIGIDITY
        DeltaReason describes structural transitions only (ZONE_ENTRY, SPATIAL_DELTA...).
        No behavioral categories (no LOITERING, INTRUSION, SUSPICIOUS).
        ActionDistribution carries the full class probability vector.
        Downstream layers compose meaning from distributions — not from frozen labels.
        ContinuityState carries identity_stability — a float [0..1], not a bool.
        ObservationValidity is a structured report, not an enum verdict.

ARCHITECTURE:

    ingestion/    FrameSampler, TemporalContract
    perception/   PersonTracker, PoseFeatures, ActionModel, ConfidenceModel
    observation/  ObservationBuilder, DeltaDetector, CandidateEmitter
    memory/       ObservationStream, Blackboard
    domain/       all dataclasses and enums

Install:
    pip install ultralytics torch torchvision opencv-python numpy scipy

Usage:
    bb  = Blackboard()
    cfg = ObservationConfig(
        input_path="cctv.mp4",
        model_path="model-2-conf.pt",
        vision_mode=2,
        zones=[{"zone_id": "zone_a", "polygon": [[0.0,0.5],[0.35,0.5],[0.35,1.0],[0.0,1.0]]}],
    )
    ObservationLeg(cfg, bb).run()
"""

# python phase1_observation.py --input "D:\Download_path\12822182_3840_2160_30fps.mp4" --model "C:\Users\ASUS\Documents\AI\lstm_model\model-2-conf.pt" --mode 2


from __future__ import annotations

import collections
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except ImportError:
    _TQDM_OK = False

try:
    from scipy.signal import savgol_filter as _savgol
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

log = logging.getLogger(__name__)
_SENTINEL = object()

DEFAULT_ACTION_MODEL_BY_MODE: dict[int, str] = {
    1: r"C:\Users\ASUS\Documents\AI\lstm_model\model-1-inf.pt",
    2: r"C:\Users\ASUS\Documents\AI\lstm_model\model-2-conf.pt",
}


def resolve_action_model_path(vision_mode: int, explicit_model_path: str = "") -> str:
    """Resolve the action model path for mode 1/2, unless CLI overrides it."""
    if explicit_model_path and explicit_model_path.strip():
        return explicit_model_path.strip()

    mode = int(vision_mode)
    try:
        return DEFAULT_ACTION_MODEL_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported vision mode {vision_mode!r}. Use 1 or 2.") from exc


def _load_yolo_model(model_path: str) -> Any:
    """Load Ultralytics checkpoints across PyTorch 2.6's weights_only change."""
    from ultralytics import YOLO

    try:
        return YOLO(model_path)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise

        import torch

        log.warning(
            "YOLO checkpoint requires legacy pickle loading. Retrying with "
            "torch.load(weights_only=False); trust this checkpoint before use: %s",
            model_path,
        )
        original_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _torch_load_compat
        try:
            return YOLO(model_path)
        finally:
            torch.load = original_load


# =============================================================================
# DOMAIN — entities, schemas, state models
# All names are structurally neutral. No semantic loading.
# =============================================================================

# ── Action vocabulary ─────────────────────────────────────────────────────────
# These are motion/posture labels only. They do not imply threat or intent.
# Downstream phases assign operational meaning compositionally.
ACTION_CLASSES: list[str] = [
    "fall", "grab", "gun", "hit", "kick",
    "lying_down", "run", "sit", "sneak",
    "stand", "struggle", "throw", "walk",
]
NUM_CLASSES   = len(ACTION_CLASSES)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}

COCO_KP_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KP = 17

_OKS_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
], dtype=np.float64)

_N_ANGLES   = 6    # confirmed from checkpoint: lstm.weight_ih_l0 input=125
FEATURE_DIM = NUM_KP * 2 + NUM_KP * 2 + NUM_KP * 2 + _N_ANGLES + NUM_KP  # 125


# ── Enumerations ──────────────────────────────────────────────────────────────

class DeltaReason(str, Enum):
    """
    Structural state transitions only.
    No behavioral categories. No threat vocabulary.
    These describe WHAT CHANGED, not WHAT IT MEANS.
    Downstream phases compose meaning from these transitions.
    """
    NEW_TRACK      = "NEW_TRACK"      # track_id first observed
    ZONE_ENTRY     = "ZONE_ENTRY"     # centroid crossed into a registered zone
    ZONE_EXIT      = "ZONE_EXIT"      # centroid left a registered zone
    ACTION_SHIFT   = "ACTION_SHIFT"   # dominant action class changed
    SPATIAL_DELTA  = "SPATIAL_DELTA"  # centroid displacement exceeded threshold
    TEMPORAL_PULSE = "TEMPORAL_PULSE" # no transition detected within pulse window


class CandidateState(str, Enum):
    """Lifecycle state of a candidate — Phase 1 only ever writes EMERGING."""
    EMERGING   = "EMERGING"
    CONFIRMED  = "CONFIRMED"   # promoted by Phase 2
    SUPPRESSED = "SUPPRESSED"  # suppressed by Phase 2
    EXPIRED    = "EXPIRED"


class TrackLifecycle(str, Enum):
    """Track lifecycle. Continuity state, not identity state."""
    ACTIVE  = "ACTIVE"
    LOST    = "LOST"
    EXPIRED = "EXPIRED"


class VisionMode(int, Enum):
    BASIC      = 1   # model-1-inf.pt
    CALIBRATED = 2   # model-2-conf.pt


# ── Zone ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneDefinition:
    """
    A named spatial region.
    zone_type is a free-form tag — not an operational enum.
    Downstream phases assign meaning to zone_type values.
    """
    zone_id:   str
    polygon:   tuple[tuple[float, float], ...]  # normalised [0..1], immutable
    zone_type: str = "OBSERVATION"              # tag only — no operational enum
    priority:  int = 1
    metadata:  dict = field(default_factory=dict)

    def __post_init__(self):
        # Ensure polygon is a tuple-of-tuples (frozen)
        object.__setattr__(self, "polygon",
            tuple(tuple(p) for p in self.polygon))


# ── Temporal contract ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TemporalContract:
    """
    Explicit temporal guarantees for Phase 1.

    CRITICAL: frame_skip defines the observable universe of this system.
    Changing it changes what behaviors can exist in the system.
    It must be a deliberate, documented decision — not a runtime optimization.

    observation_window_sec: observations older than this are stale.
    temporal_pulse_sec: maximum silence before a TEMPORAL_PULSE delta fires.
    infer_every: LSTM runs once every N processed frames per track.
    wall_clock_budget_ms: target processing budget per frame (observability only).
    """
    frame_skip:             int   = 1       # process every Nth frame
    observation_window_sec: float = 30.0    # stale observation threshold
    temporal_pulse_sec:     float = 5.0     # max silence before forced delta
    infer_every:            int   = 8       # LSTM cadence in processed frames
    wall_clock_budget_ms:   float = 40.0    # observability budget (not enforced)

    def temporal_resolution_sec(self, src_fps: float) -> float:
        """The finest time grain this system can observe at this fps."""
        return self.frame_skip / max(src_fps, 1e-6)

    def is_observation_fresh(self, obs_ts: float, current_ts: float) -> bool:
        return (current_ts - obs_ts) <= self.observation_window_sec


# ── Confidence — multi-dimensional, never a scalar ───────────────────────────

@dataclass(frozen=True)
class ConfidenceProfile:
    """
    Multi-dimensional uncertainty. Never compressed to a scalar.

    Each dimension captures a different failure mode:
        detection_conf  — YOLO bbox probability (low → bbox noise)
        pose_conf       — mean visible keypoint confidence (low → skeleton unreliable)
        tracking_oks    — OKS continuity to previous frame (low → identity uncertain)
        temporal_weight — recency weight, decays with observation age

    Validity gates are conditional:
        pose_valid    = n_visible_kpts >= threshold
        action_valid  = pose_valid (action certainty depends on skeleton quality)
        tracking_valid = tracking_oks >= floor

    These are NOT additive. They are conditional and hierarchical.
    """
    detection_conf:  float
    pose_conf:       float
    tracking_oks:    float
    n_visible_kpts:  int
    temporal_weight: float = 1.0  # [0..1] recency weight, not yet decayed at Phase 1

    # Conditional validity — set by ConfidenceModel
    pose_valid:     bool = True
    tracking_valid: bool = True
    action_valid:   bool = True   # conditional on pose_valid

    def composite(
        self,
        w_detection: float,
        w_pose: float,
        w_action: float,
        w_tracking: float,
        action_conf: float,
    ) -> float:
        """
        Conditional composite score.
        Invalid dimensions are excluded and weights are renormalised.
        Does not produce false certainty from degraded inputs.
        """
        terms: list[tuple[float, float]] = [
            (w_detection, self.detection_conf),
            (w_pose,      self.pose_conf),
        ]
        if self.action_valid:
            terms.append((w_action, action_conf))
        if self.tracking_valid:
            terms.append((w_tracking, self.tracking_oks))
        w_total = sum(w for w, _ in terms)
        if w_total < 1e-8:
            return 0.0
        return sum(w * v for w, v in terms) / w_total * self.temporal_weight

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form. Preserves all dimensions."""
        return {
            "detection_conf":  round(self.detection_conf, 4),
            "pose_conf":       round(self.pose_conf, 4),
            "tracking_oks":    round(self.tracking_oks, 4),
            "n_visible_kpts":  self.n_visible_kpts,
            "temporal_weight": round(self.temporal_weight, 4),
            "pose_valid":      self.pose_valid,
            "tracking_valid":  self.tracking_valid,
            "action_valid":    self.action_valid,
        }


@dataclass(frozen=True)
class ObservationValidity:
    """
    Structured validity report — not an enum verdict.

    A single VALID/INVALID flag would compress information.
    This preserves the full reason tree so downstream phases
    can make informed decisions rather than inheriting a binary.
    """
    detection_ok: bool
    pose_ok:      bool
    tracking_ok:  bool
    detail:       str = ""   # human-readable explanation for debugging

    @property
    def is_valid(self) -> bool:
        return self.detection_ok and self.pose_ok

    # NOTE: tracking_ok=False does NOT invalidate the observation.
    # It degrades it. The observation is still emitted with tracking_valid=False
    # in its ConfidenceProfile. Downstream phases decide whether to use it.


# ── Action distribution — full probability space, never compressed ────────────

@dataclass(frozen=True)
class ActionDistribution:
    """
    The complete probability distribution over action classes.

    CRITICAL: this is the primary action representation.
    top_label and top_conf are DERIVED fields — not stored truth.

    Storing only top_label would destroy future reinterpretation.
    Example: {"running": 0.61, "walking": 0.33} is richer than "running".
    The 0.33 walking mass is operationally significant and must survive.

    class_probs is a dict[str, float] — open to new classes without schema change.
    """
    class_probs: dict[str, float]   # full distribution over all action classes

    @property
    def top_label(self) -> str:
        if not self.class_probs:
            return "unknown"
        return max(self.class_probs, key=self.class_probs.__getitem__)

    @property
    def top_conf(self) -> float:
        if not self.class_probs:
            return 0.0
        return self.class_probs[self.top_label]

    @property
    def entropy(self) -> float:
        """Distribution entropy — high entropy means high behavioral uncertainty."""
        vals = np.array(list(self.class_probs.values()), dtype=np.float64)
        vals = vals[vals > 1e-10]
        return float(-np.sum(vals * np.log(vals + 1e-10)))

    def top_n(self, n: int = 3) -> list[tuple[str, float]]:
        """Top-n alternatives — preserves ambiguity for downstream."""
        return sorted(self.class_probs.items(), key=lambda x: -x[1])[:n]

    @classmethod
    def from_logits(cls, logits_np: np.ndarray, disabled: set[str]) -> ActionDistribution:
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()
        return cls(class_probs={
            a: float(probs[i]) for i, a in enumerate(ACTION_CLASSES)
            if a not in disabled
        })

    @classmethod
    def null(cls) -> ActionDistribution:
        """Uniform prior — no information."""
        n = len(ACTION_CLASSES)
        return cls(class_probs={a: 1.0 / n for a in ACTION_CLASSES})

    @classmethod
    def uninformed(cls, disabled: set[str]) -> ActionDistribution:
        """Uniform prior over ENABLED classes only.

        Builds a local filtered list from ACTION_CLASSES minus disabled.
        Does NOT mutate the caller's disabled set or ACTION_CLASSES.

        Use this — not null() — whenever a prior is needed but LSTM has not
        run yet. null() includes disabled classes; uninformed() excludes them,
        so disabled classes never receive probability mass.
        """
        enabled = [a for a in ACTION_CLASSES if a not in disabled]
        n = len(enabled)
        if n == 0:
            return cls(class_probs={})
        p = 1.0 / n
        return cls(class_probs={a: p for a in enabled})

    @classmethod
    def absent(cls) -> ActionDistribution:
        """No distribution yet — LSTM has not produced output for this track.

        Semantically distinct from uninformed()/null() which fabricate a prior.
        Detect the absent state with: if not dist.class_probs
        """
        return cls(class_probs={})


# ── ContinuityState — track is a hypothesis, not an identity ─────────────────

@dataclass
class ContinuityState:
    """
    Per-track continuity hypothesis.
    Explicitly NOT an identity. Tracks represent continuity estimates.

    identity_stability [0..1]:
        Mean OKS over recent frames. Low stability means the track may have
        jumped to a different person. Downstream phases must not treat
        low-stability tracks as confirmed identity.

    last_action_distribution:
        The most recent ActionDistribution seen for this track.
        Stored so that ACTION_SHIFT deltas can supply the actual prior
        distribution rather than a fabricated null/uniform placeholder.
        Previously, ACTION_SHIFT used ActionDistribution.null() as the prior
        because the distribution wasn't stored — that was incorrect.

    This struct lives inside DeltaDetector. Never written to the Blackboard directly.
    The Blackboard receives TrackSummary — a lightweight, intentionally limited projection.
    """
    track_id:                 int
    first_seen_sec:           float
    last_delta_sec:           float               = 0.0
    last_centroid:            tuple               = (0.0, 0.0)
    last_action_label:        str                 = ""
    last_action_distribution: ActionDistribution  = field(
        default_factory=ActionDistribution.absent)  # absent until first LSTM output
    last_zones:               frozenset           = field(default_factory=frozenset)
    identity_stability:       float               = 1.0


@dataclass(frozen=True)
class TrackSummary:
    """
    Lightweight blackboard-facing track projection.
    Contains only what downstream phases need for scheduling and coordination.
    Does NOT contain action labels, confidence scores, or semantic conclusions.
    """
    track_id:           int
    last_seen_sec:      float
    last_bbox:          tuple[float, ...]   # (x1,y1,x2,y2) normalised
    zone_contacts:      tuple[str, ...]
    lifecycle:          TrackLifecycle
    identity_stability: float              # surfaced for Phase 2 awareness


# ── Observation — epistemically neutral perception output ─────────────────────

@dataclass
class Observation:
    """
    A single structured perception output. Epistemically neutral.

    Describes what was seen. Not what it means.
    Contains the full ActionDistribution — never just a top label.
    Contains the full ConfidenceProfile — never just a scalar.
    Contains ObservationValidity — a report, not a verdict.

    ingestion_time_sec: wall-clock time of frame capture (monotonic).
    processing_latency_ms: wall-clock time from capture to observation build.
    These are needed for temporal contract enforcement.
    """
    observation_id:        str
    track_id:              int
    frame_number:          int
    timestamp_sec:         float          # video time
    ingestion_time_sec:    float          # wall-clock capture time
    processing_latency_ms: float          # wall-clock latency
    camera_id:             str
    bbox:                  list[float]    # [x1,y1,x2,y2] normalised
    keypoints:             np.ndarray     # (17,2) normalised
    kp_confidence:         np.ndarray     # (17,) per-keypoint
    action_distribution:   ActionDistribution   # FULL distribution, not a label
    confidence:            ConfidenceProfile
    zone_contacts:         list[str]
    motion_vector:         tuple[float, float]  # (dx, dy) normalised
    validity:              ObservationValidity


# ── ObservationDelta — state transition record ────────────────────────────────

@dataclass(frozen=True)
class ObservationDelta:
    """
    A detected structural state transition.

    Immutable. Carries the triggering observation by reference.
    prior_action_distribution and current_action_distribution preserve full distributions.
    No behavioral label is assigned here.
    """
    delta_id:                    str
    track_id:                    int
    reason:                      DeltaReason
    observation:                 Observation
    prior_zones:                 frozenset
    current_zones:               frozenset
    prior_action_distribution:   ActionDistribution
    current_action_distribution: ActionDistribution


# ── Candidate — first remembered unit ────────────────────────────────────────

@dataclass
class Candidate:
    """
    The first remembered unit. Created from an ObservationDelta.
    Written to the Blackboard. Immutable contract — downstream phases must not mutate.

    CRITICAL fields:
        action_distribution — full probability vector, never just top label.
        confidence          — full ConfidenceProfile, never just composite score.
        delta_reason        — structural transition only, no behavioral category.

    composite_score is stored for Phase 2 filtering convenience,
    but it is derived from confidence — not the primary record.
    """
    candidate_id:         str
    track_id:             int
    camera_id:            str
    frame_number:         int
    timestamp_sec:        float
    delta_reason:         DeltaReason
    bbox:                 list[float]
    keypoints:            list[list[float]]
    kp_confidence:        list[float]
    zone_contacts:        list[str]
    action_distribution:  ActionDistribution    # FULL distribution
    confidence:           ConfidenceProfile     # FULL profile
    composite_score:      float                 # derived convenience field
    motion_vector:        list[float]
    identity_stability:   float                 # from ContinuityState at emission time
    state:                CandidateState = CandidateState.EMERGING

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form. Preserves full distributions."""
        return {
            "candidate_id":       self.candidate_id,
            "track_id":           self.track_id,
            "camera_id":          self.camera_id,
            "frame_number":       self.frame_number,
            "timestamp_sec":      round(self.timestamp_sec, 3),
            "delta_reason":       self.delta_reason.value,
            "bbox":               [round(v, 4) for v in self.bbox],
            "zone_contacts":      self.zone_contacts,
            "action_distribution": {k: round(v, 4) for k, v in
                                    self.action_distribution.class_probs.items()},
            "action_top":         self.action_distribution.top_label,
            "action_top_conf":    round(self.action_distribution.top_conf, 4),
            "action_entropy":     round(self.action_distribution.entropy, 4),
            "action_top3":        [(a, round(c, 4)) for a, c in
                                   self.action_distribution.top_n(3)],
            "confidence":         self.confidence.as_dict(),
            "composite_score":    round(self.composite_score, 4),
            "motion_vector":      [round(v, 5) for v in self.motion_vector],
            "identity_stability": round(self.identity_stability, 4),
            "state":              self.state.value,
        }


# =============================================================================
# MEMORY — append-only stream + disciplined blackboard
# =============================================================================

class ObservationStream:
    """
    Append-only log of all valid observations.
    Primary truth source. The blackboard is derived from this, not the reverse.

    Phase 1 appends. Downstream phases read.
    Enforces observation_window_sec on snapshot reads.
    """

    def __init__(self, contract: TemporalContract) -> None:
        self._lock     = threading.RLock()
        self._stream:  list[Observation] = []
        self._contract = contract

    def append(self, obs: Observation) -> None:
        with self._lock:
            self._stream.append(obs)

    def snapshot_fresh(self, current_ts: float) -> list[Observation]:
        """Returns only observations within the temporal window."""
        with self._lock:
            return [o for o in self._stream
                    if self._contract.is_observation_fresh(o.timestamp_sec, current_ts)]

    def snapshot_all(self) -> list[Observation]:
        with self._lock:
            return list(self._stream)

    def count(self) -> int:
        with self._lock:
            return len(self._stream)


class Blackboard:
    """
    Shared situational state. Disciplined. Ascetic.

    Contains ONLY:
        zone_registry       — static spatial config
        track_summaries     — lightweight ContinuityState projections
        candidate_queue     — unresolved EMERGING candidates for Phase 2
        observation_stream  — reference to append-only truth source

    Contains NEVER:
        action labels, confidence scores, anomaly judgments, alert history,
        operational conclusions, semantic interpretations of any kind.

    Write methods are strictly typed.
    No free-form dict writes allowed.
    Every field has an explicit contract comment.

    Thread safety: RLock on all mutations.
    """

    def __init__(self, contract: TemporalContract) -> None:
        self._lock               = threading.RLock()
        self._contract           = contract
        self.observation_stream  = ObservationStream(contract)

        # Static spatial config — written once at startup, read-only after
        self.zone_registry:   dict[str, ZoneDefinition] = {}

        # Continuity state projections — lifecycle + stability only
        # NOT: action labels, confidence, semantic state
        self.track_summaries: dict[int, TrackSummary]   = {}

        # Unresolved candidates awaiting Phase 2 promotion
        self._candidate_log:  list[Candidate]           = []
        self._candidates:     dict[str, Candidate]      = {}

    # ── Write API ─────────────────────────────────────────────────────────────

    def register_zones(self, zones: list[ZoneDefinition]) -> None:
        """Called once at startup. Zone registry is immutable after this."""
        with self._lock:
            for z in zones:
                self.zone_registry[z.zone_id] = z
                log.debug("Blackboard: zone registered  zone_id=%s  type=%s",
                          z.zone_id, z.zone_type)

    def write_observation(self, obs: Observation) -> None:
        """
        Primary write path.
        Appends to the authoritative stream.
        Updates the track summary — continuity state only.
        """
        self.observation_stream.append(obs)
        summary = TrackSummary(
            track_id           = obs.track_id,
            last_seen_sec      = obs.timestamp_sec,
            last_bbox          = tuple(obs.bbox),
            zone_contacts      = tuple(obs.zone_contacts),
            lifecycle          = TrackLifecycle.ACTIVE,
            identity_stability = obs.confidence.tracking_oks,
        )
        with self._lock:
            self.track_summaries[obs.track_id] = summary

    def write_candidate(self, c: Candidate) -> None:
        """Phase 1 only. Candidate must be in EMERGING state."""
        assert c.state == CandidateState.EMERGING, \
            f"Phase 1 must only write EMERGING candidates, got {c.state}"
        with self._lock:
            self._candidates[c.candidate_id] = c
            self._candidate_log.append(c)
        log.debug(
            "Blackboard: candidate  id=%s  track=%d  reason=%s  "
            "top_action=%s(%.2f)  composite=%.3f  stability=%.3f",
            c.candidate_id[:8], c.track_id, c.delta_reason.value,
            c.action_distribution.top_label, c.action_distribution.top_conf,
            c.composite_score, c.identity_stability,
        )

    def transition_candidate_state(self, c: Candidate, new_state: CandidateState) -> Candidate:
        """
        Phase 2 write path. Atomically replace a candidate with a new-state copy.

        RISK 4 (mutable shared memory):
            Phase 2 must never write directly to self._candidates.
            This method is the only authorised state-transition path.
            It acquires self._lock, produces a new Candidate via
            dataclasses.replace(), stores it, and returns it.
            The original Candidate reference held by Phase 1 is never modified.

        Returns the new Candidate so callers can use it downstream
        without a second lookup.
        """
        import dataclasses
        transitioned = dataclasses.replace(c, state=new_state)
        with self._lock:
            self._candidates[transitioned.candidate_id] = transitioned
        return transitioned

    def expire_track(self, track_id: int) -> None:
        with self._lock:
            if track_id in self.track_summaries:
                s = self.track_summaries[track_id]
                self.track_summaries[track_id] = TrackSummary(
                    track_id           = s.track_id,
                    last_seen_sec      = s.last_seen_sec,
                    last_bbox          = s.last_bbox,
                    zone_contacts      = s.zone_contacts,
                    lifecycle          = TrackLifecycle.EXPIRED,
                    identity_stability = s.identity_stability,
                )

    # ── Read API ──────────────────────────────────────────────────────────────

    def drain_emerging(self) -> list[Candidate]:
        """
        Phase 2 consumes EMERGING candidates.

        RISK 4 (re-delivery bug):
            Returns only candidates that are CURRENTLY in EMERGING state.
            Because phase2 immediately transitions promoted/suppressed candidates
            via transition_candidate_state(), those are no longer EMERGING on
            the next poll. Candidates that fail evaluation and remain EMERGING
            (e.g. if transition fails mid-flight) will be re-evaluated — this
            is intentional: they are genuinely unresolved.
            Phase 2 must call transition_candidate_state() for every candidate
            it receives, so the set returned here shrinks to zero each cycle.
        """
        with self._lock:
            return [c for c in self._candidates.values()
                    if c.state == CandidateState.EMERGING]

    def count_candidates(self) -> int:
        with self._lock:
            return len(self._candidates)

    def get_candidates_for_track(self, track_id: int) -> list[Candidate]:
        with self._lock:
            return [c for c in self._candidate_log if c.track_id == track_id]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ObservationConfig:
    # ── Input ─────────────────────────────────────────────────────────────────
    input_path:    str   = r"D:\Download_path\12822182_3840_2160_30fps.mp4"
    camera_id:     str   = "cam_01"
    start_sec:     float = 0.0
    end_sec:       float = 0.0       # 0 = full video

    # ── Models ────────────────────────────────────────────────────────────────
    yolo_model:    str   = "yolo26n-pose.pt"
    model_path:    str   = DEFAULT_ACTION_MODEL_BY_MODE[2]
    vision_mode:   int   = VisionMode.CALIBRATED
    device:        str   = "cuda"

    # ── YOLO ──────────────────────────────────────────────────────────────────
    conf_thresh:   float = 0.25
    iou_thresh:    float = 0.70
    kpt_conf_min:  float = 0.25
    yolo_input_sz: int   = 640
    max_persons:   int   = 10

    # ── Temporal contract (see TemporalContract) ──────────────────────────────
    frame_skip:             int   = 1
    observation_window_sec: float = 30.0
    temporal_pulse_sec:     float = 5.0
    infer_every:            int   = 8
    wall_clock_budget_ms:   float = 40.0

    # ── Tracking ──────────────────────────────────────────────────────────────
    oks_alpha:     float = 0.5
    track_max_age: int   = 12

    # ── LSTM ──────────────────────────────────────────────────────────────────
    window_size:   int   = 32
    hidden_size:   int   = 128
    num_layers:    int   = 2
    # Minimum frames required before attempting LSTM inference (allows
    # shorter tracks to still receive predictions; can be reduced for
    # faster but noisier outputs).
    min_infer_frames: int = 8

    # ── Confidence gates (conditional, not additive) ──────────────────────────
    min_visible_keypoints: int   = 10      # below → pose_valid = False
    min_tracking_oks:      float = 0.10    # below → tracking_valid = False
    min_detection_conf:    float = 0.25    # below → detection_ok = False

    # Composite weights — applied only to valid dimensions
    w_detection:  float = 0.25
    w_pose:       float = 0.30
    w_action:     float = 0.35
    w_tracking:   float = 0.10

    # Minimum composite for candidate promotion
    min_composite_score:    float = 0.40

    # ── Delta thresholds ──────────────────────────────────────────────────────
    spatial_delta_threshold: float = 0.025
    action_shift_min_conf:   float = 0.55   # top_conf must exceed this for ACTION_SHIFT

    # ── Zone contact ──────────────────────────────────────────────────────────
    zone_overlap_threshold: float = 0.0     # 0.0 = centroid mode; future: bbox overlap

    # ── Zones ─────────────────────────────────────────────────────────────────
    zones: list[dict] = field(default_factory=list)

    # ── Queues ────────────────────────────────────────────────────────────────
    raw_q_size: int = 32
    ann_q_size: int = 16

    # ── Disabled classes ──────────────────────────────────────────────────────
    # These classes are excluded from ActionDistribution.
    # Downstream phases can override this for reinterpretation.
    disabled_classes: list[str] = field(default_factory=lambda: [
        "gun", "fall", "grab", "hit", "kick",
        "lying_down", "sneak", "struggle", "throw",
    ])

    def build_temporal_contract(self) -> TemporalContract:
        return TemporalContract(
            frame_skip             = self.frame_skip,
            observation_window_sec = self.observation_window_sec,
            temporal_pulse_sec     = self.temporal_pulse_sec,
            infer_every            = self.infer_every,
            wall_clock_budget_ms   = self.wall_clock_budget_ms,
        )


# =============================================================================
# INGESTION — frame sampler with explicit temporal contract
# =============================================================================

class FrameSampler:
    """
    Static frame sampler.
    The frame_skip value defines the temporal resolution of the observable universe.
    It is a system contract, not a performance knob.
    """

    def __init__(self, contract: TemporalContract) -> None:
        self._skip = max(1, contract.frame_skip)

    def should_process(self, frame_idx: int) -> bool:
        return frame_idx % self._skip == 0

    @property
    def frame_skip(self) -> int:
        return self._skip


def _resolve_device(requested: str) -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    r = requested.strip().lower()
    if r == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if r.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cpu"


# =============================================================================
# PERCEPTION — each class has exactly one job
# =============================================================================

class PoseFeatures:
    """
    Feature engineering for LSTM input construction.
    Isolated. The observer runtime consumes prepared sequences.
    It does not own preprocessing complexity.
    """

    @staticmethod
    def joint_angle(p1: np.ndarray, center: np.ndarray, p2: np.ndarray) -> float:
        v1, v2 = p1 - center, p2 - center
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

    @staticmethod
    def extract_angles(kpts: np.ndarray) -> np.ndarray:
        # 6 core limb angles — confirmed from checkpoint input_size=125:
        # pos(34)+vel(34)+acc(34)+angles(6)+conf(17)=125
        # These are the first 6 of the original 12; the trailing 6 torso angles
        # were NOT in the training feature set for these checkpoints.
        triplets = [
            (5,7,9),(6,8,10),(5,11,13),(6,12,14),(11,13,15),(12,14,16),
        ]
        angles = []
        for p1i, ci, p2i in triplets:
            p1, pc, p2 = kpts[p1i], kpts[ci], kpts[p2i]
            if np.any(p1 != 0) and np.any(pc != 0) and np.any(p2 != 0):
                angles.append(PoseFeatures.joint_angle(p1, pc, p2))
            else:
                angles.append(0.0)
        return np.array(angles, dtype=np.float32)

    @staticmethod
    def bbox_diagonal_normalize(kpts: np.ndarray, bbox: list[float]) -> np.ndarray:
        cx   = (bbox[0] + bbox[2]) / 2.0
        cy   = (bbox[1] + bbox[3]) / 2.0
        diag = np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) + 1e-8
        return (kpts - np.array([cx, cy], dtype=np.float32)) / diag

    @staticmethod
    def normalize_skeleton(kpts: np.ndarray) -> np.ndarray:
        ls, rs = kpts[5], kpts[6]
        center = (ls + rs) / 2.0
        scale  = np.linalg.norm(rs - ls) + 1e-8
        return (kpts - center) / scale

    @staticmethod
    def smooth(pos_arr: np.ndarray, window: int = 9, poly: int = 3) -> np.ndarray:
        if not _SCIPY_OK:
            return pos_arr
        T = pos_arr.shape[0]; w = min(window, T)
        if w % 2 == 0: w -= 1
        if w < poly + 2 or T < w:
            return pos_arr
        out = pos_arr.copy()
        for k in range(NUM_KP):
            for d in range(2):
                out[:, k, d] = _savgol(pos_arr[:, k, d], window_length=w, polyorder=poly)
        return out

    @classmethod
    def build_sequence(
        cls,
        pos_buf:  list[np.ndarray],
        conf_buf: list[np.ndarray],
        bbox_buf: list[list[float]] | None = None,
    ) -> np.ndarray:
        T        = len(pos_buf)
        pos_arr  = cls.smooth(np.stack(pos_buf, axis=0))
        conf_arr = np.stack(conf_buf, axis=0)
        if bbox_buf and len(bbox_buf) == T:
            diag_norm = np.stack([cls.bbox_diagonal_normalize(pos_arr[t], bbox_buf[t])
                                   for t in range(T)])
        else:
            diag_norm = pos_arr.copy()
        norm_pos = np.stack([cls.normalize_skeleton(diag_norm[t]) for t in range(T)])
        vel = np.zeros_like(norm_pos)
        acc = np.zeros_like(norm_pos)
        for k in range(NUM_KP):
            v = np.diff(norm_pos[:, k, :], axis=0); vel[1:, k, :] = v
            a = np.diff(vel[:, k, :],      axis=0); acc[1:, k, :] = a
        va, aa = np.abs(vel), np.abs(acc)
        vs  = float(np.percentile(va, 95)) if va.max() > 1e-8 else 1.0
        as_ = float(np.percentile(aa, 95)) if aa.max() > 1e-8 else 1.0
        vel = np.clip(vel / max(vs,  1e-8), -3.0, 3.0)
        acc = np.clip(acc / max(as_, 1e-8), -3.0, 3.0)
        angles = np.stack([cls.extract_angles(norm_pos[t]) for t in range(T)]) / 180.0
        return np.concatenate(
            [norm_pos.reshape(T,-1), vel.reshape(T,-1), acc.reshape(T,-1), angles, conf_arr],
            axis=1,
        ).astype(np.float32)



# =============================================================================
# PERCEPTION — ConfidenceTracker  (augmented from infer-vid-6w)
#
# Tracks a continuous confidence value [0..100] per track.
# Drives three observation modes:
#   live   — tracking actively, YOLO skeleton fresh
#   locked — high-confidence freeze, skeleton locked during brief occlusion
#   walled — full confidence, skeleton held from last lock (occlusion tolerant)
#
# Augmentation contract (risk-proofing):
#   RISK 1 — no semantic fields. Mode strings ("live","locked","walled","glitch",
#             "occluded") describe OBSERVATION STATE only, not operational meaning.
#   RISK 4 — ConfidenceTracker is owned by PersonTracker._tracks — private.
#             Its output dict is consumed in-thread; never written to Blackboard.
#             The Blackboard never sees raw confidence floats or mode strings.
#   RISK 2 — ConfidenceTracker does not produce ActionDistribution.
#             It only manages skeleton quality and observation continuity.
# =============================================================================

def _conf_rate_pct_per_sec(rate_setting: float) -> float:
    """
    Maps rate_setting [1..3] to percent-per-second gain.
    Anchors: 1.0 → 33.3%/s, 2.0 → 10%/s, 3.0 → 3.3%/s.
    Linearly interpolated between anchors.
    """
    anchors = {1.0: 100.0 / 3.0, 2.0: 100.0 / 10.0, 3.0: 100.0 / 30.0}
    if rate_setting <= 1.0: return anchors[1.0]
    if rate_setting >= 3.0: return anchors[3.0]
    if rate_setting <= 2.0:
        t = rate_setting - 1.0
        return anchors[1.0] + t * (anchors[2.0] - anchors[1.0])
    t = rate_setting - 2.0
    return anchors[2.0] + t * (anchors[3.0] - anchors[2.0])


class ConfidenceTracker:
    """
    Per-track observation-quality tracker.

    Maintains:
      - A Kalman filter per keypoint (position smoothing + glitch detection)
      - A temporal skeleton buffer for locked/walled mode output
      - A confidence score [0..100] that accumulates on detection and drains on motion
      - An entropy penalty that reduces gain when action distribution is uncertain

    Output modes (structural, not semantic):
      "live"     — YOLO detection active, confidence < 80
      "locked"   — confidence 80..100, skeleton frozen during motion
      "walled"   — confidence == 100, skeleton held through occlusion
      "glitch"   — Kalman/jump detected, previous frame output returned
      "occluded" — no detection, confidence < 100 (track still alive)

    Risk contract:
      - Never produces ActionDistribution or CandidateState
      - Never writes to Blackboard
      - Never modifies PersonTracker or ObservationBuilder internal state
      - set_entropy_penalty() receives normalised entropy [0..1], not a label
    """

    SMOOTH_N = 8
    _KF_Q    = 1e-4
    _KF_R    = 5e-3

    # Awareness defaults — used when no override provided
    _AW_DEFAULTS: dict = {
        "vel_stationary_px":   0.5,
        "vel_drain_rate":      0.8,
        "vel_drain_max":       8.0,
        "dist_near_area":      0.04,
        "dist_far_area":       0.002,
        "dist_far_gain_scale": 0.35,
        "entropy_full":        0.95,
        "entropy_none":        0.35,
    }

    def __init__(
        self,
        src_fps:      float,
        src_w:        int,
        src_h:        int,
        rate_setting: float = 2.0,
        awareness:    dict | None = None,
    ) -> None:
        self.fps     = max(src_fps, 1.0)
        self.src_w   = src_w
        self.src_h   = src_h
        self.rate_pps = _conf_rate_pct_per_sec(rate_setting)

        self._aw: dict = dict(self._AW_DEFAULTS)
        if awareness:
            self._aw.update(awareness)

        self.confidence: float = 0.0

        self._kpt_history:  collections.deque = collections.deque(maxlen=self.SMOOTH_N)
        self._conf_history: collections.deque = collections.deque(maxlen=self.SMOOTH_N)

        self._locked_kpts:  np.ndarray | None = None
        self._locked_conf:  np.ndarray | None = None
        self._lock_frames_remaining: int = 0

        self._prev_kpts:  np.ndarray | None = None
        self._wall_kpts:  np.ndarray | None = None
        self._wall_conf:  np.ndarray | None = None

        self._person_px_width:  float = src_w / 10.0
        self._bbox_area_norm:   float = 0.01

        # Per-keypoint Kalman filter state
        self._kf_x    = np.zeros((NUM_KP, 4), dtype=np.float64)
        self._kf_P    = np.tile(np.eye(4, dtype=np.float64) * 0.1, (NUM_KP, 1, 1))
        self._kf_init = False
        self._kf_F    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self._kf_H    = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self._kf_Q    = np.eye(4, dtype=np.float64) * self._KF_Q
        self._kf_R    = np.eye(2, dtype=np.float64) * self._KF_R

        self._entropy_penalty: float = 0.0

    # ── Kalman ────────────────────────────────────────────────────────────────

    def _kalman_update(
        self,
        kpts_norm: np.ndarray,
        kpt_conf:  np.ndarray,
        kpt_thresh: float,
    ) -> np.ndarray:
        F, H, Q, R = self._kf_F, self._kf_H, self._kf_Q, self._kf_R
        smoothed = np.zeros((NUM_KP, 2), dtype=np.float32)
        for k in range(NUM_KP):
            if not self._kf_init:
                self._kf_x[k] = [kpts_norm[k, 0], kpts_norm[k, 1], 0.0, 0.0]
            x, P = self._kf_x[k], self._kf_P[k]
            xp   = F @ x; Pp = F @ P @ F.T + Q
            if kpt_conf[k] > kpt_thresh:
                z = kpts_norm[k].astype(np.float64)
                y = z - H @ xp; S = H @ Pp @ H.T + R
                K = Pp @ H.T @ np.linalg.inv(S)
                xu = xp + K @ y; Pu = (np.eye(4) - K @ H) @ Pp
            else:
                xu = xp; Pu = Pp
            self._kf_x[k] = xu; self._kf_P[k] = Pu
            smoothed[k] = xu[:2].astype(np.float32)
        self._kf_init = True
        return smoothed

    def _kalman_prediction_error(
        self,
        kpts_norm:  np.ndarray,
        kpt_conf:   np.ndarray,
        kpt_thresh: float,
    ) -> float:
        if not self._kf_init:
            return 0.0
        errors = []
        for k in range(NUM_KP):
            if kpt_conf[k] <= kpt_thresh:
                continue
            errors.append(np.linalg.norm(
                (self._kf_x[k, :2] - kpts_norm[k].astype(np.float64))
                * [self.src_w, self.src_h]
            ))
        return float(np.mean(errors)) if errors else 0.0

    # ── Gain modifiers ────────────────────────────────────────────────────────

    def _distance_gain_scale(self) -> float:
        a    = self._bbox_area_norm
        near = self._aw["dist_near_area"]
        far  = self._aw["dist_far_area"]
        if a >= near: return 1.0
        if a <= far:  return self._aw["dist_far_gain_scale"]
        t = (a - far) / (near - far + 1e-9)
        return self._aw["dist_far_gain_scale"] + t * (1.0 - self._aw["dist_far_gain_scale"])

    def set_entropy_penalty(self, normalised_entropy: float) -> None:
        """
        Receives normalised entropy [0..1] from ActionDistribution.entropy.
        High entropy → high penalty → slower confidence gain.
        Does not store the distribution — only the scalar penalty.
        This is a STRUCTURAL gate, not a semantic one.
        """
        lo = self._aw["entropy_none"]
        hi = self._aw["entropy_full"]
        if normalised_entropy <= lo:
            self._entropy_penalty = 0.0
        elif normalised_entropy >= hi:
            self._entropy_penalty = 1.0
        else:
            self._entropy_penalty = (normalised_entropy - lo) / (hi - lo + 1e-9)

    # ── Glitch + velocity ─────────────────────────────────────────────────────

    def _glitch_threshold_px(self) -> float:
        return max(self._person_px_width, 8.0)

    def _is_glitch(
        self,
        new_kpts:   np.ndarray,
        new_conf:   np.ndarray,
        kpt_thresh: float,
    ) -> bool:
        if self._prev_kpts is None:
            return False
        thresh = self._glitch_threshold_px()
        valid  = new_conf > kpt_thresh
        if valid.sum() == 0:
            return False
        diffs = np.linalg.norm(
            (new_kpts[valid] - self._prev_kpts[valid])
            * np.array([self.src_w, self.src_h]),
            axis=1,
        )
        if bool(np.any(diffs > thresh)):
            return True
        return self._kalman_prediction_error(new_kpts, new_conf, kpt_thresh) > thresh * 3.0

    def _pixel_velocity(
        self,
        new_kpts:   np.ndarray,
        new_conf:   np.ndarray,
        kpt_thresh: float,
    ) -> float:
        if self._prev_kpts is None:
            return 0.0
        valid = new_conf > kpt_thresh
        if valid.sum() == 0:
            return 0.0
        diffs = np.linalg.norm(
            (new_kpts[valid] - self._prev_kpts[valid])
            * np.array([self.src_w, self.src_h]),
            axis=1,
        )
        return float(diffs.mean())

    def _smoothed_skeleton(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack(list(self._kpt_history)).mean(0),
            np.stack(list(self._conf_history)).mean(0),
        )

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        kpts_norm:  np.ndarray,
        kpt_conf:   np.ndarray,
        bbox_norm:  list[float] | None,
        kpt_thresh: float,
        detected:   bool,
        crowd_scale: float = 1.0,
    ) -> dict:
        """
        Update tracker state for one frame.

        Returns a dict:
            kpts_to_draw : np.ndarray | None  — best available skeleton
            conf_to_draw : np.ndarray | None  — corresponding confidence
            vision_conf  : float              — current confidence [0..100]
            mode         : str                — "live"|"locked"|"walled"|"glitch"|"occluded"

        The returned skeleton is used as input to PoseFeatures and ObservationBuilder.
        It is NEVER written directly to the Blackboard.
        """
        if bbox_norm is not None:
            px_w = (bbox_norm[2] - bbox_norm[0]) * self.src_w
            if px_w > 4:
                self._person_px_width = px_w
            self._bbox_area_norm = max(
                (bbox_norm[2] - bbox_norm[0]) * (bbox_norm[3] - bbox_norm[1]), 1e-6)

        if not detected:
            if self.confidence >= 100.0:
                return {"kpts_to_draw": self._wall_kpts, "conf_to_draw": self._wall_conf,
                        "vision_conf": self.confidence, "mode": "walled"}
            return {"kpts_to_draw": None, "conf_to_draw": None,
                    "vision_conf": self.confidence, "mode": "occluded"}

        kpts_smooth = self._kalman_update(kpts_norm, kpt_conf, kpt_thresh)

        if self._is_glitch(kpts_norm, kpt_conf, kpt_thresh):
            return {
                "kpts_to_draw": self._locked_kpts if self._locked_kpts is not None else kpts_smooth,
                "conf_to_draw": self._locked_conf if self._locked_conf is not None else kpt_conf,
                "vision_conf":  self.confidence,
                "mode":         "glitch",
            }

        pix_vel = self._pixel_velocity(kpts_norm, kpt_conf, kpt_thresh)
        self._prev_kpts = kpts_norm.copy()
        self._kpt_history.append(kpts_smooth.copy())
        self._conf_history.append(kpt_conf.copy())
        self._wall_kpts = kpts_smooth.copy()
        self._wall_conf = kpt_conf.copy()

        base_gain   = self.rate_pps / self.fps
        dist_scale  = self._distance_gain_scale()
        uncert_scale = max(0.0, 1.0 - self._entropy_penalty)
        gain        = base_gain * dist_scale * uncert_scale * crowd_scale

        stationary = pix_vel < self._aw["vel_stationary_px"]
        drain = 0.0 if stationary else min(
            pix_vel / max(self._person_px_width, 1.0)
            * self._aw["vel_drain_rate"],
            self._aw["vel_drain_max"],
        )

        self.confidence = float(np.clip(self.confidence + gain - drain, 0.0, 100.0))

        if self.confidence < 80.0:
            self._locked_kpts = None
            self._locked_conf = None
            self._lock_frames_remaining = 0
            return {"kpts_to_draw": kpts_smooth, "conf_to_draw": kpt_conf,
                    "vision_conf": self.confidence, "mode": "live"}

        if self.confidence < 100.0:
            if self._lock_frames_remaining <= 0 and self._kpt_history:
                self._locked_kpts, self._locked_conf = self._smoothed_skeleton()
            if not stationary:
                norm_vel = pix_vel / max(self._person_px_width, 1.0)
                lock_sec = max(0.1, 1.0 / (1.0 + norm_vel * 5.0))
                self._lock_frames_remaining = max(
                    self._lock_frames_remaining, int(lock_sec * self.fps))
            if self._lock_frames_remaining > 0:
                self._lock_frames_remaining -= 1
                return {"kpts_to_draw": self._locked_kpts, "conf_to_draw": self._locked_conf,
                        "vision_conf": self.confidence, "mode": "locked"}
            if self._kpt_history:
                sk, sc = self._smoothed_skeleton()
                return {"kpts_to_draw": sk, "conf_to_draw": sc,
                        "vision_conf": self.confidence, "mode": "locked"}

        if self._kpt_history:
            sk, sc = self._smoothed_skeleton()
            self._locked_kpts = sk
            self._locked_conf = sc

        return {
            "kpts_to_draw": self._locked_kpts if self._locked_kpts is not None else kpts_smooth,
            "conf_to_draw": self._locked_conf if self._locked_conf is not None else kpt_conf,
            "vision_conf":  self.confidence,
            "mode":         "walled",
        }

    def reset_on_redetect(self) -> None:
        """Call when a walled track is re-detected after occlusion."""
        self.confidence = 0.0
        self._kf_init   = False

class ConfidenceModel:
    """
    Builds ConfidenceProfile with conditional invalidation.
    Never produces a scalar. Preserves all dimensions.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg = cfg

    def build(
        self,
        detection_conf: float,
        kp_conf:        np.ndarray,
        tracking_oks:   float,
    ) -> ConfidenceProfile:
        n_vis     = int((kp_conf > self._cfg.kpt_conf_min).sum())
        pose_conf = float(kp_conf[kp_conf > self._cfg.kpt_conf_min].mean()) if n_vis > 0 else 0.0

        pose_valid     = n_vis >= self._cfg.min_visible_keypoints
        tracking_valid = tracking_oks >= self._cfg.min_tracking_oks
        action_valid   = pose_valid  # action certainty is meaningless without good skeleton

        return ConfidenceProfile(
            detection_conf  = detection_conf,
            pose_conf       = pose_conf,
            tracking_oks    = tracking_oks,
            n_visible_kpts  = n_vis,
            temporal_weight = 1.0,
            pose_valid      = pose_valid,
            tracking_valid  = tracking_valid,
            action_valid    = action_valid,
        )

    def observation_validity(self, profile: ConfidenceProfile, detection_conf: float) -> ObservationValidity:
        detection_ok = detection_conf >= self._cfg.min_detection_conf
        pose_ok      = profile.pose_valid
        tracking_ok  = profile.tracking_valid

        parts = []
        if not detection_ok:
            parts.append(f"detection_conf={detection_conf:.3f} < {self._cfg.min_detection_conf}")
        if not pose_ok:
            parts.append(f"visible_kpts={profile.n_visible_kpts} < {self._cfg.min_visible_keypoints}")
        if not tracking_ok:
            parts.append(f"tracking_oks={profile.tracking_oks:.3f} < {self._cfg.min_tracking_oks}")

        return ObservationValidity(
            detection_ok = detection_ok,
            pose_ok      = pose_ok,
            tracking_ok  = tracking_ok,
            detail       = "; ".join(parts) if parts else "ok",
        )


def _compute_oks(
    kpts_a: np.ndarray, conf_a: np.ndarray,
    kpts_b: np.ndarray, conf_b: np.ndarray,
    bbox_b: list[float], kpt_thresh: float = 0.1,
) -> float:
    bw = bbox_b[2] - bbox_b[0]; bh = bbox_b[3] - bbox_b[1]
    s2 = max(bw * bh, 1e-8)
    visible = (conf_a > kpt_thresh) & (conf_b > kpt_thresh)
    if not visible.any():
        return 0.0
    d2 = np.sum((kpts_a[visible] - kpts_b[visible]) ** 2, axis=1)
    k2 = (2.0 * _OKS_SIGMAS[visible]) ** 2
    return float(np.exp(-d2 / (2.0 * s2 * k2)).mean())


class PersonTracker:
    """
    Continuity estimation only.
    Tracks are continuity hypotheses — not identities.
    Instability is surfaced via identity_stability (mean OKS).
    It does not trigger semantic logic.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg          = cfg
        self._disabled_set = set(cfg.disabled_classes)   # for uninformed() init
        self._next_id      = 0
        self._tracks: dict[int, dict] = {}
        self._oks_hist: dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=60))

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / (ua + 1e-8)

    def _match_score(self, det: dict, track: dict) -> float:
        iou = self._iou(det["bbox"], track["bbox"])
        if track.get("last_kpts") is not None:
            oks = _compute_oks(
                det["kpts"], det["conf"],
                track["last_kpts"], track["last_kptconf"],
                det["bbox"],
            )
        else:
            oks = iou
        return self._cfg.oks_alpha * oks + (1.0 - self._cfg.oks_alpha) * iou

    def update(self, detections: list[dict]) -> tuple[list[dict], set[int]]:
        for t in self._tracks.values():
            t["age"] += 1

        matched: set[int] = set()
        results: list[dict] = []

        for det in detections:
            best_tid, best_score = None, self._cfg.iou_thresh
            for tid, track in self._tracks.items():
                if tid in matched:
                    continue
                s = self._match_score(det, track)
                if s > best_score:
                    best_score, best_tid = s, tid

            is_new = best_tid is None
            if is_new:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = {
                    "bbox":        det["bbox"],
                    "age":         0,
                    "pos_buf":     collections.deque(maxlen=self._cfg.window_size),
                    "conf_buf":    collections.deque(maxlen=self._cfg.window_size),
                    "bbox_buf":    collections.deque(maxlen=self._cfg.window_size),
                    # uninformed() not null() — uniform over ENABLED classes only (FIX)
                    "last_action_dist": ActionDistribution.uninformed(self._disabled_set),
                    "last_kpts":   None,
                    "last_kptconf": None,
                }

            track = self._tracks[best_tid]
            track["bbox"] = det["bbox"]
            track["age"]  = 0
            track["pos_buf"].append(det["kpts"])
            track["conf_buf"].append(det["conf"])
            track["bbox_buf"].append(det["bbox"])

            if track["last_kpts"] is not None:
                self._oks_hist[best_tid].append(
                    _compute_oks(
                        track["last_kpts"], track["last_kptconf"],
                        det["kpts"], det["conf"], det["bbox"],
                    )
                )
            track["last_kpts"]     = det["kpts"].copy()
            track["last_kptconf"]  = det["conf"].copy()
            matched.add(best_tid)

            results.append({
                "track_id":         best_tid,
                "bbox":             det["bbox"],
                "kpts_raw":         det["kpts"],
                "conf":             det["conf"],
                "conf_yolo":        det.get("conf_yolo", 0.0),
                "pos_buf":          track["pos_buf"],
                "conf_buf":         track["conf_buf"],
                "bbox_buf":         track["bbox_buf"],
                "last_action_dist": track["last_action_dist"],
                "mean_oks":         self._mean_oks(best_tid),
                "is_new":           is_new,
            })

        dead = [tid for tid, t in self._tracks.items()
                if tid not in matched and t["age"] > self._cfg.track_max_age]
        for tid in dead:
            del self._tracks[tid]
            self._oks_hist.pop(tid, None)

        return results, set(dead)

    def set_action_distribution(self, track_id: int, dist: ActionDistribution) -> None:
        """Store full distribution — not just label."""
        if track_id in self._tracks:
            self._tracks[track_id]["last_action_dist"] = dist

    def _mean_oks(self, tid: int) -> float:
        h = self._oks_hist.get(tid)
        return float(np.mean(list(h))) if h else 0.0


class ActionModel:
    """
    Behavior classification. Returns ActionDistribution — full probability vector.
    Never returns a label directly. Labels are derived by callers.
    Runs on fixed schedule. No pose-gate heuristic.
    """

    def __init__(self, model_path: str, hidden_size: int, num_layers: int, device: str) -> None:
        self._device = device
        self._model  = self._load(model_path, hidden_size, num_layers, device)

    @staticmethod
    def _load(model_path: str, hidden_size: int, num_layers: int, device: str) -> Any | None:
        """
        Auto-detecting loader — reads architecture entirely from the checkpoint.

        Fixes four wrong hardcoded assumptions in previous version:
          1. unidirectional LSTM  → BiLSTM (checkpoint has _reverse keys)
          2. linear named "fc"    → linear named "head"
          3. hidden_size from cfg → derived from lstm.weight_hh_l0.shape[1]
          4. out[:,-1,:] forward  → out.mean(dim=1) with dropout (matches training)

        Architecture is never assumed — always derived from checkpoint weights:
          hidden_size  = weight_hh_l0.shape[1]
          feat_dim     = weight_ih_l0.shape[1]   (stored on model as .feat_dim)
          num_layers   = count of weight_ih_lN keys without _reverse
          num_classes  = head.weight.shape[0]
          bidirectional = any key contains "_reverse"
        """
        if not model_path or not Path(model_path).exists():
            log.warning("ActionModel: not found at %s — YOLO-only mode", model_path)
            return None
        import torch
        import torch.nn as nn

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt                 else ckpt["model_state_dict"]

        # ── Derive architecture from checkpoint keys ──────────────────────────
        ckpt_feat_dim  = int(state["lstm.weight_ih_l0"].shape[1])
        ckpt_hidden    = int(state["lstm.weight_hh_l0"].shape[1])
        ckpt_layers    = sum(1 for k in state
                             if k.startswith("lstm.weight_ih_l") and "_reverse" not in k)
        ckpt_bidir     = any("_reverse" in k for k in state)
        head_key       = "head" if "head.weight" in state else "fc"
        ckpt_classes   = int(state[f"{head_key}.weight"].shape[0])
        head_in        = ckpt_hidden * (2 if ckpt_bidir else 1)

        if ckpt_feat_dim != FEATURE_DIM:
            log.warning(
                "ActionModel: checkpoint feat_dim=%d, FEATURE_DIM=%d — "
                "using checkpoint value. Sequence builder will be adjusted.",
                ckpt_feat_dim, FEATURE_DIM,
            )

        log.info(
            "ActionModel checkpoint: feat=%d  hidden=%d  layers=%d  "
            "bidirectional=%s  head='%s'  classes=%d",
            ckpt_feat_dim, ckpt_hidden, ckpt_layers, ckpt_bidir, head_key, ckpt_classes,
        )

        class _BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size    = ckpt_feat_dim,
                    hidden_size   = ckpt_hidden,
                    num_layers    = ckpt_layers,
                    batch_first   = True,
                    bidirectional = ckpt_bidir,
                    dropout       = 0.5 if ckpt_layers > 1 else 0.0,
                )
                self.drop = nn.Dropout(0.5)
                self.head = nn.Linear(head_in, ckpt_classes)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm(x)
                return self.head(self.drop(out.mean(dim=1)))

        m = _BiLSTM()
        m.load_state_dict(state)
        m.to(device).eval()
        m.feat_dim   = ckpt_feat_dim    # runtime access for sequence dim-check
        m.num_classes = ckpt_classes
        log.info("ActionModel loaded: %s → %s", Path(model_path).name, device)
        return m

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, seq: np.ndarray, disabled: set[str]) -> ActionDistribution:
        """
        Returns ActionDistribution over ENABLED classes only. Never a label string.

        Sequence dim-check: if seq feature dim != model.feat_dim, pad or trim.
        This handles any residual mismatch between PoseFeatures and the checkpoint.
        Correct path: seq.shape[1] == model.feat_dim == FEATURE_DIM == 125.

        Fallback: uninformed(disabled) — uniform over ENABLED classes only.
        """
        if self._model is None:
            return ActionDistribution.uninformed(disabled)
        import torch
        try:
            feat_dim = getattr(self._model, "feat_dim", FEATURE_DIM)
            T, sd = seq.shape
            if sd != feat_dim:
                log.debug("ActionModel: seq dim %d != model feat_dim %d — adjusting", sd, feat_dim)
                if sd > feat_dim:
                    seq = seq[:, :feat_dim]
                else:
                    seq = np.concatenate(
                        [seq, np.zeros((T, feat_dim - sd), dtype=seq.dtype)], axis=1)
            if not np.isfinite(seq).all():
                seq = np.nan_to_num(seq, nan=0.0, posinf=3.0, neginf=-3.0)
            t = torch.tensor(seq[np.newaxis], dtype=torch.float32, device=self._device)
            with torch.no_grad():
                logits = self._model(t).squeeze(0).cpu().numpy()
            return ActionDistribution.from_logits(logits, disabled)
        except Exception as e:
            log.debug("ActionModel predict error: %s", e)
            return ActionDistribution.uninformed(disabled)


# =============================================================================
# ZONE CONTACT
# Phase 1: centroid mode. Future: bbox overlap ratio via zone_overlap_threshold.
# =============================================================================

def _point_in_polygon(px: float, py: float, pts: tuple) -> bool:
    n, inside, j = len(pts), False, len(pts) - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _zone_contacts(
    bbox:              list[float],
    zone_registry:     dict[str, ZoneDefinition],
    overlap_threshold: float = 0.0,
) -> list[str]:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return [
        zid for zid, z in zone_registry.items()
        if _point_in_polygon(cx, cy, z.polygon)
    ]


# =============================================================================
# OBSERVATION — builder, delta detector, candidate emitter
# Three classes. Three jobs. No fusion.
# =============================================================================

class ObservationBuilder:
    """
    Constructs Observation from raw perception output.
    Sets validity report. Stores full ActionDistribution — not top label.
    Does NOT decide whether the observation matters.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg        = cfg
        self._conf_model = ConfidenceModel(cfg)

    def build(
        self,
        track_id:         int,
        frame_number:     int,
        timestamp_sec:    float,
        ingestion_time:   float,
        bbox:             list[float],
        kpts_raw:         np.ndarray,
        kp_conf:          np.ndarray,
        conf_yolo:        float,
        mean_oks:         float,
        action_dist:      ActionDistribution,
        zone_contacts:    list[str],
        motion_vector:    tuple[float, float],
    ) -> Observation:
        profile  = self._conf_model.build(conf_yolo, kp_conf, mean_oks)
        validity = self._conf_model.observation_validity(profile, conf_yolo)
        latency  = (time.monotonic() - ingestion_time) * 1000.0

        return Observation(
            observation_id         = str(uuid.uuid4()),
            track_id               = track_id,
            frame_number           = frame_number,
            timestamp_sec          = timestamp_sec,
            ingestion_time_sec     = ingestion_time,
            processing_latency_ms  = latency,
            camera_id              = self._cfg.camera_id,
            bbox                   = bbox,
            keypoints              = kpts_raw,
            kp_confidence          = kp_conf,
            action_distribution    = action_dist,   # FULL distribution
            confidence             = profile,
            zone_contacts          = zone_contacts,
            motion_vector          = motion_vector,
            validity               = validity,
        )


class DeltaDetector:
    """
    Detects structural state transitions. Answers: "What changed?"
    Never answers: "Does this matter?" or "Is this suspicious?"

    Carries ContinuityState per track — privately.
    ContinuityState is NOT written to the Blackboard.
    Only TrackSummary (via write_observation) reaches the Blackboard.

    ACTION_SHIFT compares distributions, not labels.
    If top_label changed but top_conf is below threshold → no delta.
    This prevents noisy distribution fluctuations from flooding candidates.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg        = cfg
        self._continuity: dict[int, ContinuityState] = {}

    def detect(self, obs: Observation) -> ObservationDelta | None:
        ts       = obs.timestamp_sec
        tid      = obs.track_id
        cx       = (obs.bbox[0] + obs.bbox[2]) / 2.0
        cy       = (obs.bbox[1] + obs.bbox[3]) / 2.0
        zone_set = frozenset(obs.zone_contacts)

        if tid not in self._continuity:
            self._continuity[tid] = ContinuityState(
                track_id                 = tid,
                first_seen_sec           = ts,
                last_delta_sec           = ts,
                last_centroid            = (cx, cy),
                last_action_label        = obs.action_distribution.top_label,
                last_action_distribution = obs.action_distribution,  # store from first obs
                last_zones               = zone_set,
                identity_stability       = obs.confidence.tracking_oks,
            )
            return self._make(obs, DeltaReason.NEW_TRACK,
                              frozenset(), zone_set,
                              ActionDistribution.absent(),   # no prior exists — not fake uniform
                              obs.action_distribution)

        cont = self._continuity[tid]

        # Update identity stability (rolling)
        cont.identity_stability = (
            0.9 * cont.identity_stability + 0.1 * obs.confidence.tracking_oks
        )

        entered = zone_set - cont.last_zones
        if entered:
            prior = cont.last_zones
            prior_dist = ActionDistribution(dict(obs.action_distribution.class_probs))
            cont.last_zones               = zone_set
            cont.last_centroid            = (cx, cy)
            cont.last_action_label        = obs.action_distribution.top_label
            cont.last_action_distribution = obs.action_distribution
            cont.last_delta_sec           = ts
            return self._make(obs, DeltaReason.ZONE_ENTRY,
                              prior, zone_set, prior_dist, obs.action_distribution)

        exited = cont.last_zones - zone_set
        if exited:
            prior = cont.last_zones
            prior_dist = ActionDistribution(dict(obs.action_distribution.class_probs))
            cont.last_zones               = zone_set
            cont.last_centroid            = (cx, cy)
            cont.last_action_label        = obs.action_distribution.top_label
            cont.last_action_distribution = obs.action_distribution
            cont.last_delta_sec           = ts
            return self._make(obs, DeltaReason.ZONE_EXIT,
                              prior, zone_set, prior_dist, obs.action_distribution)

        # ACTION_SHIFT: compare top labels, gated by top_conf and action_valid
        new_top = obs.action_distribution.top_label
        if (new_top != cont.last_action_label
                and obs.action_distribution.top_conf >= self._cfg.action_shift_min_conf
                and obs.confidence.action_valid):
            # prior_dist is the stored distribution from the previous observation —
            # not null(), not uniform. The actual prior is now available because
            # ContinuityState.last_action_distribution is updated every frame (below).
            prior_dist             = cont.last_action_distribution
            cont.last_action_label = new_top
            cont.last_action_distribution = obs.action_distribution
            cont.last_centroid    = (cx, cy)
            cont.last_delta_sec   = ts
            return self._make(obs, DeltaReason.ACTION_SHIFT,
                              cont.last_zones, zone_set,
                              prior_dist, obs.action_distribution)

        # Always keep last_action_distribution current (used as prior on next ACTION_SHIFT)
        cont.last_action_distribution = obs.action_distribution

        dist = np.hypot(cx - cont.last_centroid[0], cy - cont.last_centroid[1])
        if dist >= self._cfg.spatial_delta_threshold:
            cont.last_centroid  = (cx, cy)
            cont.last_delta_sec = ts
            return self._make(obs, DeltaReason.SPATIAL_DELTA,
                              cont.last_zones, zone_set,
                              obs.action_distribution, obs.action_distribution)

        elapsed = ts - cont.last_delta_sec
        if elapsed >= self._cfg.temporal_pulse_sec:
            cont.last_delta_sec = ts
            cont.last_centroid  = (cx, cy)
            return self._make(obs, DeltaReason.TEMPORAL_PULSE,
                              cont.last_zones, zone_set,
                              obs.action_distribution, obs.action_distribution)

        return None

    def get_stability(self, track_id: int) -> float:
        cont = self._continuity.get(track_id)
        return cont.identity_stability if cont else 1.0

    def get_last_centroid(self, track_id: int) -> tuple[float, float] | None:
        """Public accessor for last known centroid — avoids external _continuity access."""
        cont = self._continuity.get(track_id)
        if cont is None or cont.last_centroid == (0.0, 0.0):
            return None
        return cont.last_centroid

    def expire_track(self, track_id: int) -> None:
        self._continuity.pop(track_id, None)

    @staticmethod
    def _make(
        obs:          Observation,
        reason:       DeltaReason,
        prior_zones:  frozenset,
        current_zones: frozenset,
        prior_dist:   ActionDistribution,
        current_dist: ActionDistribution,
    ) -> ObservationDelta:
        return ObservationDelta(
            delta_id                    = str(uuid.uuid4()),
            track_id                    = obs.track_id,
            reason                      = reason,
            observation                 = obs,
            prior_zones                 = prior_zones,
            current_zones               = current_zones,
            prior_action_distribution   = prior_dist,
            current_action_distribution = current_dist,
        )


class CandidateEmitter:
    """
    Promotes ObservationDelta to Candidate when composite score is sufficient.

    Single gate: composite score >= min_composite_score.
    No behavioral heuristics. No semantic classification.
    Preserves full ActionDistribution and ConfidenceProfile in the candidate.
    """

    def __init__(self, cfg: ObservationConfig) -> None:
        self._cfg = cfg

    def evaluate(
        self,
        delta:    ObservationDelta,
        stability: float,
    ) -> Candidate | None:
        obs   = delta.observation
        score = obs.confidence.composite(
            self._cfg.w_detection,
            self._cfg.w_pose,
            self._cfg.w_action,
            self._cfg.w_tracking,
            obs.action_distribution.top_conf,
        )
        if score < self._cfg.min_composite_score:
            return None

        return Candidate(
            candidate_id        = str(uuid.uuid4()),
            track_id            = obs.track_id,
            camera_id           = obs.camera_id,
            frame_number        = obs.frame_number,
            timestamp_sec       = obs.timestamp_sec,
            delta_reason        = delta.reason,
            bbox                = obs.bbox,
            keypoints           = obs.keypoints.tolist(),
            kp_confidence       = obs.kp_confidence.tolist(),
            zone_contacts       = obs.zone_contacts,
            action_distribution = obs.action_distribution,   # FULL — not compressed
            confidence          = obs.confidence,            # FULL — not compressed
            composite_score     = score,
            motion_vector       = list(obs.motion_vector),
            identity_stability  = stability,
            state               = CandidateState.EMERGING,
        )


# =============================================================================
# RUNTIME — threads, orchestration
# =============================================================================

def _reader_thread(
    cap:         cv2.VideoCapture,
    sampler:     FrameSampler,
    start_frame: int,
    end_frame:   int,
    raw_q:       queue.Queue,
    max_q_size:  int,
    counters:    dict,
) -> None:
    frame_idx = start_frame
    total_frames = max(end_frame - start_frame, 1)
    pbar = _tqdm(
        total=total_frames,
        desc="Phase 1 — Reading",
        unit="frames",
        dynamic_ncols=True,
        colour="cyan",
        leave=True,
    ) if _TQDM_OK else None
    try:
        while True:
            ingest_time = time.monotonic()
            ret, frame  = cap.read()
            if not ret or frame_idx >= end_frame:
                break
            frame_idx += 1
            counters["total"] = counters.get("total", 0) + 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "skip": counters.get("skipped", 0),
                    "drop": counters.get("dropped", 0),
                    "q": raw_q.qsize(),
                }, refresh=False)
            if not sampler.should_process(frame_idx):
                counters["skipped"] = counters.get("skipped", 0) + 1
                continue
            if raw_q.qsize() >= max_q_size:
                counters["dropped"] = counters.get("dropped", 0) + 1
                continue
            raw_q.put((frame_idx, frame, ingest_time))
    finally:
        if pbar is not None:
            pbar.close()
        raw_q.put(_SENTINEL)
        log.info("Reader done. total=%d skipped=%d dropped=%d",
                 counters.get("total",0), counters.get("skipped",0), counters.get("dropped",0))


def _detector_thread(
    raw_q:   queue.Queue,
    ann_q:   queue.Queue,
    yolo:    Any,
    tracker: PersonTracker,
    cfg:     ObservationConfig,
    src_w:   int,
    src_h:   int,
    device:  str,
) -> None:
    yolo_w = max(32, (int(src_w * cfg.yolo_input_sz / max(src_w, src_h)) // 32) * 32) \
             if cfg.yolo_input_sz / max(src_w, src_h) < 1.0 else src_w
    yolo_h = max(32, (int(src_h * cfg.yolo_input_sz / max(src_w, src_h)) // 32) * 32) \
             if cfg.yolo_input_sz / max(src_w, src_h) < 1.0 else src_h

    try:
        while True:
            item = raw_q.get()
            if item is _SENTINEL:
                ann_q.put(_SENTINEL)
                break
            frame_idx, frame, ingest_time = item

            resized = cv2.resize(frame, (yolo_w, yolo_h)) \
                      if (yolo_w, yolo_h) != (src_w, src_h) else frame
            results = yolo(resized, conf=cfg.conf_thresh, iou=cfg.iou_thresh,
                           classes=[0], verbose=False, device=device)

            detections: list[dict] = []
            for r in results:
                if r.boxes is None:
                    continue
                boxes     = r.boxes.xyxyn.cpu().numpy()
                yolo_c    = r.boxes.conf.cpu().numpy()
                kpts_all  = r.keypoints.xyn.cpu().numpy()  if r.keypoints is not None else None
                kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else None
                for i in range(min(len(boxes), cfg.max_persons)):
                    dc = float(yolo_c[i])
                    if dc < cfg.conf_thresh:
                        continue
                    kpts = (kpts_all[i].astype(np.float32)  if kpts_all  is not None
                            else np.zeros((NUM_KP,2), dtype=np.float32))
                    kp_c = (kpts_conf[i].astype(np.float32) if kpts_conf is not None
                            else np.zeros(NUM_KP, dtype=np.float32))
                    detections.append({"bbox": boxes[i].tolist(), "conf_yolo": dc,
                                       "kpts": kpts, "conf": kp_c})

            tracked, expired_ids = tracker.update(detections)
            ann_q.put((frame_idx, tracked, expired_ids, ingest_time))

    except Exception as e:
        log.exception("Detector thread error: %s", e)
        ann_q.put(_SENTINEL)


def _observer_thread(
    ann_q:        queue.Queue,
    blackboard:   Blackboard,
    action_model: ActionModel,
    cfg:          ObservationConfig,
    contract:     TemporalContract,
    src_fps:      float,
    device:       str,
    tracker:      PersonTracker,
) -> None:
    obs_builder  = ObservationBuilder(cfg)
    delta_detect = DeltaDetector(cfg)
    emitter      = CandidateEmitter(cfg)
    disabled_set = set(cfg.disabled_classes)
    infer_ctr:   dict[int, int] = {}

    total_obs = total_deltas = total_emitted = 0
    total_latency_ms = 0.0
    t0 = time.perf_counter()

    pbar = _tqdm(
        desc="Phase 1 — Observing",
        unit="obs",
        dynamic_ncols=True,
        colour="green",
        leave=True,
    ) if _TQDM_OK else None

    try:
        while True:
            item = ann_q.get()
            if item is _SENTINEL:
                break

            frame_idx, tracked, expired_ids, ingest_time = item
            ts = frame_idx / max(src_fps, 1.0)

            for tid in expired_ids:
                delta_detect.expire_track(tid)
                blackboard.expire_track(tid)

            # LSTM on fixed schedule — allow configurable shorter-sequence
            # inference so Phase 1 can populate real action distributions
            # for short tracks (heatmap expects counts rather than uniform priors).
            if action_model.available:
                for person in tracked:
                    tid     = person["track_id"]
                    pos_buf = person["pos_buf"]
                    infer_ctr[tid] = infer_ctr.get(tid, 0) + 1
                    # require at least `min_infer_frames` samples before predicting
                    if (len(pos_buf) >= cfg.min_infer_frames
                            and infer_ctr[tid] % contract.infer_every == 0):
                        try:
                            seq  = PoseFeatures.build_sequence(
                                list(pos_buf),
                                list(person["conf_buf"]),
                                list(person["bbox_buf"]) or None,
                            )
                            dist = action_model.predict(seq, disabled_set)
                            tracker.set_action_distribution(tid, dist)
                        except Exception as e:
                            log.debug("ActionModel skip track %d: %s", tid, e)
                    else:
                        # Helpful debug log to understand why predictions are missing
                        if len(pos_buf) < cfg.min_infer_frames:
                            log.debug(
                                "ActionModel: skipping track %d — need %d frames, have %d",
                                tid, cfg.min_infer_frames, len(pos_buf)
                            )
            else:
                log.debug("ActionModel not available — Phase 1 will use uninformed priors")

            for person in tracked:
                kpts    = person["kpts_raw"]
                kp_conf = person["conf"]
                if kpts is None or kp_conf is None:
                    continue

                bbox       = person["bbox"]
                tid        = person["track_id"]
                conf_yolo  = person.get("conf_yolo", 0.0)
                mean_oks   = person["mean_oks"]
                action_dist = person["last_action_dist"]   # ActionDistribution

                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                # Motion vector from DeltaDetector public API
                last_centroid = delta_detect.get_last_centroid(tid)
                if last_centroid is not None:
                    dx = cx - last_centroid[0]
                    dy = cy - last_centroid[1]
                else:
                    dx, dy = 0.0, 0.0

                contacts = _zone_contacts(bbox, blackboard.zone_registry,
                                          cfg.zone_overlap_threshold)

                obs = obs_builder.build(
                    track_id       = tid,
                    frame_number   = frame_idx,
                    timestamp_sec  = ts,
                    ingestion_time = ingest_time,
                    bbox           = bbox,
                    kpts_raw       = kpts,
                    kp_conf        = kp_conf,
                    conf_yolo      = conf_yolo,
                    mean_oks       = mean_oks,
                    action_dist    = action_dist,
                    zone_contacts  = contacts,
                    motion_vector  = (dx, dy),
                )

                total_latency_ms += obs.processing_latency_ms

                # All detection_ok + pose_ok observations go into stream
                if obs.validity.is_valid:
                    blackboard.write_observation(obs)
                    total_obs += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            "deltas": total_deltas,
                            "cands": total_emitted,
                            "tracks": len(blackboard.track_summaries),
                        }, refresh=False)

                delta = delta_detect.detect(obs)
                if delta is not None:
                    total_deltas += 1
                    stability = delta_detect.get_stability(tid)
                    candidate = emitter.evaluate(delta, stability)
                    if candidate is not None:
                        blackboard.write_candidate(candidate)
                        total_emitted += 1

            if frame_idx % 500 == 0:
                elapsed = time.perf_counter() - t0
                avg_lat = total_latency_ms / max(total_obs, 1)
                budget  = contract.wall_clock_budget_ms
                log.info(
                    "Observer: frame=%d ts=%.1fs obs=%d deltas=%d candidates=%d "
                    "fps=%.1f avg_latency=%.1fms budget=%.0fms",
                    frame_idx, ts, total_obs, total_deltas, total_emitted,
                    frame_idx / max(elapsed, 1e-6), avg_lat, budget,
                )
                if avg_lat > budget:
                    log.warning(
                        "Temporal budget exceeded: avg=%.1fms > budget=%.0fms — "
                        "consider increasing frame_skip (changes observable universe)",
                        avg_lat, budget,
                    )

    except Exception as e:
        log.exception("Observer thread error: %s", e)

    if pbar is not None:
        pbar.close()

    elapsed = time.perf_counter() - t0
    avg_lat = total_latency_ms / max(total_obs, 1)
    log.info(
        "Observer done. obs=%d deltas=%d candidates=%d "
        "emission_rate=%.3f avg_latency=%.1fms elapsed=%.1fs",
        total_obs, total_deltas, total_emitted,
        total_emitted / max(total_deltas, 1), avg_lat, elapsed,
    )


# =============================================================================
# ObservationLeg — public entry point
# =============================================================================

class ObservationLeg:
    """
    Phase 1 entry point.

        bb       = Blackboard(cfg.build_temporal_contract())
        cfg      = ObservationConfig(...)
        leg      = ObservationLeg(cfg, bb)
        leg.run()

        # Primary truth: bb.observation_stream
        # Candidates: bb.drain_emerging()
        # Track state: bb.track_summaries (continuity only, not identity)
    """

    def __init__(self, cfg: ObservationConfig, blackboard: Blackboard) -> None:
        self._cfg      = cfg
        self._bb       = blackboard
        self._contract = cfg.build_temporal_contract()
        self._counters: dict = {}

    def run(self) -> None:
        self.start()
        self.join()

    def start(self) -> None:
        cfg    = self._cfg
        device = _resolve_device(cfg.device)

        yolo = _load_yolo_model(cfg.yolo_model)

        if cfg.vision_mode == 3:
            raise ValueError("Vision mode 3 deprecated. Use 1 or 2.")

        action_model = ActionModel(cfg.model_path, cfg.hidden_size, cfg.num_layers, device)

        cap = cv2.VideoCapture(cfg.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {cfg.input_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(cfg.start_sec * src_fps)
        end_frame   = int(cfg.end_sec   * src_fps) if cfg.end_sec > 0 else total
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        temporal_res = self._contract.temporal_resolution_sec(src_fps)
        log.info(
            "Phase 1 start: %s (%dx%d @ %.1ffps, %d frames) "
            "mode=%d frame_skip=%d temporal_resolution=%.3fs",
            cfg.input_path, src_w, src_h, src_fps, total,
            cfg.vision_mode, cfg.frame_skip, temporal_res,
        )

        zones = [
            ZoneDefinition(
                zone_id  = z["zone_id"],
                polygon  = z["polygon"],
                zone_type = z.get("zone_type", "OBSERVATION"),
                priority  = z.get("priority", 1),
                metadata  = z.get("metadata", {}),
            )
            for z in cfg.zones
        ]
        self._bb.register_zones(zones)
        log.info("Zones registered: %d", len(zones))

        sampler = FrameSampler(self._contract)
        tracker = PersonTracker(cfg)

        raw_q = queue.Queue(maxsize=cfg.raw_q_size)
        ann_q = queue.Queue(maxsize=cfg.ann_q_size)
        self._cap = cap

        self._threads = [
            threading.Thread(
                target=_reader_thread,
                args=(cap, sampler, start_frame, end_frame,
                      raw_q, cfg.raw_q_size, self._counters),
                daemon=True, name="Ph1-Reader",
            ),
            threading.Thread(
                target=_detector_thread,
                args=(raw_q, ann_q, yolo, tracker, cfg, src_w, src_h, device),
                daemon=True, name="Ph1-Detector",
            ),
            threading.Thread(
                target=_observer_thread,
                args=(ann_q, self._bb, action_model, cfg, self._contract,
                      src_fps, device, tracker),
                daemon=True, name="Ph1-Observer",
            ),
        ]
        for t in self._threads:
            t.start()

    def join(self) -> None:
        for t in self._threads:
            t.join()
        if hasattr(self, "_cap"):
            self._cap.release()
        log.info(
            "Phase 1 complete. candidates=%d observations=%d tracks=%d",
            self._bb.count_candidates(),
            self._bb.observation_stream.count(),
            len(self._bb.track_summaries),
        )

    @property
    def counters(self) -> dict:
        return dict(self._counters)


# =============================================================================
# Standalone test
# =============================================================================

def print_candidate_summary(candidates: list, limit: int = 5) -> None:
    """Print a summary of candidates, skipping high-entropy ambiguous samples."""
    print(f"  Candidates: {len(candidates)}")
    if not candidates:
        return

    meaningful = [
        c for c in candidates
        if c.action_distribution.entropy <= 1.0
    ]
    if meaningful:
        sample = meaningful[:limit]
        skipped = len(candidates) - len(meaningful)
        print(f"  Top-3 action distributions (sample — first {min(limit, len(sample))} meaningful candidates):")
    else:
        sample = candidates[:limit]
        skipped = 0
        print(f"  Top-3 action distributions (sample — first {min(limit, len(sample))} candidates):")

    for c in sample:
        top3 = c.action_distribution.top_n(3)
        ent  = round(c.action_distribution.entropy, 3)
        stab = round(c.identity_stability, 3)
        print(f"    track={c.track_id}  top3={top3}  entropy={ent}  stability={stab}")
    if skipped:
        print(f"    ({skipped} high-entropy candidates omitted from sample)")


if __name__ == "__main__":
    import argparse
    import json
    from collections import Counter

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 1 — Observation leg v3")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--model",  default="")
    parser.add_argument("--mode",   type=int, default=2, choices=[1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zones",  default=None, help="JSON file with zone definitions")
    parser.add_argument("--skip",   type=int, default=1)
    parser.add_argument("--out",    default="phase1_candidates.json")
    parser.add_argument("--end",    type=float, default=0.0)
    args = parser.parse_args()

    zones_raw = []
    if args.zones:
        with open(args.zones, encoding="utf-8") as f:
            zones_raw = json.load(f)

    model_path = resolve_action_model_path(args.mode, args.model)

    cfg = ObservationConfig(
        input_path  = args.input,
        model_path  = model_path,
        vision_mode = args.mode,
        device      = args.device,
        zones       = zones_raw,
        end_sec     = args.end,
        frame_skip  = args.skip,
    )

    bb  = Blackboard(cfg.build_temporal_contract())
    leg = ObservationLeg(cfg, bb)

    print(f"\n  Phase 1 Observation — v3")
    print(f"  Input      : {args.input}")
    print(f"  Model      : {model_path}")
    print(f"  Mode       : {args.mode}  Zones: {len(zones_raw)}  Skip: {args.skip}")
    print()

    leg.run()

    candidates = bb._candidate_log

    out_data = {
        "meta": {
            "input":          args.input,
            "model_path":     model_path,
            "vision_mode":    args.mode,
            "frame_skip":     args.skip,
            "n_candidates":   bb.count_candidates(),
            "n_observations": bb.observation_stream.count(),
            "n_tracks":       len(bb.track_summaries),
        },
        "candidates": [c.to_dict() for c in
                       sorted(candidates, key=lambda x: x.timestamp_sec)],
    }

    # Add aggregate action counts and mean probability vector to help
    # downstream tools (heatmap) that expect counts rather than priors.
    try:
        from collections import Counter
        action_counts = Counter(c.action_distribution.top_label for c in candidates)
        out_data["meta"]["action_counts"] = dict(action_counts)

        # average class probabilities across all candidates
        if candidates:
            accum = {a: 0.0 for a in ACTION_CLASSES}
            for c in candidates:
                for k, v in c.action_distribution.class_probs.items():
                    accum[k] = accum.get(k, 0.0) + float(v)
            n = float(len(candidates))
            out_data["meta"]["action_probs_mean"] = {k: round(v / n, 4) for k, v in accum.items()}
        else:
            out_data["meta"]["action_probs_mean"] = {}
    except Exception:
        # be conservative: do not fail the writer on aggregation errors
        out_data["meta"]["action_counts"] = {}
        out_data["meta"]["action_probs_mean"] = {}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"  Observations    : {bb.observation_stream.count()}")
    print(f"  Candidates      : {bb.count_candidates()}")
    print(f"  Tracks          : {len(bb.track_summaries)}")
    print(f"  Output          : {args.out}")
    print()

    reasons = Counter(c.delta_reason.value for c in candidates)
    print("  Delta reasons:")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:<25} {cnt}")
    print()

    if candidates:
        print_candidate_summary(candidates, limit=5)
        print()
