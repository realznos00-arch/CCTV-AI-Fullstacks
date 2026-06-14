"""
infer-vid-w7.py — CCTV Pose Pipeline v7 (Prototype)
=====================================================
Architecture merge of infer-vid-6w.py and phase1_observation.py.

WHY THIS EXISTS
---------------
  infer-vid-6w.py  — model + analytics engine (BiLSTM, ConfidenceTracker,
                     RoleTracker, FlagLog, SummaryLog, fence overlay, HUD).
                     LSTM gated by _event_triggered() → almost never fires on
                     normal standing/walking footage. Draws and writes video.

  phase1_observation.py — pulse/delta/emerging blackboard architecture.
                          DeltaDetector fires candidates on structural state
                          transitions. No event-trigger gate. Epistemically
                          neutral. Never draws video.

WHAT v7 DOES
------------
  Takes the MODEL and ANALYTICS engine from 6w verbatim.
  Replaces the 6w writer-thread LSTM gate (_event_triggered) with the
  phase1 pulse/delta/emerging loop:

    DeltaDetector   — emits ObservationDelta on NEW_TRACK, ZONE_ENTRY/EXIT,
                       ACTION_SHIFT, SPATIAL_DELTA, TEMPORAL_PULSE.
    CandidateEmitter — promotes delta → Candidate when composite score passes.
    EmergingQueue    — lightweight replacement for phase1 Blackboard;
                       stores Candidates for downstream consumption and
                       for the on-screen EMERGING overlay.

  LSTM still runs on a fixed cadence (infer_every), but now fires whenever
  the buffer is full (window_size frames), not when a geometry heuristic
  passes. lstm_ran_for set gates action_valid correctly.

  Everything else from 6w is preserved unchanged:
    ConfidenceTracker, RoleTracker, FlagLog, SummaryLog,
    draw_person, draw_fences, draw_hud, pure_vision_flags,
    _build_sequence, predict_sequence, load_lstm_model.

PROTOTYPE SCOPE
---------------
  - No full Blackboard (just EmergingQueue, a plain list with a lock).
  - No ObservationStream; observations are not archived.
  - No zone-registered DeltaDetector zones (zone contacts always []).
  - Candidate serialisation: emitted to JSON sidecar as .candidates.json.
  - Video output unchanged from 6w (ffmpeg pipe or OpenCV fallback).
  - EMERGING overlay: small green dot + "E" label above bbox when a
    candidate is active for that track.

WHAT IS NOT HERE (future work)
--------------------------------
  - Phase 2 filter (cooldown, score threshold, suppression).
  - Phase 3 event log.
  - Phase 4 SQLite storage.
  - Full Blackboard contracts.

Install:
    pip install ultralytics torch torchvision opencv-python numpy scipy

Usage:
    python infer-vid-w7.py --input video.mp4 --mode 2
    python infer-vid-w7.py --input video.mp4 --mode 2 --no-prompt
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Resolve model paths relative to this script's directory (project root).
# Replaces the old .\\lstm_model\\... relative paths that broke when
# the working directory was not the project root.
_ROOT = Path(__file__).resolve().parent
from typing import Any

import subprocess
import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# seek_utils: CV precision seek + seek_map scaling helpers.
# Graceful fallback — script still runs without the file; seek falls back
# to the inline cap.set() path and seek_map scaling is a no-op.
try:
    from seek_utils import seek_to_frame as _seek_to_frame, scale_seek_map as _scale_seek_map
    _SEEK_UTILS_OK = True
    log_msg = "seek_utils loaded — CV precision seek + seek_map scaling active"
except ImportError:
    _seek_to_frame   = None   # type: ignore[assignment]
    _scale_seek_map  = None   # type: ignore[assignment]
    _SEEK_UTILS_OK   = False
    log_msg = "seek_utils not found — using inline seek fallback (precision may vary)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
log.debug(log_msg)  # seek_utils import result — visible at DEBUG level

_SENTINEL = object()


# =============================================================================
# CONSTANTS — unchanged from 6w
# =============================================================================

ACTION_CLASSES = [
    "fall", "grab", "gun", "hit", "kick",
    "lying_down", "run", "sit", "sneak",
    "stand", "struggle", "throw", "walk",
]
NUM_CLASSES   = len(ACTION_CLASSES)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KP = 17

SKELETON_EDGES = [
    ("left_shoulder","right_shoulder"), ("left_shoulder","left_elbow"),
    ("left_elbow","left_wrist"), ("right_shoulder","right_elbow"),
    ("right_elbow","right_wrist"), ("left_shoulder","left_hip"),
    ("right_shoulder","right_hip"), ("left_hip","right_hip"),
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ("nose","left_shoulder"), ("nose","right_shoulder"),
]

ACTION_COLORS: dict[str, tuple] = {
    "fall":(0,50,255), "grab":(0,165,255), "gun":(0,0,200),
    "hit":(0,0,255), "kick":(30,105,210), "lying_down":(128,0,128),
    "run":(0,200,50), "sit":(255,200,0), "sneak":(200,130,0),
    "stand":(0,255,200), "struggle":(0,0,180), "throw":(0,128,255),
    "walk":(180,220,80),
}

ACTION_DISPLAY: dict[str, str] = {
    "fall":"Falling", "grab":"Grabbing", "gun":"Gun",
    "hit":"Hitting", "kick":"Kicking", "lying_down":"Lying Down",
    "run":"Running", "sit":"Sitting", "sneak":"Sneaking",
    "stand":"Idle", "struggle":"Struggling", "throw":"Throwing", "walk":"Walking",
}

_OKS_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
], dtype=np.float64)

_N_ANGLES   = 12
FEATURE_DIM = NUM_KP * 2 + NUM_KP * 2 + NUM_KP * 2 + _N_ANGLES + NUM_KP  # 131

ROLE_COLOR: dict[str, tuple] = {
    "employee": (0, 220, 180),
    "customer": (30, 130, 255),
    "unknown":  (120, 120, 120),
}
ROLE_LABEL: dict[str, str] = {
    "employee": "EMP",
    "customer": "CUST",
    "unknown":  "?",
}

_LOCO_ACTIONS   = {"run", "walk", "sneak"}
_STATIC_ACTIONS = {"sit", "stand", "lying_down"}

_FLAG_COLOR = {
    "FALLING":(0,50,255),"LYING":(128,0,128),"SITTING":(255,200,0),
    "CROUCHING":(200,130,0),"RUNNING":(0,200,50),"WALKING":(180,220,80),
    "ARMS UP":(0,165,255),"ARMS WIDE":(0,128,255),"REACHING":(30,105,210),
    "LEANING":(0,255,200),"IDLE":(160,160,160),
}

_STATIC_FLAGS  = frozenset({"SITTING","IDLE","LYING","CROUCHING","LEANING","sit","stand","lying_down"})
_DYNAMIC_FLAGS = frozenset({"RUNNING","WALKING","FALLING","run","walk","fall","sneak","grab","hit","kick","throw","struggle","gun"})

_LOG_LABELS = {
    "SITTING":"Sitting","IDLE":"Standing / Idle","LYING":"Lying Down",
    "CROUCHING":"Crouching","LEANING":"Leaning",
    "RUNNING":"Running","WALKING":"Walking","FALLING":"Falling",
    "sit":"Sitting","stand":"Standing / Idle","lying_down":"Lying Down",
    "run":"Running","walk":"Walking","fall":"Falling","sneak":"Sneaking",
    "grab":"Grabbing","hit":"Hitting","kick":"Kicking",
    "throw":"Throwing","struggle":"Struggling","gun":"Gun",
}


# =============================================================================
# DEVICE RESOLVER — unchanged from 6w
# =============================================================================

def resolve_device(requested: str) -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    r = requested.strip().lower()
    if r == "mps":
        if torch.backends.mps.is_available():
            log.info("Device : MPS"); return "mps"
        log.warning("MPS not available — fallback to CPU"); return "cpu"
    if r.startswith("cuda"):
        if torch.cuda.is_available():
            log.info("Device : %s  (%s)", requested, torch.cuda.get_device_name(0)); return requested
        log.warning("CUDA not available — fallback to CPU"); return "cpu"
    log.info("Device : cpu"); return "cpu"


# =============================================================================
# PULSE / DELTA / EMERGING — ported from phase1_observation.py
# =============================================================================

class DeltaReason(str, Enum):
    NEW_TRACK      = "NEW_TRACK"
    ZONE_ENTRY     = "ZONE_ENTRY"
    ZONE_EXIT      = "ZONE_EXIT"
    ACTION_SHIFT   = "ACTION_SHIFT"
    SPATIAL_DELTA  = "SPATIAL_DELTA"
    TEMPORAL_PULSE = "TEMPORAL_PULSE"


@dataclass
class Candidate:
    """
    First remembered unit — promoted from an ObservationDelta.
    Prototype version: no full ConfidenceProfile, just the fields
    needed for JSON serialisation and the EMERGING overlay.
    """
    candidate_id:      str
    track_id:          int
    frame_number:      int
    timestamp_sec:     float
    delta_reason:      DeltaReason
    bbox:              list[float]
    action_top:        str
    action_top_conf:   float
    action_dist:       dict[str, float]
    composite_score:   float
    action_valid:      bool     # True only after LSTM has produced real output
    zone_contacts:     list[str]
    motion_vector:     list[float]

    def to_dict(self) -> dict:
        return {
            "candidate_id":   self.candidate_id,
            "track_id":       self.track_id,
            "frame_number":   self.frame_number,
            "timestamp_sec":  round(self.timestamp_sec, 3),
            "delta_reason":   self.delta_reason.value,
            "bbox":           [round(v, 4) for v in self.bbox],
            "action_top":     self.action_top,
            "action_top_conf": round(self.action_top_conf, 4),
            "action_dist":    {k: round(v, 4) for k, v in self.action_dist.items()},
            "composite_score": round(self.composite_score, 4),
            "action_valid":   self.action_valid,
            "zone_contacts":  self.zone_contacts,
            "motion_vector":  [round(v, 5) for v in self.motion_vector],
        }


class EmergingQueue:
    """
    Lightweight replacement for phase1 Blackboard.
    Stores the most recent Candidate per track for:
      - JSON sidecar serialisation
      - EMERGING overlay in the writer thread

    Thread-safe. Prototype scope — no full Blackboard contracts.
    """
    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._by_track:  dict[int, Candidate] = {}  # most recent per track
        self._all:       list[Candidate]       = []  # full append-only log

    def write(self, c: Candidate) -> None:
        with self._lock:
            self._by_track[c.track_id] = c
            self._all.append(c)

    def get_recent(self, track_id: int) -> Candidate | None:
        with self._lock:
            return self._by_track.get(track_id)

    def expire(self, track_id: int) -> None:
        with self._lock:
            self._by_track.pop(track_id, None)

    def snapshot_all(self) -> list[Candidate]:
        with self._lock:
            return list(self._all)

    def count(self) -> int:
        with self._lock:
            return len(self._all)


@dataclass
class _TrackCont:
    """Per-track continuity state for DeltaDetector. Private."""
    track_id:          int
    first_seen_sec:    float
    last_delta_sec:    float           = 0.0
    last_centroid:     tuple           = (0.0, 0.0)
    last_action_label: str             = ""
    last_zones:        frozenset       = field(default_factory=frozenset)


class DeltaDetector:
    """
    Detects structural state transitions.
    Ported from phase1_observation.py with zone support stripped for
    prototype (zone_contacts always empty; add zone registry later).

    Returns a DeltaReason + metadata dict, or None if no delta.
    """

    def __init__(
        self,
        spatial_delta_threshold: float = 0.025,
        temporal_pulse_sec:      float = 5.0,
        action_shift_min_conf:   float = 0.55,
    ) -> None:
        self._spatial_thr    = spatial_delta_threshold
        self._pulse_sec      = temporal_pulse_sec
        self._shift_min_conf = action_shift_min_conf
        self._cont: dict[int, _TrackCont] = {}

    def detect(
        self,
        track_id:    int,
        timestamp:   float,
        bbox:        list[float],
        action_top:  str,
        action_conf: float,
        action_valid: bool,
    ) -> tuple[DeltaReason, dict] | tuple[None, None]:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        if track_id not in self._cont:
            self._cont[track_id] = _TrackCont(
                track_id       = track_id,
                first_seen_sec = timestamp,
                last_delta_sec = timestamp,
                last_centroid  = (cx, cy),
                last_action_label = action_top,
            )
            return DeltaReason.NEW_TRACK, {
                "prior_action": "",
                "current_action": action_top,
            }

        cont = self._cont[track_id]

        # ACTION_SHIFT — only when LSTM has produced real output (action_valid)
        if (action_valid
                and action_top != cont.last_action_label
                and action_conf >= self._shift_min_conf):
            prior = cont.last_action_label
            cont.last_action_label = action_top
            cont.last_centroid     = (cx, cy)
            cont.last_delta_sec    = timestamp
            return DeltaReason.ACTION_SHIFT, {
                "prior_action":   prior,
                "current_action": action_top,
            }

        # SPATIAL_DELTA
        dist = np.hypot(cx - cont.last_centroid[0], cy - cont.last_centroid[1])
        if dist >= self._spatial_thr:
            cont.last_centroid  = (cx, cy)
            cont.last_delta_sec = timestamp
            return DeltaReason.SPATIAL_DELTA, {
                "displacement": round(float(dist), 5),
                "prior_action": cont.last_action_label,
                "current_action": action_top,
            }

        # TEMPORAL_PULSE
        if (timestamp - cont.last_delta_sec) >= self._pulse_sec:
            cont.last_delta_sec = timestamp
            cont.last_centroid  = (cx, cy)
            return DeltaReason.TEMPORAL_PULSE, {
                "elapsed_sec": round(timestamp - cont.last_delta_sec, 2),
                "prior_action": cont.last_action_label,
                "current_action": action_top,
            }

        return None, None

    def expire(self, track_id: int) -> None:
        self._cont.pop(track_id, None)

    def get_stability(self, track_id: int) -> float:
        """Returns 1.0 for prototype; phase1 uses OKS rolling mean."""
        return 1.0


def _build_candidate(
    track_id:     int,
    frame_number: int,
    timestamp:    float,
    reason:       DeltaReason,
    bbox:         list[float],
    action_top:   str,
    action_top_conf: float,
    action_dist:  dict[str, float],
    composite:    float,
    action_valid: bool,
    motion_vec:   tuple[float, float],
    min_composite: float,
) -> Candidate | None:
    """Promote delta to Candidate if composite passes threshold."""
    if composite < min_composite:
        return None
    return Candidate(
        candidate_id    = str(uuid.uuid4()),
        track_id        = track_id,
        frame_number    = frame_number,
        timestamp_sec   = timestamp,
        delta_reason    = reason,
        bbox            = bbox,
        action_top      = action_top,
        action_top_conf = action_top_conf,
        action_dist     = action_dist,
        composite_score = composite,
        action_valid    = action_valid,
        zone_contacts   = [],
        motion_vector   = list(motion_vec),
    )


# =============================================================================
# OKS HELPER — unchanged from 6w
# =============================================================================

def compute_oks(kpts_a, conf_a, kpts_b, conf_b, bbox_b, kpt_thresh=0.1):
    bw = bbox_b[2] - bbox_b[0]; bh = bbox_b[3] - bbox_b[1]
    s2 = max(bw * bh, 1e-8)
    visible = (conf_a > kpt_thresh) & (conf_b > kpt_thresh)
    if not visible.any():
        return 0.0
    d2  = np.sum((kpts_a[visible] - kpts_b[visible]) ** 2, axis=1)
    k2  = (2.0 * _OKS_SIGMAS[visible]) ** 2
    return float(np.exp(-d2 / (2.0 * s2 * k2)).mean())


# =============================================================================
# CONFIDENCE RATE HELPER — unchanged from 6w
# =============================================================================

def _conf_rate_pct_per_sec(rate_setting):
    anchors = {1.0: 100.0/3.0, 2.0: 100.0/10.0, 3.0: 100.0/30.0}
    if rate_setting <= 1.0: return anchors[1.0]
    if rate_setting >= 3.0: return anchors[3.0]
    if rate_setting <= 2.0:
        t = rate_setting - 1.0
        return anchors[1.0] + t * (anchors[2.0] - anchors[1.0])
    t = rate_setting - 2.0
    return anchors[2.0] + t * (anchors[3.0] - anchors[2.0])


# =============================================================================
# CONFIDENCE TRACKER — unchanged from 6w
# =============================================================================

class ConfidenceTracker:
    SMOOTH_N = 8; _KF_Q = 1e-4; _KF_R = 5e-3

    def __init__(self, src_fps, src_w, src_h, rate_setting=2.0, awareness=None):
        self.fps = max(src_fps, 1.0); self.src_w = src_w; self.src_h = src_h
        self.rate_pps = _conf_rate_pct_per_sec(rate_setting)
        _aw_defaults = {
            "vel_stationary_px":0.5,"vel_drain_rate":0.8,"vel_drain_max":8.0,
            "dist_near_area":0.04,"dist_far_area":0.002,"dist_far_gain_scale":0.35,
            "entropy_full":0.95,"entropy_none":0.35,
        }
        self._aw = dict(_aw_defaults)
        if awareness: self._aw.update(awareness)
        self.confidence = 0.0
        self._kpt_history  = collections.deque(maxlen=self.SMOOTH_N)
        self._conf_history = collections.deque(maxlen=self.SMOOTH_N)
        self._locked_kpts = None; self._locked_conf = None
        self._lock_frames_remaining = 0
        self._prev_kpts = None
        self._wall_kpts = None; self._wall_conf = None
        self._person_px_width = src_w / 10.0; self._bbox_area_norm = 0.01
        self._kf_x = np.zeros((NUM_KP, 4), dtype=np.float64)
        self._kf_P = np.tile(np.eye(4, dtype=np.float64) * 0.1, (NUM_KP, 1, 1))
        self._kf_init = False
        self._kf_F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self._kf_H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        self._kf_Q = np.eye(4, dtype=np.float64) * self._KF_Q
        self._kf_R = np.eye(2, dtype=np.float64) * self._KF_R
        self._entropy_penalty = 0.0

    def _kalman_update(self, kpts_norm, kpt_conf, kpt_thresh):
        F,H,Q,R = self._kf_F, self._kf_H, self._kf_Q, self._kf_R
        smoothed = np.zeros((NUM_KP, 2), dtype=np.float32)
        for k in range(NUM_KP):
            if not self._kf_init: self._kf_x[k] = [kpts_norm[k,0], kpts_norm[k,1], 0., 0.]
            x, P = self._kf_x[k], self._kf_P[k]
            xp = F @ x; Pp = F @ P @ F.T + Q
            if kpt_conf[k] > kpt_thresh:
                z = kpts_norm[k].astype(np.float64); y = z - H @ xp
                S = H @ Pp @ H.T + R; K = Pp @ H.T @ np.linalg.inv(S)
                xu = xp + K @ y; Pu = (np.eye(4) - K @ H) @ Pp
            else:
                xu = xp; Pu = Pp
            self._kf_x[k] = xu; self._kf_P[k] = Pu; smoothed[k] = xu[:2].astype(np.float32)
        self._kf_init = True; return smoothed

    def _kalman_prediction_error(self, kpts_norm, kpt_conf, kpt_thresh):
        if not self._kf_init: return 0.0
        errors = []
        for k in range(NUM_KP):
            if kpt_conf[k] <= kpt_thresh: continue
            errors.append(np.linalg.norm((self._kf_x[k,:2] - kpts_norm[k].astype(np.float64)) * [self.src_w, self.src_h]))
        return float(np.mean(errors)) if errors else 0.0

    def _distance_gain_scale(self):
        a = self._bbox_area_norm; near = self._aw["dist_near_area"]; far = self._aw["dist_far_area"]
        if a >= near: return 1.0
        if a <= far:  return self._aw["dist_far_gain_scale"]
        t = (a - far) / (near - far + 1e-9)
        return self._aw["dist_far_gain_scale"] + t * (1. - self._aw["dist_far_gain_scale"])

    def set_entropy_penalty(self, ne):
        lo = self._aw["entropy_none"]; hi = self._aw["entropy_full"]
        if ne <= lo:   self._entropy_penalty = 0.
        elif ne >= hi: self._entropy_penalty = 1.
        else:          self._entropy_penalty = (ne - lo) / (hi - lo + 1e-9)

    def _glitch_threshold_px(self): return max(self._person_px_width, 8.)

    def _is_glitch(self, new_kpts, new_conf, kpt_thresh):
        if self._prev_kpts is None: return False
        thresh = self._glitch_threshold_px()
        valid = (new_conf > kpt_thresh)
        if valid.sum() == 0: return False
        diffs = np.linalg.norm((new_kpts[valid] - self._prev_kpts[valid]) * np.array([self.src_w, self.src_h]), axis=1)
        if bool(np.any(diffs > thresh)): return True
        return self._kalman_prediction_error(new_kpts, new_conf, kpt_thresh) > thresh * 3.

    def _pixel_velocity(self, new_kpts, new_conf, kpt_thresh):
        if self._prev_kpts is None: return 0.
        valid = (new_conf > kpt_thresh)
        if valid.sum() == 0: return 0.
        diffs = np.linalg.norm((new_kpts[valid] - self._prev_kpts[valid]) * np.array([self.src_w, self.src_h]), axis=1)
        return float(diffs.mean())

    def _smoothed_skeleton(self):
        return np.stack(list(self._kpt_history)).mean(0), np.stack(list(self._conf_history)).mean(0)

    def update(self, kpts_norm, kpt_conf, bbox_norm, kpt_thresh, detected, crowd_scale=1.):
        if bbox_norm is not None:
            px_w = (bbox_norm[2] - bbox_norm[0]) * self.src_w
            if px_w > 4: self._person_px_width = px_w
            self._bbox_area_norm = max((bbox_norm[2]-bbox_norm[0])*(bbox_norm[3]-bbox_norm[1]), 1e-6)
        if not detected:
            if self.confidence >= 100.:
                return {"kpts_to_draw": self._wall_kpts, "conf_to_draw": self._wall_conf, "vision_conf": self.confidence, "mode": "walled"}
            return {"kpts_to_draw": None, "conf_to_draw": None, "vision_conf": self.confidence, "mode": "occluded"}
        kpts_smooth = self._kalman_update(kpts_norm, kpt_conf, kpt_thresh)
        if self._is_glitch(kpts_norm, kpt_conf, kpt_thresh):
            return {"kpts_to_draw": self._locked_kpts if self._locked_kpts is not None else kpts_smooth,
                    "conf_to_draw": self._locked_conf if self._locked_conf is not None else kpt_conf,
                    "vision_conf": self.confidence, "mode": "glitch"}
        pix_vel = self._pixel_velocity(kpts_norm, kpt_conf, kpt_thresh)
        self._prev_kpts = kpts_norm.copy()
        self._kpt_history.append(kpts_smooth.copy()); self._conf_history.append(kpt_conf.copy())
        self._wall_kpts = kpts_smooth.copy(); self._wall_conf = kpt_conf.copy()
        base_gain = self.rate_pps / self.fps
        dist_scale = self._distance_gain_scale()
        uncert_scale = max(0., 1. - self._entropy_penalty)
        gain = base_gain * dist_scale * uncert_scale * crowd_scale
        stationary = pix_vel < self._aw["vel_stationary_px"]
        drain = 0. if stationary else min(pix_vel / max(self._person_px_width, 1.) * self._aw["vel_drain_rate"], self._aw["vel_drain_max"])
        self.confidence = float(np.clip(self.confidence + gain - drain, 0., 100.))
        if self.confidence < 80.:
            self._locked_kpts = None; self._locked_conf = None; self._lock_frames_remaining = 0
            return {"kpts_to_draw": kpts_smooth, "conf_to_draw": kpt_conf, "vision_conf": self.confidence, "mode": "live"}
        if self.confidence < 100.:
            if self._lock_frames_remaining <= 0 and self._kpt_history:
                self._locked_kpts, self._locked_conf = self._smoothed_skeleton()
            if not stationary:
                norm_vel = pix_vel / max(self._person_px_width, 1.)
                lock_sec = max(.1, 1. / (1. + norm_vel * 5.))
                self._lock_frames_remaining = max(self._lock_frames_remaining, int(lock_sec * self.fps))
            if self._lock_frames_remaining > 0:
                self._lock_frames_remaining -= 1
                return {"kpts_to_draw": self._locked_kpts, "conf_to_draw": self._locked_conf, "vision_conf": self.confidence, "mode": "locked"}
            if self._kpt_history:
                sk, sc = self._smoothed_skeleton()
                return {"kpts_to_draw": sk, "conf_to_draw": sc, "vision_conf": self.confidence, "mode": "locked"}
        if self._kpt_history:
            sk, sc = self._smoothed_skeleton(); self._locked_kpts = sk; self._locked_conf = sc
        return {"kpts_to_draw": self._locked_kpts if self._locked_kpts is not None else kpts_smooth,
                "conf_to_draw": self._locked_conf if self._locked_conf is not None else kpt_conf,
                "vision_conf": self.confidence, "mode": "walled"}

    def reset_on_redetect(self): self.confidence = 0.; self._kf_init = False


# =============================================================================
# FEATURE ENGINEERING — unchanged from 6w
# =============================================================================

def _joint_angle(p1, center, p2):
    v1, v2 = p1 - center, p2 - center
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1., 1.))))

def _extract_angles(kpts):
    triplets = [
        (5,7,9),(6,8,10),(5,11,13),(6,12,14),(11,13,15),(12,14,16),
        (0,5,6),(5,6,12),(5,11,12),(6,12,11),(5,6,11),(7,5,11),
    ]
    angles = []
    for p1i, ci, p2i in triplets:
        p1, pc, p2 = kpts[p1i], kpts[ci], kpts[p2i]
        if np.any(p1!=0) and np.any(pc!=0) and np.any(p2!=0):
            angles.append(_joint_angle(p1, pc, p2))
        else:
            angles.append(0.)
    return np.array(angles, dtype=np.float32)

def _normalize(kpts):
    ls, rs = kpts[5], kpts[6]; center = (ls+rs)/2.; scale = np.linalg.norm(rs-ls) + 1e-8
    return (kpts - center) / scale

def bbox_diagonal_normalize(kpts_norm, bbox):
    cx = (bbox[0]+bbox[2])/2.; cy = (bbox[1]+bbox[3])/2.
    diag = np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2) + 1e-8
    return (kpts_norm - np.array([cx,cy], dtype=np.float32)) / diag

def _sg_smooth_positions(pos_arr, window=9, poly=3):
    if not _SCIPY_OK: return pos_arr
    T = pos_arr.shape[0]; w = min(window, T)
    if w % 2 == 0: w -= 1
    if w < poly+2 or T < w: return pos_arr
    out = pos_arr.copy()
    for k in range(NUM_KP):
        for d in range(2):
            out[:,k,d] = _savgol(pos_arr[:,k,d], window_length=w, polyorder=poly)
    return out

def _build_sequence(position_buf, conf_buf, bbox_buf=None):
    T = len(position_buf)
    pos_arr  = np.stack(position_buf, axis=0)
    conf_arr = np.stack(conf_buf, axis=0)
    pos_arr  = _sg_smooth_positions(pos_arr)
    if bbox_buf and len(bbox_buf) == T:
        diag_norm = np.stack([bbox_diagonal_normalize(pos_arr[t], bbox_buf[t]) for t in range(T)])
    else:
        diag_norm = pos_arr.copy()
    norm_pos = np.stack([_normalize(diag_norm[t]) for t in range(T)])
    vel = np.zeros_like(norm_pos); acc = np.zeros_like(norm_pos)
    for k in range(NUM_KP):
        v = np.diff(norm_pos[:,k,:], axis=0); vel[1:,k,:] = v
        a = np.diff(vel[:,k,:], axis=0);       acc[1:,k,:] = a
    va = np.abs(vel); aa = np.abs(acc)
    vs  = float(np.percentile(va, 95)) if va.max() > 1e-8 else 1.
    as_ = float(np.percentile(aa, 95)) if aa.max() > 1e-8 else 1.
    vel = np.clip(vel / max(vs, 1e-8), -3., 3.)
    acc = np.clip(acc / max(as_, 1e-8), -3., 3.)
    angles = np.stack([_extract_angles(norm_pos[t]) for t in range(T)]) / 180.
    return np.concatenate(
        [norm_pos.reshape(T,-1), vel.reshape(T,-1), acc.reshape(T,-1), angles, conf_arr],
        axis=1).astype(np.float32)


# =============================================================================
# LSTM LOADER / PREDICTOR — unchanged from 6w (BiLSTM, auto detects dims)
# =============================================================================

def load_lstm_model(model_path, hidden_size=128, num_layers=2, device="cpu"):
    import torch, torch.nn as nn
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    ckpt_feat_dim = FEATURE_DIM
    if "lstm.weight_ih_l0" in ckpt:
        detected_hidden = ckpt["lstm.weight_ih_l0"].shape[0] // 4
        hidden_size = detected_hidden
        ckpt_feat_dim = ckpt["lstm.weight_ih_l0"].shape[1]
        if ckpt_feat_dim != FEATURE_DIM:
            log.warning("LSTM feat_dim mismatch: ckpt=%d build=%d — will pad/trim", ckpt_feat_dim, FEATURE_DIM)
    detected_layers = sum(1 for k in ckpt if k.startswith("lstm.weight_ih_l") and "_reverse" not in k)
    if detected_layers > 0: num_layers = detected_layers
    num_classes = ckpt["head.weight"].shape[0] if "head.weight" in ckpt else NUM_CLASSES

    class _BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=ckpt_feat_dim, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=True,
                                dropout=.5 if num_layers > 1 else 0.)
            self.drop = nn.Dropout(.5); self.head = nn.Linear(hidden_size*2, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x); return self.head(self.drop(out.mean(dim=1)))

    model = _BiLSTM().to(device); model.load_state_dict(ckpt); model.eval()
    model.feat_dim = ckpt_feat_dim
    log.info("LSTM: %s  (feat=%d hid=%d layers=%d classes=%d)",
             model_path, ckpt_feat_dim, hidden_size, num_layers, num_classes)
    return model

def predict_sequence(model, seq, device="cpu", disabled_classes=None):
    import torch
    if not np.isfinite(seq).all(): seq = np.nan_to_num(seq, nan=0., posinf=3., neginf=-3.)
    mfd = getattr(model, "feat_dim", FEATURE_DIM); T, sd = seq.shape
    if sd != mfd:
        seq = seq[:,:mfd] if sd > mfd else np.concatenate([seq, np.zeros((T, mfd-sd), dtype=seq.dtype)], 1)
    x = torch.tensor(seq[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1).squeeze().contiguous().cpu().numpy()
    if not np.isfinite(probs).all(): probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
    masked = probs.copy()
    if disabled_classes:
        for cls in disabled_classes:
            idx = ACTION_TO_IDX.get(cls)
            if idx is not None: masked[idx] = 0.
    best = int(np.argmax(masked))
    return {
        "action":       ACTION_CLASSES[best],
        "confidence":   float(masked[best]),
        "class_scores": {ACTION_CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


# =============================================================================
# PURE VISION FLAGS — unchanged from 6w
# =============================================================================

def pure_vision_flags(kpts_norm, conf, kpt_thresh=0.15):
    def vis(i): return bool(conf[i] > kpt_thresh)
    def y(i):   return float(kpts_norm[i, 1])
    def x(i):   return float(kpts_norm[i, 0])
    flags = []; n_visible = int((conf > kpt_thresh).sum())
    if n_visible < 4: return flags
    has_shoulder = vis(5) and vis(6); has_hip = vis(11) and vis(12)
    has_knee = vis(13) or vis(14); has_ankle = vis(15) or vis(16)
    has_wrist = vis(9) or vis(10); has_elbow = vis(7) or vis(8)
    if has_shoulder and has_hip:
        sh_my = (y(5)+y(6))/2.; hip_my = (y(11)+y(12))/2.
        sh_mx = (x(5)+x(6))/2.; hip_mx = (x(11)+x(12))/2.
        dy = abs(sh_my - hip_my); dx = abs(sh_mx - hip_mx)
        if dy < 0.12:
            if has_ankle and abs(y(15 if vis(15) else 16) - sh_my) < 0.20: flags.append("LYING")
            else: flags.append("FALLING")
        elif dx > dy*1.1: flags.append("LEANING")
        else:
            if has_hip and has_knee:
                ky = (y(13) if vis(13) else y(14))
                if ky < hip_my - 0.06: flags.append("SITTING")
                elif ky < hip_my - 0.02: flags.append("CROUCHING")
            if not any(f in flags for f in ("SITTING", "CROUCHING")):
                if has_ankle and vis(15) and vis(16):
                    asp = abs(x(15)-x(16)); sw = abs(x(5)-x(6)) + 1e-4
                    sr = asp / sw
                    if sr > 2.2: flags.append("RUNNING")
                    elif sr > 1.3: flags.append("WALKING")
    if has_shoulder and has_wrist:
        wa = []; sty = min(y(5), y(6))
        for wi in (9, 10):
            if vis(wi) and y(wi) < sty - 0.05: wa.append(wi)
        if len(wa) == 2: flags.append("ARMS UP")
        elif len(wa) == 1 and has_elbow: flags.append("REACHING")
    if vis(9) and vis(10) and has_shoulder:
        if abs(x(9)-x(10)) > abs(x(5)-x(6))*2.5:
            if "ARMS UP" not in flags: flags.append("ARMS WIDE")
    if not flags and has_shoulder and has_hip: flags.append("IDLE")
    return flags


# =============================================================================
# ROLE TRACKER — unchanged from 6w
# =============================================================================

def _point_in_polygon(px, py, pts):
    n = len(pts); inside = False; j = n - 1
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi+1e-12)+xi):
            inside = not inside
        j = i
    return inside

class RoleTracker:
    def __init__(self, manual_map, employee_zones, loco_thresh=0.35, static_thresh=0.45, window_frames=150):
        self.manual_map = manual_map; self.employee_zones = employee_zones
        self.loco_thresh = loco_thresh; self.static_thresh = static_thresh
        self.window_frames = window_frames
        self._action_hist: dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_frames))
        self._role_cache: dict[int, str] = {}

    def update(self, track_id, action, cx, cy):
        if track_id in self.manual_map: return self.manual_map[track_id]
        for zone in self.employee_zones:
            if zone and _point_in_polygon(cx, cy, zone):
                self._role_cache[track_id] = "employee"; return "employee"
        self._action_hist[track_id].append(action)
        hist = list(self._action_hist[track_id]); n = len(hist)
        if n < 20: return self._role_cache.get(track_id, "unknown")
        loco_r   = sum(1 for a in hist if a in _LOCO_ACTIONS) / n
        static_r = sum(1 for a in hist if a in _STATIC_ACTIONS) / n
        if loco_r > self.loco_thresh:     role = "customer"
        elif static_r > self.static_thresh: role = "employee"
        else: role = self._role_cache.get(track_id, "unknown")
        self._role_cache[track_id] = role; return role

    def get_role(self, track_id):
        if track_id in self.manual_map: return self.manual_map[track_id]
        return self._role_cache.get(track_id, "unknown")

    def prune(self, active_ids):
        dead = [tid for tid in self._role_cache if tid not in active_ids]
        for tid in dead: self._role_cache.pop(tid, None); self._action_hist.pop(tid, None)


# =============================================================================
# PERSON TRACKER — simplified from 6w (mode 1 only for prototype)
# ConfidenceTracker used for skeleton smoothing in mode 2.
# =============================================================================

class PersonTracker:
    def __init__(self, window_size=16, iou_thresh=0.30, max_age=30,
                 src_fps=25., src_w=1280, src_h=720,
                 confidence_rate=2., vision_mode=2,
                 awareness=None, oks_alpha=0.5):
        self.window_size = window_size; self.iou_thresh = iou_thresh; self.max_age = max_age
        self._src_fps = src_fps; self._src_w = src_w; self._src_h = src_h
        self._confidence_rate = confidence_rate; self._vision_mode = vision_mode
        self._awareness = awareness; self._oks_alpha = oks_alpha
        self._next_id = 0; self._tracks: dict[int, dict] = {}
        self._oks_history: dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=60))

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / (ua + 1e-8)

    def _match_score(self, det, track):
        iou = self._iou(det["bbox"], track["bbox"])
        if "kpts" in det and track.get("last_kpts") is not None:
            oks = compute_oks(det["kpts"], det["conf"], track["last_kpts"], track["last_kptconf"], det["bbox"])
        else:
            oks = iou
        return self._oks_alpha * oks + (1. - self._oks_alpha) * iou

    def update(self, detections, crowd_scale=1.):
        for t in self._tracks.values(): t["age"] += 1
        matched = set(); results = []
        for det in detections:
            best_tid = None; best_score = self.iou_thresh
            for tid, track in self._tracks.items():
                if tid in matched: continue
                score = self._match_score(det, track)
                if score > best_score: best_score = score; best_tid = tid
            if best_tid is None:
                best_tid = self._next_id; self._next_id += 1
                self._tracks[best_tid] = {
                    "bbox": det["bbox"], "age": 0,
                    "pos_buf":  collections.deque(maxlen=self.window_size),
                    "conf_buf": collections.deque(maxlen=self.window_size),
                    "bbox_buf": collections.deque(maxlen=self.window_size),
                    "last_pred": {"action":"?","confidence":0.,"class_scores":{}},
                    "conf_tracker": ConfidenceTracker(
                        self._src_fps, self._src_w, self._src_h,
                        self._confidence_rate, self._awareness),
                    "was_walled": False, "last_kpts": None, "last_kptconf": None,
                }
            track = self._tracks[best_tid]
            track["bbox"] = det["bbox"]; track["age"] = 0
            track["pos_buf"].append(det["kpts"])
            track["conf_buf"].append(det["conf"])
            track["bbox_buf"].append(det["bbox"])
            track["last_kpts"] = det["kpts"].copy(); track["last_kptconf"] = det["conf"].copy()
            matched.add(best_tid)

            if track.get("was_walled") and track["age"] == 0:
                track["conf_tracker"].reset_on_redetect()
            track["was_walled"] = False

            ct = track["conf_tracker"]
            if self._vision_mode == 1:
                results.append({
                    "track_id": best_tid, "bbox": det["bbox"],
                    "kpts_raw": det["kpts"], "conf": det["conf"],
                    "kpts_live": det["kpts"], "conf_live": det["conf"],
                    "pos_buf": track["pos_buf"], "conf_buf": track["conf_buf"],
                    "bbox_buf": track["bbox_buf"], "last_pred": track["last_pred"],
                    "vision_conf": 0., "vision_mode": "inference",
                    "mean_oks": self._mean_oks(best_tid),
                })
            else:
                cvr = ct.update(det["kpts"], det["conf"], det["bbox"], 0.25, True, crowd_scale)
                results.append({
                    "track_id": best_tid, "bbox": det["bbox"],
                    "kpts_raw": cvr["kpts_to_draw"], "conf": cvr["conf_to_draw"],
                    "kpts_live": det["kpts"], "conf_live": det["conf"],
                    "pos_buf": track["pos_buf"], "conf_buf": track["conf_buf"],
                    "bbox_buf": track["bbox_buf"], "last_pred": track["last_pred"],
                    "vision_conf": cvr["vision_conf"], "vision_mode": cvr["mode"],
                    "mean_oks": self._mean_oks(best_tid),
                })

        # Handle walled tracks (mode 2 only)
        for tid, track in self._tracks.items():
            if tid in matched: continue
            if self._vision_mode == 1: continue
            ct = track["conf_tracker"]
            cvr = ct.update(np.zeros((NUM_KP,2), dtype=np.float32),
                            np.zeros(NUM_KP, dtype=np.float32),
                            None, 0.25, False, crowd_scale)
            if cvr["mode"] == "walled" and cvr["kpts_to_draw"] is not None:
                track["was_walled"] = True
                results.append({
                    "track_id": tid, "bbox": track["bbox"],
                    "kpts_raw": cvr["kpts_to_draw"], "conf": cvr["conf_to_draw"],
                    "kpts_live": None, "conf_live": None,
                    "pos_buf": track["pos_buf"], "conf_buf": track["conf_buf"],
                    "bbox_buf": track.get("bbox_buf", collections.deque()),
                    "last_pred": track["last_pred"],
                    "vision_conf": cvr["vision_conf"], "vision_mode": "walled",
                    "mean_oks": self._mean_oks(tid),
                })

        if self._vision_mode == 1:
            dead = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
        else:
            dead = [tid for tid, t in self._tracks.items()
                    if t["age"] > self.max_age and t["conf_tracker"].confidence < 100.]
        for tid in dead:
            del self._tracks[tid]; self._oks_history.pop(tid, None)
        return results, set(dead)

    def _mean_oks(self, tid):
        h = self._oks_history.get(tid)
        return float(np.mean(list(h))) if h else 0.

    def set_prediction(self, track_id, pred):
        if track_id not in self._tracks: return
        self._tracks[track_id]["last_pred"] = pred
        scores = pred.get("class_scores", {})
        if scores:
            probs = np.array(list(scores.values()), dtype=np.float64)
            probs /= (probs.sum() + 1e-9)
            h = -float(np.sum(probs * np.log(probs + 1e-9)))
            self._tracks[track_id]["conf_tracker"].set_entropy_penalty(h / (np.log(len(probs)) + 1e-9))


# =============================================================================
# DRAWING HELPERS — unchanged from 6w
# =============================================================================

def _blend_rect(img, pt1, pt2, color, alpha):
    """Alpha-blend a filled rectangle without copying the whole frame."""
    h, w = img.shape[:2]
    x1, y1 = pt1
    x2, y2 = pt2
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    if alpha >= 0.99:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return

    roi = img[y1:y2, x1:x2]
    color_img = np.empty_like(roi)
    color_img[:] = color
    cv2.addWeighted(color_img, alpha, roi, 1.0 - alpha, 0, roi)


def _blend_poly(img, pts_px, color, alpha):
    """Alpha-blend a filled polygon inside its bounding ROI only."""
    if pts_px is None or len(pts_px) < 3:
        return

    h, w = img.shape[:2]
    pts = np.asarray(pts_px, dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return

    shifted = pts.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1

    roi = img[y1:y2, x1:x2]
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)

    color_img = np.empty_like(roi)
    color_img[:] = color
    blended = cv2.addWeighted(color_img, alpha, roi, 1.0 - alpha, 0)
    np.copyto(roi, blended, where=mask[:, :, None].astype(bool))


def _put_text_with_bg(img, text, org, font_scale=0.55, thickness=1,
                      fg=(255,255,255), bg=(30,30,30), alpha=0.65):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org; pad = 3
    _blend_rect(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), bg, alpha)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_person(img, bbox_norm, kpts_norm, conf, track_id, pred,
                show_scores=False, kpt_thresh=0.25, vision_conf=0.0,
                vision_mode="live", kpts_live=None, conf_live=None,
                dual_skeleton=False, mean_oks=0.0, role="unknown",
                has_emerging=False) -> None:
    """
    6w draw_person + EMERGING indicator (green dot top-left of bbox).
    has_emerging: True when an active Candidate exists for this track.
    """
    h, w = img.shape[:2]
    action = pred.get("action", "?"); confidence = pred.get("confidence", 0.)
    color  = ACTION_COLORS.get(action, (200,200,200))
    display_name = ACTION_DISPLAY.get(action, action.replace("_"," ").capitalize())
    x1 = int(bbox_norm[0]*w); y1 = int(bbox_norm[1]*h)
    x2 = int(bbox_norm[2]*w); y2 = int(bbox_norm[3]*h)
    box_w = max(x2-x1, 1); box_h = max(y2-y1, 1)

    if vision_mode == "walled":
        dash_color = (180,180,50)
        for ep in [((x1,y1),(x2,y1)),((x1,y2),(x2,y2)),((x1,y1),(x1,y2)),((x2,y1),(x2,y2))]:
            p0, p1_ = ep; total = max(abs(p1_[0]-p0[0]), abs(p1_[1]-p0[1]))
            for i in range(0, total, 16):
                t0s = i/total; t1s = min((i+8)/total, 1.)
                pt0 = (int(p0[0]+(p1_[0]-p0[0])*t0s), int(p0[1]+(p1_[1]-p0[1])*t0s))
                pt1 = (int(p0[0]+(p1_[0]-p0[0])*t1s), int(p0[1]+(p1_[1]-p0[1])*t1s))
                cv2.line(img, pt0, pt1, dash_color, 2, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2, cv2.LINE_AA)

    # EMERGING indicator — green dot + "E" at top-left of bbox
    if has_emerging:
        cv2.circle(img, (x1+6, y1+6), 6, (0, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(img, "E", (x1+2, y1+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (0,0,0), 1, cv2.LINE_AA)

    if kpts_norm is None or conf is None: return

    skel_map  = {"inference":(0,220,80),"walled":(180,180,50),"locked":(0,200,255),"glitch":(0,80,200)}
    kpt_cmap  = {"walled":(80,200,200),"locked":(0,150,255),"glitch":(0,80,200)}
    skel_color = skel_map.get(vision_mode, (0,220,80))
    kpt_color  = kpt_cmap.get(vision_mode, (0,120,255))
    kp_idx = {name: i for i, name in enumerate(COCO_KP_NAMES)}

    def _draw_skel(kp, kc, ec, dc, lt=2, dr=4, alpha=1.):
        tgt = img
        ox = oy = 0
        if alpha < 1.:
            visible = [(int(kp[i,0]*w), int(kp[i,1]*h)) for i in range(NUM_KP)
                       if kc[i] >= kpt_thresh]
            if not visible:
                return
            xs = [p[0] for p in visible]
            ys = [p[1] for p in visible]
            margin = max(8, dr + lt + 6)
            ox = max(0, min(xs) - margin)
            oy = max(0, min(ys) - margin)
            rx2 = min(w, max(xs) + margin + 1)
            ry2 = min(h, max(ys) + margin + 1)
            if rx2 <= ox or ry2 <= oy:
                return
            tgt = img[oy:ry2, ox:rx2].copy()
        for na, nb in SKELETON_EDGES:
            ia, ib = kp_idx[na], kp_idx[nb]
            if kc[ia] < kpt_thresh or kc[ib] < kpt_thresh: continue
            pa = (int(kp[ia,0]*w), int(kp[ia,1]*h))
            pb = (int(kp[ib,0]*w), int(kp[ib,1]*h))
            if alpha < 1.:
                pa = (pa[0] - ox, pa[1] - oy)
                pb = (pb[0] - ox, pb[1] - oy)
            cv2.line(tgt, pa, pb, ec, lt, cv2.LINE_AA)
        for i in range(NUM_KP):
            if kc[i] < kpt_thresh: continue
            px, py_ = int(kp[i,0]*w), int(kp[i,1]*h)
            if alpha < 1.:
                px -= ox
                py_ -= oy
            cv2.circle(tgt, (px,py_), dr, dc, -1, cv2.LINE_AA)
            cv2.circle(tgt, (px,py_), dr, (255,255,255), 1, cv2.LINE_AA)
        if alpha < 1.:
            roi = img[oy:oy+tgt.shape[0], ox:ox+tgt.shape[1]]
            cv2.addWeighted(tgt, alpha, roi, 1-alpha, 0, roi)

    if dual_skeleton and kpts_live is not None and conf_live is not None:
        _draw_skel(kpts_live, conf_live, (0,220,80), (0,120,255), lt=1, dr=3, alpha=.5)
    _draw_skel(kpts_norm, conf, skel_color, kpt_color)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(np.clip(box_h*0.33/28., 0.55, 3.)); thickness = max(1, int(font_scale*2.))
    (tw, th), bl = cv2.getTextSize(display_name, font, font_scale, thickness)
    pad = max(4, int(font_scale*5)); label_x = x1 + max(0,(box_w-tw)//2); label_y = max(y1-pad, th+pad)
    _blend_rect(img, (label_x-pad, label_y-th-pad), (label_x+tw+pad, label_y+bl+pad), color, .80)
    brightness = .299*color[2] + .587*color[1] + .114*color[0]
    cv2.putText(img, display_name, (label_x, label_y), font, font_scale,
                (0,0,0) if brightness > 128 else (255,255,255), thickness, cv2.LINE_AA)

    bar_h = max(4, int(box_h*.025)); filled = int(box_w*confidence)
    cv2.rectangle(img, (x1,y1), (x2,y1+bar_h), (50,50,50), -1)
    cv2.rectangle(img, (x1,y1), (x1+filled, y1+bar_h), color, -1)

    # Role tag
    role_col = ROLE_COLOR.get(role, (120,120,120))
    role_lbl = f"[{ROLE_LABEL.get(role,'?')}]"
    role_scale = max(0.30, font_scale * 0.45)
    (rtw,rth), rbl = cv2.getTextSize(role_lbl, font, role_scale, 1)
    rx = x1; ry = min(y2 + rth + 4, h - rbl - 2)
    _blend_rect(img, (rx-2, ry-rth-2), (rx+rtw+2, ry+rbl+2), role_col, .80)
    r_bright = .299*role_col[2] + .587*role_col[1] + .114*role_col[0]
    cv2.putText(img, role_lbl, (rx,ry), font, role_scale,
                (0,0,0) if r_bright > 128 else (255,255,255), 1, cv2.LINE_AA)

    if vision_mode != "inference":
        vc_pct = int(round(vision_conf))
        mode_icons = {"walled":"W","locked":"L","glitch":"G"}
        mode_icon  = mode_icons.get(vision_mode, "·")
        vc_bgs = {"walled":(60,60,20),"locked":(40,40,60),"glitch":(20,20,60)}
        vc_fgs = {"walled":(100,200,50),"locked":(0,200,255),"glitch":(50,80,220)}
        vc_bg  = vc_bgs.get(vision_mode, (20,20,20)); vc_fg = vc_fgs.get(vision_mode, (180,255,180))
        vc_text = f"{mode_icon} {vc_pct}% OKS:{mean_oks:.2f}"
        vc_scale = max(.35, font_scale*.50)
        (vc_tw, vc_th), vc_bl = cv2.getTextSize(vc_text, font, vc_scale, 1)
        vc_pad = 3; vc_x = x1; vc_y = min(y2+vc_th+vc_pad+2+(rth+6), h-vc_bl-vc_pad)
        _blend_rect(img, (vc_x-vc_pad, vc_y-vc_th-vc_pad),
                    (vc_x+vc_tw+vc_pad, vc_y+vc_bl+vc_pad), vc_bg, .75)
        cv2.putText(img, vc_text, (vc_x,vc_y), font, vc_scale, vc_fg, 1, cv2.LINE_AA)

    if show_scores and pred.get("class_scores"):
        scores = pred["class_scores"]; panel_x = x2+6; panel_y = y1
        for cls in sorted(scores, key=lambda c: -scores[c])[:7]:
            sc = scores[cls]; bar = int(80*sc)
            cv2.rectangle(img, (panel_x,panel_y), (panel_x+80,panel_y+10), (40,40,40), -1)
            cv2.rectangle(img, (panel_x,panel_y), (panel_x+bar,panel_y+10), ACTION_COLORS.get(cls,(180,180,180)), -1)
            cv2.putText(img, f"{cls[:8]} {sc:.2f}", (panel_x+84,panel_y+9), font, .35, (220,220,220), 1, cv2.LINE_AA)
            panel_y += 13


def draw_fences(img, fences, persons, pulse=False):
    h, w = img.shape[:2]; breaches = []
    for fence in fences:
        pts_norm = fence.get("pts", [])
        if len(pts_norm) < 3: continue
        name = fence.get("name","zone"); color_bgr = tuple(fence.get("color_bgr",[0,200,255]))
        roles_allowed = set(fence.get("roles_allowed",[]))
        pts_px = np.array([[int(p[0]*w), int(p[1]*h)] for p in pts_norm], dtype=np.int32)
        overlay = img.copy(); cv2.fillPoly(overlay, [pts_px], color_bgr)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts_px], True, color_bgr, 2, cv2.LINE_AA)
        cx_px = int(pts_px[:,0].mean()); cy_px = int(pts_px[:,1].mean())
        cv2.putText(img, name, (cx_px-20, cy_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        for person in persons:
            role = person.get("role","unknown")
            if roles_allowed and role in roles_allowed: continue
            bb = person.get("bbox")
            if bb is None: continue
            pcx = (bb[0]+bb[2])/2.; pcy = (bb[1]+bb[3])/2.
            if _point_in_polygon(pcx, pcy, pts_norm):
                breaches.append({"fence":name,"track_id":person["track_id"],"role":role,"cx":round(pcx,4),"cy":round(pcy,4)})
                red_ov = img.copy(); cv2.fillPoly(red_ov, [pts_px], (0,0,220))
                alpha = 0.35 if pulse else 0.20
                cv2.addWeighted(red_ov, alpha, img, 1-alpha, 0, img)
                cv2.polylines(img, [pts_px], True, (0,0,255), 3, cv2.LINE_AA)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bx1 = int(bb[0]*w); by1 = int(bb[1]*h)
                breach_txt = f"BREACH [{name}]"
                (btw,bth), _ = cv2.getTextSize(breach_txt, font, 0.5, 1)
                bx = bx1; by = max(by1-8, bth+4)
                ov2 = img.copy()
                cv2.rectangle(ov2, (bx-3,by-bth-3), (bx+btw+3,by+3), (0,0,180), -1)
                cv2.addWeighted(ov2, 0.85, img, 0.15, 0, img)
                cv2.putText(img, breach_txt, (bx,by), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return breaches


def draw_hud(img, frame_idx, fps, num_persons, vision_mode=0,
             queue_size=0, skip_rate=0., proc_fps=0., n_emerging=0):
    ts = frame_idx/fps if fps > 0 else 0; m, s = divmod(int(ts), 60)
    mode_label = {
        0:"MODE 0 · PURE VISION", 1:"MODE 1 · LSTM BASIC",
        2:"MODE 2 · LSTM+THRESHOLD", 3:"MODE 3 · COMBINED",
    }.get(vision_mode, f"MODE {vision_mode}")
    fps_str = f"  |  {proc_fps:.1f} fps" if proc_fps > 0 else ""
    # v7: add emerging count to HUD
    emg_str = f"  |  emerging:{n_emerging}"
    text = (f"Frame {frame_idx}  |  {m:02d}:{s:02d}  |  persons:{num_persons}"
            f"  |  {mode_label}  |  Q:{queue_size} skip:{skip_rate:.0%}{fps_str}{emg_str}")
    _put_text_with_bg(img, text, (10, 55), font_scale=2.00, bg=(20,20,20))  # was (10, 22) — baseline at 22 clipped glyphs at scale 2.0


# =============================================================================
# FLAG LOG + DRAW — unchanged from 6w
# =============================================================================

class FlagLog:
    def __init__(self, obs_window_sec=5., fps=25.):
        self.obs_window_sec = obs_window_sec; self.fps = max(fps, 1.)
        self._static_now = {}; self._dyn_hist = collections.deque(); self._frame_idx = 0

    def update(self, frame_idx, flag_counts):
        self._frame_idx = frame_idx; static_snap = {}; dyn_snap = {}
        for flag, cnt in flag_counts.items():
            if flag in _STATIC_FLAGS or flag.upper() in _STATIC_FLAGS:
                key = flag if flag in _STATIC_FLAGS else flag.upper()
                static_snap[key] = static_snap.get(key, 0) + cnt
            elif flag in _DYNAMIC_FLAGS or flag.upper() in _DYNAMIC_FLAGS:
                dyn_snap[flag] = dyn_snap.get(flag, 0) + cnt
        self._static_now = static_snap
        if dyn_snap: self._dyn_hist.append((frame_idx, dyn_snap))
        cutoff = frame_idx - int(self.obs_window_sec * self.fps)
        while self._dyn_hist and self._dyn_hist[0][0] < cutoff: self._dyn_hist.popleft()

    def static_rows(self): return sorted([(k,v) for k,v in self._static_now.items() if v>0], key=lambda x:-x[1])
    def dynamic_rows(self):
        peak = {}
        for _, snap in self._dyn_hist:
            for flag, cnt in snap.items(): peak[flag] = max(peak.get(flag,0), cnt)
        return sorted([(k,v) for k,v in peak.items() if v>0], key=lambda x:-x[1])


def draw_flag_log(img, flag_log, position=1, obs_sec=5., font_scale=2.00, alpha=0.84, max_rows=10):
    h, w = img.shape[:2]; font = cv2.FONT_HERSHEY_SIMPLEX; thick = 2
    line_h = int(font_scale*40); pad = 8; acc_w = 4
    static_rows  = flag_log.static_rows()[:max_rows]
    dynamic_rows = flag_log.dynamic_rows()[:max_rows]
    static_labels = {_LOG_LABELS.get(r[0], r[0]) for r in static_rows}
    display = []; rank = 1
    for flag, cnt in static_rows:
        lbl = _LOG_LABELS.get(flag, flag.replace("_"," ").capitalize())
        display.append((f"({rank}) {lbl}", cnt, False)); rank += 1
    for flag, cnt in dynamic_rows:
        lbl = _LOG_LABELS.get(flag, flag.replace("_"," ").capitalize())
        if lbl in static_labels: continue
        display.append((f"({rank}) {lbl}", cnt, True)); rank += 1
    if not display: return
    title_str = "  Activity Log"
    (ttw, tth), _ = cv2.getTextSize(title_str, font, font_scale, thick); max_tw = ttw
    for lbl, cnt, is_dyn in display:
        suffix = f"  ~ {cnt} persons" if is_dyn else f"  {cnt} persons"
        (tw, _), _ = cv2.getTextSize(lbl+suffix, font, font_scale, thick); max_tw = max(max_tw, tw)
    panel_w = max_tw+pad*2+acc_w+10; title_h = line_h+6; footer_h = line_h-4
    panel_h = title_h + len(display)*line_h + footer_h + pad
    margin = 10; hud_offset = 80  # was 36 — increased so Activity Log clears the HUD bar
    positions = {1:(margin, hud_offset+margin), 2:(w-panel_w-margin, hud_offset+margin),
                 3:(margin, h-panel_h-margin), 4:(w-panel_w-margin, h-panel_h-margin)}
    px, py = positions.get(position, (margin, hud_offset+margin))
    ov = img.copy(); cv2.rectangle(ov, (px,py), (px+panel_w, py+panel_h), (15,15,15), -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (px,py), (px+panel_w, py+title_h-2), (30,30,30), -1)
    cv2.putText(img, title_str, (px+acc_w+4, py+tth+pad-2), font, font_scale, (255,255,255), 2, cv2.LINE_AA)
    cv2.line(img, (px, py+title_h-1), (px+panel_w, py+title_h-1), (55,55,55), 1)
    ry = py + title_h
    # FIX: one overlay for the whole row panel, not one img.copy() per row.
    # Original allocated a full frame (23.7 MiB at 4K) per row, causing
    # _ArrayMemoryError on CPU. Now: one copy -> all backgrounds -> one blend.
    row_ov = img.copy()
    _ry = py + title_h
    for lbl, cnt, is_dyn in display:
        rb = (10,38,60) if is_dyn else (50,30,8)
        cv2.rectangle(row_ov, (px,_ry), (px+panel_w,_ry+line_h-1), rb, -1)
        _ry += line_h
    cv2.addWeighted(row_ov, .50, img, .50, 0, img)
    del row_ov
    for lbl, cnt, is_dyn in display:
        rf = (20,200,255) if is_dyn else (20,235,255)
        hl = (0,140,230) if is_dyn else (160,90,20)
        cv2.rectangle(img, (px,ry), (px+acc_w,ry+line_h-1), hl, -1)
        suffix = f"  ~ {cnt} persons" if is_dyn else f"  {cnt} persons"
        cv2.putText(img, lbl+suffix, (px+acc_w+6, ry+line_h-int(line_h*.20)), font, font_scale, rf, thick, cv2.LINE_AA)
        ry += line_h
    cv2.putText(img, f"  ~ = peak in last {int(obs_sec)}s",
                (px+acc_w+4, ry+footer_h-4), font, max(font_scale-.10,.30), (90,90,90), 1, cv2.LINE_AA)


# =============================================================================
# SUMMARY LOG — unchanged from 6w (includes roles + fence breaches)
# =============================================================================

class SummaryLog:
    def __init__(self, output_path, src_fps=25., checkpoint_frames=300, meta=None):
        self.out_path = str(Path(output_path).with_suffix(".summary.json"))
        self.src_fps = max(src_fps, 1.); self.checkpoint_frames = checkpoint_frames; self.meta = meta or {}
        self._frame_idx = 0; self._counts = {}; self._maxes = {}; self._total_frames = 0
        self._timeline = []; self._last_ckpt = 0
        self._role_frames: dict[str,int] = {"employee":0,"customer":0,"unknown":0}
        self._fence_breaches: dict[str,int] = {}
        import atexit, signal as _sig
        atexit.register(self.flush)
        try:
            _orig = _sig.getsignal(_sig.SIGTERM)
            def _handler(s, f): self.flush(); (callable(_orig) and _orig(s, f))
            _sig.signal(_sig.SIGTERM, _handler)
        except (OSError, ValueError): pass

    _LOG_LABELS_LOCAL = _LOG_LABELS

    def update(self, frame_idx, flag_counts, role_counts=None, breach_list=None):
        self._frame_idx = frame_idx; self._total_frames += 1; snapshot = {}
        for key, cnt in flag_counts.items():
            if cnt <= 0: continue
            lbl = self._LOG_LABELS_LOCAL.get(key, self._LOG_LABELS_LOCAL.get(key.upper(), key))
            self._counts.setdefault(lbl, []).append(cnt)
            self._maxes[lbl] = max(self._maxes.get(lbl, 0), cnt); snapshot[lbl] = cnt
        if role_counts:
            for role, n in role_counts.items():
                self._role_frames[role] = self._role_frames.get(role, 0) + n
        if breach_list:
            for b in breach_list:
                fn = b.get("fence","unknown")
                self._fence_breaches[fn] = self._fence_breaches.get(fn, 0) + 1
        if frame_idx - self._last_ckpt >= self.checkpoint_frames:
            if snapshot:
                self._timeline.append({"frame": frame_idx, "ts": round(frame_idx/self.src_fps, 2),
                                        "counts": dict(sorted(snapshot.items(), key=lambda x:-x[1]))})
            self._last_ckpt = frame_idx; self.flush()

    def _build(self):
        poses = {}
        for lbl, counts in self._counts.items():
            active = len(counts)
            poses[lbl] = {"avg_persons": round(sum(counts)/active, 2) if active else 0.,
                          "max_persons": self._maxes.get(lbl, 0), "active_frames": active,
                          "total_frames": self._total_frames,
                          "presence_pct": round(100.*active/max(self._total_frames, 1), 1)}
        poses = dict(sorted(poses.items(), key=lambda x:-x[1]["max_persons"]))
        total_r = max(sum(self._role_frames.values()), 1)
        return {
            "meta": {**self.meta, "frames_processed": self._total_frames,
                     "duration_sec": round(self._total_frames/self.src_fps, 1),
                     "last_frame": self._frame_idx, "partial": True},
            "poses": poses, "timeline": self._timeline,
            "roles": {r: {"frames": n, "pct": round(100.*n/total_r, 1)} for r,n in self._role_frames.items()},
            "fence_breaches": dict(self._fence_breaches),
        }

    def flush(self, partial=True):
        try:
            data = self._build(); data["meta"]["partial"] = partial
            with open(self.out_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        except Exception as e: log.warning("SummaryLog.flush failed: %s", e)

    def close(self):
        self.flush(partial=False)
        log.info("Summary → %s  (%d poses, %d frames)", self.out_path, len(self._counts), self._total_frames)


# =============================================================================
# PIPELINE THREADS
# =============================================================================

def _reader_thread(cap, start_frame, end_frame, raw_q, max_q_size, skip_counters):
    """Read frames from cap and push them into raw_q.

    Adaptive backpressure strategy:
      - NORMAL:   blocking put() — reader pauses when queue is full, zero frames dropped.
      - PRESSURE: if the queue is already at capacity when we're about to put(), the
                  detector is running slow. We attempt a non-blocking put; if it fails
                  we skip this frame (count it) and immediately decode the next one.
                  This is the old-w7 skip behaviour, preserved for high-fps sources
                  where falling behind would OOM before the detector catches up.

    Frame index is taken from CAP_PROP_POS_FRAMES (authoritative decoder position)
    rather than a local counter so keyframe-snap drift doesn't accumulate silently.
    cap.isOpened() is checked each iteration so a mid-stream capture failure exits
    cleanly without spinning on failed reads.
    """
    frame_idx = start_frame; skipped = 0; total_put = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Authoritative frame index — absorbs any keyframe-seek drift.
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_idx >= end_frame: break

            # Adaptive put: try non-blocking first when queue is near-full.
            # If it would block (queue full), skip this frame to shed load.
            if raw_q.qsize() >= max_q_size:
                try:
                    raw_q.put_nowait((frame_idx, frame))
                    total_put += 1
                except Exception:
                    # Queue genuinely full — skip frame under pressure.
                    skipped += 1
                    skip_counters["skipped"] = skipped
                    skip_counters["total"]   = total_put + skipped
                    continue
            else:
                # Normal path: block until a slot opens (backpressure, no drops).
                raw_q.put((frame_idx, frame))
                total_put += 1

            skip_counters["total"] = total_put + skipped
    finally:
        raw_q.put(_SENTINEL)

    if skipped:
        log.warning("[Reader] Dropped %d/%d frames under queue pressure (%.1f%%)",
                    skipped, total_put + skipped, 100 * skipped / max(total_put + skipped, 1))
    log.info("[Reader] Done. %d frames enqueued.", total_put)


def _detector_thread(raw_q, ann_q, yolo, tracker, device, conf_thresh, iou_thresh,
                     kpt_conf_min, max_persons, yolo_w, yolo_h, crowd_k, n_detect, src_w, src_h,
                     src_fps=30.0):
    """
    Crash-isolated detector loop.

    Dual-layer wrapper:
      - INNER try/except around the YOLO forward pass only — a single bad
        frame (corrupt decode, CUDA hiccup, NaNs in keypoints) falls back to
        the previous frame's detections (last_detections) instead of
        aborting the whole pipeline.
      - OUTER try/except around the entire per-frame body — anything else
        unexpected (tracker.update raising, resize failure, etc.) is logged
        with frame_idx/timestamp_sec and that single frame is dropped from
        ann_q so the writer thread never sees it, but the loop continues
        with the next pre-fetched frame.

    The sentinel is always forwarded on raw_q exhaustion so the writer
    thread can still shut down cleanly even if every frame errors out.
    """
    last_detections = []; lfc = 0
    while True:
        item = raw_q.get()
        if item is _SENTINEL: ann_q.put(_SENTINEL); break
        frame_idx, frame = item; lfc += 1
        ts = frame_idx / max(src_fps, 1.0)
        try:
            yolo_frame = cv2.resize(frame, (yolo_w, yolo_h), interpolation=cv2.INTER_LINEAR)
            run_yolo = (lfc % n_detect == 1) or (not last_detections)
            if run_yolo:
                try:
                    res = yolo(yolo_frame, conf=conf_thresh, iou=iou_thresh,
                               device=device, max_det=max_persons, verbose=False)[0]
                    detections = []
                    if res.keypoints is not None and len(res.keypoints) > 0:
                        for pi in range(len(res.keypoints)):
                            kpd = res.keypoints[pi]; xy = kpd.xy.cpu().numpy()[0]; ck = kpd.conf.cpu().numpy()[0]
                            kpts_norm = np.zeros((NUM_KP, 2), dtype=np.float32)
                            kpts_norm[:,0] = xy[:,0]/yolo_w; kpts_norm[:,1] = xy[:,1]/yolo_h
                            if res.boxes is not None and pi < len(res.boxes):
                                bx = res.boxes[pi].xyxyn.cpu().numpy()[0]
                                bbox = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
                            else:
                                xs = kpts_norm[ck>kpt_conf_min, 0]; ys = kpts_norm[ck>kpt_conf_min, 1]
                                bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else [0,0,1,1]
                            ck_clean = ck.copy(); ck_clean[ck < kpt_conf_min] = 0.; kpts_norm[ck < kpt_conf_min] = 0.
                            detections.append({"bbox": bbox, "kpts": kpts_norm, "conf": ck_clean})
                    last_detections = detections
                except Exception as e:
                    log.warning("[Detector] YOLO inference failed at frame %d (t=%.3fs): %s — "
                                "reusing previous detections", frame_idx, ts, e)
                    detections = last_detections
            else:
                detections = last_detections
            crowd_scale = 1. / (1. + crowd_k * max(0, len(detections)-1))
            tracked, expired_ids = tracker.update(detections, crowd_scale=crowd_scale)
            ann_q.put((frame_idx, frame, tracked, expired_ids))
        except Exception as e:
            log.error("[Detector] Skipping frame %d (t=%.3fs) — unhandled error: %s",
                      frame_idx, ts, e)
            continue


def _writer_thread(
    ann_q, write_frame_fn, lstm, device,
    window_size, infer_every, disabled_classes, show_scores,
    src_fps, kpt_conf_min, vision_mode, display_preview,
    src_w, src_h, tracker, preview_width, json_log,
    infer_counter_ref, skip_counters,
    flag_log=None, log_panel_position=1, log_obs_window_sec=5.,
    log_panel_show=True, summary_log=None,
    role_tracker=None, fences=None,
    # v7: pulse/delta/emerging
    emerging_queue=None,
    delta_detector=None,
    min_composite_score=0.40,
):
    t0 = time.perf_counter(); proc_frames = 0

    # Per-track LSTM state (v7: no event-trigger gate)
    lstm_ran_for: set[int] = set()

    # Per-track composite score weights (simplified — same as phase1 defaults)
    W_DET = 0.25; W_POSE = 0.30; W_ACT = 0.35; W_TRK = 0.10

    frame_count = 0
    last_log_time = time.perf_counter()

    while True:
        try:
            item = ann_q.get(timeout=6.0)  # generous timeout
            if item is _SENTINEL:
                log.info("[writer] Sentinel received → shutting down")
                break

            frame_idx, frame, tracked, expired_ids = item
            proc_frames += 1
            ts = frame_idx / max(src_fps, 1.0)

            # Expire tracks in delta detector + emerging queue
            for tid in expired_ids:
                if delta_detector: delta_detector.expire(tid)
                if emerging_queue: emerging_queue.expire(tid)
                lstm_ran_for.discard(tid)
                infer_counter_ref.pop(tid, None)

            # ── LSTM inference — fixed cadence, NO event-trigger gate ─────────────
            if vision_mode != 0 and lstm is not None:
                for person in tracked:
                    tid      = person["track_id"]
                    pos_buf  = person["pos_buf"]
                    conf_buf = person["conf_buf"]
                    bbox_buf = person.get("bbox_buf", [])
                    infer_counter_ref[tid] = infer_counter_ref.get(tid, 0) + 1
                    if (len(pos_buf) >= window_size
                            and infer_counter_ref[tid] % infer_every == 0):
                        try:
                            seq  = _build_sequence(
                                list(pos_buf), list(conf_buf),
                                list(bbox_buf) if bbox_buf else None)
                            pred = predict_sequence(lstm, seq, device=device,
                                                    disabled_classes=disabled_classes)
                            if pred is not None:
                                tracker.set_prediction(tid, pred)
                                lstm_ran_for.add(tid)
                        except Exception as e:
                            log.debug("LSTM skip %d: %s", tid, e)

            # ── Pulse / delta / emerging ──────────────────────────────────────────
            if delta_detector is not None and emerging_queue is not None:
                for person in tracked:
                    tid       = person["track_id"]
                    bbox      = person["bbox"]
                    pred      = person.get("last_pred", {})
                    action    = pred.get("action", "stand") if vision_mode != 0 \
                                else (pure_vision_flags(
                                        person["kpts_raw"] if person.get("kpts_raw") is not None
                                        else np.zeros((NUM_KP,2), dtype=np.float32),
                                        person["conf"] if person.get("conf") is not None
                                        else np.zeros(NUM_KP, dtype=np.float32),
                                        kpt_conf_min) or ["stand"])[0].lower()
                    action_conf  = pred.get("confidence", 0.) if vision_mode != 0 else 0.25
                    action_valid = (tid in lstm_ran_for) if vision_mode != 0 else False

                    # Build a lightweight composite score for candidate gate
                    # detection_conf: use mean YOLO conf from kp_conf as proxy
                    kp_conf = person.get("conf")
                    n_vis = int((kp_conf > kpt_conf_min).sum()) if kp_conf is not None else 0
                    detection_conf = float(kp_conf[kp_conf > kpt_conf_min].mean()) if n_vis > 0 else 0.0
                    pose_valid = n_vis >= 10
                    mean_oks   = person.get("mean_oks", 0.)
                    tracking_valid = mean_oks >= 0.10

                    terms: list[tuple[float, float]] = [(W_DET, detection_conf), (W_POSE, detection_conf)]
                    if action_valid: terms.append((W_ACT, action_conf))
                    if tracking_valid: terms.append((W_TRK, mean_oks))
                    w_total = sum(ww for ww, _ in terms)
                    composite = sum(ww*vv for ww,vv in terms) / max(w_total, 1e-8) if pose_valid else 0.0

                    # Centroid motion vector
                    cx = (bbox[0]+bbox[2])/2.; cy = (bbox[1]+bbox[3])/2.
                    cont = delta_detector._cont.get(tid)
                    if cont and cont.last_centroid != (0.0, 0.0):
                        dx = cx - cont.last_centroid[0]; dy = cy - cont.last_centroid[1]
                    else:
                        dx, dy = 0.0, 0.0

                    reason, _meta = delta_detector.detect(
                        track_id     = tid,
                        timestamp    = ts,
                        bbox         = bbox,
                        action_top   = action,
                        action_conf  = action_conf,
                        action_valid = action_valid,
                    )
                    if reason is not None:
                        # Build action_dist dict from class_scores or uniform prior
                        scores = pred.get("class_scores", {})
                        if scores and tid in lstm_ran_for:
                            total_s = sum(scores.values()) + 1e-9
                            action_dist = {k: round(v/total_s, 4) for k, v in scores.items()
                                           if k not in (disabled_classes or [])}
                        else:
                            enabled = [a for a in ACTION_CLASSES if a not in (disabled_classes or [])]
                            action_dist = {a: round(1./len(enabled), 4) for a in enabled} if enabled else {}

                        candidate = _build_candidate(
                            track_id       = tid,
                            frame_number   = frame_idx,
                            timestamp      = ts,
                            reason         = reason,
                            bbox           = bbox,
                            action_top     = action,
                            action_top_conf = action_conf,
                            action_dist    = action_dist,
                            composite      = composite,
                            action_valid   = action_valid,
                            motion_vec     = (dx, dy),
                            min_composite  = min_composite_score,
                        )
                        if candidate is not None:
                            emerging_queue.write(candidate)

            # ── Role inference ────────────────────────────────────────────────────
            annotated = frame.copy()
            person_roles: dict[int, str] = {}
            if role_tracker is not None:
                active_ids = set()
                for person in tracked:
                    if person.get("kpts_raw") is None: continue
                    tid  = person["track_id"]; active_ids.add(tid)
                    pred = person.get("last_pred", {})
                    action = pred.get("action", "stand") if vision_mode != 0 \
                             else (pure_vision_flags(person["kpts_raw"], person["conf"], kpt_conf_min) or ["stand"])[0].lower()
                    bb = person["bbox"]; cx = (bb[0]+bb[2])/2.; cy = (bb[1]+bb[3])/2.
                    role = role_tracker.update(tid, action, cx, cy)
                    person_roles[tid] = role
                    if json_log is not None:
                        p = person.get("last_pred", {})
                        if p.get("action","?") != "?":
                            json_log.append({
                                "frame": frame_idx, "track_id": tid,
                                "action": p.get("action","?"),
                                "confidence": round(p.get("confidence",0.), 3),
                                "mean_oks": round(person.get("mean_oks",0.), 3),
                                "ts": round(frame_idx/src_fps, 2),
                                "role": role, "bbox": [round(v,4) for v in bb],
                            })
                role_tracker.prune(active_ids)

            # ── Fence check ───────────────────────────────────────────────────────
            fence_persons = [{"bbox": p["bbox"],
                              "role": person_roles.get(p["track_id"],"unknown"),
                              "track_id": p["track_id"]}
                             for p in tracked if p.get("bbox") is not None]
            breaches = []
            if fences:
                pulse = bool((frame_idx // int(src_fps)) % 2 == 0)
                breaches = draw_fences(annotated, fences, fence_persons, pulse=pulse)
                if json_log is not None:
                    for b in breaches:
                        json_log.append({
                            "frame": frame_idx, "ts": round(frame_idx/src_fps, 2),
                            "type": "fence_breach", "fence": b["fence"],
                            "track_id": b["track_id"], "role": b["role"],
                            "cx": b["cx"], "cy": b["cy"],
                        })

            # ── Flag counts ───────────────────────────────────────────────────────
            _need_counts = (flag_log is not None and log_panel_show) or (summary_log is not None)
            frame_flag_counts = {}; role_counts = {"employee":0,"customer":0,"unknown":0}
            if _need_counts:
                for person in tracked:
                    if person.get("kpts_raw") is None: continue
                    if vision_mode == 0:
                        for flg in pure_vision_flags(person["kpts_raw"], person["conf"], kpt_conf_min):
                            frame_flag_counts[flg] = frame_flag_counts.get(flg, 0) + 1
                    else:
                        action = person.get("last_pred",{}).get("action","stand")
                        frame_flag_counts[action] = frame_flag_counts.get(action, 0) + 1
                    r = person_roles.get(person["track_id"], "unknown")
                    role_counts[r] = role_counts.get(r, 0) + 1
            if flag_log is not None and log_panel_show: flag_log.update(frame_idx, frame_flag_counts)
            if summary_log is not None: summary_log.update(frame_idx, frame_flag_counts, role_counts, breaches)

            # ── Draw persons ──────────────────────────────────────────────────────
            n_emerging = 0
            for person in tracked:
                if person.get("kpts_raw") is None: continue
                kp = person["kpts_raw"]; cf = person["conf"]; bb = person["bbox"]
                tid = person["track_id"]; oks = person.get("mean_oks", 0.)
                role = person_roles.get(tid, "unknown")
                cand = emerging_queue.get_recent(tid) if emerging_queue else None
                has_emerging = cand is not None
                if has_emerging: n_emerging += 1
                draw_person(
                    annotated, bb, kp, cf, tid,
                    person.get("last_pred", {"action":"?","confidence":0.}),
                    show_scores, kpt_conf_min,
                    person.get("vision_conf", 0.),
                    person.get("vision_mode", "inference" if vision_mode == 1 else "live"),
                    person.get("kpts_live"), person.get("conf_live"),
                    dual_skeleton=(vision_mode == 3),
                    mean_oks=oks, role=role,
                    has_emerging=has_emerging,
                )

            # ── HUD + panel ───────────────────────────────────────────────────────
            total = skip_counters.get("total", 1); skip_rate = skip_counters.get("skipped", 0) / max(total, 1)
            _elapsed = time.perf_counter() - t0; _proc_fps = proc_frames / max(_elapsed, 1e-6)
            draw_hud(annotated, frame_idx, src_fps, len(tracked), vision_mode,
                     ann_q.qsize(), skip_rate, proc_fps=_proc_fps, n_emerging=n_emerging)
            if flag_log is not None and log_panel_show:
                draw_flag_log(annotated, flag_log, position=log_panel_position, obs_sec=log_obs_window_sec)
            write_frame_fn(annotated)

            frame_count += 1
            if frame_count % 25 == 0 or (time.perf_counter() - last_log_time > 3.0):
                log.info(f"[Writer] ✅ Wrote frame {frame_count} | ts={ts:.2f}s | ann_q={ann_q.qsize()}")
                last_log_time = time.perf_counter()
            ann_q.task_done()

            if display_preview:
                pw = min(src_w, preview_width); ph = int(src_h*pw/src_w)
                cv2.imshow("v7 Preview (Q to quit)", cv2.resize(annotated, (pw, ph)))
                if cv2.waitKey(1) & 0xFF == ord("q"): break

            if proc_frames % 100 == 0:
                log.info("Writer: frame %d  (%.1f fps proc)  candidates=%d",
                         frame_idx, proc_frames/max(time.perf_counter()-t0, 1e-6),
                         emerging_queue.count() if emerging_queue else 0)

        except queue.Empty:
            if not (t_reader is None or t_reader.is_alive() or t_detect is None or t_detect.is_alive()):
                log.info("[writer] Upstream threads dead + queue empty → exiting")
                break
            continue

        except Exception as e:
            log.error(f"[writer] Critical error: {e}", exc_info=True)
            break


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_pipeline(
    input_path, model_path, output_path="output_v7.mp4",
    yolo_model="yolo26n-pose.pt", device="cpu",
    conf_thresh=0.25, iou_thresh=0.70, kpt_conf_min=0.25,
    window_size=16, infer_every=1, show_scores=False, max_persons=10,
    hidden_size=128, num_layers=2, display_preview=False,
    start_sec=0., end_sec=0., yolo_input_size=640,
    disabled_classes=None, confidence_rate=2., vision_mode=2,
    n_detect=5, oks_alpha=0.5, raw_q_size=32, ann_q_size=16,
    json_sidecar=True,
    # role inference
    role_map=None, employee_zones=None,
    role_loco_thresh=0.35, role_static_thresh=0.45,
    # virtual fencing
    fences=None,
    # v7 pulse/delta/emerging
    spatial_delta_threshold=0.025,
    temporal_pulse_sec=5.0,
    action_shift_min_conf=0.55,
    min_composite_score=0.40,
) -> None:
    device = resolve_device(device)
    from ultralytics import YOLO
    yolo = YOLO(yolo_model)
    if not _SCIPY_OK: log.warning("scipy missing — SG smoothing disabled. pip install scipy")

    lstm = None
    if vision_mode != 0:
        lstm = load_lstm_model(model_path, hidden_size=hidden_size, num_layers=num_layers, device=device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {input_path}")
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_sec * src_fps)
    end_frame   = int(end_sec   * src_fps) if end_sec > 0 else total_frm
    if start_frame > 0:
        if _SEEK_UTILS_OK:
            # CV precision seek: corrects keyframe snap + overshot drift automatically.
            _seek_ok, _peek_frame, _actual = _seek_to_frame(cap, start_frame)
            if not _seek_ok:
                raise RuntimeError(f"seek_utils.seek_to_frame failed seeking to frame {start_frame}")
            log.info("[Seek] seek_utils landed on frame %d (wanted %d)", _actual, start_frame)
            # Rewind to the confirmed frame so _reader_thread starts from there.
            cap.set(cv2.CAP_PROP_POS_FRAMES, _actual)
        else:
            # Inline fallback (original logic) when seek_utils is absent.
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _peek_ok, _peek_frame = cap.read()
            _actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if _peek_ok and _actual < start_frame:
                log.warning("[Seek] Decoder drifted to frame %d (wanted %d) — stepping forward",
                            _actual, start_frame)
                while _actual < start_frame:
                    _peek_ok, _peek_frame = cap.read()
                    if not _peek_ok:
                        raise RuntimeError(f"Seek correction failed: ran out of frames before {start_frame}")
                    _actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                log.info("[Seek] Corrected to frame %d", _actual)
            elif _peek_ok and _actual > start_frame:
                log.warning("[Seek] Decoder jumped past target (frame %d > %d) — accuracy compromised",
                            _actual, start_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, _actual)
    scale  = yolo_input_size / max(src_w, src_h)
    yolo_w = max(32, (int(src_w*scale)//32)*32) if scale < 1. else src_w
    yolo_h = max(32, (int(src_h*scale)//32)*32) if scale < 1. else src_h
    log.info("v7 pipeline: %s  (%dx%d @ %.1f fps)  window=%d infer_every=%d",
             input_path, src_w, src_h, src_fps, window_size, infer_every)

    output_path = str(Path(output_path).with_suffix(".mp4"))
    ffmpeg_cmd = ["ffmpeg","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
                  "-s",f"{src_w}x{src_h}","-r",str(src_fps),"-i","pipe:0",
                  "-vcodec","libx264","-preset","veryfast","-crf","23",
                  "-pix_fmt","yuv420p","-movflags","+faststart",output_path]
    try:
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        cv_writer = None; log.info("ffmpeg pipe → %s", output_path)
    except FileNotFoundError:
        log.warning("ffmpeg not found — OpenCV fallback")
        ffmpeg_proc = None
        cv_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))

    def write_frame(bgr):
        """Safe frame writer with fallback"""
        global ffmpeg_proc, cv_writer
        if ffmpeg_proc:
            try:
                if not bgr.flags.get("C_CONTIGUOUS", False):
                    bgr = np.ascontiguousarray(bgr)
                ffmpeg_proc.stdin.write(bgr.tobytes())
                # ffmpeg_proc.stdin.flush()  # uncomment only if needed
            except (BrokenPipeError, IOError, ValueError) as e:
                log.error(f"[ffmpeg] Pipe broken: {e}")
                ffmpeg_proc = None  # disable ffmpeg
        else:
            if cv_writer is not None:
                cv_writer.write(bgr)

    tracker = PersonTracker(
        window_size=window_size, iou_thresh=0.30, max_age=30,
        src_fps=src_fps, src_w=src_w, src_h=src_h,
        confidence_rate=confidence_rate,
        vision_mode=1 if vision_mode in (0, 1) else vision_mode,
        awareness=CONFIG.get("awareness"), oks_alpha=oks_alpha,
    )

    # v7: pulse / delta / emerging
    emerging_queue = EmergingQueue()
    delta_detector = DeltaDetector(
        spatial_delta_threshold = spatial_delta_threshold,
        temporal_pulse_sec      = temporal_pulse_sec,
        action_shift_min_conf   = action_shift_min_conf,
    )

    role_tracker = RoleTracker(
        manual_map     = role_map or {},
        employee_zones = employee_zones or [],
        loco_thresh    = role_loco_thresh,
        static_thresh  = role_static_thresh,
    )

    crowd_k    = CONFIG.get("awareness", {}).get("crowd_k", 0.15)
    json_log: list | None = [] if json_sidecar else None

    _log_panel_show = CONFIG.get("log_panel_show", True)
    _log_panel_pos  = CONFIG.get("log_panel_position", 1)
    _log_obs_sec    = CONFIG.get("log_obs_window_sec", 5.)
    flag_log = FlagLog(obs_window_sec=_log_obs_sec, fps=src_fps) if _log_panel_show else None

    summary_log = SummaryLog(
        output_path=output_path, src_fps=src_fps,
        checkpoint_frames=CONFIG.get("summary_checkpoint_frames", 300),
        meta={
            "input": input_path, "output": output_path, "vision_mode": vision_mode,
            "yolo_model": yolo_model, "src_w": src_w, "src_h": src_h, "src_fps": src_fps,
            "window_size": window_size, "infer_every": infer_every,
            # time.py fields — pre-process video info
            "input_total_frames":   total_frm,
            "input_duration_sec":   round(total_frm / max(src_fps, 1.0), 3),
            "start_sec":            start_sec,
            "end_sec":              end_sec if end_sec > 0 else round(total_frm / max(src_fps, 1.0), 3),
            "segment_duration_sec": round((end_frame - start_frame) / max(src_fps, 1.0), 3),
        },
    )

    # ── Dynamic queue cap ─────────────────────────────────────────────────────
    # main.py computes a memory-safe frame cap and forwards it via the env var
    # POSE_MAX_QUEUE_FRAMES rather than as a CLI flag (avoids argparse rejection).
    # Apply it here; fall back to the default safe sizes if the var is absent.
    _env_cap = os.environ.get("POSE_MAX_QUEUE_FRAMES", "").strip()
    if _env_cap.isdigit():
        raw_q_size = max(8, min(96, int(_env_cap)))
        log.info("[queue] POSE_MAX_QUEUE_FRAMES=%d applied to raw_q", raw_q_size)
    else:
        raw_q_size = 96
        log.info("[queue] raw_q=%d (default)", raw_q_size)

    infer_counter_ref = {}; skip_counters = {"skipped": 0, "total": 0}

    _env_ann = os.environ.get("POSE_ANN_QUEUE_SIZE", "").strip()
    if _env_ann.isdigit():
        ann_q_size = max(8, min(48, int(_env_ann)))
        log.info("[queue] POSE_ANN_QUEUE_SIZE=%d applied to ann_q", ann_q_size)
    else:
        ann_q_size = 48
        log.info("[queue] ann_q=%d (default)", ann_q_size)

    raw_q = queue.Queue(maxsize=raw_q_size)
    ann_q = queue.Queue(maxsize=ann_q_size)

    t_reader = threading.Thread(
        target=_reader_thread,
        args=(cap, start_frame, end_frame, raw_q, raw_q_size, skip_counters),
        daemon=True, name="v7-Reader")
    t_detect = threading.Thread(
        target=_detector_thread,
        args=(raw_q, ann_q, yolo, tracker, device, conf_thresh, iou_thresh,
              kpt_conf_min, max_persons, yolo_w, yolo_h, crowd_k, n_detect, src_w, src_h,
              src_fps),
        daemon=True, name="v7-Detector")
    t_writer = threading.Thread(
        target=_writer_thread,
        args=(ann_q, write_frame, lstm, device, window_size, infer_every,
              disabled_classes, show_scores, src_fps, kpt_conf_min, vision_mode,
              display_preview, src_w, src_h, tracker,
              CONFIG.get("preview_width", 960), json_log,
              infer_counter_ref, skip_counters),
        kwargs=dict(
            flag_log=flag_log, log_panel_position=_log_panel_pos,
            log_obs_window_sec=_log_obs_sec, log_panel_show=_log_panel_show,
            summary_log=summary_log,
            role_tracker=role_tracker, fences=fences or [],
            emerging_queue=emerging_queue,
            delta_detector=delta_detector,
            min_composite_score=min_composite_score,
            t_reader=t_reader, t_detect=t_detect,
        ),
        daemon=True, name="v7-Writer")

    log.info("3-thread v7 pipeline starting…")
    t0 = time.perf_counter()

    t_reader.start(); t_detect.start(); t_writer.start()
    try:
        t_reader.join(timeout=180)   # 3 minutes safety
        t_detect.join(timeout=180)
        t_writer.join(timeout=60)
    except KeyboardInterrupt:
        log.info("[Pipeline] KeyboardInterrupt received")
    finally:
        # Force shutdown
        for q in (raw_q, ann_q):
            try:
                q.put(_SENTINEL, timeout=1)
            except Exception:
                pass

    cap.release()
    if ffmpeg_proc: ffmpeg_proc.stdin.close(); ffmpeg_proc.wait(); log.info("ffmpeg done.")
    else: cv_writer.release()
    if display_preview: cv2.destroyAllWindows()

    elapsed    = time.perf_counter() - t0
    total_fr   = skip_counters.get("total", 1)
    skipped    = skip_counters.get("skipped", 0)
    n_cands    = emerging_queue.count()
    log.info("Done.  %.1fs  |  frames:%d  skipped:%d (%.1f%%)  candidates:%d",
             elapsed, total_fr, skipped, 100*skipped/max(total_fr,1), n_cands)
    summary_log.close()

    if json_log:
        jp = str(Path(output_path).with_suffix(".events.json"))
        with open(jp, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "input": input_path, "output": output_path,
                    "vision_mode": vision_mode,
                    "src_fps": src_fps, "src_w": src_w, "src_h": src_h,
                    "input_total_frames":   total_frm,
                    "input_duration_sec":   round(total_frm / max(src_fps, 1.0), 3),
                    "start_sec":            start_sec,
                    "end_sec":              end_sec if end_sec > 0 else round(total_frm / max(src_fps, 1.0), 3),
                    "segment_duration_sec": round((end_frame - start_frame) / max(src_fps, 1.0), 3),
                    "n_events": len(json_log),
                },
                "events": json_log,
            }, f, indent=2)
        log.info("Events JSON → %s  (%d events)", jp, len(json_log))

    # v7: candidates sidecar
    candidates = emerging_queue.snapshot_all()
    if candidates:
        cp = str(Path(output_path).with_suffix(".candidates.json"))

        # ── seek_map output-time scaling ──────────────────────────────────────
        # Candidate timestamp_sec values are in INPUT-video time.
        # The frontend <video> plays the OUTPUT video, whose duration may differ
        # (e.g. frame_skip / segment trimming).  Rescale here so heatmap
        # builders receive already-correct output timestamps and seek_map
        # entries land on the right position without a second pass.
        input_duration_sec  = round(total_frm / max(src_fps, 1.0), 3)
        segment_duration_sec = round((end_frame - start_frame) / max(src_fps, 1.0), 3)

        def _to_output_ts(input_ts: float) -> float:
            """Rescale a single input-time timestamp to output-video time."""
            if _SEEK_UTILS_OK and segment_duration_sec > 0:
                # Use seek_utils.scale_seek_map logic directly on a single value.
                ratio = segment_duration_sec / max(input_duration_sec, 1e-6)
                return round(input_ts * ratio, 3)
            return input_ts  # no-op if seek_utils absent or durations unknown

        candidates_out = []
        for c in candidates:
            d = c.to_dict()
            d["timestamp_sec"] = _to_output_ts(d["timestamp_sec"])
            candidates_out.append(d)

        with open(cp, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "input": input_path, "output": output_path,
                    "vision_mode": vision_mode,
                    "src_fps": src_fps, "src_w": src_w, "src_h": src_h,
                    "input_total_frames":    total_frm,
                    "input_duration_sec":    input_duration_sec,
                    "start_sec":             start_sec,
                    "end_sec":               end_sec if end_sec > 0 else input_duration_sec,
                    "segment_duration_sec":  segment_duration_sec,
                    "window_size": window_size, "infer_every": infer_every,
                    "n_candidates": n_cands,
                    "seek_map_scaled": _SEEK_UTILS_OK,   # diagnostic flag
                },
                "candidates": candidates_out,
            }, f, indent=2)
        log.info("Candidates JSON → %s  (%d candidates%s)", cp, n_cands,
                 ", seek_map scaled to output-time" if _SEEK_UTILS_OK else "")

    # ── time.py: duration report + timestamps ─────────────────────────────────
    # Runs automatically after all sidecars are flushed.
    # Writes <output_stem>.timestamps.json alongside the other sidecars.
    _time_py = Path(__file__).with_name("time.py")
    if _time_py.exists():
        _time_cmd = [
            sys.executable, str(_time_py),
            "--input",  input_path,
            "--output", output_path,
            "--auto",                   # auto-derive summary/events/candidates paths
            "--write-timestamps",       # write <stem>.timestamps.json
        ]
        log.info("time.py → %s", _time_py)
        try:
            subprocess.run(_time_cmd, check=True)
        except subprocess.CalledProcessError as _e:
            log.warning("time.py exited with code %d (non-fatal)", _e.returncode)
        except Exception as _e:
            log.warning("time.py failed to run: %s (non-fatal)", _e)
    else:
        log.info("time.py not found at %s — skipping duration/timestamp report", _time_py)


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "input_path":  r"D:\Download_path\15574390_3840_2160_50fps.mp4",
    "output_path": r"C:\Users\ASUS\Videos\v7_output.mp4",

    "vision_mode": 2,   # 0=pure vision  1=LSTM basic  2=LSTM+thresh  3=combined

    "model_paths": {
        0: "",
        1: str(_ROOT / "lstm_model" / "model-1-inf.pt"),
        2: str(_ROOT / "lstm_model" / "model-2-conf.pt"),
        3: str(_ROOT / "lstm_model" / "model-3-dual.pt"),
    },

    "yolo_model":  "yolo26n-pose.pt",
    "device":      "cuda",

    "conf_thresh":  0.25,
    "iou_thresh":   0.70,
    "kpt_conf_min": 0.25,

    # v7 patched defaults (from phase1 patch v3.1)
    "window_size":  16,   # was 32 — see patch notes
    "hidden_size":  128,
    "num_layers":   2,
    "yolo_input_size": 640,
    "infer_every":  1,    # was 8 — run every frame once buffer is full

    "max_persons":     10,
    "show_scores":     False,
    "display_preview": False,
    "preview_width":   960,

    "disabled_classes": ["gun","fall","grab","hit","kick",
                         "lying_down","sneak","struggle","throw"],
    "confidence_rate": 1,

    "n_detect":   5,
    "oks_alpha":  0.5,
    "raw_q_size": 32,
    "ann_q_size": 16,
    "json_sidecar": True,
    "summary_checkpoint_frames": 300,
    "log_panel_position":  1,
    "log_obs_window_sec":  5.0,
    "log_panel_show":      True,

    # Role inference
    "role_map":          {},
    "employee_zones":    [],
    "role_loco_thresh":  0.35,
    "role_static_thresh":0.45,

    # Virtual fencing
    "fences": [],

    "start_sec": 0.0,
    "end_sec":   0.0,

    # v7 pulse/delta parameters
    "spatial_delta_threshold": 0.025,
    "temporal_pulse_sec":      5.0,
    "action_shift_min_conf":   0.55,
    "min_composite_score":     0.40,

    "awareness": {
        "vel_stationary_px":   0.5,
        "vel_drain_rate":      0.8,
        "vel_drain_max":       8.0,
        "dist_near_area":      0.04,
        "dist_far_area":       0.002,
        "dist_far_gain_scale": 0.35,
        "entropy_none":        0.35,
        "entropy_full":        0.95,
        "crowd_k":             0.15,
    },
}


# =============================================================================
# CLI
# =============================================================================

_MODE_DESCRIPTIONS = {
    0: "Pure Vision   — YOLO flags only, no model loaded",
    1: "LSTM Basic    — model-1-inf.pt, raw inference",
    2: "LSTM+Thresh   — model-2-conf.pt, confidence UI",
    3: "Combined      — model-3-dual.pt, dual skeleton",
}

def _prompt_vision_mode(default):
    print("\n  Vision modes:")
    for k, desc in _MODE_DESCRIPTIONS.items():
        print(f"    {k}  {desc}{'  ◄ default' if k == default else ''}")
    while True:
        raw = input(f"\n  Vision mode [0-3] (Enter={default}): ").strip()
        if raw == "": return default
        if raw in ("0","1","2","3"): return int(raw)
        print("  Please enter 0-3.")

def _prompt_preview(default):
    d = "Y" if default else "N"
    while True:
        raw = input(f"  Live preview? [Y/N] (Enter={d}): ").strip().upper()
        if raw == "": return default
        if raw in ("Y","YES"): return True
        if raw in ("N","NO"):  return False

def _prompt_input_path(default):
    print(f'  Default: "{default}"')
    raw = input("  Video path (Enter=default): ").strip()
    return default if raw == "" else raw.strip('"').strip("'")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="v7 CCTV Pose Pipeline — Pulse + Delta + Emerging",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input",    default=None)
    parser.add_argument("--output",   default=CONFIG["output_path"])
    parser.add_argument("--device",   default=CONFIG["device"])
    parser.add_argument("--yolo",     default=CONFIG["yolo_model"])
    parser.add_argument("--mode",     type=int, default=None)
    parser.add_argument("--model",    default=None)
    parser.add_argument("--preview",  action="store_true")
    parser.add_argument("--scores",   action="store_true")
    parser.add_argument("--start",    type=float, default=CONFIG["start_sec"])
    parser.add_argument("--end",      type=float, default=CONFIG["end_sec"])
    parser.add_argument("--yolo-size",type=int,   default=CONFIG["yolo_input_size"])
    parser.add_argument("--n-detect", type=int,   default=CONFIG["n_detect"])
    parser.add_argument("--oks-alpha",type=float, default=CONFIG["oks_alpha"])
    parser.add_argument("--window",   type=int,   default=CONFIG["window_size"])
    parser.add_argument("--infer-every", type=int, default=CONFIG["infer_every"])
    parser.add_argument("--disable",  nargs="*",  default=None, metavar="CLASS")
    parser.add_argument("--fences",   default=None, metavar="JSON")
    parser.add_argument("--no-prompt",action="store_true")
    parser.add_argument("--seek_sec", type=float, default=None,
                        help=(
                            "Precision seek: jump to this absolute source-video second and "
                            "process a 10-second window around it (overrides --start/--end). "
                            "Uses frame-index seeking with keyframe-drift correction via seek_utils."
                        ))
    args = parser.parse_args()

    # Non-interactive: no prompts ever.
    # mode=2 (LSTM+Thresh) unless overridden by --mode.
    # --preview is False by default (headless subprocess).
    vision_mode     = args.mode if args.mode is not None else 2
    display_preview = args.preview
    if args.input is None:
        print("ERROR: --input <path> is required", file=sys.stderr)
        sys.exit(1)
    input_path = args.input

    model_path = args.model if args.model else CONFIG.get("model_paths", {}).get(vision_mode, "")

    if args.disable is None:
        disabled_classes = set(CONFIG.get("disabled_classes", []))
    else:
        disabled_classes = set(args.disable)
    disabled_classes -= disabled_classes - set(ACTION_CLASSES)

    fences = CONFIG.get("fences", [])
    if args.fences:
        try:
            with open(args.fences, encoding="utf-8") as f: fences = json.load(f)
            log.info("Loaded %d fences from %s", len(fences), args.fences)
        except Exception as e: log.warning("Could not load fences file: %s", e)

    # ── Precision seek mode ───────────────────────────────────────────────────
    # --seek_sec overrides --start / --end: process a 10-second window centred
    # on the requested timestamp so the LSTM buffer fills cleanly.
    _start_sec = args.start
    _end_sec   = args.end
    if args.seek_sec is not None and args.seek_sec >= 0:
        _SEEK_WINDOW_SEC = 10.0   # seconds either side of target
        _start_sec = max(0.0, args.seek_sec - _SEEK_WINDOW_SEC / 2)
        _end_sec   = args.seek_sec + _SEEK_WINDOW_SEC / 2
        log.info(
            "[seek_sec] precision mode: target=%.3fs  window=[%.3fs, %.3fs]",
            args.seek_sec, _start_sec, _end_sec,
        )

    run_pipeline(
        input_path       = input_path,
        model_path       = model_path,
        output_path      = args.output,
        yolo_model       = args.yolo,
        device           = args.device,
        conf_thresh      = CONFIG["conf_thresh"],
        iou_thresh       = CONFIG["iou_thresh"],
        kpt_conf_min     = CONFIG["kpt_conf_min"],
        window_size      = args.window,
        infer_every      = args.infer_every,
        show_scores      = args.scores or CONFIG["show_scores"],
        max_persons      = CONFIG["max_persons"],
        hidden_size      = CONFIG["hidden_size"],
        num_layers       = CONFIG["num_layers"],
        display_preview  = display_preview,
        start_sec        = _start_sec,
        end_sec          = _end_sec,
        yolo_input_size  = args.yolo_size,
        disabled_classes = disabled_classes or None,
        confidence_rate  = float(CONFIG.get("confidence_rate", 2.)),
        vision_mode      = vision_mode,
        n_detect         = args.n_detect,
        oks_alpha        = args.oks_alpha,
        raw_q_size       = CONFIG.get("raw_q_size", 32),
        ann_q_size       = CONFIG.get("ann_q_size", 16),
        json_sidecar     = CONFIG.get("json_sidecar", True),
        role_map         = CONFIG.get("role_map", {}),
        employee_zones   = CONFIG.get("employee_zones", []),
        role_loco_thresh = CONFIG.get("role_loco_thresh", 0.35),
        role_static_thresh = CONFIG.get("role_static_thresh", 0.45),
        fences           = fences,
        spatial_delta_threshold = CONFIG.get("spatial_delta_threshold", 0.025),
        temporal_pulse_sec      = CONFIG.get("temporal_pulse_sec", 5.0),
        action_shift_min_conf   = CONFIG.get("action_shift_min_conf", 0.55),
        min_composite_score     = CONFIG.get("min_composite_score", 0.40),
    )