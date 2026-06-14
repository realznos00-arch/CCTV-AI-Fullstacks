"""
seek_utils.py — CV Precision Seek Engine
=========================================
Two complementary strategies for landing on the exact frame in a
high-fps / heavily-compressed source video:

  STRATEGY A — Frame-index seek with drift correction  (seek_to_frame)
      Uses CAP_PROP_POS_FRAMES and steps forward manually when the
      hardware decoder snaps to a keyframe instead of the exact index.
      Always available; zero extra dependencies.

  STRATEGY B — OCR Top-Left Protocol  (seek_by_top_left_clock)
      Reads the burned-in "MM:SS" timestamp from the HUD top-left of
      each frame and advances until the visual clock matches the target.
      Requires: pytesseract + Tesseract-OCR installed on PATH.
      Falls back gracefully to Strategy A if pytesseract is absent.

Both strategies are exposed through a single public entry-point:

    frame, actual_frame_idx = get_precise_frame(video_path, target_seconds, src_fps)

─────────────────────────────────────────────────────────────────────
seek_map scaling helpers (used by main.py / build_heatmap_payload)
─────────────────────────────────────────────────────────────────────
Output video duration often differs from input/source video duration
(e.g. frame_skip downsampling: 15s input → 5s output).

seek_map values are computed against SOURCE/input timestamps. Before
sending to the frontend, every seek_map must be rescaled so that
clicking a cell seeks to the correct position in the OUTPUT video
(the one actually loaded in <video>).

    output_ts = input_ts * (output_duration_sec / input_duration_sec)

If durations are not provided, ratio=1 and seek_map passes through
unchanged — backward compatible with existing callers.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import cv2

log = logging.getLogger(__name__)

# ── optional OCR import ────────────────────────────────────────────────────────
try:
    import pytesseract as _tesseract
    _OCR_AVAILABLE = True
except ImportError:
    _tesseract = None          # type: ignore[assignment]
    _OCR_AVAILABLE = False
    log.debug("pytesseract not installed — OCR strategy unavailable, using frame-index only")


# =============================================================================
# STRATEGY A — Frame-index seek with keyframe-drift correction
# =============================================================================

def seek_to_frame(
    cap: cv2.VideoCapture,
    target_frame: int,
) -> Tuple[bool, Optional["cv2.typing.MatLike"], int]:
    """
    Seek ``cap`` to ``target_frame`` and correct any keyframe snap.

    Returns
    -------
    (success, frame, actual_frame_idx)
        success          — False if the video ran out before reaching target
        frame            — the verified frame (or best-effort if overshot)
        actual_frame_idx — the frame index that was actually read
    """
    # Validate target_frame is reasonable
    if target_frame < 0:
        log.warning("[seek_to_frame] target_frame=%d is negative, clamping to 0", target_frame)
        target_frame = max(0, target_frame)
    
    # Try to get video bounds (best effort)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and target_frame >= total_frames:
        log.warning(
            "[seek_to_frame] target_frame=%d >= total=%d, clamping to last frame",
            target_frame, total_frames
        )
        target_frame = max(0, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok:
        log.error("[seek_to_frame] cap.read() failed at target_frame=%d", target_frame)
        return False, None, -1

    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # read() already advanced

    if actual == target_frame:
        log.debug("[seek_to_frame] exact match at frame %d", actual)
        return True, frame, actual

    if actual < target_frame:
        # Decoder snapped to an earlier keyframe — step forward
        log.warning(
            "[seek_to_frame] drift: decoder landed on %d (wanted %d), stepping forward",
            actual, target_frame,
        )
        while actual < target_frame:
            ok, frame = cap.read()
            if not ok:
                log.error("[seek_to_frame] ran out of frames before reaching %d", target_frame)
                return False, frame, actual
            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        log.info("[seek_to_frame] corrected to frame %d", actual)
        return True, frame, actual

    # actual > target_frame — decoder overshot (rare, broken index)
    log.warning(
        "[seek_to_frame] overshot: decoder at %d > target %d — accuracy compromised",
        actual, target_frame,
    )
    return True, frame, actual


# =============================================================================
# STRATEGY B — OCR Top-Left Clock Protocol
# =============================================================================

def _read_top_left_clock(frame: "cv2.typing.MatLike") -> Optional[str]:
    """
    Crop the top-left HUD header, run Tesseract, and return the first
    "MM:SS" string found, or None if OCR produced no match.
    """
    if not _OCR_AVAILABLE:
        return None

    h, w = frame.shape[:2]
    # Crop: top 8 % of height, left 25 % of width — adjust if your HUD differs
    crop = frame[0: max(1, int(h * 0.08)), 0: max(1, int(w * 0.25))]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Whitelist only digits and colon so Tesseract doesn't hallucinate letters
    cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789:"
    try:
        raw = _tesseract.image_to_string(thresh, config=cfg).strip()
    except Exception as exc:
        log.debug("[ocr] tesseract error: %s", exc)
        return None

    m = re.search(r"\d{2}:\d{2}", raw)
    return m.group(0) if m else None


def seek_by_top_left_clock(
    cap: cv2.VideoCapture,
    target_seconds: float,
    src_fps: float,
) -> Tuple[bool, Optional["cv2.typing.MatLike"], int]:
    """
    Jump to approximately ``target_seconds``, then walk forward frame-by-frame
    until the burned-in HUD clock matches the target MM:SS.

    A 2-second runway back-off (100 frames at 50 fps) is used to ensure we
    don't overshoot the target before OCR confirms it.

    Falls back to ``seek_to_frame`` if pytesseract is unavailable.

    Returns
    -------
    (success, frame, actual_frame_idx)
    """
    # Validate target_seconds is reasonable
    if target_seconds < 0:
        log.warning("[ocr] target_seconds=%.3f is negative, clamping to 0", target_seconds)
        target_seconds = max(0.0, target_seconds)
    
    # Get video total duration (best effort)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = (total_frames / src_fps) if src_fps > 0 else 0
    
    if total_duration > 0 and target_seconds >= total_duration:
        log.warning(
            "[ocr] target_seconds=%.3f >= video_duration=%.3f, clamping to near-end",
            target_seconds, total_duration
        )
        target_seconds = max(0.0, total_duration - 1.0)

    if not _OCR_AVAILABLE:
        log.info("[ocr] pytesseract not available — falling back to frame-index seek")
        target_frame = int(target_seconds * src_fps)
        return seek_to_frame(cap, target_frame)

    # Build expected "MM:SS" string
    mins = int(target_seconds // 60)
    secs = int(target_seconds % 60)
    expected = f"{mins:02d}:{secs:02d}"

    # Back off ~2 s to create a scanning runway
    runway_start = max(0, int(target_seconds * src_fps) - int(2.0 * src_fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, runway_start)

    log.info(
        "[ocr] scanning for clock '%s' (target=%.3fs) from frame %d",
        expected, target_seconds, runway_start,
    )

    ok, frame = cap.read()
    while ok:
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        stamp = _read_top_left_clock(frame)
        if stamp == expected:
            log.info("[ocr] clock match '%s' at frame %d", stamp, actual)
            return True, frame, actual
        # Safety: don't scan more than 5 s past the target
        if actual > int((target_seconds + 5.0) * src_fps):
            log.warning(
                "[ocr] scanned 5s past target without finding '%s' — "
                "falling back to frame-index result at frame %d",
                expected, actual,
            )
            break
        ok, frame = cap.read()

    # Fall back: re-seek with Strategy A
    log.warning("[ocr] OCR scan did not confirm target; falling back to frame-index seek")
    target_frame = int(target_seconds * src_fps)
    return seek_to_frame(cap, target_frame)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def get_precise_frame(
    video_path: str,
    target_seconds: float,
    src_fps: float = 50.0,
    use_ocr: bool = True,
) -> Tuple[bool, Optional["cv2.typing.MatLike"], int]:
    """
    Open ``video_path``, seek to ``target_seconds``, and return the verified frame.

    Strategy selection
    ------------------
    use_ocr=True  and pytesseract installed → OCR Top-Left Protocol (Strategy B)
                                              with automatic fallback to Strategy A
    use_ocr=False or pytesseract missing    → Frame-index seek only (Strategy A)

    Returns
    -------
    (success, frame, actual_frame_idx)
        success           — True if a valid frame was obtained
        frame             — the BGR frame as a numpy array, or None on failure
        actual_frame_idx  — the exact frame index that was read
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("[get_precise_frame] cannot open: %s", video_path)
        return False, None, -1

    try:
        if use_ocr and _OCR_AVAILABLE:
            return seek_by_top_left_clock(cap, target_seconds, src_fps)
        else:
            target_frame = int(target_seconds * src_fps)
            return seek_to_frame(cap, target_frame)
    finally:
        cap.release()


# =============================================================================
# seek_map SCALING HELPERS  (used by main.py / build_heatmap_payload)
# =============================================================================

def scale_seek_map(
    seek_map: dict,
    input_duration_sec: Optional[float],
    output_duration_sec: Optional[float],
) -> dict:
    """
    Rescale seek_map values from input-video time to output-video time.

    seek_map : {col_index: timestamp_sec}  (input-time)
    Returns  : {col_index: timestamp_sec}  (output-time)
    """
    if not seek_map:
        return seek_map
    if not input_duration_sec or not output_duration_sec or input_duration_sec <= 0:
        return seek_map  # no scaling info — pass through unchanged

    ratio = output_duration_sec / input_duration_sec
    return {k: round(v * ratio, 3) for k, v in seek_map.items()}


def unscale_seek_map(
    seek_map: dict,
    input_duration_sec: Optional[float],
    output_duration_sec: Optional[float],
) -> dict:
    """
    Reverse-scale seek_map values from output-video time BACK to input-video time.
    This is the INVERSE of scale_seek_map().
    
    Used by frontend: when sending seek target to /run-again, convert 
    OUTPUT-time back to INPUT-time before sending to backend.
    
    seek_map : {col_index: timestamp_sec}  (output-time)
    Returns  : {col_index: timestamp_sec}  (input-time)
    """
    if not seek_map:
        return seek_map
    if not input_duration_sec or not output_duration_sec or input_duration_sec <= 0:
        return seek_map
    
    # Reverse the ratio: input / output instead of output / input
    inverse_ratio = input_duration_sec / output_duration_sec
    return {k: round(v * inverse_ratio, 3) for k, v in seek_map.items()}


def scale_all_seek_maps(
    payload: dict,
    input_duration_sec: Optional[float],
    output_duration_sec: Optional[float],
) -> dict:
    """
    Walk a heatmap payload dict and rescale every 'seek_map' key found
    in any top-level panel value. Mutates and returns payload.
    """
    for key, val in payload.items():
        if isinstance(val, dict) and "seek_map" in val:
            val["seek_map"] = scale_seek_map(
                val["seek_map"], input_duration_sec, output_duration_sec
            )
    return payload