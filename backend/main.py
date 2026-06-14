from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import io
import uuid
import subprocess
import shutil
import os
import sys
import json
import re
import mimetypes
from pathlib import Path
from collections import defaultdict
import math
import time
from seek_utils import scale_seek_map

app = FastAPI()

# ── Processing mode time-estimate constants ────────────────────────────────────
# Calibrated against the observed 5.5 fps proc rate on a CPU-only 4K/30fps run.
# "full":       ~1 output-second per 5.5 input-seconds  → ratio ≈ 5.5
# "compressed": n_detect=5 means YOLO runs 1/5 as often, LSTM every 4th frame
#               → empirically ~3–4× faster than full → ratio ≈ 1.4
_PROC_SEC_PER_INPUT_SEC = {"full": 5.5, "compressed": 1.4}


def _estimate_minutes(video_path: str, processing_mode: str) -> float | None:
    """Return estimated processing time in minutes, or None if duration unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        dur = float(out.stdout.strip())
        ratio = _PROC_SEC_PER_INPUT_SEC.get(processing_mode, 5.5)
        return round(dur * ratio / 60, 1)
    except Exception:
        return None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── paths ──────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR    = BACKEND_DIR.parent
SCRIPT_SYNC       = ROOT_DIR / "infer-vid-w7-sync.py"       # full precision
SCRIPT_COMPRESSED = ROOT_DIR / "infer-vid-w7-compress.py"   # fast compressed
PYTHON_EXE        = Path(sys.executable)
UPLOAD_DIR   = BACKEND_DIR / "uploads"
OUTPUT_DIR   = BACKEND_DIR / "outputs"
JOBS_DIR     = BACKEND_DIR / "jobs"
MAX_ARTIFACT_AGE_DAYS = 30


def _cleanup_old_artifacts(age_days: int = MAX_ARTIFACT_AGE_DAYS) -> None:
    now = time.time()
    cutoff = now - age_days * 86400
    for root in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR):
        for path in root.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            except Exception:
                continue

for d in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR):
    d.mkdir(exist_ok=True, parents=True)
_cleanup_old_artifacts()

# serve finished videos at /outputs/<file>, with HTTP Range support so the
# <video> element can seek. StaticFiles alone was returning 200 + the full
# body for Range requests instead of 206 + Content-Range, which causes
# browsers to reset currentTime back to 0 after a seek.
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 1024 * 1024  # 1 MB


@app.get("/outputs/{filename:path}")
async def get_output_file(filename: str, request: Request):
    file_path = OUTPUT_DIR / filename
    if not file_path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)

    file_size = file_path.stat().st_size
    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header:
        match = _RANGE_RE.match(range_header)
        if match:
            start_str, end_str = match.groups()
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)

            if start >= file_size or start > end:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )

            content_length = end - start + 1

            def ranged_stream():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(_CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                ranged_stream(),
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                },
            )

    # No (usable) Range header — return the whole file, but advertise that
    # range requests are supported so the player knows seeking will work.
    def full_stream():
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        full_stream(),
        status_code=200,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


# Fallback mount (kept for any tooling that expects a StaticFiles mount at
# this path, e.g. directory listings); the route above takes precedence for
# /outputs/<file> requests since FastAPI matches routes before mounts.
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ── job state store (in-memory, good enough for single-user dev) ───────────────
JOBS: dict[str, dict] = {}


import re as _re
import threading as _threading

# ── Milestone targets — log line triggers a target jump ───────────────────────
# Progress eases toward target; never goes backward.
_MILESTONES = [
    (_re.compile(r"ffmpeg pipe",                   _re.I),  12, "opening input"),
    (_re.compile(r"3-thread.*v7|pipeline.*start",  _re.I),  25, "pipeline starting"),
    (_re.compile(r"reader.*thread|writer.*thread", _re.I),  38, "threads ready"),
    (_re.compile(r"yolo|pose|keypoint|lstm",       _re.I),  52, "inferring poses"),
    (_re.compile(r"Writer.*frame",                 _re.I),  68, "writing frames"),
    (_re.compile(r"ffmpeg done",                   _re.I),  78, "encoding output"),
    (_re.compile(r"candidate|event.*json|summary", _re.I),  88, "saving results"),
    (_re.compile(r"sqlite|storage|\.db",           _re.I),  94, "writing DB"),
    (_re.compile(r"\bDone\b.*frame|Done\.\s*\d",   _re.I),  98, "finishing"),
]

_TICK_SEC = 0.3
_CAP      = 99.0


def _get_decay(current: float) -> float:
    if current < 20:  return 0.25   # fast burst  — setup flies by
    if current < 50:  return 0.04   # slow grind  — inference drag
    if current < 65:  return 0.18   # pickup      — encoding / writing
    return 0.02                      # long taper  — crawls to 99, never quite arrives


def _run_pipeline(job_id: str, input_path: str, output_path: str,
                  seek_sec: float | None = None,
                  processing_mode: str = "full"):
    job = JOBS[job_id]
    job["status"]           = "running"
    job["log"]              = []
    job["progress"]         = 0.0
    job["phase_label"]      = "starting…"
    job["processing_mode"]  = processing_mode
    if seek_sec is not None:
        job["seek_sec"] = seek_sec   # store so /pipeline-status can echo it back

    # shared state touched by both the timer thread and the log-reader loop
    _state = {
        "current":  0.0,
        "target":   8.0,   # give the bar an initial nudge immediately
        "label":    "starting…",
        "alive":    True,
    }

    # ── Zeno timer — runs in its own thread ───────────────────────────────
    def _ticker():
        while _state["alive"]:
            c, t = _state["current"], _state["target"]
            if c < t:
                c = c + (t - c) * _get_decay(c)
                c = min(c, _CAP)
                _state["current"] = c
                job["progress"]    = round(c, 1)
                job["phase_label"] = _state["label"]
            _threading.Event().wait(_TICK_SEC)

    timer = _threading.Thread(target=_ticker, daemon=True)
    timer.start()

    # ── Pipeline subprocess ────────────────────────────────────────────────
    script_path = SCRIPT_COMPRESSED if processing_mode == "compressed" else SCRIPT_SYNC
    cmd = [
        PYTHON_EXE, script_path,
        "--input",     input_path,
        "--output",    output_path,
        "--mode",      "2",
        "--no-prompt",
    ]
    job["log"].append(f"[pipeline] processing_mode={processing_mode}  script={script_path}")
    # Precision seek: bypass full sequential read, jump to exact frame window
    if seek_sec is not None and seek_sec >= 0:
        cmd += ["--seek_sec", str(seek_sec)]

    # Compute a dynamic, conservative max queue frames cap based on input
    # resolution and available system memory. Helpful especially for 4K inputs.
    def _compute_max_queue_frames(input_path: str) -> int:
        try:
            out = subprocess.run([
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_path)
            ], capture_output=True, text=True, timeout=10)
            parts = [p for p in out.stdout.splitlines() if p.strip()]
            if len(parts) >= 2:
                w = int(parts[0]); h = int(parts[1])
            else:
                w, h = 1280, 720
        except Exception:
            w, h = 1280, 720

        # approximate raw frame bytes (3 channels RGB)
        frame_bytes = max(1, w * h * 3)

        # available memory (fallback to 8GB if psutil missing)
        try:
            import psutil as _ps
            avail = _ps.virtual_memory().available
        except Exception:
            avail = 8 * 1024 ** 3

        # target to use at most 20% of available memory for the queue buffer
        allowed = max(32 * 1024 ** 2, int(avail * 0.20))
        frames = allowed // frame_bytes
        # clamp to sane bounds
        frames = max(8, min(frames, 500))
        return int(frames)

    # Compute dynamic queue cap and store on job for diagnostics.
    # We do NOT pass this as a CLI flag because infer-vid-w7.py's argparse
    # does not declare --max_queue_frames → that was the cause of exit-code-2.
    # Instead we forward it via an environment variable (POSE_MAX_QUEUE_FRAMES)
    # so the child script can read it with os.environ.get() whenever it is ready
    # to consume it, without any argparse changes required on that side.
    child_env = os.environ.copy()
    try:
        max_queue_frames = _compute_max_queue_frames(input_path)
        job["max_queue_frames"] = max_queue_frames
        child_env["POSE_MAX_QUEUE_FRAMES"] = str(max_queue_frames)
        child_env["POSE_ANN_QUEUE_SIZE"] = str(48)
        job["log"].append(
            f"[queue] dynamic cap = {max_queue_frames} frames "
            f"(forwarded via POSE_MAX_QUEUE_FRAMES, ann_q=48)"
        )
    except Exception as _qe:
        job["log"].append(f"[queue] cap computation skipped: {_qe}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

        for line in proc.stdout:
            line = line.rstrip()
            job["log"].append(line)
            if len(job["log"]) > 100:
                job["log"] = job["log"][-100:]

            # check milestones — bump target if this line matches
            for pat, tgt, label in _MILESTONES:
                if pat.search(line):
                    if tgt > _state["target"]:
                        _state["target"] = tgt
                        _state["label"]  = label
                    break

        proc.wait()

        _state["alive"] = False   # stop the ticker

        if proc.returncode == 0 and Path(output_path).exists():
            job["status"]      = "done"
            job["progress"]    = 100
            job["phase_label"] = "complete"
            job["output_file"] = Path(output_path).name

            # ── Run time.py: duration report + timestamps sidecar ─────────────
            # Writes <output_stem>.timestamps.json alongside the other sidecars.
            _time_py = ROOT_DIR / "time.py"
            if _time_py.exists():
                _time_cmd = [
                    PYTHON_EXE, str(_time_py),
                    "--input",  input_path,
                    "--output", output_path,
                    "--auto",
                    "--write-timestamps",
                ]
                try:
                    subprocess.run(_time_cmd, timeout=60, check=False)
                    job["log"].append("[time.py] duration + timestamps written")
                except Exception as _te:
                    job["log"].append(f"[time.py] skipped: {_te}")
        else:
            job["status"] = "error"
            job["error"]  = f"Process exited with code {proc.returncode}"

    except Exception as e:
        _state["alive"] = False
        job["status"]   = "error"
        job["error"]    = str(e)


# ── Heatmap data generator ─────────────────────────────────────────────────────

# Canonical action name mapping (same as heatmap-pose2.py)
_CANONICAL: dict[str, str] = {
    "stand": "Stand/Idle", "Standing / Idle": "Stand/Idle", "IDLE": "Stand/Idle",
    "sit": "Sit", "Sitting": "Sit", "SITTING": "Sit",
    "lying_down": "Lying Down", "Lying Down": "Lying Down", "LYING": "Lying Down",
    "Crouching": "Crouch", "CROUCHING": "Crouch",
    "Leaning": "Lean", "LEANING": "Lean",
    "run": "Run", "Running": "Run", "RUNNING": "Run",
    "walk": "Walk", "Walking": "Walk", "WALKING": "Walk",
    "sneak": "Sneak", "Sneaking": "Sneak",
    "fall": "Fall", "Falling": "Fall", "FALLING": "Fall",
    "grab": "Grab", "Grabbing": "Grab",
    "hit": "Hit", "Hitting": "Hit",
    "kick": "Kick", "Kicking": "Kick",
    "throw": "Throw", "Throwing": "Throw",
    "struggle": "Struggle", "Struggling": "Struggle",
    "gun": "Gun",
}

_POSE_GROUPS = {
    "Stand/Idle": "static", "Sit": "static", "Lying Down": "static",
    "Crouch": "static", "Lean": "static",
    "Run": "locomotion", "Walk": "locomotion", "Sneak": "locomotion",
    "Fall": "action", "Grab": "action", "Hit": "action", "Kick": "action",
    "Throw": "action", "Struggle": "action", "Gun": "action",
}

_POSE_RAMP = ["#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"]


def _canon(k: str) -> str:
    return _CANONICAL.get(k, k)


def _ffprobe_duration_sec(path: Path) -> float | None:
    """Return media duration in seconds via ffprobe, or None on failure."""
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrapper=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        val = out.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


def _read_summary(job_id: str) -> dict | None:
    """Read the summary.json for a completed job."""
    sp = OUTPUT_DIR / f"{job_id}_output.summary.json"
    if not sp.exists():
        return None
    with open(sp, encoding="utf-8") as f:
        return json.load(f)


def _read_events(job_id: str) -> list[dict]:
    """Read events from events.json for a completed job."""
    ep = OUTPUT_DIR / f"{job_id}_output.events.json"
    if not ep.exists():
        return []
    with open(ep, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])


def _build_pose_timeline_heatmap(summary: dict) -> dict:
    """
    Build a pose × time heatmap from summary timeline data.
    Rows = canonical pose names, cols = time bins, values = person count.
    """
    timeline = summary.get("timeline", [])
    if not timeline:
        return {"type": "grid", "title": "Pose activity over time", "rows": [], "cols": [], "values": [],
                "legend": ["Low", "High"], "ramp": _POSE_RAMP}

    # Collect all poses across all snapshots
    all_poses_set: set[str] = set()
    for snap in timeline:
        for k in snap.get("counts", {}):
            cn = _canon(k)
            if cn != "?":
                all_poses_set.add(cn)

    # Sort by group then name
    def _pose_sort_key(p: str) -> tuple:
        group_order = {"static": 0, "locomotion": 1, "action": 2}
        g = _POSE_GROUPS.get(p, "other")
        return (group_order.get(g, 99), p)

    all_poses = sorted(all_poses_set, key=_pose_sort_key)

    # Time bins
    timestamps = [snap.get("ts", 0.0) for snap in timeline]
    cols = [f"{ts:.1f}s" for ts in timestamps]

    # Build matrix
    values = []
    for pose in all_poses:
        row_vals = []
        for snap in timeline:
            cnt = 0
            for k, v in snap.get("counts", {}).items():
                if _canon(k) == pose:
                    cnt += int(v)
            row_vals.append(cnt)
        values.append(row_vals)

    # Normalize to [0,1] by grid max
    grid_max = max((v for row in values for v in row), default=1)
    norm_values = [
        [round(v / max(grid_max, 1), 4) for v in row]
        for row in values
    ]

    # Seek map: col_index -> timestamp
    seek_map = {i: round(ts, 3) for i, ts in enumerate(timestamps)}

    return {
        "type": "grid",
        "title": "Phase 1 — Pose activity density (pose × time)",
        "rows": all_poses,
        "cols": cols,
        "values": norm_values,
        "raw_counts": [[int(v) for v in row] for row in values],
        "legend": ["Low count", "High count"],
        "ramp": _POSE_RAMP,
        "seek_map": seek_map,
    }


def _build_presence_bar(summary: dict) -> dict:
    """Build presence percentage bar chart per pose."""
    poses = summary.get("poses", {})
    if not poses:
        return {"type": "bars", "title": "Pose presence (% of frames)", "rows": [], "values": [], "color": "#16a34a"}

    rows = []
    values = []
    for k, s in sorted(poses.items(), key=lambda x: -x[1].get("presence_pct", 0)):
        cn = _canon(k)
        if cn == "?":
            continue
        rows.append(cn)
        values.append(round(s.get("presence_pct", 0.0), 1))

    return {
        "type": "bars",
        "title": "Phase 2 — Pose presence (% of frames active)",
        "rows": rows,
        "values": values,
        "color": "#16a34a",
        "bgColor": "#f0fdf4",
    }


def _build_events_heatmap(events: list[dict], summary: dict) -> dict:
    """Build action × time event density heatmap from events.json."""
    if not events:
        return {"type": "grid", "title": "Action event density", "rows": [], "cols": [], "values": [],
                "legend": ["Low", "High"], "ramp": ["#fef9c3", "#fde047", "#f59e0b", "#b45309", "#78350f"]}

    # Determine time range
    timestamps = [e.get("ts", 0.0) for e in events]
    t_min = min(timestamps)
    t_max = max(timestamps)
    duration = max(t_max - t_min, 1e-6)

    n_bins = min(12, max(4, len(events) // 10))
    bin_width = duration / n_bins

    # Collect all actions
    action_set = sorted({_canon(e.get("action", "?")) for e in events if _canon(e.get("action", "?")) != "?"})

    # Count per action per time bin
    matrix: dict[str, list[int]] = {a: [0] * n_bins for a in action_set}
    for e in events:
        a = _canon(e.get("action", "?"))
        if a == "?":
            continue
        bi = min(int((e.get("ts", 0.0) - t_min) / bin_width), n_bins - 1)
        matrix[a][bi] += 1

    col_labels = [f"{t_min + i * bin_width:.1f}s" for i in range(n_bins)]
    values = [[matrix[a][i] for i in range(n_bins)] for a in action_set]

    grid_max = max((v for row in values for v in row), default=1)
    norm_values = [[round(v / max(grid_max, 1), 4) for v in row] for row in values]

    seek_map = {i: round(t_min + i * bin_width, 3) for i in range(n_bins)}

    return {
        "type": "grid",
        "title": "Phase 3 — Action event density (action × time)",
        "rows": action_set,
        "cols": col_labels,
        "values": norm_values,
        "raw_counts": [[int(v) for v in row] for row in values],
        "legend": ["Low density", "High density"],
        "ramp": ["#fef9c3", "#fde047", "#f59e0b", "#b45309", "#78350f"],
        "seek_map": seek_map,
    }


def _build_throughput_timeline(events: list[dict]) -> dict:
    """Build event throughput timeline (events per second over bins)."""
    if not events:
        return {"type": "timeline", "title": "Event throughput", "labels": [], "values": [], "color": "#ea580c"}

    timestamps = [e.get("ts", 0.0) for e in events]
    t_min = min(timestamps)
    t_max = max(timestamps)
    duration = max(t_max - t_min, 1e-6)

    n_bins = min(14, max(6, len(events) // 8))
    bin_width = duration / n_bins

    counts = [0] * n_bins
    for ts in timestamps:
        bi = min(int((ts - t_min) / bin_width), n_bins - 1)
        counts[bi] += 1

    labels = [f"{t_min + i * bin_width:.1f}s" for i in range(n_bins)]
    seek_map = {i: round(t_min + i * bin_width, 3) for i in range(n_bins)}

    return {
        "type": "timeline",
        "title": "Phase 4 — Event throughput per time bin",
        "labels": labels,
        "values": counts,
        "color": "#ea580c",
        "seek_map": seek_map,
    }


def _build_confidence_histogram(events: list[dict]) -> dict:
    """Build confidence distribution histogram."""
    if not events:
        return {"type": "bars", "title": "Action confidence distribution", "rows": [], "values": [], "color": "#7c3aed"}

    bins = 10
    hist = [0] * bins
    for e in events:
        c = e.get("confidence", 0.5)
        bi = min(int(c * bins), bins - 1)
        hist[bi] += 1

    labels = [f"{i/bins:.1f}–{(i+1)/bins:.1f}" for i in range(bins)]
    total = sum(hist) or 1

    return {
        "type": "bars",
        "title": "Phase 2 — Action confidence distribution",
        "rows": labels,
        "values": [round(v / total * 100, 1) for v in hist],
        "color": "#7c3aed",
    }


def _build_role_distribution(events: list[dict]) -> dict:
    """Build role distribution bar chart."""
    if not events:
        return {"type": "bars", "title": "Role distribution", "rows": [], "values": [], "color": "#0891b2"}

    counts: dict[str, int] = defaultdict(int)
    for e in events:
        role = e.get("role", "unknown")
        counts[role] += 1

    total = sum(counts.values()) or 1
    ordered = sorted(counts.items(), key=lambda x: -x[1])

    return {
        "type": "bars",
        "title": "Phase 3 — Role distribution across events",
        "rows": [k for k, _ in ordered],
        "values": [round(v / total * 100, 1) for _, v in ordered],
        "color": "#0891b2",
    }


def _build_spatial_data(events: list[dict]) -> dict:
    """
    Build spatial occupancy data for each pose.
    Returns {pose: {xs: [...], ys: [...]}} in normalized [0,1] frame coords.
    Also returns metadata for the spatial heatmap rendering.
    """
    if not events:
        return {"poses": {}, "ramp": _POSE_RAMP, "title": "Spatial occupancy by pose"}

    pose_pts: dict[str, dict[str, list[float]]] = {}
    for e in events:
        a = _canon(e.get("action", "?"))
        if a == "?":
            continue
        if a not in pose_pts:
            pose_pts[a] = {"xs": [], "ys": []}
        bbox = e.get("bbox")
        if bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            pose_pts[a]["xs"].append(round(cx, 4))
            pose_pts[a]["ys"].append(round(cy, 4))

    return {
        "poses": pose_pts,
        "ramp": _POSE_RAMP,
        "title": "Spatial occupancy by pose",
    }


def build_heatmap_payload(job_id: str) -> dict | None:
    """
    Build full heatmap payload for a completed job.
    Returns None if job hasn't produced output files yet.
    """
    summary = _read_summary(job_id)
    if not summary:
        return None

    events = _read_events(job_id)

    meta = summary.get("meta", {})

    # Phase 1: Pose activity timeline (from summary.json timeline)
    phase1_pose_timeline = _build_pose_timeline_heatmap(summary)

    # Phase 2: Presence bars + confidence histogram
    phase2_presence = _build_presence_bar(summary)
    phase2_confidence = _build_confidence_histogram(events)

    # Phase 3: Events heatmap + role distribution
    phase3_events = _build_events_heatmap(events, summary)
    phase3_roles = _build_role_distribution(events)

    # Phase 4: Throughput timeline + spatial data
    phase4_throughput = _build_throughput_timeline(events)
    phase4_spatial = _build_spatial_data(events)

    # ── seek_map scaling: input video time -> output video time ──────────────
    # seek_map timestamps are in INPUT-video time (frame_idx / src_fps).
    # The output video is shorter when frame_skip is active.
    # We need the ratio: output_dur / input_dur
    #
    # Key names in summary.json meta (written by infer-vid-w7.py SummaryLog):
    #   "input_duration_sec"  = total_frm / src_fps  (raw input length)  ← CORRECT
    #   "input_total_frames"  = total_frm
    #   "duration_sec"        = frames_processed / src_fps  (processed frames only) ← NOT input dur
    #   "frames_processed"    = _total_frames (writer thread frames)
    #
    # Always prefer input_duration_sec; fall back to input_total_frames / src_fps.
    input_duration_sec = (
        meta.get("input_duration_sec")                          # direct field (v7+)
        or (meta.get("input_total_frames", 0) / max(meta.get("src_fps", 25.0), 1.0))  # computed
        or meta.get("duration_sec", 0)                          # last resort (may be wrong)
        or None
    )

    # Output duration: try cv2 first (always available), then ffprobe
    output_path_mp4 = OUTPUT_DIR / f"{job_id}_output.mp4"
    output_duration_sec: float | None = None
    
    # Wait for file to be fully written (race condition protection)
    import time
    for attempt in range(3):
        if not output_path_mp4.exists():
            time.sleep(0.5)  # File might still be written
            continue
        
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(str(output_path_mp4))
            if cap.isOpened():
                _fps = cap.get(_cv2.CAP_PROP_FPS)
                _fc  = cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if _fps and _fps > 0 and _fc and _fc > 0:
                    output_duration_sec = round(_fc / _fps, 3)
                    break  # Success, exit retry loop
        except Exception as e:
            import logging as _log_attempt
            _log_attempt.getLogger(__name__).warning("[heatmap] cv2 failed attempt %d: %s", attempt+1, e)
            time.sleep(0.2)
    
    if output_duration_sec is None:
        # Fall back to ffprobe
        output_duration_sec = _ffprobe_duration_sec(output_path_mp4)

    # Compute ratio; log it so it's visible in the FastAPI console
    if input_duration_sec and output_duration_sec and input_duration_sec > 0:
        _seek_ratio = output_duration_sec / input_duration_sec
        # Bounds check: ratio should be between 0.001 and 1.0 (frame skip scenario)
        if _seek_ratio < 0.001 or _seek_ratio > 1.0:
            import logging as _warn_log
            _warn_log.getLogger(__name__).warning(
                "[heatmap] seek_ratio=%.6f out of expected bounds (0.001-1.0), "
                "frame skip or duration mismatch suspected", _seek_ratio
            )
        _seek_ratio = round(max(0.001, min(_seek_ratio, 1.0)), 6)
    else:
        import logging as _log_ratio
        _log_ratio.getLogger(__name__).warning(
            "[heatmap] cannot compute seek_ratio: input_dur=%s, output_dur=%s",
            input_duration_sec, output_duration_sec
        )
        _seek_ratio = 1.0

    import logging as _logging
    _logging.getLogger(__name__).info(
        "[heatmap] job=%s  input_dur=%.3fs  output_dur=%s  seek_ratio=%.4f",
        job_id,
        input_duration_sec or 0,
        f"{output_duration_sec:.3f}s" if output_duration_sec else "None",
        _seek_ratio,
    )

    payload = {
        "job_id": job_id,
        "meta": {
            "input":               meta.get("input", ""),
            "duration_sec":        input_duration_sec or 0,      # INPUT duration (for seek ratio)
            "output_duration_sec": output_duration_sec or 0,     # OUTPUT duration (actual video)
            "seek_ratio":          _seek_ratio,                   # output / input (frontend uses this)
            "frames_processed":    meta.get("frames_processed", 0),
            "n_events":            len(events),
            "n_poses":             len(summary.get("poses", {})),
        },
        "phases": {
            "1": {
                "label": "Observation",
                "panels": [
                    {"id": "pose_timeline", "data": phase1_pose_timeline},
                ],
            },
            "2": {
                "label": "Filter",
                "panels": [
                    {"id": "presence_bar", "data": phase2_presence},
                    {"id": "confidence_hist", "data": phase2_confidence},
                ],
            },
            "3": {
                "label": "Derivation",
                "panels": [
                    {"id": "events_heatmap", "data": phase3_events},
                    {"id": "role_distribution", "data": phase3_roles},
                ],
            },
            "4": {
                "label": "Storage",
                "panels": [
                    {"id": "throughput_timeline", "data": phase4_throughput},
                    {"id": "spatial_occupancy", "data": phase4_spatial},
                ],
            },
        },
    }

    # Rescale every panel's seek_map (input-time -> output-time)
    # NOTE: seek_map is now in OUTPUT-time. Frontend should NOT rescale again.
    # Frontend reads these values directly and sends them to video player.
    # If frontend needs to send to /run-again, it must reverse-scale using
    # unscale_seek_map(seek_map, input_duration_sec, output_duration_sec).
    for phase in payload["phases"].values():
        for panel in phase["panels"]:
            data = panel["data"]
            if isinstance(data, dict) and "seek_map" in data:
                data["seek_map"] = scale_seek_map(
                    data["seek_map"], input_duration_sec, output_duration_sec
                )

    return payload


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.post("/hello-cv")
async def hello_computer_vision(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.putText(img, "Hello World!", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, encoded_img = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


@app.post("/run-pipeline")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    processing_mode: str = "full",   # form field: "full" | "compressed"
):
    if processing_mode not in ("full", "compressed"):
        processing_mode = "full"

    job_id      = str(uuid.uuid4())
    suffix      = Path(file.filename).suffix or ".mp4"
    input_path  = str(UPLOAD_DIR / f"{job_id}_input{suffix}")
    output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")

    # Save uploaded file to disk in streaming chunks to avoid holding
    # the entire upload in memory (important for large 4K files).
    try:
        total = 0
        with open(input_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        print(f"[run-pipeline] wrote {total} bytes to {input_path}")
    except Exception as e:
        print(f"[run-pipeline] failed to write file: {e}")
        return JSONResponse({"error": f"file write failed: {str(e)}"}, status_code=400)

    est_minutes = _estimate_minutes(input_path, processing_mode)

    JOBS[job_id] = {
        "status":           "queued",
        "input_file":       file.filename,
        "input_path":       input_path,
        "output_path":      output_path,
        "processing_mode":  processing_mode,
        "est_minutes":      est_minutes,
        "log":              [],
    }

    background_tasks.add_task(
        _run_pipeline, job_id, input_path, output_path,
        processing_mode=processing_mode,
    )

    return JSONResponse({
        "job_id":          job_id,
        "status":          "queued",
        "processing_mode": processing_mode,
        "est_minutes":     est_minutes,
    })


@app.get("/pipeline-status/{job_id}")
def pipeline_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    resp = {
        "status":           job["status"],
        "input_file":       job.get("input_file", ""),
        "log_tail":         job["log"][-40:],
        "progress":         job.get("progress", 0.0),
        "phase_label":      job.get("phase_label", ""),
        "seek_sec":         job.get("seek_sec"),
        "processing_mode":  job.get("processing_mode", "full"),
        "est_minutes":      job.get("est_minutes"),
    }
    if job["status"] == "done":
        resp["output_url"] = f"/outputs/{job['output_file']}"
    if job["status"] == "error":
        resp["error"] = job.get("error", "unknown error")

    return JSONResponse(resp)


# ── /run-again: re-run pipeline on existing job's input at a precise timestamp ─

from pydantic import BaseModel

class RunAgainRequest(BaseModel):
    job_id:       str
    target_second: float       # absolute source-video second (e.g. 45.5)

@app.post("/run-again")
async def run_again(req: RunAgainRequest, background_tasks: BackgroundTasks):
    """
    Re-run the pipeline on an existing job's input video, seeking precisely
    to `target_second` (absolute source-video time) instead of reading from 0.

    The seek is forwarded as --seek_sec to infer-vid-w7.py, which bypasses
    the full sequential queue and jumps directly to the requested frame window.
    """
    parent = JOBS.get(req.job_id)
    if not parent:
        return JSONResponse({"error": "parent job not found"}, status_code=404)

    input_path  = parent.get("input_path")
    if not input_path or not Path(input_path).exists():
        return JSONResponse({"error": "input video no longer on disk"}, status_code=404)

    job_id      = str(uuid.uuid4())
    output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")

    JOBS[job_id] = {
        "status":      "queued",
        "input_file":  parent.get("input_file", ""),
        "input_path":  input_path,
        "output_path": output_path,
        "seek_sec":    req.target_second,
        "log":         [],
    }

    background_tasks.add_task(
        _run_pipeline, job_id, input_path, output_path, req.target_second
    )
    return JSONResponse({
        "job_id":        job_id,
        "status":        "queued",
        "target_second": req.target_second,
    })


@app.get("/heatmap/{job_id}")
def heatmap_data(job_id: str):
    """
    Return computed heatmap data for a completed job.
    Reads summary.json + events.json and builds structured heatmap payloads
    matching the frontend's HEATMAP_DEFS schema for each phase.
    """
    # Check job exists
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    if job["status"] != "done":
        return JSONResponse({"error": "job not yet complete", "status": job["status"]}, status_code=409)

    payload = build_heatmap_payload(job_id)
    if payload is None:
        return JSONResponse({"error": "output files not found for this job"}, status_code=404)

    return JSONResponse(payload)