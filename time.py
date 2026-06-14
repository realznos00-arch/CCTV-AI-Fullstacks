"""
time.py — Video Duration Reporter + Dual Timestamper
=====================================================
Standalone companion to infer-vid-w7.py.
No imports from main.py or infer-vid-w7.py.

WHAT IT DOES
------------
  1. PRE-process length   — reads the raw input video via cv2 and reports
                            its real duration in seconds before any pipeline
                            processing. Also captures FPS, frame count,
                            resolution.

  2. POST-process length  — reads the pipeline's summary.json (written by
                            SummaryLog in infer-vid-w7) and reports the
                            duration actually processed, plus the output
                            video length via cv2 on the .mp4 output.

  3. TIMESTAMPER JSON     — reads all three sidecars produced by infer-vid-w7
                            (summary.json, events.json, candidates.json) and
                            writes a single  <stem>.timestamps.json  that has:

        {
          "source": { pre  video_path, duration_sec, fps, frames, w, h },
          "output": { post video_path, duration_sec, fps, frames, w, h },
          "timeline_pre":  [ { frame, ts_pre,  ts_post, ... snapshot ... } ],
          "timeline_post": [ { frame, ts_pre,  ts_post, ... snapshot ... } ],
          "events_pre":    [ { ...event..., ts_pre, ts_post } ],
          "events_post":   [ { ...event..., ts_pre, ts_post } ],
          "candidates_pre":  [ { ...candidate..., ts_pre, ts_post } ],
          "candidates_post": [ { ...candidate..., ts_pre, ts_post } ],
        }

      "pre"  = timestamp calculated from frame_number / src_fps  (raw clock)
      "post" = timestamp as-recorded in the sidecar  (pipeline clock, may
               differ if frames were dropped or start_sec != 0)

USAGE
-----
  # Minimal — just print durations
  python time.py --input  video.mp4
                 --output outputs/abc123_output.mp4

  # Full timestamper (writes <output_stem>.timestamps.json)
  python time.py --input   uploads/abc123_input.mp4
                 --output  outputs/abc123_output.mp4
                 --summary outputs/abc123_output.summary.json
                 --events  outputs/abc123_output.events.json
                 --candidates outputs/abc123_output.candidates.json
                 --write-timestamps

  # Auto-discover sidecars (infer stems from --output path, same convention
  # as infer-vid-w7: replaces .mp4 suffix with .summary.json etc.)
  python time.py --input  uploads/abc123_input.mp4
                 --output outputs/abc123_output.mp4
                 --auto   --write-timestamps

MAIN.PY PROOF
-------------
  No symbols from main.py are imported.  main.py can call this as a
  subprocess exactly the same way it calls infer-vid-w7.py:

      subprocess.run([
          PYTHON_EXE, "time.py",
          "--input",  input_path,
          "--output", output_path,
          "--auto",
          "--write-timestamps",
      ], check=True)

  Or import and call directly (zero shared state):

      from time import run as time_run
      result = time_run(input_path, output_path, auto=True, write_timestamps=True)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 stdout/stderr so box-drawing or arrow chars in log messages
# never crash on Thai/non-Latin Windows code pages (cp874, cp1252, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# VIDEO PROBE  (cv2 only — no ffprobe dependency)
# =============================================================================

def probe_video(path: str | Path) -> dict:
    """
    Return basic metadata for a video file.

    Returns
    -------
    dict with keys:
        path        str   — absolute path
        exists      bool
        fps         float
        frame_count int
        duration_sec float
        width       int
        height      int
        error       str | None
    """
    p = Path(path)
    base: dict[str, Any] = {
        "path":         str(p.resolve()),
        "exists":       p.exists(),
        "fps":          0.0,
        "frame_count":  0,
        "duration_sec": 0.0,
        "width":        0,
        "height":       0,
        "error":        None,
    }
    if not p.exists():
        base["error"] = f"File not found: {p}"
        return base

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        base["error"] = f"cv2 cannot open: {p}"
        return base

    try:
        fps    = cap.get(cv2.CAP_PROP_FPS)   or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Some containers report 0 or −1 for frame count.
        # Fall back: scan the file (slow for large files, but correct).
        if frames <= 0:
            log.warning("CAP_PROP_FRAME_COUNT unavailable for %s - scanning...", p.name)
            frames = 0
            while cap.grab():
                frames += 1

        dur = round(frames / fps, 3) if fps > 0 else 0.0

        base.update({
            "fps":          round(fps, 4),
            "frame_count":  frames,
            "duration_sec": dur,
            "width":        w,
            "height":       h,
        })
    finally:
        cap.release()

    return base


# =============================================================================
# SIDECAR READERS  (same file naming as infer-vid-w7 SummaryLog / json_log)
# =============================================================================

def _read_json(path: str | Path) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not read %s: %s", p, e)
        return None


def read_summary(path: str | Path) -> dict | None:
    """Read <output>.summary.json produced by SummaryLog."""
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    return data


def read_events(path: str | Path) -> tuple[list[dict], dict]:
    """
    Read events.json produced by infer-vid-w7.
    Returns (events_list, meta_dict) — meta carries src_fps, input_total_frames,
    input_duration_sec, start_sec, end_sec, segment_duration_sec, etc.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        return [], {}
    return data.get("events", []), data.get("meta", {})


def read_candidates(path: str | Path) -> tuple[list[dict], dict]:
    """
    Read candidates.json produced by infer-vid-w7.
    Returns (candidates_list, meta_dict).
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        return [], {}
    return data.get("candidates", []), data.get("meta", {})


def auto_sidecar_paths(output_path: str | Path) -> dict[str, Path]:
    """
    Derive the standard sidecar paths from the output video path,
    mirroring the naming convention in infer-vid-w7.py:

        <stem>.summary.json
        <stem>.events.json
        <stem>.candidates.json
    """
    base = Path(output_path).with_suffix("")   # strips .mp4
    return {
        "summary":    base.with_suffix(".summary.json"),
        "events":     base.with_suffix(".events.json"),
        "candidates": base.with_suffix(".candidates.json"),
    }


# =============================================================================
# DURATION REPORT
# =============================================================================

def _best_meta(*metas: dict) -> dict:
    """
    Merge meta dicts from sidecars — first non-None value for each key wins.
    Sidecars should be passed in priority order: summary > events > candidates.
    """
    merged: dict = {}
    for m in metas:
        if not m:
            continue
        for k, v in m.items():
            if k not in merged and v is not None:
                merged[k] = v
    return merged


def duration_report(
    pre_probe:    dict,
    post_probe:   dict,
    summary:      dict | None,
    events_meta:  dict | None = None,
    cands_meta:   dict | None = None,
) -> dict:
    """
    Build a concise duration comparison dict.

    Pulls pre-process info from the enriched meta blocks that
    infer-vid-w7 now writes into every sidecar:
        input_total_frames, input_duration_sec,
        start_sec, end_sec, segment_duration_sec, src_fps
    Falls back to cv2 probe values when meta is absent.
    """
    summary_meta: dict = summary.get("meta", {}) if summary else {}
    meta = _best_meta(summary_meta, events_meta or {}, cands_meta or {})

    src_fps         = float(meta.get("src_fps", 0) or pre_probe.get("fps") or 25.0)
    pipeline_frames = meta.get("frames_processed")   # set by SummaryLog._build
    pipeline_dur    = (
        round(pipeline_frames / max(src_fps, 1.0), 3) if pipeline_frames else
        meta.get("segment_duration_sec")              # fallback to w7 segment calc
    )

    # Pre-process ground truth — prefer json meta over cv2 probe
    # (cv2 sometimes misreports frame count on re-encoded files)
    pre_total_frames = meta.get("input_total_frames") or pre_probe.get("frame_count") or 0
    pre_duration_sec = (
        meta.get("input_duration_sec") or
        pre_probe.get("duration_sec") or
        round(pre_total_frames / max(src_fps, 1.0), 3)
    )

    return {
        "pre_process": {
            "video_path":        pre_probe["path"],
            # duration from meta is authoritative (w7 measured it at open time)
            "duration_sec":      pre_duration_sec,
            "frame_count":       pre_total_frames,
            "fps":               src_fps,
            "resolution":        f"{meta.get('src_w', pre_probe['width'])}x{meta.get('src_h', pre_probe['height'])}",
            "error":             pre_probe["error"],
            # segment actually processed
            "segment_start_sec": meta.get("start_sec"),
            "segment_end_sec":   meta.get("end_sec"),
            "segment_duration_sec": meta.get("segment_duration_sec"),
        },
        "post_process": {
            "video_path":     post_probe["path"],
            "duration_sec":   post_probe["duration_sec"],
            "frame_count":    post_probe["frame_count"],
            "fps":            post_probe["fps"],
            "resolution":     f"{post_probe['width']}x{post_probe['height']}",
            "error":          post_probe["error"],
            # pipeline accounting (from summary.json meta)
            "pipeline_duration_sec":     pipeline_dur,
            "pipeline_frames_processed": pipeline_frames,
        },
        "delta_sec": round(
            (post_probe["duration_sec"] or 0.0) - pre_duration_sec,
            3,
        ),
        # raw meta for downstream consumers
        "_meta": meta,
    }


# =============================================================================
# TIMESTAMP ENRICHMENT
# =============================================================================

def _enrich(
    record: dict,
    frame_key: str,
    ts_key: str,
    src_fps: float,
) -> dict:
    """
    Add ts_pre (frame / src_fps) and ts_post (as-recorded ts) to a copy
    of record.  frame_key / ts_key are the field names in the source dict.
    """
    out = dict(record)
    frame_num = record.get(frame_key, 0) or 0
    recorded_ts = record.get(ts_key)

    ts_pre  = round(frame_num / max(src_fps, 1.0), 3)
    ts_post = round(float(recorded_ts), 3) if recorded_ts is not None else ts_pre

    out["ts_pre"]  = ts_pre   # timestamp from raw frame clock
    out["ts_post"] = ts_post  # timestamp recorded by pipeline
    return out


def build_timestamps(
    summary:    dict | None,
    events:     list[dict],
    candidates: list[dict],
    src_fps:    float,
    pre_probe:  dict,
    post_probe: dict,
) -> dict:
    """
    Build the full timestamps payload.

    Each list is provided in two versions:
      *_pre  — entries sorted by ts_pre  (raw video clock)
      *_post — entries sorted by ts_post (pipeline clock as recorded)
    """

    # ── timeline from summary.json ────────────────────────────────────────────
    timeline_raw: list[dict] = []
    if summary:
        timeline_raw = summary.get("timeline", [])

    timeline_enriched = [
        _enrich(snap, frame_key="frame", ts_key="ts", src_fps=src_fps)
        for snap in timeline_raw
    ]
    timeline_pre  = sorted(timeline_enriched, key=lambda x: x["ts_pre"])
    timeline_post = sorted(timeline_enriched, key=lambda x: x["ts_post"])

    # ── events from events.json ───────────────────────────────────────────────
    events_enriched = [
        _enrich(ev, frame_key="frame", ts_key="ts", src_fps=src_fps)
        for ev in events
    ]
    events_pre  = sorted(events_enriched, key=lambda x: x["ts_pre"])
    events_post = sorted(events_enriched, key=lambda x: x["ts_post"])

    # ── candidates from candidates.json ──────────────────────────────────────
    # candidates use "frame_number" and "timestamp_sec" (Candidate.to_dict)
    candidates_enriched = [
        _enrich(c, frame_key="frame_number", ts_key="timestamp_sec", src_fps=src_fps)
        for c in candidates
    ]
    candidates_pre  = sorted(candidates_enriched, key=lambda x: x["ts_pre"])
    candidates_post = sorted(candidates_enriched, key=lambda x: x["ts_post"])

    return {
        "source": {
            "video_path":   pre_probe["path"],
            "duration_sec": pre_probe["duration_sec"],
            "fps":          pre_probe["fps"],
            "frame_count":  pre_probe["frame_count"],
            "width":        pre_probe["width"],
            "height":       pre_probe["height"],
        },
        "output": {
            "video_path":   post_probe["path"],
            "duration_sec": post_probe["duration_sec"],
            "fps":          post_probe["fps"],
            "frame_count":  post_probe["frame_count"],
            "width":        post_probe["width"],
            "height":       post_probe["height"],
        },
        # ── seek precision metadata ───────────────────────────────────────────
        # Frontend uses seek_ratio to map raw-source seconds → output-video seconds.
        # seek_ratio = output_duration / source_duration  (1.0 = no frame drop)
        "seek_precision": {
            "seek_ratio": round(
                post_probe["duration_sec"] / max(pre_probe["duration_sec"], 1e-6), 6
            ) if post_probe["duration_sec"] and pre_probe["duration_sec"] else 1.0,
            "src_fps":            round(src_fps, 4),
            "input_duration_sec": pre_probe["duration_sec"],
            "output_duration_sec": post_probe["duration_sec"],
        },
        # ── timeline (summary checkpoints) ────────────────────────────────────
        # pre:  ordered by raw video clock (frame / src_fps)
        # post: ordered by pipeline-recorded timestamp
        "timeline_pre":      timeline_pre,
        "timeline_post":     timeline_post,
        # ── per-person action events ──────────────────────────────────────────
        "events_pre":        events_pre,
        "events_post":       events_post,
        # ── delta/emerging candidates ─────────────────────────────────────────
        "candidates_pre":    candidates_pre,
        "candidates_post":   candidates_post,
        # ── stats ──────────────────────────────────────────────────────────────
        "stats": {
            "n_timeline_snaps":  len(timeline_enriched),
            "n_events":          len(events_enriched),
            "n_candidates":      len(candidates_enriched),
            "src_fps_used":      round(src_fps, 4),
        },
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def _fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "n/a"
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}  ({sec:.3f}s)"


def print_duration_report(report: dict) -> None:
    pre  = report["pre_process"]
    post = report["post_process"]

    # ASCII-only box drawing so this works on any Windows code page (cp874, cp1252, etc.)
    SEP = "+---------------------------------------------+"
    print("\n" + SEP)
    print("|          VIDEO DURATION REPORT              |")
    print(SEP)
    print("|  PRE-PROCESS  (raw input)                   |")
    print(f"|    Path      : {Path(pre['video_path']).name}")
    print(f"|    Duration  : {_fmt_dur(pre['duration_sec'])}")
    print(f"|    Frames    : {pre['frame_count']}")
    print(f"|    FPS       : {pre['fps']}")
    print(f"|    Res       : {pre['resolution']}")
    if pre.get("segment_start_sec") is not None:
        print(f"|    Segment   : {pre['segment_start_sec']:.3f}s -> {pre['segment_end_sec']:.3f}s")
        print(f"|    Seg dur   : {_fmt_dur(pre['segment_duration_sec'])}")
    if pre["error"]:
        print(f"|    ERROR     : {pre['error']}")
    print(SEP)
    print("|  POST-PROCESS  (output video)               |")
    print(f"|    Path      : {Path(post['video_path']).name}")
    print(f"|    Duration  : {_fmt_dur(post['duration_sec'])}")
    print(f"|    Frames    : {post['frame_count']}")
    print(f"|    FPS       : {post['fps']}")
    print(f"|    Res       : {post['resolution']}")
    if post["pipeline_duration_sec"] is not None:
        print(f"|    Pipeline  : {_fmt_dur(post['pipeline_duration_sec'])}  (summary.json)")
        print(f"|    Proc frames: {post['pipeline_frames_processed']}")
    if post["error"]:
        print(f"|    ERROR     : {post['error']}")
    print(SEP)
    delta = report["delta_sec"]
    sign  = "+" if delta >= 0 else ""
    print(f"|  Delta (post - pre) : {sign}{delta:.3f}s")
    print(SEP + "\n")


# =============================================================================
# MAIN ENTRY POINT  (callable as function or CLI)
# =============================================================================

def run(
    input_path:       str | Path,
    output_path:      str | Path,
    summary_path:     str | Path | None = None,
    events_path:      str | Path | None = None,
    candidates_path:  str | Path | None = None,
    auto:             bool = False,
    write_timestamps: bool = False,
    timestamps_path:  str | Path | None = None,
    silent:           bool = False,
) -> dict:
    """
    Core runner. Returns a dict:
        {
          "duration": <duration_report>,
          "timestamps": <timestamps_payload> | None,
          "timestamps_file": <str path> | None,
        }

    Parameters
    ----------
    input_path       : raw input video (pre-process)
    output_path      : pipeline output video (post-process)
    summary_path     : path to *.summary.json  (or None)
    events_path      : path to *.events.json   (or None)
    candidates_path  : path to *.candidates.json (or None)
    auto             : if True, derive sidecar paths from output_path
    write_timestamps : if True, write <output_stem>.timestamps.json
    timestamps_path  : override the output path for the timestamps file
    silent           : suppress print output
    """
    # ── Auto-discover sidecars ────────────────────────────────────────────────
    if auto:
        sidecars = auto_sidecar_paths(output_path)
        summary_path    = summary_path    or sidecars["summary"]
        events_path     = events_path     or sidecars["events"]
        candidates_path = candidates_path or sidecars["candidates"]

    # ── Probe both videos ─────────────────────────────────────────────────────
    log.info("Probing pre-process video: %s", input_path)
    pre_probe = probe_video(input_path)

    log.info("Probing post-process video: %s", output_path)
    post_probe = probe_video(output_path)

    # ── Read sidecars ─────────────────────────────────────────────────────────
    summary        = read_summary(summary_path)    if summary_path    else None
    events,   ev_meta   = read_events(events_path)      if events_path     else ([], {})
    candidates, cand_meta = read_candidates(candidates_path) if candidates_path else ([], {})

    if summary_path and summary is None:
        log.warning("summary.json not found or unreadable: %s", summary_path)
    if events_path and not events:
        log.warning("events.json not found or empty: %s", events_path)
    if candidates_path and not candidates:
        log.info("candidates.json not found or empty: %s", candidates_path)

    # ── Duration report ───────────────────────────────────────────────────────
    dur_report = duration_report(pre_probe, post_probe, summary,
                                 events_meta=ev_meta, cands_meta=cand_meta)
    if not silent:
        print_duration_report(dur_report)

    # ── Timestamps ────────────────────────────────────────────────────────────
    timestamps_payload: dict | None = None
    timestamps_file: str | None = None

    need_timestamps = write_timestamps or bool(summary or events or candidates)

    if need_timestamps:
        # src_fps: summary meta > events meta > candidates meta > cv2 probe > 25
        # dur_report["_meta"] already merged all three sidecar metas in priority order
        _merged_meta = dur_report.get("_meta", {})
        src_fps: float = float(
            _merged_meta.get("src_fps", 0) or
            pre_probe.get("fps") or
            25.0
        )

        log.info(
            "Building timestamps — src_fps=%.4f  events=%d  candidates=%d  timeline_snaps=%d",
            src_fps, len(events), len(candidates),
            len(summary.get("timeline", [])) if summary else 0,
        )

        timestamps_payload = build_timestamps(
            summary    = summary,
            events     = events,
            candidates = candidates,
            src_fps    = src_fps,
            pre_probe  = pre_probe,
            post_probe = post_probe,
        )

        if not silent:
            stats = timestamps_payload["stats"]
            print(
                f"[time.py] Timestamps built: "
                f"{stats['n_timeline_snaps']} timeline snaps  |  "
                f"{stats['n_events']} events  |  "
                f"{stats['n_candidates']} candidates"
            )

        if write_timestamps:
            if timestamps_path:
                out_p = Path(timestamps_path)
            else:
                out_p = Path(output_path).with_suffix("").with_suffix(".timestamps.json")

            try:
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(timestamps_payload, f, indent=2)
                timestamps_file = str(out_p.resolve())
                log.info("Timestamps -> %s", timestamps_file)
                if not silent:
                    print(f"[time.py] Timestamps written -> {timestamps_file}")
            except Exception as e:
                log.error("Failed to write timestamps: %s", e)

    return {
        "duration":        dur_report,
        "timestamps":      timestamps_payload,
        "timestamps_file": timestamps_file,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="time.py — video duration reporter + dual timestamper for infer-vid-w7",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--input",       required=True,  help="Raw input video (pre-process)")
    p.add_argument("--output",      required=True,  help="Pipeline output video (post-process)")
    p.add_argument("--summary",     default=None,   help="Path to *.summary.json  (optional if --auto)")
    p.add_argument("--events",      default=None,   help="Path to *.events.json   (optional if --auto)")
    p.add_argument("--candidates",  default=None,   help="Path to *.candidates.json (optional if --auto)")
    p.add_argument("--auto",        action="store_true",
                   help="Auto-derive sidecar paths from --output (infer-vid-w7 naming convention)")
    p.add_argument("--write-timestamps", action="store_true",
                   help="Write <output_stem>.timestamps.json")
    p.add_argument("--timestamps-out", default=None,
                   help="Override path for the timestamps output file")
    p.add_argument("--silent",      action="store_true", help="Suppress stdout report")
    p.add_argument("--json-out",    default=None,
                   help="If given, write the full result dict (duration + timestamps) to this path as JSON")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    result = run(
        input_path       = args.input,
        output_path      = args.output,
        summary_path     = args.summary,
        events_path      = args.events,
        candidates_path  = args.candidates,
        auto             = args.auto,
        write_timestamps = args.write_timestamps,
        timestamps_path  = args.timestamps_out,
        silent           = args.silent,
    )

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result["duration"], f, indent=2)
            log.info("Duration report -> %s", args.json_out)
        except Exception as e:
            log.error("Failed to write json-out: %s", e)
            sys.exit(1)

    # Exit non-zero if either video couldn't be opened
    if result["duration"]["pre_process"]["error"]:
        log.error("Pre-process probe failed: %s", result["duration"]["pre_process"]["error"])
        sys.exit(1)
    if result["duration"]["post_process"]["error"]:
        log.warning("Post-process probe failed: %s", result["duration"]["post_process"]["error"])