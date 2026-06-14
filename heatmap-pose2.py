"""
heatmap_pose.py — Pose-Activity Heatmap Generator  (v2)
=========================================================
Reads checkpoint JSON files produced by infer-vid-5w-negus.py and renders
two families of heatmaps:

  ① TEMPORAL heatmaps  — pose × checkpoint-time, person-count values
  ② SPATIAL  heatmaps  — KDE occupancy density per pose, drawn over
                          a camera-frame canvas  (the "where" map)

Input auto-detection (--path accepts ANY of these):
    output_annotated                      bare base name
    output_annotated.mp4                  video file path
    output_annotated.summary.json         summary file directly
    output_annotated.events.json          events file (resolves sibling)
    C:/Videos/                            directory → scans for *.summary.json
Or explicit:
    --summary foo.summary.json  --events foo.events.json

Spatial heatmaps need bbox data in events.json.
  If absent the tool falls back to zone-proxy positions and advises how
  to add bbox to the events writer in infer-vid-5w-negus.py.

Outputs (written to --outdir, default = folder of summary JSON):
  Temporal ─────────────────────────────────────────────────────────
    heatmap_combined.png
    heatmap_static.png
    heatmap_locomotion.png
    heatmap_actions.png
    heatmap_presence_bar.png
    heatmap_events_timeline.png
  Spatial KDE occupancy ────────────────────────────────────────────
    spatial_all.png             all poses colour-coded, overlaid
    spatial_static.png          idle/sit/lie/crouch/lean
    spatial_locomotion.png      run + walk + sneak
    spatial_actions.png         fall/grab/hit/kick/throw/struggle/gun

Install:
    pip install seaborn matplotlib pandas numpy scipy

Run:
    python heatmap_pose.py --path output_annotated.mp4
    python heatmap_pose.py --path C:/Videos/1-2annotated_output
    python heatmap_pose.py --demo --outdir ./reports
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# Pose taxonomy + canonical name map
# ─────────────────────────────────────────────────────────────────────────────

_CANONICAL: dict[str, str] = {
    # static
    "stand": "Stand/Idle", "Standing / Idle": "Stand/Idle", "IDLE": "Stand/Idle",
    "sit": "Sit", "Sitting": "Sit", "SITTING": "Sit",
    "lying_down": "Lying Down", "Lying Down": "Lying Down", "LYING": "Lying Down",
    "Crouching": "Crouch", "CROUCHING": "Crouch",
    "Leaning": "Lean", "LEANING": "Lean",
    # locomotion
    "run": "Run", "Running": "Run", "RUNNING": "Run",
    "walk": "Walk", "Walking": "Walk", "WALKING": "Walk",
    "sneak": "Sneak", "Sneaking": "Sneak",
    # actions
    "fall": "Fall", "Falling": "Fall", "FALLING": "Fall",
    "grab": "Grab", "Grabbing": "Grab",
    "hit": "Hit", "Hitting": "Hit",
    "kick": "Kick", "Kicking": "Kick",
    "throw": "Throw", "Throwing": "Throw",
    "struggle": "Struggle", "Struggling": "Struggle",
    "gun": "Gun", "Gun": "Gun",
}

_STATIC_SET     = {"Stand/Idle", "Sit", "Lying Down", "Crouch", "Lean"}
_LOCOMOTION_SET = {"Run", "Walk", "Sneak"}
_ACTION_SET     = {"Fall", "Grab", "Hit", "Kick", "Throw", "Struggle", "Gun"}

def _canon(k: str) -> str:
    return _CANONICAL.get(k, k)

def _group(c: str) -> str:
    if c in _STATIC_SET:     return "static"
    if c in _LOCOMOTION_SET: return "locomotion"
    if c in _ACTION_SET:     return "actions"
    return "other"

# RGB colours per pose (for spatial overlay)
_POSE_RGB: dict[str, tuple] = {
    "Stand/Idle":  (0.20, 0.80, 0.80),
    "Sit":         (1.00, 0.80, 0.00),
    "Lying Down":  (0.60, 0.00, 0.60),
    "Crouch":      (0.80, 0.50, 0.00),
    "Lean":        (0.40, 0.80, 0.40),
    "Run":         (0.00, 0.85, 0.20),
    "Walk":        (0.70, 0.90, 0.30),
    "Sneak":       (0.90, 0.60, 0.10),
    "Fall":        (1.00, 0.10, 0.10),
    "Grab":        (1.00, 0.50, 0.00),
    "Hit":         (1.00, 0.00, 0.00),
    "Kick":        (0.20, 0.40, 0.90),
    "Throw":       (0.00, 0.50, 1.00),
    "Struggle":    (0.80, 0.00, 0.60),
    "Gun":         (0.30, 0.00, 0.80),
}

# ─────────────────────────────────────────────────────────────────────────────
# Seaborn theme
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(
    style="dark", context="notebook", font_scale=1.0,
    rc={"figure.dpi": 150, "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13, "axes.labelsize": 11},
)

CMAP_COMBINED   = "YlOrRd"
CMAP_STATIC     = "Blues"
CMAP_LOCOMOTION = "Greens"
CMAP_ACTIONS    = "Reds"


# ─────────────────────────────────────────────────────────────────────────────
# Smart path resolver
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".mts"}
_SUMMARY_SUFF = ".summary.json"
_EVENTS_SUFF  = ".events.json"


def _strip_to_stem(p: Path) -> Path:
    """
    Remove any known suffix so we get the bare output base.
      output_annotated.mp4           → <dir>/output_annotated
      output_annotated.summary.json  → <dir>/output_annotated
      output_annotated.events.json   → <dir>/output_annotated
      output_annotated               → <dir>/output_annotated
    """
    name = p.name
    for suff in (_SUMMARY_SUFF, _EVENTS_SUFF):
        if name.endswith(suff):
            return p.parent / name[: -len(suff)]
    if p.suffix.lower() in _VIDEO_EXTS:
        return p.parent / p.stem
    # No known suffix — return as-is (bare base or unknown ext)
    return p.parent / p.stem if p.suffix else p


def _find_summary(base: Path) -> Path | None:
    """Try <base>.summary.json, then <base>/<stem>.summary.json."""
    direct = base.parent / (base.name + _SUMMARY_SUFF)
    if direct.exists():
        return direct
    # User passed something like "output_annotated" without parent resolved
    alt = Path(str(base) + _SUMMARY_SUFF)
    return alt if alt.exists() else None


def _auto_events(sp: Path) -> Path | None:
    name = sp.name
    if name.endswith(_SUMMARY_SUFF):
        stem = name[: -len(_SUMMARY_SUFF)]
        ep   = sp.parent / (stem + _EVENTS_SUFF)
        return ep if ep.exists() else None
    return None


def _scan_dir(d: Path) -> list[Path]:
    return sorted(d.glob(f"*{_SUMMARY_SUFF}"))


def _die(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n")
    sys.exit(1)


def _prompt_pick(candidates: list[Path]) -> Path:
    print("  Found multiple summary files:")
    for i, c in enumerate(candidates):
        print(f"    [{i}] {c}")
    try:
        raw = input("  Select [0]: ").strip()
        idx = int(raw) if raw else 0
    except (EOFError, ValueError):
        idx = 0
    return candidates[max(0, min(idx, len(candidates) - 1))]


def resolve_input_paths(
    path_arg:    str | None,
    summary_arg: str | None,
    events_arg:  str | None,
) -> tuple[str, str | None]:
    """
    Return (summary_path_str, events_path_str_or_None).

    Priority:
      1. --summary / --events  (explicit, always wins)
      2. --path  smart resolver
      3. Current-directory scan fallback
    """

    # ── 1. Explicit --summary ─────────────────────────────────────────────────
    if summary_arg:
        sp = Path(summary_arg)
        if not sp.exists():
            _die(f"--summary not found: {sp}")
        ep = Path(events_arg) if events_arg else _auto_events(sp)
        return str(sp), (str(ep) if ep and ep.exists() else None)

    # ── 2. --path smart resolver ──────────────────────────────────────────────
    if path_arg:
        p = Path(path_arg)

        # Directory → scan
        if p.is_dir():
            candidates = _scan_dir(p)
            if not candidates:
                _die(f"No *{_SUMMARY_SUFF} files found in: {p}")
            sp = _prompt_pick(candidates) if len(candidates) > 1 else candidates[0]
            ep = Path(events_arg) if events_arg else _auto_events(sp)
            return str(sp), (str(ep) if ep and ep.exists() else None)

        # File or base name
        base = _strip_to_stem(p)
        sp   = _find_summary(base)

        # If user passed the summary file itself directly
        if sp is None and p.name.endswith(_SUMMARY_SUFF) and p.exists():
            sp = p

        if sp is None:
            _die(
                f"Cannot locate summary JSON from: {path_arg}\n"
                f"  Tried: {base}{_SUMMARY_SUFF}\n"
                f"  Tip  : pass the bare output base without extension\n"
                f"         e.g.  --path output_annotated\n"
                f"         e.g.  --path C:\\Videos\\1-2annotated_output"
            )

        ep = Path(events_arg) if events_arg else _auto_events(sp)
        return str(sp), (str(ep) if ep and ep.exists() else None)

    # ── 3. Scan current directory ─────────────────────────────────────────────
    candidates = _scan_dir(Path("."))
    if not candidates:
        _die(
            "No summary JSON found in current directory.\n"
            "  Use : --path <output_base>   or   --summary <file.summary.json>\n"
            "  Demo: --demo"
        )
    sp = _prompt_pick(candidates) if len(candidates) > 1 else candidates[0]
    print(f"  Auto-detected: {sp}")
    ep = _auto_events(sp)
    return str(sp), (str(ep) if ep and ep.exists() else None)


# ─────────────────────────────────────────────────────────────────────────────
# JSON loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_events(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("events", [])


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builders
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline_df(summary: dict) -> pd.DataFrame:
    timeline = summary.get("timeline", [])
    if not timeline:
        raise ValueError(
            "summary.json has no 'timeline' entries — "
            "run too short or checkpoint_frames too large."
        )
    rows = []
    for snap in timeline:
        ts  = float(snap.get("ts", snap.get("frame", 0)))
        row: dict[str, Any] = {"ts": ts}
        for k, cnt in snap.get("counts", {}).items():
            row[_canon(k)] = int(cnt)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ts").fillna(0)
    return df.T.groupby(level=0).sum().T


def build_presence_df(summary: dict) -> pd.DataFrame:
    rows = [
        {"pose":         _canon(k),
         "presence_pct": s.get("presence_pct", 0.0),
         "max_persons":  s.get("max_persons", 0),
         "avg_persons":  s.get("avg_persons", 0.0),
         "group":        _group(_canon(k))}
        for k, s in summary.get("poses", {}).items()
    ]
    df = pd.DataFrame(rows)
    df = df.groupby("pose", as_index=False).agg(
        {"presence_pct": "max", "max_persons": "max",
         "avg_persons": "max", "group": "first"}
    )
    return df.sort_values("presence_pct", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ① TEMPORAL heatmap functions
# ─────────────────────────────────────────────────────────────────────────────

def _ts_labels(index: pd.Index, max_labels: int = 20) -> list[str]:
    step = max(1, len(index) // max_labels)
    out  = []
    for i, ts in enumerate(index):
        if i % step == 0:
            m, s = divmod(int(ts), 60)
            out.append(f"{m:02d}:{s:02d}")
        else:
            out.append("")
    return out


def _temporal_heatmap(
    df: pd.DataFrame, title: str, out_path: str,
    cmap: str = CMAP_COMBINED, figsize: tuple = (14, 5),
    vmax: float | None = None,
) -> None:
    if df.empty or df.shape[1] == 0:
        print(f"  [skip] {title} — no data.")
        return

    ts_lbl     = _ts_labels(df.index)
    annot_data = df.copy().astype(str)
    annot_data[df <= 0] = ""

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.heatmap(
        df.T, ax=ax, cmap=cmap,
        linewidths=0.3, linecolor="#222222",
        annot=annot_data.T, fmt="",
        annot_kws={"size": 7, "color": "white", "weight": "bold"},
        cbar_kws={"label": "Person count", "shrink": 0.7},
        vmin=0, vmax=vmax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Video time (MM:SS)", fontsize=10)
    ax.set_ylabel("Pose / Action", fontsize=10)
    ax.set_xticks(range(len(ts_lbl)))
    ax.set_xticklabels(ts_lbl, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    # Yellow box on peak column per pose
    for ri, pose in enumerate(df.columns):
        if df[pose].max() > 0:
            ci = list(df.index).index(df[pose].idxmax())
            ax.add_patch(plt.Rectangle((ci, ri), 1, 1,
                fill=False, edgecolor="yellow", lw=1.5, clip_on=True))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {Path(out_path).name}")


def _presence_bar(df: pd.DataFrame, title: str, out_path: str) -> None:
    GC = {"static": "#4C72B0", "locomotion": "#55A868",
          "actions": "#C44E52", "other": "#8172B2"}
    palette = [GC.get(g, "#888") for g in df["group"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.42)),
                           constrained_layout=True)
    bars = ax.barh(df["pose"], df["presence_pct"],
                   color=palette, edgecolor="#111", linewidth=0.4)
    for bar, val in zip(bars, df["presence_pct"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=8, color="white")
    legend_els = [mpatches.Patch(facecolor=v, label=k.capitalize())
                  for k, v in GC.items() if k != "other"]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8,
              framealpha=0.6, facecolor="#1a1a2e")
    ax.set_xlim(0, min(100, df["presence_pct"].max() * 1.15))
    ax.set_xlabel("Presence (% of total frames)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.set_facecolor("#1a1a2e");  fig.patch.set_facecolor("#12121e")
    ax.spines[["top", "right"]].set_visible(False)
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_color("white")
    ax.tick_params(colors="white")
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {Path(out_path).name}")


def _events_timeline(
    events: list[dict], title: str, out_path: str,
    fps: float = 25.0, bin_sec: float = 5.0,
) -> None:
    if not events:
        print("  [skip] events timeline — no events.")
        return
    actions = sorted({_canon(e.get("action", "?")) for e in events})
    max_ts  = max(e.get("ts", 0.0) for e in events)
    bins    = np.arange(0, max_ts + bin_sec, bin_sec)
    n_bins  = max(1, len(bins) - 1)
    matrix  = np.zeros((len(actions), n_bins))
    aidx    = {a: i for i, a in enumerate(actions)}
    for e in events:
        a = _canon(e.get("action", "?"))
        b = int(e.get("ts", 0.0) // bin_sec)
        if a in aidx and 0 <= b < n_bins:
            matrix[aidx[a], b] += 1
    col_labels = [f"{int(bins[b])//60:02d}:{int(bins[b])%60:02d}"
                  for b in range(n_bins)]
    df_ev = pd.DataFrame(matrix, index=actions, columns=col_labels)
    fig, ax = plt.subplots(
        figsize=(max(12, n_bins // 3), max(4, len(actions) * 0.6)),
        constrained_layout=True,
    )
    sns.heatmap(
        df_ev, ax=ax, cmap="magma",
        linewidths=0.2, linecolor="#111",
        cbar_kws={"label": f"LSTM events / {bin_sec:.0f}s bin", "shrink": 0.6},
        annot=(df_ev > 0), fmt=".0f",
        annot_kws={"size": 7, "color": "white"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(f"Video time ({bin_sec:.0f}s bins)", fontsize=10)
    ax.set_ylabel("Action (LSTM)", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# ② SPATIAL KDE occupancy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _build_kde_grid(
    xs: list[float], ys: list[float],
    grid_w: int, grid_h: int, sigma: float,
) -> np.ndarray:
    """
    Drop (x,y) points (normalised 0-1) into a 2-D accumulation grid,
    apply Gaussian blur, and peak-normalise to [0,1].
    """
    grid    = np.zeros((grid_h, grid_w), dtype=np.float32)
    col_idx = np.clip((np.array(xs) * grid_w).astype(int), 0, grid_w - 1)
    row_idx = np.clip((np.array(ys) * grid_h).astype(int), 0, grid_h - 1)
    np.add.at(grid, (row_idx, col_idx), 1.0)
    grid    = gaussian_filter(grid, sigma=sigma).astype(np.float32)
    peak    = grid.max()
    if peak > 0:
        grid /= peak
    return grid


def _spatial_single(
    pose_pts: dict[str, tuple[list, list]],
    poses: list[str],
    title: str,
    out_path: str,
    grid_w: int = 160,
    grid_h: int = 90,
    sigma: float = 4.0,
    alpha: float = 0.72,
    cmap_name: str = "hot",
) -> None:
    """
    Spatial KDE occupancy map.

    Single pose  → monochrome heatmap + colorbar (density scale).
    Multi-pose   → per-pose RGB channel blended into one RGBA composite
                   (each pose keeps its _POSE_RGB colour; brighter = denser).

    Axes show normalised frame coordinates [0,1] so the map aligns with
    whatever resolution the CCTV camera was running at.
    """
    present = [p for p in poses if p in pose_pts and pose_pts[p][0]]
    if not present:
        print(f"  [skip] {title} — no spatial data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")

    # Faint zone grid (camera-frame reference)
    for gx in np.linspace(0, 1, 5):
        ax.axvline(gx, color="#2a2a2a", lw=0.6, ls="--")
    for gy in np.linspace(0, 1, 4):
        ax.axhline(gy, color="#2a2a2a", lw=0.6, ls="--")

    legend_patches = []

    if len(present) == 1:
        # ── Monochrome KDE + colorbar ─────────────────────────────────────────
        p       = present[0]
        xs, ys  = pose_pts[p]
        grid    = _build_kde_grid(xs, ys, grid_w, grid_h, sigma)
        masked  = np.ma.masked_where(grid < 0.02, grid)
        img     = ax.imshow(
            masked, extent=[0, 1, 1, 0], origin="upper",
            cmap=cmap_name, alpha=alpha, aspect="auto",
            vmin=0, vmax=1, interpolation="bilinear",
        )
        cb = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label("Relative density", color="white", fontsize=9)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        rgb = _POSE_RGB.get(p, (1, 1, 1))
        legend_patches.append(mpatches.Patch(color=rgb, label=f"{p}  ({len(xs):,} pts)"))

    else:
        # ── Multi-pose RGBA composite ─────────────────────────────────────────
        composite = np.zeros((grid_h, grid_w, 4), dtype=np.float32)
        for p in present:
            xs, ys = pose_pts[p]
            grid   = _build_kde_grid(xs, ys, grid_w, grid_h, sigma)
            rgb    = _POSE_RGB.get(p, (1.0, 1.0, 1.0))
            mask   = grid > 0.02
            for c, ch in enumerate(rgb):
                composite[:, :, c] = np.where(
                    mask,
                    np.maximum(composite[:, :, c], grid * ch),
                    composite[:, :, c],
                )
            composite[:, :, 3] = np.where(
                mask,
                np.maximum(composite[:, :, 3], grid * alpha),
                composite[:, :, 3],
            )
            legend_patches.append(
                mpatches.Patch(color=rgb, label=f"{p}  ({len(xs):,} pts)")
            )
        ax.imshow(
            composite, extent=[0, 1, 1, 0], origin="upper",
            aspect="auto", interpolation="bilinear",
        )

    # Axes cosmetics
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=8)
    ax.set_xlabel("Normalised X  (left → right of frame)", fontsize=9, color="white")
    ax.set_ylabel("Normalised Y  (top → bottom of frame)", fontsize=9, color="white")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333333")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 6)],
                       color="#aaaaaa", fontsize=8)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{int(v*100)}%" for v in np.linspace(0, 1, 5)],
                       color="#aaaaaa", fontsize=8)

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  framealpha=0.55, facecolor="#1a1a2e", labelcolor="white")

    total_pts = sum(len(pose_pts[p][0]) for p in present)
    ax.text(0.01, 0.97, f"{total_pts:,} total detections",
            transform=ax.transAxes, fontsize=7, color="#666666", va="top")

    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {Path(out_path).name}")


# Fallback action→zone map used when no bbox is available in events.json
_PROXY_ZONES: dict[str, tuple[float, float]] = {
    "Stand/Idle": (0.50, 0.65), "Sit":        (0.40, 0.72),
    "Lying Down": (0.50, 0.85), "Crouch":     (0.50, 0.62),
    "Lean":       (0.30, 0.50), "Run":        (0.50, 0.38),
    "Walk":       (0.50, 0.55), "Sneak":      (0.22, 0.60),
    "Fall":       (0.50, 0.70), "Grab":       (0.62, 0.45),
    "Hit":        (0.55, 0.40), "Kick":       (0.45, 0.35),
    "Throw":      (0.72, 0.35), "Struggle":   (0.50, 0.50),
    "Gun":        (0.68, 0.40),
}


def _collect_pose_points(events: list[dict]) -> dict[str, tuple[list, list]]:
    """
    Return {pose: ([cx, ...], [cy, ...])} in normalised [0,1] frame coords.

    Reads from events records in priority order:
      1. "cx" / "cy"  float fields
      2. "bbox"       [x1,y1,x2,y2] normalised → derive centre
      3. Proxy zones  (Gaussian scatter around known CCTV hotspots)
         — prints a one-time advisory on how to add bbox to events writer
    """
    pose_pts: dict[str, tuple[list, list]] = {}

    has_spatial = any("cx" in e or "bbox" in e for e in events)
    if not has_spatial:
        print(
            "\n  [spatial] NOTE: events.json has no 'bbox'/'cx'/'cy' fields.\n"
            "  To get pixel-accurate spatial maps, add to _writer_thread in\n"
            "  infer-vid-5w-negus.py (inside the json_log.append block):\n"
            "      'bbox': person['bbox'],   # [x1,y1,x2,y2] normalised\n"
            "  Using zone-proxy positions for now.\n"
        )
        rng = random.Random(0)
        for e in events:
            pose   = _canon(e.get("action", "?"))
            mx, my = _PROXY_ZONES.get(pose, (0.5, 0.5))
            cx     = float(np.clip(rng.gauss(mx, 0.16), 0.02, 0.98))
            cy     = float(np.clip(rng.gauss(my, 0.11), 0.02, 0.98))
            xs, ys = pose_pts.setdefault(pose, ([], []))
            xs.append(cx); ys.append(cy)
    else:
        for e in events:
            pose   = _canon(e.get("action", "?"))
            cx, cy = None, None
            if "cx" in e and "cy" in e:
                cx, cy = float(e["cx"]), float(e["cy"])
            elif "bbox" in e:
                b  = e["bbox"]
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
            if cx is not None:
                xs, ys = pose_pts.setdefault(pose, ([], []))
                xs.append(cx); ys.append(cy)

    return pose_pts


def generate_spatial(
    events: list[dict],
    subtitle: str,
    base_out: Path,
    grid_w: int = 160,
    grid_h: int = 90,
    sigma: float = 4.0,
) -> None:
    if not events:
        print("  [skip] spatial heatmaps — no events loaded.")
        return

    pose_pts  = _collect_pose_points(events)
    all_poses = list(pose_pts.keys())
    static_p  = [p for p in all_poses if _group(p) == "static"]
    loco_p    = [p for p in all_poses if _group(p) == "locomotion"]
    action_p  = [p for p in all_poses if _group(p) == "actions"]
    kw        = dict(grid_w=grid_w, grid_h=grid_h, sigma=sigma)

    _spatial_single(
        pose_pts, all_poses,
        title=f"Spatial Occupancy — All Poses{subtitle}",
        out_path=str(base_out / "spatial_all.png"),
        cmap_name="hot", **kw,
    )
    _spatial_single(
        pose_pts, static_p,
        title=f"Spatial Occupancy — Static (Idle · Sit · Lie · Crouch · Lean){subtitle}",
        out_path=str(base_out / "spatial_static.png"),
        cmap_name="Blues", **kw,
    )
    _spatial_single(
        pose_pts, loco_p,
        title=f"Spatial Occupancy — Locomotion (Run · Walk · Sneak){subtitle}",
        out_path=str(base_out / "spatial_locomotion.png"),
        cmap_name="Greens", **kw,
    )
    _spatial_single(
        pose_pts, action_p,
        title=f"Spatial Occupancy — Actions (Fall · Hit · Kick · Grab · …){subtitle}",
        out_path=str(base_out / "spatial_actions.png"),
        cmap_name="Reds", **kw,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    summary_path: str,
    events_path:  str | None = None,
    outdir:       str | None = None,
) -> None:
    print(f"\n{'='*62}")
    print(f"  Pose Heatmap Generator  (v2 — temporal + spatial)")
    print(f"  Summary : {summary_path}")
    print(f"  Events  : {events_path or '(none)'}")
    print(f"{'='*62}\n")

    summary  = load_summary(summary_path)
    meta     = summary.get("meta", {})
    fps      = float(meta.get("src_fps", 25.0))
    src_name = Path(meta.get("input", "video")).stem
    mode_str = f"mode{meta.get('vision_mode', '?')}"
    dur_str  = f"{meta.get('duration_sec', 0):.0f}s"
    subtitle = f"\n[{src_name} · {mode_str} · {dur_str}]"

    base_out = Path(outdir) if outdir else Path(summary_path).parent
    base_out.mkdir(parents=True, exist_ok=True)

    print("  Building DataFrames…")
    try:
        tl_df = build_timeline_df(summary)
    except ValueError as e:
        print(f"  ERROR: {e}"); sys.exit(1)

    pres_df   = build_presence_df(summary)
    all_poses = list(tl_df.columns)
    static_c  = [c for c in all_poses if _group(c) == "static"]
    loco_c    = [c for c in all_poses if _group(c) == "locomotion"]
    action_c  = [c for c in all_poses if _group(c) == "actions"]
    vmax      = float(tl_df.max().max()) or 1.0
    fw        = max(14, len(tl_df) // 2)

    # ── TEMPORAL ──────────────────────────────────────────────────────────────
    print("\n  ── Temporal heatmaps ──────────────────────────────────────")

    print("  [1/6] Combined…")
    _temporal_heatmap(
        tl_df[all_poses] if all_poses else tl_df,
        title=f"All Poses — Person Count per Checkpoint{subtitle}",
        out_path=str(base_out / "heatmap_combined.png"),
        cmap=CMAP_COMBINED, vmax=vmax,
        figsize=(fw, max(5, len(all_poses) * 0.55 + 2)),
    )
    print("  [2/6] Static…")
    _temporal_heatmap(
        tl_df[static_c] if static_c else pd.DataFrame(),
        title=f"Static Poses (Idle · Sit · Lie · Crouch · Lean){subtitle}",
        out_path=str(base_out / "heatmap_static.png"),
        cmap=CMAP_STATIC, vmax=vmax,
        figsize=(fw, max(4, len(static_c) * 0.7 + 2)),
    )
    print("  [3/6] Locomotion…")
    _temporal_heatmap(
        tl_df[loco_c] if loco_c else pd.DataFrame(),
        title=f"Locomotion Poses (Run · Walk · Sneak){subtitle}",
        out_path=str(base_out / "heatmap_locomotion.png"),
        cmap=CMAP_LOCOMOTION, vmax=vmax,
        figsize=(fw, max(4, len(loco_c) * 0.7 + 2)),
    )
    print("  [4/6] Actions…")
    _temporal_heatmap(
        tl_df[action_c] if action_c else pd.DataFrame(),
        title=f"Action Poses (Fall · Grab · Hit · Kick · Throw · Struggle · Gun){subtitle}",
        out_path=str(base_out / "heatmap_actions.png"),
        cmap=CMAP_ACTIONS, vmax=vmax,
        figsize=(fw, max(4, len(action_c) * 0.7 + 2)),
    )
    print("  [5/6] Presence bar…")
    _presence_bar(
        pres_df,
        title=f"Pose Presence (% of frames active){subtitle}",
        out_path=str(base_out / "heatmap_presence_bar.png"),
    )
    print("  [6/6] LSTM events timeline…")
    events: list[dict] = []
    if events_path and Path(events_path).exists():
        events = load_events(events_path)
    _events_timeline(
        events,
        title=f"LSTM Event Density (5s bins){subtitle}",
        out_path=str(base_out / "heatmap_events_timeline.png"),
        fps=fps,
    )

    # ── SPATIAL ───────────────────────────────────────────────────────────────
    print("\n  ── Spatial KDE occupancy heatmaps ─────────────────────────")
    generate_spatial(events, subtitle, base_out)

    # ── Stats table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'Pose':<20} {'Presence%':>10}  {'Avg':>6}  {'Max':>4}  Group")
    print(f"  {'─'*20} {'─'*10}  {'─'*6}  {'─'*4}  {'─'*10}")
    for _, row in pres_df.iterrows():
        print(f"  {row['pose']:<20} {row['presence_pct']:>9.1f}%"
              f"  {row['avg_persons']:>6.2f}  {int(row['max_persons']):>4}"
              f"  {row['group']}")
    print(f"{'='*62}")
    print(f"  Output → {base_out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Demo data (with realistic bbox so spatial maps are pixel-accurate)
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data() -> tuple[dict, list[dict]]:
    rng = random.Random(42)
    all_actions = ["stand", "walk", "run", "sit", "lying_down",
                   "fall", "grab", "hit", "kick", "throw", "struggle", "sneak", "gun"]

    timeline = []
    for frame in range(0, 9000, 300):
        counts = {a: rng.randint(1, 5)
                  for a in rng.sample(all_actions, rng.randint(2, 6))}
        timeline.append({"frame": frame, "ts": round(frame / 25.0, 2), "counts": counts})

    poses = {
        a: {
            "avg_persons":   round(rng.uniform(0.5, 3.5), 2),
            "max_persons":   rng.randint(1, 8),
            "active_frames": (n := rng.randint(10, 900)),
            "total_frames":  1000,
            "presence_pct":  round(100 * n / 1000, 1),
        }
        for a in all_actions
    }

    _zones = {
        "stand":      (0.50, 0.65), "sit":        (0.40, 0.72),
        "lying_down": (0.50, 0.85), "walk":       (0.50, 0.55),
        "run":        (0.50, 0.38), "sneak":      (0.20, 0.60),
        "fall":       (0.50, 0.70), "grab":       (0.62, 0.45),
        "hit":        (0.55, 0.40), "kick":       (0.45, 0.35),
        "throw":      (0.72, 0.35), "struggle":   (0.50, 0.50),
        "gun":        (0.68, 0.40),
    }
    events = []
    for _ in range(500):
        act    = rng.choice(all_actions)
        mx, my = _zones.get(act, (0.5, 0.5))
        cx     = float(np.clip(rng.gauss(mx, 0.14), 0.02, 0.98))
        cy     = float(np.clip(rng.gauss(my, 0.09), 0.02, 0.98))
        hw, hh = rng.uniform(0.04, 0.10), rng.uniform(0.10, 0.28)
        bbox   = [round(v, 4) for v in [cx - hw, cy - hh, cx + hw, cy + hh]]
        events.append({
            "frame": rng.randint(0, 9000), "track_id": rng.randint(0, 4),
            "action": act, "confidence": round(rng.uniform(0.5, 1.0), 3),
            "ts": round(rng.uniform(0, 360), 2), "bbox": bbox,
        })

    summary = {
        "meta": {
            "input": "demo_cctv.mp4", "vision_mode": 2,
            "src_fps": 25.0, "duration_sec": 360.0,
            "frames_processed": 9000, "partial": False,
        },
        "poses": poses, "timeline": timeline,
    }
    return summary, events


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate temporal + spatial pose heatmaps from negus CCTV pipeline JSON.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python heatmap_pose.py --path output_annotated
  python heatmap_pose.py --path output_annotated.mp4
  python heatmap_pose.py --path output_annotated.summary.json
  python heatmap_pose.py --path C:/Videos/
  python heatmap_pose.py --summary foo.summary.json --events foo.events.json
  python heatmap_pose.py --demo --outdir ./reports
""",
    )
    parser.add_argument(
        "--path", default=None, metavar="PATH",
        help=(
            "Smart path — accepts any of:\n"
            "  bare base     :  output_annotated\n"
            "  video file    :  output_annotated.mp4\n"
            "  summary file  :  output_annotated.summary.json\n"
            "  events file   :  output_annotated.events.json\n"
            "  directory     :  C:/Videos/  (scans for *.summary.json)\n"
            "Sibling *.events.json is detected automatically."
        ),
    )
    parser.add_argument("--summary", default=None, metavar="FILE",
                        help="Explicit *.summary.json (overrides --path).")
    parser.add_argument("--events",  default=None, metavar="FILE",
                        help="Explicit *.events.json (optional).")
    parser.add_argument("--outdir",  default=None, metavar="DIR",
                        help="Output directory (default: same folder as summary).")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data — no JSON files needed.")
    args = parser.parse_args()

    if args.demo:
        print("\n  DEMO MODE — generating synthetic data with bbox…")
        summary_demo, events_demo = _make_demo_data()
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "demo.summary.json")
            ep = os.path.join(tmp, "demo.events.json")
            with open(sp, "w") as f: json.dump(summary_demo, f, indent=2)
            with open(ep, "w") as f: json.dump({"events": events_demo}, f, indent=2)
            generate_all(sp, ep, outdir=args.outdir or ".")
    else:
        sp, ep = resolve_input_paths(args.path, args.summary, args.events)
        generate_all(sp, ep, outdir=args.outdir)


if __name__ == "__main__":
    main()