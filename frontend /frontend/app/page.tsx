"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const API = "http://localhost:8000";

// ── Processing mode definitions ────────────────────────────────────────────────
type ProcessingMode = "full" | "compressed";

interface ModeConfig {
  id:          ProcessingMode;
  label:       string;
  tagline:     string;
  detail:      string;
  color:       string;
  accent:      string;
  border:      string;
  warning:     string;
}


const MODES: ModeConfig[] = [
  {
    id:      "full",
    label:   "Full Precision Analysis",
    tagline: "infer-vid-w7-sync.py — every frame processed",
    detail:  "YOLO runs on every frame. LSTM fires every eligible window. No frames dropped. Best for short clips or when accuracy is critical.",
    color:   "#2563eb",
    accent:  "#eff6ff",
    border:  "#bfdbfe",
    warning: "Full Precision processes every frame sequentially using infer-vid-w7-sync.py. This can take several minutes on long or high-resolution videos.",
  },
  {
    id:      "compressed",
    label:   "Fast Compressed Analysis",
    tagline: "infer-vid-w7-compress.py — frame-skip + LSTM cadence",
    detail:  "YOLO runs every 5th frame; detections are reused in between. LSTM fires every 4th eligible window. Frames are skipped under queue pressure. Best for long recordings where speed matters more than per-frame precision.",
    color:   "#ca8a04",
    accent:  "#fefce8",
    border:  "#fde68a",
    warning: "Compressed mode uses infer-vid-w7-compress.py and skips frames/reuses detections. Some fast-moving events between keyframes may be missed. Suitable for long-duration surveillance footage.",
  },
];

// ── Phase definitions ──────────────────────────────────────────────────────────
const PHASES = [
  { id: 1, label: "Observation", short: "Obs",    color: "#2563eb", accent: "#eff6ff", border: "#bfdbfe" },
  { id: 2, label: "Filter",      short: "Filter", color: "#16a34a", accent: "#f0fdf4", border: "#bbf7d0" },
  { id: 3, label: "Derivation",  short: "Deriv",  color: "#ca8a04", accent: "#fefce8", border: "#fde68a" },
  { id: 4, label: "Storage",     short: "Store",  color: "#ea580c", accent: "#fff7ed", border: "#fed7aa" },
];

// ── Heatmap data types (from backend /heatmap/{job_id}) ────────────────────────
interface GridData {
  type: "grid";
  title: string;
  rows: string[];
  cols: string[];
  values: number[][];
  raw_counts?: number[][];
  legend: string[];
  ramp: string[];
  seek_map?: Record<string | number, number>;
}

interface BarsData {
  type: "bars";
  title: string;
  rows: string[];
  values: number[];
  color: string;
  bgColor?: string;
}

interface TimelineData {
  type: "timeline";
  title: string;
  labels: string[];
  values: number[];
  color: string;
  seek_map?: Record<string | number, number>;
}

interface SpatialData {
  type?: "spatial";
  title: string;
  poses: Record<string, { xs: number[]; ys: number[] }>;
  ramp: string[];
}

type PanelData = GridData | BarsData | TimelineData | SpatialData;

interface PhaseData {
  label: string;
  panels: { id: string; data: PanelData }[];
}

interface HeatmapPayload {
  job_id: string;
  meta: {
    input: string;
    duration_sec: number;
    output_duration_sec: number;
    seek_ratio: number;          // output_dur / input_dur — 1.0 means no scaling
    frames_processed: number;
    n_events: number;
    n_poses: number;
  };
  phases: Record<string, PhaseData>;
}

// ── Heatmap helpers ────────────────────────────────────────────────────────────
function rampColor(value: number, ramp: string[]): string {
  const idx = Math.min(ramp.length - 1, Math.floor(value * ramp.length));
  return ramp[idx];
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec % 1) * 10);
  return `${m}:${String(s).padStart(2, "0")}.${ms}`;
}

interface SeekInfo { label: string; timestamp: number; formattedTime: string; rawSourceSec: number; }

// ── Grid heatmap component ─────────────────────────────────────────────────────
interface GridHeatmapProps {
  data: GridData;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  videoDuration?: number;
  onSeek?: (info: SeekInfo) => void;
  // NOTE: seek_map values are already OUTPUT-time (backend scaled them).
  // No client-side rescaling needed. inputDuration prop removed to prevent confusion.
}

function GridHeatmap({ data: d, videoRef, videoDuration, onSeek }: GridHeatmapProps) {
  const [activeCell, setActiveCell] = useState<string | null>(null);
  const canSeek = !!videoRef;

  const handleCellClick = useCallback((row: string, col: string, colIdx: number) => {
    const vid = videoRef?.current;
    if (!vid) { console.warn("[GridHeatmap] videoRef.current is null"); return; }

    // rawTs is already in OUTPUT-time (from backend's scale_seek_map)
    const rawTs = d.seek_map?.[colIdx] ?? d.seek_map?.[String(colIdx)];
    if (rawTs === undefined) return;

    // Backend has already rescaled seek_map from INPUT-time to OUTPUT-time.
    // rawTs is already in OUTPUT-time and can be used directly with the player.
    // If duration isn't known yet (NaN before metadata loads), don't clamp the
    // upper bound to 0 — that would force every seek to land at 0.
    const outDur = vid.duration && isFinite(vid.duration) ? vid.duration : Infinity;
    const seekTo = Math.min(Math.max(0, rawTs), outDur);

    // NOTE: rawTs is already output-video time from the backend's rescaling.
    // Do NOT rescale again here. Just clamp to video bounds.
    console.debug(
      `[GridHeatmap] rawTs=${rawTs.toFixed(3)}s  seekTo=${seekTo.toFixed(3)}s  outDur=${isFinite(outDur) ? outDur.toFixed(3) : "unknown"}s`
    );

    const wasPlaying = !vid.paused;
    vid.currentTime  = seekTo;
    if (wasPlaying) vid.play().catch(() => {});

    const cellKey = `${row}-${col}`;
    setActiveCell(cellKey);
    // If we need to send to /run-again, we'd need to reverse-scale this back
    // to INPUT-time. For now, we're only using it for video player seeking,
    // which expects OUTPUT-time.
    onSeek?.({ label: `${row} × ${col}`, timestamp: seekTo, formattedTime: fmtTime(seekTo), rawSourceSec: rawTs });
  }, [d.seek_map, videoRef, videoDuration, onSeek]);

  if (!d.rows.length || !d.cols.length) {
    return <div style={{ fontSize: 10, color: "#ccc", padding: "12px 0" }}>No data available for this view.</div>;
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#555", marginBottom: 8, fontFamily: "monospace" }}>{d.title}</div>
      {/* col labels */}
      <div style={{ display: "flex", marginLeft: 72, gap: 3, marginBottom: 3 }}>
        {d.cols.map((c, i) => (
          <div key={i} style={{ width: 32, fontSize: 9, color: "#888", textAlign: "center" }}>{c}</div>
        ))}
      </div>
      {d.rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex", alignItems: "center", gap: 3, marginBottom: 3 }}>
          <div style={{ width: 68, fontSize: 9, color: "#888", textAlign: "right", paddingRight: 6, flexShrink: 0 }}>{row}</div>
          {d.values[ri].map((v, ci) => {
            const cellKey = `${row}-${d.cols[ci]}`;
            const isActive = activeCell === cellKey;
            const rawVal = d.raw_counts?.[ri]?.[ci];
            const seekTs = d.seek_map?.[ci] ?? d.seek_map?.[String(ci)];
            const seekHint = canSeek && seekTs !== undefined
              ? ` — click to seek`
              : "";
            const tooltip = `${row} × ${d.cols[ci]}: ${rawVal ?? Math.round(v * 100)}${seekHint}`;
            return (
              <div key={ci}
                title={tooltip}
                onClick={() => handleCellClick(row, d.cols[ci], ci)}
                style={{
                  width: 32, height: 22, borderRadius: 3, flexShrink: 0,
                  background: rampColor(v, d.ramp),
                  cursor: canSeek && seekTs !== undefined ? "pointer" : "default",
                  transition: "transform .1s, box-shadow .1s",
                  outline: isActive ? "2.5px solid #111" : "none",
                  outlineOffset: "1px",
                  boxShadow: isActive ? "0 0 0 1px #fff" : "none",
                }}
                onMouseEnter={e => { e.currentTarget.style.transform = "scale(1.3)"; e.currentTarget.style.zIndex = "10"; }}
                onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; e.currentTarget.style.zIndex = ""; }}
              />
            );
          })}
        </div>
      ))}
      {/* legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10, fontSize: 9, color: "#aaa", marginLeft: 72 }}>
        <span>{d.legend[0]}</span>
        {d.ramp.map((c, i) => <div key={i} style={{ width: 18, height: 10, borderRadius: 2, background: c }} />)}
        <span>{d.legend[1]}</span>
      </div>
    </div>
  );
}

// ── Bars heatmap component ─────────────────────────────────────────────────────
function BarsHeatmap({ data: d }: { data: BarsData }) {
  if (!d.rows.length) {
    return <div style={{ fontSize: 10, color: "#ccc", padding: "12px 0" }}>No data available for this view.</div>;
  }
  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#555", marginBottom: 8, fontFamily: "monospace" }}>{d.title}</div>
      {d.rows.map((row, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7 }}>
          <div style={{ width: 130, fontSize: 10, color: "#666", flexShrink: 0, textAlign: "right" }}>{row}</div>
          <div style={{ flex: 1, height: 22, background: "#f0f0ea", borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              width: `${Math.min(d.values[i], 100)}%`, height: "100%", borderRadius: 4,
              background: `${d.color}${Math.round(0.3 * 255 + (d.values[i] / 100) * 0.5 * 255).toString(16).padStart(2, "0")}`,
              display: "flex", alignItems: "center", paddingLeft: 8,
              fontSize: 10, fontWeight: 700, color: d.color,
              transition: "width 1s ease",
            }}>
              {d.values[i] > 25 ? `${d.values[i]}%` : ""}
            </div>
          </div>
          <div style={{ width: 32, fontSize: 10, color: "#888", textAlign: "right", flexShrink: 0 }}>{d.values[i]}%</div>
        </div>
      ))}
    </div>
  );
}

// ── Timeline heatmap component ─────────────────────────────────────────────────
interface TimelineHeatmapProps {
  data: TimelineData;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  videoDuration?: number;
  // NOTE: seek_map values are already OUTPUT-time (backend scaled). No rescaling here.
  onSeek?: (info: SeekInfo) => void;
}

function TimelineHeatmap({ data: d, videoRef, videoDuration, onSeek }: TimelineHeatmapProps) {
  const maxV = Math.max(...d.values);
  const total = d.values.reduce((a, b) => a + b, 0);
  const [activeBar, setActiveBar] = useState<number | null>(null);
  const canSeek = !!videoRef;

  const handleBarClick = useCallback((i: number, label: string) => {
    const vid = videoRef?.current;
    if (!vid) { console.warn("[TimelineHeatmap] videoRef.current is null"); return; }

    const rawTs = d.seek_map?.[i] ?? d.seek_map?.[String(i)];
    if (rawTs === undefined) return;

    // Backend has already rescaled seek_map from INPUT-time to OUTPUT-time.
    // rawTs is already in OUTPUT-time and can be used directly with the player.
    // If duration isn't known yet (NaN before metadata loads), don't clamp the
    // upper bound to 0 — that would force every seek to land at 0.
    const outDur = vid.duration && isFinite(vid.duration) ? vid.duration : Infinity;
    const seekTo = Math.min(Math.max(0, rawTs), outDur);

    // NOTE: rawTs is already output-video time from the backend's rescaling.
    // Do NOT rescale again here. Just clamp to video bounds.
    console.debug(
      `[TimelineHeatmap] rawTs=${rawTs.toFixed(3)}s  seekTo=${seekTo.toFixed(3)}s`
    );

    const wasPlaying = !vid.paused;
    vid.currentTime  = seekTo;
    if (wasPlaying) vid.play().catch(() => {});

    setActiveBar(i);
    onSeek?.({ label, timestamp: seekTo, formattedTime: fmtTime(seekTo), rawSourceSec: rawTs });
  }, [d.seek_map, videoRef, videoDuration, onSeek]);

  if (!d.labels.length) {
    return <div style={{ fontSize: 10, color: "#ccc", padding: "12px 0" }}>No data available for this view.</div>;
  }

  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#555", marginBottom: 8, fontFamily: "monospace" }}>{d.title}</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 8, height: 110, marginBottom: 6 }}>
        {d.values.map((v, i) => (
          <div key={i}
            style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4, cursor: canSeek && (d.seek_map?.[i] ?? d.seek_map?.[String(i)]) !== undefined ? "pointer" : "default" }}
            onClick={() => handleBarClick(i, d.labels[i])}
            title={(() => {
              const seekTs = d.seek_map?.[i] ?? d.seek_map?.[String(i)];
              return canSeek && seekTs !== undefined
                ? `${d.labels[i]}: ${v} — click to seek`
                : `${d.labels[i]}: ${v}`;
            })()}
          >
            <div style={{ fontSize: 9, color: "#888" }}>{v}</div>
            <div style={{
              width: "100%", height: `${maxV > 0 ? Math.max(Math.round((v / maxV) * 80), 2) : 0}px`,
              background: d.color, borderRadius: "3px 3px 0 0",
              opacity: 0.5 + (maxV > 0 ? (v / maxV) * 0.5 : 0), transition: "height .8s ease",
              outline: activeBar === i ? "2.5px solid #111" : "none",
              outlineOffset: "1px",
            }} />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        {d.labels.map((l, i) => (
          <div key={i} style={{ flex: 1, fontSize: 9, color: activeBar === i ? d.color : "#aaa", textAlign: "center", fontWeight: activeBar === i ? 700 : 400 }}>{l}</div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 20, marginTop: 10, fontSize: 10, color: "#888" }}>
        <span>Peak: <strong style={{ color: d.color }}>{maxV}</strong></span>
        <span>Total: <strong style={{ color: d.color }}>{total}</strong></span>
        <span>Avg: <strong style={{ color: d.color }}>{d.values.length > 0 ? Math.round(total / d.values.length) : 0}</strong></span>
      </div>
    </div>
  );
}

// ── Spatial occupancy component ─────────────────────────────────────────────────
function SpatialOccupancy({ data: d }: { data: SpatialData }) {
  const poses = d.poses || {};
  const poseNames = Object.keys(poses);
  const totalPoints = poseNames.reduce((sum, p) => sum + (poses[p]?.xs?.length ?? 0), 0);

  if (!poseNames.length) {
    return <div style={{ fontSize: 10, color: "#ccc", padding: "12px 0" }}>No spatial data available (bbox not in events).</div>;
  }

  return (
    <div>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#555", marginBottom: 4, fontFamily: "monospace" }}>{d.title}</div>
      <div style={{ fontSize: 9, color: "#aaa", marginBottom: 8 }}>{totalPoints.toLocaleString()} detection points across {poseNames.length} poses</div>
      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {poseNames.map((pose) => {
          const xs = poses[pose].xs;
          const ys = poses[pose].ys;
          const n = xs.length;
          const meanX = n > 0 ? xs.reduce((a, b) => a + b, 0) / n : 0.5;
          const meanY = n > 0 ? ys.reduce((a, b) => a + b, 0) / n : 0.5;
          return (
            <div key={pose} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <div style={{ width: 80, fontSize: 9, color: "#666", textAlign: "right", flexShrink: 0 }}>{pose}</div>
              <div style={{ flex: 1, height: 14, background: "#f0f0ea", borderRadius: 3, overflow: "hidden", position: "relative" }}>
                <div style={{
                  width: `${Math.min((meanX ?? 0.5) * 100, 100)}%`, height: "100%",
                  background: "#3b82f6", borderRadius: 3,
                  opacity: 0.3 + (n / Math.max(...Object.values(poses).map(p => p.xs.length), 1)) * 0.7,
                  transition: "width 0.6s ease",
                }} />
                <div style={{
                  position: "absolute", left: `${Math.min((meanX ?? 0.5) * 100, 100)}%`,
                  top: 0, width: 2, height: "100%", background: "#1e3a8a",
                  transform: "translateX(-50%)",
                }} />
              </div>
              <div style={{ fontSize: 8, color: "#bbb", width: 40, textAlign: "right" }}>{n} pts</div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 8, color: "#bbb", marginTop: 6 }}>
        Bar = relative density, tick = mean X position (normalized 0–1)
      </div>
    </div>
  );
}

// ── Heatmap panel (per-phase accordion content) ────────────────────────────────
interface HeatmapPanelProps {
  phaseId: number;
  phaseData: PhaseData | undefined;
  onClose: () => void;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  videoDuration?: number;
  lastSeek?: SeekInfo | null;
  onSeek?: (info: SeekInfo) => void;
  onLockTarget?: (outputTimeSec: number) => void;  // OUTPUT-time from scaled seek_map
}

function HeatmapPanel({ phaseId, phaseData, onClose, videoRef, videoDuration, lastSeek, onSeek, onLockTarget }: HeatmapPanelProps) {
  const phase = PHASES[phaseId - 1];
  const canSeek = !!videoRef;
  const panels = phaseData?.panels ?? [];

  // Wrap onSeek so every click also locks the OUTPUT-time timestamp for Run Again.
  // The backend /run-again endpoint reverse-scales it to INPUT-time automatically.
  const handleSeekWithLock = useCallback((info: SeekInfo, outputTimeSec: number) => {
    onSeek?.(info);
    if (outputTimeSec !== undefined) {
      onLockTarget?.(outputTimeSec);
      console.debug(`[HeatmapPanel] target locked -> ${outputTimeSec.toFixed(3)}s (OUTPUT-time; backend will reverse-scale)`);
    }
  }, [onSeek, onLockTarget]);

  return (
    <div style={{
      background: "#fff", border: `1.5px solid ${phase.border}`,
      borderRadius: 10, padding: "16px 20px", marginBottom: 12,
      animation: "slideDown .2s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: phase.color }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: phase.color, fontFamily: "monospace" }}>
            P{phaseId} — {phaseData?.label ?? phase.label}
          </span>
        </div>
        <button onClick={onClose} style={{
          background: "none", border: "none", cursor: "pointer",
          fontSize: 16, color: "#bbb", lineHeight: 1, padding: "0 4px",
        }}>×</button>
      </div>

      {/* seek hint */}
      {canSeek && panels.some(p => p.data.type === "grid" || p.data.type === "timeline") && (
        <div style={{
          fontSize: 9, color: "#bbb", marginBottom: 10,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{ fontFamily: "monospace" }}>⌖ click cell/bar → seek video</span>
          {lastSeek && (
            <span style={{
              background: phase.accent, border: `1px solid ${phase.border}`,
              borderRadius: 4, padding: "2px 8px", color: phase.color, fontWeight: 700, fontFamily: "monospace",
              animation: "slideDown .2s ease",
            }}>
              ↳ {lastSeek.label} @ {lastSeek.formattedTime}
            </span>
          )}
        </div>
      )}

      {panels.length === 0 && (
        <div style={{ fontSize: 10, color: "#bbb", padding: "8px 0" }}>
          No heatmap data available for this phase.
        </div>
      )}

      {panels.map((panel) => (
        <div key={panel.id} style={{ marginBottom: panels.length > 1 ? 16 : 0 }}>
          {panel.data.type === "grid" && (
            <GridHeatmap data={panel.data as GridData} videoRef={videoRef} videoDuration={videoDuration}
              onSeek={(info) => handleSeekWithLock(info, info.rawSourceSec)} />
          )}
          {panel.data.type === "bars" && (
            <BarsHeatmap data={panel.data as BarsData} />
          )}
          {panel.data.type === "timeline" && (
            <TimelineHeatmap data={panel.data as TimelineData} videoRef={videoRef} videoDuration={videoDuration}
              onSeek={(info) => handleSeekWithLock(info, info.rawSourceSec)} />
          )}
          {(panel.data as SpatialData).poses !== undefined && panel.data.type !== "grid" && panel.data.type !== "bars" && panel.data.type !== "timeline" && (
            <SpatialOccupancy data={panel.data as SpatialData} />
          )}
        </div>
      ))}
    </div>
  );
}

// ── Log guessing ───────────────────────────────────────────────────────────────
function guessPhase(logLines: string[]): number {
  const last = [...logLines].reverse().find(Boolean) || "";
  if (/phase.?3|deriv/i.test(last))           return 3;
  if (/phase.?2|filter|promot/i.test(last))   return 2;
  if (/phase.?4|heatmap|sqlite|storage/i.test(last)) return 4;
  if (/frame|yolo|lstm|delta|track/i.test(last)) return 1;
  return 1;
}

// ── Phase pill ─────────────────────────────────────────────────────────────────
function PhasePip({ phase, active, done }: { phase: typeof PHASES[0]; active: boolean; done: boolean }) {
  const bg     = done ? phase.color + "22" : active ? phase.color + "14" : "#f0f0ea";
  const border = done || active ? phase.color : "#e0e0da";
  const text   = done || active ? phase.color : "#bbb";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 10px",
      borderRadius: 5, border: `1px solid ${border}`, background: bg }}>
      {done
        ? <span style={{ fontSize: 10, color: phase.color }}>✓</span>
        : <div style={{ width: 7, height: 7, borderRadius: "50%",
            background: active ? phase.color : "#ddd",
            animation: active ? "blink 1.2s infinite" : "none" }} />}
      <span style={{ fontSize: 10, fontFamily: "monospace", color: text, fontWeight: 700 }}>P{phase.id}</span>
    </div>
  );
}

// ── Log panel ──────────────────────────────────────────────────────────────────
function LogPanel({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines]);
  return (
    <div ref={ref} style={{ flex: 1, overflowY: "auto", padding: "8px 12px",
      fontFamily: "monospace", fontSize: 10, color: "#555", lineHeight: 1.7, background: "#fafaf8" }}>
      {lines.length === 0
        ? <span style={{ color: "#ccc" }}>— waiting for output —</span>
        : lines.map((l, i) => {
            const isWarn = /warning|warn/i.test(l);
            const isErr  = /error|fail/i.test(l);
            const col    = isErr ? "#dc2626" : isWarn ? "#ca8a04" : "#555";
            return <div key={i} style={{ color: col }}>{l || " "}</div>;
          })}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function Page() {
  const [file,           setFile]           = useState<File | null>(null);
  const [jobId,          setJobId]          = useState<string | null>(null);
  const [status,         setStatus]         = useState<"idle"|"uploading"|"running"|"done"|"error">("idle");
  const [logLines,       setLogLines]       = useState<string[]>([]);
  const [outputUrl,      setOutputUrl]      = useState<string | null>(null);
  const [errMsg,         setErrMsg]         = useState("");
  const [activePhase,    setActivePhase]    = useState(0);
  const [donePhases,     setDonePhases]     = useState<number[]>([]);
  const [openHeatmap,    setOpenHeatmap]    = useState<number | null>(null);
  const [lastSeek,       setLastSeek]       = useState<SeekInfo | null>(null);
  const [videoDuration,  setVideoDuration]  = useState<number>(0);
  const [progress,       setProgress]       = useState<number>(0);
  const [heatmapData,    setHeatmapData]    = useState<HeatmapPayload | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);

  // ── Processing mode ──────────────────────────────────────────────────────────
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("full");
  const [estMinutes,     setEstMinutes]     = useState<number | null>(null);
  const [showWarning,    setShowWarning]    = useState(false);   // confirmation modal

  // ── Precision seek state ────────────────────────────────────────────────────
  // targetSecond: the raw SOURCE-video second locked by the last heatmap click.
  // This is sent to /run-again so the backend jumps to the exact frame.
  const [targetSecond,   setTargetSecond]   = useState<number | null>(null);

  const fileRef  = useRef<HTMLInputElement>(null);
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // ── Fetch heatmap data when job completes ────────────────────────────────────
  const fetchHeatmap = useCallback(async (id: string) => {
    setHeatmapLoading(true);
    try {
      const res = await fetch(`${API}/heatmap/${id}`);
      if (res.ok) {
        const data = await res.json();
        setHeatmapData(data);
      }
    } catch {
      // silently fail — heatmap is optional enhancement
    } finally {
      setHeatmapLoading(false);
    }
  }, []);

  // ── polling ──────────────────────────────────────────────────────────────────
  const startPolling = useCallback((id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/pipeline-status/${id}`);
        const data = await res.json();
        setLogLines(data.log_tail || []);
        if (typeof data.progress === "number") setProgress(data.progress);
        const gp = guessPhase(data.log_tail || []);
        setActivePhase(gp);
        setDonePhases(Array.from({ length: gp - 1 }, (_, i) => i + 1));
        if (data.status === "done") {
          clearInterval(pollRef.current!);
          setStatus("done");
          setProgress(100);
          setOutputUrl(API + data.output_url);
          setActivePhase(0);
          setDonePhases([1, 2, 3, 4]);
          // Fetch real heatmap data
          fetchHeatmap(id);
        }
        if (data.status === "error") {
          clearInterval(pollRef.current!);
          setStatus("error");
          setProgress(0);
          setErrMsg(data.error || "Pipeline failed");
        }
      } catch { /* keep polling on blip */ }
    }, 1500);
  }, [fetchHeatmap]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ── handlers ─────────────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setStatus("idle"); setOutputUrl(null);
    setLogLines([]); setErrMsg(""); setDonePhases([]); setActivePhase(0); setJobId(null);
    setHeatmapData(null);
  };

  // ── Run: show warning first, then actually submit ────────────────────────────
  const handleRunConfirm = async () => {
    setShowWarning(false);
    if (!file) return;
    setStatus("uploading"); setLogLines([]); setOutputUrl(null);
    setErrMsg(""); setDonePhases([]); setActivePhase(1);
    setHeatmapData(null); setEstMinutes(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("processing_mode", processingMode);
      const res  = await fetch(`${API}/run-pipeline`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "Upload failed");
      setJobId(data.job_id);
      if (typeof data.est_minutes === "number") setEstMinutes(data.est_minutes);
      setStatus("running");
      startPolling(data.job_id);
    } catch (e: any) {
      setStatus("error");
      setErrMsg(e.message);
    }
  };

  const handleRun = () => {
    if (!file) return;
    setShowWarning(true);   // always show warning/confirmation before submitting
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setFile(null); setJobId(null); setStatus("idle");
    setLogLines([]); setOutputUrl(null); setErrMsg("");
    setActivePhase(0); setDonePhases([]); setOpenHeatmap(null);
    setLastSeek(null); setVideoDuration(0); setProgress(0);
    setHeatmapData(null); setTargetSecond(null);
    setEstMinutes(null); setShowWarning(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  // ── Run Again: re-run pipeline at the heatmap-clicked source timestamp ──────
  const handleRunAgain = useCallback(async () => {
    if (!jobId || targetSecond === null) return;
    setStatus("running"); setLogLines([]); setOutputUrl(null);
    setErrMsg(""); setDonePhases([]); setActivePhase(1);
    setHeatmapData(null); setProgress(0);
    try {
      const res  = await fetch(`${API}/run-again`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId, target_second: targetSecond }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || "run-again failed");
      setJobId(data.job_id);
      startPolling(data.job_id);
      console.log(`[RunAgain] new job ${data.job_id} targeting ${targetSecond.toFixed(3)}s`);
    } catch (e: any) {
      setStatus("error");
      setErrMsg(e.message);
    }
  }, [jobId, targetSecond, startPolling]);

  const toggleHeatmap = (id: number) =>
    setOpenHeatmap(prev => (prev === id ? null : id));

  // ── derived state ────────────────────────────────────────────────────────────
  const isIdle      = status === "idle";
  const isUploading = status === "uploading";
  const isRunning   = status === "running";
  const isDone      = status === "done";
  const isError     = status === "error";
  const isBusy      = isUploading || isRunning;

  const btnLabel = isUploading ? "Uploading…"
                 : isRunning   ? "Running…"
                 : isDone      ? "Run Again"
                 : "Run Pipeline";

  const btnColor = isError ? "#dc2626" : isDone ? "#16a34a" : "#2563eb";

  // ── render ───────────────────────────────────────────────────────────────────
  return (
    <div style={{ fontFamily: "'JetBrains Mono','Fira Code',monospace",
      background: "#f5f5f0", color: "#222", minHeight: "100vh",
      display: "flex", flexDirection: "column" }}>

      {/* ── top bar ──────────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "0 16px",
        height: 44, borderBottom: "1px solid #e0e0da", background: "#fff", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%",
            background: isDone ? "#16a34a" : isBusy ? "#2563eb" : "#ddd",
            animation: isBusy ? "blink 1.2s infinite" : "none" }} />
          <span style={{ fontSize: 11, fontWeight: 700,
            color: isDone ? "#16a34a" : isBusy ? "#2563eb" : "#ccc" }}>
            {isDone ? "DONE" : isBusy ? "LIVE" : "IDLE"}
          </span>
        </div>
        <span style={{ fontSize: 11, color: "#bbb" }}>
          Pose Pipeline v7 · {processingMode === "compressed" ? "infer-vid-w7-compress.py" : "infer-vid-w7-sync.py"}
        </span>
        <div style={{ display: "flex", gap: 5, marginLeft: "auto" }}>
          {PHASES.map(p => (
            <PhasePip key={p.id} phase={p}
              active={activePhase === p.id}
              done={donePhases.includes(p.id)} />
          ))}
        </div>
      </div>

      {/* ── body ─────────────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>

        {/* ── left panel ──────────────────────────────────────────────────────── */}
        <div style={{ width: 240, borderRight: "1px solid #e8e8e2",
          background: "#fafaf8", display: "flex", flexDirection: "column",
          padding: "16px 14px", gap: 12, flexShrink: 0 }}>

          <div style={{ fontSize: 10, color: "#bbb", letterSpacing: 1 }}>VIDEO INPUT</div>

          {/* file picker */}
          <input ref={fileRef} type="file" accept="video/*,.mp4,.avi,.mkv,.mov"
            style={{ display: "none" }} onChange={handleFileChange} />
          <button onClick={() => fileRef.current?.click()} disabled={isBusy}
            style={{ width: "100%", padding: "13px 0", borderRadius: 7, cursor: "pointer",
              border: "1px solid #bfdbfe", background: "#eff6ff",
              color: "#2563eb", fontSize: 12, fontWeight: 700, letterSpacing: 1,
              opacity: isBusy ? 0.5 : 1 }}>
            ▲ SELECT VIDEO
          </button>

          {/* filename */}
          {file && (
            <div style={{ fontSize: 9, color: "#888", wordBreak: "break-all",
              background: "#f0f0ea", borderRadius: 4, padding: "5px 9px",
              border: "1px solid #e8e8e2" }}>
              {file.name}
              <span style={{ color: "#bbb", marginLeft: 4 }}>
                ({(file.size / 1024 / 1024).toFixed(1)} MB)
              </span>
            </div>
          )}

          {/* ── Processing mode picker — shown after file is selected ── */}
          {file && !isBusy && !isDone && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ fontSize: 10, color: "#bbb", letterSpacing: 1, marginTop: 2 }}>
                PROCESSING MODE
              </div>
              {MODES.map(m => {
                const active = processingMode === m.id;
                return (
                  <button key={m.id}
                    onClick={() => setProcessingMode(m.id)}
                    style={{
                      width: "100%", padding: "10px 10px", borderRadius: 7,
                      cursor: "pointer", textAlign: "left",
                      border: `1.5px solid ${active ? m.color : "#e8e8e2"}`,
                      background: active ? m.accent : "transparent",
                      transition: "all 0.15s",
                    }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{
                        width: 10, height: 10, borderRadius: "50%",
                        border: `2px solid ${m.color}`,
                        background: active ? m.color : "transparent",
                        flexShrink: 0,
                      }} />
                      <span style={{ fontSize: 11, fontWeight: 700, color: active ? m.color : "#555" }}>
                        {m.label}
                      </span>
                    </div>
                    <div style={{ fontSize: 9, color: "#888", marginTop: 4, paddingLeft: 16, lineHeight: 1.5 }}>
                      {m.tagline}
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* est time shown while running */}
          {isBusy && estMinutes !== null && (
            <div style={{ fontSize: 9, color: "#ca8a04", background: "#fefce8",
              border: "1px solid #fde68a", borderRadius: 4, padding: "5px 9px", lineHeight: 1.6 }}>
              ⏱ Est. ~{estMinutes} min remaining
            </div>
          )}

          {/* run / run-again button */}
          {isDone && targetSecond !== null ? (
            <button
              onClick={handleRunAgain}
              disabled={isBusy}
              style={{ width: "100%", padding: "14px 0", borderRadius: 7,
                cursor: isBusy ? "not-allowed" : "pointer",
                border: "1px solid #16a34a55",
                background: isBusy ? "#f0f0ea" : "#16a34a18",
                color: isBusy ? "#ccc" : "#16a34a",
                fontSize: 12, fontWeight: 700, letterSpacing: 1, transition: "all 0.2s",
                lineHeight: 1.4 }}>
              ▶ Run Again<br />
              <span style={{ fontSize: 10, fontWeight: 400, opacity: 0.8 }}>
                @ {targetSecond.toFixed(2)}s
              </span>
            </button>
          ) : (
          <button
            onClick={isDone ? handleReset : handleRun}
            disabled={!file || isBusy}
            style={{ width: "100%", padding: "14px 0", borderRadius: 7,
              cursor: !file || isBusy ? "not-allowed" : "pointer",
              border: `1px solid ${btnColor}55`,
              background: !file || isBusy ? "#f0f0ea" : btnColor + "18",
              color: !file || isBusy ? "#ccc" : btnColor,
              fontSize: 13, fontWeight: 700, letterSpacing: 1, transition: "all 0.2s" }}>
            {btnLabel}
          </button>
          )}

          {/* reset */}
          {(isDone || isError) && (
            <button onClick={handleReset} style={{ width: "100%", padding: "10px 0",
              borderRadius: 7, cursor: "pointer", border: "1px solid #e8e8e2",
              background: "transparent", color: "#bbb", fontSize: 11 }}>
              ↺ Reset
            </button>
          )}

          {/* error */}
          {isError && (
            <div style={{ fontSize: 9, color: "#dc2626", background: "#fef2f2",
              border: "1px solid #fecaca", borderRadius: 4, padding: "6px 8px", lineHeight: 1.6 }}>
              <strong>Error:</strong><br />{errMsg}
            </div>
          )}

          <div style={{ borderTop: "1px solid #e8e8e2", marginTop: 2 }} />

          {/* ── per-phase heatmap buttons ────────────────────────────────────── */}
          <div style={{ fontSize: 10, color: "#bbb", letterSpacing: 1 }}>
            HEATMAP {heatmapLoading && <span style={{ fontSize: 8, color: "#2563eb" }}>…loading</span>}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {PHASES.map(p => (
              <button key={p.id} onClick={() => toggleHeatmap(p.id)}
                style={{
                  width: "100%", padding: "10px 12px", borderRadius: 7,
                  border: `1px solid ${openHeatmap === p.id ? p.color : p.border}`,
                  background: openHeatmap === p.id ? p.color + "18" : p.accent,
                  color: p.color, fontSize: 11, fontWeight: 700,
                  cursor: "pointer", textAlign: "left",
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  transition: "all .15s",
                }}>
                <span>P{p.id} — {p.label}</span>
                <span style={{ fontSize: 12, opacity: 0.6 }}>
                  {openHeatmap === p.id ? "▲" : "▼"}
                </span>
              </button>
            ))}
          </div>

          <div style={{ borderTop: "1px solid #e8e8e2", marginTop: 2 }} />

          {/* info */}
          <div style={{ fontSize: 9, color: "#bbb", lineHeight: 1.9 }}>
            <div><span style={{ color: "#2563eb" }}>Script:</span></div>
            <div style={{ wordBreak: "break-all", color: "#999" }}>
              {processingMode === "compressed" ? "infer-vid-w7-compress.py" : "infer-vid-w7-sync.py"}
            </div>
            <div style={{ marginTop: 4 }}><span style={{ color: "#2563eb" }}>Mode:</span> 2 (LSTM+Thresh)</div>
            <div><span style={{ color: "#2563eb" }}>Device:</span> cuda</div>
            {jobId && (
              <div style={{ marginTop: 6 }}>
                <span style={{ color: "#2563eb" }}>Job:</span><br />
                <span style={{ color: "#aaa" }}>{jobId.slice(0, 8)}…</span>
              </div>
            )}
            {heatmapData?.meta && (
              <div style={{ marginTop: 6 }}>
                <span style={{ color: "#16a34a" }}>Data:</span><br />
                <span style={{ color: "#aaa" }}>
                  {heatmapData.meta.n_events.toLocaleString()} events, {heatmapData.meta.n_poses} poses
                </span>
              </div>
            )}
            {heatmapData?.meta && heatmapData.meta.seek_ratio !== undefined && (
              <div style={{ marginTop: 6 }}>
                <span style={{ color: heatmapData.meta.seek_ratio === 1.0 ? "#bbb" : "#ca8a04" }}>
                  Seek ratio:
                </span><br />
                <span style={{ color: "#aaa", fontFamily: "monospace" }}>
                  {heatmapData.meta.seek_ratio.toFixed(4)}×
                  {heatmapData.meta.seek_ratio !== 1.0
                    ? ` (${Math.round(heatmapData.meta.output_duration_sec)}s out / ${Math.round(heatmapData.meta.duration_sec)}s in)`
                    : " (1:1)"}
                </span>
              </div>
            )}
            {targetSecond !== null && (
              <div style={{ marginTop: 6, padding: "5px 8px",
                background: "#16a34a12", border: "1px solid #16a34a44",
                borderRadius: 5 }}>
                <span style={{ color: "#16a34a", fontWeight: 700 }}>Target locked:</span><br />
                <span style={{ color: "#16a34a", fontFamily: "monospace", fontSize: 10 }}>
                  {targetSecond.toFixed(3)}s
                </span>
                <button onClick={() => setTargetSecond(null)}
                  style={{ float: "right", background: "none", border: "none",
                    cursor: "pointer", color: "#bbb", fontSize: 12, lineHeight: 1 }}>×</button>
              </div>
            )}
          </div>
        </div>

        {/* ── centre ────────────────────────────────────────────────────────── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>

          {/* HUD strip */}
          <div style={{ height: 36, background: "#fafaf8", borderBottom: "1px solid #e8e8e2",
            display: "flex", alignItems: "center", padding: "0 16px", gap: 12, flexShrink: 0 }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%",
              background: isBusy ? "#dc2626" : isDone ? "#16a34a" : "#ddd",
              display: "inline-block", animation: isBusy ? "blink 1.2s infinite" : "none" }} />
            <span style={{ fontSize: 10, color: "#aaa", letterSpacing: 2 }}>
              {isBusy ? "REC" : isDone ? "PLAYBACK" : "STANDBY"}
            </span>
            <span style={{ fontSize: 11, color: "#666", fontWeight: 700 }}>camera_01</span>
            <span style={{ marginLeft: "auto", fontSize: 10, color: "#bbb" }}>MODE 2 · v7</span>
          </div>

          {/* heatmap panels (inline above main content) */}
          {openHeatmap !== null && (
            <div style={{ padding: "12px 16px 0", flexShrink: 0, maxHeight: "42vh", overflowY: "auto" }}>
              <HeatmapPanel
                phaseId={openHeatmap}
                phaseData={heatmapData?.phases?.[String(openHeatmap)]}
                onClose={() => { setOpenHeatmap(null); setLastSeek(null); }}
                videoRef={videoRef}
                videoDuration={videoDuration}
                lastSeek={lastSeek}
                onSeek={setLastSeek}
                onLockTarget={setTargetSecond}
              />
            </div>
          )}

          {/* main content */}
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            background: "#f5f5f0", overflow: "hidden", position: "relative" }}>

            {isIdle && !file && (
              <div style={{ textAlign: "center", color: "#ccc", lineHeight: 2.4 }}>
                <div style={{ fontSize: 48, opacity: 0.1 }}>▶</div>
                <div style={{ fontSize: 11, letterSpacing: 2 }}>NO SIGNAL</div>
                <div style={{ fontSize: 10 }}>Select a video and run the pipeline</div>
              </div>
            )}

            {isIdle && file && (
              <div style={{ textAlign: "center", color: "#aaa", lineHeight: 2.4 }}>
                <div style={{ fontSize: 36, opacity: 0.18 }}>🎞</div>
                <div style={{ fontSize: 11 }}>{file.name}</div>
                <div style={{ fontSize: 10, color: "#ccc" }}>
                  Press <strong style={{ color: "#2563eb" }}>Run Pipeline</strong> to start
                </div>
              </div>
            )}

            {isUploading && (
              <div style={{ textAlign: "center", color: "#2563eb", lineHeight: 2.4 }}>
                <div style={{ fontSize: 11, letterSpacing: 2 }}>UPLOADING…</div>
                <div style={{ fontSize: 10, color: "#bbb" }}>{file?.name}</div>
              </div>
            )}

            {isRunning && (
              <div style={{ textAlign: "center", color: "#aaa", lineHeight: 2.8 }}>
                <div style={{ fontSize: 36, opacity: 0.08, marginBottom: 8 }}>⚙</div>
                <div style={{ fontSize: 11, color: "#2563eb", letterSpacing: 3, fontWeight: 700 }}>PROCESSING…</div>
                <div style={{ fontSize: 10, color: "#ccc" }}>{file?.name}</div>
              </div>
            )}

            {/* Video always mounted once outputUrl exists — keeps videoRef.current alive for heatmap seeking */}
            {outputUrl && (
              <div style={{ position: "relative", maxWidth: "100%", maxHeight: "100%", display: isDone ? "block" : "none" }}>
                <video ref={videoRef} src={outputUrl} controls
                  onLoadedMetadata={() => setVideoDuration(videoRef.current?.duration || 0)}
                  style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 6, background: "#000", display: "block" }}>
                  Your browser does not support video.
                </video>
                {/* Seek flash badge — shows where we jumped to */}
                {lastSeek && (
                  <div key={lastSeek.timestamp + lastSeek.label} style={{
                    position: "absolute", bottom: 52, left: "50%", transform: "translateX(-50%)",
                    background: "rgba(0,0,0,0.82)", color: "#fff",
                    fontSize: 11, fontFamily: "monospace", fontWeight: 700,
                    padding: "5px 14px", borderRadius: 20,
                    pointerEvents: "none",
                    animation: "seekFadeOut 1.8s ease forwards",
                    whiteSpace: "nowrap",
                    letterSpacing: 1,
                  }}>
                    ⌖ {lastSeek.label} → {lastSeek.formattedTime}
                  </div>
                )}
              </div>
            )}

            {isError && (
              <div style={{ textAlign: "center", color: "#dc2626", padding: 24 }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>⚠</div>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>Pipeline Error</div>
                <div style={{ fontSize: 10, color: "#888", maxWidth: 400, lineHeight: 1.8 }}>{errMsg}</div>
              </div>
            )}
          </div>

          {/* download bar */}
          {isDone && outputUrl && (
            <div style={{ height: 44, borderTop: "1px solid #e8e8e2", background: "#fafaf8",
              display: "flex", alignItems: "center", padding: "0 16px", gap: 12, flexShrink: 0 }}>
              <span style={{ fontSize: 10, color: "#16a34a", fontWeight: 700 }}>✓ Output ready</span>
              <a href={outputUrl} download
                style={{ marginLeft: "auto", fontSize: 11, color: "#2563eb",
                  textDecoration: "none", border: "1px solid #bfdbfe",
                  background: "#eff6ff", padding: "6px 16px", borderRadius: 6 }}>
                ⬇ Download
              </a>
            </div>
          )}
        </div>

        {/* ── right panel — log ─────────────────────────────────────────────── */}
        <div style={{ width: 300, borderLeft: "1px solid #e8e8e2",
          background: "#fafaf8", display: "flex", flexDirection: "column", flexShrink: 0 }}>

          <div style={{ padding: "8px 12px", borderBottom: "1px solid #e8e8e2",
            display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <span style={{ fontSize: 10, color: "#bbb", letterSpacing: 1 }}>PIPELINE LOG</span>
            {isBusy && (
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#2563eb",
                animation: "blink 1.2s infinite", marginLeft: "auto" }} />
            )}
            {isDone && <span style={{ fontSize: 10, color: "#16a34a", marginLeft: "auto" }}>complete</span>}
          </div>

          {/* Progress bar — visible while running */}
          {isRunning && (
            <div style={{ padding: "10px 12px 8px", borderBottom: "1px solid #e8e8e2", flexShrink: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between",
                fontSize: 9, fontFamily: "monospace", marginBottom: 5 }}>
                <span style={{ color: "#2563eb", fontWeight: 700, letterSpacing: 1 }}>PROCESSING</span>
                <span style={{ color: "#2563eb", fontWeight: 700 }}>{progress}%</span>
              </div>
              <div style={{ height: 6, background: "#e8e8e2", borderRadius: 99, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${progress}%`,
                  background: "linear-gradient(90deg, #1d4ed8, #2563eb, #60a5fa)",
                  borderRadius: 99,
                  transition: "width 1s cubic-bezier(0.4,0,0.2,1)",
                }} />
              </div>
            </div>
          )}

          <LogPanel lines={logLines} />

          <div style={{ padding: "6px 12px", borderTop: "1px solid #e8e8e2",
            fontSize: 9, color: "#ccc", flexShrink: 0 }}>
            status:{" "}
            <span style={{ color: isError ? "#dc2626" : isDone ? "#16a34a" : isBusy ? "#2563eb" : "#bbb",
              fontWeight: 700 }}>{status}</span>
            {jobId && <span style={{ marginLeft: 8 }}>job: {jobId.slice(0, 8)}</span>}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
        @keyframes slideDown { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }
        @keyframes seekFadeOut { 0%{opacity:1;transform:translateX(-50%) translateY(0)} 60%{opacity:1} 100%{opacity:0;transform:translateX(-50%) translateY(-8px)} }
      `}</style>

      {/* ── Processing mode warning / confirmation modal ── */}
      {showWarning && (() => {
        const m = MODES.find(x => x.id === processingMode)!;
        return (
          <div style={{
            position: "fixed", inset: 0, zIndex: 1000,
            background: "rgba(0,0,0,0.45)", display: "flex",
            alignItems: "center", justifyContent: "center",
          }}>
            <div style={{
              background: "#fff", borderRadius: 12, padding: "28px 28px 22px",
              width: 420, maxWidth: "92vw", boxShadow: "0 8px 40px rgba(0,0,0,0.18)",
              border: `1.5px solid ${m.border}`,
              animation: "slideDown 0.18s ease",
            }}>
              {/* header */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <div style={{
                  width: 32, height: 32, borderRadius: "50%",
                  background: m.accent, border: `2px solid ${m.border}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16,
                }}>⚠</div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#1a1a1a" }}>
                    Confirm: {m.label}
                  </div>
                  <div style={{ fontSize: 10, color: "#888", marginTop: 1 }}>{m.tagline}</div>
                </div>
              </div>

              {/* warning text */}
              <div style={{
                background: m.accent, border: `1px solid ${m.border}`,
                borderRadius: 7, padding: "10px 13px",
                fontSize: 11, color: "#555", lineHeight: 1.65, marginBottom: 14,
              }}>
                {m.warning}
              </div>

              {/* mode detail */}
              <div style={{ fontSize: 10, color: "#888", lineHeight: 1.7, marginBottom: 18 }}>
                {m.detail}
              </div>

              {/* file info + est time */}
              <div style={{
                background: "#f8f8f6", borderRadius: 6, padding: "8px 12px",
                fontSize: 10, color: "#666", marginBottom: 20, lineHeight: 1.7,
              }}>
                <strong style={{ color: "#333" }}>File:</strong> {file?.name}
                {file && <span style={{ color: "#bbb" }}> ({(file.size/1024/1024).toFixed(1)} MB)</span>}
                <br />
                <strong style={{ color: "#333" }}>Estimated time:</strong>{" "}
                <span style={{ color: m.color, fontWeight: 700 }}>
                  calculating after upload…
                </span>
              </div>

              {/* buttons */}
              <div style={{ display: "flex", gap: 10 }}>
                <button onClick={() => setShowWarning(false)}
                  style={{
                    flex: 1, padding: "11px 0", borderRadius: 7, cursor: "pointer",
                    border: "1px solid #e8e8e2", background: "transparent",
                    color: "#888", fontSize: 12, fontWeight: 600,
                  }}>
                  ← Cancel
                </button>
                <button onClick={handleRunConfirm}
                  style={{
                    flex: 2, padding: "11px 0", borderRadius: 7, cursor: "pointer",
                    border: `1px solid ${m.color}55`,
                    background: m.accent, color: m.color,
                    fontSize: 12, fontWeight: 700, letterSpacing: 0.5,
                  }}>
                  ▶ Confirm & Run
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}