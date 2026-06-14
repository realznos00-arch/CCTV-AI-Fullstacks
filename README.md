# CCTV-AI-Fullstacks

Production-ready full-stack CCTV surveillance AI system with real-time video processing, event detection, and analytics.

## 🏗️ Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      CCTV VIDEO INPUT                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
        ┌────────────────────────────────────────┐
        │   LAYER 1: OBSERVATION (Detection)    │
        │   phase1_observation.py                │
        │   - Frame analysis                     │
        │   - Entity detection                   │
        └────────────────────────────────────────┘
                             │
                             ↓
                    phase1_candidates.json
                             │
                             ↓
        ┌────────────────────────────────────────┐
        │     LAYER 2: FILTERING (Decision)      │
        │     phase2_filter.py                   │
        │     - False positive reduction          │
        │     - Rule-based filtering             │
        └────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
          phase2_decisions.json   phase2_filter_results.json
                    │                 │
                    └────────┬────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │   LAYER 3: DERIVATION (Synthesis)      │
        │   phase3_derivation.py                 │
        │   - Event extraction                   │
        │   - Pattern recognition (LSTM)         │
        └────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
          phase3_events.json     phase3_derivation_results.json
                    │                 │
                    └────────┬────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │   LAYER 4: STORAGE (Persistence)       │
        │   phase4_storage.py                    │
        │   - Database persistence               │
        │   - Log streaming                      │
        └────────────────────────────────────────┘
                    │
         ┌──────────┼──────────┐
         ↓          ↓          ↓
    SQLite DB   JSONL Log  Analytics
    (Queries)  (Streaming) (Heatmaps)
         │          │          │
         └──────────┼──────────┘
                    ↓
        ┌────────────────────────────────────────┐
        │   FRONTEND LAYER (Dashboard)           │
        │   Next.js + TypeScript                 │
        │   - Real-time visualization            │
        │   - Event management                   │
        └────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Backend:** Python (91.1%)
- **Frontend:** TypeScript/Next.js (8.4%)
- **AI/ML:** YOLOv8 Pose Detection, LSTM Networks
- **Storage:** SQLite + JSONL
- **Processing:** Real-time synchronized/compressed inference

## 📊 Layer Details

| Layer | Module | Input | Process | Output |
|-------|--------|-------|---------|--------|
| **1** | `phase1_observation.py` | Video frames | Detection | `phase1_candidates.json` |
| **2** | `phase2_filter.py` | Candidates | Filtering | `phase2_decisions.json` |
| **3** | `phase3_derivation.py` | Filtered events | Synthesis | `phase3_events.json` |
| **4** | `phase4_storage.py` | Events | Persistence | `phase4_events.db`, `.jsonl` |
| **5** | Frontend (Next.js) | Database | Visualization | Dashboard UI |

## 🚀 Quick Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python ../phase1_observation.py
```

**Frontend:**
```bash
cd frontend
npm install && npm run dev
# Open http://localhost:3000
```

## 📁 Key Components

- **Video Inference:** `infer-vid-w7-sync.py`, `infer-vid-w7-compress.py`
- **AI Models:** `yolo26n-pose.pt` (7.9MB), `lstm_model/` (temporal patterns)
- **Analytics:** `heatmap_phase1-4.py` (spatial analysis)
- **Performance:** `time.py` (benchmarking)

## 💾 Data Storage

- **SQLite** (`phase4_events.db`) - Structured queries, analytics
- **JSONL** (`phase4_events.jsonl`) - Log streaming, integrations

---

**Repository:** [realznos00-arch/CCTV-AI-Fullstacks](https://github.com/realznos00-arch/CCTV-AI-Fullstacks)
