# CCTV-AI-Fullstacks

Production-ready full-stack CCTV surveillance AI system with real-time video processing, event detection, and analytics.

## 🏗️ Architecture

**4-Phase Event Processing Pipeline:**

| Phase | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Observation | `phase1_observation.py` | Video frames | `phase1_candidates.json` |
| 2. Filtering | `phase2_filter.py` | Candidates | `phase2_decisions.json` |
| 3. Derivation | `phase3_derivation.py` | Filtered events | `phase3_events.json` |
| 4. Storage | `phase4_storage.py` | Derived events | `phase4_events.db`, `phase4_events.jsonl` |

## 🛠️ Tech Stack

- **Backend:** Python (91.1%)
- **Frontend:** TypeScript/Next.js (8.4%)
- **AI/ML:** YOLOv8 Pose Detection, LSTM Networks
- **Storage:** SQLite + JSONL
- **Video Processing:** Real-time synchronized/compressed inference

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

## 📊 Key Components

- **Video Inference:** `infer-vid-w7-sync.py` (synchronized), `infer-vid-w7-compress.py` (optimized)
- **AI Models:** `yolo26n-pose.pt` (7.9MB), `lstm_model/` (temporal patterns)
- **Analytics:** Heatmap generation (`heatmap_phase1-4.py`)
- **Performance:** `time.py` for benchmarking

## 📁 Project Structure

```
├── backend/                          # Python backend
├── frontend/                         # Next.js UI
├── lstm_model/                       # LSTM models
├── phase1_observation.py             # Detection
├── phase2_filter.py                  # Filtering
├── phase3_derivation.py              # Event extraction
├── phase4_storage.py                 # Persistence
├── infer-vid-w7-sync.py              # Sync inference
├── infer-vid-w7-compress.py          # Compressed inference
├── heatmap_phase1-4.py               # Spatial analysis
└── yolo26n-pose.pt                   # Pose model
```

## 💾 Data Storage

- **SQLite** (`phase4_events.db`) - Structured queries
- **JSONL** (`phase4_events.jsonl`) - Log streaming, integrations

## 📝 License

Private repository. Contact owner for access.

---

**Repository:** [realznos00-arch/CCTV-AI-Fullstacks](https://github.com/realznos00-arch/CCTV-AI-Fullstacks)
