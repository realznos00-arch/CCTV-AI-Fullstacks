# CCTV-AI-Fullstacks

A **production-ready full-stack CCTV surveillance AI system** that detects, filters, analyzes, and stores security events with real-time video processing and visual analytics capabilities.

## 📋 Overview

**CCTV-AI-Fullstacks** is an intelligent surveillance platform combining computer vision, machine learning, and event-driven architecture to process CCTV footage in real-time. The system employs a 4-phase pipeline for robust anomaly detection and event persistence.

### Technology Stack
- **Backend:** Python (91.1% of codebase)
- **Frontend:** TypeScript/Next.js (8.4% of codebase)
- **AI/ML:** YOLOv8 Pose Detection, LSTM Neural Networks
- **Database:** SQLite for event persistence
- **Architecture:** Event-driven, microservice-ready

---

## 🏗️ System Architecture

### 4-Phase Processing Pipeline

#### **Phase 1: Observation**
- **File:** `phase1_observation.py`
- **Purpose:** Initial video frame analysis and candidate detection
- **Output:** `phase1_candidates.json` (detected entities)
- Identifies potential security events from CCTV frames

#### **Phase 2: Filtering**
- **File:** `phase2_filter.py`
- **Purpose:** Decision-making and intelligent filtering of candidates
- **Input:** `phase1_candidates.json`
- **Output:** `phase2_decisions.json`, `phase2_filter_results.json`
- Reduces false positives through heuristic and ML-based filtering

#### **Phase 3: Derivation**
- **File:** `phase3_derivation.py`
- **Purpose:** Event derivation and pattern extraction
- **Input:** `phase2_filter_results.json`
- **Output:** `phase3_events.json`, `phase3_derivation_results.json`
- Synthesizes filtered data into meaningful security events

#### **Phase 4: Storage**
- **File:** `phase4_storage.py`
- **Purpose:** Event persistence and database management
- **Storage:** `phase4_events.db` (SQLite), `phase4_events.jsonl` (JSONL format)
- Ensures event durability and accessibility

---

## 🎯 Key Features

### Real-Time Video Processing
- **Synchronized Inference:** `infer-vid-w7-sync.py` - Maintains video frame synchronization
- **Compressed Processing:** `infer-vid-w7-compress.py` - Optimized for bandwidth-constrained environments
- **YOLO Pose Detection:** `yolo26n-pose.pt` (~7.9 MB model) for skeleton-based activity recognition

### AI/ML Components
- **LSTM Models:** `lstm_model/` directory - Neural networks for temporal pattern recognition
- **Pose Estimation:** Real-time skeleton tracking for human activity analysis
- **Event Classification:** Multi-stage filtering and event derivation

### Spatial Analytics
- **Heatmap Generation:** 4-phase heatmap processing (`heatmap_phase1-4.py`)
- **Heatmap Data:** Visualization-ready outputs (`heatmap_phase1-4.json`)
- Spatial analysis of detected events for threat mapping

### Event Persistence
- **SQLite Database:** Structured event storage (`phase4_events.db`)
- **JSONL Export:** Line-delimited JSON for log streaming and integration
- **Timestamped Records:** Full audit trail of all detected events

---

## 📁 Project Structure

```
CCTV-AI-Fullstacks/
├── backend/                          # Python backend services
├── frontend/                         # Next.js React application
├── lstm_model/                       # LSTM neural network models
│
├── phase1_observation.py             # Observation & detection
├── phase1_candidates.json            # Detected candidates
│
├── phase2_filter.py                  # Filtering logic
├── phase2_decisions.json             # Filter decisions
├── phase2_filter_results.json        # Filter results
│
├── phase3_derivation.py              # Event derivation
├── phase3_events.json                # Derived events
├── phase3_derivation_results.json    # Derivation results
│
├── phase4_storage.py                 # Storage & persistence
├── phase4_events.db                  # SQLite database
├── phase4_events.db-shm              # Shared memory file
├── phase4_events.db-wal              # Write-ahead log
├── phase4_events.jsonl               # JSONL event log
│
├── heatmap_phase1-4.py               # Heatmap generation scripts
├── heatmap_phase1-4.json             # Heatmap data outputs
│
├── infer-vid-w7-sync.py              # Synchronized video inference
├── infer-vid-w7-compress.py          # Compressed video inference
├── yolo26n-pose.pt                   # YOLOv8 pose model
│
├── time.py                           # Timing & performance utilities
├── run.txt                           # Execution documentation
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- pip/npm package managers
- GPU support recommended (CUDA for accelerated inference)

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Run Phase 1 (Observation)
python ../phase1_observation.py

# Run Phase 2 (Filtering)
python ../phase2_filter.py

# Run Phase 3 (Derivation)
python ../phase3_derivation.py

# Run Phase 4 (Storage)
python ../phase4_storage.py
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to access the dashboard.

---

## 📊 Data Flow

```
CCTV Video Input
    ↓
[Phase 1: Observation]
    ↓
phase1_candidates.json
    ↓
[Phase 2: Filtering]
    ↓
phase2_decisions.json → phase2_filter_results.json
    ↓
[Phase 3: Derivation]
    ↓
phase3_events.json → phase3_derivation_results.json
    ↓
[Phase 4: Storage]
    ↓
phase4_events.db (SQLite) + phase4_events.jsonl (JSONL)
    ↓
[Frontend Dashboard] ← Visualization & Analytics
```

---

## 🎬 Video Inference Options

### Synchronized Inference
Best for: Real-time monitoring with strict frame synchronization
```bash
python infer-vid-w7-sync.py --input <video_path> --output <output_dir>
```

### Compressed Inference
Best for: High-throughput processing with bandwidth optimization
```bash
python infer-vid-w7-compress.py --input <video_path> --output <output_dir>
```

---

## 📈 Heatmap Analytics

Generate spatial heatmaps for threat analysis:
```bash
python heatmap_phase1.py   # Phase 1 heatmap
python heatmap_phase2.py   # Phase 2 heatmap
python heatmap_phase3.py   # Phase 3 heatmap
python heatmap_phase4.py   # Phase 4 heatmap
```

Outputs are saved as `.json` files for visualization integration.

---

## 💾 Data Storage

### SQLite Database (`phase4_events.db`)
- Structured relational storage
- WAL mode enabled for concurrent access
- Optimized for event queries and analytics

### JSONL Format (`phase4_events.jsonl`)
- Line-delimited JSON for streaming
- Compatible with log aggregation systems
- Suitable for integration with analytics platforms (ELK, Splunk, etc.)

---

## 🔧 Performance & Timing

Use `time.py` to profile system performance:
```bash
python time.py
```

Useful for:
- Benchmarking inference speed
- Identifying bottlenecks
- Optimizing pipeline phases

---

## 📝 Configuration

See `run.txt` for execution guidelines and configuration notes.

---

## 🛠️ Development

### Frontend Development
- Framework: [Next.js](https://nextjs.org/docs)
- Language: TypeScript
- UI auto-refreshes on code changes

### Backend Development
- Language: Python
- Each phase is independently executable
- JSON-based data interchange for flexibility

---

## 📦 Dependencies

**Backend:**
- OpenCV (video processing)
- YOLOv8 (pose detection)
- PyTorch/TensorFlow (LSTM models)
- SQLite3 (database)

**Frontend:**
- React (Next.js)
- TypeScript

---

## 📜 License

This project is private. For access and licensing inquiries, contact the repository owner.

---

## 👤 Author

**realznos00-arch**

Repository: [realznos00-arch/CCTV-AI-Fullstacks](https://github.com/realznos00-arch/CCTV-AI-Fullstacks)

---

## 🤝 Contributing

For contributions, please:
1. Create a feature branch
2. Commit changes with clear messages
3. Submit a pull request with documentation

---

## 📞 Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.

---

**Last Updated:** June 14, 2026
