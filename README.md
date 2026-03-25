# Open-World Continual Learning for Autonomous Agents
### Team Delaware

A deep learning framework that integrates **Open-Set Recognition** and **Continual Learning** for autonomous vehicle perception — enabling agents to detect unknown objects and adapt to new environments without catastrophic forgetting.

---

## Project Overview

| Component | Details |
|-----------|---------|
| **Source Domain** | Waymo Open Dataset (San Francisco, Phoenix, Mountain View) |
| **Target Domain** | nuScenes (Boston, Singapore) |
| **Baseline Detector** | YOLOv8 |
| **Continual Learning** | Elastic Weight Consolidation (EWC) |
| **Open-Set Recognition** | Entropy-based uncertainty thresholding |
| **MLOps** | MLflow experiment tracking, DVC data versioning |

---

## Repository Structure

```
owcl_project/
├── configs/                  # YAML config files for all experiments
│   ├── base_config.yaml
│   ├── waymo_config.yaml
│   └── nuscenes_config.yaml
├── data/
│   ├── waymo/                # Raw Waymo TFRecords (not tracked by Git)
│   ├── nuscenes/             # Raw nuScenes data (not tracked by Git)
│   └── processed/            # DVC-tracked processed splits
├── notebooks/
│   ├── 01_waymo_eda.ipynb
│   ├── 02_nuscenes_eda.ipynb
│   └── 03_baseline_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── waymo_loader.py   # Waymo dataset loader & parser
│   │   ├── nuscenes_loader.py# nuScenes dataset loader & parser
│   │   └── transforms.py     # Shared augmentation pipeline
│   ├── models/
│   │   └── yolo_detector.py  # YOLOv8 wrapper & trainer
│   ├── continual/
│   │   └── ewc.py            # Elastic Weight Consolidation
│   ├── openset/
│   │   └── uncertainty.py    # Entropy-based open-set recognition
│   └── utils/
│       ├── mlflow_utils.py   # MLflow logging helpers
│       ├── metrics.py        # mAP, F1, uncertainty metrics
│       └── visualization.py  # Plotting & result visualization
├── tests/
│   ├── test_loaders.py
│   ├── test_ewc.py
│   └── test_uncertainty.py
├── train_baseline.py         # Phase 1: Train baseline YOLOv8 on Waymo
├── train_continual.py        # Phase 2: EWC continual learning on nuScenes
├── evaluate.py               # Unified evaluation script
├── requirements.txt
├── .dvcignore
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-org/owcl-autonomous-agents.git
cd owcl-autonomous-agents
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Data (DVC)

```bash
# Pull processed data splits (requires DVC remote access)
dvc pull

# Or manually place raw data:
# - Waymo TFRecords → data/waymo/
# - nuScenes files  → data/nuscenes/
```

### 3. Phase 1 — Baseline Training (Waymo)

```bash
python train_baseline.py --config configs/waymo_config.yaml
```

### 4. Phase 2 — Continual Learning (nuScenes + EWC)

```bash
python train_continual.py --config configs/nuscenes_config.yaml \
                          --checkpoint runs/waymo_baseline/best.pt \
                          --ewc_lambda 0.4
```

### 5. Evaluate

```bash
python evaluate.py --checkpoint runs/continual/best.pt \
                   --dataset nuscenes \
                   --open_set
```

### 6. Launch MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 7. Phase 3 — Deployment (Docker API & UI)

For deploying the Open-Set Recognition pipeline seamlessly across any teammate's machine, we have packaged the FastAPI backend and Streamlit UI dashboard into Docker containers.

```bash
# Ensure Docker Desktop is running, then spin up the stack:
docker-compose up --build
```

Once the containers are successfully running:
- **UI Dashboard:** Open [http://localhost:8501](http://localhost:8501) in your browser to upload images and see the open-set bounding boxes.
- **FastAPI Swagger:** Open [http://localhost:8000/docs](http://localhost:8000/docs) to test the API endpoints directly.

---

## Phase Progress

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 (Weeks 1–2) | Data acquisition, EDA, Baseline YOLOv8 | ✅ Complete |
| Phase 2 (Weeks 3–5) | Feature engineering, EWC, Open-Set, MLflow | ✅ Complete |
| Phase 3 (Weeks 6–10) | Docker, API, UI, Integration | ✅ Complete |
| Phase 4 (Weeks 10+) | Final report, Overleaf docs | 🔜 Upcoming |

---

## Team

| Member | Role |
|--------|------|
| Aditya | Lead Developer & Documentation |
| Amit | MLOps & Backend |
| Sudamshu | Data Engineer & QA |
| Shreyas | Project Manager & Integration |

---

## Tech Stack

- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Data:** Waymo Open Dataset SDK, nuscenes-devkit, DVC
- **MLOps:** MLflow, DVC
- **Dev:** VS Code, Jupyter, GitHub Projects
