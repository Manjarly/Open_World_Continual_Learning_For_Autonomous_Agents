# Open-World Continual Learning (OWCL) for Autonomous Agents

A deep learning and MLOps framework integrating **Open-Set Recognition (OSR)** and **Continual Learning (CL)** for autonomous vehicle perception. This system enables vision models to flag unknown obstacles in real-time while adapting to new geographic domains without suffering from catastrophic forgetting.

---

## 📌 Project Overview

Autonomous perception models deployed in real-world driving environments face two fundamental challenges:
1. **Closed-World Assumption**: Standard detectors classify every bounding box into a fixed set of predefined categories, misclassifying novel or unexpected obstacles (e.g., unusual construction debris, localized vehicles) into known classes with high confidence.
2. **Catastrophic Forgetting**: Fine-tuning an existing model on a new operational domain (e.g., moving from US west-coast driving to Singapore urban streets) drastically degrades its accuracy on the original domain.

The **OWCL Framework** addresses both issues by combining a **YOLOv8 object detector**, **Elastic Weight Consolidation (EWC)** for sequential domain adaptation, and **Entropy-based Uncertainty Scoring** for out-of-distribution (OOD) unknown detection.

```
                              ┌─────────────────────────────┐
                              │  Waymo Dataset (Source)     │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Baseline YOLOv8 Model   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
┌─────────────────────────────┐  ┌───────────────────────────┐
│   nuScenes Dataset (Target) ├─►│ EWC Continual Learning   │◄── Fisher Penalty (θ*)
└─────────────────────────────┘  └─────────────┬─────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Open-Set Uncertainty Engine │
                              │ (Entropy / MaxSoftmax / Energy)
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │  FastAPI & Streamlit Dashboard│
                              └─────────────────────────────┘
```

---

## 🧠 Machine Learning & Architecture

### 1. Baseline Object Detection (YOLOv8)
The base detector leverages Ultralytics YOLOv8 fine-tuned on primary navigation classes: **Vehicle**, **Pedestrian**, **Cyclist**, and **Sign**. 

### 2. Continual Learning via Elastic Weight Consolidation (EWC)
When adapting the model from the source domain (Waymo) to a target domain (nuScenes), EWC calculates the diagonal **Fisher Information Matrix ($F$)** to quantify the importance of each model parameter to the source domain task:

$$F_{ii} \approx \mathbb{E} \left[ \left( \frac{\partial \log p(y|x, \theta)}{\partial \theta_i} \right)^2 \right]$$

During target domain training, the total loss function penalizes parameter drift on weights critical to Task A:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{target}}(\theta) + \frac{\lambda}{2} \sum_{i} F_{i} (\theta_i - \theta_{A,i}^*)^2$$

- $\theta_{A,i}^*$: Optimal parameters learned on the source task (Waymo).
- $\lambda$: Regularization strength balancing target domain performance against source task retention.

### 3. Open-Set Uncertainty Recognition
When encountering unseen or out-of-distribution obstacles, the network's output logits exhibit flatter class distributions. The framework normalizes Shannon entropy across known categories:

$$H(p) = -\sum_{c=1}^{C} p_c \log(p_c) \quad \implies \quad U(x) = \frac{H(p)}{\log(C)}$$

Supported uncertainty scoring metrics in `src/openset/uncertainty.py`:
- **Normalized Entropy** (Default): Measures distribution uniformity over known classes.
- **Max-Softmax**: $1 - \max(p)$, flagging low maximum prediction confidence.
- **Energy Score**: $-\log \sum \exp(z_i)$, leveraging logit magnitudes for OOD detection.

Detections with uncertainty scores exceeding the calibrated threshold are automatically flagged as **`UNKNOWN`** (`cls = -1`).

---

## 📊 Datasets & Data Pipeline

| Domain | Dataset | Environment / Locations | Primary Classes |
|--------|---------|-------------------------|-----------------|
| **Source Domain** | Waymo Open Dataset | San Francisco, Mountain View, Phoenix | Vehicle, Pedestrian, Cyclist, Sign |
| **Target Domain** | nuScenes | Boston, Singapore | Vehicle, Pedestrian, Cyclist, Barrier |

### Data Preprocessing & Versioning
- **Parsing & Conversion**: `src/data/waymo_loader.py` and `src/data/nuscenes_loader.py` parse raw sensor annotations into standard YOLO bounding box formats (`[x_center, y_center, width, height]`).
- **Data Versioning (DVC)**: Processed dataset splits under `data/processed/` are tracked using DVC for reproducible data pipelines across training environments.

---

## ⚙️ System Workflow & Serving Pipeline

```
                                  Client Request
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    FastAPI Backend API    │
                          │        (/predict)         │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │     Inference Engine      │
                          │  (src/inference.py)       │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
           ┌──────────────────────────┐   ┌──────────────────────────┐
           │     YOLOv8 Detector      │   │   Uncertainty Flagger    │
           │ (src/models/yolo_detector│   │(src/openset/uncertainty) │
           └─────────────┬────────────┘   └─────────────┬────────────┘
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   Visualization Engine    │
                          │ (src/utils/visualization) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │    Streamlit Web UI       │
                          │       (ui/app.py)         │
                          └───────────────────────────┘
```

### 1. REST API (`api/app.py`)
Built with FastAPI utilizing modern `lifespan` context management:
- `POST /predict`: Uploads image frames and accepts dynamic parameters (`conf_threshold`, `uncertainty_threshold`, `metric`). Returns JSON bounding box detections with `is_unknown` flags.
- `GET /health`: Returns service health and model initialization status.
- `GET /config`: Exposes active threshold configurations and class mappings.

### 2. Streamlit Web Dashboard (`ui/app.py`)
Interactive visualization dashboard providing:
- **Object Detection & Open-Set Canvas**: Side-by-side comparison of input vs. predicted frames with color-coded bounding boxes (Emerald Green for known, Crimson Red for `UNKNOWN`).
- **Uncertainty Analytics**: Entropy distribution histograms and tabular detection record inspection.
- **Continual Learning Overview**: Real-time parameter controls and architectural details.

---

## 🛠️ CI/CD, Testing & MLOps

### Automated Testing Suite
Unit tests cover loaders, EWC penalty calculations, uncertainty metrics, and visualization utilities in `tests/`:
- `tests/test_loaders.py`: Dataset loading and annotation parsing.
- `tests/test_ewc.py`: Fisher matrix diagonal estimation and weight penalty computation.
- `tests/test_uncertainty.py`: Entropy, Max-Softmax, and Energy score calculations.
- `tests/test_visualization.py`: Bounding box annotation and histogram generation.

Run tests locally:
```bash
pytest tests/
```

### MLOps & Experiment Tracking
- **MLflow**: Logs loss curves, mAP50/mAP50-95 metrics, EWC $\lambda$ hyperparameter sweeps, and model checkpoint artifacts.
- **DVC**: Manages versioned data storage and remote caching.

### Containerization (Docker)
Packaged via `docker-compose.yml`:
- `Dockerfile.api`: FastAPI backend container running on port `8000`.
- `Dockerfile.ui`: Streamlit web dashboard container running on port `8501`.

---

## 📁 Repository Structure

```
owcl_project/
├── api/
│   └── app.py                # FastAPI REST service
├── ui/
│   └── app.py                # Streamlit web UI dashboard
├── configs/
│   ├── base_config.yaml      # General hyperparameters
│   ├── waymo_config.yaml     # Task A (Waymo) config
│   └── nuscenes_config.yaml  # Task B (nuScenes + EWC) config
├── data/
│   ├── waymo/                # Raw Waymo files (git-ignored)
│   ├── nuscenes/             # Raw nuScenes files (git-ignored)
│   └── processed/            # Processed splits (DVC-tracked)
├── src/
│   ├── data/
│   │   ├── waymo_loader.py   # Waymo dataset parser
│   │   ├── nuscenes_loader.py# nuScenes dataset parser
│   │   └── transforms.py     # Image augmentation pipeline
│   ├── models/
│   │   └── yolo_detector.py  # YOLOv8 model wrapper & trainer
│   ├── continual/
│   │   └── ewc.py            # Elastic Weight Consolidation module
│   ├── openset/
│   │   └── uncertainty.py    # Entropy & energy uncertainty flagger
│   ├── utils/
│   │   ├── mlflow_utils.py   # MLflow experiment tracking helpers
│   │   ├── metrics.py        # mAP, forgetting & open-set metrics
│   │   └── visualization.py  # Image annotation & histogram plotting
│   └── inference.py          # Unified inference engine
├── tests/
│   ├── test_loaders.py
│   ├── test_ewc.py
│   ├── test_uncertainty.py
│   └── test_visualization.py
├── train_baseline.py         # Baseline YOLOv8 training on Waymo
├── train_continual.py        # Continual learning with EWC on nuScenes
├── evaluate.py               # Unified evaluation & reporting script
├── docker-compose.yml        # Docker composition manifest
├── Dockerfile.api            # Dockerfile for FastAPI backend
├── Dockerfile.ui             # Dockerfile for Streamlit frontend
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Manjarly/Open_World_Continual_Learning_For_Autonomous_Agents.git
cd Open_World_Continual_Learning_For_Autonomous_Agents

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

```bash
pytest tests/
```

### 3. Training & Evaluation Pipeline

```bash
# Step 1: Train Baseline Detector on Waymo
python train_baseline.py --config configs/waymo_config.yaml

# Step 2: Continual Learning with EWC on nuScenes
python train_continual.py --config configs/nuscenes_config.yaml \
                          --checkpoint runs/waymo_baseline/weights/best.pt \
                          --ewc_lambda 0.4

# Step 3: Run Evaluation & Save Visualizations
python evaluate.py --checkpoint runs/continual_ewc/weights/best.pt \
                   --dataset nuscenes \
                   --open_set \
                   --save_visualizations
```

### 4. Local Deployment (API & UI)

#### Option A: Running Python Services Directly
```bash
# Terminal 1: Launch FastAPI Backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Streamlit Dashboard UI
streamlit run ui/app.py
```

- **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Option B: Running with Docker Compose
```bash
docker-compose up --build
```

---

## 🧰 Tech Stack

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Datasets & Tools**: Waymo Open Dataset SDK, nuscenes-devkit, OpenCV, Pillow
- **MLOps & Tracking**: MLflow, DVC
- **Web & Serving**: FastAPI, Uvicorn, Streamlit, Requests
- **Testing & Containerization**: Pytest, Docker, Docker Compose

