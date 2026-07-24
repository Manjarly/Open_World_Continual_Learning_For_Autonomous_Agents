# Project Quad Chart: Open-World Continual Learning for Autonomous Agents

**Team:** Delaware

---

## Top-Left: Latest Accomplishments & KPIs
*What we have achieved so far*

- **Phase 1 Complete:** Baseline YOLOv8 detector trained on Waymo Open Dataset (Source Domain).
- **Phase 2 Complete:** Continual Learning pipeline with Elastic Weight Consolidation (EWC) on nuScenes dataset (Target Domain) to prevent catastrophic forgetting.
- **Open-Set Recognition:** Entropy, Max Softmax, and Energy-based uncertainty scoring implemented to detect "unknown" obstacles.
- **Phase 3 Complete:** Built FastAPI backend (`api/app.py`), modern multi-tab Streamlit dashboard (`ui/app.py`), and visualization utilities (`src/utils/visualization.py`).
- **MLOps & Quality:** 43 automated unit tests passing across loaders, EWC, uncertainty detection, and visualization.

---

## Top-Right: Major Next Tasks (RAIL)
*Immediate action items and ownership*

- **Docker Stack Finalization (Amit):** Build & test full stack (`docker-compose up --build`).
- **End-to-End Field Validation (Sudamshu/Shreyas):** Conduct benchmark testing against nuScenes validation splits.
- **Final Report & Paper (Aditya/Shreyas):** Consolidate findings into final project report & presentation slides.

---

## Bottom-Left: Major Risks, Barriers, & Obstacles
*Potential roadblocks we are monitoring*

- **Environment Inconsistency:** Risk of dependency conflicts across host machines. *(Mitigation: Strict Docker containerization).*
- **Inference Latency:** High-resolution entropy calculations during real-time stream inference. *(Mitigation: Dynamic thresholding and vectorized matrix operations).*
- **Data Quality:** Ensuring seamless handling when transitioning between raw Waymo TFRecords and nuScenes annotations.

---

## Bottom-Right: Remaining Major Activities & Timeline
*The path to completion (Phase 3 & 4)*

- **Phase 1 & 2 (Weeks 1–5):** Model training, EWC continual learning, open-set recognition ✅
- **Phase 3 (Weeks 6–9):** FastAPI backend, Streamlit Dashboard UI, Visualization engine, Unit testing ✅
- **Phase 4 (Weeks 10+):** System integration testing, final benchmark validation, LaTeX report & slides 🔜

