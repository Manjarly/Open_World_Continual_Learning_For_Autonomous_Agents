# Project Quad Chart: Open-World Continual Learning for Autonomous Agents

**Team:** Delaware

---

## Top-Left: Latest Accomplishments & KPIs
*What we have achieved so far*

- **Phase 1 Complete:** Baseline YOLOv8 model successfully trained on the Waymo Open Dataset (Source Domain).
- **Phase 2 Complete:** Continual Learning pipeline implemented using Elastic Weight Consolidation (EWC) on the nuScenes dataset (Target Domain) to mitigate catastrophic forgetting.
- **Open-Set Recognition:** Entropy-based uncertainty scoring implemented to detect "unknown" objects.
- **MLOps & Tracking:** Experiment tracking established using MLflow, with data versioning set up via DVC.
- **Prototyping:** End-to-end pipeline tested and validated in a self-contained Google Colab environment.

---

## Top-Right: Major Next Tasks (RAIL)
*Immediate action items and ownership*

- **Containerization (Amit):** Dockerize the entire application (training and inference pipelines) to ensure cross-environment consistency.
- **API Development (Amit/Shreyas):** Build a FastAPI backend to serve model predictions in real-time.
- **UI & Integration (Shreyas):** Develop a frontend user interface and integrate it smoothly with the FastAPI backend.
- **Testing & QA (Sudamshu):** Conduct rigorous testing of the integrated model outputs against ground truth data.

---

## Bottom-Left: Major Risks, Barriers, & Obstacles
*Potential roadblocks we are monitoring*

- **Environment Inconsistency:** Risk of dependency conflicts between team members' machines. *(Mitigation: Strict adherence to Docker containerization).*
- **Integration Bottlenecks:** Delays integrating the ML tracking (MLflow), backend (FastAPI), and frontend (UI). *(Mitigation: Daily monitoring of the GitHub Project board).*
- **Data Quality & Pipeline Issues:** Ensuring the synthetic/mock data pipelines scale correctly when transitioned to full local datasets. *(Mitigation: Strict preprocessing checks by the Data Engineer).*
- **Performance Overhead:** The EWC and uncertainty calculations might introduce latency during real-time inference.

---

## Bottom-Right: Remaining Major Activities & Timeline
*The path to completion (Phase 3 & 4)*

- **Weeks 6-8: API & Containerization:** Finalize Docker setups and build the FastAPI endpoints.
- **Week 9: UI Development:** Complete the frontend dashboard for visualizing "unknown" object detection and model metrics.
- **Week 10: End-to-End Testing:** System integration testing, bug fixing, and final validation.
- **Weeks 10+: Final Reporting:** Consolidate findings, complete the Overleaf (LaTeX) documentation, and prepare the final project presentation.
