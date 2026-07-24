"""
ui/app.py
─────────
Streamlit frontend for the Open-World Continual Learning (OWCL) demo.
Interactive dashboard for open-set object detection and continual learning analytics.
"""

import os
import requests
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

# Import project visualization utilities if available locally
try:
    from src.utils.visualization import draw_detections, plot_uncertainty_distribution
    LOCAL_VIS_AVAILABLE = True
except Exception:
    LOCAL_VIS_AVAILABLE = False

# Class mappings
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Vehicle",
    2: "Cyclist",
    3: "Sign",
    -1: "UNKNOWN",
}

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="OWCL Autonomous Vision",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0E1117;
        color: #F0F6FC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Gradient Banner */
    .main-header {
        background: linear-gradient(135deg, #1F2937 0%, #111827 50%, #0F172A 100%);
        border: 1px solid #374151;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-known { color: #34D399; }
    .metric-unknown { color: #F87171; }
    .metric-total { color: #60A5FA; }

    /* Alert Banners */
    .alert-unknown {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #FCA5A5;
        font-weight: 600;
        margin-top: 1rem;
    }
    .alert-safe {
        background: rgba(52, 211, 153, 0.15);
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        color: #A7F3D0;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #1F2937;
        color: #9CA3AF;
        border: 1px solid #374151;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚗 Open-World Continual Learning (OWCL)</h1>
    <p class="main-subtitle">Autonomous Perception Engine with Entropy-Based Open-Set Recognition & EWC Adaptation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/autonomous-car.png", width=64)
    st.header("⚙️ Control Panel")
    
    # API Backend Health Indicator
    api_endpoint = st.text_input("FastAPI Host", value=API_URL, help="Base URL of backend service")
    predict_url = f"{api_endpoint.rstrip('/')}/predict"
    health_url = f"{api_endpoint.rstrip('/')}/health"
    
    try:
        r_health = requests.get(health_url, timeout=2)
        if r_health.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.warning("🟡 API Error")
    except Exception:
        st.error("🔴 API Offline (Check Uvicorn)")
    
    st.markdown("---")
    st.subheader("🎯 Open-Set Parameters")
    
    metric_choice = st.selectbox(
        "Uncertainty Metric",
        options=["entropy", "max_softmax", "energy"],
        index=0,
        help="Algorithm used to estimate output uncertainty"
    )
    
    uncertainty_thresh = st.slider(
        "Uncertainty Threshold (Flag Unknown)",
        min_value=0.1,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Detections exceeding this score are flagged as UNKNOWN"
    )
    
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Filter out low-confidence bounding box candidates"
    )
    
    st.markdown("---")
    st.info("💡 **Team Delaware (OWCL Project)**\nWaymo Open Dataset ➔ nuScenes")

# ── Main Content Area: Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Object Detection & Open-Set", "📊 Uncertainty Analytics", "🧠 Continual Learning Stats"])

with tab1:
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("📷 Image Source")
        uploaded_file = st.file_uploader(
            "Upload a camera frame (Waymo / nuScenes / driving test)...",
            type=["jpg", "jpeg", "png"]
        )
        
        # Test synthetic frame generator if user has no image ready
        generate_sample = st.button("🖼️ Generate Sample Driving Frame")

    image_to_process = None
    file_bytes = None
    filename = "upload.jpg"

    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
    elif generate_sample:
        # Create synthetic demo image (dark road with simulated objects)
        synthetic_img = Image.new("RGB", (640, 480), color=(30, 35, 45))
        draw_syn = ImageDraw.Draw(synthetic_img)
        # Draw road perspective lines
        draw_syn.polygon([(200, 480), (300, 240), (340, 240), (440, 480)], fill=(50, 55, 65))
        # Draw simulated vehicle & unknown box shapes
        draw_syn.rectangle([340, 300, 480, 400], fill=(70, 130, 180), outline=(255, 255, 255), width=2)
        draw_syn.rectangle([150, 320, 220, 420], fill=(180, 70, 90), outline=(255, 255, 255), width=2)
        
        image_to_process = synthetic_img
        buf = BytesIO()
        synthetic_img.save(buf, format="JPEG")
        file_bytes = buf.getvalue()
        filename = "synthetic_sample.jpg"

    with col_input:
        if image_to_process is not None:
            st.image(image_to_process, caption="Input Frame", use_container_width=True)

    with col_output:
        st.subheader("🔍 OWCL Prediction Output")
        if file_bytes is not None:
            with st.spinner("Processing through OWCL API Backend..."):
                try:
                    params = {
                        "conf_threshold": conf_thresh,
                        "uncertainty_threshold": uncertainty_thresh,
                        "metric": metric_choice
                    }
                    files = {"file": (filename, file_bytes, "image/jpeg")}
                    
                    resp = requests.post(predict_url, files=files, params=params, timeout=10)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        detections = data.get("detections", [])
                        known_cnt = sum(1 for d in detections if not d.get("is_unknown", False))
                        unknown_cnt = sum(1 for d in detections if d.get("is_unknown", False))
                        
                        # Store in session state for tab 2
                        st.session_state["last_detections"] = detections
                        st.session_state["last_threshold"] = uncertainty_thresh
                        st.session_state["last_metric"] = metric_choice
                        
                        # Draw annotated image using project visualization module
                        if LOCAL_VIS_AVAILABLE and image_to_process is not None:
                            annotated_img = draw_detections(
                                image_to_process,
                                detections,
                                class_names=CLASS_NAMES,
                                show_uncertainty=True
                            )
                        else:
                            # Fallback drawing
                            annotated_img = image_to_process.copy()
                            draw = ImageDraw.Draw(annotated_img)
                            for det in detections:
                                box = det.get("box", [0, 0, 0, 0])
                                is_unk = det.get("is_unknown", False)
                                color = "#FF1744" if is_unk else "#00E676"
                                draw.rectangle(box, outline=color, width=3)
                        
                        st.image(annotated_img, caption="Annotated Bounding Boxes", use_container_width=True)
                        
                        # Summary Cards
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value metric-total">{len(detections)}</div><div>Total Detections</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value metric-known">{known_cnt}</div><div>Known Objects</div></div>', unsafe_allow_html=True)
                        with c3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value metric-unknown">{unknown_cnt}</div><div>Unknown Objects</div></div>', unsafe_allow_html=True)
                        
                        if unknown_cnt > 0:
                            st.markdown(f'<div class="alert-unknown">🚨 <b>{unknown_cnt} UNKNOWN obstacle(s) detected!</b> High entropy signal detected — flagged for continual learning review.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-safe">✅ All detected objects belong to known baseline categories.</div>', unsafe_allow_html=True)

                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API backend: {e}")
                    st.info("Make sure FastAPI backend is running on `http://localhost:8000` via `uvicorn api.app:app`")
        else:
            st.info("👈 Upload an image or click 'Generate Sample Driving Frame' to test.")

with tab2:
    st.subheader("📊 Detailed Uncertainty & Entropy Analytics")
    
    if "last_detections" in st.session_state and st.session_state["last_detections"]:
        dets = st.session_state["last_detections"]
        thresh = st.session_state.get("last_threshold", uncertainty_thresh)
        metric = st.session_state.get("last_metric", metric_choice)
        
        col_chart, col_table = st.columns([1, 1])
        
        with col_chart:
            st.markdown("### Entropy Distribution")
            if LOCAL_VIS_AVAILABLE:
                fig = plot_uncertainty_distribution(dets, threshold=thresh, metric_name=metric)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("No uncertainty scores available to plot.")
            else:
                st.info("Local visualization module not loaded.")

        with col_table:
            st.markdown("### Detection Records")
            table_data = []
            for i, d in enumerate(dets):
                table_data.append({
                    "ID": i + 1,
                    "Class": d.get("name", CLASS_NAMES.get(d.get("cls", 0), "Unknown")),
                    "Confidence": f"{d.get('conf', 0.0):.4f}",
                    "Uncertainty": f"{d.get('uncertainty', 0.0):.4f}",
                    "Status": "🚨 UNKNOWN" if d.get("is_unknown", False) else "✅ KNOWN",
                    "Box [x1,y1,x2,y2]": [round(v, 1) for v in d.get("box", [])]
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Run a detection on Tab 1 to view entropy analytics.")

with tab3:
    st.subheader("🧠 Continual Learning & Forgetting Dashboard")
    
    st.markdown("""
    ### Project Architecture Highlights (Team Delaware)
    
    The **Open-World Continual Learning (OWCL)** system addresses two fundamental challenges in autonomous vehicle perception:
    
    1. **Catastrophic Forgetting (EWC)**:
       When adapting the detector from **Waymo (Source Domain)** to **nuScenes (Target Domain)**, Elastic Weight Consolidation adds a quadratic penalty proportional to the diagonal Fisher Information matrix:
    """)
    
    st.latex(r"L(\theta) = L_{\text{nuScenes}}(\theta) + \sum_{i} \frac{\lambda}{2} F_{i} (\theta_i - \theta_{A,i}^*)^2")
    
    st.markdown("""
    2. **Open-Set Uncertainty Detection**:
       Novel or out-of-distribution obstacles trigger uniform class logits. We normalize Shannon entropy across known categories:
    """)
    
    st.latex(r"H(p) = -\sum_{c=1}^{C} p_c \log(p_c) \quad \implies \quad U(x) = \frac{H(p)}{\log(C)}")
    
    st.markdown("---")
    c_stat1, c_stat2, c_stat3 = st.columns(3)
    with c_stat1:
        st.metric("Source Task", "Waymo Open Dataset", "4 Classes")
    with c_stat2:
        st.metric("Target Task", "nuScenes (Boston/SG)", "EWC λ = 0.40")
    with c_stat3:
        st.metric("Target FPR95", "< 5.0%", "Calibrated")

