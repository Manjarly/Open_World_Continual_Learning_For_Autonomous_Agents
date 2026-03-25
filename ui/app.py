"""
ui/app.py
─────────
Streamlit frontend for the Open-World Continual Learning demo.
Uploads images to the FastAPI backend and visualizes predictions.
"""

import streamlit as st
import requests
from PIL import Image, ImageDraw

import os

# Mapping class IDs to human-readable names based on the Colab notebook
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Vehicle",
    2: "Cyclist",
    3: "Sign",
    -1: "UNKNOWN"
}

# Use environment variable to support Docker routing (http://api:8000/predict)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="OWCL Autonomous Agents", layout="wide")

st.title("🚗 Open-World Continual Learning")
st.markdown("Upload a camera frame to detect objects and flag **unknown** obstacles that the model has never seen before.")

st.markdown("---")

# Colors for bounding boxes
COLOR_KNOWN = "#00FF00"  # Bright Green
COLOR_UNKNOWN = "#FF0000" # Bright Red

uploaded_file = st.file_uploader("Choose an image (from Waymo or nuScenes dataset)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image with PIL
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        
    with st.spinner("Analyzing image through FastAPI Backend..."):
        try:
            # Reset file pointer and send to API
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                detections = data.get("detections", [])
                
                # Draw boxes
                draw = ImageDraw.Draw(image)
                
                known_count = 0
                unknown_count = 0
                
                for det in detections:
                    box = det["box"] # [x1, y1, x2, y2]
                    is_unknown = det.get("is_unknown", False)
                    conf = det.get("conf", 0.0)
                    cls_id = det.get("cls", 0)
                    
                    if is_unknown:
                        color = COLOR_UNKNOWN
                        label_name = "UNKNOWN"
                        unknown_count += 1
                    else:
                        color = COLOR_KNOWN
                        label_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                        known_count += 1
                        
                    label = f"{label_name} ({conf:.2f})"
                    
                    # Draw rectangle
                    draw.rectangle(box, outline=color, width=4)
                    
                    # Draw text label background
                    text_x = box[0]
                    text_y = max(0, box[1] - 15)
                    draw.rectangle([text_x, text_y, text_x + len(label)*6, text_y + 15], fill=color)
                    draw.text((text_x + 2, text_y), label, fill="black")
                
                with col2:
                    st.subheader("Model Predictions")
                    st.image(image, use_container_width=True)
                    
                    st.markdown("### Detection Summary")
                    st.write(f"- **Known Objects Detected:** {known_count}")
                    
                    if unknown_count > 0:
                        st.error(f"🚨 **Unknown Objects Flagged:** {unknown_count} (High Entropy)")
                    else:
                        st.success(f"**Unknown Objects Flagged:** 0")
                        
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            st.info("Make sure the FastAPI backend is running in another terminal on http://localhost:8000! Run: `uvicorn api.app:app`")
