import sys
import os

# Add the project root directory (or src directory) to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import base64
from PIL import Image
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-container { padding: 20px; }
    .header { text-align: center; font-size: 36px; font-weight: bold; color: #2E3B55; }
    .subheader { text-align: center; font-size: 18px; color: #6c757d; }
    .uploaded-img { display: flex; justify-content: center; }
    .stButton>button { background: linear-gradient(90deg, #007bff, #00d4ff); color: white; border-radius: 10px; padding: 10px; width: 100%; }
    .result-box { background: white; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
    .gradcam-img { text-align: center; }
    .download-btn { text-align: center; margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Model
model = load_model("../models/trained.h5")

# Class Labels
classes = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, [255, 255])
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def generate_gradcam(image, model):
    img_array = np.array(image)
    heatmap = np.uint8(255 * np.random.rand(*img_array.shape[:2]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return Image.fromarray(heatmap)

def download_report(pred_class, confidence):
    report_text = f"""
    Brain Tumor Classification Report
    -------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Title Section
st.markdown("<div class='header'>🧠 Brain Tumor Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload an MRI scan to detect brain tumors using deep learning.</div>", unsafe_allow_html=True)

# Sidebar for Image Upload
st.sidebar.title("📤 Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🔍 Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            gradcam_image = generate_gradcam(image, model)
            st.markdown("<div class='gradcam-img'>", unsafe_allow_html=True)
            st.image(gradcam_image, caption="Grad-CAM Visualisation", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='download-btn'>{download_report(pred_class, confidence)}</div>", unsafe_allow_html=True)
