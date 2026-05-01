import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

from src.components.gradcam import make_gradcam_heatmap

# ---------------- CONFIG ----------------
MODEL_PATH = "models/tensorflow/vgg_model.h5"
LAST_CONV_LAYER = "block5_conv3"
IMG_SIZE = (224, 224)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TB Detection AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align: center;'>🩺 Tuberculosis Detection System</h1>
<p style='text-align: center;'>AI-powered chest X-ray analysis with explainability</p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:

    col1, col2 = st.columns(2)

    # -------- LEFT: IMAGE --------
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", width="stretch")

    # -------- RIGHT: PREDICTION --------
    with col2:

        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0][0]

        if pred > 0.5:
            label = "🦠 TB Detected"
            color = "red"
            confidence = pred
        else:
            label = "✅ Normal"
            color = "green"
            confidence = 1 - pred

        st.markdown(f"## Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        st.metric("Confidence Score", f"{confidence*100:.1f}%")
        st.progress(float(confidence))

    # -------- GRAD-CAM --------
    st.markdown("---")
    st.subheader("🔥 Model Attention (Grad-CAM)")

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)

    img_cv = cv2.cvtColor(np.array(image.resize(IMG_SIZE)), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + img_cv
    overlay = np.uint8(overlay)

    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width="stretch")