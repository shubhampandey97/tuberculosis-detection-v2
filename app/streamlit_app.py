import os
import sys

import streamlit as st
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.components import inference  # noqa: E402

DOCS_URL = "https://github.com/shubhampandey97/tuberculosis-detection-v2/blob/main/docs/failure_analysis.md"

st.set_page_config(page_title="TB screening research demo", page_icon=":material/radiology:", layout="wide")

st.title("TB chest X-ray screening: research demo")
st.warning(
    "**Research demo, not for clinical use.** The model shows dataset source bias: on the test set it "
    "still separates TB from Normal (AUC about 0.97) with the lungs hidden, so it partly recognises "
    f"where an image comes from rather than TB. See [docs/failure_analysis.md]({DOCS_URL}).",
    icon=":material/warning:",
)


@st.cache_resource
def load_model_and_threshold():
    return inference.load_model(), inference.load_threshold()


model, threshold = load_model_and_threshold()
st.caption(
    f"Model: ResNet50 (ImageNet, frozen base), trained on the Kaggle TB chest X-ray dataset. "
    f"Decision threshold {threshold:.3f}, tuned on the validation split for 95% sensitivity "
    f"(test sensitivity was 0.935)."
)

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.stop()

raw = uploaded.getvalue()
try:
    image = Image.open(uploaded).convert("RGB")  # validates the file; grayscale/RGBA/palette -> RGB
    x = inference.preprocess_bytes(raw)  # same TensorFlow decode + resize as training
except Exception:  # PIL or TensorFlow could not decode the file
    st.error("Could not read this file as a JPEG or PNG image.", icon=":material/error:")
    st.stop()

score = inference.predict(model, x)
flagged = score >= threshold

left, right = st.columns(2)
with left:
    st.image(image, caption="Uploaded image")
with right:
    with st.container(border=True):
        if flagged:
            st.error("TB suspected: flag for review", icon=":material/flag:")
        else:
            st.success("No TB flag", icon=":material/check_circle:")
        st.metric(
            "Uncalibrated model score",
            f"{score:.3f}",
            help="The model's sigmoid output. It is not a probability of TB: the scores are not calibrated.",
        )
        st.caption(
            f"Flag if score ≥ {threshold:.3f}. The score is not a probability and not a measure of certainty: "
            "on test data the model under-estimated P(TB) in the middle range."
        )

st.subheader("Where the model looked (Grad-CAM)")
overlay, empty = inference.explain(model, x, score, threshold)
if empty:
    st.info("The Grad-CAM map is empty for this image (no positive evidence at the last conv layer).",
            icon=":material/info:")
st.image(overlay, caption=f"Evidence for the decision ({'TB' if flagged else 'Normal'}), ResNet50 conv5_block3_out",
         width=480)
st.caption(
    "Grad-CAM is coarse (7×7 grid) and only partly passed a randomisation sanity check for this model. "
    "On test images, both models put weight on image borders and handwritten annotations, not only on the lungs."
)
