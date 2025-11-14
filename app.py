# app.py
import streamlit as st
from detect_pipeline import process_frame
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(page_title="Car Color Detection", layout="wide")
st.title("🚦 Car Colour Detection (HSV) & People Counter")

st.markdown(
    "Upload an image. Red rectangle = **blue car**, Blue rectangle = **other**. "
    "Use the slider to tune how strict the blue test is."
)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("**Input**")
    uploaded = st.file_uploader(
        "Upload image (jpg/png/webp)", 
        type=["jpg", "jpeg", "png", "webp"]
    )
with col2:
    st.markdown("**Controls**")
    
    blue_thresh = st.slider(
    "Blue pixel fraction threshold", 0.02, 0.30, 0.25, 0.01
)

    conf_thresh = st.slider(
        "Detection confidence", 0.1, 0.8, 0.35, 0.05
    )

if uploaded:
    pil = Image.open(uploaded).convert("RGB")

    # 👇 NO MORE DEPRECATION POPUP
    st.image(pil, caption="Input Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Processing..."):
            arr = np.array(pil)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            out_bgr, total, blue_c, other_c, people = process_frame(
                bgr,
                blue_threshold=blue_thresh,
                conf_thresh=conf_thresh
            )
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        st.image(out_rgb, caption="Annotated Output", use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cars", total)
        c2.metric("Blue Cars", blue_c)
        c3.metric("Other Cars", other_c)
        c4.metric("People", people)

        # Download Button
        buf = io.BytesIO()
        Image.fromarray(out_rgb).save(buf, format="JPEG")
        st.download_button(
            "Download Result",
            buf.getvalue(),
            file_name="result.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload an image to start.")

