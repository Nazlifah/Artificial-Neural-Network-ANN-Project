
import streamlit.components.v1 as components
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
from streamlit_drawable_canvas import st_canvas
import pickle
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load model and label encoder
model = load_model("color_ann_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

    # Set AI-themed page config
st.set_page_config(
    page_title="AI Color Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Background: animated gradient neon theme */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #533483);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    z-index: -1;
    filter: blur(0px);
}

/* Animate the gradient */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Centered container text */
h1 {
    font-size: 3rem;
    text-align: center;
    margin-top: 40px;
    margin-bottom: 20px;
    color: #fff;
    text-shadow: 0 0 10px #6a11cb, 0 0 20px #2575fc;
    letter-spacing: 1px;
}

/* Description text */
body, .main, .block-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #ffffff;
    text-shadow: 0 0 5px rgba(0,0,0,0.7);
}

/* Upload area */
.stFileUploader label {
    color: #ffffff;
    font-weight: bold;
    font-size: 1.2rem;
    text-shadow: 0 0 6px #8c52ff;
}

.stFileUploader div[data-baseweb="file-uploader"] {
    background-color: rgba(20, 20, 30, 0.6);
    border: 2px dashed #8c52ff;
    border-radius: 10px;
    padding: 20px;
    transition: all 0.3s ease;
    color: white;
}

.stFileUploader div[data-baseweb="file-uploader"]:hover {
    border-color: #a56bff;
    background-color: rgba(40, 40, 60, 0.7);
}

/* Button styling */
.stButton>button {
    background: linear-gradient(to right, #6a11cb, #8e44ad);
    color: white;
    font-weight: bold;
    font-size: 1rem;
    padding: 14px 30px;
    border: none;
    border-radius: 40px;
    cursor: pointer;
    transition: 0.4s ease;
    box-shadow: 0 0 12px #6a11cb, 0 0 20px #8e44ad;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #00f7ff, 0 0 30px #6a11cb;
}

/* Color box result */
div[data-testid="stMarkdownContainer"] > div {
    font-size: 1.1rem;
    color: white;
    text-shadow: 0 0 5px #000000;
}

/* Footer or small note */
small {
    display: block;
    margin-top: 40px;
    text-align: center;
    color: #ccc;
    font-size: 0.85rem;
    text-shadow: 0 0 4px #000;
}

/* Optional: body background to match */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    z-index: -1;
    background: linear-gradient(-45deg, #1f1b24, #2d2a33, #1f1b24, #292733);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    opacity: 1;
}

body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    z-index: -1;
    background: linear-gradient(-45deg, #1a1c22, #1e2028, #2a2d38, #20232c);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    opacity: 1;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
} 
            
/* Force the entire main app background to change */
section.main > div {
    background: none !important;
    background-color: transparent !important;
}

/* Optional: Remove default white in inner blocks */
.stApp {
    background: none !important;
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<h1>🤖 AI COLOUR DETECTION</h1>", unsafe_allow_html=True)
st.write("Upload an image and click to detect the top 3 closest color names using our ANN model.")

# Neon royal blue style for "Upload an image"
st.markdown("""
<h5 style="
    color: white;
    font-size: 16px;
    text-shadow: 0 0 5px #4169e1, 0 0 10px #4169e1, 0 0 20px #4169e1;
    font-weight: bold;
">Upload an image</h5>
""", unsafe_allow_html=True)

# File uploader with no default label
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy array (OpenCV)
    img_array = np.array(image)

    # Create a canvas for clicking
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)", 
        stroke_width=10,
        background_image=image,
        update_streamlit=True,
        height=img_array.shape[0],
        width=img_array.shape[1],
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data is not None:
      if'objects' in canvas_result.json_data and len(canvas_result.json_data['objects']) > 0:
        obj = canvas_result.json_data['objects'][-1] 
        if 'left' in obj and 'top' in obj:
            x = int(obj['left'])
            y = int(obj['top'])

            if 0 <= x < img_array.shape[1] and 0 <= y < img_array.shape[0]:
                r, g, b = img_array[y, x]
                rgb = np.array([[r, g, b]])
                prediction = model.predict(rgb)
                top3 = prediction[0].argsort()[-3:][::-1]
                top_colors = le.inverse_transform(top3)
                top_probs = prediction[0][top3]

                st.markdown("---")
                st.markdown("<h3 style='color: white;'>🌟 Top 3 Predicted Colors</h3>", unsafe_allow_html=True)
                for i in range(3):
                    color_box = f"<div style='display:flex; align-items:center; color:white;'><div style='background-color: rgb({r},{g},{b}); width: 20px; height: 20px; margin-right:10px; border-radius: 3px;'></div><b>{top_colors[i]}</b> — {top_probs[i]*100:.2f}% confidence</div>"
                    st.markdown(color_box, unsafe_allow_html=True)
            else:
                st.warning("Click inside the image area.")

    st.markdown("---")
    st.markdown("<small style='color:gray;'>Designed with ❤️ by your AI Team.</small>", unsafe_allow_html=True)
