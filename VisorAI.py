import time
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from playsound import playsound  # Import playsound
from ultralytics import YOLO

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="VisorAI", layout="wide", page_icon="assets/icon.png")

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert logo to base64
logo_base64 = get_base64_image('assets/icon.png')

# Streamlit UI: Logo & Title
st.markdown(
    f"""
    <div style="display: flex; align-items: center; padding-top: 50px;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .stTabs [role="tablist"] button {
        font-size: 1.2rem;
        padding: 12px 24px;
        margin-right: 10px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- SESSION STATE INITIALIZATION -------------------
if "last_play_time" not in st.session_state:
    st.session_state.last_play_time = 0  # Prevents audio spam
if "last_detected_classes" not in st.session_state:
    st.session_state.last_detected_classes = set()  # Tracks detected classes
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "play_sound_flag" not in st.session_state:
    st.session_state.play_sound_flag = False  # Flag to control sound playback

# ------------------- SOUND MAPPING -------------------
SOUND_FILES = {
    "Child-Pedestrian Crossing": "assets/child_pedestrian_crossing.mp3",
    "Give Way": "assets/give_way.mp3",
    "Speed Limit": "assets/speed_limit.mp3",
    "Stop": "assets/stop.mp3",
}

# ------------------- LOAD YOLO MODEL -------------------
@st.cache_resource
def load_model():
    model_path = "assets/visor.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    print("✅ YOLO Model Loaded")
    return YOLO(model_path)

model = load_model()
if model is None:
    st.stop()

# ------------------- IMAGE PROCESSING -------------------
def process_image(image):
    """Processes an image for object detection."""
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image, verbose=False)
    detected_img = results[0].plot(conf=True)

    detected_classes = set()
    for detection in results[0].boxes:
        confidence = detection.conf[0].item()
        class_index = int(detection.cls[0].item())
        class_name = model.names[class_index]

        if confidence > 0.3 and class_name in SOUND_FILES:
            detected_classes.add(class_name)

    new_detections = detected_classes - st.session_state.last_detected_classes
    st.session_state.last_detected_classes = detected_classes  # Update detected classes

    return detected_img, new_detections

# ------------------- STREAMLIT UI -------------------
detect, model_info = st.tabs(["Detection", "Model Information"])

with detect:
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create two columns with equal width for the images
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Detect Traffic Signs"):
                # Process image and detect traffic signs
                detected_img, new_detections = process_image(image)
                
                # Display the detected image next to the uploaded image
                col2.image(cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_BGR2RGB), caption="Detected Image", use_container_width=True)
                
                # Wait for the detected image to render before playing sound
                time.sleep(1.5)  # Allow time for Streamlit to render the image
                
                # Trigger the sound feedback after the image has been displayed
                if new_detections:
                    # Play sound using Streamlit's audio widget
                    for class_name in new_detections:
                        audio_file = SOUND_FILES.get(class_name)
                        if audio_file and os.path.exists(audio_file):
                            with open(audio_file, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format="audio/mp3", start_time=0)
    else:
        st.session_state.processed_image = None  # Reset detected image when file is removed
        st.session_state.last_detected_classes.clear()  # Clear detected classes
        st.image("assets/bg.jpg")

with model_info:
    st.write("Model Information: This model detects traffic signs and plays corresponding sou")
    
# Footer Section
footer = f"""
<hr>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 10px 0;">
  <div style="flex-grow: 1; text-align: left;">
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
  </div>
  <!-- Copyright -->
  <div style="flex-grow: 1; text-align: center;">
    <span>Copyright 2024 | All Rights Reserved</span>
  </div>
  <!-- Social media icons -->
  <div style="flex-grow: 1; text-align: right;">
    <a href="https://www.linkedin.com" class="fa fa-linkedin" style="padding: 10px; font-size: 24px; background: #0077B5; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.instagram.com" class="fa fa-instagram" style="padding: 10px; font-size: 24px; background: #E1306C; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.youtube.com" class="fa fa-youtube" style="padding: 10px; font-size: 24px; background: #FF0000; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.facebook.com" class="fa fa-facebook" style="padding: 10px; font-size: 24px; background: #3b5998; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://twitter.com" class="fa fa-twitter" style="padding: 10px; font-size: 24px; background: #1DA1F2; color: white; text-decoration: none; margin: 5px;"></a>
  </div>
</div>
"""

# Display footer
st.markdown(footer, unsafe_allow_html=True)
