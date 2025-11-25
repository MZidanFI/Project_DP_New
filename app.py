import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from image_enhancement import enhance_image
from detection import video_detection
from counting import video_counting

# =============== 1. PAGE CONFIGURATION ===============
st.set_page_config(
    page_title="YOLOv11 Web GUI",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# =============== 2. STYLES & ASSETS ===============
st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
""", unsafe_allow_html=True)

# Load CSS
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ File 'assets/style.css' tidak ditemukan. Menggunakan style default.")

# Header UI
st.markdown("""
<div class="header">
    <div class="neon-icon"><span class="material-icons">psychology</span></div>
    <h1>YOLOv11 Web GUI</h1>
    <p>Futuristic Object Detection — Enhance, Detect, and Count in Style 🚀</p>
    <div class="pulse"></div>
</div>
""", unsafe_allow_html=True)

# =============== 3. HELPER FUNCTIONS ===============

@st.cache_resource
def load_model(model_path):
    """Memuat model ke cache agar tidak reload terus menerus."""
    return YOLO(model_path)

def save_uploaded_file(uploaded_file, folder="inputs"):
    """Menyimpan file upload ke folder sementara."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ============== 4. SIDEBAR SETTINGS ==============
st.sidebar.markdown("""<h2><span class="material-icons">tune</span> Settings</h2>""", unsafe_allow_html=True)

# Global Settings
model_size = st.sidebar.selectbox("YOLOv11 Model Size", ["nano", "small", "medium"], index=0)
mode = st.sidebar.radio("Mode", ("Image", "Video"))

# Variable Initialization
enhance_type = "None"
video_mode = "Detection"
confidence = 0.25
save_outputs = True

# Mode Specific Settings
if mode == "Image":
    enhance_type = st.sidebar.selectbox(
        "Image Enhancement",
        ["None", "HE", "CLAHE", "CS", "Brightness", "Gamma", "Unsharp", "Bilateral", "Saturation"]
    )
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    save_outputs = st.sidebar.checkbox("💾 Save annotated outputs", value=True)

elif mode == "Video":
    video_mode = st.sidebar.selectbox("Video Mode", ["Detection", "Counting"])
    enhance_type = "None"
    save_outputs = st.sidebar.checkbox("💾 Save annotated outputs", value=True)
    confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# Upload File
uploaded_file = st.file_uploader(
    f"📤 Upload {'Image' if mode == 'Image' else 'Video'}",
    type=["jpg", "jpeg", "png", "bmp"] if mode == "Image" else ["mp4", "mov", "avi", "mkv"]
)

# Model Loading Section
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Model Status")
MODEL_PATH = f"weights/best_{model_size}.pt"

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        st.sidebar.success(f"✅ Model loaded: **{model_size.upper()}**")
    else:
        st.sidebar.error(f"❌ Weights not found: {MODEL_PATH}")
        st.sidebar.info("Please place your .pt files in the 'weights/' folder.")
except Exception as e:
    st.sidebar.error("❌ Failed to load model")
    st.sidebar.warning(str(e))

# =============== 5. MAIN LOGIC ===============
st.markdown("<div class='main-section'>", unsafe_allow_html=True)

if uploaded_file is None:
    # Tampilan Awal Kosong
    st.markdown("""
    <div class='empty-upload'>
        <span class="material-icons">cloud_upload</span><br>
        <p>Upload an image or video to begin detection.</p>
    </div>
    """, unsafe_allow_html=True)

elif model is None:
    st.error("⚠️ Model belum dimuat. Periksa folder weights Anda.")

else:
    # Setup Paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Menentukan folder output
    if mode == "Image":
        folder_category = "images"
        sub_category = enhance_type
    else:
        folder_category = "videos"
        sub_category = video_mode.capitalize() # Detection / Counting

    #/outputs/videos/model/counting/detecton
    base_output_dir = os.path.join("outputs", folder_category, model_size, sub_category)
    os.makedirs(base_output_dir, exist_ok=True)

    input_dir = os.path.join("inputs", "images" if mode == "Image" else "videos")
    os.makedirs(input_dir, exist_ok=True)
    
    # --- IMAGE MODE ---
    if mode == "Image":
        # Simpan input sementara
        input_path = save_uploaded_file(uploaded_file, folder="inputs/images")
        
        # Baca & Proses Image
        img = cv2.imread(input_path)
        img_proc = enhance_image(img, enhance_type)

        with st.spinner("🧠 Running YOLOv11 detection..."):
            results = model.predict(source=img_proc, conf=confidence, imgsz=640, verbose=False)
        
        res = results[0]
        annotated = res.plot() if hasattr(res, "plot") else img_proc.copy()
        count = len(res.boxes) if hasattr(res, "boxes") else 0

        # Menampilkan Hasil
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original / Enhanced Input")
            st.image(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.caption(f"Result (Objects: {count})")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Kartu Statistik
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-header'>
                <span class="material-icons result-icon">visibility</span>
                <h2>Detection Result</h2>
            </div>
            <p><strong>Detected objects:</strong> {count}</p>
        </div>
        """, unsafe_allow_html=True)

        # Simpan Output
        if save_outputs:
            out_name = os.path.join(base_output_dir, f"det_{enhance_type}_{timestamp}.jpg")
            cv2.imwrite(out_name, annotated)
            st.success(f"Saved to: `{out_name}`")

    # --- VIDEO MODE ---
    elif mode == "Video":
        # Simpan input sementara
        input_path = save_uploaded_file(uploaded_file, folder="inputs/videos")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            st.error("❌ Cannot open uploaded video.")
        else:
            # Video Info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Setup Writer
            writer = None
            out_video_path = None
            if save_outputs:
                out_video_path = os.path.join(base_output_dir, f"output_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

            # UI Placeholder
            stframe = st.empty()
            progress_bar = st.progress(0)
            
            st.info(f"Processing Video: {width}x{height} @ {fps:.1f} FPS")

            try:
                # Panggil Fungsi Eksternal sesuai Mode
                if video_mode == "Detection":
                    video_detection(
                        cap, model, enhance_image, enhance_type, confidence,
                        writer, stframe, progress_bar, total_frames
                    )
                elif video_mode == "Counting":
                    video_counting(
                        cap, model, enhance_image, enhance_type, confidence,
                        writer, stframe, progress_bar, total_frames
                    )
                
                if save_outputs and out_video_path:
                    st.success(f"Video processing complete! Saved to `{out_video_path}`")
                    
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
            finally:
                cap.release()
                if writer:
                    writer.release()

st.markdown("</div>", unsafe_allow_html=True)