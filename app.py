import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# 1. Konfigurasi Halaman & Identitas
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.sidebar.title("Aplikasi Deteksi Objek")
st.sidebar.markdown("**Tim: When yahh**")
st.sidebar.text("1. Aditya Setyo N\n2. Dito Ramadhan\n3. Husein Hanafiyah")
st.sidebar.markdown("---")

# 2. Pengaturan Parameter
conf_threshold = st.sidebar.slider("Confidence Score Threshold", 0.0, 1.0, 0.25)
media_type = st.sidebar.radio("Pilih Jenis Media", ["Gambar", "Video"])

# 3. Load Model YOLO (Di-cache agar tidak meload ulang setiap ada interaksi)
@st.cache_resource
def load_model():
    # Menggunakan YOLOv8 Nano untuk keseimbangan performa dan akurasi (otomatis diunduh)
    return YOLO('yolov8n.pt') 

model = load_model()

st.title("Aplikasi Deteksi Objek Gambar & Video Berbasis YOLO")
st.write("Unggah media Anda untuk melihat kelas objek, bounding box, dan confidence score.")

# 4. Logika Pemrosesan Gambar
if media_type == "Gambar":
    uploaded_file = st.file_uploader("Unggah berkas gambar", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Tampilkan dalam dua kolom (Sebelum & Sesudah)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
            
        if st.button("Deteksi Objek"):
            with st.spinner("Sedang memproses..."):
                # Menjalankan inferensi YOLO
                results = model.predict(image, conf=conf_threshold)
                
                # Menggambar bounding box bawaan ultralytics (.plot())
                res_plotted = results[0].plot()
                
                # Konversi BGR (OpenCV) ke RGB (Streamlit/Pillow)
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.image(res_plotted_rgb, caption="Hasil Deteksi", use_container_width=True)
                    
                st.success("Deteksi selesai!")

# 5. Logika Pemrosesan Video
elif media_type == "Video":
    uploaded_file = st.file_uploader("Unggah berkas video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        if st.button("Mulai Proses Video"):
            with st.spinner("Memproses video frame demi frame..."):
                # Simpan video upload ke temporary file karena OpenCV butuh path file
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                
                # Placeholder untuk merender frame secara berurutan di UI
                stframe = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Inferensi per frame
                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    res_plotted = results[0].plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    
                    # Tampilkan frame yang sudah diberi bounding box
                    stframe.image(res_plotted_rgb, channels="RGB", use_container_width=True)
                    
                cap.release()
                st.success("Pemrosesan video selesai!")