import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import time
import gdown
import os

# --- KONFIGURASI ---
MODEL_PATH = "model.keras"
FILE_ID = "1zV1rbvywa1PO8R783G8n2tSsmbTQ945l"  # ⬅️ GANTI INI
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Leaf rust', 'Leaf spot', 'Sehat'] 

st.set_page_config(
    page_title="Prediksi Kualitas Kopi - ResNet50",
    page_icon="☕",
    layout="centered"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Mengunduh model dari Google Drive..."):
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error("❌ Gagal memuat model.")
        st.error(e)
        return None

# --- FUNGSI PREPROCESSING ---
def preprocess_image(image):
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI ---
st.title("☕ Klasifikasi Citra Kopi")
st.markdown("Analisis penyakit daun kopi menggunakan model **ResNet50**.")

model = load_trained_model()

if model is not None:
    input_method = st.radio("Pilih metode input:", ("Upload File", "Ambil Foto dari Kamera"))
    image = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    elif input_method == "Ambil Foto dari Kamera":
        camera_image = st.camera_input("Ambil foto")
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is not None:
        image = ImageOps.exif_transpose(image)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption='Gambar Input', use_container_width=True)

        with col2:
            st.write("🔍 **Menganalisis...**")
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.005)
                my_bar.progress(percent_complete + 1)

            # Prediksi
            processed_image = preprocess_image(image)
            prediction_prob = model.predict(processed_image)
            predicted_class_idx = np.argmax(prediction_prob)
            predicted_class_label = CLASS_NAMES[predicted_class_idx]
            confidence = prediction_prob[0][predicted_class_idx] * 100

            st.success("Analisis Selesai!")
            st.metric(
                label="Hasil Klasifikasi", 
                value=predicted_class_label, 
                delta=f"{confidence:.2f}% Confidence"
            )

        st.divider()

        # Detail Probabilitas (Hanya dalam Expander, tanpa Bar Chart)
        prob_df = pd.DataFrame({
            'Kelas': CLASS_NAMES,
            'Probabilitas': prediction_prob[0]
        })

        with st.expander("📊 Lihat detail skor probabilitas"):
            st.dataframe(
                prob_df.style.format({'Probabilitas': '{:.4%}'}),
                use_container_width=True
            )
else:
    st.warning("Pastikan model tersedia untuk melanjutkan.")
