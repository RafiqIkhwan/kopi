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
        # Download model jika belum ada
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Mengunduh model dari Google Drive..."):
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error("❌ Gagal memuat model dari Google Drive")
        st.error(e)
        return None

# --- FUNGSI PREPROCESSING ---
def preprocess_image(image):
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    # pastikan RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# --- UI ---
st.title("☕ Klasifikasi Citra Kopi")
st.markdown("Upload gambar biji/daun kopi untuk diprediksi menggunakan model **ResNet50 + TPE Optimization**.")

model = load_trained_model()

if model is not None:
    input_method = st.radio("Pilih metode input gambar:", ("Upload File", "Ambil Foto dari Kamera"))

    image = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    elif input_method == "Ambil Foto dari Kamera":
        camera_image = st.camera_input("Ambil foto menggunakan kamera")
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is not None:
        image = ImageOps.exif_transpose(image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption='Gambar yang dipilih', use_container_width=True)

        with col2:
            st.write("Menganalisis...")

            # Progress bar
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)

            # Prediksi
            processed_image = preprocess_image(image)
            prediction_prob = model.predict(processed_image)
            predicted_class_idx = np.argmax(prediction_prob)
            predicted_class_label = CLASS_NAMES[predicted_class_idx]
            confidence = prediction_prob[0][predicted_class_idx] * 100

            st.success("Selesai!")
            st.metric(
                label="Hasil Prediksi",
                value=predicted_class_label,
                delta=f"{confidence:.2f}% Akurat"
            )

        st.divider()

        # Visualisasi
        st.subheader("📊 Tingkat Keyakinan Model")

        prob_df = pd.DataFrame({
            'Kelas': CLASS_NAMES,
            'Probabilitas': prediction_prob[0]
        })

        st.bar_chart(prob_df.set_index('Kelas'))

        with st.expander("Lihat detail probabilitas"):
            st.dataframe(prob_df.style.format({'Probabilitas': '{:.4%}'}))

else:
    st.warning("Model tidak dapat dimuat.")
