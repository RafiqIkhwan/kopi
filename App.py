import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import time

# --- KONFIGURASI ---
MODEL_PATH = "model/best_model3test_resnet_tpe_kopi.keras"
IMG_SIZE = (224, 224)

# ⚠️ PENTING: Ubah list ini sesuai dengan urutan abjad nama folder kelas di dataset kamu
# Contoh: CLASS_NAMES = ['Biji Rusak', 'Biji Sehat', 'Biji Pecah', ...]
# Karena di kode training kamu pakai flow_from_directory, urutannya otomatis Alphabetical.
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
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{MODEL_PATH}' ada di folder yang sama.")
        st.error(f"Error detail: {e}")
        return None

# --- FUNGSI PREPROCESSING ---
def preprocess_image(image):
    """
    Menyamakan preprocessing dengan ImageDataGenerator yang digunakan saat training:
    1. Resize ke 224x224
    2. Konversi ke Array
    3. Preprocess input ala ResNet50
    """
    # Resize gambar agar sesuai input model
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Konversi ke array numpy
    img_array = np.asarray(image)
    
    # Pastikan format RGB (jika PNG ada alpha channel, buang)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Preprocessing khusus ResNet50 (sama seperti training)
    # Ini melakukan normalisasi zero-center sesuai ImageNet
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # Tambahkan dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- UI UTAMA ---
st.title("☕ Klasifikasi Citra Kopi")
st.markdown("Upload gambar biji/daun kopi untuk diprediksi menggunakan model **ResNet50 + TPE Optimization**.")

model = load_trained_model()

if model is not None:
    # Pilih metode input
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
        # Tampilkan gambar yang diupload
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Gambar yang dipilih', use_container_width=True)
        
        with col2:
            st.write("Menganalisis...")
            
            # Progress bar simulasi (biar terlihat interaktif)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            
            # Proses Prediksi
            processed_image = preprocess_image(image)
            prediction_prob = model.predict(processed_image)
            predicted_class_idx = np.argmax(prediction_prob)
            predicted_class_label = CLASS_NAMES[predicted_class_idx]
            confidence = prediction_prob[0][predicted_class_idx] * 100

            st.success("Selesai!")
            st.metric(label="Hasil Prediksi", value=predicted_class_label, delta=f"{confidence:.2f}% Akurat")

        st.divider()
        
        # --- VISUALISASI PROBABILITAS ---
        st.subheader("📊 Tingkat Keyakinan Model")
        
        # Membuat DataFrame untuk chart
        prob_df = pd.DataFrame({
            'Kelas': CLASS_NAMES,
            'Probabilitas': prediction_prob[0]
        })
        
        
        # Tampilkan Data Mentah (Opsional, dalam expander)
        with st.expander("Lihat detail probabilitas"):
            st.dataframe(prob_df.style.format({'Probabilitas': '{:.4%}'}))

else:
    
    st.warning("Silakan jalankan kode training Anda terlebih dahulu untuk menghasilkan file 'best_model_resnet_tpe_kopi.keras'.",  )