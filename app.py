import streamlit as st
import os
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import pytz
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from sklearn.neighbors import KNeighborsClassifier

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Absensi Face ID", layout="wide")
st.title("Sistem Absensi Berbasis Face ID")
st.write("Aplikasi ini menggunakan deteksi wajah OpenCV untuk mencatat kehadiran.")

# --- TIMEZONE WIB ---
WIB = pytz.timezone('Asia/Jakarta')

# --- PATH PENYIMPANAN ---
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'

# --- LOAD HAAR CASCADE untuk deteksi wajah ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- FUNGSI BANTU ---
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                names = data.get("names", [])
                embeddings = data.get("embeddings", [])
            else:
                encodings, names = data
                embeddings = encodings

            # pastikan semua embedding panjang sama
            clean_names, clean_embs = [], []
            target_len = None
            for n, e in zip(names, embeddings):
                e = np.array(e, dtype=np.float32).flatten()
                if target_len is None:
                    target_len = len(e)
                if len(e) == target_len:
                    clean_names.append(n)
                    clean_embs.append(e)
            return {"names": clean_names, "embeddings": clean_embs}
    except FileNotFoundError:
        return {"names": [], "embeddings": []}

def save_known_faces(data):
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def log_attendance(name):
    """ Catat kehadiran hanya kalau nama valid (bukan Unknown) """
    if name == "Unknown":
        return False

    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])

    # Cek duplikasi (1 menit terakhir)
    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            last_time = WIB.localize(datetime.strptime(last_entry['Waktu'].iloc[-1], '%Y-%m-%d %H:%M:%S'))
            if (datetime.now(WIB) - last_time).total_seconds() < 60:
                return False

    now = datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.concat([df, pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True

def get_embedding(img):
    """Ambil fitur wajah menggunakan OpenCV (versi lebih cepat)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize untuk processing lebih cepat
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces = face_cascade.detectMultiScale(small_gray, 1.2, 4, minSize=(30, 30))
    
    if len(faces) > 0:
        # Ambil wajah pertama, scale kembali koordinat
        (x, y, w, h) = faces[0] * 2
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize ke ukuran lebih kecil untuk processing cepat
        face_roi = cv2.resize(face_roi, (64, 64))
        
        # Hitung histogram sebagai "embedding" sederhana
        hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256])  # Reduce bins
        hist = cv2.normalize(hist, hist).flatten()
        
        # Tambahkan fitur ukuran dan posisi
        features = np.concatenate([hist, [x/img.shape[1], y/img.shape[0], w/img.shape[1], h/img.shape[0]]])
        
        return features.astype(np.float32)
    
    return None

# --- MUAT DATA WAJAH ---
faces_data = load_known_faces()

# --- SIDEBAR MODE ---
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])

# --- RESET DATA ---
if st.sidebar.button("🔴 Reset Data (Hapus Semua)"):
    if os.path.exists(ENCODINGS_PATH):
        os.remove(ENCODINGS_PATH)
    if os.path.exists(ATTENDANCE_PATH):
        os.remove(ATTENDANCE_PATH)
    st.sidebar.success("✅ Semua data wajah & absensi berhasil dihapus!")
    st.rerun()

# --- MODE PENDAFTARAN WAJAH ---
if app_mode == "Pendaftaran Wajah":
    st.header("📝 Form Pendaftaran Wajah Baru")
    new_name = st.text_input("Masukkan Nama Anda:")

    img_file_buffer = st.camera_input("📸 Ambil Foto Wajah")

    if img_file_buffer and new_name:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        emb = get_embedding(cv2_img)
        if emb is not None:
            faces_data["names"].append(new_name)
            faces_data["embeddings"].append(emb.tolist())
            save_known_faces(faces_data)
            st.success(f"✅ Wajah '{new_name}' berhasil disimpan!")
            st.balloons()
        else:
            st.error("❌ Tidak ada wajah terdeteksi. Coba lagi dengan pencahayaan yang lebih baik.")

# --- MODE ABSENSI REAL-TIME ---
elif app_mode == "Absensi Real-time":
    st.header("📹 Absensi Menggunakan Kamera")

    # Siapkan model KNN
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e, dtype=np.float32).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
            clf.fit(X, y)
            st.success(f"✅ Model siap! {len(faces_data['names'])} wajah terdaftar.")
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
    else:
        st.warning("⚠️ Belum ada wajah terdaftar. Silakan daftar dulu di menu 'Pendaftaran Wajah'.")

    # Frame counter untuk skip processing (optimasi performa)
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    notification_placeholder = st.empty()

    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_skip = 2  # Process setiap 2 frame (optimasi)
            self.counter = 0
            self.last_name = None
            
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Skip beberapa frame untuk performa lebih baik
            self.counter += 1
            if self.counter % self.frame_skip != 0:
                return img
            
            # Resize untuk processing lebih cepat
            small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah dengan parameter dioptimasi
            faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Scale kembali koordinat ke ukuran asli
                x, y, w, h = x*2, y*2, w*2, h*2
                
                # Gambar kotak
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Ambil embedding (hanya jika ada model)
                name = "Unknown"
                if clf:
                    emb = get_embedding(img)
                    if emb is not None:
                        emb = emb.reshape(1, -1)
                        name = clf.predict(emb)[0]

                # Catat kehadiran
                if name != "Unknown" and name != self.last_name:
                    if log_attendance(name):
                        notification_placeholder.success(f"✅ Absensi berhasil: {name}")
                        self.last_name = name

                # Tulis nama di frame
                cv2.putText(img, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tambah jam WIB realtime di pojok kiri atas
            wib_time = datetime.now(WIB).strftime("%H:%M:%S WIB")
            cv2.putText(img, wib_time, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img

    # WebRTC dengan setting optimasi
    webrtc_ctx = webrtc_streamer(
        key="absensi",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceRecognitionTransformer,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20}  # Limit FPS untuk performa
            },
            "audio": False
        },
        async_processing=True,
    )

    st.divider()
    
    st.subheader("📊 Laporan Kehadiran Hari Ini")
    try:
        attendance_df = pd.read_csv(ATTENDANCE_PATH)
        today = datetime.now(WIB).strftime('%Y-%m-%d')
        today_df = attendance_df[attendance_df['Waktu'].str.contains(today)]
        
        if not today_df.empty:
            st.dataframe(today_df.sort_values(by='Waktu', ascending=False), use_container_width=True)
            st.metric("Total Kehadiran Hari Ini", len(today_df))
        else:
            st.info("Belum ada data kehadiran hari ini.")
            
    except FileNotFoundError:
        st.info("Belum ada data kehadiran yang tercatat.")
    
    # Tampilkan semua data
    with st.expander("📜 Lihat Semua Riwayat Kehadiran"):
        try:
            all_df = pd.read_csv(ATTENDANCE_PATH)
            st.dataframe(all_df.sort_values(by='Waktu', ascending=False), use_container_width=True)
        except FileNotFoundError:
            st.info("Belum ada riwayat.")
