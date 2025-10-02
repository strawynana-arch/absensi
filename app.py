import streamlit as st
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from sklearn.neighbors import KNeighborsClassifier

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Absensi Face ID (Mediapipe)", layout="wide")
st.title("Sistem Absensi Berbasis Face ID (Mediapipe)")
st.write("Aplikasi ini menggunakan deteksi wajah Mediapipe untuk mencatat kehadiran.")

# --- PATH PENYIMPANAN ---
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'

# --- INISIALISASI MEDIAPIPE ---
mp_face = mp.solutions.face_detection

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
            last_time = datetime.strptime(last_entry['Waktu'].iloc[-1], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_time).total_seconds() < 60:
                return False

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.concat([df, pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True

def get_embedding(img):
    """Ambil fitur wajah sederhana (bbox + keypoints)"""
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        results = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            det = results.detections[0]
            box = det.location_data.relative_bounding_box
            keypoints = []
            for kp in det.location_data.relative_keypoints:
                keypoints.extend([kp.x, kp.y])
            emb = np.array([box.xmin, box.ymin, box.width, box.height] + keypoints, dtype=np.float32)
            return emb.flatten()
    return None

# --- MUAT DATA WAJAH ---
faces_data = load_known_faces()

# --- SIDEBAR MODE ---
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])
selected_camera = st.sidebar.selectbox("Pilih Kamera", ["default"])

# --- RESET DATA ---
if st.sidebar.button("🔴 Reset Data (Hapus Semua)"):
    if os.path.exists(ENCODINGS_PATH):
        os.remove(ENCODINGS_PATH)
    if os.path.exists(ATTENDANCE_PATH):
        os.remove(ATTENDANCE_PATH)
    st.sidebar.success("✅ Semua data wajah & absensi berhasil dihapus!")
    st.sidebar.info("Silakan mulai ulang aplikasi untuk daftar wajah baru.")

# --- MODE PENDAFTARAN WAJAH ---
if app_mode == "Pendaftaran Wajah":
    st.header("Form Pendaftaran Wajah Baru")
    new_name = st.text_input("Masukkan Nama Anda:")

    img_file_buffer = st.camera_input("Ambil Foto Wajah")

    if img_file_buffer and new_name:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        emb = get_embedding(cv2_img)
        if emb is not None:
            faces_data["names"].append(new_name)
            faces_data["embeddings"].append(emb.tolist())
            save_known_faces(faces_data)
            st.success(f"Wajah '{new_name}' berhasil disimpan!")
        else:
            st.error("Tidak ada wajah terdeteksi. Coba lagi.")

# --- MODE ABSENSI REAL-TIME ---
elif app_mode == "Absensi Real-time":
    st.header("Absensi Menggunakan Kamera")

    # Siapkan model KNN
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e, dtype=np.float32).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X, y)
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")

    placeholder = st.empty()  # buat notifikasi realtime

    class FaceRecognitionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            recognized_name = None

            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
                results = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results.detections:
                    for det in results.detections:
                        box = det.location_data.relative_bounding_box
                        h, w, _ = img.shape
                        x, y, ww, hh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)

                        # Gambar kotak
                        cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

                        # Ambil embedding
                        emb = get_embedding(img)
                        name = "Unknown"
                        if clf and emb is not None:
                            emb = emb.reshape(1, -1)
                            name = clf.predict(emb)[0]

                        # Hanya catat kalau bukan Unknown
                        if name != "Unknown":
                            if log_attendance(name):
                                recognized_name = f"✅ Hadir: {name}"
                        else:
                            recognized_name = "❌ Wajah tidak dikenali"

                        # Tulis nama di frame
                        cv2.putText(img, name, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Tambah jam realtime di pojok kiri atas
            cv2.putText(img, datetime.now().strftime("%H:%M:%S"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            # Update notifikasi di UI
            if recognized_name:
                placeholder.info(recognized_name)

            return img

    webrtc_streamer(
        key="absensi",
        video_transformer_factory=FaceRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"controls": False, "autoPlay": True}
    )

    st.subheader("Laporan Kehadiran")
    try:
        attendance_df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(attendance_df.sort_values(by='Waktu', ascending=False), use_container_width=True)
    except FileNotFoundError:
        st.info("Belum ada data kehadiran yang tercatat.")
