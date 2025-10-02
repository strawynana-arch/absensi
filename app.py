import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Absensi Face ID", layout="wide")
st.title("Sistem Absensi Berbasis Face ID")
st.write("Aplikasi ini menggunakan pengenalan wajah untuk mencatat kehadiran.")

# --- PATH PENYIMPANAN ---
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'

# --- FUNGSI BANTU ---
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            # Data lama: tuple
            if isinstance(data, tuple):
                encodings, names = data
                return list(encodings), list(names)
            # Data baru: harus list
            if isinstance(data, dict):
                return list(data.get("encodings", [])), list(data.get("names", []))
            # Kalau error format → reset
            return [], []
    except Exception:
        return [], []

def save_known_faces(encodings, names):
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump((encodings, names), f)

def log_attendance(name):
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])

    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            last_time = datetime.strptime(last_entry['Waktu'].iloc[-1], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_time).total_seconds() < 60:
                return

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry = pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)

# --- MUAT DATA WAJAH ---
known_face_encodings, known_face_names = load_known_faces()

# Validasi: pastikan selalu list
if not isinstance(known_face_encodings, list):
    known_face_encodings = []
if not isinstance(known_face_names, list):
    known_face_names = []

# --- SIDEBAR ---
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])

# --- MODE 1: PENDAFTARAN WAJAH ---
if app_mode == "Pendaftaran Wajah":
    st.header("Form Pendaftaran Wajah Baru")
    new_name = st.text_input("Masukkan Nama Anda:")

    # Inisialisasi session state
    if "captured_images" not in st.session_state:
        st.session_state.captured_images = []

    img_file_buffer = st.camera_input("Ambil 5 Foto Wajah (dari berbagai angle)")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if st.button(f"Tambahkan Foto ke-{len(st.session_state.captured_images)+1}"):
            st.session_state.captured_images.append(cv2_img)
            st.success(f"Foto ke-{len(st.session_state.captured_images)} berhasil ditambahkan.")

    st.write(f"Jumlah foto yang sudah diambil: **{len(st.session_state.captured_images)}/5**")

    if st.session_state.captured_images:
        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in st.session_state.captured_images], width=150)

    if len(st.session_state.captured_images) == 5:
        if st.button("Proses & Simpan"):
            if new_name:
                with st.spinner("Memproses wajah..."):
                    for img in st.session_state.captured_images:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_img)
                        if face_locations:
                            face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                            # Pastikan append ke list
                            if isinstance(known_face_encodings, list):
                                known_face_encodings.append(face_encoding)
                            if isinstance(known_face_names, list):
                                known_face_names.append(new_name)
                        else:
                            st.warning("Salah satu foto tidak mendeteksi wajah.")

                    save_known_faces(known_face_encodings, known_face_names)
                    st.success(f"Wajah '{new_name}' berhasil disimpan!")
                    st.session_state.captured_images = []
            else:
                st.error("Nama tidak boleh kosong!")

# --- MODE 2: ABSENSI REAL-TIME ---
elif app_mode == "Absensi Real-time":
    st.header("Absensi Real-time dengan Kamera")

    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.encodings, self.names = load_known_faces()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                face_distances = face_recognition.face_distance(self.encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.names[best_match_index]
                        log_attendance(name)
                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(img, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

            return img

    # Tambahkan STUN server untuk WebRTC (Cloud Ready)
    webrtc_streamer(
        key="absensi",
        video_transformer_factory=FaceRecognitionTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"autoPlay": True, "playsinline": True, "muted": True}
    )

    st.subheader("Laporan Kehadiran")
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(df.sort_values(by="Waktu", ascending=False), use_container_width=True)
    except FileNotFoundError:
        st.info("Belum ada data absensi.")
