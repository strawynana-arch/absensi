import streamlit as st
import os
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import pytz
from sklearn.neighbors import KNeighborsClassifier
import time

# Pengaturan
st.set_page_config(page_title="Absensi Face ID", layout="wide")
WIB = pytz.timezone('Asia/Jakarta')
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Session state untuk auto-capture
if 'capture_mode' not in st.session_state:
    st.session_state.capture_mode = False
if 'last_capture_time' not in st.session_state:
    st.session_state.last_capture_time = 0

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
    if name == "Unknown":
        return False
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])
    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            try:
                last_time_str = last_entry['Waktu'].iloc[-1]
                last_time = WIB.localize(datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S'))
                if (datetime.now(WIB) - last_time).total_seconds() < 60:
                    return False
            except:
                pass
    now = datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.concat([df, pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True

def detect_face_ultra_fast(img):
    """Ultra optimized untuk HP"""
    # Resize super kecil
    h, w = img.shape[:2]
    scale = 0.3  # 30% dari ukuran asli
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Deteksi dengan parameter sangat agresif
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(15, 15))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Scale kembali ke ukuran asli
        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        
        # Extract face untuk embedding
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = face_gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (40, 40))  # Super kecil untuk speed
        
        # Embedding simpel: histogram 32 bins + posisi
        hist = cv2.calcHist([face_roi], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        img_h, img_w = img.shape[:2]
        features = np.concatenate([hist, [x/img_w, y/img_h, w/img_w, h/img_h]])
        
        return features.astype(np.float32), (x, y, w, h)
    
    return None, None

# Load data
faces_data = load_known_faces()

# Sidebar
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])

if st.sidebar.button("Reset Data"):
    if os.path.exists(ENCODINGS_PATH):
        os.remove(ENCODINGS_PATH)
    if os.path.exists(ATTENDANCE_PATH):
        os.remove(ATTENDANCE_PATH)
    st.sidebar.success("Data berhasil dihapus!")
    st.rerun()

# PENDAFTARAN
if app_mode == "Pendaftaran Wajah":
    st.header("Pendaftaran Wajah Baru")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        new_name = st.text_input("Nama Lengkap:")
        img_file = st.camera_input("Ambil Foto Wajah")
    with col2:
        st.info(f"Terdaftar: {len(faces_data['names'])} orang")

    if img_file and new_name:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        emb, bbox = detect_face_ultra_fast(cv2_img)
        
        if emb is not None:
            faces_data["names"].append(new_name)
            faces_data["embeddings"].append(emb.tolist())
            save_known_faces(faces_data)
            st.success(f"Wajah '{new_name}' berhasil disimpan!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Wajah tidak terdeteksi")

# ABSENSI REAL-TIME
elif app_mode == "Absensi Real-time":
    st.header("Absensi Real-time (Mode Interval)")
    
    # Load model
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e, dtype=np.float32).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
            clf.fit(X, y)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.warning("Belum ada wajah terdaftar")
        st.stop()
    
    st.info("Mode ini akan mengambil foto otomatis setiap 3 detik untuk mengenali wajah")
    
    # Interval setting
    col1, col2 = st.columns([3, 1])
    with col1:
        interval = st.slider("Interval capture (detik)", 2, 10, 3)
    with col2:
        if st.button("Stop" if st.session_state.capture_mode else "Start"):
            st.session_state.capture_mode = not st.session_state.capture_mode
            st.rerun()
    
    st.write(f"Waktu: {datetime.now(WIB).strftime('%H:%M:%S WIB')}")
    
    # Placeholder untuk notifikasi dan gambar
    status_placeholder = st.empty()
    image_placeholder = st.empty()
    
    # Mode capture aktif
    if st.session_state.capture_mode:
        status_placeholder.success("Mode deteksi aktif - Posisikan wajah Anda")
        
        # Auto-capture dengan interval
        current_time = time.time()
        if current_time - st.session_state.last_capture_time >= interval:
            st.session_state.last_capture_time = current_time
            
            # Ambil foto
            camera_photo = st.camera_input("Capture", label_visibility="hidden", key=f"cam_{current_time}")
            
            if camera_photo:
                bytes_data = camera_photo.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                emb, bbox = detect_face_ultra_fast(cv2_img)
                
                if emb is not None and bbox is not None:
                    emb_reshaped = emb.reshape(1, -1)
                    name = clf.predict(emb_reshaped)[0]
                    
                    # Gambar kotak
                    x, y, w, h = bbox
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(cv2_img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(cv2_img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Tampilkan
                    image_placeholder.image(cv2_img, channels="BGR", use_column_width=True)
                    
                    if name != "Unknown":
                        if log_attendance(name):
                            status_placeholder.success(f"Absensi berhasil: {name}")
                            time.sleep(2)
                        else:
                            status_placeholder.info(f"{name} sudah absen")
                    else:
                        status_placeholder.warning("Wajah tidak dikenali")
                else:
                    status_placeholder.warning("Tidak ada wajah terdeteksi")
            
            time.sleep(0.5)
            st.rerun()
    else:
        status_placeholder.info("Klik 'Start' untuk memulai deteksi")
    
    st.divider()
    
    # Laporan
    st.subheader("Laporan Hari Ini")
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        today = datetime.now(WIB).strftime('%Y-%m-%d')
        today_df = df[df['Waktu'].str.contains(today)]
        
        if not today_df.empty:
            st.metric("Total Hadir", len(today_df))
            st.dataframe(today_df.sort_values(by='Waktu', ascending=False), use_column_width=True)
        else:
            st.info("Belum ada yang absen hari ini")
    except FileNotFoundError:
        st.info("Belum ada data kehadiran")
