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
st.set_page_config(
    page_title="Absensi Face ID", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

WIB = pytz.timezone('Asia/Jakarta')
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi helper
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
    new_row = pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True

def detect_face_optimized(img):
    h, w = img.shape[:2]
    scale = 0.4
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(20, 20))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = face_gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        hist = cv2.calcHist([face_roi], [0], None, [48], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        img_h, img_w = img.shape[:2]
        features = np.concatenate([hist, [x/img_w, y/img_h, w/img_w, h/img_h]])
        return features.astype(np.float32), (x, y, w, h)
    return None, None

# Load data
faces_data = load_known_faces()

# Header
st.title("🎯 Sistem Absensi Face ID")
st.caption("Mobile Optimized - Gunakan mode foto untuk performa terbaik")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Absensi", "📝 Pendaftaran", "📊 Laporan"])

# TAB ABSENSI
with tab1:
    st.header("Absensi dengan Foto")
    
    # Load model
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e, dtype=np.float32).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
            clf.fit(X, y)
            st.success(f"✅ Model siap - {len(faces_data['names'])} wajah terdaftar")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Belum ada wajah terdaftar. Silakan daftar di tab Pendaftaran")
    
    st.info(f"🕒 Waktu: **{datetime.now(WIB).strftime('%H:%M:%S WIB')}**")
    
    # Camera input native Streamlit (paling ringan!)
    camera_photo = st.camera_input("Ambil foto untuk absensi")
    
    if camera_photo and clf:
        try:
            # Decode image
            bytes_data = camera_photo.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Detect face
            with st.spinner("Mengenali wajah..."):
                emb, bbox = detect_face_optimized(img)
            
            if emb is not None:
                # Predict
                name = clf.predict(emb.reshape(1, -1))[0]
                
                # Draw on image
                x, y, w, h = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Show result
                st.image(img, channels="BGR", use_container_width=True)
                
                # Log attendance
                if name != "Unknown":
                    if log_attendance(name):
                        st.success(f"✅ **Absensi berhasil: {name}**")
                        st.balloons()
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.info(f"ℹ️ {name} sudah absen dalam 1 menit terakhir")
                else:
                    st.error("❌ Wajah tidak dikenali. Silakan daftar terlebih dahulu")
            else:
                st.warning("⚠️ Tidak ada wajah terdeteksi di foto. Coba lagi dengan pencahayaan yang lebih baik")
                
        except Exception as e:
            st.error(f"Error processing: {str(e)}")

# TAB PENDAFTARAN
with tab2:
    st.header("Pendaftaran Wajah Baru")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_name = st.text_input("Nama Lengkap:", key="reg_name")
    
    with col2:
        st.metric("Total Terdaftar", len(faces_data['names']))
    
    if new_name:
        camera_reg = st.camera_input("Ambil foto wajah untuk pendaftaran")
        
        if camera_reg:
            try:
                bytes_data = camera_reg.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                with st.spinner("Memproses wajah..."):
                    emb, bbox = detect_face_optimized(img)
                
                if emb is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img, new_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    st.image(img, channels="BGR", use_container_width=True)
                    
                    if st.button("💾 Simpan Wajah Ini", type="primary"):
                        faces_data["names"].append(new_name)
                        faces_data["embeddings"].append(emb.tolist())
                        save_known_faces(faces_data)
                        st.success(f"✅ Wajah {new_name} berhasil disimpan!")
                        st.balloons()
                        time.sleep(1.5)
                        st.rerun()
                else:
                    st.error("❌ Wajah tidak terdeteksi. Pastikan wajah terlihat jelas")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Masukkan nama terlebih dahulu")
    
    # Daftar yang sudah terdaftar
    if faces_data['names']:
        with st.expander("👥 Lihat Daftar Terdaftar"):
            for i, name in enumerate(faces_data['names'], 1):
                st.write(f"{i}. {name}")

# TAB LAPORAN
with tab3:
    st.header("Laporan Kehadiran")
    
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        today = datetime.now(WIB).strftime('%Y-%m-%d')
        today_df = df[df['Waktu'].str.contains(today)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hadir Hari Ini", len(today_df) if not today_df.empty else 0)
        with col2:
            st.metric("Total Terdaftar", len(faces_data['names']))
        with col3:
            st.metric("Total Absensi", len(df))
        
        st.subheader("📅 Kehadiran Hari Ini")
        if not today_df.empty:
            st.dataframe(
                today_df.sort_values(by='Waktu', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Belum ada yang absen hari ini")
        
        with st.expander("📜 Semua Riwayat Kehadiran"):
            st.dataframe(
                df.sort_values(by='Waktu', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
    except FileNotFoundError:
        st.info("Belum ada data kehadiran")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    with col2:
        if st.button("🗑️ Reset Semua Data", type="secondary"):
            if os.path.exists(ENCODINGS_PATH):
                os.remove(ENCODINGS_PATH)
            if os.path.exists(ATTENDANCE_PATH):
                os.remove(ATTENDANCE_PATH)
            st.success("✅ Data berhasil dihapus!")
            time.sleep(1)
            st.rerun()
