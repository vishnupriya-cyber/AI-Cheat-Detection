import os
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import mysql.connector
from ultralytics import YOLO
# ---------------- OPTIONAL LIBS ----------------
try:
    import moviepy.editor as mp_video
    MOVIEPY_AVAILABLE = True
except Exception:
    MOVIEPY_AVAILABLE = False

# ---------------- YOLO MODEL ----------------
try:
    yolo_model = YOLO("yolov8n.pt")  # auto downloads
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_ERROR = ""
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = str(e)

# ---------------- GEMINI ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

# ---------------- CONFIG ----------------
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "1234"   # change if needed
DB_NAME = "ai_cheat_detector"

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Cheat Detector System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.set_page_config(
    page_title="AI Cheat Detector System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")

# ---------------- DATABASE ----------------
def get_db():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def init_db():
    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cheat_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_name VARCHAR(255),
                cheating_level VARCHAR(20),
                reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", ("admin",))
        exists = cursor.fetchone()[0]

        if exists == 0:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ("admin", "1234")
            )

        db.commit()
        cursor.close()
        db.close()
        return True
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return False

def save_to_db(video_name, level, reason):
    try:
        conn = get_db()
        cursor = conn.cursor()
        query = """
        INSERT INTO cheat_logs (video_name, cheating_level, reason)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (video_name, level, reason))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"DB Error: {e}")

def load_logs():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, video_name, cheating_level, reason, timestamp
            FROM cheat_logs
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        st.warning(f"Could not load logs: {e}")
        return []

# ---------------- LOGIN ----------------
def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if st.session_state.logged_in:
        return True
    st.markdown("<div class='hero-title'>🔐 Login</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>AI Cheat Detector Access Portal</div>", unsafe_allow_html=True)
    st.title("🔐 Login")
    with st.container():
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username=%s AND password=%s",
                (username, password)
            )
            user = cursor.fetchone()
            cursor.close()
            db.close()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")
        except Exception as e:
            st.error(f"Login error: {e}")

            st.markdown("<div class='small-note'>Default login: admin / 1234</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    return False

# ---------------- DETECTOR ----------------
class Detector:
    def __init__(self):
        self.alerts = []
        self.score = 0
        self.cooldown = 0

        if MEDIAPIPE_AVAILABLE:
            self.face = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.35
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        else:
            self.face = None
            self.face_mesh = None

        self.haar_face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def add_alert(self, msg, weight):
        if self.cooldown == 0:
            self.alerts.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg": msg
            })
            self.score += weight
            self.cooldown = 20

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

# ---------------- AUDIO ----------------
def detect_voice(video_path):
    if not MOVIEPY_AVAILABLE:
        return False

    try:
        clip = mp_video.VideoFileClip(video_path)
        audio = clip.audio

        if audio is None:
            clip.close()
            return False

        samples = audio.to_soundarray()
        volume = np.mean(np.abs(samples))
        clip.close()
        return volume > 0.02
    except Exception:
        return False

# ---------------- PHONE ----------------
def detect_phone_yolo(frame):
    if not YOLO_AVAILABLE:
        return False, frame

    results = yolo_model(frame, verbose=False)

    phone_detected = False

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            label = r.names[cls]

            # 📱 Detect phone class
            if label == "cell phone":
                phone_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "PHONE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return phone_detected, frame

# ---------------- GEMINI ----------------
def gemini_analysis(alerts, score, patterns):
    if not GEMINI_AVAILABLE:
        if score == 0 and len(alerts) == 0:
            return "Final Decision: CLEAN\nReason: No suspicious activity detected."
        return "Final Decision: SUSPICIOUS\nReason: Gemini not configured."

    prompt = f"""
You are a strict AI interview proctor.

Decide whether the candidate is cheating.

Consider:
- reading from another screen
- AI/ChatGPT usage
- phone usage
- outside help
- suspicious voice
- unnatural stillness
- suspicious answer delivery

DATA:
Alerts: {alerts}
Score: {score}
Patterns: {patterns}

Return in this exact structure:

Cheating Type: ...
Confidence: ...%
Risk Level: LOW / MEDIUM / HIGH
Final Decision: CLEAN / SUSPICIOUS / CHEATING
Reason: ...
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        res = model.generate_content(prompt)
        return res.text
    except Exception:
        if score == 0 and len(alerts) == 0:
            return "Final Decision: CLEAN."
        return "Final Decision: SUSPICIOUS\nReason: AI analysis unavailable."

# ---------------- DASHBOARD ----------------
def show_dashboard():
    st.markdown("<div class='hero-title'>📊 Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Analysis Results Overview</div>", unsafe_allow_html=True)

    rows = load_logs()

    if not rows:
        st.info("No logs found.")
        return

    import pandas as pd
    df = pd.DataFrame(
        rows,
        columns=["ID", "Video Name", "Cheating Level", "Reason", "Timestamp"]
    )

    # ---------- TABLE ----------
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📋 Session Records")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- GRAPH ----------
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📊 Cheating Distribution")

    level_counts = df["Cheating Level"].value_counts()

    # convert to dataframe for better chart
    chart_df = level_counts.reset_index()
    chart_df.columns = ["Level", "Count"]

    st.bar_chart(chart_df.set_index("Level"))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- HYBRID FACE DETECTION ----------------
def detect_faces_hybrid(detector, frame, gray, rgb):
    boxes = []

    if detector.face is not None:
        try:
            res = detector.face.process(rgb)
            if res and res.detections:
                h, w, _ = frame.shape
                for d in res.detections:
                    bbox = d.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    bw = max(1, int(bbox.width * w))
                    bh = max(1, int(bbox.height * h))
                    x2 = min(w - 1, x + bw)
                    y2 = min(h - 1, y + bh)
                    boxes.append((x, y, x2 - x, y2 - y))
        except Exception:
            pass

    if len(boxes) == 0 and detector.face_mesh is not None:
        try:
            mesh = detector.face_mesh.process(rgb)
            if mesh and mesh.multi_face_landmarks:
                h, w, _ = frame.shape
                for face_landmarks in mesh.multi_face_landmarks:
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x = max(0, min(xs))
                    y = max(0, min(ys))
                    x2 = min(w - 1, max(xs))
                    y2 = min(h - 1, max(ys))
                    bw = max(1, x2 - x)
                    bh = max(1, y2 - y)
                    boxes.append((x, y, bw, bh))
        except Exception:
            pass

    if len(boxes) == 0:
        faces = detector.haar_face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        for (x, y, w, h) in faces:
            boxes.append((x, y, w, h))

    return boxes

# ---------------- DETECTOR PAGE ----------------
def show_detector():
    st.markdown("""
<div style='text-align: center; margin-top: 10px;'>
    <h1 style='
        font-size: 42px;
        font-weight: 800;
        color: #4f46e5;
        margin-bottom: 5px;
    '>
        🤖 AI Cheat Detector System
    </h1>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style='
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    color: #64748b;
    margin-top: -5px;
    margin-bottom: 15px;
'>
    Cheating detection using face, phone, audio and AI reasoning
</div>
""", unsafe_allow_html=True)
    
    if not MEDIAPIPE_AVAILABLE:
        st.warning(f"MediaPipe not available: {MEDIAPIPE_ERROR}")
    if not MOVIEPY_AVAILABLE:
        st.warning("moviepy is not installed. Audio analysis will be skipped.")
    if not GEMINI_AVAILABLE:
        st.warning("Gemini API key not configured. AI decision will use fallback mode.")

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📁 Upload Interview Video")
    file = st.file_uploader("Drag and drop or browse video", type=["mp4", "avi", "mov"])
    st.markdown("</div>", unsafe_allow_html=True)

    if file is None:
        st.info("Upload a video to start.")
        return

    st.success("Video uploaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(file.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    detector = Detector()
    frame_display = st.empty()

    col1, col2, col3 = st.columns(3)
    no_face_ui = col1.empty()
    multi_face_ui = col2.empty()
    phone_ui = col3.empty()

    consecutive_no_face = 0
    multi_face = 0
    phone_count = 0
    reading_count = 0
    still_frames = 0
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        detector.update()

        h, w = frame.shape[:2]
        scale = 640 / max(w, h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_boxes = detect_faces_hybrid(detector, frame, gray, rgb)
        face_count = len(face_boxes)

        for (x, y, bw, bh) in face_boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        if face_count == 0:
            consecutive_no_face += 1
        else:
            consecutive_no_face = 0

        if consecutive_no_face > 15:
            detector.add_alert("No face detected", 2)

        if face_count > 1:
            multi_face += 1
            detector.add_alert("Multiple faces detected", 3)

        phone_flag, frame = detect_phone_yolo(frame)
        if phone_flag:
            phone_count += 1
            detector.add_alert("Phone detected", 5)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion = np.sum(diff)

            if motion < 300000:
                still_frames += 1
            else:
                still_frames = 0

            if still_frames > 50:
                reading_count += 1
                detector.add_alert("Reading from screen", 4)

        prev_gray = gray

        frame_display.image(
            frame,
            channels="BGR",
            use_container_width=True,
            caption="📹 Video Analysis"
        )

        with col1:
            no_face_ui.metric("🚫 No Face", consecutive_no_face)
        with col2:
            multi_face_ui.metric("👥 Multiple Faces", multi_face)
        with col3:
            phone_ui.metric("📱 Phone", phone_count)

        time.sleep(0.03)

    cap.release()

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("🎤 Audio Analysis")
    voice_flag = detect_voice(temp_video_path)

    if voice_flag:
        detector.add_alert("Voice detected", 3)
        st.error("Voice detected ⚠️")
    else:
        st.success("No suspicious voice")
    st.markdown("</div>", unsafe_allow_html=True)

    patterns = {
        "no_face_frames": consecutive_no_face,
        "multiple_faces": multi_face,
        "phone": phone_count,
        "reading": reading_count,
        "voice": voice_flag
    }

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📊 Score")

    if detector.score < 10:
        level = "LOW"
    elif detector.score < 25:
        level = "MEDIUM"
    else:
        level = "HIGH"

    st.metric("Score", detector.score)
    st.metric("Risk", level)

    progress = min(detector.score / 30, 1.0)
    st.progress(progress)
    st.write(f"Cheating Probability: {int(progress * 100)}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("🚨 Result")
    if detector.score >= 25:
        st.error("❌ CHEATING")
    elif detector.score >= 10:
        st.warning("⚠️ SUSPICIOUS")
    else:
        st.success("✅ CLEAN")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.subheader("📋 Alerts")
    if detector.alerts:
        for a in detector.alerts:
            st.warning(f"{a['time']} - {a['msg']}")
    else:
        st.info("No alerts")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🤖 AI Final Decision")
    result = gemini_analysis(detector.alerts, detector.score, patterns)
    if detector.score == 0 and len(detector.alerts) == 0:
        st.success("✅ NO CHEATING DETECTED")
    elif "CHEATING" in result.upper():
        st.error("❌ CHEATING DETECTED")
    elif "SUSPICIOUS" in result.upper():
        st.warning("⚠️ SUSPICIOUS")
    else:
        st.success("✅ NO CHEATING DETECTED")

    reason_text = f"Alerts: {detector.alerts}\n\nAI Decision:\n{result}"
    save_to_db(file.name, level, reason_text)

    try:
        os.remove(temp_video_path)
    except Exception:
        pass

# ---------------- APP FLOW ----------------
if not init_db():
    st.stop()

if not login():
    st.stop()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.success(f"👤 {st.session_state.username}")
    page = st.selectbox("Menu", ["Detector", "Dashboard"])
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

if page == "Detector":
    show_detector()
else:
    show_dashboard()
