import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
import time
import librosa

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MedAi", layout="wide")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 MedAi</h1>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "doctors" not in st.session_state:
    st.session_state.doctors = [{"name": "Dr. Shibin Kallis", "id": "D23"}]
if "recording" not in st.session_state:
    st.session_state.recording = False

if "transcription" not in st.session_state:
    st.session_state.transcription = ""

if "last_audio_time" not in st.session_state:
    st.session_state.last_audio_time = 0

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return whisper.load_model("small")   # 🔥 better than small

model = load_model()

# ---------------- AUDIO SETTINGS ----------------
sd.default.samplerate = 16000   # 🔥 native Whisper rate
sd.default.channels = 1

# ---------------- AUDIO PROCESSING ----------------
def clean_audio(audio):
    # Normalize
    audio = audio / np.max(np.abs(audio) + 1e-6)

    # Reduce noise (simple gate)
    audio[np.abs(audio) < 0.01] = 0

    return audio

def record_audio(duration=3):
    try:
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        st.error(f"Mic error: {e}")
        return None

# ---------------- DOCTOR SECTION ----------------
st.subheader("👨‍⚕️ Doctor Selection")

col1, col2, col3 = st.columns([3,1,1])

doctor_options = [f"{d['name']} ({d['id']})" for d in st.session_state.doctors]
selected_doctor = col1.selectbox("Select Doctor", doctor_options)

new_name = col2.text_input("Name")
new_id = col3.text_input("ID")

col_add, col_del = st.columns(2)

if col_add.button("➕ Add Doctor"):
    if new_name and new_id:
        st.session_state.doctors.append({"name": new_name, "id": new_id})
        st.success("Doctor Added")

if col_del.button("❌ Delete Doctor"):
    if st.session_state.doctors:
        st.session_state.doctors.pop()
        st.warning("Last Doctor Removed")

# ---------------- PATIENT DETAILS ----------------
st.subheader("🧾 Patient Details")

p1, p2, p3 = st.columns(3)
patient_name = p1.text_input("Patient Name")
patient_age = p2.number_input("Age", min_value=0, max_value=120)
patient_id = p3.text_input("Patient ID")

# ---------------- RECORD BUTTON ----------------
st.subheader("🎙️ Recording")

col_start, col_stop = st.columns(2)

if col_start.button("▶️ Start Recording"):
    st.session_state.recording = True
    st.success("Recording Started")

if col_stop.button("⏸️ Stop Recording"):
    st.session_state.recording = False
    st.warning("Recording Stopped")

# ---------------- LIVE TRANSCRIPTION ----------------
st.subheader("📝 Live Transcription")

placeholder = st.empty()

if st.session_state.recording:
    current_time = time.time()

    # 🔥 control loop timing (NO BLINK)
    if current_time - st.session_state.last_audio_time > 3:
        st.session_state.last_audio_time = current_time

        audio = record_audio(3)

        if audio is not None:
            audio = clean_audio(audio)

            # skip silence
            if np.max(np.abs(audio)) > 0.02:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    wav.write(f.name, 16000, audio)

                    result = model.transcribe(
                        f.name,
                        language="en",
                        fp16=False,
                        temperature=0.2,   # 🔥 improves accuracy
                        best_of=3
                    )

                    text = result["text"].strip()

                    if text:
                        st.session_state.transcription += " " + text

# 🔥 Stable text box (NO BLINK)
transcription = st.text_area(
    "Transcription Output",
    value=st.session_state.transcription,
    height=250
)

# ---------------- ACTION BUTTONS ----------------
col_v, col_r = st.columns(2)

if col_v.button("✅ Verify"):
    st.success("Verified!")

if col_r.button("📄 Generate Report"):
    report = f"""
Doctor: {selected_doctor}
Patient: {patient_name}
Age: {patient_age}
Patient ID: {patient_id}

Report:
{transcription}
"""
    st.download_button("⬇️ Download Report", report, file_name="report.txt")