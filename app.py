import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav

from report_generator import extract_findings
from pdf_generator import generate_pdf

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MedAi", layout="wide")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 MedAi</h1>", unsafe_allow_html=True)

# ---------------- AUDIO SETTINGS ----------------
sd.default.device = 1
sd.default.samplerate = 16000
sd.default.channels = 1

# ---------------- SESSION STATE ----------------
if "transcription" not in st.session_state:
    st.session_state.transcription = ""

if "doctors" not in st.session_state:
    st.session_state.doctors = [{"name": "Dr. Shibin Kallis", "id": "D23"}]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return whisper.load_model("medium", device="cpu")

model = load_model()

# ---------------- RECORD FUNCTION ----------------
def record_audio(duration=5):
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

selected_doctor = col1.selectbox(
    "Select Doctor",
    st.session_state.doctors,
    format_func=lambda d: f"{d['name']} ({d['id']})"
)

doctor_name = selected_doctor["name"]

col2.text_input("Name")
col3.text_input("ID")

# ---------------- PATIENT DETAILS ----------------
st.subheader("🧾 Patient Details")

p1, p2, p3 = st.columns(3)
patient_name = p1.text_input(
    "Patient Name",
    key="patient_name"
)

patient_age = p2.number_input(
    "Age",
    min_value=0,
    max_value=120,
    key="patient_age"
)

patient_id = p3.text_input(
    "Patient ID",
    key="patient_id"
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"],
    key="gender"
)



# ---------------- RECORD BUTTON ----------------
st.subheader("🎙️ Recording")

col1, col2 = st.columns(2)

if "recording" not in st.session_state:
    st.session_state.recording = False

if col1.button("▶️ Start Recording"):
    st.session_state.recording = True

if col2.button("⏹️ Stop Recording"):
    st.session_state.recording = False


# 🎤 RECORD ONLY WHEN ACTIVE
if st.session_state.recording:

    st.info("Recording... Speak now")

    audio = record_audio(5)

    if audio is not None:
        st.write("Audio level:", np.max(np.abs(audio)))

        audio = audio * 10.0   # 🔥 BOOST

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            wav.write(f.name, 16000, audio)

            try:
                result = model.transcribe(f.name, fp16=False)
                text = result["text"].strip()

                if text:
                    st.success(f"Detected: {text}")
                    st.session_state.transcription += " " + text

            except Exception as e:
                st.error(f"Whisper error: {e}")

# ---------------- TRANSCRIPTION BOX ----------------
st.subheader("📝 Live Transcription")

transcription = st.text_area(
    "Transcription Output (Editable)",
    value=st.session_state.transcription,
    height=200
)

# 🔥 sync manually
st.session_state.transcription = transcription

# ---------------- ACTION BUTTONS ----------------
col_v, col_r = st.columns(2)

if col_v.button("✅ Verify"):
    st.success("Verified!")

if col_r.button("📄 Generate Report"):

    if not st.session_state.transcription:
        st.warning("No transcription available")
    else:
        findings = extract_findings(st.session_state.transcription)

        st.write("### 🧠 Extracted Findings")
        st.write(findings)

        # ✅ GET NAME SAFELY
        final_name = st.session_state.get("patient_name", "").strip()

        st.write("FINAL NAME:", final_name)  # debug

        if not final_name:
            st.error("⚠️ Please enter patient name before generating report")
            st.stop()

        # ✅ GENERATE PDF
        generate_pdf(
            findings,
            final_name,
            patient_age,
            gender,
            doctor_name,
            transcription
        )

        with open("ultrasound_report.pdf", "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                f,
                file_name="ultrasound_report.pdf"
            )