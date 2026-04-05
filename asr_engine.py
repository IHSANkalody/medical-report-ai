import whisper
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import os
import time

# Load Whisper model
model = whisper.load_model("small")  # use "small" for better accuracy

SAMPLE_RATE = 16000


def record_audio(duration):
    print("Recording...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    print("Recording finished.")
    print("Audio mean:", np.mean(np.abs(audio)))

    return audio.flatten()


def preprocess_audio(audio):

    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return None

    normalized_audio = audio / max_val

    normalized_audio = (normalized_audio * 32767).astype("int16")

    return normalized_audio


def transcribe_audio(audio):

    if audio is None or len(audio) < 1000:
        return ""

    temp_path = "temp_audio.wav"

    wav.write(temp_path, SAMPLE_RATE, audio)

    time.sleep(0.2)
    print("Transcribing...")
    result = model.transcribe(temp_path, language="en")
    print("Done transcribing")
    time.sleep(0.2)

    try:
        os.remove(temp_path)
    except:
        pass
    print("Transcribed text:", result["text"])
    return result["text"]