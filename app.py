
import streamlit as st
from google.cloud import texttospeech
from pydub import AudioSegment
import tempfile
import os
from io import BytesIO
from scipy.spatial.distance import cosine
import numpy as np
import librosa

st.set_page_config(page_title="NameCall AI", layout="centered")

st.title("🎓 NameCall AI")
st.markdown("Say every name right — every time.")

# --- Sidebar settings ---
st.sidebar.header("Settings")
language_code = st.sidebar.selectbox("Language Code", ["en-US", "en-GB", "en-IN"])
gender = st.sidebar.selectbox("Voice Gender", ["NEUTRAL", "MALE", "FEMALE"])

# --- Google TTS Function ---
def synthesize_speech(text, lang="en-US", gender="NEUTRAL"):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=lang,
        ssml_gender=texttospeech.SsmlVoiceGender[gender]
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )
    return response.audio_content

# --- Name Input ---
name = st.text_input("Enter a name to pronounce:")

if st.button("🔊 Generate AI Pronunciation") and name:
    with st.spinner("Synthesizing voice..."):
        audio_data = synthesize_speech(name, lang=language_code, gender=gender)
        st.audio(audio_data, format="audio/mp3")

# --- Upload Real Pronunciation ---
st.markdown("### 📤 Upload Real Pronunciation (Recorded Voice)")
uploaded_file = st.file_uploader("Upload WAV/MP3 of actual name pronunciation", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        uploaded_audio_path = tmp_file.name

    # Compare similarity with AI voice
    if name:
        ai_audio = synthesize_speech(name, lang=language_code, gender=gender)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ai_file:
            ai_file.write(ai_audio)
            ai_audio_path = ai_file.name

        # Extract MFCC embeddings
        def extract_embedding(path):
            y, sr = librosa.load(path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return np.mean(mfcc, axis=1)

        try:
            emb1 = extract_embedding(uploaded_audio_path)
            emb2 = extract_embedding(ai_audio_path)
            similarity = 1 - cosine(emb1, emb2)
            st.success(f"🧠 Similarity Score: {similarity:.2f} (1.0 = identical)")
        except Exception as e:
            st.warning(f"Could not compare voices: {e}")
