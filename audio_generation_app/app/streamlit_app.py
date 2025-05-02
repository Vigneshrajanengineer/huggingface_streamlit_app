import streamlit as st
from bark import SAMPLE_RATE, generate_audio
import soundfile as sf

st.title("Text to Speech Generator (Bark)")

text = st.text_area("Enter your text", "This is AI generated speech using Bark.")

if st.button("Generate Audio"):
    with st.spinner("Generating..."):
        audio = generate_audio(text)
        sf.write("output.wav", audio, SAMPLE_RATE)
        st.audio("output.wav", format="audio/wav")
        with open("output.wav", "rb") as f:
            st.download_button("Download Audio", f, "output.wav")
