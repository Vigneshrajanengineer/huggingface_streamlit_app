
import streamlit as st
from transformers import pipeline

st.title("Gen_AI_Task(Sentiment Analysis ,Text Generation, Summarization)")

task = st.selectbox("Choose Task", ["Sentiment Analysis", "Text Generation", "Summarization"])
text_input = st.text_area("Enter Text")

if st.button("Run"):
    if task == "Sentiment Analysis":
        pipe = pipeline("sentiment-analysis")
        result = pipe(text_input)
    elif task == "Text Generation":
        pipe = pipeline("text-generation", model="gpt2")
        result = pipe(text_input, max_length=50)
    elif task == "Summarization":
        pipe = pipeline("summarization")
        result = pipe(text_input)
    st.json(result)
