
# Hugging Face Transformers: Advanced Tutorial (Text Generation, Summarization, Streamlit API)

## 1. Text Generation (GPT-2)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

## 2. Summarization
from transformers import pipeline

summarizer = pipeline("summarization")
text = '''
The Hugging Face Transformers library provides general-purpose architectures for Natural Language Understanding and Natural Language Generation with over 32+ pretrained models.
'''
print(summarizer(text, max_length=40, min_length=10, do_sample=False))

## 3. Translation
translator = pipeline("translation_en_to_fr")
print(translator("Hugging Face is creating a tool that democratizes AI."))

## 4. Streamlit App (Save as streamlit_app.py and run with `streamlit run streamlit_app.py`)

import streamlit as st
from transformers import pipeline

st.title("Hugging Face Transformers Demo")

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
