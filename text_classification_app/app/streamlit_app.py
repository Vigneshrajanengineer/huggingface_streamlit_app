import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Text Classification App")
text_input = st.text_area("Enter your text:", "I love this movie!")

if st.button("Classify"):
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        label = "Positive" if prediction == 1 else "Negative"
        st.write(f"**Prediction:** {label}")
