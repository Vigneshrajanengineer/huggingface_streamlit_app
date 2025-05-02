import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Offline Text Generation App")
prompt = st.text_area("Enter your prompt:", "The future of AI is")

max_length = st.slider("Max length", 20, 200, 100)

if st.button("Generate"):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Generated Text:")
    st.write(generated)
