import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("Stable Diffusion Image Generator")

prompt = st.text_input("Enter your prompt", "A fantasy landscape with dragons and castles")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        pipe = load_pipeline()
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("generated.png")
        with open("generated.png", "rb") as file:
            st.download_button("Download Image", file, "image.png", "image/png")
