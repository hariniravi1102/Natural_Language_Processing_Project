import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. Please check your PyTorch / GPU install.")

device = "cuda"
st.write(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0)}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    device_map="cuda",
)


def generate_image(prompt: str):
    with torch.autocast("cuda"):
        out = pipe(prompt, num_inference_steps=30)
        image = out.images[0]
    return image

st.title("Text to Image Generator")

user_prompt = st.text_input("Text to image generation:")

if st.button("Enter"):
    if user_prompt:
        st.write("Generating image on GPU...")
        img = generate_image(user_prompt)
        st.image(img, caption=f"Generated for: {user_prompt}", use_column_width=True)
    else:
        st.warning("Please enter some text before pressing Enter.")

if st.button(" back to main page"):
    st.switch_page("main.py")
