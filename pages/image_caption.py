import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from huggingface_hub import login
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login(st.secrets["HUGGINGFACE_TOKEN"])
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    return processor, model

processor, model = load_model()

st.title("Image caption")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare inputs
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    st.markdown("")
    st.write(caption)

if st.button(" back to main page"):
    st.switch_page("main.py")

