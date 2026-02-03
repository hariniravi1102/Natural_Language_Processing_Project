import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    with st.spinner("Downloading punkt..."):
        nltk.download("punkt")
model_name = "deep-learning-analytics/GrammarCorrector"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct_grammar(text):

    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(input_ids, max_length=1000, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return corrected_text

st.title("Grammatical check")

# Text input box
user_prompt = st.text_area("Enter the text for grammatical error:", height=300)


def correct_long_text(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sentence in sentences:
        if sentence.strip():
            try:
                corrected = correct_grammar(sentence)
                corrected_sentences.append(corrected)
            except Exception as e:
                corrected_sentences.append(sentence)
    return ' '.join(corrected_sentences)


if st.button("Enter"):
    if user_prompt:
        text = correct_long_text(user_prompt)
        st.text_area(
            label="Output",
            value= text,
            height=300,
            max_chars=None,
            key="output_text_area",
            disabled=True,
            help="This is corrected sentence output."
        )
    else:
        st.warning("Please enter some text before pressing Enter.")


if st.button("back to main page"):
    st.switch_page("main.py")
