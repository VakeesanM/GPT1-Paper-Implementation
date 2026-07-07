import streamlit as st
from model import generate
import time

st.set_page_config(page_title="GPT-1", page_icon="", layout="wide")
st.header("""GPT1 Impemented from "Improving Language Understanding by Generative Pre-Training" """)
st.write("This model was trained on the TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories")
st.write("Specifically, it trained on a corpus of 11,000 short stories over 15 Epochs")




query = st.text_area("""Enter Query. Its best to give sentences like 'once upon a time', rather than a questions""")
temp = st.number_input("Temperature", min_value=0.01, value=0.7)
length = st.number_input("Max Length of Generation", min_value=1,value=50)
if st.button("Generate"):
    st.write_stream(generate(query, temp=temp, max_len=length))