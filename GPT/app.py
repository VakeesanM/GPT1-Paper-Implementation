import streamlit as st
from model import generate

st.set_page_config(page_title="GPT-1", page_icon="", layout="wide")
st.header("""GPT1 Impemented from "Improving Language Understanding by Generative Pre-Training" """)
st.write("By Vakeesan Moorthy")
st.write("This model was trained on the TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories")
st.write("Specifically, it was trained on a corpus of 11,000 short stories over 15 Epochs")


query = st.text_area("""Enter Query. Its best to give sentences like 'once upon a time', rather than questions""")

"Model Parameters:"
col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("Temperature", min_value=0.01, value=0.7)
with col2:
    length = st.number_input("Max Length of Generation", min_value=1,value=50)
with col3:
    top_k = st.number_input("Top K Sampling", min_value=1, max_value=50256, value=40)
if st.button("Generate"):
    st.write_stream(generate(query, temp=temp, max_len=length, top_k=top_k))