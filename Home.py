import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from utils import load_model

st.title('Klasifikasi Soal Otomatis')
model = load_model()
question = st.text_input('Soal')
if st.button('🔍 Mulai Klasifikasi!'):
  if question != '':
    status = st.text('🔃 mengklasifikasikan soal...')
    result = model(question)
    status.text('✅ klasifikasi selesai')
    st.write("💡 Tingkat Kognitif:", result[0]['label'])
    st.write(f"🎯 Keyakinan: {(result[0]['score'] * 100):.2f}%")