import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

model = BertForSequenceClassification.from_pretrained('nufa/model')
tokenizer = BertTokenizer.from_pretrained('nufa/model')

predict = pipeline("text-classification", model=model, tokenizer=tokenizer)

st.title('🤖 Klasifikasi Soal Otomatis')
question = st.text_input('Soal')
if st.button('🔍 Mulai Klasifikasi!'):
  if question != '':
    status = st.text('🔃 mengklasifikasikan soal...')
    result = predict(question)
    status.text('✅ klasifikasi selesai')
    st.write("💡 Tingkat Kognitif:", result[0]['label'])
    st.write(f"🎯 Keyakinan: {(result[0]['score'] * 100):.2f}%")