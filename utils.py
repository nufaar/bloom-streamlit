import torch
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1  # Gunakan GPU jika tersedia
    
    model = BertForSequenceClassification.from_pretrained('indobert-bloom')
    tokenizer = BertTokenizer.from_pretrained('indobert-bloom')
    # model = BertForSequenceClassification.from_pretrained('nufa/model')
    # tokenizer = BertTokenizer.from_pretrained('nufa/model')
    predict = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return predict