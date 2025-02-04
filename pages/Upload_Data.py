import streamlit as st
import pandas as pd
import numpy as np
import io
from utils import load_model

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    processed_data = output.getvalue()
    return processed_data

model = load_model()

st.title('Klasifikasi Soal Otomatis')
data_soal = st.file_uploader('Masukan data soal')

if data_soal is not None:
    df = pd.read_excel(data_soal, names=['soal'], header=None)
    list_soal = df.soal.tolist()
    
    if st.button('🔍 Mulai Klasifikasi!'):
        if list_soal is not None:
            indikator = st.text('🔃 mengklasifikasi soal...')
            result = model(list_soal)
            indikator.text('✅ klasifikasi selesai')
            
            csv = convert_df(pd.concat([df, pd.DataFrame(result)], axis=1))
            excel_data = convert_df_to_excel(pd.concat([df, pd.DataFrame(result)], axis=1))

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='hasil_klasifikasi.csv',
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name="hasil_klasifikasi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
