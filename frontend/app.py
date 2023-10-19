import streamlit as st
import numpy as np
import requests


url = 'http://api:8080/predict/'

st.title("Image Classifier")

uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)

    files = {'upload_file': (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
    res = requests.post(url, files=files)

    if res.status_code == 200:
        result = res.json()
        st.success(f"予測結果は {result['result_class']}")
    else:
        st.error(f"Error uploading file: {res.status_code}")

