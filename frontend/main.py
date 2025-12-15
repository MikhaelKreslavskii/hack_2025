

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests


st.title('MVP Хакатона')
uploaded_zip = st.file_uploader("Загрузите папку как ZIP", type="zip")
if uploaded_zip is not None:
    files = {'zip': uploaded_zip}
    response = requests.post("http://localhost:8000/api/v1/upload_files/", files=files)

    if response.status_code == 200:
        st.success("Архив обработан!")
        st.json(response.json())
    else:
        st.error(f"Ошибка: {response.text}")



st.text('Интерполяция')
