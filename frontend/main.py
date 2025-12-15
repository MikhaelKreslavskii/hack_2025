import streamlit as st
import pandas as pd
import numpy as np
import lasio
import io
import plotly.express as px

st.title('MVP Хакатона')
uploaded_file = st.file_uploader("Выбери файл с данными", type=['las'])
if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        las_text = uploaded_file.read().decode('utf-8', errors='ignore')

        # Создаем StringIO для lasio
        las_stringio = io.StringIO(las_text)
        las = lasio.read(las_stringio)

        # Конвертируем в DataFrame
        df = las.df()
        df = df.reset_index()
        df.columns = ['DEPTH', 'VALUE']
        df.dropna(inplace=True)
        st.dataframe(df.head(10))

        fig = px.bar(
            df,
            x=df.index,  # или другой индекс, если есть
            y='DEPTH',
            color='VALUE',
            color_discrete_map={0: 'lightblue', 1: 'orange'},  # цвета: синий для 0, оранжевый для 1
            orientation='h',  # горизонтальные столбцы для лучшего вида глубины
            title='Глубина и наличие материала',
            labels={'глубина': 'Глубина', 'наличие_материала': 'Наличие материала'}
        )

        fig.update_layout(

            xaxis_title="Индекс/Позиция",
            height=800  # высота для длинной оси глубины
        )

        st.plotly_chart(fig, use_container_width=True)

        # Показываем информацию о файле
        st.success(f"✅ Файл загружен: {uploaded_file.name}")


    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a file.")