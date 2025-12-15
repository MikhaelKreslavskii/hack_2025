import zipfile

import streamlit as st
import pandas as pd
import numpy as np
import lasio
import io
import plotly.express as px

st.title('MVP Хакатона')
uploaded_zip = st.file_uploader("Загрузите папку как ZIP", type="zip")
if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip) as zip_ref:
        file_names = zip_ref.namelist()
        file_names_show = [file.split(sep='/')[1] for file in file_names]
        try:
            df_list = {}
            for name in file_names:
                content_bytes = zip_ref.read(name)
                las_text = content_bytes.decode('utf-8', errors='ignore')  # или 'utf-8'
                try:
                    # Парсим LAS
                    las = lasio.read(io.StringIO(las_text))
                    df = las.df().reset_index()
                except Exception as e:
                    print(e)
                    continue

                # Конвертируем в DataFrame
                df = las.df()
                df = df.reset_index()
                df.columns = ['DEPTH', 'VALUE']
                key_name = str(name).split(sep='/')[1].split(sep='.')[0]
                df.dropna(inplace=True)
                df_list[key_name] = df
                # st.dataframe(df.head(10))
                #
                # fig = px.bar(
                #     df,
                #     x=df.index,  # или другой индекс, если есть
                #     y='DEPTH',
                #     color='VALUE',
                #     color_discrete_map={0: 'lightblue', 1: 'orange'},  # цвета: синий для 0, оранжевый для 1
                #     orientation='h',  # горизонтальные столбцы для лучшего вида глубины
                #     title='Глубина и наличие материала',
                #     labels={'глубина': 'Глубина', 'наличие_материала': 'Наличие материала'}
                # )
                #
                # fig.update_layout(
                #
                #     xaxis_title="Индекс/Позиция",
                #     height=800  # высота для длинной оси глубины
                # )
                #
                # st.plotly_chart(fig, use_container_width=True)

                # Показываем информацию о файле
                #st.success(f"✅ Файл загружен: {name}")

            st.dataframe(df_list['WELL_076'])
        except Exception as e:
            st.error(f"Error reading file: {e}")
        else:
            st.info("Please upload a file.")

st.text('Интерполяция')
