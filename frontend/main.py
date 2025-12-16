import zipfile
from io import BytesIO
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

from interpolation import read_navigator_file, create_extrapolated_surface, create_original_area_boundary

method_descriptions = {
    'linear': 'Линейная интерполяция (scipy)',
    'cubic': 'Кубическая интерполяция (scipy)',
    'nearest': 'Ближайший сосед (scipy)',
    'rbf': 'Радиальные базисные функции',
    'idw': 'Обратное расстояние (Inverse Distance Weighting)',
    'kriging': 'Кригинг (Гауссовские процессы)',
    'svr': 'Support Vector Regression'
}

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
uploaded_h = st.file_uploader("Загрузите архив с фактическими данными EFF_H, H", type="zip")
uploaded_h_predict = st.file_uploader("Загрузите файл с предсказанными данными EFF_H, H")

#читаем предсказанный файл и конвертим в датафрейм
if uploaded_h is not None and uploaded_h_predict is not None:
    with st.spinner("🔄 Обработка данных..."):
        df_predict_h = pd.DataFrame()
        file_content = BytesIO(uploaded_h_predict.read())
        df_predict_h = read_navigator_file(file_content)
        df_predict_h = df_predict_h.rename(columns={'value': 'h_kol'})
        st.success('Данные для предсказания успешно загружены')
        st.dataframe(df_predict_h)
        x_predict = df_predict_h['x'].values.tolist()
        y_predict = df_predict_h['y'].values.tolist()
        h_kol_predict = df_predict_h['h_kol'].values.tolist()

    if uploaded_h is not None:
        df_eff_h = pd.DataFrame()
        df_h = pd.DataFrame()
        zip_content = BytesIO(uploaded_h.read())

        with zipfile.ZipFile(zip_content) as zip_ref:
            file_names = zip_ref.namelist()
            st.write("Файлы в архиве:", [f.split('/')[-1] for f in file_names])

            for name in file_names:
                try:
                    content_bytes = zip_ref.read(name)
                    las_text = content_bytes.decode('utf-8', errors='ignore')

                    # Создаем BytesIO объект вместо filepath
                    file_like = BytesIO(content_bytes)
                    print(name)
                    if 'FF' in name.upper():
                        print('EFF')
                        df_eff_h = read_navigator_file(file_like)  # Модифицируйте функцию
                        print(df_eff_h.head())
                    else:
                        print('H')
                        df_h = read_navigator_file(file_like)
                        print(df_h.head())

                except Exception as e:
                    st.error(f"Ошибка файла {name}: {e}")
                    continue

        # Обработка DataFrame...
        if not df_eff_h.empty and not df_h.empty:
            df_eff_h = df_eff_h.rename(columns={'value': 'eff_h'})
            df_h = df_h.rename(columns={'value': 'h'})

        df_merged = pd.merge(df_eff_h, df_h, on=['x', 'y', 'z', 'well'], how='inner')
        df_merged['h_kol'] = df_merged['eff_h'] / df_merged['h']
        st.success(f"✅ Объединено {len(df_merged)} скважин")
        st.dataframe(df_merged.head())
        methods = ['rbf', 'idw', 'linear', 'cubic', 'kriging', 'svr']
        # Получаем данные для визуализации
        x = df_merged['x'].values
        y = df_merged['y'].values

        x_predict = df_predict_h['x'].values
        y_predict = df_predict_h['y'].values
        z_predict = df_predict_h['z'].values

        z_coords = df_merged['z'].values  # координаты Z скважин
        z_values = df_merged['h_kol'].values  # значения h_kol
        # Dropdown
        selected_method = st.selectbox(
            "Выберите метод интерполяции:",
            methods,
            index=0,  # По умолчанию первый элемент
            help="rbf - рекомендуется"
        )

        st.write(f"Выбран: {selected_method}")

        # Параметры для выбранного метода
        method_params = {}
        if selected_method == 'rbf':
            method_params = {'rbf_function': 'linear', 'smooth': 0.1}
        elif selected_method == 'idw':
            method_params = {'power': 2, 'neighbors': min(10, len(x))}
        elif selected_method == 'svr':
            method_params = {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}

        try:
            print(f"\nВыполняем интерполяцию методом: {selected_method}...")
            xi, yi, zi_extrapolated, xi_grid, yi_grid = create_extrapolated_surface(
                x_predict, y_predict, z_predict,
                grid_points=150,
                expansion=0.3,
                method=selected_method,
                **method_params
            )
            print(f"Интерполяция успешно выполнена!")

        except Exception as e:
            print(f"\nОшибка при интерполяции методом {selected_method}: {e}")
            print("Использую линейную интерполяцию как запасной вариант...")
            selected_method = 'linear'
            xi, yi, zi_extrapolated, xi_grid, yi_grid = create_extrapolated_surface(
                x_predict, y_predict, z_predict,
                grid_points=150,
                expansion=0.3,
                method=selected_method
            )

        # Создаем границу исходной области
        boundary_x, boundary_y = create_original_area_boundary(x_predict, y_predict)
        # КОНВЕРТАЦИЯ ВСЕГО в чистые Python типы
        xi_list = xi.tolist()
        yi_list = yi.tolist()
        zi_list = zi_extrapolated.tolist()  # 2D list для contour
        x_predict = df_predict_h['x'].values.tolist()
        y_predict = df_predict_h['y'].values.tolist()
        z_predict = df_predict_h['z'].values.tolist()
        # x_list = df_merged['x'].values.tolist()
        # y_list = df_merged['y'].values.tolist()
        h_kol_list = df_predict_h['h_kol'].values.tolist()
        well_list = df_merged['well'].values.tolist()
        boundary_x_list = list(boundary_x)  # list()
        boundary_y_list = list(boundary_y)

        # ✅ Merge с координатами для правильного соответствия
        merged_comparison = pd.merge(
            df_merged[['well', 'x', 'y', 'h_kol']],
            df_predict_h[['well', 'x', 'y', 'h_kol']],
            on='well',
            how='inner',
            suffixes=('_fact', '_pred')
        )

        merged_comparison['delta'] = merged_comparison['h_kol_pred'] - merged_comparison['h_kol_fact']

        # ✅ Используем координаты из merged_comparison
        x_pred = merged_comparison['x_pred'].tolist()
        y_pred = merged_comparison['y_pred'].tolist()
        h_pred = merged_comparison['h_kol_pred'].tolist()
        h_fact = merged_comparison['h_kol_fact'].tolist()
        delta = merged_comparison['delta'].tolist()
        wells = merged_comparison['well'].tolist()
        customdata = [[pred, fact, d] for pred, fact, d in zip(h_pred, h_fact, delta)]
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_pred,
            y=y_pred,
            mode='markers+text',
            textposition="top center",
            textfont=dict(size=10, color='black'),
            text=wells,
            customdata=customdata,
            marker=dict(
                size=12,
                color=delta,
                colorscale='Reds',  # Красная палитра для предсказаний
                line=dict(width=2, color='darkorange'),
                symbol='diamond'  # Ромбы для отличия
            ),
            hovertemplate=(
                "X: %{x:.1f} | Y: %{y:.1f}<br>Pred: %{customdata[0]:.3f}<br>Fact: %{customdata[1]:.3f}<br><b>Δ: %{customdata[2]:+.3f}</b><extra></extra>"
            ),
            name='Предсказанные точки'
        ))

        # 1. САМЫЙ ПРОСТОЙ CONTOUR
        fig.add_trace(go.Contour(
            x=xi_list,
            y=yi_list,
            z=zi_extrapolated.tolist(),
            colorscale='Viridis',
            hoverinfo='skip',
            opacity=0.8,
            name='Интерполяция'
        ))

        # 2. Граница
        fig.add_trace(go.Scatter(
            x=boundary_x_list,
            y=boundary_y_list,
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Граница',
            hoverinfo='skip'

        ))

        # # 3. Скважины БЕЗ text
        # fig.add_trace(go.Scatter(
        #     x=x_predict,
        #     y=y_predict,
        #     mode='markers',
        #     marker=dict(
        #         size=12,
        #         color=h_kol_list,
        #         colorscale='Viridis',
        #         line=dict(width=2, color='white')
        #     ),
        #     name='Скважины'
        # ))

        # МИНИМАЛЬНЫЙ layout
        fig.update_layout(
            title=f'h_kol - {selected_method}',
            xaxis_title='X',
            yaxis_title='Y',
            height=700,
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)
