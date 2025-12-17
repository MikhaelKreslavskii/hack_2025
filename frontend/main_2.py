import zipfile
from io import BytesIO
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from interpolation import read_navigator_file, create_extrapolated_surface, create_original_area_boundary

# Конфигурация страницы
st.set_page_config(layout="wide", page_title="MVP Хакатона")

method_descriptions = {
    'linear': 'Линейная интерполяция (scipy)',
    'cubic': 'Кубическая интерполяция (scipy)',
    'nearest': 'Ближайший сосед (scipy)',
    'rbf': 'Радиальные базисные функции',
    'idw': 'Обратное расстояние (Inverse Distance Weighting)',
    'kriging': 'Кригинг (Гауссовские процессы)',
    'svr': 'Support Vector Regression'
}

# Создаем боковую панель
with st.sidebar:
    st.title("📁 Загрузка данных")
    
    # Загрузка ZIP-архива через API
    st.subheader("Загрузка через API")
    uploaded_zip = st.file_uploader("Загрузите папку как ZIP", type="zip", key="zip_api")
    if uploaded_zip is not None:
        files = {'zip': uploaded_zip}
        with st.spinner("Отправка на сервер..."):
            response = requests.post("http://localhost:8000/api/v1/upload_files/", files=files)
            
            if response.status_code == 200:
                st.success("✅ Архив обработан!")
                # st.json(response.json())
            else:
                st.error(f"❌ Ошибка: {response.text}")
    
    st.divider()
    
    # Загрузка данных для интерполяции
    st.subheader("Данные для интерполяции")
    uploaded_h = st.file_uploader("Фактические данные EFF_H, H (ZIP)", type="zip", key="actual_data")
    uploaded_h_predict = st.file_uploader("Предсказанные данные EFF_H, H", type=["las", "txt", "csv"], key="predicted_data")
    
    # Настройки отображения
    st.divider()
    st.subheader("⚙️ Настройки отображения")
    
    show_contour = st.checkbox("Показать контурную карту", value=True)
    show_boundary = st.checkbox("Показать границу области", value=True)
    show_wells = st.checkbox("Показать скважины", value=True)
    
    opacity = st.slider("Прозрачность карты", 0.1, 1.0, 0.8)
    colorscale = st.selectbox("Цветовая схема", 
                             ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'RdBu', 'RdYlBu'])

# Основная область
st.title('🎯 MVP Хакатона - Визуализация данных')

# Читаем предсказанный файл и конвертируем в датафрейм
if uploaded_h is not None and uploaded_h_predict is not None:
    with st.spinner("🔄 Обработка данных..."):
        # Создаем два столбца для отображения статуса
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📊 Загрузка предсказанных данных...")
            df_predict_h = pd.DataFrame()
            file_content = BytesIO(uploaded_h_predict.read())
            df_predict_h = read_navigator_file(file_content)
            df_predict_h = df_predict_h.rename(columns={'value': 'h_kol'})
            st.success(f'✅ Загружено {len(df_predict_h)} записей')
            
        with col2:
            st.info("📦 Обработка архивных данных...")
            df_eff_h = pd.DataFrame()
            df_h = pd.DataFrame()
            zip_content = BytesIO(uploaded_h.read())

            with zipfile.ZipFile(zip_content) as zip_ref:
                file_names = zip_ref.namelist()
                
                for name in file_names:
                    try:
                        content_bytes = zip_ref.read(name)
                        file_like = BytesIO(content_bytes)
                        
                        if 'FF' in name.upper():
                            df_eff_h = read_navigator_file(file_like)
                        else:
                            df_h = read_navigator_file(file_like)
                            
                    except Exception as e:
                        st.error(f"Ошибка файла {name}: {e}")
                        continue

        # Обработка DataFrame
        if not df_eff_h.empty and not df_h.empty:
            df_eff_h = df_eff_h.rename(columns={'value': 'eff_h'})
            df_h = df_h.rename(columns={'value': 'h'})
        
        df_eff_h.drop(columns=['z'], inplace=True)
        df_h.drop(columns=['z'], inplace=True)

        # Подготовка данных
        for df in [df_eff_h, df_h, df_predict_h]:
            df[['x', 'y']] = df[['x', 'y']].astype(float).round(4)
        
        # Объединение данных
        df_merged = pd.merge(df_eff_h, df_h, on=['x', 'y', 'well'], how='inner')
        df_merged['h_kol'] = df_merged['eff_h'] / df_merged['h']

        merged_comparison = pd.merge(
            df_merged[['well', 'x', 'y', 'h_kol']],
            df_predict_h[['well', 'x', 'y', 'h_kol']],
            on=['well', 'x', 'y'],
            how='inner',
            suffixes=('_fact', '_pred')
        )
        
        merged_comparison['delta'] = merged_comparison['h_kol_pred'] - merged_comparison['h_kol_fact']
        
        # Статистика
        st.success(f"✅ Обработка завершена!")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Всего скважин", df_merged['well'].nunique())
        with col_stats2:
            st.metric("Объединенных точек", len(merged_comparison))
        with col_stats3:
            avg_delta = merged_comparison['delta'].mean()
            st.metric("Средняя разница", f"{avg_delta:.3f}")

    # Выбор метода интерполяции
    st.divider()
    st.header("📐 Метод интерполяции")
    
    methods = ['rbf', 'idw', 'linear', 'cubic', 'kriging', 'svr']
    
    # Создаем две колонки для выбора метода и информации
    col_method, col_info = st.columns([1, 2])
    
    with col_method:
        selected_method = st.selectbox(
            "Выберите метод:",
            methods,
            index=0,
            help="Выберите метод интерполяции для построения поверхности"
        )
        
        # Параметры для выбранного метода
        method_params = {}
        if selected_method == 'rbf':
            rbf_func = st.selectbox("Функция RBF", ['linear', 'cubic', 'gaussian', 'multiquadric'])
            smooth_val = st.slider("Сглаживание", 0.0, 1.0, 0.1)
            method_params = {'rbf_function': rbf_func, 'smooth': smooth_val}
        elif selected_method == 'idw':
            power_val = st.slider("Степень", 1, 5, 2)
            neighbors_val = st.slider("Соседей", 1, 20, min(10, len(df_merged)))
            method_params = {'power': power_val, 'neighbors': neighbors_val}
        elif selected_method == 'svr':
            kernel_val = st.selectbox("Ядро", ['rbf', 'linear', 'poly'])
            C_val = st.slider("C", 1, 1000, 100)
            method_params = {'kernel': kernel_val, 'C': C_val}
    
    with col_info:
        st.info(f"**{method_descriptions.get(selected_method, selected_method)}**")
        if selected_method == 'rbf':
            st.caption("Рекомендуется для большинства задач. Использует радиальные базисные функции.")
        elif selected_method == 'idw':
            st.caption("Метод обратного расстояния - простой и быстрый метод.")
        elif selected_method == 'kriging':
            st.caption("Статистический метод, учитывающий пространственную корреляцию.")

    # Интерполяция
    x_predict = df_merged['x'].values
    y_predict = df_merged['y'].values
    z_values = df_merged['h_kol'].values

    try:
        with st.spinner(f"Выполняем интерполяцию методом {selected_method}..."):
            xi, yi, zi_extrapolated, xi_grid, yi_grid = create_extrapolated_surface(
                x_predict, y_predict, z_values,
                grid_points=200,  # Увеличиваем разрешение для лучшего ховера
                expansion=0.3,
                method=selected_method,
                **method_params
            )
    except Exception as e:
        st.warning(f"Ошибка при интерполяции методом {selected_method}: {e}")
        st.info("Использую линейную интерполяцию как запасной вариант...")
        selected_method = 'linear'
        xi, yi, zi_extrapolated, xi_grid, yi_grid = create_extrapolated_surface(
            x_predict, y_predict, z_values,
            grid_points=200,
            expansion=0.3,
            method=selected_method
        )

    # Создаем границу исходной области
    boundary_x, boundary_y = create_original_area_boundary(x_predict, y_predict)
    
    # Подготовка данных для визуализации
    xi_list = xi.tolist()
    yi_list = yi.tolist()
    zi_list = zi_extrapolated.tolist()
    
    # Данные для точек
    x_pred = merged_comparison['x'].tolist()
    y_pred = merged_comparison['y'].tolist()
    h_pred = merged_comparison['h_kol_pred'].tolist()
    h_fact = merged_comparison['h_kol_fact'].tolist()
    delta = merged_comparison['delta'].tolist()
    wells = merged_comparison['well'].tolist()
    
    # Создаем кастомные данные для ховера
    customdata = [[pred, fact, d, w] for pred, fact, d, w in zip(h_pred, h_fact, delta, wells)]

    # Создание фигуры
    fig = go.Figure()
    
    # 1. CONTOUR - Улучшенный ховер
    if show_contour:
        # Создаем meshgrid для более точного ховера
        X, Y = np.meshgrid(xi_list, yi_list)
        Z = np.array(zi_list)
        
        fig.add_trace(go.Contour(
            x=xi_list,
            y=yi_list,
            z=zi_list,
            colorscale=colorscale,
            opacity=opacity,
            name='Интерполяция',
            hovertemplate=(
                "<b>Карта значений</b><br>"
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}<br>"
                "Значение: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            ),
            colorbar=dict(
                title="h_kol",

                titlefont=dict(size=14)
            )
        ))
    
    # 2. Граница области
    if show_boundary:
        fig.add_trace(go.Scatter(
            x=list(boundary_x),
            y=list(boundary_y),
            mode='lines',
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='red', width=2, dash='dash'),
            name='Граница области',
            hovertemplate=(
                "<b>Граница области</b><br>"
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}<br>"
                "<extra></extra>"
            )
        ))
    
    # 3. Точки скважин с улучшенным ховером
    if show_wells:
        fig.add_trace(go.Scatter(
            x=x_pred,
            y=y_pred,
            mode='markers+text',
            text=wells,
            textposition="top center",
            textfont=dict(size=10, color='black'),
            customdata=customdata,
            marker=dict(
                size=14,
                color=delta,
                colorscale='RdBu',
                colorbar=dict(
                    title="Δ (Pred - Fact)",
                    x=1.05,

                ),
                line=dict(width=2, color='black'),
                symbol='diamond',
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Скважина: %{customdata[3]}</b><br>"
                "Координаты: (%{x:.1f}, %{y:.1f})<br>"
                "Предсказание: %{customdata[0]:.3f}<br>"
                "Факт: %{customdata[1]:.3f}<br>"
                "<b>Разница: %{customdata[2]:+.3f}</b><br>"
                "<extra></extra>"
            ),
            name='Скважины'
        ))
    
    # Настройка layout с улучшенным ховером
    fig.update_layout(
        title=dict(
            text=f'Визуализация h_kol - {method_descriptions.get(selected_method, selected_method)}',
            font=dict(size=24),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Координата X',
        yaxis_title='Координата Y',
        height=800,
        hovermode='x unified',  # Улучшенный режим ховера
        hoverdistance=20,  # Увеличиваем дистанцию срабатывания
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    # Настройка осей
    fig.update_xaxes(
        gridcolor='lightgray',
        griddash='dash',
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='lightgray',
        griddash='dash',
        showgrid=True
    )
    
    # Отображение графика
    st.plotly_chart(fig, use_container_width=True)
    
    # Дополнительная информация
    with st.expander("📊 Подробная статистика"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Распределение разниц")
            fig_dist = px.histogram(merged_comparison, x='delta', 
                                   title="Распределение Δ (Pred - Fact)",
                                   nbins=30)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("Корреляция")
            correlation = merged_comparison['h_kol_fact'].corr(merged_comparison['h_kol_pred'])
            st.metric("Коэффициент корреляции", f"{correlation:.3f}")
            fig_scatter = px.scatter(merged_comparison, 
                                    x='h_kol_fact', 
                                    y='h_kol_pred',
                                    trendline='ols',
                                    title="Pred vs Fact")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col3:
            st.subheader("Данные")
            st.dataframe(
                merged_comparison[['well', 'x', 'y', 'h_kol_fact', 'h_kol_pred', 'delta']]
                .sort_values('delta', ascending=False)
                .head(10)
            )
    
else:
    # Инструкции при отсутствии данных
    st.info("👈 Пожалуйста, загрузите данные через боковую панель слева")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📁 Загрузите:
        1. ZIP архив с фактическими данными
        2. Файл с предсказанными данными
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ Настройте:
        1. Метод интерполяции
        2. Параметры отображения
        3. Цветовую схему
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Получите:
        1. Интерактивную карту
        2. Подробную статистику
        3. Анализ точности
        """)
    
    # Пример графика
    st.divider()
    st.subheader("Пример визуализации")
    
    # Создаем примерный график для демонстрации
    fig_demo = go.Figure()
    
    # Примерные данные
    x_demo = np.random.uniform(0, 100, 50)
    y_demo = np.random.uniform(0, 100, 50)
    z_demo = np.sin(x_demo/10) * np.cos(y_demo/10)
    
    fig_demo.add_trace(go.Contour(
        x=np.linspace(0, 100, 100),
        y=np.linspace(0, 100, 100),
        z=[[np.sin(i/10) * np.cos(j/10) for i in np.linspace(0, 100, 100)] 
           for j in np.linspace(0, 100, 100)],
        colorscale='Viridis',
        opacity=0.7,
        hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Значение: %{z:.3f}<extra></extra>"
    ))
    
    fig_demo.add_trace(go.Scatter(
        x=x_demo,
        y=y_demo,
        mode='markers',
        marker=dict(
            size=10,
            color=z_demo,
            colorscale='RdBu',
            line=dict(width=2, color='black')
        ),
        hovertemplate="Скважина<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Значение: %{marker.color:.3f}<extra></extra>"
    ))
    
    fig_demo.update_layout(
        title="Пример интерактивной карты",
        height=500
    )
    
    st.plotly_chart(fig_demo, use_container_width=True)