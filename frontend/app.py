import zipfile
from io import BytesIO
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from interpolation import read_navigator_file, create_extrapolated_surface, create_original_area_boundary
import requests
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
    
    # Загрузка данных для интерполяции
    st.subheader("Данные для интерполяции")
    uploaded_h = st.file_uploader("Фактические данные EFF_H, H (ZIP)", type="zip", key="actual_data")
    uploaded_h_predict = st.file_uploader("Предсказанные данные (CSV)", type=["csv"], key="predicted_data")
    
    # Настройки отображения
    st.divider()
    st.subheader("⚙️ Настройки отображения")
    
    show_contour = st.checkbox("Показать контурную карту", value=True)
    show_boundary = st.checkbox("Показать границу области", value=True)
    show_wells = st.checkbox("Показать скважины", value=True)
    
    opacity = st.slider("Прозрачность карты", 0.1, 1.0, 0.8)
    colorscale = st.selectbox("Цветовая схема", 
                             ['RdYlBu', 'RdBu', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',])

# Основная область
st.title('🎯 MVP Хакатона - Визуализация данных')

# Читаем предсказанный файл и конвертируем в датафрейм
if uploaded_h is not None and uploaded_h_predict is not None:
    with st.spinner("🔄 Обработка данных..."):
        # Создаем два столбца для отображения статуса
        col1, col2 = st.columns(2)
        df_merged = pd.DataFrame()
        with col1:
            st.info("📊 Загрузка архивных данных...")

            files = {'zip': uploaded_h}
            response = requests.post("http://localhost:8000/api/v1/upload_tvt_fact/", files=files)

            if response.status_code == 200:
                st.success("Архив обработан!")
                df_merged = pd.DataFrame(response.json()['data'])
                st.dataframe(df_merged)

            else:
                st.error(f"Ошибка: {response.text}")


        with col2:
            st.info("📦 Обработка предсказанных данных...")

            files = {'csv': uploaded_h_predict}
            response = requests.post("http://localhost:8000/api/v1/upload_tvt_pred/", files=files)
            # Проверяем необходимые колонки
            if response.status_code!=200:
                st.error(response.reason)
            else:
                print(response.json())
                df_predict_h = pd.DataFrame(response.json()['data'])
                st.success(f'✅ Загружено {len(df_predict_h)} записей')
                st.write(f"Колонки в файле: {list(df_predict_h.columns)}")



        df_merged.rename(columns={'name':'well'}, inplace=True)
        df_predict_h.rename(columns={'name':'well'}, inplace=True)
        merged_comparison = pd.merge(
            df_merged[['well', 'x', 'y', 'h_kol']],
            df_predict_h[['well', 'x', 'y', 'h_kol']],
            on=['well', 'x', 'y'],
            how='inner',
            suffixes=('_fact', '_pred')
        )
        
        if merged_comparison.empty:
            st.error("❌ Нет общих точек между фактическими и предсказанными данными")
            st.write("Проверьте совпадение координат и названий скважин")
            st.stop()
        
        # Вычисление разницы и процента ошибки
        merged_comparison['delta'] = merged_comparison['h_kol_pred'] - merged_comparison['h_kol_fact']
        merged_comparison['error_percent'] = np.where(
            merged_comparison['h_kol_fact'] != 0,
            (merged_comparison['delta'] / merged_comparison['h_kol_fact']) * 100,
            np.nan
        )
        
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
                grid_points=200,
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
    error_percent = merged_comparison['error_percent'].tolist()
    wells = merged_comparison['well'].tolist()
    
    # Создаем кастомные данные для ховера - ТОЛЬКО нужные данные
    customdata = [[w, pred, fact, d, ep] for w, pred, fact, d, ep in zip(wells, h_pred, h_fact, delta, error_percent)]

    # Создание фигуры
    fig = go.Figure()
    
    # 1. CONTOUR - Упрощенный ховер (только координаты и значение)
    if show_contour:
        fig.add_trace(go.Contour(
            x=xi_list,
            y=yi_list,
            z=zi_list,
            colorscale=colorscale,
            opacity=opacity,
            name='Интерполяция',
            hovertemplate=(
                "<b>Контурная карта</b><br>"
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
                titleside="right",
                titlefont=dict(size=14)
            )
        ))
    
    # 2. Граница области - упрощенный ховер
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
    
    # 3. Точки скважин - ДЕТАЛЬНЫЙ ховер с требуемыми данными
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
                    titleside="right"
                ),
                line=dict(width=2, color='black'),
                symbol='diamond',
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Скважина: %{customdata[0]}</b><br>"
                "Координаты: (%{x:.1f}, %{y:.1f})<br>"
                "Предсказание: %{customdata[1]:.3f}<br>"
                "Факт: %{customdata[2]:.3f}<br>"
                "<b>Разница: %{customdata[3]:+.3f}</b><br>"
                "Ошибка: %{customdata[4]:.1f}%<br>"
                "<extra></extra>"
            ),
            name='Скважины'
        ))
    
    # Настройка layout
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
        hovermode='closest',
        hoverdistance=20,
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
        # Первая строка: гистограмма и корреляция
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Распределение разниц")
            fig_dist = px.histogram(merged_comparison, x='delta', 
                                   title="Распределение Δ (Pred - Fact)",
                                   nbins=30,
                                   labels={'delta': 'Разница (Pred - Fact)'})
            fig_dist.update_layout(showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("Корреляция и точность")
            
            # Метрики точности
            correlation = merged_comparison['h_kol_fact'].corr(merged_comparison['h_kol_pred'])
            mae = merged_comparison['delta'].abs().mean()
            rmse = np.sqrt((merged_comparison['delta']**2).mean())
            mape = merged_comparison['error_percent'].abs().mean()
            
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric("Коэф. корреляции", f"{correlation:.3f}")
                st.metric("MAE", f"{mae:.3f}")
            with col_metrics2:
                st.metric("RMSE", f"{rmse:.3f}")
                st.metric("MAPE", f"{mape:.1f}%")
            
            # Scatter plot
            fig_scatter = px.scatter(merged_comparison, 
                                    x='h_kol_fact', 
                                    y='h_kol_pred',
                                    trendline='ols',
                                    title="Предсказание vs Факт",
                                    labels={'h_kol_fact': 'Фактическое значение', 
                                           'h_kol_pred': 'Предсказанное значение'},
                                    hover_data=['well'])
            
            # Добавляем линию идеального предсказания
            fig_scatter.add_trace(
                go.Scatter(
                    x=[merged_comparison['h_kol_fact'].min(), merged_comparison['h_kol_fact'].max()],
                    y=[merged_comparison['h_kol_fact'].min(), merged_comparison['h_kol_fact'].max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Идеальное предсказание'
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Вторая строка: таблица данных (ПОД графиками)
        st.divider()
        st.subheader("📋 Таблица данных")
        
        # Добавляем фильтры для таблицы
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            sort_by = st.selectbox("Сортировать по:", 
                                  ['delta', 'error_percent', 'h_kol_fact', 'h_kol_pred'],
                                  index=0)
        with col_filter2:
            sort_order = st.radio("Порядок:", ['По убыванию', 'По возрастанию'], 
                                 horizontal=True)
        
        # Подготовка данных для отображения
        display_df = merged_comparison[['well', 'x', 'y', 'h_kol_fact', 'h_kol_pred', 'delta', 'error_percent']].copy()
        display_df['error_percent'] = display_df['error_percent'].round(1)
        
        # Сортировка
        ascending = (sort_order == 'По возрастанию')
        display_df = display_df.sort_values(sort_by, ascending=ascending)
        
        # Форматирование колонок для отображения
        display_df_display = display_df.copy()
        display_df_display['x'] = display_df_display['x'].round(1)
        display_df_display['y'] = display_df_display['y'].round(1)
        display_df_display['h_kol_fact'] = display_df_display['h_kol_fact'].round(3)
        display_df_display['h_kol_pred'] = display_df_display['h_kol_pred'].round(3)
        display_df_display['delta'] = display_df_display['delta'].round(3)
        
        # Отображение таблицы
        st.dataframe(
            display_df_display,
            column_config={
                "well": "Скважина",
                "x": "X",
                "y": "Y",
                "h_kol_fact": "Факт",
                "h_kol_pred": "Предсказание",
                "delta": "Разница",
                "error_percent": st.column_config.NumberColumn(
                    "Ошибка (%)",
                    format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Кнопка для скачивания данных
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать данные как CSV",
            data=csv,
            file_name=f"comparison_results_{selected_method}.csv",
            mime="text/csv",
        )
    
else:
    # Инструкции при отсутствии данных
    st.info("👈 Пожалуйста, загрузите данные через боковую панель слева")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📁 Загрузите:
        1. **ZIP архив** с фактическими данными
           - Формат: ZIP с файлами .las или .txt
           - Должны содержать EFF_H и H данные
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Загрузите:
        2. **CSV файл** с предсказанными данными
           - Обязательные колонки: `x`, `y`, `well`, `h_kol`
           - Координаты должны совпадать с фактическими данными
        """)
    
    with col3:
        st.markdown("""
        ### ⚙️ Настройте:
        1. Метод интерполяции
        2. Параметры отображения
        3. Цветовую схему
        """)
    
    # Пример структуры CSV файла
    st.divider()
    st.subheader("📋 Пример структуры CSV файла")
    
    example_data = pd.DataFrame({
        'well': ['WELL_001', 'WELL_002', 'WELL_003', 'WELL_004'],
        'x': [100.1234, 120.5678, 140.9012, 160.3456],
        'y': [200.7890, 220.1234, 240.5678, 260.9012],
        'h_kol': [0.85, 0.92, 0.78, 0.88]
    })
    
    st.write("**Требуемые колонки в CSV файле:**")
    st.dataframe(example_data, hide_index=True)
    
    st.write("**Формат ZIP архива с фактическими данными:**")
    st.code("""
    archive.zip
    ├── actual_data_eff_h.las  # или .txt
    └── actual_data_h.las      # или .txt
    """, language="text")