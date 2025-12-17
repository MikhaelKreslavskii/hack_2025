import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import requests
import zipfile
from io import BytesIO

# Настройка страницы
st.set_page_config(layout="wide", page_title="Геолого-геофизический анализ скважин")

# Боковое меню для навигации
st.sidebar.title("📊 Навигация")

# Выбор приложения
app_mode = st.sidebar.radio(
    "Выберите режим:",
    ["📈 Диаграмма скважин", "🗺️ 3D Визуализация h_kol"],
    index=0
)

# Описания методов для второго приложения
method_descriptions = {
    'linear': 'Линейная интерполяция (scipy)',
    'cubic': 'Кубическая интерполяция (scipy)',
    'nearest': 'Ближайший сосед (scipy)',
    'rbf': 'Радиальные базисные функции',
    'idw': 'Обратное расстояние (Inverse Distance Weighting)',
    'kriging': 'Кригинг (Гауссовские процессы)',
    'svr': 'Support Vector Regression'
}


# Функция для создания интерактивной гистограммы с индивидуальными шкалами глубин для каждой скважины
def create_individual_scale_histogram(filtered_df, selected_wells, stats_df):
    """Создает гистограмму с индивидуальными шкалами глубин для каждой скважины"""

    # Настройки
    VISIBLE_WELLS = 10  # Количество скважин в видимой области
    bar_width = 0.35  # Ширина каждого столбца
    gap_between_wells = 0.6  # Промежуток между скважинами

    # ЕДИНАЯ цветовая палитра
    COLOR_COLLECTOR = '#FFD700'  # Желтый - коллектор
    COLOR_NONCOLLECTOR = '#CCCCCC'  # Серый - неколлектор

    # Создаем фигуру с подграфиками для каждой скважины
    fig = go.Figure()

    # Собираем данные о диапазонах глубин для каждой скважины
    well_ranges = {}
    for well in selected_wells:
        well_data = filtered_df[filtered_df['name'] == well]
        if len(well_data) > 0:
            min_depth = well_data['depth'].min()
            max_depth = well_data['depth'].max()
            margin = (max_depth - min_depth) * 0.05 if max_depth > min_depth else 5
            well_ranges[well] = {
                'min': min_depth - margin,
                'max': max_depth + margin,
                'range': max_depth - min_depth
            }

    # Нормализуем все глубины к общему диапазону для визуального выравнивания
    normalized_ranges = {}
    if well_ranges:
        max_range = max(r['range'] for r in well_ranges.values())
        for well, range_info in well_ranges.items():
            normalized_ranges[well] = {
                'min': 0,
                'max': range_info['range'] * (100 / max_range) if max_range > 0 else 100,
                'original_min': range_info['min'],
                'original_max': range_info['max'],
                'scale_factor': 100 / max_range if max_range > 0 else 1
            }

    # Для каждой скважины
    for well_idx, well in enumerate(selected_wells):
        well_data = filtered_df[filtered_df['name'] == well].sort_values('depth')

        if len(well_data) > 0:
            # Позиция скважины на оси X
            x_center = well_idx * (1 + gap_between_wells)

            # Получаем информацию о диапазоне для этой скважины
            well_range = normalized_ranges.get(well, {'min': 0, 'max': 100, 'original_min': 0, 'original_max': 100,
                                                      'scale_factor': 1})

            # Нормализуем глубины для этой скважины
            min_depth_original = well_data['depth'].min()
            scale_factor = well_range['scale_factor']

            # Создаем массивы для каждого типа данных
            fact_heights = []
            fact_bottoms = []
            fact_colors = []
            fact_hovertexts = []

            pred_heights = []
            pred_bottoms = []
            pred_colors = []
            pred_hovertexts = []

            # Обрабатываем каждую точку данных
            for i in range(len(well_data)):
                row = well_data.iloc[i]
                depth_original = row['depth']

                # Нормализуем глубину
                depth_normalized = (depth_original - min_depth_original) * scale_factor

                # Определяем высоту сегмента в нормализованных единицах
                if i < len(well_data) - 1:
                    next_depth_original = well_data.iloc[i + 1]['depth']
                    height_original = next_depth_original - depth_original
                    height_normalized = height_original * scale_factor
                else:
                    if len(well_data) > 1:
                        avg_step_original = (well_data.iloc[-1]['depth'] - well_data.iloc[0]['depth']) / (
                                    len(well_data) - 1)
                        height_original = avg_step_original
                        height_normalized = avg_step_original * scale_factor
                    else:
                        height_original = 1.0
                        height_normalized = 1.0 * scale_factor

                # Фактические данные
                fact_color = COLOR_COLLECTOR if row['value'] == 1 else COLOR_NONCOLLECTOR
                fact_heights.append(height_normalized)
                fact_bottoms.append(depth_normalized)
                fact_colors.append(fact_color)
                fact_hovertexts.append(
                    f"<b>{well}</b> (факт)<br>"
                    f"Глубина: {depth_original:.1f} м<br>"
                    f"Интервал: ~{height_original:.1f} м<br>"
                    f"Значение: {row['value']} {'(коллектор)' if row['value'] == 1 else '(неколлектор)'}"
                )

                # Предсказанные данные
                pred_color = COLOR_COLLECTOR if row['value_predict'] == 1 else COLOR_NONCOLLECTOR
                pred_heights.append(height_normalized)
                pred_bottoms.append(depth_normalized)
                pred_colors.append(pred_color)
                pred_hovertexts.append(
                    f"<b>{well}</b> (предсказание)<br>"
                    f"Глубина: {depth_original:.1f} м<br>"
                    f"Интервал: ~{height_original:.1f} м<br>"
                    f"Значение: {row['value_predict']} {'(коллектор)' if row['value_predict'] == 1 else '(неколлектор)'}<br>"
                    f"{'✓ Совпадение' if row['value'] == row['value_predict'] else '✗ Расхождение'}"
                )

            # Добавляем фактические данные (левый столбец)
            fig.add_trace(go.Bar(
                x=[x_center - bar_width / 2] * len(fact_heights),
                y=fact_heights,
                base=fact_bottoms,
                width=bar_width,
                marker_color=fact_colors,
                name=f"{well}",
                legendgroup=well,
                showlegend=False,
                hovertext=fact_hovertexts,
                hoverinfo="text",
                orientation='v'
            ))

            # Добавляем предсказанные данные (правый столбец)
            fig.add_trace(go.Bar(
                x=[x_center + bar_width / 2] * len(pred_heights),
                y=pred_heights,
                base=pred_bottoms,
                width=bar_width,
                marker_color=pred_colors,
                name=f"{well}",
                legendgroup=well,
                showlegend=False,
                hovertext=pred_hovertexts,
                hoverinfo="text",
                orientation='v'
            ))

            # Добавляем красную границу для расхождений
            for i in range(len(well_data)):
                row = well_data.iloc[i]
                if row['value'] != row['value_predict']:
                    x_pos = x_center + bar_width / 2
                    y_bottom = fact_bottoms[i]
                    height = fact_heights[i]

                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos],
                        y=[y_bottom, y_bottom + height],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False,
                        name='',
                        hoverinfo='skip'
                    ))

    # Настройка осей
    fig.update_xaxes(
        title_text="Скважины",
        tickvals=[i * (1 + gap_between_wells) for i in range(len(selected_wells))],
        ticktext=selected_wells,
        tickangle=45,
        showgrid=False,
        zeroline=False,
        range=[-0.5, min(VISIBLE_WELLS, len(selected_wells)) * (1 + gap_between_wells) - 0.5]
    )

    fig.update_yaxes(
        title_text="Нормализованная глубина",
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=False,
        tickmode='array',
        tickvals=[0, 25, 50, 75, 100],
        ticktext=['Мин', '', 'Средн', '', 'Макс']
    )

    # Настройка макета
    fig.update_layout(
        height=700,
        title_text=f"Гистограмма скважин ({len(selected_wells)} шт.) | Индивидуальные шкалы глубин",
        barmode='group',
        bargap=0,
        bargroupgap=0,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='LightGray'
            ),
            type="linear",
            range=[-0.5, min(VISIBLE_WELLS, len(selected_wells)) * (1 + gap_between_wells) - 0.5],
            fixedrange=False
        ),
        dragmode='zoom',
        width=1200
    )

    # Добавляем процент ошибок под скважинами
    for well_idx, well in enumerate(selected_wells):
        error_row = stats_df[stats_df['Скважина'] == well]
        if not error_row.empty:
            error_pct = float(error_row['Ошибка (%)'].values[0].replace('%', ''))
            color = 'red' if error_pct > 20 else 'orange' if error_pct > 10 else 'green'
            x_pos = well_idx * (1 + gap_between_wells)

            fig.add_annotation(
                x=x_pos,
                y=-5,
                text=f"{error_pct:.1f}%",
                showarrow=False,
                font=dict(color=color, size=10, weight='bold'),
                yref="y"
            )

    # Легенда
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=COLOR_COLLECTOR,
        name='Коллектор',
        showlegend=True,
        width=0
    ))

    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=COLOR_NONCOLLECTOR,
        name='Неколлектор',
        showlegend=True,
        width=0
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Расхождение',
        showlegend=True
    ))

    fig.add_annotation(
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        text="ℹ️ Каждая скважина имеет свою шкалу глубин (нормализована для сравнения)",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )

    return fig


# Основной код приложения
if app_mode == "📈 Диаграмма скважин":
    # ПЕРВОЕ ПРИЛОЖЕНИЕ: Диаграмма скважин

    st.title("📊 Диаграмма скважин: Фактические vs Предсказанные значения")
    st.markdown("Единая цветовая палитра: желтый - коллектор, серый - неколлектор")

    # Сайдбар для загрузки файлов
    st.sidebar.header("📂 Загрузка данных")

    # Загрузка фактических данных через ZIP архив и API
    uploaded_zip = st.sidebar.file_uploader(
        "Загрузите razrez.zip (фактические данные через API)",
        type=['zip'],
        help="ZIP архив с фактическими данными для обработки через API"
    )

    # Загрузка предсказанных данных через CSV файл
    uploaded_csv = st.sidebar.file_uploader(
        "Загрузите razrez_predict.csv (предсказанные данные)",
        type=['csv'],
        help="CSV файл с предсказанными данными"
    )

    # Переменные для хранения данных
    actual_df = None
    predict_df = None

    # Обработка загруженного ZIP архива с фактическими данными
    if uploaded_zip is not None:
        try:
            with st.spinner("🔄 Отправка фактических данных на сервер..."):
                files = {'zip': uploaded_zip}
                response = requests.post("http://localhost:8000/api/v1/upload_well/", files=files)

                if response.status_code == 200:
                    st.sidebar.success("✅ Фактические данные успешно обработаны!")

                    # Получаем данные из ответа
                    response_data = response.json()

                    # Проверяем структуру ответа
                    if 'data' in response_data:
                        # Предполагаем, что API возвращает список словарей
                        data_list = response_data['data']

                        if data_list and len(data_list) > 0:
                            # Создаем DataFrame из полученных данных
                            actual_df = pd.DataFrame(data_list)

                            # Проверяем наличие необходимых колонок
                            required_columns = ['name', 'depth', 'value']
                            missing_columns = [col for col in required_columns if col not in actual_df.columns]

                            if missing_columns:
                                st.error(f"❌ В фактических данных отсутствуют необходимые колонки: {missing_columns}")
                                st.info("Ожидаемые колонки в фактических данных: id, name, depth, value")
                                actual_df = None
                            else:
                                # Оставляем только нужные колонки
                                actual_df = actual_df[required_columns].copy()
                                st.sidebar.success(f"✅ Загружено {len(actual_df)} фактических записей")
                        else:
                            st.error("❌ Получены пустые данные от API")
                    else:
                        st.error(f"❌ Неожиданный формат ответа от API")
                        st.json(response_data)  # Показываем ответ для отладки
                else:
                    st.error(f"❌ Ошибка сервера: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Не удалось подключиться к серверу. Проверьте, запущен ли сервер на localhost:8000")
        except Exception as e:
            st.error(f"❌ Ошибка при обработке фактических данных: {str(e)}")

    # Обработка загруженного CSV файла с предсказанными данными
    if uploaded_csv is not None:
        try:
            with st.spinner("🔄 Загрузка предсказанных данных..."):
                # Читаем CSV файл напрямую
                predict_df = pd.read_csv(uploaded_csv)

                # Проверяем наличие необходимых колонок
                required_columns = ['name', 'depth', 'value_predict']
                missing_columns = [col for col in required_columns if col not in predict_df.columns]

                if missing_columns:
                    st.error(f"❌ В предсказанных данных отсутствуют необходимые колонки: {missing_columns}")
                    st.info("Ожидаемые колонки в предсказанных данных: id, name, depth, value_predict")
                    predict_df = None
                else:
                    # Оставляем только нужные колонки
                    predict_df = predict_df[required_columns].copy()
                    st.sidebar.success(f"✅ Загружено {len(predict_df)} предсказанных записей")

        except Exception as e:
            st.error(f"❌ Ошибка при чтении CSV файла: {str(e)}")

    # Основной контент приложения
    if actual_df is not None and predict_df is not None:
        try:
            # Объединение данных
            df = pd.merge(actual_df, predict_df, on=['name', 'depth'], how='inner')

            if len(df) == 0:
                st.error("❌ Нет совпадающих записей между фактическими и предсказанными данными")
                st.info("Проверьте совпадение id, name и depth в обоих наборах данных")
            else:
                df = df.sort_values(['name', 'depth'])

                # Показать информацию о данных
                with st.expander("📋 Информация о данных"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Фактические данные:**")
                        st.write(f"- Записей: {len(actual_df):,}")
                        st.write(f"- Уникальных скважин: {actual_df['name'].nunique()}")
                        st.write(
                            f"- Диапазон глубин: {actual_df['depth'].min():.1f} - {actual_df['depth'].max():.1f} м")

                    with col2:
                        st.write("**Предсказанные данные:**")
                        st.write(f"- Записей: {len(predict_df):,}")
                        st.write(f"- Уникальных скважин: {predict_df['name'].nunique()}")
                        st.write(
                            f"- Диапазон глубин: {predict_df['depth'].min():.1f} - {predict_df['depth'].max():.1f} м")

                    st.write("**Объединенные данные:**")
                    st.write(f"- Совпадающих записей: {len(df):,}")
                    st.write(f"- Уникальных скважин: {df['name'].nunique()}")

                # Уникальные скважины
                wells = df['name'].unique().tolist()

                # Настройки визуализации
                st.sidebar.header("⚙️ Настройки отображения")

                # Выбор скважин - ВСЕ по умолчанию
                selected_wells = st.sidebar.multiselect(
                    "Выберите скважины:",
                    options=wells,
                    default=wells[:min(10, len(wells))],  # Первые 10 по умолчанию
                    help="Все скважины можно выбрать через Ctrl+A"
                )

                # Если ничего не выбрано, показываем первые 10
                if not selected_wells:
                    selected_wells = wells[:min(10, len(wells))]

                # Информация о диапазоне глубин
                st.sidebar.header("📊 Информация о глубинах")

                # Собираем статистику по выбранным скважинам
                depth_stats = []
                for well in selected_wells:
                    well_data = df[df['name'] == well]
                    if len(well_data) > 0:
                        depth_stats.append({
                            'Скважина': well,
                            'Мин. глубина': f"{well_data['depth'].min():.1f} м",
                            'Макс. глубина': f"{well_data['depth'].max():.1f} м",
                            'Диапазон': f"{well_data['depth'].max() - well_data['depth'].min():.1f} м",
                            'Точек': len(well_data)
                        })

                if depth_stats:
                    depth_stats_df = pd.DataFrame(depth_stats)

                    with st.sidebar.expander("📈 Диапазоны глубин по скважинам"):
                        st.dataframe(depth_stats_df.set_index('Скважина'), height=300)

                # Информация о выбранных скважинах
                st.sidebar.header("📊 Информация")
                st.sidebar.write(f"Выбрано скважин: {len(selected_wells)}")

                if len(selected_wells) > 0:
                    # Фильтрация данных
                    filtered_df = df[df['name'].isin(selected_wells)].copy()

                    # Сортировка
                    filtered_df['name'] = pd.Categorical(filtered_df['name'],
                                                         categories=selected_wells,
                                                         ordered=True)
                    filtered_df = filtered_df.sort_values(['name', 'depth'])

                    st.sidebar.write(f"Точек данных: {len(filtered_df):,}")

                    # Статистика ошибок
                    stats_data = []
                    for well in selected_wells:
                        well_data = filtered_df[filtered_df['name'] == well]
                        if len(well_data) > 0:
                            correct = (well_data['value'] == well_data['value_predict']).sum()
                            total = len(well_data)
                            accuracy = correct / total * 100 if total > 0 else 0
                            error = 100 - accuracy

                            stats_data.append({
                                'Скважина': well,
                                'Всего точек': total,
                                'Правильно': correct,
                                'Ошибок': total - correct,
                                'Точность (%)': f"{accuracy:.1f}",
                                'Ошибка (%)': f"{error:.1f}"
                            })

                    stats_df = pd.DataFrame(stats_data)

                    # Основная статистика
                    st.header("📊 Статистика точности")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        total_points = stats_df['Всего точек'].sum()
                        st.metric("Всего точек", f"{total_points:,}")

                    with col2:
                        overall_accuracy = stats_df['Правильно'].sum() / total_points * 100 if total_points > 0 else 0
                        st.metric("Общая точность", f"{overall_accuracy:.1f}%")

                    with col3:
                        avg_error = stats_df['Ошибка (%)'].str.replace('%', '').astype(float).mean()
                        st.metric("Средняя ошибка", f"{avg_error:.1f}%")

                    with col4:
                        if len(stats_df) > 0:
                            worst_idx = stats_df['Ошибка (%)'].str.replace('%', '').astype(float).idxmax()
                            worst = stats_df.loc[worst_idx]
                            worst_error = worst['Ошибка (%)']
                            worst_well = worst['Скважина']
                            st.metric("Макс. ошибка",
                                      f"{worst_error}",
                                      delta=worst_well)

                    # Детальная таблица статистики
                    with st.expander("📋 Подробная статистика по скважинам"):
                        st.dataframe(stats_df.set_index('Скважина'), width='stretch')

                    # Визуализация
                    st.header("📈 Гистограмма скважин с индивидуальными шкалами")

                    # Информация о визуализации
                    st.markdown(f"""
                    **Настройки отображения:**
                    - 📍 **Видимая область:** 10 скважин (остальные через скролл)
                    - 📊 **Всего скважин:** {len(selected_wells)}
                    - 🎨 **Цветовая схема:** 🟡 Желтый = коллектор, ⚪ Серый = неколлектор
                    - 📐 **Шкала глубин:** Каждая скважина имеет свою собственную шкалу (нормализована для сравнения)

                    **ℹ️ Важно:** 
                    - Каждая скважина отображается с оптимальным масштабом для её собственных данных
                    - Все глубины нормализованы, чтобы скважины были сравнимы по высоте
                    - Реальные значения глубин видны при наведении курсора
                    """)

                    # Создаем и отображаем график
                    fig = create_individual_scale_histogram(filtered_df, selected_wells, stats_df)

                    # Отображаем график
                    st.plotly_chart(fig, use_container_width=True, height=750)

                    # Инструкция по использованию
                    st.info("""
                    **🎮 Как использовать график:**

                    ### 🔍 Приближение (Zoom):
                    - **Выделите область мышкой** - зажмите левую кнопку мыши и выделите прямоугольник для приближения
                    - **Колесико мыши** - прокрутите для приближения/отдаления
                    - **Двойной клик** - сбросить масштаб

                    ### 📜 Горизонтальный скролл:
                    - **Ползунок внизу** - для прокрутки всех скважин
                    - **На ползунке:** 
                      * Левая часть - видимая область (10 скважин)
                      * Правая часть = все остальные скважины

                    ### 🖱️ Интерактивность:
                    - **Наведите курсор** на любой столбец для детальной информации (реальные глубины)
                    - **Красные пунктирные границы** показывают расхождения
                    - **Процент ошибки** под каждой скважиной

                    ### 📐 Особенности шкал:
                    - **Ось Y:** Нормализованная глубина (все скважины приведены к одной высоте)
                    - **Реальные значения:** Видны при наведении курсора
                    - **Мин/Макс:** Относительные значения для каждой скважины
                    """)

                    # Анализ расхождений
                    st.header("🔍 Анализ расхождений")

                    discrepancies = filtered_df[filtered_df['value'] != filtered_df['value_predict']].copy()

                    if len(discrepancies) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Типы ошибок:**")

                            error_types = pd.DataFrame({
                                'Тип ошибки': ['Ложные срабатывания (0→1)', 'Пропущенные коллекторы (1→0)'],
                                'Количество': [
                                    len(discrepancies[discrepancies['value'] == 0]),
                                    len(discrepancies[discrepancies['value'] == 1])
                                ],
                                '% от всех точек': [
                                    f"{len(discrepancies[discrepancies['value'] == 0]) / len(filtered_df) * 100:.1f}%",
                                    f"{len(discrepancies[discrepancies['value'] == 1]) / len(filtered_df) * 100:.1f}%"
                                ]
                            })

                            st.dataframe(error_types, width='stretch')

                        with col2:
                            st.write("**Матрица ошибок:**")

                            confusion = pd.crosstab(
                                filtered_df['value'],
                                filtered_df['value_predict'],
                                rownames=['Факт'],
                                colnames=['Предсказание']
                            )

                            # Создаем визуализацию матрицы ошибок
                            fig_conf, ax_conf = plt.subplots(figsize=(6, 5))

                            im = ax_conf.imshow(confusion.values, cmap='Reds', aspect='auto')

                            # Добавляем текст
                            for i in range(confusion.shape[0]):
                                for j in range(confusion.shape[1]):
                                    text = ax_conf.text(j, i, confusion.iloc[i, j],
                                                        ha="center", va="center",
                                                        color="white" if confusion.iloc[
                                                                             i, j] > confusion.values.max() / 2 else "black",
                                                        fontsize=14, fontweight='bold')

                            ax_conf.set_xticks(range(2))
                            ax_conf.set_yticks(range(2))
                            ax_conf.set_xticklabels(['0 (неколлектор)', '1 (коллектор)'])
                            ax_conf.set_yticklabels(['0 (неколлектор)', '1 (коллектор)'])
                            ax_conf.set_xlabel('Предсказание', fontsize=12)
                            ax_conf.set_ylabel('Факт', fontsize=12)
                            ax_conf.set_title('Матрица ошибок', fontsize=14, fontweight='bold')

                            plt.colorbar(im, ax=ax_conf)
                            plt.tight_layout()

                            st.pyplot(fig_conf)
                            plt.close(fig_conf)

                        # Таблица с расхождениями
                        with st.expander("📄 Детализация расхождений (первые 50)"):
                            st.dataframe(
                                discrepancies[['name', 'depth', 'value', 'value_predict']]
                                .sort_values(['name', 'depth'])
                                .head(50),
                                width='stretch',
                                height=300
                            )
                    else:
                        st.success("🎉 Совпадение 100%! Нет расхождений между фактическими и предсказанными данными.")

                else:
                    st.warning("⚠️ Нет данных для выбранных параметров")

        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")
            st.info("Проверьте структуру файлов. Ожидаемые столбцы: id, name, depth, value (value_predict)")

    elif actual_df is not None and predict_df is None:
        st.warning("⚠️ Загружены только фактические данные. Загрузите также файл razrez_predict.csv")
    elif actual_df is None and predict_df is not None:
        st.warning("⚠️ Загружены только предсказанные данные. Загрузите также архив razrez.zip")
    else:
        # Инструкция при первом запуске
        st.info("👈 Загрузите оба файла для начала работы")

        with st.expander("📖 Инструкция по использованию"):
            st.markdown("""
            ### 📋 Формат данных:

            **Файл 1 (razrez.zip):**
            - ZIP архив с фактическими данными
            - Отправляется на API для обработки
            - API должен возвращать данные с полями: `id`, `name`, `depth`, `value`

            **Файл 2 (razrez_predict.csv):**
            - CSV файл с предсказанными данными
            - Обязательные колонки: `id`, `name`, `depth`, `value_predict`

            ### 🎨 Визуализация:

            **Основной график:**
            - По оси X: названия скважин
            - По оси Y: нормализованная глубина (каждая скважина имеет свою шкалу)
            - Для каждой скважины: 2 столбца (факт и предсказание)
            - **Видимо сразу:** 10 скважин
            - **Остальные:** через горизонтальный скролл

            **Цвета:**
            - 🟡 **Желтый:** Коллектор (значение = 1)
            - ⚪ **Серый:** Неколлектор (значение = 0)

            **Особенности:**
            - 📐 **Индивидуальные шкалы:** Каждая скважина отображается с оптимальным масштабом
            - 🔄 **Нормализация:** Все скважины приведены к одинаковой высоте для сравнения
            - 🎯 **Реальные значения:** Видны при наведении курсора

            ### 🎮 Управление:

            **Приближение:**
            1. Зажмите левую кнопку мыши
            2. Выделите прямоугольную область
            3. Отпустите кнопку для приближения

            **Скролл:**
            - Используйте ползунок внизу графика
            - Левая часть ползунка = видимая область
            - Правая часть = все скважины

            **Информация:**
            - Наведите курсор на столбец для реальных значений глубин
            - Процент ошибки под каждой скважиной
            - Красные границы = расхождения
            """)

        # Пример ожидаемой структуры данных
        st.divider()
        st.subheader("📋 Пример структуры данных")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Фактические данные (от API):**")
            example_actual = pd.DataFrame({
                'id': [1, 2, 3, 4, 5, 6],
                'name': ['WELL_001', 'WELL_001', 'WELL_002', 'WELL_002', 'WELL_003', 'WELL_003'],
                'depth': [100.0, 110.0, 95.0, 105.0, 120.0, 130.0],
                'value': [1, 0, 1, 1, 0, 0]
            })
            st.dataframe(example_actual, hide_index=True)

        with col2:
            st.write("**Предсказанные данные (CSV):**")
            example_predict = pd.DataFrame({
                'id': [1, 2, 3, 4, 5, 6],
                'name': ['WELL_001', 'WELL_001', 'WELL_002', 'WELL_002', 'WELL_003', 'WELL_003'],
                'depth': [100.0, 110.0, 95.0, 105.0, 120.0, 130.0],
                'value_predict': [1, 0, 0, 1, 0, 1]
            })
            st.dataframe(example_predict, hide_index=True)

else:
        # ВТОРОЕ ПРИЛОЖЕНИЕ: 3D Визуализация h_kol

        st.title('🎯 MVP Хакатона - Визуализация данных h_kol')

        # Загрузка данных для интерполяции
        st.sidebar.header("📁 Загрузка данных")
        uploaded_h = st.sidebar.file_uploader("Фактические данные EFF_H, H (ZIP)", type="zip", key="actual_data")
        uploaded_h_predict = st.sidebar.file_uploader("Предсказанные данные (CSV)", type=["csv"], key="predicted_data")

        # Настройки отображения
        st.sidebar.header("⚙️ Настройки отображения")
        show_contour = st.sidebar.checkbox("Показать контурную карту", value=True)
        show_boundary = st.sidebar.checkbox("Показать границу области", value=True)
        show_wells = st.sidebar.checkbox("Показать скважины", value=True)
        opacity = st.sidebar.slider("Прозрачность карты", 0.1, 1.0, 0.8)
        colorscale = st.sidebar.selectbox("Цветовая схема",
                                          ['RdYlBu', 'RdBu', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'])

        # Основная логика второго приложения (оставляем без изменений)
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

                    if response.status_code != 200:
                        st.error(response.reason)
                    else:
                        df_predict_h = pd.DataFrame(response.json()['data'])
                        st.success(f'✅ Загружено {len(df_predict_h)} записей')
                        st.write(f"Колонки в файле: {list(df_predict_h.columns)}")

                # Переименование колонок для объединения
                df_merged.rename(columns={'name': 'well'}, inplace=True)
                df_predict_h.rename(columns={'name': 'well'}, inplace=True)

                # Объединение данных
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
                    # Исправляем ошибку - используем nunique() вместо вывода Series
                    unique_wells_count = df_merged['well'].nunique()
                    st.metric("Всего скважин", unique_wells_count)

                with col_stats2:
                    merged_count = len(merged_comparison)
                    st.metric("Объединенных точек", merged_count)

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

            # Импортируем функции интерполяции
            try:
                from interpolation import create_extrapolated_surface, create_original_area_boundary
            except ImportError:
                st.error("❌ Не удалось импортировать модуль interpolation")
                st.info("Убедитесь, что файл interpolation.py находится в той же директории")


                # Создаем заглушки для функций
                def create_extrapolated_surface(x, y, z, grid_points=200, expansion=0.3, method='linear', **kwargs):
                    # Заглушка для тестирования
                    import numpy as np
                    xi = np.linspace(x.min() - expansion, x.max() + expansion, grid_points)
                    yi = np.linspace(y.min() - expansion, y.max() + expansion, grid_points)
                    xi_grid, yi_grid = np.meshgrid(xi, yi)
                    zi = np.zeros_like(xi_grid)
                    return xi, yi, zi, xi_grid, yi_grid


                def create_original_area_boundary(x, y):
                    # Простая граница - выпуклая оболочка
                    from scipy.spatial import ConvexHull
                    points = np.column_stack([x, y])
                    hull = ConvexHull(points)
                    boundary_x = points[hull.vertices, 0]
                    boundary_y = points[hull.vertices, 1]
                    # Замыкаем контур
                    return np.append(boundary_x, boundary_x[0]), np.append(boundary_y, boundary_y[0])

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
            customdata = [[w, pred, fact, d, ep] for w, pred, fact, d, ep in
                          zip(wells, h_pred, h_fact, delta, error_percent)]

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
                    rmse = np.sqrt((merged_comparison['delta'] ** 2).mean())
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
                display_df = merged_comparison[
                    ['well', 'x', 'y', 'h_kol_fact', 'h_kol_pred', 'delta', 'error_percent']].copy()
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

