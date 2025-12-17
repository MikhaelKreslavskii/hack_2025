import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
import base64
from io import BytesIO

# Настройка страницы
st.set_page_config(layout="wide", page_title="Диаграмма скважин")

# Заголовок приложения
st.title("📊 Диаграмма скважин: Фактические vs Предсказанные значения")
st.markdown("Единая цветовая палитра: желтый - коллектор, серый - неколлектор")

# Сайдбар для загрузки файлов
st.sidebar.header("📂 Загрузка данных")

# Загрузка файлов
uploaded_actual = st.sidebar.file_uploader(
    "Загрузите razrez.csv",
    type=['csv'],
    help="id, name, depth, value"
)

uploaded_predict = st.sidebar.file_uploader(
    "Загрузите razrez_predict.csv", 
    type=['csv'],
    help="id, name, depth, value_predict"
)

# Функция для создания интерактивной гистограммы с индивидуальными шкалами глубин для каждой скважины
def create_individual_scale_histogram(filtered_df, selected_wells, stats_df):
    """Создает гистограмму с индивидуальными шкалами глубин для каждой скважины"""
    
    # Настройки
    VISIBLE_WELLS = 10  # Количество скважин в видимой области
    bar_width = 0.35  # Ширина каждого столбца
    gap_between_wells = 0.6  # Промежуток между скважинами
    
    # ЕДИНАЯ цветовая палитра
    COLOR_COLLECTOR = '#FFD700'    # Желтый - коллектор
    COLOR_NONCOLLECTOR = '#CCCCCC' # Серый - неколлектор
    
    # Создаем фигуру с подграфиками для каждой скважины
    # Используем make_subplots для создания общей структуры
    fig = go.Figure()
    
    # Собираем данные о диапазонах глубин для каждой скважины
    well_ranges = {}
    for well in selected_wells:
        well_data = filtered_df[filtered_df['name'] == well]
        if len(well_data) > 0:
            min_depth = well_data['depth'].min()
            max_depth = well_data['depth'].max()
            # Добавляем небольшой запас сверху и снизу для лучшей визуализации
            margin = (max_depth - min_depth) * 0.05 if max_depth > min_depth else 5
            well_ranges[well] = {
                'min': min_depth - margin,
                'max': max_depth + margin,
                'range': max_depth - min_depth
            }
    
    # Нормализуем все глубины к общему диапазону для визуального выравнивания
    # Это нужно, чтобы все скважины имели примерно одинаковую высоту на графике
    normalized_ranges = {}
    if well_ranges:
        max_range = max(r['range'] for r in well_ranges.values())
        for well, range_info in well_ranges.items():
            normalized_ranges[well] = {
                'min': 0,  # Все начинаются с 0
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
            well_range = normalized_ranges.get(well, {'min': 0, 'max': 100, 'original_min': 0, 'original_max': 100, 'scale_factor': 1})
            
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
                    # Для последней точки используем средний шаг
                    if len(well_data) > 1:
                        avg_step_original = (well_data.iloc[-1]['depth'] - well_data.iloc[0]['depth']) / (len(well_data) - 1)
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
                x=[x_center - bar_width/2] * len(fact_heights),
                y=fact_heights,
                base=fact_bottoms,
                width=bar_width,
                marker_color=fact_colors,
                name=f"{well}",
                legendgroup=well,
                showlegend=False,  # Не показываем в легенде имена скважин
                hovertext=fact_hovertexts,
                hoverinfo="text",
                orientation='v'
            ))
            
            # Добавляем предсказанные данные (правый столбец)
            fig.add_trace(go.Bar(
                x=[x_center + bar_width/2] * len(pred_heights),
                y=pred_heights,
                base=pred_bottoms,
                width=bar_width,
                marker_color=pred_colors,
                name=f"{well}",
                legendgroup=well,
                showlegend=False,  # Не показываем в легенде имена скважин
                hovertext=pred_hovertexts,
                hoverinfo="text",
                orientation='v'
            ))
            
            # Добавляем красную границу для расхождений
            for i in range(len(well_data)):
                row = well_data.iloc[i]
                if row['value'] != row['value_predict']:
                    # Определяем координаты для красной границы
                    x_pos = x_center + bar_width/2
                    y_bottom = fact_bottoms[i]
                    height = fact_heights[i]
                    
                    # Добавляем невидимый след для границы
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
        # Фиксируем видимую область (первые 10 скважин)
        range=[-0.5, min(VISIBLE_WELLS, len(selected_wells)) * (1 + gap_between_wells) - 0.5]
    )
    
    # Настраиваем ось Y для нормализованных значений
    fig.update_yaxes(
        title_text="Нормализованная глубина",
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=False,
        # Используем тики, но подписываем их реальными значениями из первой скважины для ориентира
        tickmode='array',
        tickvals=[0, 25, 50, 75, 100],
        ticktext=['Мин', '', 'Средн', '', 'Макс']
    )
    
    # Вычисляем общую ширину для всех скважин
    total_width = len(selected_wells) * (1 + gap_between_wells)
    
    # Настройка макета с горизонтальным скроллом
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
        # Включаем зум и скролл
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor='LightGray'
            ),
            type="linear",
            # Устанавливаем общий диапазон для всех скважин
            range=[-0.5, min(VISIBLE_WELLS, len(selected_wells)) * (1 + gap_between_wells) - 0.5],
            # Включаем зум
            fixedrange=False
        ),
        # Настройки зума
        dragmode='zoom',
        # Ширина для отображения всех скважин в скроллере
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
                y=-5,  # Немного ниже графика
                text=f"{error_pct:.1f}%",
                showarrow=False,
                font=dict(color=color, size=10, weight='bold'),
                yref="y"
            )
    
    # УПРОЩЕННАЯ ЛЕГЕНДА: только коллектор/неколлектор
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=COLOR_COLLECTOR,
        name='Коллектор',
        showlegend=True,
        width=0  # Невидимый бар для легенды
    ))
    
    fig.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color=COLOR_NONCOLLECTOR,
        name='Неколлектор',
        showlegend=True,
        width=0  # Невидимый бар для легенды
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Расхождение',
        showlegend=True
    ))
    
    # Добавляем информацию о глубинах в легенду или как аннотацию
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

# Основной контент приложения
if uploaded_actual is not None and uploaded_predict is not None:
    try:
        # Загрузка данных
        actual_df = pd.read_csv(uploaded_actual)
        predict_df = pd.read_csv(uploaded_predict)
        
        st.sidebar.success("✅ Файлы успешно загружены!")
        
        # Показать информацию о данных
        with st.expander("📋 Информация о данных"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Фактические данные:**")
                st.write(f"- Записей: {len(actual_df):,}")
                st.write(f"- Уникальных скважин: {actual_df['name'].nunique()}")
                st.write(f"- Диапазон глубин: {actual_df['depth'].min():.1f} - {actual_df['depth'].max():.1f} м")
            
            with col2:
                st.write("**Предсказанные данные:**")
                st.write(f"- Записей: {len(predict_df):,}")
                st.write(f"- Уникальных скважин: {predict_df['name'].nunique()}")
                st.write(f"- Диапазон глубин: {predict_df['depth'].min():.1f} - {predict_df['depth'].max():.1f} м")
        
        # Объединение данных
        df = pd.merge(actual_df, predict_df, on=['id', 'name', 'depth'])
        df = df.sort_values(['name', 'depth'])
        
        # Уникальные скважины
        wells = df['name'].unique().tolist()
        
        # Настройки визуализации
        st.sidebar.header("⚙️ Настройки отображения")
        
        # Выбор скважин - ВСЕ по умолчанию
        selected_wells = st.sidebar.multiselect(
            "Выберите скважины:",
            options=wells,
            default=wells,  # ВСЕ скважины по умолчанию
            help="Все скважины выбраны по умолчанию. Можно изменить выбор."
        )
        
        # Если ничего не выбрано, показываем все
        if not selected_wells:
            selected_wells = wells
        
        # Ползунок для диапазона глубин (теперь информационный, а не фильтрующий)
        min_depth = float(df['depth'].min())
        max_depth = float(df['depth'].max())
        
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
            # Фильтрация данных (без фильтрации по глубине, чтобы сохранить все данные каждой скважины)
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
                    st.metric("Макс. ошибка", 
                             f"{worst['Ошибка (%)']}",
                             delta=worst['Скважина'])
            
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
              * Правая часть - все остальные скважины
            
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
                            f"{len(discrepancies[discrepancies['value'] == 0])/len(filtered_df)*100:.1f}%",
                            f"{len(discrepancies[discrepancies['value'] == 1])/len(filtered_df)*100:.1f}%"
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
                                              color="white" if confusion.iloc[i, j] > confusion.values.max()/2 else "black",
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

else:
    # Инструкция при первом запуске
    st.info("👈 Загрузите CSV файлы для начала работы")
    
    with st.expander("📖 Инструкция по использованию"):
        st.markdown("""
        ### 📋 Формат данных:
        
        **Файл 1 (razrez.csv):**
        - id, name, depth, value
        - value: 1=коллектор, 0=неколлектор
        
        **Файл 2 (razrez_predict.csv):**
        - id, name, depth, value_predict
        - value_predict: 1=коллектор, 0=неколлектор
        
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

# Футер
st.markdown("---")
st.caption("Диаграмма скважин | Индивидуальные шкалы глубин | Zoom мышкой")