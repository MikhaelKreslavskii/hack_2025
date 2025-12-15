import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata, Rbf
from scipy.spatial import ConvexHull, KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBF_kernel
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')

def read_navigator_file(filepath):
    """Чтение файла тНавигатора"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'Float Value':
            data_start = i + 1
            break

    data_lines = lines[data_start:]

    data = []
    for line in data_lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    well = parts[3]
                    value = float(parts[4])
                    data.append([x, y, z, well, value])
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'well', 'value'])
    return df


def idw_interpolation(x, y, z, xi_grid, yi_grid, power=2, neighbors=10):
    """
    Интерполяция методом обратных расстояний (Inverse Distance Weighting).

    Parameters:
    -----------
    power : степень обратного расстояния (обычно 2)
    neighbors : количество ближайших соседей для использования
    """
    xi = xi_grid.flatten()
    yi = yi_grid.flatten()
    zi = np.zeros_like(xi)

    # Создаем KD-дерево для быстрого поиска соседей
    tree = KDTree(np.column_stack([x, y]))

    for i in range(len(xi)):
        # Находим k ближайших соседей
        distances, indices = tree.query([[xi[i], yi[i]]], k=min(neighbors, len(x)))

        # Вычисляем веса
        weights = 1.0 / (distances[0] ** power + 1e-10)  # добавляем маленькое число чтобы избежать деления на 0
        weights = weights / weights.sum()

        # Взвешенное среднее
        zi[i] = np.sum(z[indices[0]] * weights)

    return zi.reshape(xi_grid.shape)


def kriging_interpolation(x, y, z, xi_grid, yi_grid):
    """
    Интерполяция методом Кригинга (гауссовские процессы).
    Это упрощенная версия обычного кригинга.
    """
    try:
        # Используем Gaussian Process Regressor как аналог кригинга
        X = np.column_stack([x, y])

        # Определяем ядро (вариограмму)
        kernel = RBF_kernel(length_scale=1.0)

        # Создаем модель гауссовского процесса
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                      normalize_y=True, n_restarts_optimizer=10)

        # Обучаем модель
        gp.fit(X, z)

        # Прогнозируем значения на сетке
        X_grid = np.column_stack([xi_grid.flatten(), yi_grid.flatten()])
        zi, sigma = gp.predict(X_grid, return_std=True)

        return zi.reshape(xi_grid.shape)
    except Exception as e:
        print(f"Ошибка при кригинге: {e}")
        # Возвращаем NaN если не удалось
        return np.full(xi_grid.shape, np.nan)


def svr_interpolation(x, y, z, xi_grid, yi_grid, kernel='rbf', C=100, gamma=0.1):
    """
    Интерполяция методом Support Vector Regression (SVR).
    """
    try:
        # Подготовка данных
        X = np.column_stack([x, y])

        # Создаем и обучаем модель SVR
        svr_model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=0.1)
        svr_model.fit(X, z)

        # Прогнозируем значения на сетке
        X_grid = np.column_stack([xi_grid.flatten(), yi_grid.flatten()])
        zi = svr_model.predict(X_grid)

        return zi.reshape(xi_grid.shape)
    except Exception as e:
        print(f"Ошибка при SVR интерполяции: {e}")
        # Возвращаем NaN если не удалось
        return np.full(xi_grid.shape, np.nan)


def create_extrapolated_surface(x, y, z, grid_points=150, expansion=0.3,
                                method='rbf', **kwargs):
    """
    Создает экстраполированную поверхность с расширением за пределы исходных точек.

    Parameters:
    -----------
    method : str
        'linear' - линейная интерполяция
        'cubic' - кубическая интерполяция
        'nearest' - ближайший сосед
        'rbf' - радиальные базисные функции
        'idw' - обратное расстояние
        'kriging' - кригинг (гауссовские процессы)
        'svr' - Support Vector Regression
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min_exp = x_min - x_range * expansion
    x_max_exp = x_max + x_range * expansion
    y_min_exp = y_min - y_range * expansion
    y_max_exp = y_max + y_range * expansion

    xi = np.linspace(x_min_exp, x_max_exp, grid_points)
    yi = np.linspace(y_min_exp, y_max_exp, grid_points)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    if method == 'rbf':
        # Используем разные ядра RBF
        rbf_function = kwargs.get('rbf_function', 'linear')
        smooth = kwargs.get('smooth', 0.1)
        rbf = Rbf(x, y, z, function=rbf_function, smooth=smooth)
        zi_extrapolated = rbf(xi_grid, yi_grid)

    elif method == 'idw':
        # IDW интерполяция
        power = kwargs.get('power', 2)
        neighbors = kwargs.get('neighbors', min(10, len(x)))
        zi_extrapolated = idw_interpolation(x, y, z, xi_grid, yi_grid,
                                            power=power, neighbors=neighbors)

    elif method == 'kriging':
        # Кригинг (гауссовские процессы)
        zi_extrapolated = kriging_interpolation(x, y, z, xi_grid, yi_grid)

    elif method == 'svr':
        # SVR интерполяция
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 100)
        gamma = kwargs.get('gamma', 0.1)
        zi_extrapolated = svr_interpolation(x, y, z, xi_grid, yi_grid,
                                            kernel=kernel, C=C, gamma=gamma)

    elif method in ['linear', 'cubic', 'nearest']:
        # Стандартные методы из scipy
        zi_extrapolated = griddata(
            (x, y), z, (xi_grid, yi_grid),
            method=method, fill_value=np.nan
        )
    else:
        raise ValueError(f"Неизвестный метод интерполяции: {method}")

    # Заполнение NaN значений (если есть)
    if np.any(np.isnan(zi_extrapolated)):
        from scipy.ndimage import distance_transform_edt
        mask = ~np.isnan(zi_extrapolated)
        if np.any(mask):
            distances, indices = distance_transform_edt(~mask, return_indices=True)
            zi_filled = zi_extrapolated.copy()
            zi_filled[~mask] = zi_extrapolated[indices[0][~mask], indices[1][~mask]]
            zi_extrapolated = zi_filled

    return xi, yi, zi_extrapolated, xi_grid, yi_grid


def create_original_area_boundary(x, y):
    """
    Создает границу исходной области данных на основе выпуклой оболочки.
    """
    if len(x) >= 3:
        try:
            points = np.column_stack([x, y])
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            return hull_points[:, 0], hull_points[:, 1]
        except:
            pass

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    padding_x = (x_max - x_min) * 0.05
    padding_y = (y_max - y_min) * 0.05

    boundary_x = [
        x_min - padding_x, x_max + padding_x,
        x_max + padding_x, x_min - padding_x,
        x_min - padding_x
    ]
    boundary_y = [
        y_min - padding_y, y_min - padding_y,
        y_max + padding_y, y_max + padding_y,
        y_min - padding_y
    ]

    return boundary_x, boundary_y


# Чтение файлов
df_eff_h = read_navigator_file(eff_h_file)
df_h = read_navigator_file(h_file)

# Переименовываем колонки
df_eff_h = df_eff_h.rename(columns={'value': 'eff_h'})
df_h = df_h.rename(columns={'value': 'h'})

# Объединение данных
df_merged = pd.merge(
    df_eff_h,
    df_h,
    on=['x', 'y', 'z', 'well'],
    how='inner'
)

print(f"Объединено {len(df_merged)} скважин")

# Создание новой колонки h_kol = eff_h / h
df_merged['h_kol'] = df_merged['eff_h'] / df_merged['h']

# Получаем данные для визуализации
x = df_merged['x'].values
y = df_merged['y'].values
z_coords = df_merged['z'].values  # координаты Z скважин
z_values = df_merged['h_kol'].values  # значения h_kol

# Список доступных методов интерполяции
available_methods = ['linear', 'cubic', 'nearest', 'rbf', 'idw', 'kriging', 'svr']

method_descriptions = {
    'linear': 'Линейная интерполяция (scipy)',
    'cubic': 'Кубическая интерполяция (scipy)',
    'nearest': 'Ближайший сосед (scipy)',
    'rbf': 'Радиальные базисные функции',
    'idw': 'Обратное расстояние (Inverse Distance Weighting)',
    'kriging': 'Кригинг (Гауссовские процессы)',
    'svr': 'Support Vector Regression'
}

print("\n" + "=" * 80)
print("ДОСТУПНЫЕ МЕТОДЫ ИНТЕРПОЛЯЦИИ:")
print("=" * 80)
for i, method in enumerate(available_methods, 1):
    print(f"{i:2d}. {method:10s} - {method_descriptions[method]}")

print("\n" + "=" * 80)
print("ВЫБЕРИТЕ МЕТОД ИНТЕРПОЛЯЦИИ:")
print("=" * 80)
print("1. linear  - быстрая, простая линейная интерполяция")
print("2. cubic   - гладкая кубическая интерполяция")
print("3. rbf     - радиальные базисные функции (рекомендуется)")
print("4. idw     - обратное расстояние (хорошо для локальных особенностей)")
print("5. kriging - кригинг (геостатистика, может быть медленным)")
print("6. svr     - метод опорных векторов (машинное обучение)")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   Выбор метода интерполяции.   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
selected_method = 'rbf'  # Можете изменить на любой из available_methods
print(f"\nВыбран метод: {selected_method} - {method_descriptions[selected_method]}")

# Параметры для выбранного метода
method_params = {}
if selected_method == 'rbf':
    method_params = {'rbf_function': 'linear', 'smooth': 0.1}
elif selected_method == 'idw':
    method_params = {'power': 2, 'neighbors': min(10, len(x))}
elif selected_method == 'svr':
    method_params = {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}

# Создаем экстраполированную поверхность
try:
    print(f"\nВыполняем интерполяцию методом: {selected_method}...")
    xi, yi, zi_extrapolated, xi_grid, yi_grid = create_extrapolated_surface(
        x, y, z_values,
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
        x, y, z_values,
        grid_points=150,
        expansion=0.3,
        method=selected_method
    )

# Создаем границу исходной области
boundary_x, boundary_y = create_original_area_boundary(x, y)

# Создаем figure с экстраполяцией
fig = go.Figure()

# Подготовка данных для всплывающей подсказки контурной карты
hover_texts = []
for i in range(len(xi)):
    row_texts = []
    for j in range(len(yi)):
        x_val = xi_grid[i, j]
        y_val = yi_grid[i, j]
        z_interp = zi_extrapolated[i, j]

        # Получаем ближайшую скважину для Z координаты
        if len(x) > 0:
            distances = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
            nearest_idx = np.argmin(distances)
            z_coord = z_coords[nearest_idx]
            well_name = df_merged['well'].iloc[nearest_idx]
            well_h_kol = z_values[nearest_idx]
            distance = distances[nearest_idx]
        else:
            z_coord = np.nan
            well_name = "Н/Д"
            well_h_kol = np.nan
            distance = np.nan

        text = (f"<b>Метод интерполяции: {method_descriptions[selected_method]}</b><br><br>"
                f"<b>Координаты точки:</b><br>"
                f"• X: {x_val:.2f}<br>"
                f"• Y: {y_val:.2f}<br>"
                # f"• Z (ближайшей скважины): {z_coord:.2f}<br><br>"
                # f"<b>Ближайшая скважина:</b><br>"
                # f"• {well_name}<br>"
                # f"• Расстояние: {distance:.2f}<br>"
                f"• Значение h_kol: {well_h_kol:.4f}<br><br>"
                f"<b>Интерполированное значение:</b><br>"
                f"• h_kol: {z_interp:.4f}")
        row_texts.append(text)
    hover_texts.append(row_texts)

hover_texts = np.array(hover_texts)

# 1. Экстраполированная тепловая карта (вся область) с улучшенными подсказками
fig.add_trace(go.Contour(
    x=xi,
    y=yi,
    z=zi_extrapolated,
    colorscale='Viridis',
    opacity=0.8,
    contours=dict(
        showlabels=True,
        labelfont=dict(size=11, color='white'),
        coloring='heatmap'
    ),
    colorbar=dict(
        title=dict(
            text='h_kol = eff_h / h',
            side='right',
            font=dict(size=14)
        ),
        thickness=20,
        len=0.8
    ),
    name=f'Интерполяция: {selected_method}',
    customdata=hover_texts,
    hovertemplate='%{customdata}<extra></extra>',
    hoverinfo='text'
))

# 2. Граница исходной области (если бы не было экстраполяции)
fig.add_trace(go.Scatter(
    x=boundary_x,
    y=boundary_y,
    mode='lines',
    line=dict(
        color='rgba(255, 0, 0, 0.8)',
        width=3,
        dash='dash'
    ),
    fill='none',
    name='Граница области без экстраполяции',
    hoverinfo='skip'
))

# 3. Исходные точки (скважины) с полной информацией
fig.add_trace(go.Scatter(
    x=df_merged['x'],
    y=df_merged['y'],
    mode='markers+text',
    marker=dict(
        size=12,
        color=df_merged['h_kol'],
        colorscale='Viridis',
        showscale=False,
        line=dict(width=2, color='white'),
        symbol='circle'
    ),
    text=df_merged['well'],
    textposition="top center",
    textfont=dict(size=10, color='black'),
    customdata=np.stack((
        df_merged['x'].values,
        df_merged['y'].values,
        df_merged['z'].values,
        df_merged['well'].values,
        df_merged['h_kol'].values,
        df_merged['eff_h'].values,
        df_merged['h'].values
    ), axis=-1),
    hovertemplate=(
        '<b>Скважина:</b> %{customdata[3]}<br>'
        '<b>Координаты:</b><br>'
        '  X: %{customdata[0]:.2f}<br>'
        '  Y: %{customdata[1]:.2f}<br>'
        '  Z: %{customdata[2]:.2f}<br>'
        '<b>Значения:</b><br>'
        '  h_kol: %{customdata[4]:.4f}<br>'
        '  eff_h: %{customdata[5]:.4f}<br>'
        '  h: %{customdata[6]:.4f}<br>'
        '<extra></extra>'
    ),
    name='Скважины',
    showlegend=True
))

# Настройка layout
fig.update_layout(
    title=dict(
        text=f'Карта значений h_kol с экстраполяцией<br>Метод: {method_descriptions[selected_method]}',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        title='Координата X',
        scaleanchor="y",
        scaleratio=1,
        constrain='domain',
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    yaxis=dict(
        title='Координата Y',
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1,
        font=dict(size=12)
    ),
    width=1100,
    height=850,
    template='plotly_white',
    hovermode='closest'
)

# Добавляем аннотацию с пояснением
fig.add_annotation(
    x=0.02,
    y=0.02,
    xref="paper",
    yref="paper",
    text=f"Метод: {selected_method}<br>Красная линия - граница без экстраполяции",
    showarrow=False,
    font=dict(size=11, color='red'),
    bgcolor='rgba(255, 255, 255, 0.8)',
    bordercolor='red',
    borderwidth=1,
    borderpad=4
)

fig.show()

# Выводим статистику
print("\n" + "=" * 80)
print("СТАТИСТИКА ПО ДАННЫМ:")
print("=" * 80)
print(f"Общее количество скважин: {len(df_merged)}")
print(f"Диапазон значений h_kol: {df_merged['h_kol'].min():.4f} - {df_merged['h_kol'].max():.4f}")
print(f"Среднее значение h_kol: {df_merged['h_kol'].mean():.4f}")
print(f"Стандартное отклонение: {df_merged['h_kol'].std():.4f}")
print(f"Медиана: {df_merged['h_kol'].median():.4f}")

# Сохраняем результаты
output_filename = f'results_h_kol_{selected_method}.csv'
df_merged.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\nРезультаты сохранены в файл: {output_filename}")

# Создаем HTML файл с графиком
import plotly.io as pio

html_filename = f'h_kol_map_{selected_method}.html'
pio.write_html(fig, html_filename)
print(f"Интерактивная карта сохранена в файл: {html_filename}")

print("\n" + "=" * 80)
print("ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ РАЗНЫХ МЕТОДОВ:")
print("=" * 80)
print("Чтобы изменить метод интерполяции, измените строку:")
print("selected_method = 'rbf'")
print("\nДоступные методы:")
for method in available_methods:
    print(f"  '{method}' - {method_descriptions[method]}")

print("\nРекомендации:")
print("1. 'rbf' - баланс скорости и качества, рекомендуется по умолчанию")
print("2. 'idw' - хорошо сохраняет локальные особенности")
print("3. 'linear' - самый быстрый метод")
print("4. 'cubic' - гладкая интерполяция")
print("5. 'kriging' - геостатистический метод, точный но медленный")
print("6. 'svr' - метод машинного обучения, требует настройки параметров")

print("\n" + "=" * 80)
print("ДЕТАЛЬНЫЕ ДАННЫЕ ПО СКВАЖИНАМ:")
print("=" * 80)
for idx, row in df_merged.iterrows():
    print(f"\nСкважина: {row['well']}")
    print(f"  Координаты: X={row['x']:.2f}, Y={row['y']:.2f}, Z={row['z']:.2f}")
    print(f"  eff_h: {row['eff_h']:.4f}")
    print(f"  h: {row['h']:.4f}")
    print(f"  h_kol (eff_h/h): {row['h_kol']:.4f}")