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

def read_navigator_file(file_like):
    """Чтение файла тНавигатора из BytesIO или файла"""
    if hasattr(file_like, 'read'):  # BytesIO
        lines = file_like.readlines()
        file_like.seek(0)  # Сброс позиции
    else:  # Файловый путь
        with open(file_like, 'r', encoding='utf-8') as f:
            lines = f.readlines

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
                    well = parts[3].decode('utf-8')
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

