from http.client import HTTPException
import sqlalchemy as sa
import lasio
import io
import zipfile
from fastapi import File
from sqlalchemy.orm import Session, sessionmaker

from backend.data.TVT_Fact import TVT_Fact
from backend.data.Well import Well
from backend.data.database import SessionLocal, engine
import pandas as pd

from frontend.interpolation import read_navigator_file


async def upload_files_well_service(zip: File()):
    df_list = {}
    content = await zip.read()
    zip_buffer = io.BytesIO(content)
    with zipfile.ZipFile(zip_buffer) as zip_ref:
        file_names = zip_ref.namelist()
        try:
            for name in file_names:
                print('0')
                content_bytes = zip_ref.read(name)
                las_text = content_bytes.decode('utf-8', errors='ignore')
                print('1')
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
                df.columns = ['depth', 'value']
                key_name = str(name).split(sep='/')[1].split(sep='.')[0]
                df.dropna(inplace=True)

                # Добавляем имя скважины в DataFrame
                df['name'] = key_name

                print(f'3 - добавлена скважина {key_name} с {len(df)} строками')
                df_list[key_name] = df

        except Exception as e:
            print(e)
            raise HTTPException(500, "Внутренняя ошибка сервера")

    print(f"Обработано скважин: {len(df_list)}")

    with SessionLocal() as session:
        total_inserted = 0
        for well_name, df in df_list.items():
            print(f"Вставка {well_name}: {len(df)} строк")

            for _, row in df.iterrows():
                # id автоматически генерируется PostgreSQL
                well = Well(
                    name=well_name,
                    depth=float(row['depth']),
                    value=float(row['value'])
                )
                session.add(well)
                total_inserted += 1

            session.commit()  # commit по скважине

        session.close()

        # Объединяем все DataFrame в один
        # Способ 1: Преобразуем словарь в список DataFrame и объединяем
        combined_df = pd.concat(list(df_list.values()), ignore_index=True)

        # Альтернативный способ: Собираем в список с явным указанием
        # combined_dfs = []
        # for well_name, df in df_list.items():
        #     df_copy = df.copy()
        #     df_copy['name'] = well_name  # Убедимся, что имя есть
        #     combined_dfs.append(df_copy)
        # combined_df = pd.concat(combined_dfs, ignore_index=True)

        print(f'Win - объединено {len(combined_df)} строк из {len(df_list)} скважин')

        # Сортируем для удобства
        combined_df = combined_df.sort_values(['name', 'depth']).reset_index(drop=True)

        # Добавляем id для соответствия ожидаемой структуре
        combined_df['id'] = range(1, len(combined_df) + 1)

        # Переупорядочиваем колонки: id, name, depth, value
        combined_df = combined_df[['id', 'name', 'depth', 'value']]

        print("Структура combined_df:")
        print(combined_df.head())
        print(f"Колонки: {combined_df.columns.tolist()}")

        return {
            "status": "success",
            "count": len(df_list),
            "total_rows": len(combined_df),
            "data": combined_df.to_dict(orient='records')  # Используем 'records' для списка словарей
        }


async def upload_tvt_fact_files_service(zip: File()):  # ✅ UploadFile вместо File()
    content = await zip.read()
    zip_buffer = io.BytesIO(content)

    df_eff_h_list, df_h_list = [], []

    with zipfile.ZipFile(zip_buffer) as zip_ref:
        for name in zip_ref.namelist():
            try:
                file_like = io.BytesIO(zip_ref.read(name))
                if 'FF' in name.upper():
                    df_eff_h_list.append(read_navigator_file(file_like))
                else:
                    df_h_list.append(read_navigator_file(file_like))
            except Exception as e:
                print(f"Ошибка файла {name}: {e}")
                continue

    if not df_eff_h_list or not df_h_list:
        raise HTTPException(500, "Не найдены файлы FF и H")

    df_eff_h = pd.concat(df_eff_h_list, ignore_index=True)
    df_h = pd.concat(df_h_list, ignore_index=True)

    # Обработка данных
    df_eff_h = df_eff_h.rename(columns={'value': 'eff_h'}).drop(columns=['z'])
    df_h = df_h.rename(columns={'value': 'h'}).drop(columns=['z'])

    df_merged = pd.merge(df_eff_h, df_h, on=['x', 'y', 'well'], how='inner')
    df_merged['h_kol'] = df_merged['eff_h'] / df_merged['h']
    df_merged[['x', 'y']] = df_merged[['x', 'y']].astype(float).round(2)

    # ✅ Подготовка колонок для таблицы (без id)
    df_to_insert = df_merged[['well', 'x', 'y', 'h_kol']].copy()
    df_to_insert.rename(columns={'well': 'name'}, inplace=True)  # well -> name

    # 🔥 Быстрая вставка через Pandas (1000+ строк/сек)
    try:
        # if_exists='append' добавляет строки
        # method='multi' для скорости
        inserted = df_to_insert.to_sql(
            name='tvt_fact',
            con=engine,
            if_exists='replace',
            index=False,  # не вставляем индекс DataFrame
            method='multi',  # батчевая вставка
            dtype={
                'name': sa.String(255),
                'x': sa.Float,
                'y': sa.Float,
                'h_kol': sa.Float
            }
        )
        print('success')
    except Exception as e:
        print(f"Ошибка вставки: {e}")
        raise HTTPException(500, f"Ошибка сохранения в БД: {str(e)}")

    return {"status": "success", "inserted": len(df_to_insert), "data": df_to_insert.to_dict()}





async def upload_tvt_pred_files_service(csv: File()):  # ✅ UploadFile
    # ✅ Читаем CSV напрямую (НЕ ZIP!)
    print('0')
    content = await csv.read()
    df = pd.read_csv(io.BytesIO(content))

    print('1')
    df_to_insert = df[['x', 'y', 'well', 'h_kol']].copy()  # Измените колонки под ваш CSV
    df_to_insert.rename(columns={'well': 'name'}, inplace=True)

    required_columns = ['x', 'y', 'name', 'h_kol']
    if not all(col in df_to_insert.columns for col in required_columns):
        return HTTPException(400, 'Некорректные столбцы в таблице')
    print('2')
    # Конвертируем типы данных
    df_to_insert['x'] = pd.to_numeric(df_to_insert['x'], errors='coerce')
    df_to_insert['y'] = pd.to_numeric(df_to_insert['y'], errors='coerce')
    df_to_insert['h_kol'] = pd.to_numeric(df_to_insert['h_kol'], errors='coerce')
    print('3')
    # Удаляем строки с NaN значениями
    df_to_insert = df_to_insert.dropna(subset=['x', 'y', 'h_kol'])
    print('Dataframe ready')
    # 🔥 Вставка с заменой (без ошибок дублирования)
    try:


        # ✅ Вставляем новые данные
        inserted = df_to_insert.to_sql(
            name='tvt_predict',  # ✅ Правильная таблица!
            con=engine,
            if_exists='replace',
            index=False,
            method='multi',
            dtype={
                'name': sa.String(255),
                'x': sa.Float,
                'y': sa.Float,
                'h_kol': sa.Float
            }
        )
        print(f'Успешно вставлено: {len(df_to_insert)} строк')
        return {
            "status": "success",
            "inserted": len(df_to_insert),
            "data": df_to_insert.to_dict('records')
        }

    except Exception as e:
        print(f"Ошибка вставки: {e}")
        raise HTTPException(500, f"Ошибка сохранения в БД: {str(e)}")

