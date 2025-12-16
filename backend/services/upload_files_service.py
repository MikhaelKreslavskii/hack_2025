from http.client import HTTPException

import lasio
import io
import zipfile
from fastapi import File
from sqlalchemy.orm import Session, sessionmaker


from backend.data.Well import Well
from backend.data.database import SessionLocal


async def upload_files_service(zip: File()):
    df_list = {}
    content = await zip.read()
    zip_buffer = io.BytesIO(content)
    with zipfile.ZipFile(zip_buffer) as zip_ref:
        file_names = zip_ref.namelist()
        #file_names_show = [file.split(sep='/')[1] for file in file_names]
        try:

            for name in file_names:
                print('0')
                content_bytes = zip_ref.read(name)
                las_text = content_bytes.decode('utf-8', errors='ignore')  # или 'utf-8'
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
                print('3')
                df_list[key_name] = df
        except Exception as e:
            print(e)
            raise HTTPException(500, "Внутренняя ошибка сервера")
    print(df_list)
    with SessionLocal() as session:
        total_inserted = 0
        for well_name, df in df_list.items():
            print(f"Вставка {well_name}: {len(df)} строк")

            for _, row in df.iterrows():
                # id автоматически генерируется PostgreSQL
                well = Well(
                    name=well_name,  # ✅ ФИКСИРУЕМ ИМЯ
                    depth=float(row['depth']),
                    value=float(row['value'])
                )
                session.add(well)
                total_inserted += 1

            session.commit()  # commit по скважине

        session.close()
        print('Win')
        return {"status": "success", "count": len(df_list)}


