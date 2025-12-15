from http.client import HTTPException

import lasio
import io
import zipfile
from fastapi import File


async def upload_files(zip: File()):
    with zipfile.ZipFile(zip) as zip_ref:
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
            return {"status": "success", "count": len(df_list)}

        except Exception as e:
            print(e)
            raise HTTPException(500, "Внутренняя ошибка сервера")
