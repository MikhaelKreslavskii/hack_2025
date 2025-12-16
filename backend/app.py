from fastapi import FastAPI, UploadFile, File
import zipfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.data.Well import Base
from backend.data.database import engine
from backend.endpoint.upload_files import router
app = FastAPI()



app = FastAPI()
app.include_router(router, prefix="/api/v1")
def create_tables():
    Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup():
    create_tables()


@app.get("/")
def read_root():
    return {"Hello": "World"}
