from fastapi import FastAPI, UploadFile, File
import zipfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.Well import Well, Base
from endpoint import upload_router
app = FastAPI()

engine = create_engine("postgresql://postgres:password@localhost:6422/tatneft")
SessionLocal = sessionmaker(bind=engine)

app = FastAPI()
app.include_router(upload_router, prefix="/api/v1")
def create_tables():
    Base.metadata.create_all(bind=engine)

@app.on_event("startup")
def startup():
    create_tables()


@app.get("/")
def read_root():
    return {"Hello": "World"}
