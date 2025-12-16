from fastapi import FastAPI, UploadFile, File, APIRouter
import os
import sys

from backend.services.upload_files_service import upload_files_service

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




router = APIRouter()

@router.post('/upload_files')
async def upload_files(zip: UploadFile = File(...)):

    return await upload_files_service(zip=zip)
