from fastapi import FastAPI, UploadFile, File, APIRouter
import os
import sys

from backend.services.upload_files_service import upload_files_well_service, upload_tvt_fact_files_service, \
    upload_tvt_pred_files_service

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




router = APIRouter()

@router.post('/upload_well')
async def upload_well_files(zip: UploadFile = File(...)):

    return await upload_files_well_service(zip=zip)

@router.post('/upload_tvt_fact')
async def upload_tpv_fact_files(zip: UploadFile = File(...)):

    return await upload_tvt_fact_files_service(zip=zip)

@router.post('/upload_tvt_pred')
async def upload_tpv_pred_files(csv: UploadFile = File(...)):

    return await upload_tvt_pred_files_service(csv=csv)