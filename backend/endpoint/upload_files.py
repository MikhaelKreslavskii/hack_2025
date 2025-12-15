from fastapi import FastAPI, UploadFile, File, APIRouter

router = APIRouter()

@router.post('/upload_files')
async def upload_files(zip: UploadFile = File(...)):
    return {"status": "success"}
