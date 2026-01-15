from fastapi import APIRouter, UploadFile, File, Form
import os, shutil
from app.services.speech_service import generate_report

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_audio(
    child_name: str = Form(...),
    audio: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIR, audio.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    result = generate_report(file_path, child_name)

    return {"message": result}
