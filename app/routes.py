from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
import os, shutil
from app.services.speech_service import generate_report

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/analyze")
async def upload_audio(
    child_name: str = Form(...),
    audio: UploadFile = File(...)
):
    file_path = os.path.join(UPLOAD_DIR, audio.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    zip_path = generate_report(file_path, child_name)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="Babblebunch_AI_Reports.zip"
    )
