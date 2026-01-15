from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from api.report_service import ReportService

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/generate-report")
async def generate_report(
    child_name: str = Form(...),
    audio: UploadFile = File(...)
):
    audio_path = os.path.join(UPLOAD_DIR, audio.filename)

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    service = ReportService()
    result = service.generate_reports(child_name, audio_path)

    return JSONResponse(content=result)
