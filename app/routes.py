from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import shutil
import uuid

from app.services.speech_service import generate_report, safe_delete_file

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/analyze")
async def upload_audio(
    background_tasks: BackgroundTasks,
    child_name: str = Form(...),
    audio: UploadFile = File(...)
):
    try:
        # -------------------------------------------------
        # STEP 1: Validation
        # -------------------------------------------------
        if not child_name.strip():
            raise HTTPException(status_code=400, detail="Child name is required")

        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file received")

        # -------------------------------------------------
        # STEP 2: Save uploaded file temporarily
        # -------------------------------------------------
        safe_filename = audio.filename.replace(" ", "_")
        unique_id = uuid.uuid4().hex[:8]
        temp_upload_name = f"{unique_id}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_upload_name)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        print("📥 Uploaded file saved:", file_path)

        # -------------------------------------------------
        # STEP 3: Generate reports
        # -------------------------------------------------
        result = generate_report(file_path, child_name)

        zip_path = result["zip_path"]
        temp_files = result["temp_files"]

        # Also include original uploaded file in cleanup
        temp_files.append(file_path)
        temp_files.append(zip_path)

        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(status_code=500, detail="ZIP report was not created")

        # -------------------------------------------------
        # STEP 4: Cleanup after response is sent
        # -------------------------------------------------
        for temp_file in temp_files:
            background_tasks.add_task(safe_delete_file, temp_file)

        # -------------------------------------------------
        # STEP 5: Send ZIP to browser (Downloads folder)
        # -------------------------------------------------
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="Babblebunch_AI_Reports.zip",
            background=background_tasks
        )

    except Exception as e:
        print("❌ ERROR in /analyze:", str(e))
        raise HTTPException(status_code=500, detail=str(e))