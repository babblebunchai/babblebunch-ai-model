import os
import zipfile
import uuid
import librosa
import soundfile as sf
from speaker_recognition import SpeechFeedback

REPORT_DIR = "reports"
UPLOAD_DIR = "uploads"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def safe_delete_file(file_path):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Could not delete file: {file_path} -> {e}")


def clean_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        print(f"Could not clean folder {folder_path}: {e}")


def sanitize_name(name):
    return "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================
def generate_report(file_path, child_name):
    temp_files = []

    try:
        # Optional: clean old junk before new run
        clean_folder(REPORT_DIR)

        safe_name = sanitize_name(child_name)
        unique_id = uuid.uuid4().hex[:8]

        # -------------------------------------------------
        # STEP 1: Convert uploaded file to ONE clean WAV
        # -------------------------------------------------
        y, sr = librosa.load(file_path, sr=16000)

        wav_path = os.path.join(UPLOAD_DIR, f"{safe_name}_{unique_id}_final.wav")
        sf.write(wav_path, y, sr)
        temp_files.append(wav_path)

        print("🎧 Converted to WAV:", wav_path)

        # -------------------------------------------------
        # STEP 2: Generate reports
        # -------------------------------------------------
        speech = SpeechFeedback(wav_path, safe_name)

        internal_pdf = speech.generate_pdf()
        parent_pdf = speech.generate_parent_report()

        print("📄 Internal PDF:", internal_pdf)
        print("📄 Parent PDF:", parent_pdf)

        if not internal_pdf or not os.path.exists(internal_pdf):
            raise Exception(f"Internal PDF not created properly: {internal_pdf}")

        if not parent_pdf or not os.path.exists(parent_pdf):
            raise Exception(f"Parent PDF not created properly: {parent_pdf}")

        temp_files.extend([internal_pdf, parent_pdf])

        # -------------------------------------------------
        # STEP 3: Create ZIP (temporary)
        # -------------------------------------------------
        zip_path = os.path.join(REPORT_DIR, f"{safe_name}_{unique_id}_reports.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(internal_pdf, os.path.basename(internal_pdf))
            zipf.write(parent_pdf, os.path.basename(parent_pdf))

        print("✅ ZIP CREATED:", zip_path)

        if not os.path.exists(zip_path):
            raise Exception("ZIP file was not created properly")

        # Return zip path + temp files list for cleanup later
        return {
            "zip_path": zip_path,
            "temp_files": temp_files
        }

    except Exception as e:
        print("❌ ERROR in generate_report:", str(e))

        # Cleanup on failure
        for temp_file in temp_files:
            safe_delete_file(temp_file)

        raise