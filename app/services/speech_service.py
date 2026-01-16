import os
import zipfile
import uuid
from speaker_recognition import SpeechFeedback

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report(file_path, child_name):
    sf = SpeechFeedback(file_path, child_name)

    internal_pdf = sf.generate_pdf()
    parent_pdf = sf.generate_parent_report()

    zip_id = uuid.uuid4().hex
    zip_path = os.path.join(REPORT_DIR, f"{child_name}_{zip_id}.zip")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(internal_pdf, os.path.basename(internal_pdf))
        zipf.write(parent_pdf, os.path.basename(parent_pdf))

    return zip_path
