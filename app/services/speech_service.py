import os
from pathlib import Path

# IMPORT YOUR EXISTING CLASS
# ⚠️ If SpeechFeedback is in SAME FILE, remove this import
from speaker_recognition import SpeechFeedback

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_report(audio_path: str, child_name: str):
    """
    Generates both internal + parent PDF reports
    """

    sf = SpeechFeedback(audio_path)

    # override name from UI
    sf.child_name = child_name

    sf.generate_pdf()
    sf.generate_parent_report()

    return f"Reports generated for {child_name}"
