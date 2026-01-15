from speaker_recognition import SpeechFeedback
import os

class ReportService:

    def generate_reports(self, child_name, audio_path):
        sf = SpeechFeedback(audio_path)

        # override child name from form
        sf.child_name = child_name

        # generate both reports
        sf.generate_pdf()
        sf.generate_parent_report()

        return {
            "status": "success",
            "message": "Reports generated successfully",
            "child_name": child_name
        }
