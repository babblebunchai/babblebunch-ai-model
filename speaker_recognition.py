# =========================================================
# SpeechSpark Kids AI - Single PDF Per Child (Final Fixed Version)
# =========================================================

import os
import glob
import datetime
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

import librosa
import torchaudio
import torch
from speechbrain.pretrained import EncoderClassifier


# --------------------- CONFIG ---------------------
OUTPUT_DIR = r"C:\Users\Isha Arora\OneDrive\Documents\AI_Reports"
BRAND_NAME = "Babblebunch AI"
WEBSITE_URL = "www.babblebunchai.com"
LOGO_PATH = "logo.png"
AVATAR_PATH = "avatar.png"
BACKGROUND_COLOR = colors.HexColor("#FFFFFF")  # soft peach
HeaderBackground = colors.HexColor("#FFFFFF")  # light pink
# or "#F9FBFF" (light blue)
# or "#F6FFF8" (mint)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- UI Banner ---------------------
class ColorBanner(Flowable):
    def __init__(self, width, height=14 * mm):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        colors_list = ["#ffb3c7", "#ffd480", "#b3e6ff", "#c3ffa6"]
        stripe = self.width / len(colors_list)
        x = 0
        for col in colors_list:
            c.setFillColor(colors.HexColor(col))
            c.rect(x, 0, stripe, self.height, stroke=0, fill=1)
            x += stripe

# =========================================================
# MAIN CLASS
# =========================================================
class SpeechFeedback:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.child_name = os.path.splitext(os.path.basename(audio_path))[0]

        self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)

        # Metrics
        self.energy = 0
        self.tempo = 0
        self.pauses_ratio = 0
        self.pitch_mean = 0
        self.pitch_std = 0
        self.duration = 0
        self.words_per_minute = 0
        self.clarity_db = 0

        self.analyze_audio()

    # -----------------------------------------------------
    # Audio Analysis
    # -----------------------------------------------------
    def analyze_audio(self):

        # Energy
        try:
            self.energy = float(np.mean(librosa.feature.rms(y=self.y)))
        except:
            self.energy = 0

        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            self.tempo = float(tempo)
        except:
            self.tempo = 0

        # Pauses
        try:
            parts = librosa.effects.split(self.y, top_db=20)
            voiced = sum(end - start for start, end in parts)
            self.pauses_ratio = 1 - (voiced / len(self.y))
        except:
            self.pauses_ratio = 1.0

        # Pitch
        try:
            f0, _, _ = librosa.pyin(self.y, fmin=80, fmax=450, sr=self.sr)
            vals = f0[~np.isnan(f0)]
            self.pitch_mean = float(np.mean(vals)) if len(vals) else 0
            self.pitch_std = float(np.std(vals)) if len(vals) else 0
        except:
            self.pitch_mean = 0
            self.pitch_std = 0

        # Duration
        self.duration = len(self.y) / self.sr

        # Speaking rate
        if self.duration > 0:
            estimated_words = self.duration / 0.4
            self.words_per_minute = (estimated_words / self.duration) * 60

        # Clarity
        try:
            rms = librosa.feature.rms(y=self.y)[0]
            noise_floor = np.mean(np.sort(rms)[:50])
            sig = np.mean(rms)
            self.clarity_db = float(20 * np.log10((sig + 1e-9) / (noise_floor + 1e-9)))
        except:
            self.clarity_db = 0

    # -----------------------------------------------------
    # Matric Chart
    # -----------------------------------------------------
    def metric_rows(self):
        metrics = [
            ("Energy", self.energy, 0.02, 0.10),
            ("Tempo (BPM)", self.tempo, 90, 160),
            ("Smoothness", 1 - self.pauses_ratio, 0.5, 1.0),
            ("Pitch Mean (Hz)", self.pitch_mean, 150, 300),
            ("Pitch Variability (Hz)", self.pitch_std, 20, 80),
            ("Duration (s)", self.duration, 5, 15),
            ("Speaking Rate (WPM)", self.words_per_minute, 80, 160),
            ("Clarity (dB)", self.clarity_db, 10, 40),
        ]

        rows = []
        for name, val, mn, mx in metrics:
            if val < mn:
                status = "Below target"
            elif val > mx:
                status = "Above target"
            else:
                status = "Within target"
            rows.append([name, f"{val:.2f}", f"{mn}–{mx}", status])

        return rows
        
    # -----------------------------------------------------
    # Chart
    # -----------------------------------------------------
    def radar_chart(self):
        labels = ["Energy", "Tempo", "Smooth", "PitchMean", "PitchVar", "Duration", "Rate", "Clarity"]
        values = [
            self.energy, self.tempo, 1 - self.pauses_ratio,
            self.pitch_mean, self.pitch_std, self.duration,
            self.words_per_minute, self.clarity_db
        ]
        ranges = [
            (0.0, 0.12), (60, 200), (0.0, 1.0), (80, 350),
            (0, 140), (1, 15), (40, 200), (5, 40)
        ]

        norm = [(v - mn) / (mx - mn) for v, (mn, mx) in zip(values, ranges)]

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        norm += norm[:1]
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(figsize=(6,5), subplot_kw=dict(polar=True))
        ax.plot(angles, norm, linewidth=2)
        ax.fill(angles, norm, alpha=0.4)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title(f"{self.child_name} Speech Profile")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=140)
        plt.close()
        buf.seek(0)
        return buf

    # -----------------------------------------------------
    # Observations
    # -----------------------------------------------------

    def observations(self):
        obs = []

        if self.energy < 0.02:
            obs.append("• Volume is low — encourage louder speaking and breath support.")
        elif self.energy > 0.10:
            obs.append("• Volume is high — encourage softer, controlled voice.")
        else:
            obs.append("• Volume is within the comfortable range.")

        if self.pauses_ratio > 0.5:
            obs.append("• Too many pauses — practice short scripted sentences to improve fluency.")
        else:
            obs.append("• Flow is good with minimal pauses — well done!")

        if self.pitch_mean > 300:
            obs.append("• Pitch is very high — work on calm and deeper tone exercises.")
        elif self.pitch_mean < 150:
            obs.append("• Pitch is low — encourage expressive reading.")
        else:
            obs.append("• Pitch range is appropriate.")

        if self.pitch_std > 80:
            obs.append("• Pitch varies a lot — practice controlled intonation.")
        else:
            obs.append("• Pitch variation is healthy.")

        if self.words_per_minute < 80:
            obs.append("• Speaking rate is slow — try rhythmic speaking or read-aloud drills.")
        elif self.words_per_minute > 160:
            obs.append("• Speaking rate is too fast — teach pausing at punctuation.")
        else:
            obs.append("• Speaking rate is appropriate.")

        if self.clarity_db < 10:
            obs.append("• Clarity is low — check microphone and distance.")
        else:
            obs.append("• Clarity is acceptable.")

        return obs


    # -----------------------------------------------------
    # Improvement Plan
    # -----------------------------------------------------
    def plan(self):
        plan = []

        if self.energy < 0.02:
            plan.append("1) Volume games: call-and-response exercises with increasing loudness.")
        elif self.energy > 0.10:
            plan.append("1) Controlled speech drills: whisper reading + microphone distance practice.")
        else:
            plan.append("1) Maintain current volume but practice consistency.")

        if self.pauses_ratio > 0.5:
            plan.append("2) Fluency drills: repeat-after-me sentences & timed speaking challenges.")
        else:
            plan.append("2) Continue phrase-linking games to keep fluency strong.")

        if self.pitch_std < 20:
            plan.append("3) Pitch expression games: robot voice → happy voice → sad voice.")
        elif self.pitch_std > 80:
            plan.append("3) Pitch control drill: practice reading in neutral tone at steady pitch.")
        else:
            plan.append("3) Maintain expressive reading activities.")

        if self.words_per_minute < 80:
            plan.append("4) Speed up: do 30-second quick read games.")
        elif self.words_per_minute > 160:
            plan.append("4) Slow down: practice pacing using commas/full stops.")
        else:
            plan.append("4) Speaking pace is good — keep practicing.")

        if self.clarity_db < 10:
            plan.append("5) Audio clarity: use quieter room, reduce distance from mic.")
        else:
            plan.append("5) Clarity is good — keep the same recording setup.")

        plan.append("6) Track progress weekly by recording 2 short samples & comparing metrics.")
        plan.append("7) Add rewards like stickers or praise to keep motivation high.")

        return plan


    #-----------Header Building Function----------------- 
    def build_header(self, styles):
        header = []

        # Logo (simple, no box, no padding)
        if os.path.exists(LOGO_PATH):
            logo = Image(LOGO_PATH, width=30*mm, height=30*mm)
        else:
            logo = Spacer(30*mm, 30*mm)

        brand = Paragraph(
            f"""
            <para>
            <b><font size="18">{BRAND_NAME}</font></b><br/>
            <b><font size="10" color="#555555">{WEBSITE_URL}</font></b>
            </para>
            """,
            styles["Normal"]
        )

        header_table = Table(
          [[logo, brand]],
          colWidths=[32*mm, 128*mm]
        )

        header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))

        header.append(header_table)
        header.append(Spacer(1, 8))

        # Child info (bold + clean)
        info = Paragraph(
        f"""
        <para>
        <font size="11">
        <b>Child:</b> <b>{self.child_name}</b>
        &nbsp;&nbsp; | &nbsp;&nbsp;
        <b>Date:</b> <b>{datetime.date.today()}</b>
        </font>
        </para>
        """,
        styles["Normal"]
       )

        header.append(info)
        header.append(Spacer(1, 14))

        return header



    # -----------------------------------------------------
    # Parent Friendly Scores
    # -----------------------------------------------------
    def parent_scores(self):
       clarity = min(max((self.clarity_db / 40) * 100, 0), 100)
       confidence = min(max((self.energy / 0.10) * 100 * (1 - self.pauses_ratio), 0), 100)
       pace = 100 - min(abs(self.words_per_minute - 120), 60) * (100 / 60)
       expression = min(max((self.pitch_std / 80) * 100, 0), 100)

       overall = (clarity + confidence + pace + expression) / 4
       star_score = round((overall / 100) * 5, 1)

       return {
        "clarity": round(clarity),
        "confidence": round(confidence),
        "pace": round(pace),
        "expression": round(expression),
        "overall": round(overall),
        "stars": star_score
    }

        # -----------------------------------------------------
    # Parent Score Table (Visual)
    # -----------------------------------------------------
    def parent_score_table(self):
        scores = self.parent_scores()

        data = [
            ["Skill", "Score"],
            ["Clarity", f"{scores['clarity']} / 100"],
            ["Confidence", f"{scores['confidence']} / 100"],
            ["Speaking Pace", f"{scores['pace']} / 100"],
            ["Expression", f"{scores['expression']} / 100"],
            ["Overall", f"{scores['overall']} / 100  ⭐ {scores['stars']} / 5"],
        ]

        table = Table(data, colWidths=[70*mm, 70*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F4A261")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
            ("TOPPADDING", (0,0), (-1,0), 8),
        ]))

        return table

        # -----------------------------------------------------
        # Parent Friendly Summary
        # -----------------------------------------------------
    def parent_summary(self):
       scores = self.parent_scores()
       summary = []

       # Overall level
       if scores["overall"] >= 80:
        level = "Excellent progress"
       elif scores["overall"] >= 60:
        level = "Good progress"
       else:
        level = "Developing skills"

       summary.append(
        f"<b>{self.child_name}</b> participated actively in today’s speaking activity. "
        f"Our AI analysis indicates <b>{level}</b> in overall speech development, "
        "appropriate for their age."
    )

       # Clarity
       if scores["clarity"] >= 70:
        summary.append("• Your child speaks clearly and is easy to understand.")
       else:
        summary.append("• Your child is developing clearer speech with practice.")

       # Confidence
       if scores["confidence"] >= 70:
        summary.append("• Your child shows good confidence while speaking.")
       else:
        summary.append("• Speaking confidence is emerging and improving gradually.")

       # Pace
       if scores["pace"] >= 70:
        summary.append("• Speaking pace is comfortable and listener-friendly.")
       else:
        summary.append("• Speaking pace is developing and will improve with guided practice.")

       # Expression
       if scores["expression"] >= 70:
        summary.append("• Voice expression is natural and engaging.")
       else:
        summary.append("• Expressive voice skills are developing with age.")

       # Closing reassurance
       summary.append(
        "Overall, this session reflects healthy communication growth. "
        "With regular encouragement, simple speaking games, and positive reinforcement at home, "
        "your child will continue to build strong speech confidence."
        )

       return summary


        # -----------------------------------------------------
    # Parent Highlights (Strength-Focused)
    # -----------------------------------------------------
    def parent_highlights(self):
        highlights = []

        highlights.append("✔ Shows willingness to speak and participate.")
        highlights.append("✔ Demonstrates age-appropriate speech clarity.")
        highlights.append("✔ Responds well to guided speaking activities.")
        highlights.append("✔ Learning to express thoughts with confidence.")

        return highlights
    
        # -----------------------------------------------------
    # Gentle Tips for Parents (Non-Clinical)
    # -----------------------------------------------------
    def parent_tips(self):
        return [
            "• Encourage your child to talk freely about their day without correction.",
            "• Read short stories together and ask fun questions.",
            "• Praise effort and confidence, not just clear words.",
            "• Keep speaking activities playful and pressure-free.",
        ]

    # -----------------------------------------------------
    # Parent Report PDF
    # -----------------------------------------------------
    def generate_parent_report(self):
        pdf_path = os.path.join(
            OUTPUT_DIR, f"{self.child_name}_Parent_Report.pdf"
        )

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        styles["Normal"].fontSize = 11
        styles["Normal"].leading = 14

        styles["Heading3"].fontSize = 13
        styles["Heading3"].leading = 16

        styles["Title"].fontSize = 18
        styles["Title"].leading = 22

        story = []

        # Banner + Header (reuse existing)
        story.append(ColorBanner(170 * mm))
        story.append(Spacer(1, 6))

        for item in self.build_header(styles):
            story.append(item)

        # Friendly Title
        story.append(Spacer(1, 10))
        story.append(
            Paragraph(
                "<b>Parent Speech Progress Report</b>",
                styles["Title"]
            )
        )
        story.append(Spacer(1, 10))

        # Summary Section
        story.append(Paragraph("<b>Session Summary</b>", styles["Heading3"]))
        for line in self.parent_summary():
            story.append(Paragraph(line, styles["Normal"]))

        story.append(Spacer(1, 10))

        # ===== SCORE SECTION =====
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Speech Skill Scores</b>", styles["Heading3"]))
        story.append(Spacer(1, 6))
        story.append(self.parent_score_table())


        # Highlights Section
        story.append(Paragraph("<b>Positive Highlights</b>", styles["Heading3"]))
        for h in self.parent_highlights():
            story.append(Paragraph(h, styles["Normal"]))

        story.append(Spacer(1, 14))

        # ===== Parent Tips =====
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Helpful Tips for Parents</b>", styles["Heading3"]))

        for tip in self.parent_tips():
            story.append(Paragraph(tip, styles["Normal"]))

        # Closing Trust Message
        story.append(
            Paragraph(
                "<b>Why Babblebunch AI?</b><br/>"
                "Our AI-guided speech sessions are designed to gently nurture confidence, "
                "clarity, and expressive communication in children through structured play "
                "and expert-backed techniques.",
                styles["Normal"]
            )
        )

        doc.build(
           story,
           onFirstPage=self.draw_page_decorations,
           onLaterPages=self.draw_page_decorations
             )
               
        print("✅ Parent Report Saved:", pdf_path)


    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------
    def generate_pdf(self):
        pdf_path = os.path.join(OUTPUT_DIR, f"{self.child_name}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)

        styles = getSampleStyleSheet()
        styles["Normal"].fontSize = 11
        styles["Normal"].leading = 14

        styles["Heading3"].fontSize = 13
        styles["Heading3"].leading = 16

        styles["Title"].fontSize = 18
        styles["Title"].leading = 22
        story = []

        story.append(ColorBanner(170 * mm))
        story.append(Spacer(1, 6))

        for item in self.build_header(styles):
         story.append(item)

        chart_bytes = self.radar_chart()
        story.append(Image(chart_bytes, width=160*mm, height=110*mm))
        story.append(Spacer(1, 15))

        # ===== METRICS TABLE =====
        story.append(Paragraph("<b>Speech Analysis Metrics</b>", styles["Heading3"]))
        data = [["Metric", "Value", "Range", "Status"]] + self.metric_rows()

        table = Table(data, colWidths=[60*mm, 30*mm, 45*mm, 35*mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0B3D91")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

        # ===== AI OBSERVATIONS =====
        story.append(Paragraph("<b>AI Observations</b>", styles["Heading3"]))
        for x in self.observations():
            story.append(Paragraph(x, styles["Normal"]))

        story.append(Spacer(1, 8)) 
        story.append(Paragraph("<b>Improvement Plan</b>", styles["Heading3"]))
        for line in self.plan():
            story.append(Paragraph("• " + line, styles["Normal"]))

        doc.build(
           story,
           onFirstPage=self.draw_page_decorations,
           onLaterPages=self.draw_page_decorations
             )

        print("✅ Saved:", pdf_path)
        try:
            os.startfile(pdf_path)
        except Exception:
            pass
    # -----------------------------------------------------
# Page Background (Full Page)
# -----------------------------------------------------
    def draw_page_background(self, canvas, doc):
      canvas.saveState()

      canvas.setFillColor(colors.HexColor("#FFFFFF"))  # light pastel bg
      canvas.rect(
        0,
        0,
        A4[0],
        A4[1],
        stroke=0,
        fill=1
    )

      canvas.restoreState()



    # -----------------------------------------------------
    # Common Footer (Used by all reports)
    # -----------------------------------------------------
    # -----------------------------------------------------
     # Page Background + Footer (Combined)
    # -----------------------------------------------------
    def draw_page_decorations(self, canvas, doc):
       canvas.saveState()

       # Background
       canvas.setFillColor(colors.white)
       canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)

       # Border
       canvas.setStrokeColor(colors.HexColor("#DADADA"))
       canvas.setLineWidth(1)
       canvas.rect(
        12, 12,
        A4[0] - 24,
        A4[1] - 24,
        stroke=1,
        fill=0
        )

       canvas.restoreState()

       # Footer (moved UP)
       canvas.saveState()
       canvas.setFont("Helvetica", 9)
       canvas.setFillColor(colors.grey)

       canvas.drawCentredString(
        A4[0] / 2,
        30,
        "Generated by Babblebunch AI"
       )
       canvas.drawCentredString(
        A4[0] / 2,
        18,
        "www.babblebunchai.com • Building confident young communicators"
        )

       canvas.restoreState()

    # ------Border-------
    def draw_page_border(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#E0E0E0"))
        canvas.setLineWidth(1)

        canvas.rect(
            15, 15,
            A4[0] - 30,
            A4[1] - 30,
            stroke=1,
            fill=0
        )

        canvas.restoreState()

# =========================================================
# RUN FOR ALL MP3
# =========================================================
if __name__ == "__main__":
    folder = r"C:\Users\Isha Arora\OneDrive\Desktop"
    mp3s = glob.glob(os.path.join(folder, "*.mp3"))

    if not mp3s:
        print("⚠️ No MP3 files found!")

    for audio in mp3s:
        print("\nProcessing:", audio)

        sf = SpeechFeedback(audio)
        sf.generate_pdf()                 # Team / Internal Report
        try:
           sf.generate_parent_report()
        except Exception as e:
           print("⚠️ Parent report failed:", e)     # Parent Report




