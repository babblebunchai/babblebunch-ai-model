import os
import re
import io
import math
import datetime
import numpy as np
import librosa
import whisper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch


# =========================================================
# CONFIG
# =========================================================
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Whisper model
WHISPER_MODEL = whisper.load_model("medium")

# Pronunciation helper model
WAV2VEC_MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


# =========================================================
# BANNER
# =========================================================
class ColorBanner(Flowable):
    def __init__(self, width, height=12 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        cols = ["#ffb3c7", "#ffd480", "#b3e6ff", "#c3ffa6"]
        w = self.width / len(cols)
        x = 0
        for col in cols:
            c.setFillColor(colors.HexColor(col))
            c.rect(x, 0, w, self.height, stroke=0, fill=1)
            x += w


# =========================================================
# MAIN CLASS
# =========================================================
class SpeechFeedback:
    def __init__(self, audio_path, child_name):
        self.audio_path = audio_path
        self.child_name = child_name
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        self.y, self.sr = librosa.load(audio_path, sr=None)

        # Limit to first 20 sec for stable evaluation
        if len(self.y) / self.sr > 20:
            self.y = self.y[:int(self.sr * 20)]

        # -------------------------
        # Raw Audio Metrics
        # -------------------------
        self.energy = 0.0
        self.pitch_mean = 0.0
        self.pitch_std = 0.0
        self.pauses_ratio = 0.0
        self.duration = 0.0
        self.words_per_minute = 0.0
        self.clarity_db = 0.0
        self.smoothness = 0.0

        # -------------------------
        # Transcript / Language Metrics
        # -------------------------
        self.transcript = ""
        self.cleaned_transcript = ""
        self.grammar_score = 0
        self.fluency_score = 0
        self.pronunciation = 0
        self.pause_count = 0
        self.avg_pause = 0
        self.filler_count = 0
        self.vocab_score = 0
        self.avg_sentence_length = 0
        self.word_count = 0
        self.unique_word_count = 0
        self.transcript_quality = 0

        # -------------------------
        # Unified Score Cache
        # -------------------------
        self.score_cache = {}

        self.analyze_audio()
        self.run_ai_analysis()
        self.compute_all_scores()

    # =========================================================
    # HELPERS
    # =========================================================
    def normalize(self, value, min_val, max_val):
        if max_val <= min_val:
            return 0.0
        score = (value - min_val) / (max_val - min_val)
        return max(0.0, min(score, 1.0))

    def normalize_score(self, value, min_val, max_val):
        return round(self.normalize(value, min_val, max_val) * 100, 2)

    def clamp_score(self, value, min_score=0, max_score=100):
        return round(max(min_score, min(value, max_score)), 2)

    def safe_text(self, text):
        if not text or not text.strip():
            return "No clear transcript detected from the speech sample."
        return text.encode("latin-1", "replace").decode("latin-1")

    def add_page_border(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9DDE8"))
        canvas.setLineWidth(1)
        canvas.rect(10 * mm, 10 * mm, A4[0] - 20 * mm, A4[1] - 20 * mm)
        canvas.restoreState()

    def clean_text(self, text):
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s\.\!\?']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def transcript_is_weak(self):
        return (
            self.word_count < 4 or
            self.transcript_quality < 35
        )

    def transcript_quality_score(self, text):
        if not text.strip():
            return 0

        words = text.split()
        if not words:
            return 0

        alpha_words = [w for w in words if re.match(r"^[a-zA-Z']+$", w)]
        alpha_ratio = len(alpha_words) / max(len(words), 1)

        avg_len = np.mean([len(w) for w in words]) if words else 0
        repetition_ratio = len(set(words)) / max(len(words), 1)

        score = (
            alpha_ratio * 45 +
            min(avg_len / 5, 1) * 20 +
            repetition_ratio * 35
        )
        return round(max(0, min(score, 100)), 2)
    
    def reliability_score(self):
        """
        Reliability reflects how trustworthy the current report is,
        based on transcript quality, duration, word count, clarity,
        and whether the sample is repetitive / memorized.
        """
        score = 0

        score += self.normalize(self.transcript_quality, 30, 85) * 35
        score += self.normalize(self.duration, 5, 15) * 20
        score += self.normalize(self.word_count, 5, 35) * 20
        score += self.normalize(self.clarity_db, 5, 25) * 15

        # Penalty if repetitive / rhyme-like sample
        if self.is_repetitive_sample():
            score -= 15

        if self.transcript_is_weak():
            score -= 10

        return round(max(10, min(score, 95)))

    def reliability_note(self):
        r = self.reliability_score()

        if self.is_repetitive_sample():
            return (
                "This speech sample appears to include repetitive or patterned speech "
                "(such as a rhyme, chant, or memorized phrase), so language-based results "
                "should be interpreted with extra caution."
            )

        if r >= 75:
            return (
                "This speech sample was clear enough and contained enough spoken content "
                "to provide a fairly reliable AI-assisted communication snapshot."
            )
        elif r >= 50:
            return (
                "This speech sample provided a usable speech snapshot, but some results "
                "should still be interpreted with moderate caution."
            )
        else:
            return (
                "This speech sample was short, unclear, or limited in spoken content, "
                "so the report should be treated as a preliminary screening snapshot only."
            )
        
    def reliability_label(self):
        r = self.reliability_score()
        if r >= 75:
            return "High Reliability"
        elif r >= 50:
            return "Moderate Reliability"
        else:
            return "Low Reliability"

    def reliability_note(self):
        r = self.reliability_score()

        if r >= 75:
            return (
                "This speech sample was clear enough and contained enough spoken content "
                "to provide a fairly reliable AI-assisted communication snapshot."
            )
        elif r >= 50:
            return (
                "This speech sample provided a usable speech snapshot, but some results "
                "should still be interpreted with moderate caution."
            )
        else:
            return (
                "This speech sample was short, unclear, or limited in spoken content, "
                "so the report should be treated as a preliminary screening snapshot only."
            )

    def score_band(self, score):
        if score >= 85:
            return "Advanced"
        elif score >= 70:
            return "Strong"
        elif score >= 55:
            return "Developing"
        elif score >= 40:
            return "Emerging"
        else:
            return "Needs Support"

    def medical_disclaimer(self):
        return (
            "This report is an AI-assisted communication snapshot and is intended for "
            "screening, progress tracking, and guided support only. It is not a medical "
            "or clinical diagnosis."
        )

    # =========================================================
    # AUDIO ANALYSIS
    # =========================================================
    def analyze_audio(self):
        self.duration = len(self.y) / self.sr

        # Energy
        try:
            self.energy = float(np.mean(librosa.feature.rms(y=self.y)))
        except:
            self.energy = 0

        # Pauses + Smoothness
        try:
            intervals = librosa.effects.split(self.y, top_db=25)
            voiced = sum(e - s for s, e in intervals)
            self.pauses_ratio = 1 - (voiced / len(self.y))
            self.smoothness = round(max(0, min(1 - self.pauses_ratio, 1)), 2)
        except:
            self.pauses_ratio = 0
            self.smoothness = 0

        # Pitch
        try:
            f0 = librosa.yin(self.y, fmin=80, fmax=400, sr=self.sr)
            f0 = f0[np.isfinite(f0)]
            f0 = f0[(f0 > 50) & (f0 < 500)]

            self.pitch_mean = float(np.mean(f0)) if len(f0) else 0
            self.pitch_std = float(np.std(f0)) if len(f0) else 0
        except Exception as e:
            print("Pitch analysis error:", e)
            self.pitch_mean = 0
            self.pitch_std = 0

        # Clarity proxy
        try:
            rms = librosa.feature.rms(y=self.y)[0]
            noise = np.mean(np.sort(rms)[:max(5, len(rms)//10)])
            signal = np.mean(rms)
            ratio = (signal + 1e-9) / (noise + 1e-9)
            self.clarity_db = min(40, max(0, 10 * np.log10(ratio)))
        except:
            self.clarity_db = 0

    # =========================================================
    # AI ANALYSIS
    # =========================================================
    def run_ai_analysis(self):
        result = {}

        # -------------------------
        # Whisper Transcription
        # -------------------------
        try:
            result = WHISPER_MODEL.transcribe(
                self.audio_path,
                word_timestamps=True,
                language="en"
            )
            self.transcript = result.get("text", "").strip()
            self.cleaned_transcript = self.clean_text(self.transcript)
        except Exception as e:
            print("Whisper transcription error:", e)
            self.transcript = ""
            self.cleaned_transcript = ""

        # -------------------------
        # Pause Detection
        # -------------------------
        try:
            segments = result.get("segments", [])
            prev_end = 0
            pauses = []

            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)

                if start - prev_end > 0.5:
                    pauses.append(start - prev_end)

                prev_end = end

            self.pause_count = len(pauses)
            self.avg_pause = round(float(np.mean(pauses)), 2) if pauses else 0
        except:
            self.pause_count = 0
            self.avg_pause = 0

        # -------------------------
        # Text Analysis
        # -------------------------
        try:
            text = self.cleaned_transcript
            words = [w for w in text.split() if w.strip()]
            self.word_count = len(words)
            self.unique_word_count = len(set(words)) if words else 0

            fillers = ["um", "uh", "like", "you know", "hmm", "aaa", "mmm"]
            self.filler_count = sum(text.count(f) for f in fillers)

            # Vocabulary ratio
            if self.word_count:
                self.vocab_score = round((self.unique_word_count / self.word_count) * 100, 2)
            else:
                self.vocab_score = 0

            # Sentence length
            sentences = [
                s.strip() for s in text.replace("?", ".").replace("!", ".").split(".")
                if s.strip()
            ]
            lengths = [len(s.split()) for s in sentences]
            self.avg_sentence_length = round(float(np.mean(lengths)), 2) if lengths else 0

            # Transcript quality
            self.transcript_quality = self.transcript_quality_score(text)

            # WPM based on actual transcript
            if self.duration > 0:
                self.words_per_minute = round((self.word_count / self.duration) * 60, 2)
            else:
                self.words_per_minute = 0

            # Grammar score
            self.grammar_score = self.compute_grammar_score()

            # Fluency score
            self.fluency_score = self.compute_fluency_score()

        except Exception as e:
            print("AI text analysis error:", e)
            self.grammar_score = 0
            self.fluency_score = 0
            self.vocab_score = 0
            self.filler_count = 0
            self.avg_sentence_length = 0
            self.word_count = 0
            self.unique_word_count = 0
            self.transcript_quality = 0

        # -------------------------
        # Pronunciation
        # -------------------------
        try:
            self.pronunciation = self.pronunciation_score()
        except Exception as e:
            print("Pronunciation error:", e)
            self.pronunciation = 35

    # =========================================================
    # GRAMMAR / FLUENCY / PRONUNCIATION
    # =========================================================
    def compute_grammar_score(self):
        if self.word_count == 0:
            return 0

        sentence_part = self.normalize(self.avg_sentence_length, 3, 10) * 100
        vocab_part = self.normalize(self.vocab_score, 28, 78) * 100
        transcript_part = self.normalize(self.transcript_quality, 35, 85) * 100

        grammar_raw = (
            sentence_part * 0.40 +
            vocab_part * 0.35 +
            transcript_part * 0.25
        )

        if self.word_count < 6:
            grammar_raw -= 22
        elif self.word_count < 12:
            grammar_raw -= 12
        elif self.word_count < 20:
            grammar_raw -= 6

        if self.filler_count >= 4:
            grammar_raw -= 6

        # NEW: repetitive sample penalty
        if self.is_repetitive_sample():
            grammar_raw -= 10

        return round(max(10, min(grammar_raw, 90)))

    def compute_fluency_score(self):
        if self.word_count == 0:
            return 0

        pace_score = self.score_pace()
        smoothness_score = self.normalize(self.smoothness, 0.45, 0.95) * 100

        pause_penalty = self.pause_count * 4
        filler_penalty = self.filler_count * 4
        avg_pause_penalty = max(0, (self.avg_pause - 1.2) * 10)

        fluency_raw = (
            smoothness_score * 0.45 +
            pace_score * 0.30 +
            (100 - min(pause_penalty + filler_penalty + avg_pause_penalty, 50)) * 0.25
        )

        if self.word_count < 6:
            fluency_raw -= 18
        elif self.word_count < 12:
            fluency_raw -= 10

        return round(max(10, min(fluency_raw, 95)))

    def pronunciation_score(self):
        try:
            if not self.cleaned_transcript or len(self.cleaned_transcript.split()) < 2:
                return 35

            y16k = librosa.resample(self.y, orig_sr=self.sr, target_sr=16000)

            input_values = WAV2VEC_PROCESSOR(
                y16k,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values

            with torch.no_grad():
                logits = WAV2VEC_MODEL(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted = WAV2VEC_PROCESSOR.batch_decode(predicted_ids)[0].lower()
            predicted = self.clean_text(predicted)

            original_words = set(self.cleaned_transcript.split())
            predicted_words = set(predicted.split())

            overlap = len(original_words & predicted_words)
            total = max(len(original_words), 1)

            score = (overlap / total) * 100
            adjusted = (score * 0.65) + 28

            # Penalize low transcript quality slightly
            if self.transcript_quality < 35:
                adjusted -= 8

            return round(max(20, min(adjusted, 92)), 2)

        except Exception as e:
            print("Pronunciation score error:", e)
            return 35

    # =========================================================
    # CORE UNIFIED SCORES (SINGLE SOURCE OF TRUTH)
    # =========================================================
    def score_clarity(self):
        clarity_part = self.normalize(self.clarity_db, 6, 28) * 100
        pron_part = self.normalize(self.pronunciation, 25, 85) * 100
        transcript_part = self.normalize(self.transcript_quality, 35, 85) * 100

        score = (
            clarity_part * 0.45 +
            pron_part * 0.35 +
            transcript_part * 0.20
        )
        return round(max(10, min(score, 92)))

    def score_confidence(self):
        """
        Confidence should reflect speaking energy + flow,
        but should not over-reward loud or fast speech alone.
        """
        energy_part = self.normalize(self.energy, 0.015, 0.065) * 100
        pause_part = (1 - self.normalize(self.pauses_ratio, 0.12, 0.50)) * 100
        pace_part = self.normalize(self.words_per_minute, 65, 125) * 100

        score = (
            energy_part * 0.38 +
            pause_part * 0.42 +
            pace_part * 0.20
        )

        if self.word_count < 6:
            score -= 10
        elif self.word_count < 12:
            score -= 5

        if self.transcript_is_weak():
            score -= 8

        return round(max(15, min(score, 90)))

    def score_pace(self):
        ideal_low, ideal_high = 70, 130

        if self.words_per_minute < ideal_low:
            return max(0, 100 - (ideal_low - self.words_per_minute) * 2)
        elif self.words_per_minute > ideal_high:
            return max(0, 100 - (self.words_per_minute - ideal_high) * 2)
        else:
            return 100

    def score_expression(self):
        """
        Expression should reflect healthy vocal variation,
        but should NOT easily become 100 for rhyme / chant / memorized samples.
        """
        pitch_part = self.normalize(self.pitch_std, 18, 95) * 100

        reliability_penalty = 0
        if self.transcript_is_weak():
            reliability_penalty += 10

        if self.duration < 5:
            reliability_penalty += 8
        elif self.duration < 8:
            reliability_penalty += 4

        # NEW: repetitive rhyme / chant penalty
        if self.is_repetitive_sample():
            reliability_penalty += 18

        score = pitch_part - reliability_penalty
        return round(max(18, min(score, 85)))
    
    def repetition_ratio(self):
        """
        Detect how repetitive the transcript is.
        High repetition often means rhyme, memorized phrase,
        chanting, or limited spontaneous language.
        """
        words = self.cleaned_transcript.split()
        if not words:
            return 1.0

        unique = len(set(words))
        total = len(words)

        return round(1 - (unique / total), 2)

    def is_repetitive_sample(self):
        """
        Detect likely rhyme / repetitive / memorized speech.
        """
        rep = self.repetition_ratio()

        repeated_phrase_markers = [
            "one finger", "two finger", "three finger",
            "tap tap", "cut cut", "bend bend",
            "la la", "na na", "baby shark"
        ]

        marker_hit = any(marker in self.cleaned_transcript for marker in repeated_phrase_markers)

        return rep >= 0.45 or marker_hit
    
    def score_vocabulary(self):
        """
        Vocabulary should reward variety and sentence richness,
        but should be careful with short, weak, or repetitive samples.
        """
        vocab_part = self.normalize(self.vocab_score, 30, 80) * 100
        sentence_part = self.normalize(self.avg_sentence_length, 3.5, 9) * 100
        transcript_part = self.normalize(self.transcript_quality, 40, 85) * 100

        score = (
            vocab_part * 0.42 +
            sentence_part * 0.28 +
            transcript_part * 0.30
        )

        if self.word_count < 6:
            score -= 18
        elif self.word_count < 12:
            score -= 10
        elif self.word_count < 20:
            score -= 5

        # NEW: repetitive sample penalty
        if self.is_repetitive_sample():
            score -= 12

        return round(max(12, min(score, 85)))
    
    def compute_all_scores(self):
        self.score_cache = {
            "clarity": self.score_clarity(),
            "confidence": self.score_confidence(),
            "fluency": round(self.fluency_score),
            "pronunciation": round(self.pronunciation),
            "expression": self.score_expression(),
            "grammar": round(self.grammar_score),
            "vocabulary": self.score_vocabulary(),
        }

    # =========================================================
    # FINAL SCORE
    # =========================================================
    def overall_score(self):
        s = self.score_cache

        base = (
            s["clarity"] * 0.19 +
            s["confidence"] * 0.14 +
            s["fluency"] * 0.17 +
            s["pronunciation"] * 0.16 +
            s["expression"] * 0.08 +
            s["grammar"] * 0.13 +
            s["vocabulary"] * 0.13
        )

        # Hard penalties to avoid fake high reports
        if s["fluency"] < 30:
            base -= 12
        if s["grammar"] < 30:
            base -= 10
        if s["pronunciation"] < 35:
            base -= 8
        if s["clarity"] < 35:
            base -= 8
        if self.transcript_is_weak():
            base -= 10
        if self.duration < 5:
            base -= 8

        return round(max(18, min(base, 94)))

    def rating_out_of_5(self):
        return round((self.overall_score() / 100) * 5, 1)

    # =========================================================
    # SUMMARIES
    # =========================================================
    def ai_summary(self):
        score = self.overall_score()
        reliability = self.reliability_label()

        if score >= 85:
            summary = (
                f"{self.child_name} demonstrates strong speaking ability with clear speech, "
                f"good confidence, and effective expressive communication across this sample."
            )
        elif score >= 70:
            summary = (
                f"{self.child_name} shows a developing speaking profile with several visible strengths, "
                f"along with a few areas that can improve through guided speaking practice."
            )
        elif score >= 50:
            summary = (
                f"{self.child_name} is building important speech and communication foundations and may "
                f"benefit from more support in fluency, pronunciation, and sentence-building."
            )
        else:
            summary = (
                f"{self.child_name} is still developing early speech foundations and may benefit from "
                f"structured support in clarity, confidence, and spoken language development."
            )

        return f"{summary} Reliability of this sample: {reliability}."

    def parent_dynamic_summary(self):
        s = self.score_cache
        reliability = self.reliability_label()

        if s["clarity"] >= 75 and s["pronunciation"] >= 60:
            clarity_text = "mostly clear and understandable speech"
        elif s["clarity"] >= 55:
            clarity_text = "speech that is understandable in parts but still needs clearer word production"
        else:
            clarity_text = "speech that currently needs more support for clarity and pronunciation"

        if s["confidence"] >= 75:
            confidence_text = "good speaking confidence"
        elif s["confidence"] >= 55:
            confidence_text = "developing confidence while speaking"
        else:
            confidence_text = "hesitation and lower speaking confidence"

        if s["fluency"] >= 75:
            fluency_text = "fairly smooth speech flow"
        elif s["fluency"] >= 55:
            fluency_text = "some natural flow with occasional pauses"
        else:
            fluency_text = "frequent pauses that affect smooth speaking"

        if s["expression"] >= 70:
            expression_text = "good vocal expression"
        elif s["expression"] >= 50:
            expression_text = "some expressive variation"
        else:
            expression_text = "limited variation in speaking tone"

        if s["vocabulary"] >= 70:
            vocab_text = "a healthy mix of words for this sample"
        elif s["vocabulary"] >= 50:
            vocab_text = "developing word usage"
        else:
            vocab_text = "a limited word range in this sample"

        return (
            f"This speech review suggests that {self.child_name} currently shows {clarity_text}, "
            f"along with {confidence_text}. The sample also reflects {fluency_text} and {expression_text}. "
            f"Language use shows {vocab_text}. "
            f"Overall, this should be interpreted as a {reliability.lower()} communication snapshot."
        )

    # =========================================================
    # METRICS TABLE
    # =========================================================
    def build_metrics(self):
        return [
            ("Energy", round(self.energy, 3), 0.015, 0.08),
            ("Tempo (WPM)", round(self.words_per_minute, 2), 70, 130),
            ("Smoothness", round(self.smoothness, 2), 0.55, 1.0),
            ("Pitch Mean", round(self.pitch_mean, 2), 180, 350),
            ("Pitch Variability", round(self.pitch_std, 2), 18, 95),
            ("Duration", round(self.duration, 2), 5, 20),
            ("Clarity", round(self.clarity_db, 2), 6, 28),
            ("Pronunciation", round(self.pronunciation, 2), 40, 92),
            ("Fluency Score", round(self.fluency_score, 2), 45, 95),
            ("Grammar Score", round(self.grammar_score, 2), 40, 95),
            ("Transcript Quality", round(self.transcript_quality, 2), 35, 85),
        ]

    def get_status(self, value, mn, mx):
        if value < mn:
            return "Below target"
        elif value > mx:
            return "Above target"
        else:
            return "Within target"

    # =========================================================
    # OBSERVATIONS
    # =========================================================
    def generate_observations(self):
        obs = []

        if self.energy < 0.015:
            obs.append("The child speaks softly in this sample, which may affect audibility and perceived confidence.")
        elif self.energy > 0.08:
            obs.append("The child speaks with strong volume, though voice control can be improved.")
        else:
            obs.append("The child’s speaking volume is generally comfortable and audible.")

        if self.pauses_ratio > 0.45:
            obs.append("Frequent pauses suggest hesitation or difficulty maintaining sentence flow.")
        elif self.pauses_ratio > 0.25:
            obs.append("Some pauses are present, but overall speech flow is manageable.")
        else:
            obs.append("Speech flow is relatively smooth with minimal hesitation.")

        if self.pitch_std < 18:
            obs.append("Speech sounds fairly flat and may benefit from more expressive variation.")
        elif self.pitch_std > 95:
            obs.append("Speech shows strong vocal variation, though tone control can improve.")
        else:
            obs.append("The child uses a healthy amount of vocal expression while speaking.")

        if self.words_per_minute < 70:
            obs.append("The child speaks slowly, which may reflect hesitation or careful word production.")
        elif self.words_per_minute > 130:
            obs.append("The child speaks quickly, which may reduce clarity in some parts.")
        else:
            obs.append("The speaking pace is age-appropriate and fairly balanced.")

        if self.score_cache["clarity"] < 45:
            obs.append("Speech clarity is currently limited and may need more pronunciation-focused practice.")
        elif self.score_cache["clarity"] < 70:
            obs.append("Speech is understandable in parts, though clearer articulation would help.")
        else:
            obs.append("Most spoken words are reasonably clear and understandable.")

        return obs

    # =========================================================
    # DETAILED FEEDBACK
    # =========================================================
    def ai_detailed_feedback(self):
        feedback = []

        if self.word_count < 8:
            feedback.append("A limited amount of spoken content was detected, so the child may need encouragement to respond in longer phrases or sentences.")

        if self.avg_sentence_length < 4:
            feedback.append("The child mostly uses very short sentences and may need support building fuller responses.")

        if self.filler_count > 3:
            feedback.append("Frequent hesitation sounds or filler words may reduce fluency and speaking confidence.")

        if self.pauses_ratio > 0.4:
            feedback.append("The child pauses often while speaking, which may reflect hesitation or difficulty organizing thoughts.")

        if self.score_cache["clarity"] < 55:
            feedback.append("Pronunciation and articulation need improvement, as some words may not be coming through clearly.")

        if self.pronunciation < 40:
            feedback.append("Pronunciation accuracy appears low in this sample and may benefit from repetition-based speaking practice.")

        if self.score_cache["vocabulary"] < 45:
            feedback.append("Vocabulary variety is currently limited, suggesting the child may rely on repeated familiar words.")

        if self.pitch_std < 14:
            feedback.append("Speech sounds somewhat monotone and could benefit from more expressive speaking activities.")

        if self.transcript_is_weak():
            feedback.append("Because the speech sample was short or unclear, some language-based results should be interpreted as preliminary rather than final.")

        if not feedback:
            feedback.append("The child shows a fairly balanced speaking profile with reasonable clarity, pace, expression, and language use.")

        return " ".join(feedback)

    # =========================================================
    # IMPROVEMENT PLAN
    # =========================================================
    def improvement_plan(self):
        plan = []

        if self.words_per_minute < 70:
            plan.append("Practice storytelling for 2–3 minutes daily to build speaking flow and verbal confidence.")

        if self.pauses_ratio > 0.4:
            plan.append("Encourage the child to describe pictures or daily events in one continuous sentence before stopping.")

        if self.pitch_std < 18:
            plan.append("Use emotion-based reading activities (happy, sad, excited voice) to improve vocal expression.")

        if self.score_cache["clarity"] < 60:
            plan.append("Practice simple pronunciation drills using short words and repeated sound patterns.")

        if self.pronunciation < 45:
            plan.append("Repeat simple words and short sentences slowly and clearly, focusing on mouth movement and sound accuracy.")

        if self.avg_sentence_length < 4:
            plan.append("Ask open-ended questions at home to encourage longer and more complete responses.")

        if self.score_cache["vocabulary"] < 50:
            plan.append("Introduce 3–5 new words each week and encourage the child to use them in daily sentences.")

        if not plan:
            plan.append("Maintain current speaking progress and introduce advanced speaking activities such as storytelling, role-play, and show-and-tell.")

        return plan[:5]

    # =========================================================
    # PARENT HELPERS
    # =========================================================
    def parent_strengths(self):
        s = self.score_cache
        strengths = []

        if s["clarity"] >= 70:
            strengths.append("Speech is generally understandable in this sample.")
        if s["confidence"] >= 65:
            strengths.append("The child shows reasonable speaking confidence.")
        if s["fluency"] >= 70:
            strengths.append("Speech flow is fairly smooth with manageable pauses.")
        if s["expression"] >= 65:
            strengths.append("The child uses good voice expression while speaking.")
        if s["vocabulary"] >= 60 and not self.is_repetitive_sample():
            strengths.append("A healthy variety of words is being used in speech.")
        if self.avg_sentence_length >= 5 and not self.is_repetitive_sample():
            strengths.append("The child is beginning to form longer and more meaningful responses.")

        if self.is_repetitive_sample():
            strengths.append("The child was able to maintain rhythm and repetition during a structured speaking pattern.")

        if not strengths:
            strengths.append("The child is showing early speech development and is building a speaking foundation.")

        return strengths[:4]

    def parent_focus_areas(self):
        s = self.score_cache
        focus = []

        if s["clarity"] < 65:
            focus.append("Speech clarity can improve with more pronunciation-focused speaking practice.")
        if s["confidence"] < 60:
            focus.append("The child may need support in speaking with a stronger and more confident voice.")
        if s["fluency"] < 65:
            focus.append("Speech flow can improve by reducing hesitation and long pauses.")
        if s["expression"] < 55:
            focus.append("More vocal expression can help speech sound more natural and engaging.")
        if s["vocabulary"] < 55:
            focus.append("Word variety can be strengthened through regular language exposure.")
        if self.avg_sentence_length < 4.5:
            focus.append("Sentence building and idea expansion need more practice.")

        if not focus:
            focus.append("Current speech profile is balanced, with room for continued development and refinement.")

        return focus[:4]

    def parent_language_insights(self):
        insights = []
        insights.append(f"Words spoken in sample: {self.word_count}")
        insights.append(f"Unique words used: {self.unique_word_count}")
        insights.append(f"Average sentence length: {round(self.avg_sentence_length, 1)} words")
        insights.append(f"Average pause duration: {round(self.avg_pause, 2)} sec")
        insights.append(f"Filler words detected: {self.filler_count}")
        return insights[:5]

    def parent_home_plan(self):
        s = self.score_cache
        tips = []

        if s["clarity"] < 65:
            tips.append("Practice repeating simple words and short sentences slowly and clearly.")
        if s["fluency"] < 65:
            tips.append("Encourage your child to speak in one full sentence before stopping.")
        if s["vocabulary"] < 55:
            tips.append("Introduce 3–5 new words every week through stories and daily conversations.")
        if self.avg_sentence_length < 4.5:
            tips.append("Ask open-ended questions like 'What happened today?' and encourage full answers.")
        if s["expression"] < 55:
            tips.append("Use story reading with emotions to make speaking more expressive and lively.")
        if s["confidence"] < 60:
            tips.append("Create low-pressure speaking opportunities at home through casual conversation and play.")

        if not tips:
            tips = [
                "Continue regular storytelling and reading aloud at home.",
                "Encourage your child to describe daily experiences in full sentences.",
                "Support confident speaking by listening patiently and positively."
            ]

        return tips[:4]

    # =========================================================
    # CHARTS
    # =========================================================
    def radar_chart(self):
        s = self.score_cache

        labels = ["Clarity", "Confidence", "Fluency", "Pronunciation", "Expression", "Grammar"]
        values = [
            s["clarity"],
            s["confidence"],
            s["fluency"],
            s["pronunciation"],
            s["expression"],
            s["grammar"]
        ]

        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"])
        ax.set_ylim(0, 100)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        buf.seek(0)
        return buf

    def bar_chart(self):
        s = self.score_cache

        labels = ["Clarity", "Confidence", "Fluency", "Pronunciation", "Expression"]
        values = [
            s["clarity"],
            s["confidence"],
            s["fluency"],
            s["pronunciation"],
            s["expression"]
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values)
        ax.set_ylim(0, 100)
        ax.set_title("Key Speech Scores")
        ax.set_ylabel("Score / 100")

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2,
                f"{round(value)}",
                ha='center',
                va='bottom',
                fontsize=9
            )

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()
        buf.seek(0)
        return buf

    # =========================================================
    # FULL REPORT PDF
    # =========================================================
    def generate_pdf(self):
        path = os.path.join(OUTPUT_DIR, f"{self.child_name}_Speech_Report.pdf")

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm
        )

        styles = getSampleStyleSheet()
        story = []

        story.append(ColorBanner(180 * mm))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Babblebunch AI</b>", styles['Title']))
        story.append(Paragraph("Advanced Speech Analysis Report", styles['Heading2']))
        story.append(Paragraph("<font color='#457B9D'>https://www.babblebunchai.com</font>", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"<b>Child Name:</b> {self.child_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {self.timestamp}", styles['Normal']))
        story.append(Paragraph(f"<b>Overall Score:</b> {self.overall_score()} / 100", styles['Normal']))
        story.append(Paragraph(f"<b>Performance Band:</b> {self.score_band(self.overall_score())}", styles['Normal']))
        story.append(Paragraph(f"<b>Rating:</b> ⭐ {self.rating_out_of_5()} / 5", styles['Normal']))
        story.append(Paragraph(f"<b>Reliability:</b> {self.reliability_label()} ({self.reliability_score()} / 100)", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Transcript Snapshot</b>", styles['Heading3']))
        story.append(Paragraph(self.safe_text(self.transcript), styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>AI Summary</b>", styles['Heading3']))
        story.append(Paragraph(self.ai_summary(), styles['Normal']))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Reliability Note</b>", styles['Heading3']))
        story.append(Paragraph(self.reliability_note(), styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Image(self.radar_chart(), width=120 * mm, height=120 * mm))
        story.append(Spacer(1, 10))

        story.append(Image(self.bar_chart(), width=150 * mm, height=70 * mm))
        story.append(Spacer(1, 10))

        data = [["Metric", "Value", "Range", "Status"]]
        for name, val, mn, mx in self.build_metrics():
            data.append([name, round(val, 2), f"{mn}-{mx}", self.get_status(val, mn, mx)])

        table = Table(data, colWidths=[45 * mm, 25 * mm, 35 * mm, 35 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        story.append(table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>AI Observations</b>", styles['Heading3']))
        for o in self.generate_observations():
            story.append(Paragraph(f"• {o}", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Detailed Feedback</b>", styles['Heading3']))
        story.append(Paragraph(self.ai_detailed_feedback(), styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Improvement Plan</b>", styles['Heading3']))
        for p in self.improvement_plan():
            story.append(Paragraph(f"• {p}", styles['Normal']))

        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Important Note</b>", styles['Heading3']))
        story.append(Paragraph(self.medical_disclaimer(), styles['Normal']))

        doc.build(story, onFirstPage=self.add_page_border, onLaterPages=self.add_page_border)

        if not os.path.exists(path):
            raise Exception("Internal report PDF was not generated")

        return path

    # =========================================================
    # PARENT REPORT PDF
    # =========================================================
    def generate_parent_report(self):
        path = os.path.join(OUTPUT_DIR, f"{self.child_name}_Parent_Report.pdf")

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm
        )

        styles = getSampleStyleSheet()
        story = []

        section_style = ParagraphStyle(
            'ParentSectionStyle',
            parent=styles['Heading3'],
            textColor=colors.HexColor("#E63946"),
            spaceAfter=6
        )

        compact_style = ParagraphStyle(
            'Compact',
            parent=styles['Normal'],
            leading=14,
            spaceAfter=4
        )

        story.append(ColorBanner(180 * mm))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Babblebunch AI</b>", styles['Title']))
        story.append(Paragraph("Parent Speech Progress Report", styles['Heading2']))
        story.append(Paragraph("<font color='#457B9D'>https://www.babblebunchai.com</font>", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"<b>Child Name:</b> {self.child_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {self.timestamp}", styles['Normal']))
        story.append(Paragraph(f"<b>Overall Score:</b> {self.overall_score()} / 100", styles['Normal']))
        story.append(Paragraph(f"<b>Performance Band:</b> {self.score_band(self.overall_score())}", styles['Normal']))
        story.append(Paragraph(f"<b>Rating:</b> ⭐ {self.rating_out_of_5()} / 5", styles['Normal']))
        story.append(Paragraph(f"<b>Reliability:</b> {self.reliability_label()} ({self.reliability_score()} / 100)", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>AI Parent Summary</b>", section_style))
        story.append(Paragraph(self.parent_dynamic_summary(), compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Reliability Note</b>", section_style))
        story.append(Paragraph(self.reliability_note(), compact_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Speech Snapshot</b>", section_style))

        s = self.score_cache
        snapshot_data = [
            ["Skill Area", "Score"],
            ["Clarity", f"{s['clarity']} / 100"],
            ["Confidence", f"{s['confidence']} / 100"],
            ["Fluency", f"{s['fluency']} / 100"],
            ["Expression", f"{s['expression']} / 100"],
            ["Vocabulary", f"{s['vocabulary']} / 100"],
        ]

        snapshot_table = Table(snapshot_data, colWidths=[80 * mm, 50 * mm])
        snapshot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4A261")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))

        story.append(snapshot_table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>What Your Child Is Doing Well</b>", section_style))
        for i, item in enumerate(self.parent_strengths(), start=1):
            story.append(Paragraph(f"{i}. {item}", compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Areas to Improve</b>", section_style))
        for i, item in enumerate(self.parent_focus_areas(), start=1):
            story.append(Paragraph(f"{i}. {item}", compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Speech & Language Insights</b>", section_style))
        for item in self.parent_language_insights():
            story.append(Paragraph(f"• {item}", compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Simple AI Feedback</b>", section_style))
        story.append(Paragraph(self.ai_detailed_feedback(), compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Next Step Plan</b>", section_style))
        for i, item in enumerate(self.improvement_plan()[:3], start=1):
            story.append(Paragraph(f"{i}. {item}", compact_style))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>How Parents Can Support at Home</b>", section_style))
        for i, item in enumerate(self.parent_home_plan(), start=1):
            story.append(Paragraph(f"{i}. {item}", compact_style))

        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Important Note</b>", section_style))
        story.append(Paragraph(self.medical_disclaimer(), compact_style))

        doc.build(story, onFirstPage=self.add_page_border, onLaterPages=self.add_page_border)

        if not os.path.exists(path):
            raise Exception("Parent report PDF was not generated")

        return path


# =========================================================
# OPTIONAL TEST RUN
# =========================================================
if __name__ == "__main__":
    # Example:
    # feedback = SpeechFeedback("sample.wav", "Aarav")
    # print(feedback.generate_pdf())
    # print(feedback.generate_parent_report())
    pass