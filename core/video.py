"""video.py — sonda, rosto, legenda e corte com ffmpeg.

SRP: só renderização. Toda decisão de scoring/seleção fica fora.
"""
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
from .models import Candidate
from .config import (
    CAPTION_MAX_CHARS_PER_LINE, CAPTION_MAX_LINES_PER_CUE,
    CAPTION_MIN_DURATION, CAPTION_PAUSE_THRESHOLD, CLIPPER_FFMPEG_THREADS,
)


# -- helpers --
def sanitize_filename(text: str, fallback: str) -> str:
    text = text or fallback
    text = re.sub(r"[^\w\-áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ ]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return (text or fallback)[:60]


def _probe_dimensions(video_path: str) -> tuple[int | None, int | None]:
    if not shutil.which("ffprobe"):
        return None, None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            capture_output=True, text=True, timeout=10,
        )
        w_h = r.stdout.strip().split("x")
        if len(w_h) == 2:
            return int(w_h[0]), int(w_h[1])
    except Exception:
        pass
    return None, None


def face_center_x(video_path: str, start: float, end: float) -> float:
    try:
        import cv2
    except ImportError:
        return 0.5
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade):
        return 0.5
    fc = cv2.CascadeClassifier(cascade)
    if fc.empty():
        return 0.5
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
        dur = max(0.1, end - start)
        times = [start + dur * f for f in (0.15, 0.5, 0.85) if start <= start + dur * f < end]
        centers: list[float] = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fc.detectMultiScale(gray, 1.1, 5)
            if len(faces):
                fx, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
                centers.append((fx + fw / 2) / width)
        if not centers:
            return 0.5
        return max(0.25, min(0.75, statistics.median(centers)))
    finally:
        cap.release()


def _srt_time(t: float) -> str:
    h, m = int(t // 3600), int((t % 3600) // 60)
    s, ms = int(t % 60), int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _wrap(text: str, max_chars: int = CAPTION_MAX_CHARS_PER_LINE) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            if len(w) > max_chars:
                for i in range(0, len(w), max_chars):
                    lines.append(w[i:i + max_chars])
                cur = ""
            else:
                cur = w
    if cur:
        lines.append(cur)
    return lines or [text[:max_chars]]


def build_srt(words: list, clip_start: float) -> str:
    if not words:
        return ""
    words = sorted(words, key=lambda w: w.start)
    cues: list[list] = []
    cur: list = []

    def flush():
        if cur:
            cues.append(list(cur))
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        is_last = i == len(words) - 1
        nxt = words[i + 1] if not is_last else None
        should = is_last
        if not should and nxt:
            gap = nxt.start - w.end
            ends = w.text.strip()[-1] in ".!?" if w.text.strip() else False
            if gap > CAPTION_PAUSE_THRESHOLD:
                should = True
            elif ends and len(cur) >= 2:
                should = True
            elif len(cur) >= 8:
                should = True
            elif len(" ".join(x.text for x in cur)) > CAPTION_MAX_CHARS_PER_LINE * CAPTION_MAX_LINES_PER_CUE:
                should = True
        if should:
            flush()

    out: list[str] = []
    idx = 1
    for chunk_words in cues:
        if not chunk_words:
            continue
        s = chunk_words[0].start - clip_start
        e = chunk_words[-1].end - clip_start
        if e <= s:
            e = s + CAPTION_MIN_DURATION
        if e - s < CAPTION_MIN_DURATION:
            e = s + CAPTION_MIN_DURATION
        raw = " ".join(w.text for w in chunk_words).strip().upper()
        wrapped = _wrap(raw)
        n_chunks = math.ceil(len(wrapped) / CAPTION_MAX_LINES_PER_CUE)
        for start in range(0, len(wrapped), CAPTION_MAX_LINES_PER_CUE):
            lines = wrapped[start:start + CAPTION_MAX_LINES_PER_CUE]
            text = "\n".join(lines)
            if n_chunks > 1:
                dur = (e - s) / n_chunks
                cs = s + (start // CAPTION_MAX_LINES_PER_CUE) * dur
                ce = cs + dur
            else:
                cs, ce = s, e
            out.append(f"{idx}\n{_srt_time(max(0, cs))} --> {_srt_time(max(0, ce))}\n{text}\n")
            idx += 1
    return "\n".join(out)


def _escape_subs(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _video_encoder() -> tuple[str, list[str]]:
    """Escolhe encoder: h264_qsv (sonda funcional 1x, cacheada) senão libx264."""
    from . import intel as intel_hw
    if intel_hw.best_video_encoder() == "h264_qsv":
        return "h264_qsv", ["-global_quality", "23", "-look_ahead", "1"]
    return "libx264", ["-preset", "veryfast", "-crf", "20"]


def cut(video_path: str, c: Candidate, out_path: Path, vertical: bool, captions: bool) -> None:
    duration = c.duration
    filters: list[str] = []

    if vertical:
        fx = face_center_x(video_path, c.start, c.end)
        w, h = _probe_dimensions(video_path)
        crop_fail = bool(w and h and (w * 1920 / h) < 1080)
        if crop_fail:
            filters.append("scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(1080-iw)/2:(1920-ih)/2:color=black")
        else:
            filters.append(f"scale=-2:1920,crop=1080:1920:x=min(max(iw*{fx:.3f}-540\\,0)\\,iw-1080):y=0")
    else:
        filters.append("scale=1280:-2")

    srt_file: str | None = None
    try:
        if captions and c.words:
            srt = build_srt(c.words, c.start)
            if srt:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8")
                srt_file = f.name
                f.write(srt)
                f.close()
                esc = _escape_subs(srt_file)
                filters.append(
                    f"subtitles='{esc}':force_style='FontName=Arial Black,FontSize=16,"
                    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=90'"
                )
        vf = ",".join(filters)
        coarse = max(0, c.start - 2)
        fine = c.start - coarse
        vcodec, vextra = _video_encoder()
        # QSV sem flag -hwaccel (sonda mostrou que -hwaccel qsv quebra o init
        # em decode software; -c:v h264_qsv sozinho funciona). 1 tentativa
        # por encoder, sem retry: QSV -> libx264.
        encoders = [(vcodec, vextra)]
        if vcodec == "h264_qsv":
            encoders.append(("libx264", ["-preset", "veryfast", "-crf", "20"]))
        last_err: Exception | None = None
        for enc, extra in encoders:
            cmd = [
                "ffmpeg", "-threads", str(CLIPPER_FFMPEG_THREADS),
                "-y", "-ss", str(coarse), "-i", video_path, "-ss", str(fine),
                "-t", str(duration), "-vf", vf,
                "-c:v", enc, *extra,
                "-threads", str(CLIPPER_FFMPEG_THREADS),
                "-c:a", "aac", "-b:a", "160k", str(out_path),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=600)
                last_err = None
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                if enc == "h264_qsv":
                    print(f"   ! QSV falhou — fallback libx264 ({e.stderr.decode(errors='ignore')[:120]})")
                    continue
                raise
            except subprocess.TimeoutExpired as e:
                last_err = e
                print(f"   ! ffmpeg timeout no clipe {out_path.name} — abortando clipe")
                raise
        if last_err is not None:
            raise last_err
    finally:
        if srt_file and os.path.exists(srt_file):
            try:
                os.unlink(srt_file)
            except Exception:
                pass
