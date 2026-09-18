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


# Tokens de controle do Whisper (nunca são fala e nunca podem ir p/ legenda).
# Origem: stream de tokens do whisper.cpp (-ojf). Formas observadas: [eot],
# [sot], [_EOT_], [SOT], [TRANSCRIBE], [BLANK] etc. Token de fala real nunca
# é 100% entre colchetes (Whisper usa parênteses p/ ex. "(risos)").
_SPECIAL_TOKEN_RE = re.compile(r"\[[^\]]*\]")


def is_special_token(text: str) -> bool:
    """True se o token inteiro é um controle do Whisper ([eot], [_EOT_]...)."""
    t = (text or "").strip()
    return bool(t) and _SPECIAL_TOKEN_RE.fullmatch(t) is not None


def clean_caption_text(text: str) -> str:
    """Remove controles [...] do texto e normaliza espaços. Nunca inventa."""
    t = _SPECIAL_TOKEN_RE.sub("", text or "")
    return re.sub(r"\s+", " ", t).strip()


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


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_cues(words: list, clip_start: float, clip_end: float) -> list[tuple]:
    """Agrupa palavras em cues (s_abs, e_abs, texto_limpo). Puro e testável.

    Mesmas garantias do build_srt: só palavras dentro do corte, texto limpo,
    sem cues vazias. Tempos ainda ABSOLUTOS; a conversão p/ relativo é feita
    pelo formatador (SRT/ASS) subtraindo clip_start de forma determinística.
    """
    words = sorted(words, key=lambda w: w.start)
    words = [w for w in words if w.end > clip_start and w.start < clip_end]
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

    out: list[tuple] = []
    for chunk_words in cues:
        if not chunk_words:
            continue
        s = chunk_words[0].start
        e = chunk_words[-1].end
        if e <= s:
            e = s + CAPTION_MIN_DURATION
        if e - s < CAPTION_MIN_DURATION:
            e = s + CAPTION_MIN_DURATION
        raw = clean_caption_text(" ".join(w.text for w in chunk_words)).upper()
        if raw:
            out.append((s, e, raw))
    return out


def _split_lines(cues: list[tuple], clip_start: float, duration: float) -> list[tuple]:
    """Divide cues longas em (cs_rel, ce_rel, texto) já relativos e clampados."""
    out: list[tuple] = []
    for s_abs, e_abs, raw in cues:
        s = max(0.0, s_abs - clip_start)
        e = min(duration, max(e_abs - clip_start, s + CAPTION_MIN_DURATION))
        if e <= s:
            continue
        wrapped = _wrap(raw)
        n_chunks = math.ceil(len(wrapped) / CAPTION_MAX_LINES_PER_CUE)
        for start in range(0, len(wrapped), CAPTION_MAX_LINES_PER_CUE):
            lines = wrapped[start:start + CAPTION_MAX_LINES_PER_CUE]
            text = "\n".join(lines)
            if n_chunks > 1:
                dur = (e - s) / n_chunks
                cs = s + (start // CAPTION_MAX_LINES_PER_CUE) * dur
                ce = min(duration, cs + dur)
            else:
                cs, ce = s, e
            if ce > cs:
                out.append((cs, ce, text))
    return out


def build_srt(words: list, clip_start: float, clip_end: float | None = None) -> str:
    """Gera SRT com tempos RELATIVOS ao corte (clip_start → 0).

    Garantias determinísticas:
    - palavras fora de [clip_start, clip_end] são descartadas;
    - cues são clampadas em [0, duration] (nunca negativas, nunca além do fim);
    - cues vazias (sem texto após limpeza) são descartadas;
    - tokens especiais ([eot], [sot], ...) nunca viram texto visível.
    """
    if not words:
        return ""
    if clip_end is None:
        clip_end = max((w.end for w in words), default=clip_start)
    duration = max(0.0, clip_end - clip_start)
    if duration <= 0:
        return ""
    cues = _split_lines(_group_cues(words, clip_start, clip_end), clip_start, duration)
    out = [f"{i}\n{_srt_time(cs)} --> {_srt_time(ce)}\n{text}\n"
           for i, (cs, ce, text) in enumerate(cues, start=1)]
    return "\n".join(out)


# Estilo Shorts/Reels/TikTok com fontes PRESENTES no sistema (Arial Black não
# existe aqui — caía p/ Noto Regular fino). Liberation Sans Bold resolve via
# fontconfig. Tamanhos em pixels do vídeo (PlayRes = dimensões de saída).
CAPTION_FONT = "Liberation Sans"
CAPTION_FONT_SIZE_VERTICAL = 54
CAPTION_FONT_SIZE_WIDE = 44
CAPTION_MARGIN_V = 140


def build_ass(words: list, clip_start: float, clip_end: float,
              width: int, height: int, font_size: int = CAPTION_FONT_SIZE_VERTICAL) -> str:
    """Gera ASS com PlayRes = dimensões REAIS de saída.

    Motivo: sem PlayRes explícito o libass assume 384x288 e escala o estilo
    de forma anamórfica (fonte gigante/esticada, MarginV fora da base). Com
    PlayRes == frame, FontSize/MarginV valem pixels do vídeo, determinístico.
    Mesmas garantias de tempo/texto do build_srt (mesmo núcleo).
    """
    duration = max(0.0, clip_end - clip_start)
    cues = _split_lines(_group_cues(words, clip_start, clip_end), clip_start, duration) \
        if duration > 0 else []
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
         "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
         "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
         "MarginL, MarginR, MarginV, Encoding"),
        (f"Style: Clip,{CAPTION_FONT}, {font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
         f"-1,0,0,0,100,100,0,0,1,3,1,2,40,40,{CAPTION_MARGIN_V},1"),
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
         "Effect, Text"),
    ]
    for cs, ce, text in cues:
        text = text.replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_ass_time(cs)},{_ass_time(ce)},Clip,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


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
        out_w, out_h, font_size = 1080, 1920, CAPTION_FONT_SIZE_VERTICAL
    else:
        filters.append("scale=1280:-2")
        out_w, out_h, font_size = None, None, CAPTION_FONT_SIZE_WIDE
        w, h = _probe_dimensions(video_path)
        if w and h:
            out_w = 1280
            out_h = max(2, (int(h * 1280 / w) // 2) * 2)

    srt_file: str | None = None
    try:
        # Corte via trim/atrim (NÃO via -ss após -i): medição ponta a ponta
        # provou que -ss de saída desloca a linha do tempo das legendas em
        # exatamente -fine (o filtro subtitles avalia na timeline do demux,
        # que o -ss de saída não desloca), enquanto trim+setpts/asetpts deixa
        # vídeo, áudio e legendas todos 0-based e alinhados. Sem offset mágico.
        # Ordem no grafo: escala/crop → trim+setpts → subtitles (por último).
        coarse = max(0, c.start - 2)
        fine = c.start - coarse
        end = fine + duration
        filters.append(f"trim=start={fine}:end={end},setpts=PTS-STARTPTS")
        af = f"atrim=start={fine}:end={end},asetpts=PTS-STARTPTS"
        if captions and c.words:
            # ASS com PlayRes = frame real → layout determinístico em pixels.
            # Sem dimensões conhecidas, cai no SRT legado (comportamento anterior).
            if out_w and out_h:
                subs = build_ass(c.words, c.start, c.end, out_w, out_h, font_size)
                suffix = ".ass"
            else:
                subs = build_srt(c.words, c.start, c.end)
                suffix = ".srt"
            if subs:
                f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
                srt_file = f.name
                f.write(subs)
                f.close()
                esc = _escape_subs(srt_file)
                if suffix == ".ass":
                    filters.append(f"subtitles='{esc}'")
                else:
                    filters.append(
                        f"subtitles='{esc}':force_style='FontName=Liberation Sans,Bold=1,FontSize=22,"
                        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=120'"
                    )
        vf = ",".join(filters)
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
                "-y", "-ss", str(coarse), "-i", video_path,
                "-vf", vf, "-af", af,
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
