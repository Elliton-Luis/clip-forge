"""video.py — sonda, rosto, legenda e corte com ffmpeg.

SRP: só renderização. Toda decisão de scoring/seleção fica fora.
"""
import json
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


def smart_join(texts: list[str]) -> str:
    """Junta tokens/palavras respeitando fronteiras reais (determinístico).

    Por que existe: o whisper.cpp emite tokens com espaço à esquerda
    (" direita" = início de palavra; "ita" = continuação) — " ".join cego
    produzia "dire ita". faster-whisper já vem sem espaços.
    Regra: se QUALQUER token traz espaço à esquerda, o lote é estilo
    whisper.cpp → concatena e normaliza (" direita"+"ita" = "direita");
    senão, junta com espaço (faster-whisper). Fidelidade > embelezamento.
    """
    texts = [t for t in (texts or []) if t and t.strip()]
    if not texts:
        return ""
    if any(t[0].isspace() for t in texts):
        return re.sub(r"\s+", " ", "".join(texts)).strip()
    parts: list[str] = []
    for t in texts:
        t = t.strip()
        if parts and not all(ch in ".,!?;:…%‰’”»)]}" for ch in t):
            parts.append(" ")
        parts.append(t)
    return "".join(parts)


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


def is_degenerate_word(w) -> bool:
    """True se o Word não tem intervalo temporal válido (end <= start).

    Tokens degenerados vêm do Whisper.cpp+VAD (ex: 20.760→20.760) como
    continuação alucinada pós-fala. Nunca devem virar caption: sem
    intervalo real, qualquer duração seria inventada. Palavras curtas
    mas válidas (0 < dur < CAPTION_MIN_DURATION, ex: "ah" 10.0→10.12)
    NÃO são degeneradas e continuam cobertas pela duração mínima visual.
    """
    try:
        return not (w.end > w.start)
    except Exception:
        return True


def valid_caption_words(words: list) -> list:
    """Fronteira transcrição→captions: só words com end > start."""
    return [w for w in (words or []) if not is_degenerate_word(w)]


# Modo experimental "intervals" (comparação A/B atrás de --caption-mode):
# cues por rajada de fala em vez de contagem de palavras. Pausas curtas
# (< GAP) não quebram (inclui gaps de medição no meio da palavra); rajadas
# longas partem em > MAX_DUR, sempre em fronteira de palavra. Ponto de
# partida para comparação — não default até evidência.
INTERVAL_GAP_SEC = 0.8
INTERVAL_MAX_DUR = 10.0


def group_interval_words(words: list, clip_start: float, clip_end: float) -> list[list]:
    """Agrupa por intervalos de fala (experimental). Puro e testável.

    Mesmas invariantes do modo words: só words válidas no clip, quebra só
    em fronteira de palavra (continuações fecham o grupo), vozes concorrentes
    separadas. Diferença: sem teto de 8 words/48 chars e sem quebra em pausa
    curta — a unidade é a rajada de fala, não o bloco de leitura.
    """
    words = sorted(valid_caption_words(words), key=lambda w: w.start)
    words = [w for w in words if w.end > clip_start and w.start < clip_end]
    has_boundaries = any(w.text[:1].isspace() for w in words if w.text)

    def is_continuation(w) -> bool:
        return has_boundaries and not (w.text[:1].isspace() if w.text else True)

    def ends_sentence(w) -> bool:
        return bool(w.text.strip()) and w.text.strip()[-1] in ".!?"

    cues: list[list] = []
    for voice in _split_voices(words):
        cur: list = []
        for w in voice:
            if cur:
                prev = cur[-1]
                gap = w.start - prev.end
                too_long = w.end - cur[0].start > INTERVAL_MAX_DUR
                if gap > INTERVAL_GAP_SEC or (too_long and not is_continuation(w)):
                    if too_long and gap <= INTERVAL_GAP_SEC:
                        # Rajada longa sem pausa: prefere fechar na última
                        # frase completa (pontuação não seguida de continuação).
                        cut_at = None
                        for k in range(len(cur) - 1, -1, -1):
                            nxt = cur[k + 1] if k + 1 < len(cur) else None
                            if ends_sentence(cur[k]) and (
                                    nxt is None or not is_continuation(nxt)):
                                cut_at = k
                                break
                        if cut_at is not None:
                            cues.append(cur[:cut_at + 1])
                            cur = cur[cut_at + 1:]
                    cues.append(list(cur))
                    cur = []
            cur.append(w)
        if cur:
            cues.append(list(cur))
    cues.sort(key=lambda g: (g[0].start, g[-1].end))
    return cues
# (quantização do Whisper ~10ms; medido 0 overlaps incidentais nos transcripts
# reais) e seguem sequenciais; acima disso, vozes concorrentes. Calibrado,
# documentado, testado — não é palpite.
VOICE_OVERLAP_MIN = 0.1


def _split_voices(words: list) -> list[list]:
    """Particiona words ordenadas em vozes concorrentes (puro e testável).

    Cada word entra na primeira voz cujo fim <= seu início (menos o piso);
    senão abre nova voz. Ordenação estável SÓ por start: em empates, vale a
    ordem de emissão do Whisper (tokens saem em ordem de decodificação, que
    acompanha os turnos percebidos) — sem isso, o desempate seria arbitrário.
    Timestamps intocados; sem labels de speaker.
    """
    ordered = sorted(words, key=lambda w: w.start)  # estável: preserva emissão
    voices: list[list] = []
    for w in ordered:
        placed = False
        for v in voices:
            if w.start >= v[-1].end - VOICE_OVERLAP_MIN:
                v.append(w)
                placed = True
                break
        if not placed:
            voices.append([w])
    return voices


def _group_cue_words(words: list, clip_start: float, clip_end: float,
                     mode: str = "words") -> list[list]:
    """Núcleo compartilhado: agrupa objetos Word em cues (listas de Word).

    Usado por _group_cues() e pela auditoria WORD→CUE do debug. A regra de
    quebra (gap > 0.4s, pontuação, 8 words, 48 chars) vive só aqui, com duas
    restrições estruturais:
    1. Quebra só em FRONTEIRA de palavra: token de continuação (sem espaço à
       esquerda, ex: "ita" em "dire"+"ita") nunca inicia cue.
    2. Falas SIMULTÂNEAS nunca são serializadas: words com overlap real
       (> VOICE_OVERLAP_MIN) formam vozes concorrentes, cada uma agrupada
       separadamente. Sem diarização não há nomes de pessoas — só cues
       coexistentes com timestamps preservados (lanes resolvem a exibição).
    mode="intervals" (experimental, --caption-mode): agrupa por rajada de
    fala via group_interval_words; "words" é o default medido.
    """
    if mode == "intervals":
        return group_interval_words(words, clip_start, clip_end)
    if mode != "words":
        raise ValueError(f"caption_mode inválido: {mode!r} (words|intervals)")
    words = sorted(valid_caption_words(words), key=lambda w: w.start)
    words = [w for w in words if w.end > clip_start and w.start < clip_end]
    has_boundaries = any(w.text[:1].isspace() for w in words if w.text)

    def is_continuation(w) -> bool:
        return has_boundaries and not (w.text[:1].isspace() if w.text else True)

    def group_voice(voice: list) -> list[list]:
        out: list[list] = []
        cur: list = []

        def flush():
            if cur:
                out.append(list(cur))
                cur.clear()

        i, n = 0, len(voice)
        while i < n:
            w = voice[i]
            cur.append(w)
            nxt = voice[i + 1] if i + 1 < n else None
            if nxt is None:
                flush()
                break
            gap = nxt.start - w.end
            ends = w.text.strip()[-1] in ".!?" if w.text.strip() else False
            should = (
                gap > CAPTION_PAUSE_THRESHOLD
                or (ends and len(cur) >= 2)
                or len(cur) >= 8
                or len(" ".join(x.text for x in cur)) > CAPTION_MAX_CHARS_PER_LINE * CAPTION_MAX_LINES_PER_CUE
            )
            if should:
                # A pausa/limite é real, mas a palavra não pode partir: TODAS
                # as continuações seguintes fecham o grupo atual; a quebra
                # vale dali.
                while i + 1 < n and is_continuation(voice[i + 1]):
                    i += 1
                    cur.append(voice[i])
                flush()
            i += 1
        return out

    cues: list[list] = []
    for voice in _split_voices(words):
        cues.extend(group_voice(voice))
    cues.sort(key=lambda g: (g[0].start, g[-1].end))
    return cues


def _group_cues(words: list, clip_start: float, clip_end: float,
                mode: str = "words") -> list[tuple]:
    """Agrupa palavras em cues (s_abs, e_abs, texto_limpo). Puro e testável.

    Mesmas garantias do build_srt: só palavras dentro do corte, texto limpo,
    sem cues vazias. Tempos ainda ABSOLUTOS; a conversão p/ relativo é feita
    pelo formatador (SRT/ASS) subtraindo clip_start de forma determinística.
    Words degenerados (end <= start, ex: VAD 20.760→20.760) são descartados
    aqui — ponto único de filtragem (cobre SRT, ASS e debug, inclusive
    transcripts em cache) — para que o fallback CAPTION_MIN_DURATION nunca
    transforme timestamp inválido em cue artificial de 1.0s.
    A extensão visual (dur < 1.0s → 1.0s) nunca invade a próxima cue: só a
    parte artificial é cortada (fim==início não é overlap); o span real das
    words nunca é reduzido (overlap real do Whisper é preservado; lanes
    resolvem a exibição).
    """
    out: list[list] = []
    for chunk_words in _group_cue_words(words, clip_start, clip_end, mode=mode):
        if not chunk_words:
            continue
        s = chunk_words[0].start
        e_real = chunk_words[-1].end
        e = e_real
        if e <= s:
            e = s + CAPTION_MIN_DURATION
        if e - s < CAPTION_MIN_DURATION:
            e = s + CAPTION_MIN_DURATION
        raw = clean_caption_text(smart_join([w.text for w in chunk_words])).upper()
        if raw:
            out.append([s, e, raw, e_real])
    for i in range(len(out) - 1):
        s, e, raw, e_real = out[i]
        s_next = out[i + 1][0]
        if e > e_real + 1e-9 and e > s_next:
            out[i][1] = max(e_real, s_next)
    return [(s, e, raw) for s, e, raw, _ in out]


def merge_event_cues(word_cues: list[tuple], event_cues: list[tuple]) -> list[tuple]:
    """Funde cues de eventos acústicos às cues de palavras. Puro e testável.

    Regra determinística: palavra sempre vence — a cue de evento é cortada
    onde cobre span de word cue (pode virar 0, 1 ou 2 pedaços; pedaço vazio
    morre). Saída ordenada por início, sem overlaps.
    """
    spans = sorted((s, e) for s, e, _ in word_cues)
    out = list(word_cues)
    for s, e, text in sorted(event_cues):
        pieces = [(s, e)]
        for ws, we in spans:
            nxt = []
            for ps, pe in pieces:
                if we <= ps or ws >= pe:
                    nxt.append((ps, pe))
                    continue
                if ws > ps:
                    nxt.append((ps, ws))
                if we < pe:
                    nxt.append((we, pe))
            pieces = nxt
        for ps, pe in pieces:
            if pe > ps:
                out.append((ps, pe, text))
    return sorted(out, key=lambda c: (c[0], c[1]))


def _split_lines(cues: list[tuple], clip_start: float, duration: float) -> list[tuple]:
    """Divide cues longas em (cs_rel, ce_rel, texto) já relativos e clampados.

    O piso visual (dur < 1.0s → 1.0s) nunca invade a próxima cue — mesma
    regra do _group_cues, aplicada na timeline relativa (o clamp absoluto
    poderia ser re-estendido aqui pelo max()).
    """
    out: list[tuple] = []
    for idx, (s_abs, e_abs, raw) in enumerate(cues):
        s = max(0.0, s_abs - clip_start)
        e_real = e_abs - clip_start
        e = min(duration, max(e_real, s + CAPTION_MIN_DURATION))
        if e > e_real + 1e-9 and idx + 1 < len(cues):
            s_next = max(0.0, cues[idx + 1][0] - clip_start)
            if e > s_next:
                e = max(e_real, s_next)
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


def build_srt(words: list, clip_start: float, clip_end: float | None = None,
              event_cues: list[tuple] | None = None,
              caption_mode: str = "words") -> str:
    """Gera SRT com tempos RELATIVOS ao corte (clip_start → 0).

    Garantias determinísticas:
    - palavras fora de [clip_start, clip_end] são descartadas;
    - cues são clampadas em [0, duration] (nunca negativas, nunca além do fim);
    - cues vazias (sem texto após limpeza) são descartadas;
    - tokens especiais ([eot], [sot], ...) nunca viram texto visível.
    - event_cues (cues absolutas de core.acoustic, opt-in): fundidas com
      merge_event_cues — palavra sempre vence; None = comportamento idêntico.
    """
    if not words and not event_cues:
        return ""
    if clip_end is None:
        clip_end = max([w.end for w in words] + [e for _, e, _ in (event_cues or [])],
                       default=clip_start)
    duration = max(0.0, clip_end - clip_start)
    if duration <= 0:
        return ""
    cues = _split_lines(merge_event_cues(_group_cues(words, clip_start, clip_end,
                                                     mode=caption_mode),
                                         event_cues or []),
                        clip_start, duration)
    out = [f"{i}\n{_srt_time(cs)} --> {_srt_time(ce)}\n{text}\n"
           for i, (cs, ce, text) in enumerate(cues, start=1)]
    return "\n".join(out)


# Estilo Shorts/Reels/TikTok. Fonte: Montserrat ExtraBold (presente e
# resolvida via fontconfig; Arial Black não existe aqui). Tamanhos em pixels
# do vídeo (PlayRes = dimensões de saída).
CAPTION_FONT = "Montserrat ExtraBold"
CAPTION_FONT_SIZE_VERTICAL = 54
CAPTION_FONT_SIZE_WIDE = 44
CAPTION_MARGIN_V = 140
# Amarelo TikTok (#FFE600) em BGR do ASS; volta ao branco após a palavra.
CAPTION_HIGHLIGHT_OPEN = r"{\1c&H0000E6FF&}"
CAPTION_HIGHLIGHT_CLOSE = r"{\1c&H00FFFFFF&}"
# Pop discreto por bloco: 92% → 104% → 100% em 280 ms (cues têm ≥1 s).
CAPTION_POP_OPEN = r"{\fscx92\fscy92\t(0,120,\fscx104\fscy104)\t(120,280,\fscx100\fscy100)}"
# Composição vertical: canvas 1080x1920, vídeo principal 1080x1400 (73%),
# faixas de 260 px em cima/embaixo com blur do próprio vídeo.
# Composição vertical 9:16: canvas fixo, vídeo principal dominante e
# centralizado, faixas com blur do próprio vídeo. Para mudar a proporção,
# ajuste LAYOUT_MAIN_H (altura do vídeo): bandas = (1920 - MAIN_H) / 2.
# 1400 → bandas de 260 (73% vídeo); 1200 → bandas de 360 (62% vídeo).
LAYOUT_W, LAYOUT_H, LAYOUT_MAIN_H = 1080, 1920, 1200
LAYOUT_BAND = (LAYOUT_H - LAYOUT_MAIN_H) // 2  # 360
# Cores fixas por locutor (quando diarização disponível).
# Determinísticas: mesma ordem de aparição → mesma cor.
SPEAKER_COLORS: dict[str, str] = {
    "A": "&H00FFFFFF",  # branco
    "B": "&H00E5D100",  # amarelo
    "C": "&H0000BFFF",  # azul
    "D": "&H0000FF00",  # verde
}
# Cor fixa por locutor


def _speaker_color(speaker_id: str) -> str:
    """Retorna cor ASS fixa para o locutor. Determinística."""
    return SPEAKER_COLORS.get(speaker_id, SPEAKER_COLORS.get("A", "&H00FFFFFF"))


# Ordem determinística de atribuição de cores quando não há diarização.
SPEAKER_ORDER: list[str] = ["A", "B", "C", "D"]

HOOK_FONT_SIZE = 84
HOOK_MAX_CHARS_PER_LINE = 16
HOOK_MARGIN_V = 100  # dentro da faixa borrada superior (0-360); nunca na área principal
HOOK_END_SEC = 5.0
HOOK_START_SEC = 0.3

# Lanes verticais (anti-overlap): faixa inferior dividida em slots de altura
# real de linha (Montserrat ExtraBold 54px + outline/sombra, medido via PIL:
# ~69px → slot 72). Cue de N linhas ocupa N slots contíguos. Regra de
# liberação: slot livre se fim_anterior <= início (fim==início NÃO é overlap).
LANE_SLOT_H = 72
LANE_MAX_SLOTS = 5
LANE_MARGIN_V_BASE = CAPTION_MARGIN_V  # 140: slot 0 encostado na base


def _speaker_id_from_text(text: str) -> str | None:
    """Extrai um ID de locutor a partir do texto da legenda.

    Procura por padrões como 'PESSOA A:', 'SPEAKER A:', 'VOICE_A:', etc.
    Retorna o ID ou None se não houver padrão reconhecido.
    """
    m = re.match(r"^[^:\n]*(?:A|B|C|D)\s*:\s*", text or "")
    if m:
        return m.group(1)
    # Padrões alternativos: VOICE_A, SPEAKER_A no início
    m = re.match(r"^[^:\n]*(?:VOICE|SPEAKER)\s*[ _-]?([A-D])[^:\n]*:?", text or "")
    if m:
        return m.group(1)
    return None


def assign_lanes(
    cues: list[tuple],
    speaker_map: dict[str, str] | None = None,
) -> list[tuple]:
    """Atribui (cs, ce, texto) → (cs, ce, texto, lane_base, span, speaker_id, color).

    Cada cue ocupa `span` slots contíguos livres no seu início; reutiliza
    as mais baixas. Máximo LANE_MAX_SLOTS; se não couber, reutiliza a lane 0.
    speaker_map: dict mapeando speaker_id → nome legível (opcional, para cores).

    O lane é determinado exclusivamente pelo conflito temporal. Duas captions
    simultâneas NUNCA compartilham a mesma lane. A cor do locutor é aplicada
    separadamente no renderer (não influencia o lane).
    """
    slots = [float("-inf")] * LANE_MAX_SLOTS
    out: list[tuple] = []
    overflowed = False

    # Determina ID de locutor para cada cue (se speaker_map fornecido)
    cue_speakers: list[str | None] = []
    if speaker_map:
        for cs, ce, text in sorted(cues, key=lambda c: (c[0], c[1])):
            sid = _speaker_id_from_text(text)
            # Se o texto já tem um ID de locutor embutido, usa-se;
            # senão, atribui o próximo da ordem determinística.
            if sid is None:
                # atribui baseado na ordem de aparição
                sid = SPEAKER_ORDER[len(cue_speakers) % len(SPEAKER_ORDER)]
                cue_speakers.append(sid)
            else:
                cue_speakers.append(sid)
    else:
        cue_speakers = [None] * len(cues)

    for idx, (cs, ce, text) in enumerate(sorted(cues, key=lambda c: (c[0], c[1]))):
        nlines = max(1, text.count("\n") + 1)
        k = min(nlines, LANE_MAX_SLOTS)
        base = None
        for i in range(LANE_MAX_SLOTS - k + 1):
            if all(slots[j] <= cs + 1e-6 for j in range(i, i + k)):
                base = i
                break
        if base is None:
            base, k, overflowed = 0, min(nlines, LANE_MAX_SLOTS), True
        for j in range(base, base + k):
            slots[j] = max(slots[j], ce)

        speaker = cue_speakers[idx] if idx < len(cue_speakers) else None
        color = _speaker_color(speaker) if speaker else None
        out.append((cs, ce, text, base, k, speaker, color, overflowed and base == 0))
    return out


def build_hook_lines(title: str) -> list[str]:
    """Hook em ≤3 linhas a partir do título do scoring (ou [] se vazio)."""
    t = (title or "").strip()
    if not t:
        return []
    return _wrap(t.upper(), HOOK_MAX_CHARS_PER_LINE)[:3]


def highlight_words_from_title(title: str) -> set:
    """Palavras do título (minúsculas, alfanuméricas, len>=5) para destaque.

    Determinístico e orientado a dados: o título é o gancho viral escolhido
    pelo scoring — palavras dele que reaparecem na legenda ganham amarelo.
    Sem LLM, sem aleatoriedade. len>=5 evita stopwords (para/como/isso).
    """
    words = set()
    for tok in re.findall(r"\w+", (title or "").lower()):
        if len(tok) >= 5:
            words.add(tok)
    return words


def _apply_highlight(line: str, highlight: set | None) -> str:
    if not highlight:
        return line
    out = []
    for tok in line.split(" "):
        key = re.sub(r"\W+", "", tok.lower())
        if key in highlight:
            out.append(f"{CAPTION_HIGHLIGHT_OPEN}{tok}{CAPTION_HIGHLIGHT_CLOSE}")
        else:
            out.append(tok)
    return " ".join(out)


def build_ass(words: list, clip_start: float, clip_end: float,
              width: int, height: int, font_size: int = CAPTION_FONT_SIZE_VERTICAL,
              highlight: set | None = None, animate: bool = True,
              hook_title: str | None = None,
              event_cues: list[tuple] | None = None,
              caption_mode: str = "words") -> str:
    """Gera ASS com PlayRes = dimensões REAIS de saída.

    Motivo: sem PlayRes explícito o libass assume 384x288 e escala o estilo
    de forma anamórfica (fonte gigante/esticada, MarginV fora da base). Com
    PlayRes == frame, FontSize/MarginV valem pixels do vídeo, determinístico.
    Mesmas garantias de tempo/texto do build_srt (mesmo núcleo).
    highlight: set de palavras (lower) pintadas de amarelo — ver
    highlight_words_from_title. animate: pop discreto por bloco (280 ms).
    hook_title: título do scoring vira cartela [0.3,5.0] na faixa borrada
    superior (fora da área principal); vazio = sem hook (nunca usa filename fallback).
    """
    duration = max(0.0, clip_end - clip_start)
    cues = _split_lines(merge_event_cues(_group_cues(words, clip_start, clip_end,
                                                     mode=caption_mode),
                                         event_cues or []),
                        clip_start, duration) if duration > 0 else []
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
        (f"Style: Hook,{CAPTION_FONT}, {HOOK_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
         f"-1,0,0,0,100,100,0,0,1,3,1,8,40,40,{HOOK_MARGIN_V},1"),
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
         "Effect, Text"),
    ]
    hook_lines = build_hook_lines(hook_title or "")
    if hook_lines and duration > HOOK_END_SEC:
        text = r"\N".join(_apply_highlight(l, highlight) for l in hook_lines)
        text = r"{\fad(150,150)}" + text
        lines.append(f"Dialogue: 0,{_ass_time(HOOK_START_SEC)},{_ass_time(HOOK_END_SEC)},"
                     f"Hook,,0,0,0,,{text}")
    # Eventos acústicos (*ÁUDIO ESTOURADO*...) saem em amarelo — mesmo destaque
    # do título, como ênfase de humor. Identidade pelo texto (constantes de
    # core/acoustic.py, ex: "*ÁUDIO ESTOURADO"); SRT não tem cor, segue texto
    # puro. Timestamps e lanes: inalterados.
    event_texts = {t for _, _, t in (event_cues or [])}
    for cs, ce, text in cues:
        parts = []
        for line in text.split("\n"):
            parts.append(_apply_highlight(line, highlight))
        text = r"\N".join(parts)
        if text in event_texts:
            text = f"{CAPTION_HIGHLIGHT_OPEN}{text}{CAPTION_HIGHLIGHT_CLOSE}"
        if animate:
            text = CAPTION_POP_OPEN + text
        lines.append(f"Dialogue: 0,{_ass_time(cs)},{_ass_time(ce)},Clip,,0,0,0,,{text}")
    _assign_dialogue_lanes(lines, cues)
    return "\n".join(lines) + "\n"


def _assign_dialogue_lanes(lines: list[str], cues: list[tuple]) -> None:
    """Aplica lanes anti-overlap in-place nas linhas Dialogue (menos hook).

    Cada cue ganha MarginV própria = base + lane*SLOT_H (a partir da base da
    faixa inferior). Cues simultâneas nunca compartilham a mesma coordenada;
    livres são reutilizadas; no máximo LANE_MAX_SLOTS, com fallback p/ lane 0.
    """
    laned = assign_lanes(cues)
    di = 0
    for i, line in enumerate(lines):
        if not line.startswith("Dialogue:") or ",Clip,," not in line:
            continue
        if di >= len(laned):
            break
        _, _, _, base, _, _, _, _ = laned[di]
        di += 1
        head, _, text = line.partition(",,0,0,0,,")
        mv = LANE_MARGIN_V_BASE + base * LANE_SLOT_H
        lines[i] = f"{head},,0,0,{mv},,{text}"


def _escape_subs(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _fmt_ts(t: float) -> str:
    if t < 0:
        sign = "-"
        t = -t
    else:
        sign = ""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{sign}{h:02d}:{m:02d}:{s:06.3f}" if h > 0 or m > 0 or s > 0 else f"{sign}00:00:00.000"


def validate_cues(words: list, clip_start: float, clip_end: float,
                  cues: list[tuple]) -> list[str]:
    """Dupla validação: absoluto esperado → relativo → escrito no arquivo.

    Retorna lista de avisos (vazia = ok). Nunca levanta: o pipeline loga e
    segue (cues inválidas já foram descartadas na construção).
    """
    warns: list[str] = []
    duration = clip_end - clip_start
    for cs, ce, text in cues:
        if not (cs >= 0):
            warns.append(f"cue negativa: cs={cs:.3f} (clip {clip_start:.3f})")
        if not (ce > cs):
            warns.append(f"cue vazia/invertida: [{cs:.3f},{ce:.3f}] {text[:30]!r}")
        if ce > duration + 1e-6:
            warns.append(f"cue além do fim: ce={ce:.3f} > dur={duration:.3f}")
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        if is_degenerate_word(w):
            # Descartado em _group_cues por regra de domínio (end <= start);
            # não é "palavra sem cobertura", é timestamp inválido do Whisper.
            continue
        # Palavra em tempo RELATIVO (abs - clip_start) contra cues relativas.
        rs, re_ = w.start - clip_start, w.end - clip_start
        covered = any(cs <= rs + 1e-6 and re_ - 1e-6 <= ce
                      or (cs <= rs + 1e-6 <= ce) or (cs <= re_ - 1e-6 <= ce)
                      for cs, ce, _ in cues)
        if not covered and w.text.strip() and not is_special_token(w.text):
            warns.append(f"palavra sem cobertura: {w.text.strip()!r} "
                         f"[{w.start:.3f},{w.end:.3f}]")
    return warns


def _timestamp_investigation(clip_start: float, clip_end: float,
                              words: list, cues: list[tuple]) -> str:
    """Gera relatório de deslocamento no início da timeline.

    Mostra a cadeia: vídeo → áudio → whisper → chunk → clip → legenda.
    Use os valores reais para identificar onde ocorre o deslocamento.
    """
    # Primeiro/última palavra do Whisper dentro do clip
    first_word_abs = None
    last_word_abs = None
    for w in sorted(words, key=lambda x: x.start):
        if w.end > clip_start and w.start < clip_end and w.text.strip():
            if first_word_abs is None:
                first_word_abs = w.start
            last_word_abs = w.end

    # Primeiro/último cue no clip
    first_cs = None
    last_ce = None
    for cs, ce, text in cues:
        if first_cs is None or cs < first_cs:
            first_cs = cs
        if last_ce is None or ce > last_ce:
            last_ce = ce

    lines = [
        "INVESTIGAÇÃO DE DESLOCAMENTO NO INÍCIO",
        "",
        "VIDEO START:        " + _fmt_ts(0.0),
        "AUDIO START:        " + _fmt_ts(clip_start),
        "CHUNK START:        " + _fmt_ts(clip_start),
        "WHISPER FIRST WORD: " + (_fmt_ts(first_word_abs) if first_word_abs is not None else "N/A"),
        "CLIP START:         " + _fmt_ts(clip_start),
        "RELATIVE FIRST WORD: " + (_fmt_ts(first_word_abs - clip_start) if first_word_abs is not None else "N/A"),
        "CAPTION FIRST START: " + _fmt_ts(first_cs) if first_cs is not None else "CAPTION FIRST START: N/A",
        "",
        "Expected (all should be 0.000 relative):",
        "  Se WHISPER FIRST WORD ≠ 0.000 relativo: o Whisper tem seu próprio início de áudio.",
        "  Se CAPTION FIRST START ≠ 0.000 relativo: o _group_cues/ _split_lines deslocou.",
        "  Se houver delta real, ele é a origem do problema de timing.",
        "",
        "Delta calculado:",
        f"  WHISPER FIRST WORD rel: {_fmt_ts(first_word_abs - clip_start) if first_word_abs is not None else 'N/A'}",
        f"  CAPTION FIRST START:   {_fmt_ts(first_cs) if first_cs is not None else 'N/A'}",
        f"  DELTA:                  {_fmt_ts((first_cs if first_cs is not None else 0) - (first_word_abs - clip_start) if first_word_abs is not None else 0)}",
        "",
        "--- FIM DO RELATÓRIO ---",
    ]
    return "\n".join(lines)


def render_caption_debug(clip_name: str, clip_start: float, clip_end: float,
                         words: list, cues: list[tuple],
                         warnings: list[str], video_path: str | None = None) -> str:
    """Texto do caption-debug.txt (§2 do diagnóstico): ABS → REL → CAP.

    video_path (opcional): adiciona seção ENERGIA POR WORD — diagnóstico
    report-only (rms do áudio no span de cada word; nunca desloca nada).
    """
    L = [f"CAPTION DEBUG — {clip_name}",
         f"Clip: {_fmt_ts(clip_start)} → {_fmt_ts(clip_end)}",
         "", "TRANSCRIÇÃO ORIGINAL [ABS]"]
    for w in sorted(words, key=lambda x: x.start):
        if w.end > clip_start and w.start < clip_end and w.text.strip():
            L.append(f"[ABS] {_fmt_ts(w.start)} → {_fmt_ts(w.end)}  {w.text.strip()!r}")
    L += ["", "TIMESTAMP RELATIVO AO CLIP [REL] (abs - clip_start)"]
    for w in sorted(words, key=lambda x: x.start):
        if w.end > clip_start and w.start < clip_end and w.text.strip():
            L.append(f"[REL] {_fmt_ts(w.start - clip_start)} → {_fmt_ts(w.end - clip_start)}  "
                     f"{w.text.strip()!r}")
    L += ["", "TIMESTAMP NO ARQUIVO DE LEGENDA [CAP]"]
    for cs, ce, text in cues:
        flat = text.replace("\n", " / ")
        L += [f"[CAP] {_fmt_ts(cs)} → {_fmt_ts(ce)}  {flat!r}",
              f"      abs esperado: {_fmt_ts(clip_start + cs)} → {_fmt_ts(clip_start + ce)}"]
    dropped = [w for w in sorted(words, key=lambda x: x.start)
               if is_degenerate_word(w) and (w.text or "").strip()]
    L += ["", f"WORDS DEGENERADOS DESCARTADOS: {len(dropped)}"]
    for w in dropped:
        L += [f'DROPPED DEGENERATE WORD: text={w.text.strip()!r} '
              f'start={w.start:.3f} end={w.end:.3f} reason=end <= start']
    outside = [w for w in sorted(words, key=lambda x: x.start)
               if not is_degenerate_word(w) and (w.text or "").strip()
               and not (w.end > clip_start and w.start < clip_end)]
    L += ["", f"WORDS VÁLIDAS FORA DO CLIP (não é erro do Whisper): {len(outside)}"]
    for w in outside[:20]:
        L += [f'DROPPED OUTSIDE CLIP: text={w.text.strip()!r} '
              f'start={w.start:.3f} end={w.end:.3f} reason=outside_selected_clip']
    if len(outside) > 20:
        L += [f"... e mais {len(outside) - 20}"]
    L += ["", "AUDITORIA WORD→CUE (absoluto | relativo ao clip | duração)"]
    groups = _group_cue_words(words, clip_start, clip_end)
    abs_cues = _group_cues(words, clip_start, clip_end)
    duration = max(0.0, clip_end - clip_start)
    for i, gw in enumerate(groups):
        if i < len(abs_cues):
            s_abs, e_abs, raw = abs_cues[i]
        else:
            s_abs, e_abs, raw = (gw[0].start, gw[-1].end, "?") if gw else (0, 0, "?")
        rs = max(0.0, s_abs - clip_start)
        re_ = min(duration, e_abs - clip_start) if duration > 0 else e_abs - clip_start
        flat = (raw or "").replace("\n", " / ")
        L.append(f"Cue {i:02d} abs {s_abs:.3f}→{e_abs:.3f} "
                 f"rel {rs:.3f}→{re_:.3f} dur {re_ - rs:.3f} {flat!r}")
        for w in gw:
            L.append(f"    {w.text.strip()!r:14} {w.start:.3f}→{w.end:.3f}")
        first_ok = abs(gw[0].start - s_abs) < 1e-6 if gw else False
        L.append(f"    status: {'OK' if first_ok else 'DIVERGENTE'} "
                 f"(cue começa na primeira word: {first_ok})")
    L += ["", "CHECAGEM FINAL"]
    if not warnings:
        L.append("OK: sem divergências (rel>=0, fim> início, dentro da duração).")
    else:
        L += [f"AVISO: {x}" for x in warnings]

    if video_path is not None:
        # Dupla validação (diagnóstico): a word afirma fala no seu span; o
        # áudio confirma energia. Divergência = Whisper errou a medição.
        # Report-only: nenhum timestamp é tocado por esta seção.
        try:
            from .acoustic import word_energy as _word_energy
            levels = _word_energy(video_path, [w for w in words
                                               if w.end > clip_start and w.start < clip_end])
        except Exception as e:
            levels = []
            L += ["", f"ENERGIA POR WORD: indisponível ({e})"]
        if levels:
            silent = sum(1 for e in levels if e["silent"])
            L += ["", f"ENERGIA POR WORD (rms no span; silent < 0.015): "
                      f"{len(levels)} words, {silent} em silêncio"]
            for e in levels:
                flag = " SILÊNCIO?" if e["silent"] else ""
                L.append(f'  {e["text"]!r:14} {e["start"]:.3f}→{e["end"]:.3f} '
                         f'rms={e["rms"]:.4f}{flag}')

    # Adiciona investigação de deslocamento no final
    L += ["", _timestamp_investigation(clip_start, clip_end, words, cues)]
    return "\n".join(L) + "\n"


def write_human_transcript(segments: list, path) -> None:
    """Transcrição legível [start → end] + texto (uma por segmento)."""
    lines = []
    for s in segments:
        lines.append(f"[{_fmt_ts(s.start)} → {_fmt_ts(s.end)}]")
        lines.append((s.text or "").strip())
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_clip_debug(debug_dir, out_path: Path, c, subs: str, suffix: str,
                      event_cues: list[tuple] | None = None,
                      acoustic_events: list | None = None,
                      video_path: str | None = None,
                      caption_mode: str = "words") -> None:
    """Artefatos de --debug-captions em debug/<clip>/ (só nesse modo)."""
    d = Path(debug_dir) / Path(out_path).stem
    d.mkdir(parents=True, exist_ok=True)
    ext = ".ass" if suffix == ".ass" else ".srt"
    (d / f"captions{ext}").write_text(subs, encoding="utf-8")
    if suffix == ".ass":
        (d / "captions.srt").write_text(
            build_srt(c.words, c.start, c.end, event_cues=event_cues,
                      caption_mode=caption_mode), encoding="utf-8")
    words = sorted(c.words, key=lambda w: w.start)
    (d / "transcript-words.json").write_text(
        json.dumps(
            [{"text": w.text, "start": w.start, "end": w.end} for w in words],
            ensure_ascii=False, indent=2), encoding="utf-8")
    if acoustic_events:
        (d / "acoustic-events.json").write_text(
            json.dumps([{"type": e.type, "start": e.start, "end": e.end,
                         "confidence": e.confidence} for e in acoustic_events],
                       ensure_ascii=False, indent=2), encoding="utf-8")
    duration = max(0.0, c.end - c.start)
    cues = _split_lines(merge_event_cues(_group_cues(c.words, c.start, c.end,
                                                     mode=caption_mode),
                                         event_cues or []),
                        c.start, duration) if duration > 0 else []
    warns = validate_cues(c.words, c.start, c.end, cues)
    (d / "caption-debug.txt").write_text(
        render_caption_debug(Path(out_path).stem, c.start, c.end,
                             c.words, cues, warns,
                             video_path=video_path), encoding="utf-8")
    for w in warns:
        print(f"   ! legenda [{Path(out_path).stem}]: {w}")


def _video_encoder() -> tuple[str, list[str]]:
    """Escolhe encoder: h264_qsv (sonda funcional 1x, cacheada) senão libx264."""
    from . import intel as intel_hw
    if intel_hw.best_video_encoder() == "h264_qsv":
        return "h264_qsv", ["-global_quality", "23", "-look_ahead", "1"]
    return "libx264", ["-preset", "veryfast", "-crf", "20"]


def cut(video_path: str, c: Candidate, out_path: Path, vertical: bool, captions: bool,
        debug_dir: Path | str | None = None,
        acoustic_events: list | None = None,
        caption_mode: str = "words") -> None:
    duration = c.duration
    filters: list[str] = []

    # Corte via trim/atrim (NÃO via -ss após -i): medição ponta a ponta
    # provou que -ss de saída desloca a linha do tempo das legendas em
    # exatamente -fine (o filtro subtitles avalia na timeline do demux,
    # que o -ss de saída não desloca), enquanto trim+setpts/asetpts deixa
    # vídeo, áudio e legendas todos 0-based e alinhados. Sem offset mágico.
    coarse = max(0, c.start - 2)
    fine = c.start - coarse
    end = fine + duration
    af = f"atrim=start={fine}:end={end},asetpts=PTS-STARTPTS"

    if vertical:
        # Composição Shorts: canvas 1080x1920, vídeo principal 1080x1400 (73%)
        # centralizado, faixas de 260 px com blur do próprio vídeo (cores
        # derivadas do conteúdo, sem cor fixa). Blur em resolução baixa
        # (270x480) = barato; sem arquivos intermediários.
        fx = face_center_x(video_path, c.start, c.end)
        w, h = _probe_dimensions(video_path)
        crop_fail = bool(w and h and (w * LAYOUT_MAIN_H / h) < LAYOUT_W)
        if crop_fail:
            fg = (f"scale={LAYOUT_W}:{LAYOUT_MAIN_H}:force_original_aspect_ratio=increase,"
                  f"crop={LAYOUT_W}:{LAYOUT_MAIN_H}")
        else:
            fg = (f"scale=-2:{LAYOUT_MAIN_H},crop={LAYOUT_W}:{LAYOUT_MAIN_H}:"
                  f"x=min(max(iw*{fx:.3f}-{LAYOUT_W // 2}\\,0)\\,iw-{LAYOUT_W}):y=0")
        filters.append(
            f"trim=start={fine}:end={end},setpts=PTS-STARTPTS,split=2[base][fgs];"
            f"[base]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"gblur=sigma=25,scale={LAYOUT_W}:{LAYOUT_H},"
            f"eq=brightness=-0.12:saturation=0.85[bg];"
            f"[fgs]{fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:{LAYOUT_BAND}")
        out_w, out_h, font_size = LAYOUT_W, LAYOUT_H, CAPTION_FONT_SIZE_VERTICAL
    else:
        filters.append("scale=1280:-2")
        filters.append(f"trim=start={fine}:end={end},setpts=PTS-STARTPTS")
        out_w, out_h, font_size = None, None, CAPTION_FONT_SIZE_WIDE
        w, h = _probe_dimensions(video_path)
        if w and h:
            out_w = 1280
            out_h = max(2, (int(h * 1280 / w) // 2) * 2)

    srt_file: str | None = None
    try:
        if captions and (c.words or acoustic_events):
            # Eventos acústicos (opt-in): cues absolutas fundidas; palavra vence.
            from .acoustic import event_cues as _event_cues
            ev_cues = _event_cues(acoustic_events or [], c.start, c.end)
            # ASS com PlayRes = frame real → layout determinístico em pixels.
            # Sem dimensões conhecidas, cai no SRT legado (comportamento anterior).
            # Destaque: palavras do título (determinístico, sem LLM).
            hl = highlight_words_from_title(c.title)
            if out_w and out_h:
                # Hook = título do scoring (sem LLM extra); vazio = sem hook.
                subs = build_ass(c.words, c.start, c.end, out_w, out_h, font_size,
                                 highlight=hl, hook_title=c.title or None,
                                 event_cues=ev_cues or None,
                                 caption_mode=caption_mode)
                suffix = ".ass"
            else:
                subs = build_srt(c.words, c.start, c.end, event_cues=ev_cues or None,
                                 caption_mode=caption_mode)
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
                        f"subtitles='{esc}':force_style='FontName=Montserrat ExtraBold,FontSize=22,"
                        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=120'"
                    )
                if debug_dir is not None:
                    _write_clip_debug(debug_dir, out_path, c, subs, suffix,
                                      event_cues=ev_cues or None,
                                      acoustic_events=acoustic_events or None,
                                      video_path=video_path,
                                      caption_mode=caption_mode)
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
