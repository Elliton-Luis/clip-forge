"""acoustic.py — análise acústica conservadora (sem transcrição).

SRP: detectar eventos de ÁUDIO ESTOURADO (clipping digital sustentado).
Sem numpy, sem modelo, sem dependência nova: stdlib (array) + ffmpeg.

O que detecta (calibrado em material real, 2026-09):
  frame "hot" (50 ms): peak >= 0.99 (teto digital) E rms >= 0.15 (alto de
  verdade, não pico isolado). Evento = sequência hot com pontes ≤ 0.2 s,
  duração ≥ 0.3 s. Confiança = fração hot dentro do evento.

  áudio limpo (teste_audio.mkv): 0 eventos. Gameplay clipado: eventos com
  confiança 0.5–1.0 (VideoMedio2 7.75→9.5 conf 0.71; GuilhermeGaivota 7x).

O que NÃO detecta (veredito documentado): RISADA. Risada não tem assinatura
confiável só com peak/rms — gritaria e batida de jogo dariam falso positivo.
Sem modelo espectral dedicado (dependência pesada), "risada estourada" seria
heurística ruim. Revisitado apenas com evidência nova.

Conceito próprio (não mistura com Word): AcousticEvent(type, start, end,
confidence). Renderização queimada é opt-in (--acoustic-captions); detecção,
manifest e debug são sempre registrados quando a etapa roda.
"""
import subprocess
from array import array
from dataclasses import dataclass

SAMPLE_RATE = 16000
FRAME_SEC = 0.05
PEAK_FLOOR = 0.99    # teto digital: sample encostando em 1.0
RMS_FLOOR = 0.15     # ...e alto de verdade (pico isolado não conta)
BRIDGE_SEC = 0.2     # une rajadas clipadas próximas num evento só
MIN_EVENT_SEC = 0.3  # abaixo disso é transiente, não evento
SAT_SAMPLE = 32000   # |x| >= ~0.977 em s16 = saturado (p/ estatísticas)

EVENT_TEXT = "*ÁUDIO ESTOURADO"
EVENT_TEXTS = {
    "clipped-audio": "*ÁUDIO ESTOURADO",
    # "laugh" (marcado pelo usuário, ver load_laughs): detector automático
    # recusado com evidência — nenhuma assinatura barata separa risada de
    # grito nos samples marcados (pico/RMS/ZCR/envelope/aspereza sobrepõem).
    "laugh": "*RISADA ESTOURADA",
}


@dataclass
class AcousticEvent:
    type: str  # hoje: só "clipped-audio"
    start: float
    end: float
    confidence: float  # 0..1 = fração de frames hot no evento


def decode_pcm(video_path: str, start: float, dur: float) -> list[float]:
    """[start, start+dur) em mono 16 kHz, floats -1..1. Erro contextualizado."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(max(0.0, start)), "-t", str(dur), "-i", video_path,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, timeout=120)
    if r.returncode != 0 or len(r.stdout) < 100:
        raise RuntimeError(f"ffmpeg decode acústico rc={r.returncode}: "
                           f"{(r.stderr or b'')[:200]!r}")
    a = array("h")
    a.frombytes(r.stdout)
    return [x / 32768.0 for x in a]


def frame_hot_flags(samples: list[float]) -> list[bool]:
    """Amostras → flags por frame de 50 ms. Puro e testável."""
    fr = int(SAMPLE_RATE * FRAME_SEC)
    out = []
    for i in range(0, len(samples) - fr + 1, fr):
        f = samples[i:i + fr]
        peak = max(abs(x) for x in f)
        rms = (sum(x * x for x in f) / fr) ** 0.5
        out.append(peak >= PEAK_FLOOR and rms >= RMS_FLOOR)
    return out


def find_events(hot: list[bool], offset: float = 0.0) -> list[AcousticEvent]:
    """Flags → eventos (ponte + duração mínima). Puro e testável."""
    idx = [i for i, h in enumerate(hot) if h]
    if not idx:
        return []
    groups = [[idx[0]]]
    for i in idx[1:]:
        if (i - groups[-1][-1]) * FRAME_SEC <= BRIDGE_SEC:
            groups[-1].append(i)
        else:
            groups.append([i])
    events = []
    for g in groups:
        s = offset + g[0] * FRAME_SEC
        e = offset + (g[-1] + 1) * FRAME_SEC
        if e - s >= MIN_EVENT_SEC:
            span_frames = g[-1] - g[0] + 1
            events.append(AcousticEvent(
                type="clipped-audio", start=round(s, 3), end=round(e, 3),
                confidence=round(len(g) / span_frames, 3)))
    return events


def signal_stats(samples: list[float]) -> dict:
    """peak/rms/% saturado do trecho (diagnóstico, sem limiar)."""
    if not samples:
        return {"peak": 0.0, "rms": 0.0, "saturated_ratio": 0.0}
    peak = max(abs(x) for x in samples)
    rms = (sum(x * x for x in samples) / len(samples)) ** 0.5
    sat = sum(1 for x in samples if abs(x) * 32768 >= SAT_SAMPLE)
    return {"peak": round(peak, 4), "rms": round(rms, 4),
            "saturated_ratio": round(sat / len(samples), 4)}


# Piso de silêncio calibrado (videofull/teste_audio, 2026-09): silêncio real
# fica em rms ~0.01; fala típica >= 0.05. Word com rms abaixo disso num span
# válido é SUSPEITA (gap do Whisper ou palavra fantasma) — só diagnóstico.
SILENCE_RMS = 0.015


def rms_between(samples: list[float], start: float, end: float) -> float:
    """RMS no intervalo [start, end) em segundos. Puro e testável."""
    s0 = max(0, int(start * SAMPLE_RATE))
    s1 = min(len(samples), int(end * SAMPLE_RATE))
    if s1 <= s0:
        return 0.0
    seg = samples[s0:s1]
    return (sum(x * x for x in seg) / len(seg)) ** 0.5


def word_energy(video_path: str, words: list) -> list[dict]:
    """Cada word válida → {text, start, end, rms, silent}. Um decode só.

    Diagnóstico: nunca desloca timestamp, nunca filtra. Falha de decode
    retorna lista vazia (debug segue sem a seção).
    """
    valid = [w for w in words if (w.end > w.start) and (w.text or "").strip()]
    if not valid:
        return []
    t0 = min(w.start for w in valid)
    t1 = max(w.end for w in valid)
    try:
        samples = decode_pcm(video_path, t0, t1 - t0)
    except RuntimeError:
        return []
    out = []
    for w in valid:
        rms = rms_between(samples, w.start - t0, w.end - t0)
        out.append({"text": w.text.strip(), "start": round(w.start, 3),
                    "end": round(w.end, 3), "rms": round(rms, 4),
                    "silent": rms < SILENCE_RMS})
    return out


def detect_clipping(video_path: str, start: float, dur: float,
                    offset: float | None = None) -> tuple[list[AcousticEvent], dict]:
    """Pipeline completo: decode → flags → eventos + estatísticas."""
    samples = decode_pcm(video_path, start, dur)
    base = start if offset is None else offset
    return find_events(frame_hot_flags(samples), base), signal_stats(samples)


def event_cues(events: list[AcousticEvent], clip_start: float,
               clip_end: float) -> list[tuple]:
    """Eventos → cues absolutas (s, e, texto), clipadas no corte.

    Piso visual 1.0 s igual às captions (consistência de timeline); nunca
    além do fim do clip. Ordenadas por início.
    """
    out = []
    for ev in events:
        s = max(ev.start, clip_start)
        e = min(ev.end, clip_end)
        if e <= s:
            continue
        if e - s < 1.0:
            e = min(clip_end, s + 1.0)
        if e > s:
            out.append((s, e, EVENT_TEXTS.get(ev.type, EVENT_TEXT)))
    return sorted(out)


def load_laughs(path: str | None) -> list[AcousticEvent]:
    """Ranges de risada marcados pelo usuário → eventos (ground truth humano).

    Formato (segundos, um range por linha; `#` comenta):
        18 23
        27 30
    Linha malformada ou end <= start = erro claro (nunca chute). Sem
    detecção: quem marca é o usuário, então o rótulo é sempre honesto.
    """
    if not path:
        return []
    from pathlib import Path as _P
    events = []
    try:
        lines = _P(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise RuntimeError(f"laughs inválido ({path}): {e}")
    for lineno, line in enumerate(lines, start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            s, e = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            raise RuntimeError(
                f"laughs {path}:{lineno}: esperado 'INICIO FIM' em segundos")
        if not (e > s >= 0) or len(parts) != 2:
            raise RuntimeError(
                f"laughs {path}:{lineno}: range inválido ({line!r})")
        events.append(AcousticEvent(type="laugh", start=s, end=e,
                                    confidence=1.0))
    return events
