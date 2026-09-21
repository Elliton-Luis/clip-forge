"""review.py — revisão humana da transcrição antes dos clipes.

Fluxo:
    Whisper → work/<stem>/transcription/{transcript.json (draft), words.txt}
        → usuário edita words.txt (texto livre, timestamps validados)
        → approve → transcript.json (approved)
        → scoring/seleção/corte consomem a aprovada (sem re-rodar Whisper)
        → finalize limpa intermediários (só com confirmação)

Regras estruturais (nunca quebradas):
- Texto e timestamps são independentes: corrigir "dire ita"→"direita" não
  toca start/end; corrigir timestamp não toca texto.
- O programa NUNCA inventa timestamp: linha nova sem timestamp válido é erro
  claro, não chute. Mapeamento segmento↔word ambíguo usa regra documentada
  (ver parse_words_txt) ou falha com mensagem.
- Título/destaque só valem se sustentados pela transcrição aprovada
  (validate_grounding): violação é sinalizada, nunca silenciada.
"""
import json
import re
from pathlib import Path

from .models import Segment, Word

WORK_ROOT = Path("work")
DRAFT = "draft"
APPROVED = "approved"

# [s02] [00:01.240 → 00:02.810] texto livre aqui
_WORD_LINE_RE = re.compile(
    r"^\s*(?:\[s(\d+)\]\s*)?\[(\d+):(\d+(?:\.\d+)?)\s*→\s*(\d+):(\d+(?:\.\d+)?)\]\s?(.*)$")
_STOPWORDS = frozenset(
    "a o os as um uma umas uns de do da dos das em no na nos nas por para com sem "
    "que e ou se tu ele ela eles elas tu tá ta te lhe isso isto esse essa este esta "
    "the a an to of in on and or is are was were be been que não nao sim não".split())


# ----------------------------------------------------------------------------
# Sessão (work/<stem>/transcription/)
# ----------------------------------------------------------------------------

def session_dir(video_path: str, work_root: Path = WORK_ROOT) -> Path:
    """work/<stem>/transcription (stem sanitizado, determinístico)."""
    stem = re.sub(r"[^\w\-]+", "_", Path(video_path).stem).strip("_") or "video"
    d = work_root / stem / "transcription"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_transcript(segments: list[Segment], path: Path, status: str = DRAFT) -> None:
    data = {"status": status, "segments": [
        {"text": s.text, "start": s.start, "end": s.end,
         "timestamp_source": getattr(s, "timestamp_source", "whisper"),
         "words": [{"text": w.text, "start": w.start, "end": w.end,
                    "p": getattr(w, "prob", getattr(w, "confidence", None)),
                    "timestamp_source": getattr(w, "timestamp_source", "whisper")}
                   for w in s.words]}
        for s in segments]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_transcript(path: Path) -> tuple[list[Segment], str]:
    """(segments, status). Erro claro se o arquivo não é uma transcrição válida."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = []
        for s in data.get("segments", []):
            words = [Word(w["text"], float(w["start"]), float(w["end"]),
                          confidence=w.get("p", w.get("confidence")),
                          timestamp_source=w.get("timestamp_source", "whisper") or "whisper")
                     for w in s.get("words", [])]
            for w, raw in zip(words, s.get("words", [])):
                w.prob = raw.get("p", raw.get("confidence"))  # type: ignore[attr-defined]
            segs.append(Segment(text=s.get("text", ""), start=float(s.get("start", 0)),
                                end=float(s.get("end", 0)), words=words,
                                timestamp_source=s.get("timestamp_source", "whisper") or "whisper"))
        return segs, data.get("status", DRAFT)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"transcrição inválida em {path}: {e}")


def require_approved(path: Path) -> list[Segment]:
    """A versão final só existe após aprovação explícita (teste §15)."""
    segs, status = load_transcript(path)
    if status != APPROVED:
        raise RuntimeError(
            f"{path} está '{status}' — aprove explicitamente antes de usar como final")
    return segs


# ----------------------------------------------------------------------------
# words.txt — formato editável (texto livre, timestamps validados)
# ----------------------------------------------------------------------------

def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def dump_words_txt(segments: list[Segment], path: Path) -> None:
    lines = ["# Edite o TEXTO à vontade. Timestamps: edite com cuidado ou não toque.",
             "# Uma linha = uma palavra. Remover a linha remove a palavra.",
             "# Linha nova precisa de timestamp válido: [MM:SS.mmm → MM:SS.mmm] texto",
             ""]
    for si, seg in enumerate(segments):
        for w in seg.words:
            lines.append(f"[s{si:02d}] [{_fmt(w.start)} → {_fmt(w.end)}] {w.text.strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_words_txt(path: Path) -> list[Segment]:
    """words.txt editado → Segments. Regras determinísticas e documentadas:

    - Linhas vazias/# são ignoradas; linha malformada = erro (nunca chute).
    - start/end vêm do texto da linha (editáveis, independentes do texto).
    - O tag [sNN] preserva o segmento original; linha nova sem tag herda o
      segmento da linha anterior; sem linha anterior, erro.
    - Remover linha remove a word; segmento esvaziado é descartado.
    - Texto do segmento = junção dos textos das words (smart_join simples).
    """
    raw = path.read_text(encoding="utf-8").splitlines()
    buckets: dict[int, list[Word]] = {}
    order: list[int] = []
    last_seg: int | None = None
    for lineno, line in enumerate(raw, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _WORD_LINE_RE.match(line)
        if not m:
            raise RuntimeError(
                f"{path}:{lineno}: linha inválida (esperado "
                f"[MM:SS.mmm → MM:SS.mmm] texto): {line[:80]!r}")
        seg_tag, m1, s1, m2, s2, text = m.groups()
        start = int(m1) * 60 + float(s1)
        end = int(m2) * 60 + float(s2)
        if end < start:
            raise RuntimeError(f"{path}:{lineno}: end < start ({line[:80]!r})")
        if seg_tag is not None:
            si = int(seg_tag)
        elif last_seg is not None:
            si = last_seg
        else:
            raise RuntimeError(
                f"{path}:{lineno}: linha nova sem [sNN] antes de qualquer segmento")
        if si not in buckets:
            buckets[si] = []
            order.append(si)
        buckets[si].append(Word(text, start, end))
        last_seg = si
    segments = []
    for si in sorted(order):
        words = buckets[si]
        text = " ".join(w.text.strip() for w in words
                        if w.text and w.text.strip())
        segments.append(Segment(
            text=text, start=min(w.start for w in words),
            end=max(w.end for w in words), words=words))
    return segments


# ----------------------------------------------------------------------------
# Vocabulário customizado (pós-processamento explícito, match exato)
# ----------------------------------------------------------------------------

def load_custom_words(path: str | None) -> dict[str, str]:
    """custom_words.json → {forma_errada_lower: forma_correta}.

    Sem heurística: só troca palavra inteira (case-insensitive) listada
    explicitamente. whisper.cpp não tem prompting de vocabulário — por isso o
    mecanismo honesto é pós-processamento registrado, não "influência no modelo".
    """
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"custom_words inválido ({path}): {e}")
    words = data.get("words", data) if isinstance(data, dict) else data
    if not isinstance(words, list):
        raise RuntimeError(f"custom_words ({path}): esperado lista ou {{\"words\": [...]}}")
    mapping = {}
    for w in words:
        if isinstance(w, str) and w.strip():
            mapping[w.strip().lower()] = w.strip()
        elif isinstance(w, dict) and w.get("from") and w.get("to"):
            mapping[str(w["from"]).strip().lower()] = str(w["to"]).strip()
    return mapping


def apply_custom_words(segments: list[Segment], mapping: dict[str, str]) -> int:
    """Aplica o vocabulário (palavra inteira, timestamps intactos). Retorna nº de trocas."""
    if not mapping:
        return 0
    n = 0
    for seg in segments:
        for w in seg.words:
            key = w.text.strip().lower()
            if key in mapping and w.text.strip() != mapping[key]:
                w.text = (" " + mapping[key]) if w.text[:1].isspace() else mapping[key]
                n += 1
        seg.text = " ".join(x.text.strip() for x in seg.words
                            if x.text and x.text.strip())
    return n


# ----------------------------------------------------------------------------
# Aprovação
# ----------------------------------------------------------------------------

def approve(session: Path, segments: list[Segment]) -> Path:
    """Persiste a versão aprovada (raw preservado) e retorna o path."""
    raw = session / "transcript.raw.json"
    if not raw.exists():
        cur = session / "transcript.json"
        if cur.exists():
            raw.write_text(cur.read_text(encoding="utf-8"), encoding="utf-8")
    final = session / "transcript.json"
    save_transcript(segments, final, status=APPROVED)
    return final


# ----------------------------------------------------------------------------
# Grounding (título/destaque sustentados pela transcrição aprovada)
# ----------------------------------------------------------------------------

def _content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", text or "", re.UNICODE)
            if w.lower() not in _STOPWORDS and len(w) > 1}


def validate_grounding(title: str, approved_text: str) -> list[str]:
    """Palavras de conteúdo do título ausentes da transcrição aprovada.

    Vazio = título sustentado. Nunca rejeita silenciosamente: o chamador
    sinaliza no manifest (title_warnings) para o usuário decidir.
    """
    approved = {w.lower() for w in re.findall(r"\w+", approved_text or "", re.UNICODE)}
    return sorted(w for w in _content_words(title) if w not in approved)


def validate_highlights(title: str, approved_text: str) -> list[str]:
    """Palavras destacadas (do título) que não existem na transcrição."""
    from .video import highlight_words_from_title
    approved = {w.lower() for w in re.findall(r"\w+", approved_text or "", re.UNICODE)}
    return sorted(w for w in highlight_words_from_title(title)
                  if w.lower() not in approved)
