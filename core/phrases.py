"""phrases.py — legendas por UNIDADES NATURAIS DE FALA (camada de conteúdo).

Separação arquitetural: aqui vive o CONTEÚDO (que palavras formam cada frase,
com timestamps individuais preservados para destaque/karokê futuro); a
APRESENTAÇÃO (SRT/ASS, quebra de linha, lanes, highlight, hook) continua em
core/video.py e consome spans+texto. Trocar o estilo nunca reprocessa a
transcrição.

Regras de agrupamento (documentadas, sem número mágico escondido):
- Vozes (fronteiras de token + overlap > 0.1 s) NUNCA se misturam: cada voz
  é agrupada separadamente (reuso de video._split_voices — sem diarização,
  sem nomes, só independência temporal).
- Fronteira de frase (.!?) = quebra, EXCETO continuação próxima: gap <=
  PHRASE_GAP_SEC (3.0, configurável) E (próxima começa minúscula OU é
  conjunção/marcador OU anterior termina em "…") → mesma frase, com "…" se
  a pausa for real (>= PHRASE_ELLIPSIS_GAP). Frase nova completa (sujeito
  próprio, maiúscula, sem marcador) sempre quebra — sem advinhação.
- Sem pontuação terminal + gap <= PHRASE_GAP_SEC → continua a frase;
  "…" se a pausa interna for significativa (ex: "Eu achei que... você tinha
  entendido"). Gap > tolerância = pensamento novo, sempre quebra.
- Proibido: esticar palavras, redistribuir tempo uniformemente, inventar
  timestamp, usar duração máxima ou nº de caracteres como critério de corte,
  partir palavra no meio, juntar vozes.
"""
import re
from dataclasses import dataclass, field

# Tolerância p/ duas partes pertencerem à mesma frase (investigada em ~3 s;
# NÃO cega: combinada com pontuação/sintaxe/voz abaixo).
PHRASE_GAP_SEC = 3.0
# Pausa a partir da qual a junção ganha "…" (pausa real, não jitter de medição).
PHRASE_ELLIPSIS_GAP = 0.8

# Conjunções e marcadores discursivos PT que continuam a frase anterior.
CONTINUATORS = frozenset(
    "e ou mas que porque porquê pois então aí né tá ué ah ó bem tipo assim "
    "só também nem como quando onde se porém todavia contudo logo portanto "
    "enfim aliás ou seja quer dizer".split())

TERMINAL_RE = re.compile(r"[.!?…]\s*$")


@dataclass
class Phrase:
    """Unidade natural de fala: words com timestamps individuais intactos."""
    words: list
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    voice: int = 0
    bridged_gap: float = 0.0  # maior pausa interna absorvida (0 = fluxo contínuo)
    has_ellipsis: bool = False
    word_spans: list = field(default_factory=list)  # [(texto,start,end)] p/ destaque


def _is_continuation_token(words: list) -> bool:
    """Há fronteiras estilo whisper.cpp? (espaço à esquerda = início real)."""
    return any(w.text[:1].isspace() for w in words if w.text)


def _is_token_continuation(w, boundaries: bool) -> bool:
    return boundaries and not (w.text[:1].isspace() if w.text else True)


def _strip(text: str) -> str:
    return (text or "").strip()


def _first_word(words: list) -> str:
    for w in words:
        t = _strip(w.text)
        if t:
            return t
    return ""


def _continues(prev_text: str, next_words: list, gap: float) -> tuple[bool, bool]:
    """(mesma_frase?, ellipsis?) — gap já validado <= PHRASE_GAP_SEC."""
    prev_done = bool(TERMINAL_RE.search(_strip(prev_text)))
    fw = _first_word(next_words)
    lower = bool(fw) and fw[0].islower()
    marker = bool(fw) and fw.lower().strip("….,!?;:") in CONTINUATORS
    ellipsis = gap >= PHRASE_ELLIPSIS_GAP
    if not prev_done:
        return True, ellipsis  # frase aberta continua; "…" só se pausa real
    if lower or marker or _strip(prev_text).endswith("…"):
        return True, ellipsis
    return False, False


def group_phrases(words: list, clip_start: float, clip_end: float,
                  gap_tol: float = PHRASE_GAP_SEC) -> list[Phrase]:
    """Words → Phrases. Puro e testável. Timestamps nunca reescritos."""
    from .video import valid_caption_words, _split_voices
    in_clip = sorted(
        (w for w in valid_caption_words(words)
         if w.end > clip_start and w.start < clip_end),
        key=lambda w: w.start)
    if not in_clip:
        return []
    boundaries = _is_continuation_token(in_clip)
    phrases: list[Phrase] = []
    for voice_idx, voice in enumerate(_split_voices(in_clip)):
        cur: list = []
        bridged = 0.0
        joints: list[tuple[int, bool]] = []  # (posição, ellipsis?) p/ montar texto

        def flush():
            if cur:
                phrases.append(_make_phrase(cur, voice_idx, bridged, joints))

        i, n = 0, len(voice)
        while i < n:
            w = voice[i]
            # Nunca partir palavra: continuações fecham o grupo atual.
            nxt_chain = [w]
            j = i
            while j + 1 < n and _is_token_continuation(voice[j + 1], boundaries):
                j += 1
                nxt_chain.append(voice[j])
            if not cur:
                cur = list(nxt_chain)
                i = j + 1
                continue
            gap = nxt_chain[0].start - cur[-1].end
            prev_text = " ".join(_strip(x.text) for x in cur)
            if gap > gap_tol:
                flush()
                cur, bridged, joints = list(nxt_chain), 0.0, []
            else:
                same, ell = _continues(prev_text, nxt_chain, gap)
                if same:
                    joints.append((len(cur), ell))
                    bridged = max(bridged, gap)
                    cur.extend(nxt_chain)
                else:
                    flush()
                    cur, bridged, joints = list(nxt_chain), 0.0, []
            i = j + 1
        flush()
    phrases.sort(key=lambda p: (p.start, p.end))
    return phrases


def _make_phrase(cur: list, voice: int, bridged: float,
                 joints: list[tuple[int, bool]]) -> Phrase:
    from .video import smart_join
    ell_at = {pos for pos, ell in joints if ell}
    toks: list[str] = []
    for k, w in enumerate(cur):
        if k in ell_at:
            toks.append("…")
        toks.append(w.text)
    # smart_join respeita fronteiras de token ("direita", não "dire ita";
    # "TÔ.", não "TÔ ."); "…" é pontuação e gruda como tal.
    text = smart_join(toks)
    spans = [(_strip(w.text), w.start, w.end) for w in cur]
    return Phrase(words=list(cur), text=text,
                  start=min(w.start for w in cur), end=max(w.end for w in cur),
                  voice=voice, bridged_gap=round(bridged, 3),
                  has_ellipsis="…" in toks, word_spans=spans)
