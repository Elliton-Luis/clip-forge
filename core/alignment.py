"""alignment.py — forced alignment opt-in (Whisper continua dono do texto).

Contrato (ver docs/20260920_2050_alignment-investigation.md):
- Entrada: (áudio + segmentos do Whisper). Saída: mesmos textos, starts/ends
  re-medidos quando possível.
- `Word.timestamp_source` / `Segment.timestamp_source` explicitam a origem:
  "whisper" (original ou fallback) vs "forced_alignment" (re-medido).
- `Word.confidence` opcional (0..1) quando o backend fornece.
- Fallback controlado: qualquer falha mantém o transcript intacto, registra o
  erro em AlignmentStats + log, nunca fabrica timestamp silenciosamente.
- Proibido aqui (decisão do projeto): offsets globais, médias, heurísticas de
  drift, VAD, retoques em _group_cues, manipulação artificial de duração.

Backends:
- "off" (default): no-op — pipeline byte-idêntico ao atual.
- "whisper-refine": re-decode do PRÓPRIO Whisper (whisper.cpp/Vulkan na B580,
  fallback faster-whisper CPU) numa janela curta por segmento + casamento de
  sequência (difflib). Zero dependência nova. Ganho moderado e honesto.
- "wav2vec2": CTC real (HF jonatasgrosman/wav2vec2-large-xlsr-53-portuguese,
  lazy-import torch/transformers). Precisão de fonema; sem pacotes = fallback.
"""
import difflib
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .models import Word, Segment

ALIGN_BACKENDS = ("off", "whisper-refine", "wav2vec2")
ALIGN_VERSION = 1
WAV2VEC2_PT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
REFINE_MARGIN_SEC = 1.5  # janela extra ao redor do segmento p/ re-decode
REFINE_MIN_MATCH = 0.5  # fração mínima de palavras casadas p/ aceitar o segmento
# Gate de qualidade CTC (calibrado no experimento docs/20260920_2109_alignment-experiment.md):
# span abaixo disso NÃO é re-medição — é o modelo dizendo que a palavra não
# está no áudio (alucinação do Whisper). Fallback ao Whisper, nunca clamp.
WAV2VEC2_MIN_CONF = 0.3
WAV2VEC2_MIN_DUR = 0.04


def resolve_backend(name: str | None) -> str:
    """CLI/env → backend válido. Erro claro em nome inválido."""
    raw = (name if name is not None
           else os.environ.get("CLIPPER_ALIGN", "off")).lower()
    if raw not in ALIGN_BACKENDS:
        raise ValueError(
            f"align backend '{raw}' desconhecido — opções: {list(ALIGN_BACKENDS)}")
    return raw


@dataclass
class AlignmentStats:
    backend: str = "off"
    n_words: int = 0
    n_aligned: int = 0
    n_fallback: int = 0
    time_sec: float = 0.0
    error: str | None = None


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _sanitize_span(start: float, end: float, lo: float, hi: float,
                   fallback: tuple[float, float]) -> tuple[float, float]:
    """Valida um span re-medido; fora de (lo, hi) ou degenerado → fallback."""
    if not (start < end) or not (lo - 0.01 <= start <= hi + 0.01) \
            or not (lo - 0.01 <= end <= hi + 0.01):
        return fallback
    return (max(lo, start), min(hi, end))


def match_and_transfer(orig_words: list[Word], new_words: list[tuple[str, float, float, float | None]],
                       lo: float, hi: float) -> tuple[list[Word], int, int]:
    """Casa sequência original × re-decodificada e transfere timestamps.

    Retorna (words_finais, n_aligned, n_fallback). Texto NUNCA muda: palavras
    sem par exato mantêm start/end originais com source "whisper".
    new_words: (texto, start, end, confidence).
    """
    out: list[Word] = []
    a = [_norm(w.text) for w in orig_words]
    b = [_norm(t) for t, _, _, _ in new_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched_b: dict[int, int] = {}  # índice em b → índice em a
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                matched_b[j1 + k] = i1 + k
    b_to_a = {v: k for k, v in matched_b.items()}
    aligned = fallback = 0
    for i, w in enumerate(orig_words):
        j = b_to_a.get(i)
        if j is None:
            out.append(Word(w.text, w.start, w.end,
                            confidence=getattr(w, "confidence", None),
                            timestamp_source="whisper"))
            fallback += 1
            continue
        _, ns, ne, nc = new_words[j]
        cs, ce = _sanitize_span(ns, ne, lo, hi, (w.start, w.end))
        is_refined = (cs, ce) != (w.start, w.end)
        out.append(Word(w.text, cs, ce,
                        confidence=nc if nc is not None else getattr(w, "confidence", None),
                        timestamp_source="forced_alignment" if is_refined else "whisper"))
        if is_refined:
            aligned += 1
        else:
            fallback += 1
    # Monotonicidade: re-decode ruidoso fora de ordem não vira clamp —
    # o par violador volta ao Whisper (fallback honesto; overlaps pequenos
    # de jitter são tolerados, inversão real não).
    orig_span = {id(nw): (w.start, w.end) for nw, w in zip(out, orig_words)}

    def _revert(ww: Word) -> None:
        nonlocal aligned, fallback
        if ww.timestamp_source == "forced_alignment":
            ww.start, ww.end = orig_span[id(ww)]
            ww.timestamp_source = "whisper"
            aligned -= 1
            fallback += 1

    for i in range(1, len(out)):
        prev, cur = out[i - 1], out[i]
        if cur.end <= cur.start or cur.start < prev.start:
            _revert(cur)
            _revert(prev)
    return out, aligned, fallback


def _extract_window_wav(video_path: str, start: float, end: float, dest: Path) -> None:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(max(0.0, start)), "-t", str(max(0.5, end - start)),
         "-i", video_path, "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg não extraiu janela [{start:.2f},{end:.2f}]: "
                           f"{(r.stderr or '')[:200]}")


def _redecode_whisper_cpp(window_wav: Path, window_start: float,
                          model_size: str, language: str | None) -> list[tuple[str, float, float, float | None]]:
    """Re-decode via whisper.cpp (B580/Vulkan). Retorna (texto, start, end, prob)."""
    from . import backends as _be
    binary = _be._whisper_cpp_bin()
    if not binary:
        raise RuntimeError("binário whisper-cli ausente (ver README § Intel Arc)")
    model = Path(f"models/ggml-{model_size}.bin")
    if not model.exists():
        alt = Path.home() / ".cache" / "whisper" / f"ggml-{model_size}.bin"
        if alt.exists():
            model = alt
        else:
            raise RuntimeError(f"modelo {model} ausente")
    segs = _be._decode_chunk(binary, model, window_wav, language, window_start)
    out = []
    for s in segs:
        for w in s.words:
            out.append((w.text, w.start, w.end, getattr(w, "confidence", None)))
    return out


def _redecode_faster_whisper(window_wav: Path, model_size: str,
                             language: str | None) -> list[tuple[str, float, float, float | None]]:
    """Re-decode via faster-whisper CPU (fallback quando sem whisper.cpp)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=int(os.environ.get("CLIPPER_CPU_THREADS", "4")),
                         num_workers=1)
    raw, _ = model.transcribe(str(window_wav), word_timestamps=True,
                              vad_filter=False, language=language or "pt")
    out = []
    for seg in raw:
        for w in (seg.words or []):
            out.append((w.word.strip(), w.start, w.end,
                        float(getattr(w, "probability", 0) or 0) or None))
    return out


def _refine_segment(seg: Segment, video_path: str, model_size: str,
                    language: str | None, stats: AlignmentStats,
                    redecode_fn=None) -> Segment:
    """Refina UM segmento via re-decode em janela curta. Nunca muda o texto."""
    lo = max(0.0, seg.start - REFINE_MARGIN_SEC)
    hi = seg.end + REFINE_MARGIN_SEC
    try:
        with tempfile.TemporaryDirectory(prefix="clipper_align_") as tmp:
            wav = Path(tmp) / "win.wav"
            _extract_window_wav(video_path, lo, hi, wav)
            if redecode_fn is not None:
                new_words = redecode_fn(wav, lo)
            else:
                try:
                    new_words = _redecode_whisper_cpp(wav, lo, model_size, language)
                except RuntimeError:
                    new_words = _redecode_faster_whisper(wav, model_size, language)
        if not new_words:
            raise RuntimeError("re-decode vazio")
        words, n_al, n_fb = match_and_transfer(seg.words, new_words, lo, hi)
        stats.n_aligned += n_al
        stats.n_fallback += n_fb
        src = "forced_alignment" if n_al > 0 else "whisper"
        s = min(w.start for w in words) if words else seg.start
        e = max(w.end for w in words) if words else seg.end
        return Segment(text=seg.text, start=s, end=e, words=words,
                       timestamp_source=src)
    except Exception as e:
        stats.n_fallback += len(seg.words)
        if stats.error is None:
            stats.error = f"{type(e).__name__}: {e}"
        fb = [Word(w.text, w.start, w.end,
                   confidence=getattr(w, "confidence", None),
                   timestamp_source="whisper") for w in seg.words]
        return Segment(text=seg.text, start=seg.start, end=seg.end,
                       words=fb, timestamp_source="whisper")


def _align_wav2vec2(segments: list[Segment], video_path: str,
                    stats: AlignmentStats) -> list[Segment]:
    """CTC real com wav2vec2-PT (lazy). Falha controlada → fallback por segmento."""
    try:
        import numpy as np
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except Exception as e:
        raise RuntimeError(
            "backend wav2vec2 exige 'pip install torch transformers torchaudio' "
            f"({e})")
    out: list[Segment] = []
    try:
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_PT_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_PT_MODEL)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"modelo {WAV2VEC2_PT_MODEL} indisponível ({e})")
    vocab = {c.lower(): i for c, i in processor.tokenizer.get_vocab().items()}
    for seg in segments:
        try:
            lo = max(0.0, seg.start - REFINE_MARGIN_SEC)
            hi = seg.end + REFINE_MARGIN_SEC
            with tempfile.TemporaryDirectory(prefix="clipper_align_") as tmp:
                wav = Path(tmp) / "win.wav"
                _extract_window_wav(video_path, lo, hi, wav)
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", str(wav), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
                    capture_output=True, timeout=120)
                audio = np.frombuffer(r.stdout, dtype=np.float32).copy()
            if audio.size < 1600:
                raise RuntimeError("janela de áudio vazia")
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt",
                               padding=True)
            with torch.inference_mode():
                logits = model(inputs.input_values).logits[0]
            logp = torch.log_softmax(logits, dim=-1)
            # Transcript construído DAS words (texto original preservado):
            # só chars do vocabulário; palavra sem char alinhável → fallback.
            cleans: list[str] = []
            bounds: list[tuple[int, int] | None] = []
            pos = 0
            for w in seg.words:
                c = "".join(ch for ch in w.text.strip().lower()
                            if ch in vocab and ch != "|")
                if not c:
                    bounds.append(None)
                    continue
                bounds.append((pos, pos + len(c)))
                cleans.append(c)
                pos += len(c) + 1  # +1 = separador '|'
            text = "|".join(cleans)
            if not text:
                raise RuntimeError("texto sem caracteres alinháveis")
            tokens = [vocab[c] for c in text]
            trellis = _get_trellis(logp, tokens, blank=0)
            path = _backtrack(trellis, logp, tokens, blank=0)
            if path is None:
                raise RuntimeError("backtrack falhou")
            char_spans = _merge_repeats(path, len(tokens))
            words, n_al, n_fb = _chars_to_words(seg, char_spans, bounds, lo, hi)
            stats.n_aligned += n_al
            stats.n_fallback += n_fb
            src = "forced_alignment" if n_al > 0 else "whisper"
            s = min(w.start for w in words) if words else seg.start
            e = max(w.end for w in words) if words else seg.end
            out.append(Segment(text=seg.text, start=s, end=e, words=words,
                               timestamp_source=src))
        except Exception as e:
            stats.n_fallback += len(seg.words)
            if stats.error is None:
                stats.error = f"{type(e).__name__}: {e}"
            fb = [Word(w.text, w.start, w.end,
                       confidence=getattr(w, "confidence", None),
                       timestamp_source="whisper") for w in seg.words]
            out.append(Segment(text=seg.text, start=seg.start, end=seg.end,
                               words=fb, timestamp_source="whisper"))
    return out


def _get_trellis(emission, tokens, blank=0):
    """Trellis CTC (mesma recorrência do tutorial de forced alignment)."""
    import torch
    n_frame = emission.size(0)
    n_tok = len(tokens) + 1
    trellis = torch.empty((n_frame, n_tok))
    trellis[0, 0] = emission[0, tokens[0]]
    trellis[1:, 0] = torch.cumsum(emission[1:, blank], dim=0)
    trellis[0, 1:] = -float("inf")
    trellis[-n_tok + 1:, 0] = float("inf")
    for t in range(1, n_frame):
        e = emission[t, tokens]
        trellis[t, 1:] = torch.maximum(
            trellis[t - 1, 1:] + e,
            trellis[t - 1, :-1] + e)
    return trellis


def _backtrack(trellis, emission, tokens, blank=0):
    """Caminho ótimo. Retorna [(frame, token_pos_ou_-1_p/blank, score)] ou None."""
    import torch
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t, blank]
        changed = (trellis[t - 1, j - 1] + emission[t, tokens[j - 1]]
                   if j > 0 else torch.tensor(-float("inf")))
        use_changed = bool(changed > stayed) and j > 0
        tok_pos = (j - 1) if use_changed else -1
        prob = float(emission[t, tokens[j - 1] if use_changed else blank].exp().item())
        path.append((t - 1, tok_pos, prob))
        if use_changed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return list(reversed(path))


def _merge_repeats(path, n_tokens):
    """Agrupa frames por posição de token. Retorna [(tok_pos, start_f, end_f, score)]."""
    groups: dict[int, list] = {}
    order: list[int] = []
    for frame, tok_pos, prob in path:
        if tok_pos < 0:
            continue
        if tok_pos not in groups:
            groups[tok_pos] = [frame, frame, [], tok_pos]
            order.append(tok_pos)
        g = groups[tok_pos]
        g[0] = min(g[0], frame)
        g[1] = max(g[1], frame)
        g[2].append(prob)
    out = []
    for k in range(n_tokens):
        if k in groups:
            g = groups[k]
            out.append((k, g[0], g[1], sum(g[2]) / len(g[2])))
        else:
            out.append((k, 0, 0, 0.0))
    return out


def _chars_to_words(seg: Segment, char_spans, bounds, lo: float, hi: float):
    """Mapeia spans de tokens CTC → words (texto original preservado).

    char_spans: [(tok_pos, start_frame, end_frame, score)] indexado por posição
    de token; bounds[i] = (t0, t1) da word i no transcript (None = sem chars
    alinháveis → fallback Whisper). 1 frame wav2vec2 = 20 ms.
    """
    FRAME_SEC = 320.0 / 16000.0
    by_pos = {k: (s, e, p) for k, s, e, p in char_spans}
    words: list[Word] = []
    aligned = fallback = 0
    for w, b in zip(seg.words, bounds):
        if b is None:
            words.append(Word(w.text, w.start, w.end,
                              confidence=getattr(w, "confidence", None),
                              timestamp_source="whisper"))
            fallback += 1
            continue
        t0, t1 = b
        fr = [by_pos[k] for k in range(t0, t1) if k in by_pos and by_pos[k][2] > 0]
        if not fr:
            words.append(Word(w.text, w.start, w.end,
                              confidence=getattr(w, "confidence", None),
                              timestamp_source="whisper"))
            fallback += 1
            continue
        ns = lo + min(s for s, _, _ in fr) * FRAME_SEC
        ne = lo + (max(e for _, e, _ in fr) + 1) * FRAME_SEC
        conf = round(sum(p for _, _, p in fr) / len(fr), 3)
        cs, ce = _sanitize_span(ns, ne, lo, hi, (w.start, w.end))
        # Gate: confiança baixa ou span implausível = palavra não está no áudio
        # (texto alucinado). Fallback preserva o Whisper; conf fica p/ diagnóstico.
        if conf < WAV2VEC2_MIN_CONF or (ce - cs) < WAV2VEC2_MIN_DUR:
            words.append(Word(w.text, w.start, w.end, confidence=conf,
                              timestamp_source="whisper"))
            fallback += 1
            continue
        refined = (cs, ce) != (w.start, w.end)
        words.append(Word(w.text, cs, ce, confidence=conf,
                          timestamp_source="forced_alignment" if refined else "whisper"))
        aligned += refined
        fallback += (not refined)
    # Ordem contradizendo a acústica = transcript fora de ordem (repetição
    # alucinada): reverte o par ao Whisper em vez de clampar e fabricar.
    orig_span = {id(nw): (w.start, w.end) for nw, w in zip(words, seg.words)}

    def _revert(ww: Word) -> None:
        nonlocal aligned, fallback
        if ww.timestamp_source == "forced_alignment":
            ww.start, ww.end = orig_span[id(ww)]
            ww.timestamp_source = "whisper"
            aligned -= 1
            fallback += 1

    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        if cur.end <= cur.start or cur.start < prev.start:
            _revert(cur)
            _revert(prev)
    return words, aligned, fallback


def align_segments(segments: list[Segment], audio_path: str | None = None,
                   backend: str | None = None, model_size: str = "medium",
                   language: str | None = None,
                   redecode_fn=None) -> tuple[list[Segment], AlignmentStats]:
    """Ponto de entrada. Nunca altera texto; fallback preserva timestamps Whisper."""
    be = resolve_backend(backend)
    stats = AlignmentStats(backend=be, n_words=sum(len(s.words) for s in segments))
    t0 = time.time()
    if be == "off" or not segments:
        out = [Segment(text=s.text, start=s.start, end=s.end,
                       words=[Word(w.text, w.start, w.end,
                                   confidence=getattr(w, "confidence", None),
                                   timestamp_source=getattr(w, "timestamp_source", "whisper") or "whisper")
                              for w in s.words],
                       timestamp_source=getattr(s, "timestamp_source", "whisper") or "whisper")
               for s in segments]
        stats.time_sec = round(time.time() - t0, 3)
        return out, stats
    if audio_path is None:
        stats.error = "align sem audio_path: fallback para timestamps Whisper"
        print(f"   ! alignment ({be}): sem áudio — mantém Whisper")
        stats.n_fallback = stats.n_words
        stats.time_sec = round(time.time() - t0, 3)
        out, _ = align_segments(segments, backend="off")
        return out, stats
    print(f"[align] Forced alignment ({be}, v{ALIGN_VERSION}) sobre {len(segments)} segmentos...")
    try:
        if be == "whisper-refine":
            out = [_refine_segment(s, audio_path, model_size, language, stats,
                                   redecode_fn=redecode_fn) for s in segments]
        elif be == "wav2vec2":
            out = _align_wav2vec2(segments, audio_path, stats)
        else:
            raise ValueError(be)
    except Exception as e:  # falha global: nunca corromper o transcript
        stats.error = f"{type(e).__name__}: {e}"
        print(f"   ! alignment ({be}) falhou: {e} — mantém Whisper")
        out, _ = align_segments(segments, backend="off")
        out_stats = AlignmentStats(backend=be, n_words=stats.n_words,
                                   n_aligned=0, n_fallback=stats.n_words,
                                   time_sec=round(time.time() - t0, 3),
                                   error=stats.error)
        return out, out_stats
    stats.time_sec = round(time.time() - t0, 3)
    print(f"   -> align: {stats.n_aligned} words re-medidas, "
          f"{stats.n_fallback} fallback Whisper ({stats.time_sec}s)")
    if stats.error:
        print(f"   ! align: {stats.error}")
    return out, stats
