"""translab.py — laboratório A/B de transcrição (SEM VAD).

Por que existe: os experimentos em docs/20260919_0813_vad-experiment.md provaram que o VAD
Silero prejudica timestamps (onset −0,67 s, cauda colapsada, truncagem de ~22 s
em gameplay). Este laboratório compara estratégias **sem VAD** sobre o mesmo
áudio, com métricas objetivas e artefatos reproduzíveis.

Eixos comparáveis (nada aqui toca o pipeline produtivo nem as captions):

    áudio:    original | normalize | clean   (só p/ transcrição; o vídeo final
                                              nunca usa esse áudio)
    modo:     chunks (30 s) | global (áudio inteiro de uma vez)
    contexto: segundos extras decodificados em cada chunk, descartados depois
              de parsear (só o intervalo efetivo vira Word)
    modelo:   qualquer ggml-<nome>.bin presente em models/ (ex: medium)

Uso:
    python clipper.py transcribe-lab VIDEO [--audio ...] [--mode ...]
                                           [--context S] [--whisper-model M]
                                           [--start S] [--dur D]
    python clipper.py lab-compare EXP_A EXP_B

Saída: debug/transcription-lab/experiment-NNN/{config.json, transcript.json,
transcript.txt, words.txt, metrics.json}. lab-compare lê dois experimentos e
imprime tabela palavra-por-palavra + resumo (também salvos em compare.txt/.json
dentro de EXP_B).
"""
import difflib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .backends import CHUNK_SECONDS, SAMPLE_RATE, probe_duration
from .models import Segment, Word
from .video import clean_caption_text as _clean_text
from .video import is_special_token as _is_special

LAB_ROOT = Path("debug/transcription-lab")

# Cadeia mínima, cada filtro com justificativa (KISS/YAGNI: se não medir ganho,
# o preset sai — ver README § laboratório). Nenhum promete recuperar clipping
# severo (informação destruída não volta).
AUDIO_PRESETS = {
    # Controle: exatamente o que o pipeline produtivo extrai hoje.
    "original": None,
    # Só nivelamento: mesmo alvo do CLIPPER_TRANSCRIBE_AUDIO_FILTER existente.
    "normalize": "loudnorm=I=-16:TP=-1.5:LRA=11",
    # Ruído grave de gameplay/vento + volume irregular + nivelamento.
    # highpass: corta sub-grave que mascara a voz; dynaudnorm: nivela fala
    # baixa sem estourar picos; loudnorm: alvo final padrão.
    "clean": "highpass=f=80,dynaudnorm,loudnorm=I=-16:TP=-1.5:LRA=11",
    # Compressão dinâmica: aproxima fala baixa do teto sem amplificar picos.
    "compressed": ("acompressor=threshold=-20dB:ratio=4:attack=20:release=200,"
                   "loudnorm=I=-16:TP=-1.5:LRA=11"),
    # Redução adaptativa de ruído (sem perfil): teste contra gameplay.
    "denoised": "afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
    # Reconstrução de picos clipados (interpolação; não recupera destruído).
    "declipped": "adeclip,loudnorm=I=-16:TP=-1.5:LRA=11",
}

CHUNK_EPS = 0.001  # tolerância p/ palavra exatamente na borda do chunk


# ----------------------------------------------------------------------------
# 1. Áudio (extração só p/ transcrição — o vídeo final nunca passa por aqui)
# ----------------------------------------------------------------------------

def resolve_preset(name: str) -> str | None:
    """Nome do preset → cadeia -af do ffmpeg. Erro claro em nome inválido."""
    if name not in AUDIO_PRESETS:
        raise ValueError(
            f"áudio '{name}' desconhecido — presets: {sorted(AUDIO_PRESETS)}")
    return AUDIO_PRESETS[name]


def _extract_wav(video: str, start: float, dur: float | None,
                 audio_filter: str | None, dest: Path) -> Path:
    """Extrai [start, start+dur) em 16 kHz mono. Erro contextualizado."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(max(0.0, start))]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", video]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(
            f"ffmpeg extração [{start:.1f}s +{dur}s, af={audio_filter!r}] "
            f"rc={r.returncode}: {(r.stderr or '')[:300]}")
    return dest


def iter_chunk_wavs(video: str, start: float, dur: float | None,
                    chunk_sec: int, context_sec: float,
                    audio_filter: str | None, tmpdir: Path):
    """Gera (wav, eff_start, win_start, win_end) por chunk.

    Com contexto, o wav cobre [s-ctx, s+chunk+ctx) mas só words com
    `win_start <= start < win_end` sobrevivem (ver clip_to_window) — timestamps
    de contexto nunca vazam para o resultado.
    """
    total = probe_duration(video) or 0.0
    end = (start + dur) if dur else total
    if total and end > total:
        end = total
    s = start
    idx = 0
    while s < end - CHUNK_EPS:
        win_end = min(s + chunk_sec, end)
        ext_start = max(0.0, s - context_sec)
        ext_end = min(win_end + context_sec, total) if total else win_end + context_sec
        wav = tmpdir / f"lab_{idx:04d}.wav"
        _extract_wav(video, ext_start, ext_end - ext_start, audio_filter, wav)
        yield wav, ext_start, s, win_end
        wav.unlink(missing_ok=True)
        s = win_end
        idx += 1


# ----------------------------------------------------------------------------
# 2. Whisper sem VAD (invocação direta; sem -vm/--vad em nenhum caminho)
# ----------------------------------------------------------------------------

def _whisper_bin() -> str:
    from .backends import _whisper_cpp_bin
    binary = _whisper_cpp_bin()
    if not binary:
        raise RuntimeError("binário whisper-cpp/whisper-cli não encontrado "
                           "(PATH ou thirdparty/whisper.cpp/build/bin)")
    return binary


def resolve_model(model_size: str) -> Path:
    """ggml-<size>.bin em models/ (ou ~/.cache/whisper). Erro lista o que há."""
    for cand in (Path(f"models/ggml-{model_size}.bin"),
                 Path.home() / ".cache" / "whisper" / f"ggml-{model_size}.bin"):
        if cand.exists():
            return cand
    have = sorted(p.name for p in Path("models").glob("ggml-*.bin")) or ["(vazio)"]
    raise RuntimeError(f"modelo ggml-{model_size}.bin ausente — disponíveis: {have}")


def run_whisper_file(wav: Path, model_size: str,
                     language: str | None = None,
                     temp: float | None = None,
                     best_of: int | None = None) -> tuple[list, float]:
    """Roda whisper-cli SEM VAD sobre um wav; retorna (transcription, segundos).

    temp/best_of (opcionais, p/ experimentos de decodificação): -tp/-bo."""
    binary = _whisper_bin()
    model = resolve_model(model_size)
    stem = wav.with_suffix("")
    out_json = wav.with_suffix(".json")
    if out_json.exists():
        out_json.unlink()
    cmd = [binary, "-m", str(model), "-f", str(wav),
           "-ojf", "-of", str(stem), "-l", language or "auto", "-nt"]
    if temp is not None:
        cmd += ["-tp", str(temp)]
    if best_of is not None:
        cmd += ["-bo", str(best_of)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    if r.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"whisper.cpp rc={r.returncode}: {(r.stderr or '')[:300]}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    out_json.unlink(missing_ok=True)
    return data.get("transcription", []), dt


def _even_words(text: str, start: float, end: float) -> list[Word]:
    tokens = text.split()
    if not tokens or end <= start:
        return []
    dur = (end - start) / len(tokens)
    return [Word(t, start + i * dur, start + (i + 1) * dur)
            for i, t in enumerate(tokens)]


def parse_transcription(items: list, offset: float) -> list[Segment]:
    """JSON do whisper-cli → Segments. Mesmas regras do pipeline (tokens de
    controle filtrados, texto limpo, offsets ms→s + offset do chunk)."""
    segments: list[Segment] = []
    for tr in items:
        words = []
        for w in (tr.get("tokens", []) or []):
            text = w.get("text") or ""
            if not text.strip() or _is_special(text):
                continue
            off = w.get("offsets", {}) or {}
            word = Word(text,
                        offset + float(off.get("from", 0)) / 1000.0,
                        offset + float(off.get("to", 0)) / 1000.0)
            word_p = w.get("p")
            words.append((word, word_p))
        text = _clean_text((tr.get("text") or "").strip())
        off = tr.get("offsets", {}) or {}
        s = offset + float(off.get("from", 0)) / 1000.0
        e = offset + float(off.get("to", 0)) / 1000.0
        plain = [w for w, _ in words]
        if not plain:
            plain = _even_words(text, s, e)
            probs = [None] * len(plain)
        else:
            probs = [p for _, p in words]
        for w, p in zip(plain, probs):
            w.prob = p  # type: ignore[attr-defined] (só no lab; não vaza p/ cache)
        segments.append(Segment(text=text, start=s, end=e, words=plain))
    return segments


def clip_to_window(segments: list[Segment], win_start: float,
                   win_end: float) -> list[Segment]:
    """Descarta words fora do intervalo efetivo (contexto só ajuda o decode).

    Mantém a word se seu INÍCIO está na janela; corta o fim estourado na borda
    (nunca estende). Segmentos sem words sobreviventes são descartados.
    """
    kept: list[Segment] = []
    for seg in segments:
        words = [w for w in seg.words
                 if win_start - CHUNK_EPS <= w.start < win_end - CHUNK_EPS]
        for w in words:
            if w.end > win_end:
                w.end = win_end
        if not words:
            continue
        s = max(seg.start, win_start)
        e = min(max(w.end for w in words), win_end)
        kept.append(Segment(text=seg.text, start=s, end=e, words=words))
    return kept


# ----------------------------------------------------------------------------
# 3. Métricas (função pura — testável sem binário)
# ----------------------------------------------------------------------------

def compute_metrics(segments: list[Segment]) -> dict:
    """Métricas objetivas da transcrição. Sem achismo: contagens e spans."""
    words = [w for s in segments for w in s.words]
    n = len(words)
    deg = sum(1 for w in words if w.end <= w.start)
    starts = [w.start for w in words]
    ends = [w.end for w in words]
    pairs = [(w.start, w.end) for w in words]
    overlap = sum(1 for prev, cur in zip(words, words[1:]) if cur.start < prev.end)
    dup = len(pairs) - len(set(pairs)) if pairs else 0
    probs = [w.prob for w in words  # type: ignore[attr-defined]
             if getattr(w, "prob", None) is not None]
    text = " ".join(w.text.strip() for w in words)
    return {
        "word_count": n,
        "degenerate_word_count": deg,
        "degenerate_ratio": round(deg / n, 4) if n else 0.0,
        "segment_count": len(segments),
        "timestamp_min": round(min(starts), 3) if starts else None,
        "timestamp_max": round(max(ends), 3) if ends else None,
        "speech_span_sec": round(max(ends) - min(starts), 3) if starts else 0.0,
        "overlap_count": overlap,
        "duplicate_timestamp_count": dup,
        "avg_prob": round(sum(probs) / len(probs), 4) if probs else None,
        "min_prob": round(min(probs), 4) if probs else None,
        "transcript_text": text,
    }


# ----------------------------------------------------------------------------
# 4. Experimento (execução + artefatos reproduzíveis)
# ----------------------------------------------------------------------------

def next_experiment_dir(root: Path = LAB_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    taken = {p.name for p in root.iterdir() if p.is_dir()}
    i = 1
    while f"experiment-{i:03d}" in taken:
        i += 1
    d = root / f"experiment-{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def words_dump(segments: list[Segment]) -> str:
    """Uma word por linha: [MM:SS.mmm → MM:SS.mmm] WORD (fácil de ler/diffar)."""
    def fmt(t: float) -> str:
        return f"{int(t // 60):02d}:{t % 60:06.3f}"
    return "\n".join(f"[{fmt(w.start)} → {fmt(w.end)}] {w.text.strip()}"
                     for s in segments for w in s.words) + "\n"


def run_experiment(video: str, audio: str = "original", mode: str = "chunks",
                   context_sec: float = 0.0, model_size: str = "medium",
                   language: str | None = None, start: float = 0.0,
                   dur: float | None = None,
                   temp: float | None = None, best_of: int | None = None,
                   root: Path = LAB_ROOT) -> Path:
    """Executa um experimento e salva config/transcript/metrics. Retorna o dir."""
    if mode not in ("chunks", "global"):
        raise ValueError(f"mode '{mode}' inválido — chunks|global")
    if context_sec < 0:
        raise ValueError("--context deve ser >= 0")
    if mode == "global" and context_sec:
        raise ValueError("--context só faz sentido com --mode chunks")
    audio_filter = resolve_preset(audio)
    resolve_model(model_size)  # falha cedo, antes de extrair áudio

    exp = next_experiment_dir(root)
    tmpdir = Path(tempfile.mkdtemp(prefix="translab_"))
    segments: list[Segment] = []
    proc_time = 0.0
    try:
        if mode == "global":
            wav = tmpdir / "full.wav"
            total = probe_duration(video) or 0.0
            span = (dur if dur is not None
                    else (total - start if total else None))
            _extract_wav(video, start, span, audio_filter, wav)
            items, dt = run_whisper_file(wav, model_size, language,
                                           temp=temp, best_of=best_of)
            proc_time = dt
            segments = parse_transcription(items, start)
            wav.unlink(missing_ok=True)
        else:
            for wav, ext_start, win_start, win_end in iter_chunk_wavs(
                    video, start, dur, CHUNK_SECONDS, context_sec,
                    audio_filter, tmpdir):
                items, dt = run_whisper_file(wav, model_size, language,
                                             temp=temp, best_of=best_of)
                proc_time += dt
                segs = parse_transcription(items, ext_start)
                if context_sec:
                    segs = clip_to_window(segs, win_start, win_end)
                segments.extend(segs)
    finally:
        try:
            for leftover in tmpdir.glob("*.wav"):
                leftover.unlink()
            tmpdir.rmdir()
        except Exception:
            pass

    metrics = compute_metrics(segments)
    metrics.update({
        "processing_time_sec": round(proc_time, 2),
        "model": model_size,
        "backend": "whisper.cpp (sem VAD)",
        "vad_enabled": False,
        "audio_preprocessing": audio,
        "audio_filter": audio_filter,
        "chunk_mode": mode,
        "context_sec": context_sec,
        "range_start": start,
        "range_dur": dur,
        "decode_temp": temp,
        "decode_best_of": best_of,
    })
    config = {"video": str(video), "audio": audio, "mode": mode,
              "context_sec": context_sec, "model": model_size,
              "language": language or "auto", "start": start, "dur": dur,
              "temp": temp, "best_of": best_of,
              "vad_enabled": False}
    (exp / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp / "transcript.json").write_text(json.dumps(
        {"segments": [
            {"text": s.text, "start": s.start, "end": s.end,
             "words": [{"text": w.text, "start": w.start, "end": w.end,
                        "p": getattr(w, "prob", None)} for w in s.words]}
            for s in segments]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (exp / "transcript.txt").write_text(
        "\n".join(s.text for s in segments) + "\n", encoding="utf-8")
    (exp / "words.txt").write_text(words_dump(segments), encoding="utf-8")
    (exp / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lab] {exp} :: {metrics['word_count']} words, "
          f"{metrics['degenerate_word_count']} deg "
          f"({metrics['degenerate_ratio']:.1%}), {proc_time:.1f}s")
    return exp


# ----------------------------------------------------------------------------
# 5. Comparação palavra-por-palavra (função pura no núcleo)
# ----------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def compare_word_lists(a: list[dict], b: list[dict]) -> dict:
    """Alinha duas listas de words ({text,start,end}) por texto normalizado.

    Retorna linhas (match/missing/extra com Δinício/Δfim) + resumo. Nunca faz
    média de timestamps: divergência é registrada, não escondida.
    """
    na = [_norm(w["text"]) for w in a]
    nb = [_norm(w["text"]) for w in b]
    sm = difflib.SequenceMatcher(None, na, nb, autojunk=False)
    rows: list[dict] = []
    d_starts, d_ends = [], []
    for tag, i0, i1, j0, j1 in sm.get_opcodes():
        if tag == "equal":
            for wa, wb in zip(a[i0:i1], b[j0:j1]):
                ds = round(wb["start"] - wa["start"], 3)
                de = round(wb["end"] - wa["end"], 3)
                d_starts.append(abs(ds))
                d_ends.append(abs(de))
                rows.append({"type": "match", "word": wa["text"],
                             "a": [wa["start"], wa["end"]],
                             "b": [wb["start"], wb["end"]],
                             "d_start": ds, "d_end": de,
                             "d_dur": round((wb["end"] - wb["start"])
                                            - (wa["end"] - wa["start"]), 3)})
        elif tag == "replace":
            for wa in a[i0:i1]:
                rows.append({"type": "missing_in_b", "word": wa["text"],
                             "a": [wa["start"], wa["end"]], "b": None})
            for wb in b[j0:j1]:
                rows.append({"type": "extra_in_b", "word": wb["text"],
                             "a": None, "b": [wb["start"], wb["end"]]})
        elif tag == "delete":
            for wa in a[i0:i1]:
                rows.append({"type": "missing_in_b", "word": wa["text"],
                             "a": [wa["start"], wa["end"]], "b": None})
        else:
            for wb in b[j0:j1]:
                rows.append({"type": "extra_in_b", "word": wb["text"],
                             "a": None, "b": [wb["start"], wb["end"]]})
    n_match = sum(1 for r in rows if r["type"] == "match")
    return {
        "rows": rows,
        "summary": {
            "words_a": len(a), "words_b": len(b), "matched": n_match,
            "missing_in_b": sum(1 for r in rows if r["type"] == "missing_in_b"),
            "extra_in_b": sum(1 for r in rows if r["type"] == "extra_in_b"),
            "mean_abs_d_start": round(sum(d_starts) / len(d_starts), 3)
            if d_starts else None,
            "mean_abs_d_end": round(sum(d_ends) / len(d_ends), 3)
            if d_ends else None,
            "max_abs_d_start": round(max(d_starts), 3) if d_starts else None,
        },
    }


def _load_exp_words(exp_dir: Path) -> list[dict]:
    data = json.loads((exp_dir / "transcript.json").read_text(encoding="utf-8"))
    return [w for s in data.get("segments", []) for w in s.get("words", [])]


def compare_experiments(dir_a: str, dir_b: str) -> Path:
    """Compara dois experimentos; salva compare.txt/compare.json em EXP_B."""
    pa, pb = Path(dir_a), Path(dir_b)
    for p in (pa, pb):
        if not (p / "transcript.json").exists():
            raise RuntimeError(f"{p} não é um experimento (sem transcript.json)")
    result = compare_word_lists(_load_exp_words(pa), _load_exp_words(pb))
    lines = [f"# compare {pa.name} (A) vs {pb.name} (B)",
             f"# matched={result['summary']['matched']} "
             f"missing_in_b={result['summary']['missing_in_b']} "
             f"extra_in_b={result['summary']['extra_in_b']} "
             f"mean|d_start|={result['summary']['mean_abs_d_start']} "
             f"mean|d_end|={result['summary']['mean_abs_d_end']}",
             f"{'PALAVRA':<24}{'A':<26}{'B':<26}ΔINI   ΔFIM",
             "-" * 88]
    for r in result["rows"]:
        if r["type"] == "match":
            a = f"{r['a'][0]:.2f}–{r['a'][1]:.2f}"
            b = f"{r['b'][0]:.2f}–{r['b'][1]:.2f}"
            lines.append(f"{r['word']:<24}{a:<26}{b:<26}"
                         f"{r['d_start']:+.2f} {r['d_end']:+.2f}")
        elif r["type"] == "missing_in_b":
            lines.append(f"{r['word']:<24}{r['a'][0]:.2f}–{r['a'][1]:.2f}"
                         f"{'—':<26}só em A")
        else:
            lines.append(f"{r['word']:<24}{'—':<26}"
                         f"{r['b'][0]:.2f}–{r['b'][1]:.2f}  só em B")
    (pb / "compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (pb / "compare.json").write_text(json.dumps(
        {"a": str(pa), "b": str(pb), **result},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"... ({len(lines) - 60} linhas em {pb / 'compare.txt'})")
    return pb
