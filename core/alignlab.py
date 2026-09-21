"""alignlab.py — experimento Whisper timestamps vs Forced Alignment.

Uso:
    python clipper.py align-compare video.mkv --start 0 --dur 30 [--backend whisper-refine]

Método (nada toca o pipeline produtivo):
1. Extrai o trecho (wav 16k mono) e transcreve com o backend padrão
   (whisper.cpp/Vulkan, fallback faster-whisper CPU) → baseline Whisper.
2. Roda forced alignment sobre os mesmos segmentos (texto intacto).
3. Mede por palavra: Δinício/Δfim, cobertura (% re-medido), monotonicidade,
   energia RMS no span de cada versão (proxy acústico independente,
   report-only — nunca desloca timestamp), pausas entre palavras.
4. Salva debug/alignment-lab/<video>_<start>-<dur>/{whisper.json, aligned.json,
   compare.json, compare.txt} e imprime tabela palavra-por-palavra.

Não declara vitória por "terminou sem erro": imprime deltas e deixa os
números falarem (ver docs/20260920_2109_alignment-experiment.md).
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .models import Segment
from .alignment import align_segments

LAB_ROOT = Path("debug/alignment-lab")


def _extract_wav(video: str, start: float, dur: float, dest: Path) -> Path:
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-t", str(dur), "-i", video,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg não extraiu trecho: {(r.stderr or '')[:200]}")
    return dest


def _transcribe_wav(wav: Path, offset: float, model_size: str) -> list[Segment]:
    """Transcreve um wav com o backend padrão (vulkan → cpu)."""
    from . import backends as _be
    binary = _be._whisper_cpp_bin()
    if binary:
        from pathlib import Path as _P
        model = _P(f"models/ggml-{model_size}.bin")
        if model.exists():
            return _be._decode_chunk(binary, model, wav, None, offset)
    # Fallback CPU honesto (mesmo do pipeline).
    from faster_whisper import WhisperModel
    from .video import is_special_token as _is_special, clean_caption_text as _clean
    import os as _os
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         cpu_threads=int(_os.environ.get("CLIPPER_CPU_THREADS", "4")),
                         num_workers=1)
    raw, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=False)
    segs = []
    for s in raw:
        from .models import Word
        words = [Word((w.word or "").strip(), offset + w.start, offset + w.end)
                 for w in (s.words or [])
                 if (w.word or "").strip() and not _is_special(w.word)]
        segs.append(Segment(text=_clean((s.text or "").strip()),
                            start=offset + s.start, end=offset + s.end, words=words))
    return segs


def _rms_in_span(wav: Path, start: float, end: float, base: float) -> float:
    """Energia RMS média no span [start, end] (relativo a base=ínicio do wav)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(wav), "-ar", "16000", "-ac", "1", "-f", "f32le", "-"],
            capture_output=True, timeout=120)
        audio = np.frombuffer(r.stdout, dtype=np.float32)
        a = max(0, int((start - base) * 16000))
        b = max(a + 1, int((end - base) * 16000))
        frag = audio[a:b]
        if frag.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(frag ** 2)))
    except Exception:
        return 0.0


def run(video: str, start: float, dur: float, backend: str = "whisper-refine",
        model_size: str = "medium") -> Path:
    t0 = time.time()
    stem = Path(video).stem.replace(" ", "_")
    outdir = LAB_ROOT / f"{stem}_{start:g}-{dur:g}_{backend}"
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clipper_alignlab_") as tmp:
        wav = _extract_wav(video, start, dur, Path(tmp) / "trecho.wav")
        base_segs = _transcribe_wav(wav, start, model_size)
        n_words = sum(len(s.words) for s in base_segs)
        if n_words == 0:
            raise RuntimeError("trecho sem fala transcrita — sem o que comparar")
        aligned, stats = align_segments(base_segs, audio_path=video,
                                        backend=backend, model_size=model_size)
    # --- medição palavra-por-palavra (ordem global preservada) ---
    flat0 = [w for s in base_segs for w in s.words]
    flat1 = [w for s in aligned for w in s.words]
    assert len(flat0) == len(flat1), "aligner não pode criar/remover palavras"
    with tempfile.TemporaryDirectory(prefix="clipper_alignlab_e_") as tmp2:
        wav2 = _extract_wav(video, start, dur, Path(tmp2) / "t.wav")
        rows = []
        d_starts, d_ends = [], []
        for w0, w1 in zip(flat0, flat1):
            assert w0.text == w1.text, "texto do aligner divergiu (proibido)"
            ds = w1.start - w0.start
            de = w1.end - w0.end
            d_starts.append(abs(ds))
            d_ends.append(abs(de))
            rows.append({
                "word": w0.text, "w_start": round(w0.start, 3), "w_end": round(w0.end, 3),
                "a_start": round(w1.start, 3), "a_end": round(w1.end, 3),
                "d_start": round(ds, 3), "d_end": round(de, 3),
                "source": w1.timestamp_source,
                "rms_w": round(_rms_in_span(wav2, w0.start, w0.end, start), 4),
                "rms_a": round(_rms_in_span(wav2, w1.start, w1.end, start), 4),
            })
    import statistics as _st
    summary = {
        "video": video, "start": start, "dur": dur, "backend": backend,
        "model": model_size, "segments": len(base_segs), "words": n_words,
        "n_aligned": stats.n_aligned, "n_fallback": stats.n_fallback,
        "align_time_sec": stats.time_sec, "align_error": stats.error,
        "mean_abs_d_start": round(_st.mean(d_starts), 3) if d_starts else 0.0,
        "mean_abs_d_end": round(_st.mean(d_ends), 3) if d_ends else 0.0,
        "max_abs_d_start": round(max(d_starts), 3) if d_starts else 0.0,
        "max_abs_d_end": round(max(d_ends), 3) if d_ends else 0.0,
        "text_identical": all(a.text == b.text for a, b in zip(base_segs, aligned)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (outdir / "whisper.json").write_text(
        json.dumps([{"text": s.text, "start": s.start, "end": s.end,
                     "words": [{"text": w.text, "start": w.start, "end": w.end}
                               for w in s.words]} for s in base_segs],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "aligned.json").write_text(
        json.dumps([{"text": s.text, "start": s.start, "end": s.end,
                     "timestamp_source": s.timestamp_source,
                     "words": [{"text": w.text, "start": w.start, "end": w.end,
                                "confidence": w.confidence,
                                "timestamp_source": w.timestamp_source}
                               for w in s.words]} for s in aligned],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "compare.json").write_text(
        json.dumps({"summary": summary, "words": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# align-compare {video} [{start:g},{start + dur:g}] backend={backend}",
             f"# segments={summary['segments']} words={n_words} "
             f"aligned={stats.n_aligned} fallback={stats.n_fallback} "
             f"text_identical={summary['text_identical']}",
             f"# mean|Δstart|={summary['mean_abs_d_start']}s "
             f"mean|Δend|={summary['mean_abs_d_end']}s "
             f"max|Δstart|={summary['max_abs_d_start']}s max|Δend|={summary['max_abs_d_end']}s",
             "",
             f"{'WORD':<22}{'WHISPER':<24}{'ALIGNED':<24}{'Δs/Δe'}  src/rms_w→rms_a"]
    for r_ in rows:
        lines.append(
            f"{r_['word'][:21]:<22}"
            f"{r_['w_start']:<11.3f}{r_['w_end']:<11.3f}  "
            f"{r_['a_start']:<11.3f}{r_['a_end']:<11.3f}  "
            f"{r_['d_start']:+.3f}/{r_['d_end']:+.3f}  "
            f"{r_['source'][:5]} {r_['rms_w']:.3f}→{r_['rms_a']:.3f}")
    (outdir / "compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nArtefatos em: {outdir.resolve()}")
    return outdir
