"""transcribe.py — orquestra backend GPU-Intel primeiro, CPU como fallback.

Ordem (auto): vulkan (whisper.cpp) -> openvino (GPU) -> cpu (faster-whisper).
Fallback NUNCA silencioso: sempre loga "GPU transcription backend
unavailable: <reason> / Falling back to CPU."
Cada backend GPU é tentado no máximo 1x (sem loop, sem retry infinito).
"""
import os
import time
from pathlib import Path
from .models import Word, Segment
from .config import (
    WHISPER_MODEL_SIZE, WHISPER_LANGUAGE, CLIPPER_CPU_THREADS,
    CLIPPER_COMPUTE_TYPE, CLIPPER_DEVICE, CLIPPER_RAM_LIMIT_GB,
    CLIPPER_TRANSCRIBE_BACKEND,
)
from .cache import fingerprint, load_transcript, save_transcript
from . import intel as intel_hw
from . import backends as gpu_backends


def _apply_process_limits() -> None:
    # Auditoria (RAM): RLIMIT_AS foi removido de propósito. Ele limita ESPAÇO
    # VIRTUAL (mmaps, stacks, libs), não RSS — num run longo (1h40) isso causa
    # MemoryError em pontos aleatórios (ex: OpenVINO/torch) sem proteger a RAM
    # de verdade (Linux ignora RLIMIT_RSS), e o limite é herdado pelos filhos
    # ffmpeg. Proteção real: modelo int8, threads limitados, áudio em chunks
    # (backends GPU) ou decode único de ~0.4 GB (CPU, 100 min).
    os.environ.setdefault("OMP_NUM_THREADS", str(CLIPPER_CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CLIPPER_CPU_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CLIPPER_CPU_THREADS))


def _transcribe_cpu(video_path: str) -> list[Segment]:
    from faster_whisper import WhisperModel  # import tardio (pesado)

    print(f"[1/5] Transcrevendo com faster-whisper ({WHISPER_MODEL_SIZE}, "
          f"device={CLIPPER_DEVICE}, compute={CLIPPER_COMPUTE_TYPE}, "
          f"threads={CLIPPER_CPU_THREADS}, ram_limit={CLIPPER_RAM_LIMIT_GB}GB) — CPU fallback")
    ctype = CLIPPER_COMPUTE_TYPE
    if CLIPPER_DEVICE == "cpu" and ctype not in ("int8", "int8_float16", "float32", "int8_float32"):
        print("   ! compute_type inválido p/ CPU, forçando int8")
        ctype = "int8"
    model = WhisperModel(WHISPER_MODEL_SIZE, device=CLIPPER_DEVICE, compute_type=ctype,
                         cpu_threads=CLIPPER_CPU_THREADS, num_workers=1)
    raw_segments, info = model.transcribe(
        video_path, word_timestamps=True, vad_filter=True, language=WHISPER_LANGUAGE,
    )
    segments: list[Segment] = []
    for seg in raw_segments:
        words = [Word(w.word.strip(), w.start, w.end) for w in (seg.words or [])]
        segments.append(Segment(text=seg.text.strip(), start=seg.start, end=seg.end, words=words))
    total = segments[-1].end if segments else 0
    print(f"   -> {len(segments)} segmentos, ~{total/60:.1f}min "
          f"(idioma: {info.language}, conf: {info.language_probability:.2f})")
    return segments


def transcribe(video_path: str, cache_dir: Path | None = None, force: bool = False,
               backend: str | None = None, metrics=None) -> list[Segment]:
    fp = fingerprint(video_path) if cache_dir else None
    if cache_dir and not force and fp:
        cached = load_transcript(cache_dir, fp)
        if cached is not None:
            if metrics is not None:
                try:
                    from . import intel as _intel
                    _d = _intel.describe(backend or CLIPPER_TRANSCRIBE_BACKEND)
                    metrics.set_transcription(
                        model=WHISPER_MODEL_SIZE,
                        backend_requested=(backend or CLIPPER_TRANSCRIBE_BACKEND),
                        backend_used="cache", device=CLIPPER_DEVICE,
                        gpu=_d.get("gpu_name") or None,
                        time_sec=0.0, segments=len(cached),
                        fallback=False, tried_backends=["cache"], error=None,
                    )
                except Exception:
                    pass
            return cached

    _apply_process_limits()
    requested = backend or CLIPPER_TRANSCRIBE_BACKEND
    # preflight já imprimiu o relatório GPU completo; aqui só o backend
    # efetivo (visível também em uso standalone do módulo)
    info = intel_hw.describe(requested)
    print(f"Transcription backend: {info['transcribe_backend'].upper()}"
          + (f" ({info['transcribe_reason']})" if info["transcribe_backend"] == "cpu" else ""))

    order = [info["transcribe_backend"]] if info["transcribe_backend"] not in ("cpu", "unavailable") else []
    require_gpu = requested == "gpu"
    if info["transcribe_backend"] == "unavailable":
        # Modo 'gpu': nunca cair para CPU em silêncio — falha clara e relatório.
        err = (f"GPU transcription requested (--transcribe-backend gpu) but unavailable: "
               f"{info['transcribe_reason']}. Use 'auto' para fallback explícito à CPU "
               f"ou instale o runtime (dnf install -y intel-level-zero / whisper.cpp).")
        if metrics is not None:
            try:
                metrics.set_transcription(
                    model=WHISPER_MODEL_SIZE, backend_requested=requested,
                    backend_used=None, device=None, gpu=info.get("gpu_name") or None,
                    time_sec=0.0, segments=None, fallback=False,
                    tried_backends=[], error=f"RuntimeError: {err}",
                )
            except Exception:
                pass
        raise RuntimeError(err)
    if requested == "auto":
        # tenta vulkan e openvino nesta ordem se disponíveis, 1x cada
        for cand in ("vulkan", "openvino"):
            if cand not in order:
                b, _ = intel_hw.resolve_transcribe_backend(cand)
                if b != "cpu":
                    order.append(cand)
    order.append("cpu")  # fallback final sempre

    tried: list[str] = []
    t0 = time.time()
    last_err: str | None = None
    for b in order:
        if b in tried:
            continue
        tried.append(b)
        try:
            if b == "vulkan":
                print("[1/5] Transcrevendo com whisper.cpp (Vulkan, B580)...")
                segments = gpu_backends.transcribe_vulkan(
                    video_path, model_size=WHISPER_MODEL_SIZE, language=WHISPER_LANGUAGE)
            elif b == "openvino":
                print("[1/5] Transcrevendo com OpenVINO (device GPU, B580)...")
                segments = gpu_backends.transcribe_openvino(
                    video_path, model_size=WHISPER_MODEL_SIZE, language=WHISPER_LANGUAGE)
            else:
                segments = _transcribe_cpu(video_path)
        except RuntimeError as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"GPU transcription backend unavailable: {e}")
            if b != "cpu" and not require_gpu:
                print("Falling back to CPU.")
            if b == "cpu":
                print("CPU backend falhou.")
            if b == "cpu" or require_gpu:
                if metrics is not None:
                    try:
                        metrics.set_transcription(
                            model=WHISPER_MODEL_SIZE, backend_requested=requested,
                            backend_used=None, device=CLIPPER_DEVICE,
                            gpu=info.get("gpu_name") or None,
                            time_sec=round(time.time() - t0, 3), segments=None,
                            fallback=len(tried) > 1, tried_backends=list(tried),
                            error=last_err,
                        )
                        metrics.add_retries(max(0, len(tried) - 1))
                    except Exception:
                        pass
                raise
            continue
        if cache_dir and fp:
            save_transcript(cache_dir, fp, segments)
        if metrics is not None:
            try:
                metrics.set_transcription(
                    model=WHISPER_MODEL_SIZE, backend_requested=requested,
                    backend_used=b, device="GPU" if b in ("vulkan", "openvino") else CLIPPER_DEVICE,
                    gpu=info.get("gpu_name") or None,
                    time_sec=round(time.time() - t0, 3), segments=len(segments),
                    fallback=len(tried) > 1, tried_backends=list(tried), error=None,
                )
                metrics.add_retries(max(0, len(tried) - 1))
            except Exception:
                pass
        return segments
    raise RuntimeError("nenhum backend de transcrição disponível")
