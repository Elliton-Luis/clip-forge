"""backends.py — backends de transcrição que usam a Intel Arc B580.

- audio_chunks(): extração em chunks via ffmpeg (streaming, nunca o vídeo todo na RAM).
- transcribe_openvino(): Whisper via optimum-intel + OpenVINO device GPU.
- transcribe_vulkan(): Whisper via binário whisper.cpp (build Vulkan).

Cada backend tenta 1x e levanta RuntimeError(motivo) se falhar — o
orquestrador (transcribe.py) faz o fallback explícito. Sem retry aqui.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from .models import Word, Segment

CHUNK_SECONDS = 30  # janela nativa do Whisper; 30s a 16kHz mono = ~1MB por chunk
SAMPLE_RATE = 16000


def probe_duration(video_path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def audio_chunks(video_path: str, chunk_sec: int = CHUNK_SECONDS):
    """Gera paths de wav temporários (16kHz mono), 1 por vez.

    Uso: for wav, offset in audio_chunks(...): ...; wav.unlink()
    Nunca carrega o vídeo/áudio inteiro na RAM.
    """
    duration = probe_duration(video_path) or 0
    # fallback: se duração desconhecida, extrai em blocos até o ffmpeg retornar vazio
    tmpdir = Path(tempfile.mkdtemp(prefix="clipper_audio_"))
    try:
        start = 0.0
        idx = 0
        while True:
            if duration and start >= duration:
                break
            wav = tmpdir / f"chunk_{idx:04d}.wav"
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(start), "-t", str(chunk_sec),
                 "-i", video_path, "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-c:a", "pcm_s16le", str(wav)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
                if wav.exists():
                    wav.unlink()
                break
            yield wav, start
            wav.unlink(missing_ok=True)
            start += chunk_sec
            idx += 1
            if not duration and idx > 2000:  # trava de segurança (~16h)
                break
    finally:
        # limpa resíduos (chunks já removidos 1 a 1; remove o dir)
        try:
            for leftover in tmpdir.glob("*.wav"):
                leftover.unlink()
            tmpdir.rmdir()
        except Exception:
            pass


def _even_words(text: str, start: float, end: float) -> list[Word]:
    """Fallback honesto: distribui palavras uniformemente no intervalo."""
    tokens = text.split()
    if not tokens or end <= start:
        return []
    dur = (end - start) / len(tokens)
    return [Word(t, start + i * dur, start + (i + 1) * dur) for i, t in enumerate(tokens)]


def transcribe_openvino(video_path: str, model_size: str = "medium",
                        language: str | None = None) -> list[Segment]:
    """Whisper via optimum-intel no device GPU (B580). Levanta RuntimeError se falhar."""
    try:
        import openvino as ov
    except Exception as e:
        raise RuntimeError(f"pacote openvino não instalado ({e})")
    try:
        devs = list(ov.Core().available_devices)
    except Exception as e:
        raise RuntimeError(f"OpenVINO não enumera devices ({e})")
    if "GPU" not in devs:
        raise RuntimeError(f"OpenVINO sem device GPU (visíveis: {devs}) — instale intel-opencl-icd/level-zero")
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
        from transformers import WhisperProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(f"optimum-intel/transformers ausentes ({e}) — pip install optimum-intel transformers")

    model_id = f"openai/whisper-{model_size}"
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device="GPU")
    except Exception as e:
        raise RuntimeError(f"falha ao carregar/exportar {model_id} no GPU ({e})")

    import numpy as np
    segments: list[Segment] = []
    try:
        import soundfile as sf
        for wav, offset in audio_chunks(video_path):
            audio, sr = sf.read(str(wav))
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            out = model.generate(**inputs, return_timestamps=True)
            text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            words = _even_words(text, offset, offset + CHUNK_SECONDS)
            segments.append(Segment(text=text, start=offset,
                                    end=offset + CHUNK_SECONDS, words=words))
    except Exception as e:
        raise RuntimeError(f"erro de inferência OpenVINO GPU ({e})")
    if not segments:
        raise RuntimeError("OpenVINO GPU não gerou segmentos (áudio vazio?)")
    return segments


def _whisper_cpp_bin() -> str | None:
    import shutil
    return shutil.which("whisper-cpp") or shutil.which("whisper-cli")


def transcribe_vulkan(video_path: str, model_size: str = "medium",
                      language: str | None = None) -> list[Segment]:
    """Whisper via binário whisper.cpp (build Vulkan usa a B580). Levanta RuntimeError se falhar."""
    binary = _whisper_cpp_bin()
    if not binary:
        raise RuntimeError("binário whisper-cpp/whisper-cli não está no PATH")
    # modelo ggml esperado em models/ggml-<size>.bin (convenção whisper.cpp)
    model = Path(f"models/ggml-{model_size}.bin")
    if not model.exists():
        alt = Path.home() / ".cache" / "whisper" / f"ggml-{model_size}.bin"
        if alt.exists():
            model = alt
        else:
            raise RuntimeError(f"modelo {model} não encontrado — baixe de HuggingFace (ex: ggerganov/whisper.cpp)")
    segments: list[Segment] = []
    try:
        for wav, offset in audio_chunks(video_path):
            out_json = wav.with_suffix(".json")
            cmd = [binary, "-m", str(model), "-f", str(wav), "-oj",
                   "-l", language or "auto", "-nt", "-nc"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0 or not out_json.exists():
                raise RuntimeError(f"whisper.cpp rc={r.returncode}: {(r.stderr or '')[:200]}")
            data = json.loads(out_json.read_text(encoding="utf-8"))
            for tr in data.get("transcription", []):
                text = (tr.get("text") or "").strip()
                s = offset + float(tr.get("offsets", {}).get("from", 0)) / 1000.0
                e = offset + float(tr.get("offsets", {}).get("to", 0)) / 1000.0
                words = [Word(w.get("text", "").strip(),
                              offset + float(w.get("offsets", {}).get("from", 0)) / 1000.0,
                              offset + float(w.get("offsets", {}).get("to", 0)) / 1000.0)
                         for w in tr.get("tokens", [])] or _even_words(text, s, e)
                segments.append(Segment(text=text, start=s, end=e, words=words))
            out_json.unlink(missing_ok=True)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"erro whisper.cpp Vulkan ({e})")
    if not segments:
        raise RuntimeError("whisper.cpp não gerou segmentos (áudio vazio?)")
    return segments
