#!/usr/bin/env python3
"""tools/smoke_gpu.py — smoke test mínimo do backend GPU (Intel Arc B580).

Valida SOMENTE: detecção da GPU → runtime → backend → modelo Whisper →
inferência pequena. Não usa nenhum vídeo real (gera WAV sintético em /tmp).

Saída objetiva (sempre as 6 linhas):
    GPU detected: YES/NO
    GPU device: <nome|none>
    Backend: <vulkan|openvino|unavailable> (<motivo>)
    Model loaded: YES/NO (<detalhe>)
    Inference: SUCCESS/FAIL (<detalhe>)
    Device used: GPU|CPU|none

Exit 0 = inferência OK na GPU. Exit 2 = GPU indisponível (motivo impresso).
"""
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import intel as intel_hw  # noqa: E402

RESULT = {"gpu": "NO", "device": "none", "backend": "unavailable",
          "reason": "", "model": "NO (n/a)", "infer": "FAIL (n/a)",
          "used": "none"}


def fail(reason: str) -> int:
    RESULT["reason"] = reason
    print(f"GPU detected: {RESULT['gpu']}")
    print(f"GPU device: {RESULT['device']}")
    print(f"Backend: {RESULT['backend']} ({reason})")
    print(f"Model loaded: {RESULT['model']}")
    print(f"Inference: {RESULT['infer']}")
    print(f"Device used: {RESULT['used']}")
    return 2


def synth_wav(path: Path, seconds: float = 2.0) -> None:
    sr = 16000
    n = int(sr * seconds)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            v = int(12000 * math.sin(2 * math.pi * 440 * i / sr))
            w.writeframes(struct.pack("<h", v))


def main() -> int:
    if not intel_hw.has_intel_gpu():
        return fail("nenhuma GPU Intel detectada (vendor 8086 ausente)")
    RESULT["gpu"] = "YES"
    RESULT["device"] = intel_hw.gpu_name() or "Intel GPU (nome não identificado)"

    backend, reason = intel_hw.resolve_transcribe_backend("gpu")
    RESULT["backend"] = backend
    if backend == "unavailable":
        return fail(reason)
    if backend == "vulkan":
        return fail("whisper.cpp detectado mas smoke de inferência Vulkan não "
                    "implementado (verificar manualmente o binário + modelo ggml)")

    # backend == openvino: prova de carga + inferência real no device GPU
    try:
        import openvino as ov  # noqa: F401
    except Exception as e:
        return fail(f"pacote openvino ausente ({e})")
    if "GPU" not in intel_hw.openvino_devices():
        return fail("OpenVINO não expõe device GPU "
                    "(falta intel-opencl-icd / level-zero?)")
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
        from transformers import WhisperProcessor  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:
        return fail(f"optimum-intel/transformers/soundfile ausentes ({e})")

    try:
        model_id = "openai/whisper-tiny"
        processor = WhisperProcessor.from_pretrained(model_id)
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, device="GPU")
        RESULT["model"] = "YES (openai/whisper-tiny, device GPU)"
    except Exception as e:
        RESULT["model"] = f"NO ({e})"
        return fail(f"falha ao carregar modelo no device GPU: {e}")

    wav = Path(tempfile.mkdtemp(prefix="clipper_smoke_")) / "tone.wav"
    try:
        synth_wav(wav)
        audio, sr = sf.read(str(wav))
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        out = model.generate(**inputs)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        RESULT["infer"] = f"SUCCESS ({len(str(text))} chars decodificados)"
        RESULT["used"] = "GPU"
    except Exception as e:
        RESULT["infer"] = f"FAIL ({e})"
        return fail(f"inferência no device GPU falhou: {e}")
    finally:
        try:
            wav.unlink()
            wav.parent.rmdir()
        except Exception:
            pass

    print(f"GPU detected: {RESULT['gpu']}")
    print(f"GPU device: {RESULT['device']}")
    print(f"Backend: {RESULT['backend']} (modo 'gpu': {reason})")
    print(f"Model loaded: {RESULT['model']}")
    print(f"Inference: {RESULT['infer']}")
    print(f"Device used: {RESULT['used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
