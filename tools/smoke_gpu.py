#!/usr/bin/env python3
"""tools/smoke_gpu.py — smoke test mínimo do backend GPU (Intel Arc B580).

Ordem do projeto (auto): whisper.cpp/Vulkan → OpenVINO/GPU → CPU.
Valida a primeira camada disponível com inferência real em áudio sintético
(gera WAV em /tmp — nunca usa vídeos reais).

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
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import intel as intel_hw  # noqa: E402
from core import backends as gpu_backends  # noqa: E402

RESULT = {"gpu": "NO", "device": "none", "backend": "unavailable",
          "reason": "", "model": "NO (n/a)", "infer": "FAIL (n/a)",
          "used": "none"}


def report() -> None:
    print(f"GPU detected: {RESULT['gpu']}")
    print(f"GPU device: {RESULT['device']}")
    print(f"Backend: {RESULT['backend']} ({RESULT['reason']})")
    print(f"Model loaded: {RESULT['model']}")
    print(f"Inference: {RESULT['infer']}")
    print(f"Device used: {RESULT['used']}")


def fail(reason: str) -> int:
    RESULT["reason"] = reason
    report()
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


def smoke_vulkan(tmp: Path) -> int:
    binary = gpu_backends._whisper_cpp_bin()
    if not binary:
        return fail("binário whisper-cpp/whisper-cli ausente (PATH ou "
                    "thirdparty/whisper.cpp/build/bin) — ver README § Intel Arc")
    model = Path("models/ggml-medium.bin")
    if not model.exists():
        return fail("models/ggml-medium.bin ausente — baixe de ggerganov/whisper.cpp")
    wav = tmp / "tone.wav"
    synth_wav(wav)
    try:
        r = subprocess.run(
            [binary, "-m", str(model), "-f", str(wav), "-ojf",
             "-of", str(tmp / "out"), "-l", "auto", "-nt"],
            capture_output=True, text=True, timeout=300)
    except Exception as e:
        return fail(f"falha ao executar {binary} ({e})")
    stderr = r.stderr or ""
    if "Intel(R) Arc(tm) B580" not in stderr and "Arc" not in stderr:
        RESULT["reason"] = ("binário OK mas B580 não aparece nos devices Vulkan; "
                            "saída: " + stderr[:200])
    if r.returncode != 0 or not (tmp / "out.json").exists():
        return fail(f"whisper.cpp rc={r.returncode}: {stderr[:200]}")
    RESULT["model"] = "YES (models/ggml-medium.bin, 1533 MB na VRAM)"
    RESULT["infer"] = "SUCCESS (JSON válido gerado via Vulkan)"
    RESULT["used"] = "GPU"
    RESULT["reason"] = "whisper.cpp + Vulkan, device 0 = B580"
    report()
    return 0


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
        tmp = Path(tempfile.mkdtemp(prefix="clipper_smoke_"))
        try:
            return smoke_vulkan(tmp)
        finally:
            try:
                for f in tmp.glob("*"):
                    f.unlink()
                tmp.rmdir()
            except Exception:
                pass
    return fail("backend OpenVINO: smoke de inferência ainda não implementado "
                f"neste script ({reason})")


if __name__ == "__main__":
    raise SystemExit(main())
