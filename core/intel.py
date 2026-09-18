"""intel.py — detecção da Intel Arc/B580 e sondas de aceleração.

SRP: só detecta e sonda. Sem side-effect, sem retry, sem alterar o sistema.
Toda sonda tem timeout curto e é chamada no máximo 1x por processo (cache).
"""
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_QSV_PROBE_TIMEOUT = 15


def _drm_vendors() -> list[str]:
    out = []
    try:
        for p in Path("/sys/class/drm").glob("card*/device/vendor"):
            try:
                out.append(p.read_text().strip())
            except Exception:
                continue
    except Exception:
        pass
    return out


def has_intel_gpu() -> bool:
    """True se houver GPU Intel discreta/integrada (vendor 8086)."""
    if "0x8086" in _drm_vendors():
        return True
    try:
        out = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=2)
        for line in out.stdout.splitlines():
            if "8086" in line and ("VGA" in line or "Display" in line or "Graphics" in line):
                return True
        if "Arc" in out.stdout and "8086" in out.stdout:
            return True
    except Exception:
        pass
    return False


def gpu_name() -> str:
    """Nome amigável da GPU Intel, ou '' se não houver."""
    try:
        out = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=2)
        for line in out.stdout.splitlines():
            low = line.lower()
            if "8086" in line and ("vga" in low or "display" in low or "graphics" in low or "arc" in low):
                # "03:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]"
                desc = line.split(":", 2)[-1].strip()
                return desc[:80]
    except Exception:
        pass
    if has_intel_gpu():
        return "Intel GPU (nome não identificado)"
    return ""


def has_openvino() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def openvino_devices() -> list[str]:
    """Devices visíveis ao OpenVINO (ex: ['CPU'] ou ['CPU', 'GPU'])."""
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except Exception:
        return []


def has_openvino_gpu() -> bool:
    return "GPU" in openvino_devices()


def has_whisper_cpp() -> bool:
    if shutil.which("whisper-cpp") is not None or shutil.which("whisper-cli") is not None:
        return True
    # Build local do projeto (thirdparty/whisper.cpp, GGML_VULKAN=1).
    root = Path(__file__).resolve().parent.parent
    for name in ("whisper-cli", "whisper-cpp"):
        cand = root / "thirdparty" / "whisper.cpp" / "build" / "bin" / name
        if cand.is_file():
            return True
    return False


@lru_cache(maxsize=1)
def qsv_works() -> bool:
    """Sonda funcional: tenta 1 encode H264 QSV de 10 frames. Sem retry."""
    if not has_intel_gpu() or not shutil.which("ffmpeg"):
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=10:duration=1",
             "-c:v", "h264_qsv", "-f", "null", "-"],
            capture_output=True, text=True, timeout=_QSV_PROBE_TIMEOUT,
        )
        return r.returncode == 0
    except Exception:
        return False


def best_video_encoder() -> str:
    return "h264_qsv" if qsv_works() else "libx264"


def resolve_transcribe_backend(requested: str) -> tuple[str, str]:
    """Resolve CLIPPER_TRANSCRIBE_BACKEND.

    Retorna (backend_efetivo, motivo). Nunca levanta exceção.
    'auto': vulkan (whisper.cpp) -> openvino (GPU) -> cpu.
    Escolha explícita indisponível => cpu + motivo claro (nunca silencioso).
    """
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", "backend explícito 'cpu'"
    if requested == "gpu":
        # Exige GPU de verdade: tenta os backends reais, sem fallback
        # silencioso. Retorna "unavailable" (não é backend válido) quando
        # nada presta — o orquestrador levanta erro claro em vez de usar CPU.
        if has_whisper_cpp():
            return "vulkan", "modo 'gpu': whisper.cpp com Vulkan detectado"
        if has_openvino_gpu():
            return "openvino", "modo 'gpu': OpenVINO com device GPU"
        if has_intel_gpu() and has_openvino():
            detail = ("GPU Intel presente mas OpenVINO não expõe device GPU "
                      "(falta o pacote intel-level-zero?)")
        elif has_intel_gpu():
            detail = ("GPU Intel presente mas nenhum runtime de inferência "
                      "(whisper.cpp / OpenVINO+GPU) disponível")
        elif has_openvino():
            detail = "OpenVINO instalado mas nenhuma GPU Intel detectada"
        else:
            detail = ("nenhuma GPU Intel detectada e nenhum runtime "
                      "(whisper.cpp / OpenVINO) instalado")
        return "unavailable", f"modo 'gpu' exige GPU, indisponível: {detail}"
    if requested in ("vulkan", "whisper.cpp", "whispercpp"):
        if has_whisper_cpp():
            return "vulkan", "binário whisper.cpp presente"
        return "cpu", "whisper.cpp não encontrado no PATH (instale whisper.cpp com Vulkan)"
    if requested in ("openvino", "ov", "intel"):
        if has_openvino_gpu():
            return "openvino", "OpenVINO com device GPU"
        if has_openvino():
            return "cpu", "OpenVINO instalado mas sem device GPU visível (falta driver Level Zero/OpenCL?)"
        return "cpu", "OpenVINO não instalado (pip install openvino optimum-intel)"
    # auto
    if has_whisper_cpp():
        return "vulkan", "auto: whisper.cpp com Vulkan detectado"
    if has_openvino_gpu():
        return "openvino", "auto: OpenVINO com device GPU"
    if has_intel_gpu() and has_openvino():
        return "cpu", "auto: GPU Intel presente mas OpenVINO não expõe device GPU — verifique o pacote intel-level-zero"
    if has_intel_gpu():
        return "cpu", "auto: GPU Intel presente mas nenhum runtime de inferência (whisper.cpp/OpenVINO+GPU) disponível"
    return "cpu", "auto: nenhuma GPU Intel detectada"


def describe(requested: str = "auto") -> dict:
    """Snapshot para log do preflight (§8)."""
    backend, reason = resolve_transcribe_backend(requested)
    return {
        "gpu_present": has_intel_gpu(),
        "gpu_name": gpu_name(),
        "transcribe_backend": backend,
        "transcribe_reason": reason,
        "video_encoder": best_video_encoder(),
        "openvino_devices": openvino_devices(),
    }
