"""preflight.py — valida ambiente antes de gastar horas de transcrição.

SRP: só checagens fail-fast. DIP: recebe Path/str, não cria estado global.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .config import _AUTO_LIMITS, CLIPPER_TRANSCRIBE_BACKEND
from .system import format_report


def check(video_path: str, out_dir: Path, top_n: int,
          transcribe_backend: str | None = None) -> None:
    # Loga specs + limites auto-detectados (sem alterar nada no sistema)
    try:
        print(format_report(_AUTO_LIMITS))
    except Exception:
        pass
    # Relatório GPU/backend (§8) — sempre visível, nunca silencioso
    try:
        from . import intel as intel_hw
        d = intel_hw.describe(transcribe_backend or CLIPPER_TRANSCRIBE_BACKEND)
        if d["gpu_present"]:
            print(f"GPU detected: {d['gpu_name']}")
        else:
            print("GPU detected: none")
        print(f"Transcription backend: {d['transcribe_backend'].upper()}"
              + (f" ({d['transcribe_reason']})" if d["transcribe_backend"] in ("cpu", "unavailable") else ""))
        print(f"Video acceleration: {d['video_encoder']}")
    except Exception:
        pass
    errors: list[str] = []

    if not os.path.exists(video_path):
        errors.append(f"Arquivo não encontrado: {video_path}")
    elif shutil.which("ffprobe"):
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=10,
            )
            if not probe.stdout.strip():
                print(f"   ! Aviso: nenhuma trilha de áudio em {video_path} — transcrição pode falhar.")
        except Exception:
            pass

    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg não encontrado no PATH. Instale: sudo apt install ffmpeg / brew install ffmpeg")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".clipper_write_test"
        probe.touch()
        probe.unlink()
    except Exception as e:
        errors.append(f"Pasta de saída não gravável ({out_dir}): {e}")

    # Auditoria (disco): falhe ANTES da operação pesada se não houver espaço
    # mínimo; avise se estiver baixo. Saídas são pequenas (clipes + manifest),
    # mas nunca avance rumo a 0 bytes livres em silêncio.
    try:
        free_gb = shutil.disk_usage(out_dir).free / 1024**3
        if free_gb < 1:
            errors.append(f"Espaço insuficiente em {out_dir} ({free_gb:.1f} GB livres, mínimo 1 GB)")
        elif free_gb < 5:
            print(f"   ! Aviso: pouco espaço livre ({free_gb:.1f} GB em {out_dir})")
    except Exception:
        pass

    if top_n <= 0:
        errors.append(f"--top deve ser > 0 (recebido {top_n})")

    if not os.environ.get("NVIDIA_API_KEY"):
        errors.append("NVIDIA_API_KEY não definida — pegue uma grátis em https://build.nvidia.com")

    if errors:
        for err in errors:
            print(f"Erro de preflight: {err}", file=sys.stderr)
        sys.exit(1)

    if not shutil.which("ffprobe"):
        print("   ! Aviso: ffprobe não encontrado — crop/pad e checagem de áudio limitados.")
