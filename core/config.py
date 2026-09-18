"""config.py — única fonte de verdade para constantes e env.

SOLID: SRP (só configuração). Sem lógica de negócio.
KISS: variáveis planas, sem classes desnecessárias.
"""
import os
from .system import safe_limits as _safe_limits

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Decisão original do projeto: GLM (README: "Melhor opção", reasoning nativo).
# Slug verificado vivo em 2026-09-18 via GET /v1/models + chamada real de
# scoring (atenção: o catálogo usa "z-ai/glm-5.3", não "zai/glm-5-3").
# Slugs expiram — confira antes de fixar outro valor.
DEFAULT_MODEL = os.environ.get("NIM_MODEL", "z-ai/glm-5.3")
WHISPER_LANGUAGE = os.environ.get("CLIPPER_LANGUAGE", None)  # None = auto

# --- Limites seguros auto-detectados (sem mexer no sistema) ---
# Detecta CPU/RAM e calcula limites que não travam PC fraco (ex: 4600G 6C/12T).
# Usuário pode sobrescrever via env CLIPPER_* (YAGNI: sem arquivo de config).
_SAFE = _safe_limits()

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

# WHISPER_MODEL_SIZE: auto downgrade para "small" se RAM <=6GB, senão "medium"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", _SAFE["whisper_model"])
CLIPPER_CPU_THREADS = _int_env("CLIPPER_CPU_THREADS", _SAFE["cpu_threads"])
CLIPPER_FFMPEG_THREADS = _int_env("CLIPPER_FFMPEG_THREADS", _SAFE["ffmpeg_threads"])
CLIPPER_COMPUTE_TYPE = os.environ.get("CLIPPER_COMPUTE_TYPE", _SAFE["compute"])
CLIPPER_DEVICE = os.environ.get("CLIPPER_DEVICE", _SAFE["device"])
CLIPPER_RAM_LIMIT_GB = _int_env("CLIPPER_RAM_LIMIT_GB", _SAFE["ram_limit_gb"])
# Backend de transcrição: auto|vulkan|openvino|cpu (auto = GPU Intel primeiro, CPU fallback explícito)
CLIPPER_TRANSCRIBE_BACKEND = os.environ.get("CLIPPER_TRANSCRIBE_BACKEND", "auto").lower()
# Guarda report para logar no preflight sem re-detectar
_AUTO_LIMITS = _SAFE

# janelas
MIN_CLIP_SECONDS = 20
MAX_CLIP_SECONDS = 90
WINDOW_STEP_SECONDS = 20
SEGMENTS_PER_SCORING_CALL = 6
OVERLAP_REJECT_RATIO = 0.25

# P1-1
SNAP_TOLERANCE_SECONDS = 1.5
DEFAULT_PAD_SECONDS = 0.8

# P1-3
CAPTION_MAX_CHARS_PER_LINE = 32
CAPTION_MAX_LINES_PER_CUE = 2
CAPTION_MIN_DURATION = 1.0
CAPTION_PAUSE_THRESHOLD = 0.4

# P1-4
DEFAULT_MIN_SCORE = 6.0
DEFAULT_MAX_PER_10MIN = 2
NMS_DECAY_STRENGTH = 0.8

# P1-2
AUDIO_ENERGY_TIMEOUT = 8
