"""config.py — única fonte de verdade para constantes e env.

SOLID: SRP (só configuração). Sem lógica de negócio.
KISS: variáveis planas, sem classes desnecessárias.
"""
import os
from .system import safe_limits as _safe_limits

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Default: Nemotron 3 Super — escolha do usuário após teste A/B real
# (mesmo momento selecionado que o GLM, bem mais rápido no scoring).
# Slugs expiram — confira antes de fixar outro valor.
DEFAULT_MODEL = os.environ.get("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
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
# Retry de chunks com alucinação (loop repetitivo): modelo maior usado UMA vez
# por chunk sinalizado; "off"/vazio desliga. Medido 2026-09: large-v3 recupera
# frase perdida pelo medium em gameplay (texto certo; sem trocar o default).
WHISPER_RETRY_MODEL = os.environ.get("WHISPER_RETRY_MODEL", "large-v3")
CLIPPER_CPU_THREADS = _int_env("CLIPPER_CPU_THREADS", _SAFE["cpu_threads"])
CLIPPER_FFMPEG_THREADS = _int_env("CLIPPER_FFMPEG_THREADS", _SAFE["ffmpeg_threads"])
CLIPPER_COMPUTE_TYPE = os.environ.get("CLIPPER_COMPUTE_TYPE", _SAFE["compute"])
CLIPPER_DEVICE = os.environ.get("CLIPPER_DEVICE", _SAFE["device"])
CLIPPER_RAM_LIMIT_GB = _int_env("CLIPPER_RAM_LIMIT_GB", _SAFE["ram_limit_gb"])
# Backend de transcrição: auto|vulkan|openvino|cpu (auto = GPU Intel primeiro, CPU fallback explícito)
CLIPPER_TRANSCRIBE_BACKEND = os.environ.get("CLIPPER_TRANSCRIBE_BACKEND", "auto").lower()
# Seleção: classic (janela+score médio, padrão calibrado) ou peak (experimental:
# detecta o auge e constrói o clip ao redor dele — ver docs/20260920_2213_peak-experiment.md).
CLIPPER_SELECTION_MODE = os.environ.get("CLIPPER_SELECTION_MODE", "classic").lower()
# Forced alignment opt-in: off|whisper-refine|wav2vec2 (off = pipeline inalterado).
# whisper-refine reusa o próprio Whisper em janela curta (B580, sem deps novas);
# wav2vec2 exige torch+transformers (fallback controlado sem eles).
CLIPPER_ALIGN = os.environ.get("CLIPPER_ALIGN", "off").lower()
# Filtro ffmpeg aplicado SOMENTE ao áudio de transcrição (nunca ao vídeo
# final). Ex: "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11". Vazio = off.
# Medido em áudio estourado: sem ganho demonstrável (clipping não se recupera)
# — por isso o padrão é off. Ver README § áudio.
CLIPPER_TRANSCRIBE_AUDIO_FILTER = os.environ.get("CLIPPER_TRANSCRIBE_AUDIO_FILTER", "")
# Guarda report para logar no preflight sem re-detectar
_AUTO_LIMITS = _SAFE

# Versão do pipeline de transcrição (entra no fingerprint do cache).
# Aumente quando mudar qualquer semântica da transcrição (VAD, modelo,
# retry, parse). v1 = era VAD (transcripts antigos NÃO são reutilizados:
# outliers temporais da era VAD jamais voltam por cache).
TRANSCRIPT_PIPELINE_VERSION = 2

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
# Largura de linha medida com PIL (Montserrat ExtraBold 54px): ~35-50px/char.
# 24 chars ≈ 850px típicos (limite útil ~1000px em 1080); WrapStyle 0 no ASS
# é a rede de segurança p/ casos extremos. 2 linhas = ~4-5 palavras/linha.
CAPTION_MAX_CHARS_PER_LINE = 24
CAPTION_MAX_LINES_PER_CUE = 2
CAPTION_MIN_DURATION = 1.0
CAPTION_PAUSE_THRESHOLD = 0.4

# P1-4
DEFAULT_MIN_SCORE = 6.0
DEFAULT_MAX_PER_10MIN = 2
NMS_DECAY_STRENGTH = 0.8

# P1-2
AUDIO_ENERGY_TIMEOUT = 8
# Versão da medição de energia (entra no cache de scores). Aumente quando
# mudar como energy é medido (ex: 1 = volumedetect só-áudio; era 0 implícita,
# com decode de vídeo e timeouts em massa). Scores antigos voltam a pontuar
# em vez de servir "media" obsoleto.
AUDIO_ENERGY_VERSION = 1
