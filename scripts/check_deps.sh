#!/usr/bin/env bash
# check_deps.sh — verifica o que já está instalado para o Clipper (Fedora)
# Não instala nada, não altera nada. Só relata.
#
# Uso (a partir da raiz do repo):
#   ./scripts/check_deps.sh          # resumo + exit 0 (essencial ok) / 1 (falta essencial)
#   ./scripts/check_deps.sh --verbose
#
# Níveis: ESSENCIAL (falhar aqui = exit 1), GPU (só importa na B580),
# OPCIONAL (nunca falha o check). Detecta venv (.venv ou $VIRTUAL_ENV).

set -uo pipefail

VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE=1 ;;
        -h|--help)
            echo "Uso: $0 [--verbose]"
            echo "Exit 0 = essencial completo; exit 1 = falta item essencial."
            exit 0 ;;
        *) echo "Argumento desconhecido: $arg" >&2; exit 2 ;;
    esac
done

# --- python resolvido com venv primeiro ---
PYBIN="python3"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYBIN="$VIRTUAL_ENV/bin/python"
    PYENV="venv ativa ($VIRTUAL_ENV)"
elif [ -x ".venv/bin/python" ]; then
    PYBIN=".venv/bin/python"
    PYENV="venv local (.venv/)"
else
    PYENV="sistema ($(command -v python3 || echo "python3 ausente"))"
fi

OK="[OK]"
FALTA="[FALTA]"
INFO="[INFO]"
ess_ok=0
ess_falta=0

req() {  # check essencial (conta no exit code)
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "$OK  $desc"; ess_ok=$((ess_ok+1)); return 0
    else
        echo "$FALTA  $desc"; ess_falta=$((ess_falta+1)); return 1
    fi
}

opt() {  # check informativo (nunca falha)
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "$OK  $desc"
    else
        echo "$INFO  $desc (opcional — só faz falta no modo correspondente)"
    fi
}

say()  { echo "=================================================="; echo " $*"; echo "=================================================="; }
have() { command -v "$1" >/dev/null 2>&1; }

say "0. Python em uso"
echo "$INFO  $PYBIN ($PYENV)"
"$PYBIN" --version 2>&1 | sed 's/^/       /'

say "1. Essencial (sem isso o Clipper não roda)"
req "ffmpeg"            command -v ffmpeg
req "ffprobe"           command -v ffprobe
req "git"               command -v git
req "curl"              command -v curl
req "python3"           command -v python3
req "pip ($PYBIN -m pip)" "$PYBIN" -m pip --version
req "python: faster_whisper" "$PYBIN" -c "import faster_whisper"
req "python: openai"         "$PYBIN" -c "import openai"
req "python: cv2"            "$PYBIN" -c "import cv2"
req "python: psutil"         "$PYBIN" -c "import psutil"
req "whisper-cli (Vulkan ou PATH)" \
    bash -c 'test -x thirdparty/whisper.cpp/build/bin/whisper-cli || command -v whisper-cli >/dev/null || command -v whisper-cpp >/dev/null'
req "modelo ggml-medium.bin" test -s models/ggml-medium.bin
if [ -n "${NVIDIA_API_KEY:-}" ] || [ -n "${NVIDIA_API_KEYS:-}" ]; then
    echo "$OK  chave NVIDIA no ambiente desta shell"; ess_ok=$((ess_ok+1))
elif [ -f ".env" ] && grep -qE "^NVIDIA_API_KEY" ".env" 2>/dev/null; then
    echo "$OK  NVIDIA_API_KEY encontrada em .env"; ess_ok=$((ess_ok+1))
else
    echo "$FALTA  nenhuma NVIDIA_API_KEY (env nem .env) — scoring não funciona sem ela"
    ess_falta=$((ess_falta+1))
fi

say "2. GPU Intel Arc (só importa com --transcribe-backend auto/gpu/vulkan)"
if have vulkaninfo && vulkaninfo --summary 2>/dev/null | grep -qi "arc"; then
    echo "$OK  GPU Intel Arc visível no Vulkan:"
    vulkaninfo --summary 2>/dev/null | grep -A1 "GPU0" | sed 's/^/       /'
else
    echo "$INFO  sem Arc visível (CPU funciona; GPU exige drivers + build Vulkan)"
fi
for pkg in intel-level-zero intel-media-driver vulkan-loader mesa-vulkan-drivers; do
    opt "pacote $pkg" rpm -q "$pkg"
done

say "3. Opcional: forced alignment (--align wav2vec2)"
opt "python: torch"        "$PYBIN" -c "import torch"
opt "python: transformers" "$PYBIN" -c "import transformers"
opt "python: soundfile"    "$PYBIN" -c "import soundfile"

say "4. Opcional: backend OpenVINO / retry large-v3"
opt "python: openvino" "$PYBIN" -c "import openvino"
opt "modelo ggml-large-v3.bin" test -s models/ggml-large-v3.bin

say "5. Espaço em disco (modelos+build pedem ~6 GB livres)"
if [ "$VERBOSE" -eq 1 ]; then df -h . | sed 's/^/       /'; fi
free_kb="$(df -k . 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "$free_kb" ] && [ "$free_kb" -lt 6291456 ]; then
    echo "$INFO  pouco espaço livre ($((free_kb/1024)) MB) — download de modelos pode falhar"
else
    echo "$OK  espaço livre suficiente (${free_kb:-?} KB)"
fi
if [ "$VERBOSE" -eq 1 ]; then
    echo
    echo "$INFO  versões:"
    "$PYBIN" -c "import faster_whisper, openai, psutil; print('      faster-whisper', faster_whisper.__version__, '| openai', openai.__version__, '| psutil', psutil.__version__)" 2>/dev/null || true
    ffmpeg -version 2>/dev/null | head -1 | sed 's/^/       /'
fi

echo
echo "=================================================="
echo " RESUMO: essencial $ess_ok ok / $ess_falta faltando"
echo "=================================================="
if [ "$ess_falta" -gt 0 ]; then
    echo "Rode ./install_deps.sh (só instala o que falta) e repita este check."
    exit 1
fi
echo "Tudo essencial pronto. (Opcionais acima só importam nos modos indicados.)"
