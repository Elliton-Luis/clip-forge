#!/usr/bin/env bash
# run.sh — Atalho para rodar o clipper com padrões sensatos.
#
# Uso (a partir da raiz do repo):
#   ./scripts/run.sh                              # abre a interface interativa (TUI)
#   ./scripts/run.sh video.mkv                    # 8 clipes em cortes/, com cache
#   ./scripts/run.sh video.mkv --top 5            # 5 clipes
#   ./scripts/run.sh video.mkv --out meus_cortes  # outra pasta de saída
#   ./scripts/run.sh video.mkv --no-cache         # sem cache (não recomendado p/ vídeos grandes)
#
# Qualquer flag extra do clipper.py é repassada (ex: --model, --min-score).
set -euo pipefail

# scripts/ → raiz do repo (clipper.py, .env).
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    # Sem argumentos: interface interativa.
    if [ -z "${NVIDIA_API_KEY:-}" ] && [ -f .env ]; then
        set -a; source .env; set +a
    fi
    exec python3 clipper.py
fi

VIDEO="$1"; shift

if [ ! -f "$VIDEO" ]; then
    echo "Erro: arquivo não encontrado: $VIDEO" >&2
    exit 1
fi

if [ -z "${NVIDIA_API_KEY:-}" ] && [ -f .env ]; then
    # shellcheck disable=SC1091
    set -a; source .env; set +a
fi

EXTRA=("$@")
USE_CACHE=1
FILTERED=()
for a in "${EXTRA[@]}"; do
    if [ "$a" = "--no-cache" ]; then
        USE_CACHE=0
    else
        FILTERED+=("$a")
    fi
done
EXTRA=("${FILTERED[@]}")

if [ "$USE_CACHE" = 1 ]; then
    CACHE_ARGS=(--cache-dir .cache/clipper)
else
    CACHE_ARGS=()
fi

python3 clipper.py "$VIDEO" "${CACHE_ARGS[@]}" "${EXTRA[@]}"

echo ""
echo "Assista aos clipes na pasta de saída (ver manifest.json p/ notas e títulos)."
