#!/usr/bin/env bash
# install_deps.sh — instala as dependências do Clipper no Fedora, sem dor.
#
# Para quem parte do zero e para quem já tem metade: cada etapa verifica
# antes se precisa fazer algo e pula o que já está pronto (idempotente).
#
# Uso (a partir da raiz do repo):
#   ./scripts/install_deps.sh                    # tudo que faltar (padrão sensato)
#   ./scripts/install_deps.sh --help             # esta ajuda
#   ./scripts/install_deps.sh --dry-run          # só mostra o que faria, sem executar
#   ./scripts/install_deps.sh --only python,models
#   ./scripts/install_deps.sh --with-large        # + ggml-large-v3 (~3 GB, retry anti-alucinação)
#   ./scripts/install_deps.sh --with-align        # + torch/transformers (forced alignment)
#   ./scripts/install_deps.sh --venv .venv        # cria/usa venv em vez de --user
#   ./scripts/install_deps.sh --rebuild           # recompila o whisper.cpp mesmo se funcionar
#   ./scripts/install_deps.sh --skip-system       # pula sudo dnf
#   ./scripts/install_deps.sh --skip-whisper-cpp  # pula clone/build do whisper.cpp
#
# Etapas: system | python | whisper-cpp | models

set -euo pipefail

WITH_LARGE=0
WITH_ALIGN=0
SKIP_SYSTEM=0
SKIP_WHISPER_CPP=0
REBUILD=0
DRY_RUN=0
ONLY=""
VENV=""

usage() { sed -n '2,20p' "$0"; }

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run) DRY_RUN=1 ;;
        --with-large) WITH_LARGE=1 ;;
        --with-align) WITH_ALIGN=1 ;;
        --skip-system) SKIP_SYSTEM=1 ;;
        --skip-whisper-cpp) SKIP_WHISPER_CPP=1 ;;
        --rebuild) REBUILD=1 ;;
        --only=*) ONLY="${arg#--only=}" ;;
        --venv=*) VENV="${arg#--venv=}" ;;
        *) echo "Argumento desconhecido: $arg (veja --help)" >&2; exit 1 ;;
    esac
done

log()  { echo -e "\n=== $* ===\n"; }
run()  { if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }
want() { # want <etapa> — true se a etapa está no --only (ou sem --only)
    [ -z "$ONLY" ] && return 0
    case ",$ONLY," in *,"$1,"*) return 0 ;; *) return 1 ;; esac
}
need_cmd() { ! command -v "$1" >/dev/null 2>&1; }

# --- python resolvido: venv ativa > --venv (cria se preciso) > sistema ---
PYBIN="python3"
PIPFLAGS=(--user)
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYBIN="$VIRTUAL_ENV/bin/python"; PIPFLAGS=()
elif [ -n "$VENV" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        log "venv inexistente em $VENV — criando"
        run python3 -m venv "$VENV"
    fi
    PYBIN="$VENV/bin/python"; PIPFLAGS=()
    # shellcheck disable=SC1091
    if [ "$DRY_RUN" -eq 0 ]; then . "$VENV/bin/activate"; fi
fi

# ------------------------------------------------------------------
# 1. Pacotes de sistema (única etapa que pode pedir sudo)
# ------------------------------------------------------------------
if want system && [ "$SKIP_SYSTEM" -eq 0 ]; then
    log "1/4 — Pacotes de sistema"
    MISSING_PKGS=()
    for pkg in ffmpeg intel-level-zero intel-media-driver git cmake gcc-c++ \
               make curl python3 python3-pip vulkan-loader vulkan-loader-devel \
               vulkan-headers vulkan-tools glslang; do
        rpm -q "$pkg" >/dev/null 2>&1 || MISSING_PKGS+=("$pkg")
    done
    if [ "${#MISSING_PKGS[@]}" -eq 0 ]; then
        echo "Todos os pacotes de sistema já instalados, pulando (sem sudo, sem nada)."
    elif [ "$(id -u)" -eq 0 ]; then
        run dnf install -y "${MISSING_PKGS[@]}"
    elif command -v sudo >/dev/null 2>&1; then
        echo "Faltando: ${MISSING_PKGS[*]} — pedindo sudo 1x para o dnf."
        run sudo dnf install -y "${MISSING_PKGS[@]}"
    else
        echo "AVISO: sem root e sem sudo — pulando sistema."
        echo "  Instale manualmente: dnf install -y ${MISSING_PKGS[*]}"
        echo "  (ou rode com --skip-system para silenciar este aviso)"
    fi
else
    log "1/4 — Sistema pulado"
fi

# ------------------------------------------------------------------
# 2. Dependências Python (só o que falta)
# ------------------------------------------------------------------
if want python; then
    log "2/4 — Dependências Python ($PYBIN)"
    if [ -f requirements.txt ]; then
        run "$PYBIN" -m pip install "${PIPFLAGS[@]}" -r requirements.txt
    else
        for mod in faster-whisper openai opencv-python psutil; do
            if "$PYBIN" -c "import ${mod//-/_}" >/dev/null 2>&1; then
                echo "$mod já instalado, pulando."
            else
                run "$PYBIN" -m pip install "${PIPFLAGS[@]}" "$mod"
            fi
        done
    fi
    if [ "$WITH_ALIGN" -eq 1 ]; then
        log "2b/4 — Forced alignment (--align wav2vec2)"
        for mod in torch transformers soundfile; do
            if "$PYBIN" -c "import $mod" >/dev/null 2>&1; then
                echo "$mod já instalado, pulando."
            else
                run "$PYBIN" -m pip install "${PIPFLAGS[@]}" "$mod"
            fi
        done
    fi
else
    log "2/4 — Python pulado"
fi

# ------------------------------------------------------------------
# 3. whisper.cpp com backend Vulkan (pula se o binário já funciona)
# ------------------------------------------------------------------
whisper_ok() {
    local bin="thirdparty/whisper.cpp/build/bin/whisper-cli"
    [ "$REBUILD" -eq 0 ] && [ -x "$bin" ] && "$bin" --help >/dev/null 2>&1
}
if want whisper-cpp && [ "$SKIP_WHISPER_CPP" -eq 0 ]; then
    log "3/4 — whisper.cpp (backend Vulkan)"
    if whisper_ok; then
        echo "whisper-cli já funciona, pulando clone/build (use --rebuild para forçar)."
    else
        for dep in git cmake g++ curl; do
            if need_cmd "$dep"; then
                echo "ERRO: '$dep' ausente — rode a etapa system primeiro." >&2; exit 1
            fi
        done
        run mkdir -p thirdparty
        if [ ! -d thirdparty/SPIRV-Headers ]; then
            run git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git thirdparty/SPIRV-Headers
        else
            echo "SPIRV-Headers já clonado, pulando."
        fi
        if [ ! -d thirdparty/whisper.cpp ]; then
            run git clone https://github.com/ggerganov/whisper.cpp.git thirdparty/whisper.cpp
        else
            echo "whisper.cpp já clonado, atualizando."
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "[dry-run] (cd thirdparty/whisper.cpp && git pull --ff-only)"
            else
                (cd thirdparty/whisper.cpp && git pull --ff-only) \
                    || echo "aviso: não foi possível atualizar, seguindo com a versão local"
            fi
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "[dry-run] cmake -B thirdparty/whisper.cpp/build -DGGML_VULKAN=1 ..."
            echo "[dry-run] cmake --build thirdparty/whisper.cpp/build -j\$(nproc)"
        else
            (cd thirdparty/whisper.cpp && \
                cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release && \
                cmake --build build -j"$(nproc)" --config Release)
        fi
        if [ "$DRY_RUN" -eq 0 ] && ! whisper_ok; then
            echo "ERRO: build terminou mas o binário não executa." >&2; exit 1
        fi
        echo "Verificando GPU via Vulkan:"
        vulkaninfo --summary 2>/dev/null | grep -A1 "GPU0" \
            || echo "aviso: vulkaninfo não encontrou GPU0 — confira os drivers"
    fi
else
    log "3/4 — whisper.cpp pulado"
fi

# ------------------------------------------------------------------
# 4. Modelos ggml (download com resume + verificação de tamanho)
# ------------------------------------------------------------------
fetch_model() {  # fetch_model <arquivo> <url> <min_bytes>
    local file="$1" url="$2" min_bytes="$3"
    if [ -f "$file" ] && [ "$(stat -c%s "$file")" -ge "$min_bytes" ]; then
        echo "$file já existe e tem tamanho válido, pulando."
        return 0
    fi
    [ -f "$file" ] && echo "AVISO: $file incompleto, retomando download..."
    log "Baixando $(basename "$file")..."
    run mkdir -p "$(dirname "$file")"
    run curl -fSL --retry 3 -C - -o "$file.part" "$url"
    if [ "$DRY_RUN" -eq 0 ]; then
        mv -f "$file.part" "$file"
        if [ "$(stat -c%s "$file")" -lt "$min_bytes" ]; then
            echo "ERRO: $file menor que o esperado após download." >&2; exit 1
        fi
    fi
}
if want models; then
    log "4/4 — Modelos Whisper (ggml)"
    HF="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    fetch_model "models/ggml-medium.bin" "$HF/ggml-medium.bin" 1000000000
    if [ "$WITH_LARGE" -eq 1 ]; then
        fetch_model "models/ggml-large-v3.bin" "$HF/ggml-large-v3.bin" 2000000000
    fi
else
    log "4/4 — Modelos pulados"
fi

log "Concluído"
if [ "$DRY_RUN" -eq 0 ] && [ -x ./check_deps.sh ]; then
    echo "Verificação final:"
    ./check_deps.sh || true
fi
echo
echo "Se faltar a chave de scoring: export NVIDIA_API_KEY=\"nvapi-...\" (https://build.nvidia.com)"
