# Gerador de Cortes — lives e gameplay

Programa em Python que pega a gravação de uma live/gameplay, transcreve o
áudio localmente, identifica os melhores momentos com IA e já entrega os
clipes cortados em 9:16 com legenda queimada, prontos pra postar.

Sem custo: Whisper local (na sua GPU) + tier grátis da NVIDIA Build.

## O que faz

```text
live/gameplay (mp4, mkv — até ~17 GB)
        ↓
transcreve tudo localmente (Intel Arc B580)
        ↓
IA avalia cada trecho (nota 0–10, título, hashtags)
        ↓
cortes/01_*.mp4 … N_*.mp4 (9:16, legendados) + manifest.json
        ↓
metrics/<data>_<video>_<id>.json (relatório da execução)
```

## Como funciona

1. **Transcrição** (local, offline) — whisper.cpp + Vulkan na B580 (padrão);
   fallback `faster-whisper` CPU int8, sempre com motivo explícito.
2. **Janelas candidatas** — janela deslizante 20–90 s, passo 20 s, sobre a
   transcrição; `start/end` ajustados à palavra mais próxima (±1,5 s) + respiro
   `--pad`.
3. **Áudio** — energia por janela via `ffmpeg volumedetect`
   (`alta/media/baixa`) + `speech_rate` (palavras/s).
4. **Scoring** — lotes de ~6 candidatos para o LLM (NVIDIA Build) com prompt
   calibrado + duração/energia/speech_rate + contexto/few-shot; retry 3× com
   backoff 10/30/60 s; falha marca `failed` (nunca nota 0 silenciosa).
5. **Seleção** — NMS com decaimento (`score * (1 - 0.8*overlap)`), filtro
   `--min-score`, diversidade `--max-per-10min`.
6. **Corte** — `ffmpeg` com seek rápido + `trim`/`atrim` frame-accurate,
   crop 9:16 centralizado no rosto (fallback: centro), legenda ASS queimada
   (Liberation Sans Bold, base, área segura), encode `h264_qsv` (B580) com
   fallback `libx264`.
7. **Relatório** — resumo no terminal + JSON próprio em `metrics/`, em
   sucesso, falha ou Ctrl+C (`interrupted`).

## Uso rápido

```bash
./run.sh video.mkv                    # 8 clipes em cortes/, com cache
./run.sh video.mkv --top 5            # 5 clipes
./run.sh video.mkv --out meus_cortes  # outra pasta
./run.sh video.mkv --no-cache         # sem cache (só p/ teste rápido)
```

O `run.sh` liga o cache (`.cache/clipper`), carrega o `.env` sozinho e
repassa qualquer flag do `clipper.py`. Equivalente manual:

```bash
export NVIDIA_API_KEY="nvapi-sua-chave"
python clipper.py minha_live.mp4 --out cortes/ --top 8 --cache-dir .cache/clipper
```

Isso gera:

```text
cortes/
├── 01_jogada_insana_no_final.mp4
├── 02_treta_com_o_chat.mp4
├── ...
└── manifest.json   # nota, título, hashtags, energy, speech_rate, snapped
```

> Vídeo grande? Use **sempre** `--cache-dir` (o `run.sh` já faz isso): se algo
> cair no meio, re-rodar reusa transcrição + scores em vez de recomeçar do zero.

## Instalação

```bash
# 1. sistema (Fedora 44 testado): ffmpeg + drivers da B580
sudo dnf install -y ffmpeg intel-level-zero intel-media-driver

# 2. Python
pip install -r requirements.txt
# -> faster-whisper, openai, opencv-python, psutil

# 3. chave grátis do scoring (https://build.nvidia.com)
export NVIDIA_API_KEY="nvapi-sua-chave-aqui"
export NIM_MODEL="z-ai/glm-5.3"   # default atual; ajuste conforme o catálogo

# 4. backend Vulkan da transcrição (uma vez; detalhe na seção GPU)
#    git clone whisper.cpp + SPIRV-Headers, cmake -DGGML_VULKAN=1,
#    curl ggml-medium.bin -> models/  (passo a passo na seção GPU)
vulkaninfo --summary | grep -A1 GPU0   # Intel(R) Arc(tm) B580 (Mesa ANV)
python tools/smoke_gpu.py              # Inference: SUCCESS, Device used: GPU
```

Sem a etapa 4, a transcrição roda em CPU (mais lento, mas funciona).

## Transcrição e GPU (Intel Arc B580)

`faster-whisper`/`CTranslate2` é CUDA-only: nunca usa Intel GPU. Por isso o
padrão é **whisper.cpp + Vulkan na B580**, com CPU como fallback explícito.

| Valor (`--transcribe-backend` ou `CLIPPER_TRANSCRIBE_BACKEND`) | Efeito |
|---|---|
| `auto` (padrão) | whisper.cpp/Vulkan → OpenVINO GPU → CPU (fallback com motivo impresso) |
| `gpu` | **Exige GPU**: Vulkan ou OpenVINO/GPU; se indisponível, **falha claramente** (sem fallback) |
| `vulkan` | Só whisper.cpp; se indisponível, CPU com motivo |
| `openvino` | Só OpenVINO GPU (exige `intel-level-zero` + `pip install openvino optimum-intel transformers soundfile`); se indisponível, CPU com motivo |
| `cpu` | Sempre CPU (`faster-whisper` int8, 4 threads) |

No início o programa sempre imprime (exemplo real):

```text
GPU detected: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]
Transcription backend: VULKAN
Video acceleration: h264_qsv
```

O encode usa `h264_qsv` (Quick Sync) quando a sonda funcional passa; senão
`libx264` com 1 aviso por clipe. Filtros e detecção de rosto ficam na CPU
(leves perto de transcrição/encode).

### Benchmark medido (60 s de áudio, Whisper medium, esta máquina)

| Backend | Tempo | RTF | CPU | RAM pico |
|---|---|---|---|---|
| whisper.cpp + Vulkan / B580 | ~4,6 s | **0,08** | ~1 core | 187 MB |
| faster-whisper int8 / 4600G 4 threads | 134,8 s | 2,25 | 366% | 2,6 GB |

Medido em 2026-09-18 com `/usr/bin/time -v` sobre o mesmo WAV. Vídeo real de
7 min: 38 s de transcrição na B580.

### Como o backend Vulkan funciona

Chunks de 30 s via ffmpeg (1 por vez, removidos após uso) → 1 processo
`whisper-cli` por chunk (sequencial, timeout 300 s) → JSON com
word-timestamps reais → segmentos. VRAM: 1533 MB durante inferência, liberada
por chunk. Modelo: `models/ggml-medium.bin` (~1,5 GB, HuggingFace
`ggerganov/whisper.cpp`); binário em `thirdparty/whisper.cpp/build/bin`
(whisper.cpp 1.9.4-dev validado; ambos ignorados no git).

Limitações: áudio altamente repetitivo pode alucinar o Whisper (traço do
modelo); cada chunk recarrega o modelo (segundos); utilização/VRAM ficam
`null` no relatório (`nvidia-smi` não serve p/ Intel, `intel_gpu_top`
ausente, sysfs do driver `xe` sem contadores — `null` honesto).

## Scoring e modelos

Pontuar "isso é viral?" usa LLM remoto **só com texto** (trecho de até 1500
chars + tempos/energia/speech_rate). **Vídeo e áudio nunca saem da máquina.**

| Modelo | Quando usar |
|---|---|
| `z-ai/glm-5.3` (padrão, vivo em 2026-09-18) | Decisão original — reasoning nativo, melhor p/ hype/humor |
| `z-ai/glm-5.3-flash` | Variante mais rápida do mesmo modelo |
| `nvidia/nemotron-3-super-120b-a12b` | Alternativa testada |
| `mistralai/mistral-nemotron` | Mais leve, se os de cima instabilizarem |

Slugs expiram (ex.: `meta/llama-3.3-70b-instruct` morreu em 2026-08-26 com
HTTP 410). Se o scoring falhar com `Gone`, liste os vivos
(`GET https://integrate.api.nvidia.com/v1/models`) e ajuste
`NIM_MODEL`/`--model`. O código remove blocos de reasoning (`<think>`) antes
de ler o JSON.

> Atenção ao custo real: o GLM raciocina antes de responder — cada lote leva
> ~1 min ou mais. Num vídeo de 7 min o scoring levou 17 min (85% do run).
> É o gargalo atual, não a máquina. A variante `flash` existe para isso.

## Métricas e cache

Toda execução gera: resumo `EXECUTION METRICS` no terminal + JSON próprio em
`metrics/<data>_<video>_<id>.json` (nunca sobrescreve; falhas e Ctrl+C também
geram). Conteúdo: vídeo, execução (tempos por etapa, args, erro),
transcrição (backend, tempo, RTF, segmentos), GPU/CPU/RAM (média/pico, `null`
se indisponível), FFmpeg, API NVIDIA (requests/retries/latências/tokens;
custo sempre `null`). Monitoramento: 1 thread, 1 amostra/2 s, só agregados.

Cache (`--cache-dir`, ligado no `run.sh`): transcrição + scores por
fingerprint `tamanho+mtime+modelo+idioma` — trocar o arquivo invalida sozinho.
`--force-retranscribe` / `--force-rescore` refazem cada camada.

## CLI completa

| Flag | Efeito | Padrão |
|---|---|---|
| `VIDEO` | Arquivo de entrada (nunca alterado) | — (obrigatório) |
| `--out DIR` | Pasta de saída | `cortes` |
| `--top N` | Quantos clipes gerar | 8 |
| `--model NOME` | Modelo de scoring | env `NIM_MODEL` |
| `--transcribe-backend B` | `auto`, `gpu`, `vulkan`, `openvino`, `cpu` | env ou `auto` |
| `--cache-dir DIR` | Ativa cache | desligado |
| `--force-retranscribe` | Ignora cache de transcrição | off |
| `--force-rescore` | Ignora cache de scores | off |
| `--pad SEC` (0–5) | Respiro antes/depois do corte | 0.8 |
| `--no-vertical` | Mantém widescreen (1280px) | 9:16 |
| `--no-captions` | Sem legenda queimada | legenda on |
| `--no-audio-features` | Pula energia de áudio (mais rápido) | áudio on |
| `--min-score F` (0–10) | Score mínimo | 6.0 |
| `--max-per-10min N` (1–20) | Máximo por janela de 10 min | 2 |
| `--context TEXTO` | Contexto injetado no prompt | — |
| `--examples JSON` | Few-shot (ver `examples.json`, máx 4) | — |

```bash
# calibrado pro seu conteúdo
python clipper.py live.mp4 --context "ranked Valorant, duo com fulano, humor ácido" \
  --examples examples.json --min-score 6.5 --max-per-10min 2

# corte mais solto e rápido
python clipper.py live.mp4 --pad 1.2 --no-audio-features --min-score 5.0
```

## Ajustando pro seu conteúdo

- **Sem editar código**: `--context` + `--examples` (2 notas altas + 2 baixas,
  trechos literais da sua live). Formato em `examples.json` (aceita lista
  direta `[{...}]`).
- **IA errando "bom momento"**: ajuste `--context` primeiro; depois few-shot.
- **Clipes do mesmo trecho**: `--max-per-10min 1` ou suba `--min-score`.
- **Corte no meio da frase**: `--pad 1.2–1.5`.
- **Legenda rápida/pequena**: `CAPTION_MIN_DURATION` / `CAPTION_FONT_SIZE_*`
  em `core/video.py` (padrão 1.0 s / 54 px vertical, 44 px widescreen).
- `SEGMENTS_PER_SCORING_CALL` (em `core/config.py`): candidatos por chamada —
  suba p/ menos requests, desça se a IA "perde o fio".

## Diagnóstico

- **Caiu pra CPU?** A linha `Transcription backend: CPU (<motivo>)` + o campo
  `transcription` do JSON dizem exatamente por quê.
- **GPU indisponível?** `python tools/smoke_gpu.py` (exit 2 = motivo impresso);
  OpenVINO sem device = falta `intel-level-zero`.
- **Scoring falhou?** `410 Gone` = modelo expirado; `429` = rate limit (backoff
  cobre); `nvidia_api.failures` no JSON; lote falho = candidatos `failed`
  (excluídos da seleção, nunca nota 0).
- **FFmpeg falhou?** `! QSV falhou` → fallback `libx264` automático;
- **Legendas (sync/tokens/estilo)**: tempos absoluto→relativo com clamp em
  `[0, duração]` (testes em `tests/`: `python -m unittest discover -s tests`);
  corte usa `trim`+`setpts` (o `-ss` após `-i` deslocava legendas em `-fine`);
  tokens `[eot]/[sot]/...` filtrados na origem; ASS com `PlayRes` = frame real,
  Liberation Sans Bold 54 px na base. Timestamps do próprio Whisper têm jitter
  natural (~0,5 s) — fora do escopo do programa corrigir.
  `! Falha clipe N` → pula o clipe, o job continua.
- **Sem candidatos?** Vídeo sem fala (ou VAD removeu tudo) — `segments: 0`.
- **Disco?** Preflight falha com < 1 GB livre, avisa com < 5 GB.
- **Interrompeu (Ctrl+C)?** Relatório `interrupted` em `metrics/` com a etapa;
  temporários se limpam; `pgrep -x ffmpeg` deve voltar vazio.
- **Original seguro?** Nenhuma saída pode ter o mesmo caminho do vídeo de
  entrada (clipe ignorado com erro se coincidir).

## Limitações conhecidas

- Scoring é textual + energia: jogada visual silenciosa continua difícil sem
  visão computacional. É também o gargalo de tempo (reasoning ~1 min/lote).
- Haar Cascade simples (webcam fixa; fora de cena → centro).
- Áudio adiciona ~7 s/candidato (`volumedetect`); `--no-audio-features` pula.
- `faster-whisper` nunca usa Intel GPU (CUDA-only) — por isso o backend Vulkan.
- Cache por tamanho+mtime: edição que preserve ambos reutiliza cache (use
  `--force-*`).
- Sem suite automatizada; Linux (Fedora 44 testado) na prática.
