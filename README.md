# Gerador de Cortes — lives e gameplay

Programa em Python que pega a gravação de uma live/gameplay, transcreve,
identifica os melhores momentos e já entrega os clipes cortados em 9:16
com legenda queimada, prontos pra postar.

Pensado pra rodar no seu ritmo: sobe o vídeo de manhã, os clipes ficam
prontos ao longo do dia, você posta à noite. Sem custo (usa Whisper
local + tier grátis da NVIDIA Build).

## 1. Instalar dependências

```bash
# ffmpeg (se ainda não tiver — precisa de ffmpeg E ffprobe)
sudo apt install ffmpeg        # Linux
brew install ffmpeg            # macOS

# dependências Python
pip install -r requirements.txt
```

> Se tiver GPU NVIDIA, o faster-whisper usa automaticamente (muito mais
> rápido). Sem GPU, roda em CPU — mais lento, mas funciona.
> Se tiver Intel Arc (ex: B580), veja a seção **Intel Arc (B580)** abaixo:
> a transcrição pode rodar na GPU via OpenVINO ou whisper.cpp/Vulkan,
> e o encode usa Quick Sync (QSV) quando o driver está funcional.

## 1b. Intel Arc B580 (transcrição + vídeo na GPU)

`faster-whisper`/`CTranslate2` é CUDA-only: nunca usa Intel GPU. O backend
padrão de transcrição é **whisper.cpp + Vulkan na B580** (validado aqui:
7 min transcritos em 38 s, RTF 0,09, contra ~13 min em CPU), com CPU como
fallback explícito (nunca silencioso).

### Instalar o backend Vulkan (uma vez)

```bash
# 1. ferramentas de compilação (único passo com sudo)
sudo dnf install -y cmake gcc-c++ vulkan-headers shaderc

# 2. whisper.cpp com Vulkan (código em thirdparty/, binário local, sem sudo)
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git thirdparty/whisper.cpp
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git thirdparty/SPIRV-Headers
cmake -B thirdparty/SPIRV-Headers/build -S thirdparty/SPIRV-Headers \
  -DCMAKE_INSTALL_PREFIX="$PWD/thirdparty/prefix" > /dev/null
cmake --install thirdparty/SPIRV-Headers/build > /dev/null
cmake -B thirdparty/whisper.cpp/build -S thirdparty/whisper.cpp \
  -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release \
  -DVulkan_LIBRARY=/usr/lib64/libvulkan.so.1 \
  -DCMAKE_PREFIX_PATH="$PWD/thirdparty/prefix" \
  "-DCMAKE_CXX_FLAGS=-I$PWD/thirdparty/prefix/include" > /dev/null
cmake --build thirdparty/whisper.cpp/build -j6 --config Release --target whisper-cli

# 3. modelo ggml-medium (~1,5 GB, mesmo "medium" do faster-whisper)
mkdir -p models
curl -sL -o models/ggml-medium.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# 4. verificar: Vulkan enxerga a B580?
vulkaninfo --summary | grep -A1 GPU0   # Intel(R) Arc(tm) B580 (driver Mesa ANV)
python tools/smoke_gpu.py              # Inference: SUCCESS, Device used: GPU
```

Versão validada: whisper.cpp 1.9.4-dev (2026-09-18), Mesa ANV 26.2.2,
ggml-medium.bin (1533 MB na VRAM durante inferência).

Seleção do backend (`--transcribe-backend` ou `CLIPPER_TRANSCRIBE_BACKEND`):

| Valor | Efeito |
|---|---|
| `auto` (padrão) | whisper.cpp/Vulkan → OpenVINO GPU → CPU (fallback com motivo impresso) |
| `gpu` | **Exige GPU**: usa Vulkan ou OpenVINO/GPU; se indisponível, **falha claramente** (sem fallback) |
| `openvino` | Só OpenVINO GPU; se indisponível, CPU com motivo |
| `vulkan` | Só whisper.cpp; se indisponível, CPU com motivo |
| `cpu` | Sempre CPU (`faster-whisper` int8, threads limitados) |

No início o programa sempre imprime (exemplo real com Vulkan ativo):

```text
GPU detected: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]
Transcription backend: VULKAN
Video acceleration: h264_qsv
```

ou, se a GPU não puder ser usada:

```text
GPU detected: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]
Transcription backend: CPU
Reason: OpenVINO instalado mas sem device GPU visível (falta driver Level Zero/OpenCL?)
```

**Vídeo:** o encode usa `h264_qsv` (Quick Sync, VRAM da B580) quando uma sonda
funcional de 10 frames passa; senão cai para `libx264` com 1 aviso por clipe.
Filtros (`scale/crop/pad/subtitles`) e detecção de rosto (OpenCV) continuam na
CPU — são leves perto da transcrição/encode.

### Benchmark medido: CPU vs B580 (60 s de áudio, Whisper medium)

| Backend | Tempo | RTF | CPU | RAM pico |
|---|---|---|---|---|
| whisper.cpp + Vulkan / B580 | ~4,6 s (incl. carga do modelo) | **0,08** | ~1 core | 187 MB |
| faster-whisper int8 / 4600G 4 threads | 134,8 s | 2,25 | 366% | 2,6 GB |

~30× mais rápido, ~14× menos RAM, CPU livre. Medido em 2026-09-18 com
`/usr/bin/time -v` nos dois binários sobre o mesmo WAV de 60 s. Sem promessa
genérica de speedup: esses são os números desta máquina.

### Como o backend Vulkan funciona (resumo honesto)

Áudio em chunks de 30 s via ffmpeg (1 por vez, removidos após uso) →
1 processo `whisper-cli` por chunk (sequencial, timeout 300 s, modelo carregado
1× por chunk) → JSON completo (`-ojf`, com word-timestamps reais) → segmentos
com tempos em segundos. VRAM: 1533 MB durante inferência, liberada ao fim de
cada chunk. Nada é paralelo: sem dezenas de threads/processos.

Limitações do backend Vulkan: áudio altamente repetitivo pode gerar alucinação
do Whisper (traço do modelo, não do backend); cada chunk recarrega o modelo
(segundos, aceitável); sem `intel-media-driver` o QSV cai para `libx264`.
O backend OpenVINO continua existindo como alternativa (exige
`intel-level-zero` + `pip install openvino optimum-intel transformers soundfile`).

**RAM:** vídeos de ~17 GB nunca são carregados inteiros. O áudio é extraído em
chunks de 30 s (~1 MB cada, 1 por vez, temporários removidos) e a transcrição
processa chunk a chunk. Com 16 GB de RAM o uso fica em poucos GB.

### Smoke test da GPU (sem vídeos reais)

```bash
python tools/smoke_gpu.py
```

Responde objetivamente (`GPU detected`, `Backend`, `Model loaded`,
`Inference`, `Device used`; exit 0 = inferência OK na GPU, exit 2 =
indisponível com motivo). Testa whisper.cpp/Vulkan primeiro (ordem do `auto`);
se Vulkan estiver indisponível, avalia o OpenVINO. Estado validado aqui:

```text
GPU detected: YES
GPU device: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]
Backend: vulkan (whisper.cpp + Vulkan, device 0 = B580)
Model loaded: YES (models/ggml-medium.bin, 1533 MB na VRAM)
Inference: SUCCESS (JSON válido gerado via Vulkan)
Device used: GPU
```

### Métricas de GPU Intel

O relatório registra o device detectado; utilização/VRAM ficam `null` quando
indisponíveis. Motivo: `nvidia-smi` não serve para a B580, `intel_gpu_top`
não está instalado e o sysfs do driver `xe` não expõe contadores confiáveis
de utilização/VRAM — `null` honesto em vez de número inventado.

## 2. Pegar uma chave grátis da NVIDIA Build

1. Crie uma conta em https://build.nvidia.com
2. Escolha um modelo de texto (ex: GLM 5.3, Nemotron, etc.)
3. Copie sua API key e o **nome exato do modelo** mostrado no code snippet
   da página do modelo (o nome muda com frequência, então confira lá)

```bash
export NVIDIA_API_KEY="nvapi-sua-chave-aqui"
export NIM_MODEL="z-ai/glm-5.3"   # default atual; ajuste conforme o catálogo
```

### Qual modelo escolher

Pontuar "isso é viral?" é julgamento subjetivo (humor, emoção, timing) —
não precisa de tool calling nem multimodal, então o critério principal é
qualidade de raciocínio/instrução, não tamanho bruto:

| Modelo | Quando usar |
|---|---|
| `z-ai/glm-5.3` (padrão atual, verificado vivo em 2026-09-18) | Decisão original do projeto — reasoning nativo, melhor para "sentir" hype/humor |
| `z-ai/glm-5.3-flash` | Variante mais rápida do mesmo modelo |
| `nvidia/nemotron-3-super-120b-a12b` | Alternativa testada, bom equilíbrio |
| `mistralai/mistral-nemotron` | Mais leve, use se os de cima estiverem instáveis no free tier |

> O modelo padrão anterior (`meta/llama-3.3-70b-instruct`, que nunca foi a
> escolha do projeto) entrou em EOL em 2026-08-26 (HTTP 410) e foi removido.
> Slugs do catálogo expiram — se o scoring falhar com `Gone`, liste os modelos
> vivos (`GET https://integrate.api.nvidia.com/v1/models`) e ajuste
> `NIM_MODEL`/`--model`.

> O nome exato do modelo no catálogo muda com frequência — sempre confira
> o slug certo no code snippet da página do modelo em build.nvidia.com
> antes de colocar no `NIM_MODEL`.

O script já trata automaticamente o caso de modelos com reasoning nativo
(que às vezes devolvem um bloco de "pensamento" antes da resposta final),
removendo esse bloco antes de interpretar o JSON.

## 3. Rodar

```bash
python clipper.py minha_live.mp4 --out cortes/ --top 8
```

Isso gera:

```
cortes/
├── 01_jogada_insana_no_final.mp4
├── 02_treta_com_o_chat.mp4
├── ...
└── manifest.json   # nota, título, hashtags, energy, speech_rate, snapped
```

### Métricas por execução

Toda execução — sucesso ou falha — gera automaticamente:

1. um resumo legível no terminal (`EXECUTION METRICS`);
2. um relatório JSON próprio em `metrics/`:

```
metrics/
├── 2026-09-18_101530_video-7min_a1b2c3.json
├── 2026-09-18_104812_video-18min_d4e5f6.json
└── 2026-09-18_112045_video-1h40_789abc.json
```

Um vídeo = um JSON, mesmo que o mesmo arquivo seja processado várias vezes
(timestamp + id único evitam colisão; nada é sobrescrito). Falhas também
geram relatório (`status: "failed"`, com etapa, tipo/mensagem do erro,
backend usado e retries).

O JSON contém: vídeo (nome, tamanho, duração, resolução, codec, FPS),
execução (início/fim, duração, status, erro, tempos por etapa, args),
transcrição (modelo, backend, tempo, RTF, segmentos), GPU/CPU/RAM
(agregados leves: média/pico — `null` quando indisponível), FFmpeg
(encoder, tempos, acertos/falhas) e API NVIDIA (requests, retries,
latências, tokens somente se a API retornar, custo sempre `null`).

Ctrl+C também gera relatório (`status: "interrupted"`, com a etapa
interrompida) antes de encerrar.

Monitoramento leve: 1 thread, 1 amostra a cada 2 s, só agregados em
memória. O vídeo nunca é carregado nem copiado (só `stat` + `ffprobe`).
Sem transcrição no JSON.

### Cache (não retranscreva 4h à toa)

```bash
python clipper.py live.mp4 --cache-dir .cache/clipper --out cortes/
# segunda vez: usa transcrição + scores do cache
python clipper.py live.mp4 --cache-dir .cache/clipper --out cortes2/ --top 5

# forçar refazer
python clipper.py live.mp4 --cache-dir .cache/clipper --force-retranscribe
python clipper.py live.mp4 --cache-dir .cache/clipper --force-rescore --context "ranked Valorant"
```

O fingerprint é `tamanho+mtime+modelo Whisper+idioma` — se trocar o arquivo,
o tamanho/mtime muda e o cache é invalidado automaticamente.

### Opções completas

| Flag | Efeito | Padrão |
|---|---|---|
| `--top N` | Quantos clipes gerar | 8 |
| `--no-vertical` | Mantém widescreen em vez de recortar pra 9:16 | 9:16 |
| `--no-captions` | Não queima legenda no vídeo | legenda on |
| `--model NOME` | Usa outro modelo do catálogo NVIDIA Build | env NIM_MODEL |
| `--cache-dir DIR` | Ativa cache de transcrição + scores | desligado |
| `--force-retranscribe` | Ignora cache de transcrição | off |
| `--force-rescore` | Ignora cache de scores | off |
| `--pad SEC` | Respiro extra antes/depois do corte (snap de frase) | 0.8s |
| `--no-audio-features` | Desativa medição de energia de áudio | áudio on |
| `--min-score F` | Score mínimo para considerar candidato | 6.0 |
| `--max-per-10min N` | Máximo de clipes por janela de 10 min (diversidade) | 2 |
| `--context TEXTO` | Contexto injetado no prompt (ex: "ranked Valorant duo com X") | — |
| `--examples JSON` | Arquivo few-shot com 2–4 exemplos (ver `examples.json`) | — |
| `--transcribe-backend B` | `auto` (Vulkan→OpenVINO→CPU), `vulkan`, `openvino` ou `cpu` | env ou `auto` |

Exemplos:

```bash
# calibrado pro seu conteúdo
python clipper.py live.mp4 --context "ranked Valorant, duo com fulano, humor ácido" \
  --examples examples.json --min-score 6.5 --max-per-10min 2

# corte mais solto, sem análise de áudio (mais rápido)
python clipper.py live.mp4 --pad 1.2 --no-audio-features --min-score 5.0
```

## 4. Como funciona por dentro

1. **Transcrição** (local, offline) — backend GPU Intel primeiro (whisper.cpp/Vulkan ou OpenVINO na B580), fallback `faster-whisper` CPU int8 com threads limitados; com cache opcional
2. **Janelas candidatas**: desliza janela 20–90s com passo 20s sobre a transcrição
3. **Snap + pad**: ajusta `start/end` para fronteira de palavra mais próxima (±1.5s) e adiciona `--pad` de respiro
4. **Áudio**: mede energia por janela via `ffmpeg volumedetect` → `alta/media/baixa` + `speech_rate` (palavras/s)
5. **Scoring**: cada lote de ~6 candidatos vai pra IA (NVIDIA Build) com prompt calibrado + duração/energia/speech_rate + contexto/few-shot; retry 3× com backoff 10/30/60s, clamp 0–10, falha marcada como `failed` (não score 0 silencioso)
6. **Seleção**: NMS com decaimento — penaliza `score * (1 - 0.8*overlap)` em vez de descartar binário; filtra por `--min-score`; espalha com `--max-per-10min`
7. **Corte**: `ffmpeg` com seek duplo (coarse `-ss` antes do `-i` + fine `-ss` depois, frame-accurate), `scale+pad` fallback se largura insuficiente, crop 9:16 centralizado na mediana do rosto (clamp [0.25,0.75]), legenda profissional (pausa >0.4s, 32 chars/linha, 2 linhas, 1.0s mínimo, FontSize 16), encode `h264_qsv` (B580) com fallback `libx264`

## 5. Ajustando pro seu conteúdo

- **Contexto sem editar código**: use `--context` e `--examples examples.json` (copie `examples.json` e troque pelos seus casos reais — 2 de nota alta, 2 de nota baixa). Sem isso, o prompt usa só o critério genérico.
- **Se a IA está errando o que é "bom momento"**: edite `--context` primeiro; se não bastar, adicione few-shot com trechos literais da sua live.
- **Se clipes vêm do mesmo trecho de 10 min**: diminua `--max-per-10min` para 1 ou aumente `--min-score`.
- **Se clipes cortam no meio da frase**: aumente `--pad` para 1.2–1.5.
- **Se legendas estão rápidas/pequenas**: ajuste `CAPTION_MIN_DURATION` / `FontSize` no topo de `clipper.py` (padrão: 1.0s / 16, antes era 0.5s / 14).
- `SEGMENTS_PER_SCORING_CALL` controla candidatos por chamada de API — aumente pra gastar menos requisições, diminua se a IA "perde o fio".

### Formato do `examples.json`

```json
{
  "examples": [
    {
      "transcript": "QUE CLUTCH INSANO 1V4",
      "score": 9.2,
      "reason": "clutch raro + hype",
      "title": "CLUTCH 1V4 ABSURDO",
      "hashtags": "#valorant #clutch #viral"
    }
  ]
}
```

Também aceita lista direta `[{...}, {...}]`. Máximo 4 exemplos lidos.

## 6. Limitações conhecidas

- **Modelo padrão pode expirar:** slugs do catálogo NVIDIA expiram (ex.:
  `meta/llama-3.3-70b-instruct`, morto em 2026-08-26 com HTTP 410 — nunca foi a
  escolha do projeto). O default atual (`z-ai/glm-5.3`) foi verificado vivo em
  2026-09-18 com chamada real de scoring; se falhar com `Gone`, liste os modelos
  vivos e ajuste `NIM_MODEL`/`--model`.
- **Espaço em disco:** o preflight falha se houver < 1 GB livre na pasta de
  saída e avisa se < 5 GB.
- **Arquivo original:** nenhuma saída pode sobrescrever o vídeo de entrada
  (o clipe é ignorado com erro se os caminhos coincidirem).
- Scoring ainda é majoritariamente textual — energia de áudio ajuda a pegar grito/risada, mas jogada visual 100% silenciosa continua difícil sem visão computacional.
- Detecção de rosto é Haar Cascade simples — funciona bem com webcam fixa, pode falhar se a câmera sai de cena (fallback: centro).
- Rate limit do free tier da NVIDIA Build é por minuto — o script já usa lotes + retry com backoff, mas lives de 4h+ ainda levam alguns minutos no scoring.
- Medição de áudio adiciona ~0.5–1s por candidato (ffmpeg `volumedetect`); use `--no-audio-features` se quiser scoring mais rápido.
- **Intel Arc:** `faster-whisper` nunca usa a B580 (CTranslate2 é CUDA-only) — por isso existem os backends OpenVINO/Vulkan. Sem o pacote `intel-level-zero`, o OpenVINO enxerga só `CPU` e o programa cai para CPU com aviso explícito. Filtros de vídeo e Haar Cascade continuam na CPU por simplicidade/correção.
