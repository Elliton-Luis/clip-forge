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

`faster-whisper`/`CTranslate2` é CUDA-only: nunca usa Intel GPU. Por isso o
clipper tem backends próprios que usam a B580, com CPU como fallback explícito
(nunca silencioso).

```bash
# Fedora: drivers para a B580 expor GPU ao OpenVINO/ffmpeg
sudo dnf install -y intel-opencl-icd level-zero intel-media-driver
# verifique: vainfo | grep -i intel ; python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# esperado: ['CPU', 'GPU']

# backend OpenVINO (recomendado, pip puro)
pip install openvino optimum-intel transformers soundfile

# alternativa: whisper.cpp com Vulkan (binário externo + modelo ggml;
# baixe o modelo medium em models/ggml-medium.bin)
```

Seleção do backend (`--transcribe-backend` ou `CLIPPER_TRANSCRIBE_BACKEND`):

| Valor | Efeito |
|---|---|
| `auto` (padrão) | whisper.cpp/Vulkan → OpenVINO GPU → CPU (fallback com motivo impresso) |
| `openvino` | Só OpenVINO GPU; se indisponível, CPU com motivo |
| `vulkan` | Só whisper.cpp; se indisponível, CPU com motivo |
| `cpu` | Sempre CPU (`faster-whisper` int8, threads limitados) |

No início o programa sempre imprime:

```text
GPU detected: Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]
Transcription backend: OPENVINO
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
CPU — são leves perto da transcrição/encode. Sem promessa de speedup: meça no
seu hardware (`time`, `intel_gpu_top`, `htop`).

**RAM:** vídeos de ~17 GB nunca são carregados inteiros. O áudio é extraído em
chunks de 30 s (~1 MB cada, 1 por vez, temporários removidos) e a transcrição
processa chunk a chunk. Com 16 GB de RAM o uso fica em poucos GB.

## 2. Pegar uma chave grátis da NVIDIA Build

1. Crie uma conta em https://build.nvidia.com
2. Escolha um modelo de texto (ex: Llama 3.3 70B Instruct, Nemotron, etc.)
3. Copie sua API key e o **nome exato do modelo** mostrado no code snippet
   da página do modelo (o nome muda com frequência, então confira lá)

```bash
export NVIDIA_API_KEY="nvapi-sua-chave-aqui"
export NIM_MODEL="meta/llama-3.3-70b-instruct"   # ajuste conforme o catálogo
```

### Qual modelo escolher

Pontuar "isso é viral?" é julgamento subjetivo (humor, emoção, timing) —
não precisa de tool calling nem multimodal, então o critério principal é
qualidade de raciocínio/instrução, não tamanho bruto:

| Modelo | Quando usar |
|---|---|
| `zai/glm-5-3` (confirme o slug no site) | Melhor opção — tem reasoning nativo, ajuda bastante a "sentir" o que é engraçado/hype |
| `nvidia/nemotron-3-super-120b-a12b` | Alternativa mais testada/popular, 1M de contexto, bom equilíbrio |
| `mistralai/mistral-nemotron` | Mais leve, use se os de cima estiverem instáveis no free tier |

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

- **Modelo padrão pode expirar:** o `NIM_MODEL` padrão (`meta/llama-3.3-70b-instruct`)
  saiu do ar em 2026-08-26 (API retorna 410). Se o scoring falhar com `Gone`,
  passe `--model` com um slug vivo (liste em
  `https://integrate.api.nvidia.com/v1/models`; ex. testado:
  `nvidia/nemotron-3-super-120b-a12b`).
- **Espaço em disco:** o preflight falha se houver < 1 GB livre na pasta de
  saída e avisa se < 5 GB.
- **Arquivo original:** nenhuma saída pode sobrescrever o vídeo de entrada
  (o clipe é ignorado com erro se os caminhos coincidirem).
- Scoring ainda é majoritariamente textual — energia de áudio ajuda a pegar grito/risada, mas jogada visual 100% silenciosa continua difícil sem visão computacional.
- Detecção de rosto é Haar Cascade simples — funciona bem com webcam fixa, pode falhar se a câmera sai de cena (fallback: centro).
- Rate limit do free tier da NVIDIA Build é por minuto — o script já usa lotes + retry com backoff, mas lives de 4h+ ainda levam alguns minutos no scoring.
- Medição de áudio adiciona ~0.5–1s por candidato (ffmpeg `volumedetect`); use `--no-audio-features` se quiser scoring mais rápido.
- **Intel Arc:** `faster-whisper` nunca usa a B580 (CTranslate2 é CUDA-only) — por isso existem os backends OpenVINO/Vulkan. Sem `intel-opencl-icd`/`level-zero`, o OpenVINO enxerga só `CPU` e o programa cai para CPU com aviso explícito. Filtros de vídeo e Haar Cascade continuam na CPU por simplicidade/correção.
