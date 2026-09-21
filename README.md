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

1. **Transcrição** (local, offline) — whisper.cpp + Vulkan na B580 (padrão),
   **sem VAD** (auditoria em `docs/20260919_1006_caption-audit.md`: o VAD apagava ~24 s de
   fala em gameplay e colapsava timestamps; ver também `docs/20260919_0813_vad-experiment.md`);
   fallback `faster-whisper` CPU int8 sem VAD, sempre com motivo explícito.
   Opcional: `CLIPPER_TRANSCRIBE_AUDIO_FILTER` (ex. loudnorm) aplicado só ao
   áudio da transcrição — off por padrão (sem ganho medido em áudio estourado).
2. **Janelas candidatas** — janela deslizante sobre a transcrição (duração
   configurável `--min-duration/--max-duration`, padrão 20–90 s; o máximo é
   teto real, nem o `--pad` estoura), passo 20 s;
   transcrição; `start/end` ajustados à palavra mais próxima (±1,5 s) + respiro
   `--pad`.
3. **Áudio** — energia por janela via `ffmpeg volumedetect`
   (`alta/media/baixa`) + `speech_rate` (palavras/s).
4. **Scoring** — lotes de ~6 candidatos para o LLM (NVIDIA Build) com prompt
   calibrado + duração/energia/speech_rate + contexto/few-shot; retry 3× com
   backoff 10/30/60 s; falha marca `failed` (nunca nota 0 silenciosa).
5. **Seleção** — NMS com decaimento (`score * (1 - 0.8*overlap)`), filtro
   `--min-score`, diversidade `--max-per-10min`. Modo experimental
   `--selection-mode peak`: detecta o auge de cada candidato (LLM em
   subjanelas de ~12 s, fallback heurístico sem rede) e constrói o clip ao
   redor dele (frases inteiras + mínimo, nunca estofado até o máximo);
   ranqueia por intensidade do auge, suprime mesmo-momento por overlap de
   peaks e re-titula só os selecionados com grounding validado. Comparar com
   `python clipper.py peak-compare video.mkv --top 5 --cache-dir .cache/clipper`
   (números em `docs/20260920_2213_peak-experiment.md`: 90 s → ~21–32 s nos mesmos momentos).
6. **Corte** — `ffmpeg` com seek rápido + `trim`/`atrim` frame-accurate;
   composição vertical 1080x1920 (vídeo 1080x1400 + faixas blur do próprio
   vídeo); legenda ASS (Montserrat ExtraBold, base, destaque amarelo nas
   palavras do título, pop de 280 ms) + **hook** (título do scoring em
   destaque grande nos primeiros 5 s, só na faixa borrada superior, nunca
   sobre o vídeo principal); encode `h264_qsv` (B580) com fallback
   `libx264`.
   Texto das cues remontado por `smart_join` (respeita fronteiras de token:
   "direita", não "dire ita"); tokens `[eot]/[sot]` filtrados na origem.
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

## Interface interativa (TUI)

```bash
python clipper.py        # sem argumentos abre a TUI
```

Menu: **Processar vídeo** · **Revisar transcrição** · **Aprovar transcrição** ·
**Finalizar sessão** · **Ver último relatório** · **Última transcrição** · **Sair**.
Navegação: setas movem, `←→`/Espaço altera opções e checkboxes, Enter edita
campos de texto ou confirma, Esc cancela. Só usa `curses` (stdlib, sem
dependência nova); sem terminal compatível, a CLI continua funcionando.

A TUI monta a **mesma configuração** da CLI (nada é reimplementado):
lista `videos/` para escolher a entrada (ou caminho manual), 9 modelos de
scoring (`config/models.json`, só LLMs de texto verificados), backend,
números (validados: `--top` > 0, `--pad` 0–5 etc.), modo das legendas
(`phrases`/`words`/`intervals` com ←→), `[x] Legenda *ÁUDIO ESTOURADO*`, arquivo de
risadas, checkboxes
(`[x] Vídeo vertical` = sem `--no-vertical`), contexto/examples e cache.
Antes de executar há uma tela de confirmação com o resumo **e o comando CLI
equivalente** (ex.: `python clipper.py video.mp4 --top 5 --model ...`),
para aprender a CLI junto. `Ctrl+C` e falhas geram relatório normalmente.

Opções avançadas (contexto, examples, forçar reprocessamento, debug de
legendas) ficam no final do formulário, sem esconder nada.

## Instalação

```bash
# 1. sistema (Fedora 44 testado): ffmpeg + drivers da B580
sudo dnf install -y ffmpeg intel-level-zero intel-media-driver

# 2. Python
pip install -r requirements.txt
# -> faster-whisper, openai, opencv-python, psutil

# 3. chave grátis do scoring (https://build.nvidia.com)
export NVIDIA_API_KEY="nvapi-sua-chave-aqui"
# Opcional: 2+ chaves (suas ou com consentimento) em rodízio por lote/tentativa
# contra quota/503 — valores nunca aparecem em log, só o índice (chave 1/2):
# export NVIDIA_API_KEYS="nvapi-chave-1,nvapi-chave-2"
export NIM_MODEL="nvidia/nemotron-3-super-120b-a12b"   # default atual; ajuste conforme o catálogo

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

Anti-alucinação: cada chunk passa por detector de repetição (`core/quality.py`:
token ×10+ ou 3-grama ×3+); chunk sinalizado é re-decodificado 1× com
`models/ggml-large-v3.bin` (~3 GB, `WHISPER_RETRY_MODEL`, `off` desliga) e o
retry só vale se limpar a sinalização (senão mantém o original + log).
Medido 2026-09 em gameplay ruidoso: medium alucinava loop ("toda a
plataforma", `p` até 1.00, frase real perdida); retry large recuperou a frase
("Agora eu vou matar... picaretada"). Temperatura (`-tp 0.2`) e `best-of 8`
não resolveram (máx suportado é 8). Default segue medium: large adiciona
prefixo alucinado no limpo ("Vamos lá" antes do onset) — maior nem sempre é
melhor; `WHISPER_MODEL_SIZE=large-v3` troca globalmente se quiser.

Limitações: áudio altamente repetitivo pode alucinar o Whisper (traço do
modelo, agora detectado e retentado); cada chunk recarrega o modelo
(segundos); utilização/VRAM ficam
`null` no relatório (`nvidia-smi` não serve p/ Intel, `intel_gpu_top`
ausente, sysfs do driver `xe` sem contadores — `null` honesto).

### Laboratório de transcrição (sem VAD)

`docs/20260919_0813_vad-experiment.md` provou que o VAD prejudica timestamps (onset −0,67 s,
cauda colapsada, truncagem de ~22 s em gameplay). O lab compara estratégias
**sem VAD** sobre o mesmo áudio, sem tocar o pipeline produtivo:

```bash
python clipper.py transcribe-lab video.mkv --dur 30                     # baseline
python clipper.py transcribe-lab video.mkv --dur 30 --audio clean       # áudio tratado
python clipper.py transcribe-lab video.mkv --dur 30 --mode global       # áudio inteiro
python clipper.py transcribe-lab video.mkv --start 15 --dur 30 --context 5  # contexto
python clipper.py lab-compare debug/transcription-lab/experiment-001 \
                          debug/transcription-lab/experiment-002
```

Eixos: `--audio original|normalize|clean|compressed|denoised|declipped` (só p/ transcrição; o vídeo final
nunca usa esse áudio), `--mode chunks|global`, `--context S` (só chunks),
`--whisper-model` (qualquer `ggml-*.bin` em `models/`), `--temp`/`--best-of`
(experimentos de decodificação), `--start/--dur` p/ recortar o trecho. Cada experimento salva `config.json`, `transcript.json`,
`transcript.txt`, `words.txt` (`[MM:SS.mmm → MM:SS.mmm] WORD`) e `metrics.json`
em `debug/transcription-lab/experiment-NNN/` (ignorado no git); `lab-compare`
imprime tabela palavra-por-palavra com Δinício/Δfim (nunca média de timestamps).

Evidência medida (2026-09-19, medium, trechos de 30 s):

| Comparação | Resultado |
|---|---|
| chunks vs global (limpo e ruim) | **idênticos** (101/101 e 23/23 words, Δ=0,000) — chunking não degrada |
| `--audio clean` no ruim | 23→26 words, 0 degenerados, pontuação/partículas melhores, mesmo tempo |
| `--audio clean` no limpo | 101→99 words, 6→5 degenerados, Δ médio ~0,25 s |
| `--context 5` (fronteira 30 s) | **diverge** (Δ médio ~0,56 s) — sem benefício medido aqui |
| presets em áudio clipado 90 s (`compressed/denoised/declipped`) | 0 deg em todos, mas +27 extras com Δ~0,6–0,8 s e confiança menor (0.72–0.75 vs 0.84) — assinatura de alucinação, **nada vira default** |

### Legendas por frases (conteúdo × apresentação)

A legenda acompanha **unidades naturais de fala**, não grupos arbitrários:

```text
words do Whisper (timestamps individuais)
        ↓
core/phrases.py — CONTEÚDO: que palavras formam cada frase
        ↓
core/video.py — APRESENTAÇÃO: SRT/ASS, quebra de linha, lanes, highlight
```

Regras do agrupamento (`phrases`, padrão): fronteira de frase (`.!?`) quebra,
exceto continuação próxima (gap ≤ 3 s + minúscula/conjunção/"…"); pausa interna
real (≥ 0,8 s) vira `…` (`Eu achei que… você tinha entendido`); vozes
concorrentes nunca se misturam; palavra nunca parte; sem esticar tempo, sem
redistribuir uniformemente, sem limite de caracteres como critério de corte.
Cada frase carrega `word_spans` (texto/início/fim por palavra) para destaque
da palavra ativa no futuro. Alternativas: `--caption-mode words` (blocos de
leitura) ou `intervals` (rajadas).

```bash
python clipper.py caption-lab video.mkv --start 0 --dur 30 --cache-dir /tmp/c
# debug/caption-lab/<video>_0-30/{words.srt,intervals.srt,phrases.srt,compare.json}
```

### Laboratório de legendas (words vs intervals vs phrases)

`--caption-mode intervals` agrupa por rajada de fala (pausas < 0,8 s não
quebram; rajadas > 10 s partem em fronteira de palavra) em vez de blocos de
leitura (gap 0,4 s, 8 words, 48 chars). Medido: limpo 30 s → 15 cues/89% vs
5 cues/95%, max 3,1 s vs 9,7 s; gameplay 60 s → 29 cues/99,9% vs 9/100%, max
4,8 s vs 10,1 s; 0 palavras partidas nos dois (regra de fronteira vale nos
três modos). Medido 2026-09-21 (DerrubandoKit 0–30, diálogo limpo):
`phrases` = 7 cues/99,9% idênticas às do `words` (sem regressão em texto
limpo; a vantagem aparece em fala fragmentada: frases unidas com `…` em vez
de balões partidos). **Default é `phrases`**.

### Forced alignment (experimental, opt-in)

O Whisper escreve o texto; o alignment só re-mede *quando* cada palavra foi
dita (texto nunca muda). Desligado por padrão — o pipeline sem a flag é
byte-idêntico ao de antes.

```bash
python clipper.py live.mp4 --out cortes/ --align whisper-refine   # sem deps novas (B580)
python clipper.py live.mp4 --out cortes/ --align wav2vec2         # exige torch+transformers
python clipper.py align-compare video.mkv --start 0 --dur 30 --backend wav2vec2
# debug/alignment-lab/<video>_0-30_<backend>/{whisper,aligned,compare}.json + compare.txt
```

| Backend | O que faz | Custo |
|---|---|---|
| `whisper-refine` | Re-decode do próprio Whisper em janela curta + casamento de sequência | Segundos na B580, zero deps |
| `wav2vec2` | CTC real (`jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`, 315 M) | ~0,7 RTF em CPU, download único ~1,2 GB |

Regras: cada word carrega `timestamp_source` (`whisper` ou
`forced_alignment`) e `confidence` opcional; sem confiança mínima (0.3) ou
span implausível (< 40 ms), o fallback mantém o Whisper e registra o motivo
— nenhum timestamp é fabricado. Sem `torch`/`transformers`, `wav2vec2` cai
para Whisper com erro explícito. Decisão entre opções, precisão esperada e
limites em `docs/20260920_2050_alignment-investigation.md`; números medidos em
`docs/20260920_2109_alignment-experiment.md` (resumo honesto: em gameplay ruidoso o gate
barrou 100% — zero inventado, zero ganho fingido).

### Áudio estourado (detecção, sem adivinhação)

`core/acoustic.py` detecta clipping digital sustentado (peak ≥ 0.99 + rms ≥
0.15 por ≥ 0.3 s; calibrado: áudio limpo = 0 eventos). Resultado vai ao
manifest (`acoustic_events` com `start/end/type/confidence`) e ao debug.
Queimar `*ÁUDIO ESTOURADO*` na legenda é opt-in: `--acoustic-captions`
(a palavra sempre vence o evento em overlap). Eventos saem em **amarelo**
(mesmo destaque do título; SRT segue texto puro, sem cor). Risada **não** é
detectada automaticamente: nos samples marcados (DerrubandoKit 18–23 e 27–30)
pico/RMS/ZCR/envelope/aspereza sobrepõem fala alta — sem assinatura confiável.
Para marcar risada (ground truth humano): `--laughs risadas.txt` (linhas
`INICIO FIM` em segundos) queima `*RISADA ESTOURADA*` em amarelo nos ranges,
via o mesmo caminho (lanes, manifest, debug).

### Duração dos clips

`--min-duration/--max-duration` (padrão 20/90, TUI tem os campos). O máximo é
respeitado de verdade: janelas nascem ≤ max, o snap corta o excesso do `--pad`
e a expansão p/ mínimo nunca estoura o máximo (mídia curta = clip curto, sem
compensação mágica). Nenhum clip passa do fim do vídeo: bounds de segmento do
Whisper podem extrapolar o áudio (overshoot do decoder), mas build+snap
clampam tudo em `media_end` (medido: janela [29.2, 60.8] em vídeo de 45.1 s
vira clip dentro da mídia). Ex.: `--max-duration 60` garante clipes de até 60 s.

### Falas simultâneas

Words com overlap real (> 0.1 s) viram **cues coexistentes** com timestamps
preservados — nunca uma sequência artificial ("EU VOU"+"NÃO VAI" em 10–11 s
gera duas cues sobrepostas, exibidas em lanes/stacks diferentes). Sem
diarização: sem nomes de pessoas, só simultaneidade mantida. Overlaps
microscópicos (jitter de medição) seguem sequenciais. Quebra de cue também
respeita fronteira de palavra: token de continuação (`ita` em `dire`+`ita`)
nunca inicia cue — sem "DIRE"/"ITA" separados.

## Scoring e modelos

Pontuar "isso é viral?" usa LLM remoto **só com texto** (trecho de até 1500
chars + tempos/energia/speech_rate). **Vídeo e áudio nunca saem da máquina.**

| Modelo | Quando usar |
|---|---|
| `nvidia/nemotron-3-super-120b-a12b` (padrão, vivo em 2026-09-18) | Padrão atual — mesmo momento que o GLM em teste A/B, bem mais rápido |
| `z-ai/glm-5.3` | Decisão original — reasoning nativo, melhor p/ hype/humor, scoring mais lento |
| `z-ai/glm-5.3-flash` | Variante mais rápida do mesmo modelo |
| `nvidia/nemotron-3-super-120b-a12b` | Alternativa testada |
| `mistralai/mistral-nemotron` | Mais leve, se os de cima instabilizarem |

Slugs expiram (ex.: `meta/llama-3.3-70b-instruct` morreu em 2026-08-26 com
HTTP 410). Se o scoring falhar com `Gone`, liste os vivos
(`GET https://integrate.api.nvidia.com/v1/models`) e ajuste
`NIM_MODEL`/`--model`. O código remove blocos de reasoning (`<think>`) antes
de ler o JSON.

> Atenção ao custo real: modelos de reasoning (ex.: GLM) raciocinam antes de
> responder — cada lote leva ~1 min ou mais. Num teste, o scoring GLM levou
> 17 min num vídeo de 7 min (85% do run). O default Nemotron é bem mais rápido
> com o mesmo momento selecionado.

### Concorrência do scoring

Lotes independentes rodam em paralelo: **1 worker por chave, teto de 3**
(`NVIDIA_API_KEYS="k1,k2,k3"` — mesma convenção de antes; 1 chave =
comportamento serial de sempre). Cada chave tem pacing próprio (~30 RPM,
margem sobre os 40 RPM da API) e todo retry passa pelo mesmo pacing;
`429` honra `Retry-After`. Resultados sempre voltam ao candidato original
pelo id — ordem de conclusão não afeta seleção. Ao final o scoring imprime
`[perf]` com wall vs estimativa serial (speedup), latências min/med/max,
retries e 429s (também em `metrics/*.json`, campos `workers`,
`min/max_latency_sec`, `rate_limited`). Prompt, modelo e parsing
inalterados: só o *quando* mudou, não o *quê*.

## Revisão humana da transcrição

A transcrição aprovada é a fonte da verdade: scoring, títulos, legendas e
cortes consomem ela, nunca o Whisper direto. Texto e timestamps são
independentes (corrigir "dire ita"→"direita" não toca `start/end`).

## Revisão de clips antes do burn-in

Depois da seleção, `--review-clips` pausa para o último filtro humano: para
cada clip selecionado (nunca dezenas de descartados) mostra título, score,
duração, transcript com timestamps e um preview **sem legenda queimada**
(`work/<video>/clipreview/preview_NN.mp4` — mesmo corte e áudio do final):

```bash
python clipper.py video.mkv --out cortes/ --review-clips
# [A]ceitar  [E]ditar (texto/tempos, sem JSON manual)  [S]pular  [P]tocar  [Q]sair
```

`A` mantém tudo e vai ao burn-in; `E` abre `clip_NN_words.txt`
(`[sNN] [MM:SS.mmm → MM:SS.mmm] texto`, timestamps absolutos) no `$EDITOR`
para corrigir texto/tempos com validação — a legenda final é gerada do texto
corrigido; `S` descarta sem erro; `Q`/Ctrl+C persiste
(`decisions.json`: accepted/edited/skipped/pending) e o rerun continua de
onde parou (candidatos diferentes arquivam em `decisions.bak.json`).
Previews morrem com a sessão (`finalize`/`reset`); originais nunca são tocados.

```bash
python clipper.py video.mkv --out cortes/ --review-transcript --review-titles
# 1. transcreve → work/<video>/transcription/{transcript.json, words.txt}
# 2. PAUSA: edite words.txt ([MM:SS.mmm → MM:SS.mmm] texto), Enter → APROVADA
# 3. scoring → PAUSA: edite work/<video>/titles.txt, Enter (valida grounding)
# 4. cortes + manifest.json (com title_warnings/highlight_warnings por clipe)
```

Regenerar sem Whisper (corrigiu legenda? só re-renderiza):

```bash
python clipper.py transcribe-approve work/<video>/transcription  # valida + aprova
python clipper.py video.mkv --out cortes/ --work-dir work/<video>  # pula o Whisper
python clipper.py finalize work/<video> --out cortes/  # confirma [Y/N], limpa
```

## Artefatos reutilizáveis + FINISH

Todo run persiste `work/<video>/artifacts/transcript.json` (fonte de verdade:
texto, segmentos, palavras, timestamps). `title.json`/`captions.json` derivam
dele com envelope de procedência (fingerprint origem+modelo+config); válido =
reutiliza, ausente/inválido = gera só o que falta. Título/legenda/render
nunca rodam Whisper:

```bash
python clipper.py finish clip.mp4 --out final/                    # tudo
python clipper.py finish clip.mp4 --only title                    # só título
python clipper.py finish clip.mp4 --only captions --out final/    # só .srt
python clipper.py finish clip.mp4 --only render --out final/      # só render
python clipper.py finish clip.mp4 --title "Meu Título" --only render --out final/
```

Detalhes: `--custom-words vocab.json` (`{"words": [...]}`) aplica correção de
palavra inteira após transcrever (whisper.cpp não tem prompting de vocabulário;
o mecanismo é pós-processamento exato e registrado). Títulos com palavras fora
da transcrição aprovada geram `title_warnings` no manifest em vez de passarem
silenciosamente (o prompt do scoring também exige título-recorte fiel). Na TUI,
as mesmas opções são checkboxes, e o menu tem **Revisar/Aprovar/Finalizar**
operando sobre `work/` sem flags (a edição abre seu `$EDITOR`). `work/` é
intermediário (ignorado no git); `finalize` recusa sessão não aprovada e guarda
`approved-transcript.json` no out.

## Métricas e cache

Toda execução gera: resumo `EXECUTION METRICS` no terminal + JSON próprio em
`metrics/<data>_<video>_<id>.json` (nunca sobrescreve; falhas e Ctrl+C também
geram). Conteúdo: vídeo, execução (tempos por etapa, args, erro),
transcrição (backend, tempo, RTF, segmentos), GPU/CPU/RAM (média/pico, `null`
se indisponível), FFmpeg, API NVIDIA (requests/retries/latências/tokens;
custo sempre `null`). Monitoramento: 1 thread, 1 amostra/2 s, só agregados.

Cache (`--cache-dir`, ligado no `run.sh`): transcrição + scores por
fingerprint `tamanho+mtime+modelo+idioma+versão-pipeline` — trocar o arquivo
invalida sozinho; mudar a semântica da transcrição (`TRANSCRIPT_PIPELINE_VERSION`)
invalida transcripts antigos (era VAD nunca volta por cache).
`--force-retranscribe` / `--force-rescore` refazem cada camada.

Zerar tudo de uma vez: `python clipper.py reset` (mostra o que vai, pede
confirmação) apaga `.cache/`, `work/`, `debug/` e temporários; `--logs`
inclui os relatórios `metrics/*.json`; `--yes` pula a confirmação. Nunca
toca vídeos, modelos, `cortes/`, código ou `.env`. Na TUI: menu
**Limpar caches** (`[C]`aches ou `[T]`udo com logs).

## CLI completa

| Flag | Efeito | Padrão |
|---|---|---|
| `VIDEO` | Arquivo de entrada (nunca alterado) | — (obrigatório) |
| `--out DIR` | Pasta de saída | `cortes` |
| `--top N` | Quantos clipes gerar | 8 |
| `--model NOME` | Modelo de scoring | env `NIM_MODEL` |
| `--transcribe-backend B` | `auto`, `gpu`, `vulkan`, `openvino`, `cpu` | env ou `auto` |
| `--whisper-model M` | Modelo Whisper (`medium`, `large-v3`, ...) — força máxima com PC em repouso | env ou `medium` |
| `--debug-captions` | Preserva ASS/SRT/transcript/diagnóstico em `debug/` | off |
| `--cache-dir DIR` | Ativa cache | desligado |
| `--force-retranscribe` | Ignora cache de transcrição | off |
| `--force-rescore` | Ignora cache de scores | off |
| `--pad SEC` (0–5) | Respiro antes/depois do corte (nunca estoura o max) | 0.8 |
| `--min-duration S` | Duração mínima do clipe (expande contexto) | 20 |
| `--max-duration S` | Duração máxima do clipe (teto real) | 90 |
| `--acoustic-captions` | Queima *ÁUDIO ESTOURADO* (experimental) | off |
| `--laughs FILE` | Ranges `INICIO FIM` p/ *RISADA ESTOURADA* amarela | — |
| `--no-vertical` | Mantém widescreen (1280px) | 9:16 |
| `--no-captions` | Sem legenda queimada | legenda on |
| `--no-title` | Sem título/hook queimado (independente das legendas) | título on |
| `--caption-mode` | `phrases` (frases, padrão), `words` (blocos) ou `intervals` (rajadas, experimental) | `phrases` |
| `--no-audio-features` | Pula energia de áudio (mais rápido) | áudio on |
| `--min-score F` (0–10) | Score mínimo | 6.0 |
| `--selection-mode` | `classic` (padrão calibrado) ou `peak` (experimental, clip ao redor do auge) | `classic` |
| `--max-per-10min N` (1–20) | Máximo por janela de 10 min | 2 |
| `--context TEXTO` | Contexto injetado no prompt | — |
| `--examples JSON` | Few-shot (ver `examples.json`, máx 4) | — |
| `--review-transcript` | Pausa p/ revisar/aprovar transcrição (`work/`) | off |
| `--review-titles` | Pausa p/ revisar títulos (`work/titles.txt`) | off |
| `--review-clips` | Revisa selecionados antes do burn-in (preview limpo, A/E/S/Q) | off |
| `--work-dir DIR` | Usa transcrição APROVADA, pula o Whisper | — |
| `--custom-words JSON` | Vocabulário (`{"words": [...]}`), correção exata | — |

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
  `! Falha clipe N` → pula o clipe, o job continua.
- **Legendas (sync/tokens/estilo)**: tempos absoluto→relativo com clamp em
  `[0, duração]` + validação matemática (avisos no log); corte usa
  `trim`+`setpts` (o `-ss` após `-i` deslocava legendas em `-fine`);
  tokens `[eot]/[sot]/...` filtrados na origem; words degenerados
  (`end <= start`, alucinação pós-fala do Whisper+VAD) descartados antes do
  agrupamento; extensão visual de 1,0 s nunca invade a próxima cue (só a
  parte artificial é cortada, span real preservado); ASS com `PlayRes` =
  frame real, Montserrat ExtraBold 54 px na base. Timestamps do próprio
  Whisper têm jitter natural (~0,5 s; chunking altera ~0,01–0,02 s) — sem
  offset global, sem VAD no caminho padrão.
  Desde a remoção do VAD do caminho padrão, a causa dominante de "legenda sem
  som" (VAD apagando fala) não existe mais; ver `docs/20260919_1006_caption-audit.md`.
- **Diagnóstico de legendas**: `--debug-captions` (ou `[x] Debug de legendas`
  na TUI) preserva por clipe `captions.ass`, `captions.srt`,
  `transcript-words.json` e `caption-debug.txt` (ABS→REL→CAP + checagem +
  auditoria WORD→CUE + drop reasons: `end <= start` vs `outside_selected_clip`
  + ENERGIA POR WORD: rms do áudio no span de cada word, `SILÊNCIO?` quando o
  Whisper afirma fala onde não há energia — diagnóstico report-only, nunca
  desloca timestamp)
  em `debug/`, mais `transcript.txt` legível do vídeo. Testes:
  `python -m unittest discover -s tests`.
- **Sem candidatos?** Vídeo sem fala — `segments: 0`.
- **Disco?** Preflight falha com < 1 GB livre, avisa com < 5 GB.
- **Interrompeu (Ctrl+C)?** Relatório `interrupted` em `metrics/` com a etapa;
  temporários se limpam; `pgrep -x ffmpeg` deve voltar vazio.
- **Original seguro?** Nenhuma saída pode ter o mesmo caminho do vídeo de
  entrada (clipe ignorado com erro se coincidir).

## Limitações conhecidas

- Scoring é textual + energia: jogada visual silenciosa continua difícil sem
  visão computacional. É também o gargalo de tempo (reasoning ~1 min/lote).
- Haar Cascade simples (webcam fixa; fora de cena → centro).
- Áudio adiciona ~0,2 s/candidato (`volumedetect` só-áudio, sem decode de
  vídeo); `--no-audio-features` pula.
- `faster-whisper` nunca usa Intel GPU (CUDA-only) — por isso o backend Vulkan.
- Cache por tamanho+mtime: edição que preserve ambos reutiliza cache (use
  `--force-*`).
- Sem suite automatizada; Linux (Fedora 44 testado) na prática.

### Título e legendas independentes

Título (hook + destaque) e legenda são recursos separados, ambos ligados
por padrão:

```bash
python clipper.py live.mp4 --out cortes/          # título + legenda
python clipper.py live.mp4 --out so_titulo --no-captions   # só título
python clipper.py live.mp4 --out so_legenda --no-title     # só legenda
python clipper.py live.mp4 --out limpo --no-title --no-captions  # nenhum
```

Na TUI: checkboxes `Título (hook)` e `Legendas`. Desligar um recurso pula
seu processamento de render (sem filtro `subtitles` quando não há o que
queimar); transcrição, scoring e seleção continuam iguais.
