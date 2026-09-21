# AUDITORIA DE DESEMPENHO — Clipper

Data: 2026-09-20. Execução de referência:
`metrics/2026-09-20_034436_2026-09-19_22-53-44_a96f46.json`
(vídeo 1h50m45s → pipeline 1h22m31s).
Etapa de auditoria: nenhuma otimização implementada, nenhum código
produtivo alterado, nenhum commit criado.

---

## 1. Executive Summary

Vídeo: **AV1 1080p60, 22,0 GB, 6645 s** (`videos/2026-09-19 22-53-44.mkv`).
Args: `large-v3`, `--top 50`, captions OFF. Total **4951 s**:

| Etapa | Tempo | % | Mecanismo |
|---|---|---|---|
| audio (energia) | 1722 s (28m43s) | 35% | 215× `ffmpeg volumedetect`, todos no timeout de 8 s |
| scoring | 1263 s (21m03s) | 26% | 36 lotes seriais, latência média 27,3 s + 7 retries com sleep |
| transcribe | 992 s (16m32s) | 20% | 222 chunks × `whisper-cli`, modelo 2,9 GB recarregado 222× |
| cutting | 971 s (16m12s) | 20% | 26 renders seriais, decode AV1 em software (libdav1d) por clip |
| candidates/selection | ~0,1 s | ~0% | puro, irrelevante |

Resposta curta: 1h50 vira 1h22 porque (1) a etapa de áudio decodifica
vídeo AV1 60 fps à toa e passa 28 min só esperando timeout; (2) o scoring
são 36 requests HTTP seriais de ~27 s; (3) cada clip re-decodifica AV1 em
CPU; (4) o Whisper recarrega o modelo a cada 30 s de áudio.

**Primeira alteração com maior ganho e menor risco:
adicionar `-vn -map 0:a:0` ao `volumedetect` em `core/audio.py`**
(medido 27,7 s → 0,14 s por janela; apaga ~28 min — hoje o resultado dessa
etapa é quase todo `"media"` por timeout mesmo).

---

## 2. Measured Pipeline

Fluxo real confirmado no código (`clipper.py:386` `run_pipeline`):

```text
vídeo (22 GB AV1)
 ↓ preflight (~0,3 s)
 ↓ transcribe: audio_chunks() → 222× ffmpeg (30 s PCM) + 222× whisper-cli serial
 ↓ candidates: build+snap (~0,1 s, 215 janelas)
 ↓ audio: annotate_audio → 215× ffmpeg volumedetect SERIAL, timeout 8 s cada
 ↓ scoring: 36 lotes de 6, SERIAL, 1 request por lote + retry 3× (sleep 10/30/60)
 ↓ selection (~0,001 s)
 ↓ cut: 26× [face_center_x (cv2) + ffprobe + ffmpeg trim/scale/blur/overlay + h264_qsv] SERIAL
 ↓ manifest.json + metrics JSON
```

---

## 3. Bottleneck #1 — Áudio: 1722 s, quase 100% desperdício (P0)

Onde: `core/audio.py:12` `measure()` → `core/audio.py:40` `annotate()`.
Comando atual (sem `-vn`/`-map`, saída `-f null` aceita vídeo):

```text
ffmpeg -threads 2 -ss <start> -t <dur> -i video -filter:a volumedetect -f null -
```

O FFmpeg **decodifica o vídeo AV1 1080p60 junto** (log medido:
`video:1242KiB audio:9365KiB`, 3000 frames p/ 50 s).

### Medições (amostras reais do arquivo de 22 GB, arquivos temp removidos)

| Método (janela 50 s em `t=3000`) | Tempo |
|---|---|
| atual (sem `-vn`, decodifica AV1) | **27,7 s** (também em `t=100`: 27,7 s) |
| mesmo comando + `-vn -map 0:a:0` (só áudio) | **0,14 s (~200×)** |
| `volumedetect` sobre WAV local | 0,13 s |
| extração de 600 s de áudio puro (`-vn`) | 0,98 s (≈683× realtime → áudio integral de 6645 s ≈ 11 s) |

### Prova do timeout em produção

`1722,8 / 215 = 8,01 s` = exatamente `AUDIO_ENERGY_TIMEOUT = 8`
(`core/config.py:80`). Uma janela mínima de 20 s tem 1200 frames AV1 →
~11 s de decode > 8 s de timeout. **Praticamente todos os 215 processos
foram mortos pelo timeout** e retornaram o fallback `"media"`.
Confirmação no produto: `cortes/manifest.json` tem **26/26 clips com
`energy: media`** (duração média 62 s — candidatos longos, decode ainda
mais caro). 28 min de CPU para produzir um rótulo constante.

Paralelismo aqui é irrelevante — o fix é eliminar o decode de vídeo.

---

## 4. Bottleneck #2 — Scoring serial: 1263 s (P0)

Onde: `core/scoring.py:252` loop `for start in range(0, len, 6)` →
`_call_with_retry` → `time.sleep` (`core/scoring.py:210`). Confirmado por
grep: **nenhum thread/pool/async/semáforo** no caminho.

Números do JSON: 215 candidatos / 6 por lote = **36 lotes = 36 sucessos**;
43 requests − 36 = **7 retries**; latência média 27,26 s; tokens 151836
(64907 prompt + 86929 completion); `total_time_sec` 1172 vs
`stages_sec.scoring` 1263 → **~91 s são só sleeps de backoff** (10/30/60).
Concorrência atual = **1**. Chaves: 1 (`keys: 1` — o código já faz
rodízio via `NVIDIA_API_KEYS` se houver 2+).

- Limite artificial: backoff fixo 10/30/60 s, timeout 180 s/request.
- Trabalho redundante: **não há** — cada candidato é pontuado 1×; lotes são
  partição disjunta (overlap de janelas é o desenho do produto, step 20 s).
- Concorrência potencialmente segura: **3–4 workers**. Batches
  independentes (escritas em índices disjuntos de `id_map`; `stats`
  precisaria de lock), sem ordem entre lotes, sem GPU/VRAM local.
  Risco: rate-limit 429 (retry com backoff já existe; 4×27 s não é
  agressivo com 1 chave). Ganho esperado: 1172 s → ~300–400 s (**~3×**).
- Dedup de candidatos semelhantes: **não recomendado** — seria mudança
  funcional.

---

## 5. Bottleneck #3 — Cutting: 971 s, re-decode AV1 em software (P1)

Onde: `core/video.py:961` `cut()`, chamado em série em
`clipper.py:494`. Por clip: `face_center_x` (cv2, **~0,37 s** medido —
barato, não é gargalo) + `_probe_dimensions` (ffprobe, ms) + 1 `ffmpeg`.

Medição com o filtro vertical idêntico ao código (30 s, sem legenda — o
run estava com captions OFF):

```text
Decoder confirmado no log: av1 (libdav1d) -> h264 (h264_qsv)
30 s de clip → 20,2 s wall (0,67× realtime), CPU 307%
```

- **Encoder QSV funciona de verdade** (não só flag no código). O custo está
  no **decode**: `libdav1d` em CPU de 1080p60 + filtergraph pesado (split,
  2× scale, crop, `gblur=sigma=25`, overlay) a 60 fps.
- Produção: 971,8 / 26 = **37,4 s/clip**, consistente com duração média
  62 s × 0,67 + overhead (áudio AAC + mux).
- Cada um dos 26 clips **reabre o arquivo de 22 GB e re-decodifica do
  zero** (`-ss <coarse> -i` + `trim`, `core/video.py:1059`).
- Paralelismo: **2 workers no máximo**. Decode dav1d já usa ~3 cores;
  2 renders ≈ 6 cores, QSV aguenta; 3+ arrisca throttling/VRAM e disputa
  de I/O no mesmo arquivo. Ganho **~1,7–1,9×**. Risco médio.

---

## 6. Bottleneck #4 — Transcrição: 992 s, modelo recarregado 222× (P1)

Onde: `core/backends.py:37` `audio_chunks()` (1 `ffmpeg` de 30 s por vez —
barato: **0,12–0,16 s** medido; saída WAV não carrega vídeo) +
`core/backends.py:205` `_decode_chunk()` → **1 processo `whisper-cli` por
chunk, serial, timeout 300 s**.

- Chunks: `ceil(6645/30)` = **222**. Produção: 992,6 / 222 = **4,47 s/chunk**
  com `large-v3` (2,9 GB; default seria `medium` 1,5 GB — o run forçou max).
- Piso medido: `large-v3` em 30 s de **silêncio = 3,19 s** (quase só load +
  init Vulkan); em **fala real = 6,33 s**; `medium` em silêncio = 3,97 s.
  Logo **~2–3 s por chunk é custo fixo** (mmap do modelo + init Vulkan)
  pago 222× ≈ **440–660 s dos 992 s**. RTF global 0,149 — saudável.
- O modelo é **recarregado a cada chunk** por construção (novo processo,
  `-m` a cada chamada). Eliminar isso exige daemon/servidor whisper.cpp —
  complexidade alta, risco de divergência de timestamps (o lab provou
  chunks ≡ global, então há caminho, mas é P2, não P0).
- Nota: `large-v3` global foi escolha do run; voltar ao `medium` é decisão
  de qualidade, fora do escopo de performance.

---

## 7. Redundant Work

| OPERAÇÃO | ONDE | VEZES | CUSTO | EVITÁVEL? | RISCO |
|---|---|---|---|---|---|
| Decode de vídeo AV1 p/ medir áudio | `core/audio.py:12` | 215× | ~1720 s | **Sim, `-vn -map 0:a:0`** | Baixo |
| Abertura + seek no MKV de 22 GB | `audio.py` + `backends.py:37` | 215 + 222 = 437× | ~1720 s + ~35 s | Parcial (extrair 1× ≈ 11 s) | Baixo |
| Load do modelo Whisper 2,9 GB | `backends.py:205` | 222× | ~440–660 s (est.) | Sim, c/ daemon | Médio–Alto |
| Decode AV1 1080p60 em software | `video.py:1059` (+ `audio.py`) | 26 + 215× | 971 s + 1720 s | Parcial (HW decode; `-vn` no audio) | Médio |
| `ffprobe` dimensões por clip | `video.py:1004` | 26× | ms | Sim (cachear), ganho ~0 | Baixo |
| Requests de scoring | `scoring.py` | 43 (36+7) | 1263 s serial | Só via concorrência | Baixo–Médio |
| `face_center_x` (3 seeks cv2) | `video.py:46` | 26× | ~10 s total | Não precisa | — |

---

## 8. Parallelization Opportunities

| Etapa | Hoje | Seguro | Ganho est. | Risco |
|---|---|---|---|---|
| audio `volumedetect` | serial 1 | **não paralelizar — eliminar** (`-vn`) | 1722 s → ~30 s (~50×) | Baixo |
| scoring (lotes) | serial 1 | **3–4 workers** (+ lock em `stats`) | ~3× (1172→~350 s) | Baixo–Médio (429) |
| cutting (clips) | serial 1 | **2 workers** | ~1,8× (971→~540 s) | Médio (CPU/VRAM/I/O) |
| transcrição (chunks) | serial 1 | **não** (1 GPU, VRAM ocupada) | — | Alto (Vulkan/VRAM, OOM) |
| candidates/selection | — | n/a (0,1 s) | 0 | — |

---

## 9. Cache Opportunities

O cache atual (`transcript` + `scores` por fingerprint
tamanho+mtime+modelo) já existe e funciona; **não é gargalo** (o run foi
cold-start legítimo). Registro: `scores` salva `energy`/`speech_rate`
junto — se o fix `-vn` mudar as energias, caches antigos continuam
formalmente válidos mas semanticamente diferentes; avaliar bump de
`TRANSCRIPT_PIPELINE_VERSION` ou invalidação de scores na implementação.

---

## 10. Risks

- `-vn` no audio: `volumedetect` consome só áudio — sem risco de cálculo;
  porém energias passam a ser reais em vez de `"media"`, o que muda
  notas/seleção (comportamento pretendido do campo; registrar como mudança
  de dados, não de seleção).
- Scoring paralelo: 429/503 (retry existe); logs embaralham; `stats`
  precisa de lock.
- Cutting paralelo: pico de CPU/RAM (dav1d 1080p60 × 2), pressão em QSV,
  leitura concorrente do mesmo arquivo de 22 GB.
- HW decode AV1 (`-hwaccel qsv`): o próprio código documenta que quebrou o
  init QSV no encode (`core/video.py:1051`) — só com prova por log.
- Whisper persistente: risco de divergência de timestamps e do retry
  anti-alucinação; revalidar contra `transcribe-lab`.

---

## 11. Recommended Optimization Order

| # | Otimização | Impacto | Risco | Complexidade | Prioridade |
|---|---|---|---|---|---|
| 1 | `-vn -map 0:a:0` no `volumedetect` (`core/audio.py`) | Alto (−28 min) | Baixo | Baixa (1 linha) | **P0** |
| 2 | Scoring paralelo limitado (3–4 workers + lock) | Alto (−13 min) | Baixo–Médio | Média | **P0** |
| 3 | Cutting com 2 workers | Médio (−7 min) | Médio | Média | **P1** |
| 4 | Extrair áudio 1× (≈11 s), derivar chunks/volumedetect do WAV | Médio (alternativa a 1) | Baixo | Média | **P1** (1 ou 4) |
| 5 | Decode AV1 por hardware no cutting (se sonda passar) | Médio | Médio | Média | **P1** |
| 6 | Whisper persistente (sem reload por chunk) | Médio–Alto | Alto | Alta | **P2** |
| 7 | Default `medium` vs `large-v3` (qualidade) | Médio | qualidade | Baixa | **P2** |

Estimativa combinada (P0+P1): 4951 s → ~1900–2300 s (**~2,2–2,6×**,
de 1h22 para ~35 min) sem tocar modelos, timestamps, VAD, captions,
seleção ou qualidade.

---

## 12. Anexos da auditoria

**Arquivos lidos:** `README.md`, `clipper.py`, `core/transcribe.py`,
`core/backends.py`, `core/audio.py`, `core/scoring.py`,
`core/selection.py`, `core/video.py`, `core/candidates.py`,
`core/cache.py`, `core/config.py`, `core/intel.py`, `core/metrics.py`,
`core/acoustic.py`, `docs/audits/2026-09-19-1349-timing-root-cause.md`,
`docs/experiments/2026-09-19-0813-vad-experiment.md` (parcial), `run.sh`, `cortes/manifest.json`.

**Medições realizadas:** `ffprobe` (AV1 1080p60 + 2× AAC, 6645 s,
22 GB); `volumedetect` atual 27,7 s vs audio-only 0,14 s vs WAV local
0,13 s (janela 50 s, offsets 100/3000); extração 600 s áudio puro 0,98 s;
chunk 30 s p/ transcrição 0,12–0,16 s; `whisper-cli large-v3` silêncio
3,19 s / fala real 6,33 s / `medium` silêncio 3,97 s; `face_center_x`
0,37 s; corte vertical 30 s 20,2 s com `av1 (libdav1d) → h264_qsv`
confirmado em log; `energy: 26× media` no manifest.

**Arquivos temporários criados:** somente em `/tmp/opencode/` (WAVs de
amostra, MP4s de teste, JSONs do whisper) — todos removidos ao final.

**Nenhum código produtivo alterado. Nenhum commit criado.**
