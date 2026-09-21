# AUDITORIA DE ÁUDIO — por que a etapa `audio` domina o pipeline

Data: 2026-09-20. Etapa de auditoria: **nenhum código alterado, nenhum
commit, nenhuma otimização implementada**. Medições somente leitura sobre
os vídeos existentes (comandos `ffmpeg`/`ffprobe` avulsos em `/tmp`, nada
persistido no repo).

---

## 1. Executive Summary

A etapa `audio` é o maior gargalo em todos os runs recentes:

| Run | Candidatos | Tempo `audio` | % do total | Média/candidato |
|---|---|---|---|---|
| 22 GB AV1 1h50 | 215 | 1722,8 s (28m43s) | 35% | **8,01 s** |
| VideoMedio h264 7min (frio) | 14 | 103,8 s | 35% | **7,41 s** |
| VideoMedio h264 7min (cache) | 14 | 102,8 s | 47% | 7,34 s |

`AUDIO_ENERGY_TIMEOUT = 8` (`core/config.py:80`). Média de **8,01 s** =
o timeout. A etapa não está "processando por 28 min" — está **esperando
215 processos morrerem pelo timeout**, um por um, em série.

Causa raiz (uma linha): `core/audio.py:12` monta o comando sem `-vn` e
sem `-map`, e a saída `-f null` aceita vídeo — então **cada medição de
volume decodifica o vídeo inteiro da janela** (codec completo, 30/60 fps)
só para ler o áudio.

Medido (mesma janela, mesmo offset, só muda a seleção de streams):

| Janela | Atual (com vídeo) | Só áudio (`-vn -map 0:a:0`) |
|---|---|---|
| AV1 1080p60, 20 s (mínimo) | **11,07 s** (1200 frames) | **0,13 s (85×)** |
| AV1 1080p60, 50 s | **27,7 s** (3000 frames) | **0,14 s (~200×)** |
| h264, 20 s | 2,09 s (600 frames) | — |
| h264, 60 s | 5,81 s (1800 frames) | — |
| h264, 90 s (máximo) | **8,27 s** (2700 frames, estoura o timeout) | **0,22 s (37×)** |

Consequência nos dados: janelas que estouram o timeout retornam o
fallback `"media"` (`except TimeoutExpired: return "media"`). No run de
1h50, **26/26 clips do `manifest.json` têm `energy: media`** — 28 min de
CPU para produzir um rótulo constante. O `speech_rate` (palavras/s) é
calculado localmente e custa zero; só a energia é cara.

---

## 2. Todos os caminhos que encostam em áudio

```text
vídeo
 ├── A. audio_chunks() → WAV 30 s p/ transcrição   [core/backends.py:37]
 ├── B. measure() → volumedetect por candidato     [core/audio.py:12]  ← GARGALO
 ├── C. decode_pcm() → s16le p/ análise acústica   [core/acoustic.py:53]
 └── D. cut() → -af atrim + -c:a aac (render final)[core/video.py]
```

### A. `audio_chunks()` — barato, sem vídeo (não é gargalo)

```text
ffmpeg -ss <start> -t 30 -i video [-af filtro] -ar 16000 -ac 1 -c:a pcm_s16le chunk.wav
```

Saída WAV só carrega áudio: o FFmpeg **não decodifica o vídeo** (medido:
0,12–0,16 s por chunk de 30 s). Custo total na transcrição do vídeo de
1h50: 222 × ~0,15 s ≈ 35 s — irrelevante perto dos 992 s do Whisper.
Chamado 1× por chunk em `transcribe_vulkan` (`backends.py:250`) e no
`transcribe_openvino` (`backends.py:123`).

### B. `measure()` — o gargalo (28 min)

```text
ffmpeg -threads 2 -ss <start> -t <dur> -i video -filter:a volumedetect -f null -
```

Chamado 1× por candidato, **serial**, em `annotate()` (`audio.py:40`).
Provas de que decodifica vídeo:

- log real: `video:1242KiB audio:9365KiB` (50 s AV1), `video:497KiB`
  (20 s AV1), `video:1118KiB` (90 s h264) — frames contados no `frame=`.
- com `-vn -map 0:a:0`: `video:0KiB`, mesmo `audio:` byte a byte
  (ex.: 3742 KiB nos dois) — **o áudio lido é idêntico**, só o vídeo some.

Fronteira de timeout medida (acima = morto pelo timeout, vira `"media"`):

| Codec | Janela que estoura os 8 s |
|---|---|
| AV1 1080p60 (~1,8× realtime com 2 threads) | **qualquer uma ≥ ~15 s — inclusive o mínimo de 20 s** |
| h264 30 fps (~10× realtime) | **janelas longas, ~80 s+** (curtas passam em 2–6 s) |

Por isso o AV1 dá média cravada no timeout (8,01 s) e o h264 dá 7,4 s
(mistura de janelas curtas que passam e longas que morrem).

Detalhes secundários do comando atual:

- `-ss` **antes** do `-i` (seek rápido — correto para custo, não é o
  problema; com `-vn` o seek de áudio puro leva ~0,1 s mesmo em t=3000).
- `-threads 2` (`CLIPPER_FFMPEG_THREADS`; auto = cpu/2 nesta máquina).
  Mais threads acelerariam o decode de vídeo — mas otimizariam o trabalho
  errado.
- Sem `-map`: o arquivo AV1 tem **2 streams AAC** (índices 1 e 2); o
  comando usa o primeiro áudio por padrão. Comportamento igual com `-map
  0:a:0` — documentado para não virar dúvida na implementação.
- `mean_volume` vs `max_volume`: o código prefere `max_volume`
  (`vol = max_vol ... else mean_vol`); irrelevante para performance.

### C. `decode_pcm()` — barato, fora do caminho quente

```text
ffmpeg -ss <start> -t <dur> -i video -ar 16000 -ac 1 -f s16le -
```

Saída `s16le` = só áudio (sem decode de vídeo). Usado por:

- `detect_clipping()` — **só com `--acoustic-captions`** (opt-in, off nos
  runs medidos);
- `word_energy()` — **só em `--debug-captions`**, via
  `render_caption_debug()` (`video.py:885`).

Nenhum dos dois rodou nas execuções de referência. Quando rodam, é 1
decode de áudio por clip/trecho — ordem de décimos de segundo.

### D. Render final — áudio do clip, sem desperdício visível

`-af atrim+asetpts` + `-c:a aac 160k` dentro do `ffmpeg` do `cut()`:
re-encode do trecho uma única vez, junto ao vídeo — custo proporcional
ao clip, sem processo extra. Não é gargalo separável (está dentro dos
37,4 s/clip, dominados pelo decode AV1 de vídeo).

---

## 3. Por que serial + timeout = pior caso

`annotate()` é um `for` serial sem threads (`audio.py:48`). Cada iteração:

```text
candidato → fork ffmpeg → decode vídeo+nul → volumedetect
    → ou resultado em ~2–11 s, ou morte aos 8 s ("media")
```

Pior dos mundos: paga-se o custo do decode **e** descarta-se o resultado
quando ele importa (janelas longas/AV1 = justamente as de clips longos).
Paralelizar isso sem o `-vn` só multiplicaria decodes de vídeo
concorrentes — o fix é eliminar o decode, não distribuí-lo.

Estimativa com `-vn` (medições §1, sem implementar): 215 × ~0,15 s ≈
**~30 s** no lugar de 1722 s; 14 × ~0,2 s ≈ **~3 s** no lugar de 103 s.
Alternativa equivalente: extrair o áudio integral 1× (~11 s p/ 6645 s a
683× realtime, medido) e rodar `volumedetect` local (0,13 s). As duas
zeram o decode de vídeo; a primeira é 1 flag, a segunda muda a
arquitetura — decisão da etapa de implementação, não desta auditoria.

---

## 4. O que NÃO é problema de áudio (para não virar escopo)

- `audio_chunks()` (transcrição): ~35 s em 1h50 — ok.
- `speech_rate`: divisão local, custo zero.
- `--no-audio-features`: já pula tudo (válvula de escape existente).
- Streams duplos AAC: sem impacto (só o primeiro é lido; documentado).
- `CLIPPER_TRANSCRIBE_AUDIO_FILTER`: só afeta chunks de transcrição,
  nunca o `measure()` nem o vídeo final.

---

## 5. Riscos registrados para a implementação (não corrigidos aqui)

- Energias passarão a ser reais em vez de `"media"` quase constante →
  notas do scoring e seleção **mudam** (é o comportamento pretendido do
  campo, mas é mudança de dados: comparar manifests antes/depois).
- Cache de scores salva `energy`/`speech_rate` junto (`core/cache.py`) —
  avaliar invalidação na implementação.
- Nenhum risco de qualidade de vídeo/áudio final: o áudio do clip (D)
  não passa por este caminho.

---

## 6. Anexos

**Arquivos lidos:** `core/audio.py`, `core/backends.py` (audio_chunks),
`core/acoustic.py`, `core/config.py` (`AUDIO_ENERGY_TIMEOUT`,
`CLIPPER_FFMPEG_THREADS`), `core/system.py` (limites), `clipper.py`
(stage `audio`, flags `--no-audio-features`/`--acoustic-captions`),
`core/video.py` (`word_energy`, `cut`), `cortes/manifest.json`
(26/26 `energy: media`), `metrics/*.json` (3 runs).

**Medições realizadas** (comandos avulsos, só leitura; nada persistido):
probes A–F acima (h264 20/60/90 s, AV1 20/50 s, com e sem `-vn -map
0:a:0`); `ffprobe` de streams (AV1+2×AAC 48 kHz; h264+1×AAC);
`video:/audio:` bytes por run; fronteira de timeout por codec.

**Arquivos temporários criados:** nenhum (saídas foram para `-f null`).

**Nenhum código produtivo alterado. Nenhum commit criado.**
