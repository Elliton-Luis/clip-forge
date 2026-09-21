# RELATÓRIO DE OTIMIZAÇÃO — energia de áudio sem decode de vídeo

Data: 2026-09-20. Implementação do P0 apontado em
`20260920_1232_auditoria_audio.md` (etapa `audio` = 35% do pipeline,
215 processos mortos pelo timeout de 8 s). Commit `76cfee1`
(`perf(audio): avoid video decoding during energy analysis`).

---

## 1. O que foi feito

Uma flag no comando de medição (`core/audio.py:12`):

```text
ANTES
ffmpeg -threads 2 -ss <start> -t <dur> -i <video>
    -filter:a volumedetect -f null -

DEPOIS
ffmpeg -threads 2 -ss <start> -t <dur> -i <video>
    -vn -map 0:a:0 -filter:a volumedetect -f null -
```

`-vn` impede o decode do vídeo; `-map 0:a:0` seleciona explicitamente o
primeiro stream de áudio. Preservados: `volumedetect`, limiares
alta/media/baixa (`max_volume` preferido, `mean_volume` como fallback),
interface de `measure()`, timeout de 8 s e fallbacks para `"media"`,
áudio da transcrição, áudio do vídeo final, scoring, seleção, captions,
timestamps.

Mudanças de apoio (mínimas, mesma causa):

- `core/config.py`: `AUDIO_ENERGY_VERSION = 1` — versão da medição.
- `core/cache.py`: `save/load_scores` gravam e exigem a versão; cache
  antigo (era dos `"media"` por timeout) é rejeitado com mensagem
  explícita e re-pontuado — sem ele, o ganho não apareceria em
  re-execuções com cache.
- `core/preflight.py`: aceita `NVIDIA_API_KEYS` (plural) além da
  singular — sem isso, nenhum pipeline com 3 chaves passava do
  preflight (o scoring já aceitava; o preflight não).
- `README.md`: 1 linha (`~0,2 s/candidato` no lugar de `~7 s`).

Não foi feito: paralelizar `measure()` (a auditoria vetou — multiplicaria
decodes), aumentar threads, refatoração, dependências novas.

---

## 2. Confirmações antes de alterar

1. `measure()` chamada só por `annotate()` (`audio.py:51`), 1× por
   candidato, serial; `annotate()` chamada 1× em `clipper.py:456`.
2. Retorno: `"alta"` (`> -7 dB`), `"media"` (`> -18 dB`), `"baixa"`,
   `"media"` em timeout/erro/sem volume.
3. `energy` chega ao scoring como pista textual no payload
   (`scoring.py:110`); não altera prompt, critério ou parsing.
4. Cache persistia `energy` sem versão — invalidação explícita
   implementada (acima).
5. Nenhum teste dependia do comando (`grep` em `tests/` vazio).
6. `-map 0:a:0` validado nos formatos do projeto: mp4/h264+AAC e
   mkv/AV1+2×AAC (usa o primeiro áudio, como antes). Vídeo sem áudio:
   erro do ffmpeg → `"media"`, igual ao comportamento anterior.

---

## 3. Benchmark antes/depois (função `measure()` real)

| Caso | Antes (auditoria) | Depois | Energy depois |
|---|---|---|---|
| AV1 1080p60, 20 s | 11,07 s | **0,14 s** | alta (real) |
| AV1 1080p60, 50 s | 27,7 s | **0,20 s** | alta (real) |
| H264, 20 s | ~2,1 s | **0,14 s** | alta (real) |
| H264, 90 s | 8,27 s (timeout) | **0,19 s** | alta (real) |

Speedup por janela: **~40–200×**. Prova de que o vídeo saiu do caminho
(mesmo comando da função, janela AV1 20 s):

```text
ANTES: video:497KiB audio:3742KiB
DEPOIS: video:0KiB audio:3742KiB + max_volume: -0.2 dB
```

Mesmos bytes de áudio, zero bytes de vídeo, volume medido de verdade.

---

## 4. Impacto no pipeline (run real: DerrubandoKit.mp4, 32 s)

```text
[2/5] 1 janelas candidatas geradas.
   -> medindo energia de 1 candidatos...      (instantâneo, antes ~7 s)
      distribuição: {'alta': 1}               (real; antes seria timeout→"media")
[3/5] Pontuando 1 candidatos em 1 chamada(s)... → 1 pontuado
[4/5] 1 clipes selecionados
[5/5] Cortando ... → clip h264+aac válido, mesmo título de runs anteriores
```

Candidatos, timestamps, captions e render final inalterados. (Duas
tentativas caíram em `503` transitório da API e a terceira passou —
instabilidade da NVIDIA no dia, sem relação com esta mudança.)

---

## 5. Testes

- Novos `tests/test_audio.py` (6, mockados, sem ffmpeg): `-vn`/`-map`
  presentes com ordem correta, limiares alta/media/baixa, timeout e erro
  → `"media"`, cache antigo rejeitado sem tocar candidatos, cache atual
  aceito. **6/6 OK.**
- Suite completa: **223 testes, 222 OK**. Único erro é pré-existente
  (`test_subs` exige fixture local `metrics_test/`, ausente; confirmado
  via `git stash` em etapa anterior, sem relação com esta mudança).

---

## 6. Resposta objetiva

```text
ANTES
audio: 1722,8 s (215 processos × ~8 s de timeout)
energy real: ~0 (26/26 "media" no manifest do run de 1h50)
energy media: 26/26

DEPOIS (medido por janela, função real)
audio: ~0,15–0,2 s por janela (projeção 1h50: ~30 s no lugar de 1722 s)
energy real: 4/4 janelas medidas + 1/1 no pipeline
energy media: 0 por timeout
speedup: ~40–200× por janela
```

E, principalmente:

```text
measure() agora decodifica somente áudio.
```

---

## 7. Anexos

**Arquivos alterados:** `core/audio.py`, `core/config.py`,
`core/cache.py`, `core/preflight.py`, `tests/test_audio.py` (novo),
`README.md`. Diff: +122/−5 (6 arquivos). **Commit `76cfee1`.**

**Risco conhecido (não é regressão):** energias agora são reais, então
notas/seleção mudam onde antes era tudo `"media"` — comportamento
pretendido do campo; comparar manifests antes/depois em runs com cache
invalidado (a invalidação força isso automaticamente).

**Arquivos temporários criados:** nenhum persistido (medições em `-f
null`; clips de validação em `/tmp`, removidos).
