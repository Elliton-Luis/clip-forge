# Timing das legendas — problema, experimentos e conclusões

Data: 2026-09-19. Projeto: clipper. Idioma dos commits: inglês; este documento é operacional, em português.

## 1. O problema observado

Nos vídeos finais, as legendas apresentavam comportamento temporal irregular:

```text
início: praticamente sincronizado
depois: pequena divergência que aumenta
em determinado trecho: volta a ficar sincronizado
no final: algumas legendas se sobrepõem
```

Padrão típico (não é offset constante):

```text
+0.0, +0.1, +0.2, +0.4, +0.2, +0.0, -0.1, +0.3, ...
```

Ocorrência mesmo em condições favoráveis (áudio limpo, só uma voz, sem
gameplay, modelo de scoring mais forte, sem `min-score`, sem limite de
clipes, reconhecimento das palavras bom).

## 2. Arquitetura relevante

```text
vídeo
 ↓ (ffmpeg, chunks de 30 s, 16 kHz mono)
whisper.cpp (Vulkan/B580, + VAD Silero) ou faster-whisper (CPU fallback)
 ↓ word timestamps (absolutos = chunk_offset + token_offset)
transcript global em cache
 ↓ scoring / seleção (define clip_start/clip_end, snap + --pad)
_group_cues() → _split_lines() (absoluto → relativo, uma única conversão)
 ↓ SRT / ASS (mesmo núcleo, timelines consistentes)
ffmpeg trim/atrim + setpts (corte 0-based) + filtro subtitles
```

## 3. Causa 1 (corrigida): timestamps degenerados → cues artificiais

**Diagnóstico.** Whisper.cpp+VAD emite tokens com `from == to`
(ex: `vou 20.760 → 20.760`) como continuação alucinada pós-fala.
`_group_cues()` aplicava `e = s + CAPTION_MIN_DURATION` (1,0 s) sobre
`e <= s`, transformando `20.760 → 20.760` em `20.760 → 21.760`.
Vários tokens no mesmo timestamp geravam 5 cues idênticas durante um
silêncio real de 9,24 s (GuilhermeGaivota.mp4).

**Correção** — `21163e9 fix(captions): drop degenerate word timestamps
before grouping`: regra de domínio `end > start` para gerar caption;
`end <= start` descartado no ponto único `_group_cues()` (cobre SRT, ASS,
debug e cache). Palavras reais curtas (`0 < dur < 1,0 s`) preservadas com
duração mínima visual. Debug registra `DROPPED DEGENERATE WORD`.

**Resultado.** GuilhermeGaivota: 14 → 9 cues, overlaps 10 pares → 0,
silêncio sem captions. 9 testes novos; suite 57 → 66.

## 4. Causa 2 (corrigida): extensão visual invadindo a próxima cue

**Diagnóstico.** A extensão de 1,0 s em palavra curta válida podia cruzar
o início da próxima cue (ex: `AH 10.00 → 10.10` virava `[10.00 → 11.00]`
sobre `VAMOS [10.50 → ...]`). Medição real no VideoMedio2: 24 overlaps em
13 clips/202 cues, todos com cue de duração exatamente 1,0 s.

**Correção** — `d3f844d fix(captions): clamp min-duration extension at
next cue start`: só a parte artificial é cortada (`e = max(e_real,
s_next)`); span real nunca reduzido (overlap real do Whisper preservado,
lanes resolvem a exibição); `fim == início` permitido. Mesmo clamp em
`_split_lines()`. Auditoria `WORD → CUE` adicionada ao `caption-debug.txt`.

**Resultado.** VideoMedio2: 24 → 0 overlaps. Suite 66 → 71.

## 5. O que foi descartado com evidência (não são a causa)

| Hipótese | Evidência |
|---|---|
| Offset global | Erro varia por região (`+0.9 … −3.6`); offset pioraria parte do vídeo |
| Chunk offset | Full vs chunks difere 0,01–0,02 s |
| Conversão abs→rel | `CUE.start − WORD.start = +0.000` em todos os testes |
| FFmpeg/PTS | `start_time` 0,0; `trim+setpts` correto; SRT == ASS |
| Padding | `--pad` validado (caption em 0,8 s com pad 0,8 é correto, não drift) |
| Duplicação de words | 79 → 79 words do JSON ao grouping |
| Lanes | Resolvem exibição espacial, nunca foram causa temporal |

## 6. Causa 3 (raiz residual): jitter do Whisper, agravado pelo VAD

Medição com áudio limpo (`teste_audio.mkv`, onset acústico por energia
≈ 0,700 s, modelo medium, mesmo chunk de 30 s):

| Config | Words | Degenerados | Erro no onset |
|---|---|---|---|
| sem VAD | 101 | 6 | −0,030 s |
| com VAD | 96 | 18 | −0,670 s |

Em áudio ruim (gameplay, mesmo chunk): sem VAD 0/33 degenerados, com VAD
21/28 (75%). O VAD além de alucinar continuação, **colapsa a cauda de
fala real** (`som 26.98 → 29.95` válidos sem VAD viram `25.49 → 25.49`
com VAD, mesmo com confiança `p` 0,92–1,00). Mecanismo interno:
`thold_pt` (limiar 0,01) só atribui timestamp a token com probabilidade
suficiente; o resto empilha no último timestamp válido.

**Status: documentado, sem correção por offset.** README registra jitter
natural (~0,5 s, VAD tende a adiantar) e proíbe offset global.

## 7. Experimento DTW (concluído, descartado)

whisper.cpp 1.9.4-dev suporta `--dtw PRESET` (heads do próprio modelo,
sem download). Detalhes encontrados: com `flash-attn` (default) o DTW é
**desabilitado em silêncio** (exige `-nfa`); ativo, preenche `t_dtw`
(ponto único por token, sem span) e **não altera `offsets`**.

Resultado A/B (mesmo áudio, 4 configs): `offsets` idênticos com/sem DTW;
`t_dtw` pior que offsets no onset (+1,10 vs −0,03) e inconsistente
(atrassa sem VAD, adianta com VAD); não recupera a cauda colapsada;
custo 1,2–2,2× tempo + perda do flash-attn. **Recomendação: descartar.**

## 8. Estado atual

- 71 testes passando; `21163e9` e `d3f844d` intactos;
- VideoMedio2: 0 overlaps; GuilhermeGaivota correto; SRT == ASS;
- padding, lanes, animações, hook, Vulkan/B580, fallback CPU: intactos;
- filtro de degenerados + clamp + auditoria WORD→CUE no `--debug-captions`.

## 9. Próximos passos (um por vez, sem offset global)

1. Tuning controlado do VAD (`-vt`, `-vp`, `-vsd`) ou via sem-VAD para
   passes sensíveis a timing — experimento separado, default inalterado.
2. Avaliar sinais de confiança descartados hoje (`p` do token,
   `probability`/`no_speech_prob` do faster-whisper) como critério de
   qualidade — sem heurística arbitrária, só com evidência A/B.
3. Se restar dessincronia em words válidos: nova investigação isolada,
   nunca compensação via etapas posteriores.
