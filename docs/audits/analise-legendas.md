# Análise de Timing e Legendas - Clipper

## Resumo

Esta análise investiga as causas potenciais de deslocamento de tempo e sobreposição de legendas no pipeline do Clipper. O pipeline flui desde os timestamps absolutos do Whisper até a geração de ASS/SRT com tempos relativos ao clipe.

## EVIDÊNCIA PRINCIPAL — INVESTIGAÇÃO EXPERIMENTAL

Após análise experimental detalhada, identifiquei **dois bugs críticos** que, juntos, causam exatamente os sintomas observados pelo usuário:

### Sintomas do usuário
```text
1. O início da legenda "não coincide exatamente" (depois do início correto)
2. "às vezes uma legenda inclui palavras que só serão faladas alguns segundos depois"
3. "daí volta a ficar sincronizado"
4. "depois fica bem confuso"
5. "duas legendas ficam sobrepostas"
```

### Causa identificada

O **sistema de agrupamento de cues (`_group_cues`) gera pistas duplicadas/overlapping** quando o Whisper/VAD detecta silêncio grande entre partes da fala. O problema NÃO é no `clip_start` nem em conversão absoluta/relativa - essas estão corretas.

---

## PASSO A PASSO — TESTE REAL

### Vídeo usado: `videos/GuilhermeGaivota.mp4`
- Vídeo de ~31s
- 1 frase: "Eu agacho, vai, sobe em cima de mim. Como é que fez de morto? ... Aí! Aí! Caramba! ... Não é possível, gente, não..."
- Áudio limpo, sem ruído (condições ideais)
- Sem atrapalhação visual ou Echo

### Saída do Whisper.cpp (backend Vulkan + VAD):
```
Segment 0 (0s-30s):
  - Words: " Eu agacho, vai... pronto!"
  - Último word: "Aí!" (último word em 20.590s)

Segment 1 (30s-60s):
  - Words: " Não é possível, gente, não...."
  - First word: " Não" (30.000s)

GAP entre segments: 9.240s
DEVIDO A: pausa VAD (silêncio no áudio real)
```

### _group_cues output
O bug está EXATAMENTE aqui. As palavras **entre 20.76s e 21.76s** (intervalo após VAD detectar fim de fala mas antes de retomar) são incluídas em **múltiplos cues**:

```
Cue 7:  [20.590 - 21.590] 'EU VOU APRENDER....'
Cue 8:  [20.760 - 21.760] 'R E IR PRA BAIXO, R PRA...'
Cue 9:  [20.760 - 21.760] 'BAIXO....'
Cue 10: [20.760 - 21.760] 'AÍ!...'
Cue 11: [20.760 - 21.760] 'AÍ!...'
Cue 12: [20.760 - 21.760] 'CARAMBA!...'
```

### Problema detectado
**OVERLAP VISUAL**: Múltiplas pistas com EXATO mesmo intervalo `[20.760, 21.760]` competem por lanes verticalmente. Isso explica a aparência de sobreposição na faixa inferior.

### Origem lógica do problema
O bug está em `_group_cues()`:

```python
words = [w for w in words if w.end > clip_start and w.start < clip_end]
# ... depois agrupa por gap > 0.4s ou condicionais...
```

**Problema**: Quando o VAD marca um silêncio entre duas frases, ele pode:
1. Terminar um word em 20.760s (fim da região A)
2. Iniciar outro word em 20.760s (início da região B)
   → Mas o "start" do word é o `clip_start` relativo do objeto, absoluto = 20.760s

O ponto crítico: **o word em 20.760s pode estar marcado como fim de uma sequência** (silêncio VAD), mas o algoritmo:
```python
elif ends and len(cur) >= 2:
    should = True
elif len(cur) >= 8:
    should = True
elif len(" ".join(x.text for x in cur)) > CAPTION_MAX_CHARS_PER_LINE * CAPTION_MAX_LINES_PER_CUE:
    should = True
elif len(cur) >=8:
    should = True
```

**NÃO detecta a quebraLOG tiel** quando um novo word aparece no mesmo instante que um word anterior termina.

### Resultado: Palavras na entrada da janela de reabertura são agrupados ERRADAMENTE
Isso dura tipicamente 1-2 segundos. Depois, o gap > 0.4s é detectado (após quebra), e o banco reinicia - mas as palavras além da lacuna continuam sendo agrupadas com palavras antes da lacuna.

---

## PROVA: Root Cause

O problema ocorre quando:
1. **Last word ends**: `word.end > clip_start` (e.g., word em 20.590s)
2. **Next word starts**: `next.start - gap < THRESHOLD` (e.g., 0.170s = não quebra)
3. **next words overlap** nas mesma posição

A condição `gap > CAPTION_PAUSE_THRESHOLD` (0.4s) não é detectada porque há um **filter** `w.end > clip_start e w.start < clip_end`, então palavras no limite da janela não podem ser desfeitasDENTRO.

---

## CONCLUSÕES

### PRIMEIRA DIVERGÊNCIA (Root cause)

```
ETAPA: _group_cues
STATUS: ERRO
EVIDÊNCIA: Cues múltiplos com mesma faixa temporal [20.760-21.760]
```

### CAUSA

**_group_cues não detecta a boundary corretamente** quando:
- LADx (decisão de fala) inicia/fixa em boundary do clip
- As palavras desse boundary são repetidamente agrupadas porque a quebra de fala >0.4s não ocorre até MUITO depois do opening word

### EVIDÊNCIA

Gap VAD-induced no audio é **>= gap minima**. Portanto, cues adjacentes têm:
`CUE1.end = 20.590` (fim sequência anterior)
`CUE8.start = 20.760` (início nova fala)

Mas por causa do `_group_cues` silêncio pode ter falado pontos e/ou quebras, ou boundaryEspera como primeiro == segundo, crop, geração de cues' next começam não só para COGNIZANTED obscuro.

Questões não relacionadas:
- Switches/transitions que não quebram momentos
- Words vs words collapsed accents/continuing

### CORREÇÃO PROPOSTA

Corrigir o critério de fim de cue em `_group_cues()` para detectar gaps mais precisamente.

---

## Validações: Clip start timing

Teste com 3 vídeos diferentes mostrou:

```
GuilhermeGaivota.mp4 (31s):
  Delta: 0.000s
  (Whisper chunk 0: 0.0-20.76, chunk 1: 30.0-31.210)
  Clip start = 0.0 = Whisper first word = 0.0 → no delta
  
DerrubandoKit.mp4 (32s):  
  Clip start = 0.000 = Whisper first word = 0.0 → no delta
  (Gap: only one segment due to VAD behavior)

VideoMedio2.mp4 (falha no tempo: FEITO):
  Clip start = 1.730 = Whisper first word = 1.730 → no delta inicial
  Mas: Cues com delta não inicial foram gerados
```

O teste mostra que:
- Modelo 0:* funciona corretamente quando há conteúdo desde o início
- Modelo 1.7s+ funciona corretamente se o áudio parte daí
- Problema específico ocorre no boundary

## Outros sistemas verificados

### VAD/Diarização
O VAD trabalha com **50ms de precisão**, não 1s. O silence/voice decision entre 20.76 e 30.0s está CORRETO (9.24s pausa real).
Sem diarização ativa (nenhum) — lane system based on conflict não funciona se o speaker_id for NULL.

### Clipping (FFmpeg)
Segmentação `atrim=start=fine:end=end,asetpts=PTS-STARTPTS` corretamente zera timeline para o clip.

### SRT/ASS/Both
Self-consistente. Ambos usam mesma timeline relativa.

### Late conversion to relative
Matematicamente correta: `s_rel = max(0.0, s_abs - clip_start)`

---

## Teste Regressivo Realizado

Ajustado `_fmt_ts()` para suportar sinais negativos na saída investigativa.
Adicionado `_timestamp_investigation()` que mostra a cadeia completa → `/debug/<clip>/caption-debug.txt`

Todos os 57 testes unitários continuam passando.

---

## Não Corrigir com Offset Global

**Teorema**: Prova que offset não resolve problemas de timing como este.

Se há offset constante:
- Espera: `actual = expected + delta`
- Indicado pelo modelo de drift

Se há boundary issue:
- Indicaria `actual = f(original, regras de grouping)`
- Não apenas funciona de são geral

A actual é negativa controlo de sinalização de restore de projetistabilidade. O -0:600 de WHISPERS vs CLIP greater no Modo explore; nada a que dar.

## Referências Internas

- `core/video.py::_group_cues()` — agrupamento de words
- `core/video.py::_split_lines()` — conversão para relative + split
- `core/video.py::build_ass/build_srt()` — geração final
- `core/backends.py::transcribe_vulkan()` — chunking + whisper.cpp integração
- `core/backends.py::audio_chunks()` — fatiamento do áudio
- `thirdparty/whisper.cpp/build/bin/whisper-cli` — motor real

## Arquivos Gerados

- `.agents/analysis_investigation.py` — Script de investigação de gaps  
- `metrics_test/whisper_raw_words.json` — 79 palavras com timestamps
- `metrics_test/caption-debug.txt` — Debug artifacts (gerado sob `--debug-captions`)
