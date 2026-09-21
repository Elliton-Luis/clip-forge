# TIMING-ROOT-CAUSE — onde nasce o erro temporal

Investigação empírica, sem nenhuma alteração de código/commit/config.
Arquivo: `videos/videofull.mkv` (297.283.589 bytes, mtime 2026-09-18,
duração 88.576 s). Modelo: `models/ggml-medium.bin`. Sem cache em nenhum
passo (whisper-cli direto + extração WAV fresca). Data: 2026-09-19.

---

## 1. Fala real no áudio (evidência independente, sem Whisper)

Método: `silencedetect` + envelope RMS em passos de 0.1 s, áudio integral.

- Silêncio de 3.13 s em 51.14→54.27.
- Subida abrupta de energia em 54.2→54.3 (rms 0.015→0.062).
- Rajada de fala até ~58.4; silêncio a partir de ~58.5 (rms < 0.01).

```text
fala real: START ≈ 54.3s, END ≈ 58.4s  (margem ±0.15s, bordas por energia)
duração ≈ 4.1s — compatível com a frase de ~15 palavras em ritmo de gameplay
```

## 2. JSON RAW do Whisper (pipeline atual, sem VAD, sem cache)

Comando direto, mesmo WAV que o pipeline extrairia (chunk 30–60, 16 kHz mono):

```text
whisper-cli -m models/ggml-medium.bin -f chunk30.wav -ojf -l auto/pt -nt
```

Resultado (idêntico com `-l auto` e `-l pt`): **1 segmento local [0.00→30.00]**,
18 tokens — alucinação repetitiva, com `p` alto nos tokens alucinados:

```text
' e' 0.00→0.42 p=0.09 | ' ó' 0.42→0.84 p=0.06 | 'leo' 0.84→2.09 p=0.67
' de' 2.09→2.94 | ' céu' 2.96→4.20 p=0.93 | ' por' 4.20→5.46
' toda' 5.46→7.15 p=0.99 | ' a' 7.15→7.57 | ' plataforma' 7.57→11.79 p=0.99
... "toda a plataforma" em loop até 24.45 (p até 1.00) ...
' você' 24.45→26.06 p=0.20 | '...' 26.09→29.99 p=0.18 | [_EOT_] 30.00
texto: "e óleo de céu por toda a plataforma toda a plataforma..."
```

A frase real ("Agora eu vou matar..."), audível em local 24.3→28.4 do chunk
(WAV conferido: rms 0.055 nesse trecho), **não foi reconhecida**. Confiança
alta (até 1.00) em palavra alucinada: `p` não discrimina alucinação.

Pipeline atual completo (`transcribe_vulkan`, sem VAD) em `videofull.mkv`:

```text
SEG [0.00->30.00]  "♪ ♪ ♪ ..."      (alucinação)
SEG [30.00->60.00] "e óleo de céu..." (alucinação; frase ausente)
SEG [60.00->90.00] "Ver se não vai dar bom..." (correto — mesmo texto da era VAD)
```

## 3. Tabela fundamental

| Fonte | Início | Fim |
|---|---|---|
| Áudio real | 54.30 | 58.40 |
| Whisper RAW (era VAD, `work/videofull`) | 30.00 | 47.80 (words; +7 degenerados em 47.80) |
| Whisper RAW (atual, sem VAD) | — | — (frase ausente; loop "plataforma") |
| Word após parse | = RAW + 30 (soma exata, verificada) | idem |
| Transcript global | = parse (cópia, sem retoque) | idem |
| Cue | = start da 1ª word (código verificado) | = end última word (+piso 1 s) |
| SRT / ASS | = cue − clip_start (subtração exata) | idem |
| Vídeo final | = SRT (trim+setpts 0-based) | idem |

Primeira etapa com valor diferente do real: **o próprio Whisper (etapa 1)**.
Nenhuma etapa posterior reescreve `Word.start/end` (verificado função a
função na auditoria anterior e reconferido neste HEAD).

## 4. Segmento vs words (o ponto da §4 do procedimento)

Era VAD: `segmento 30.0→58.61`, words `30.0→47.80` + pilha degenerada.
O Clipper usa **segment.start/end só para janelas candidatas** e
**word.start/end para captions** (código verificado). Respostas:

- segment 30→58 + words 55→59 → legenda começa em **55** (da word).
- segment 30→58 + words em 30 → problema nasceu **no Whisper**.
- Nosso caso era o segundo: words em 30–47.8 (Caso 4, §6 do procedimento —
  words erradas, não deslocamento constante; 17.8 s de words vs 4.1 s reais
  exclui offset). Nota: o `segment.end 58.61` coincide com o fim real (58.4),
  mas é fronteira de decodificação, não medição da fala.

## 5. Linha 45–65 s (Caso 4 confirmado para a era VAD)

```text
45.0 ───────────────────────────── 65.0
áudio real:      [silêncio 51.1-54.3][FALA 54.3→58.4][silêncio]
VAD-era words:   [30.0→47.8 + pilha DEG em 47.8]
VAD-era segment: [30.0→58.61]
atual sem VAD:   [ausente — alucinação "plataforma"]
```

## 6. Cache e proveniência (§8)

Nenhum cache do Clipper foi usado nesta investigação (CLI direto). O
`work/videofull/transcription/transcript.raw.json` é artefato antigo da era
VAD (status draft→approved pelo usuário) e foi usado só como objeto de
estudo, não como evidência do comportamento atual.

---

## VEREDITO

```text
O timestamp já está errado no RAW do Whisper.
```

Com o qualificador honesto de duas eras: na era VAD, words certas no tempo
errado (30–47.8 vs 54.3–58.4, timeline comprimida — mecanismo em código);
hoje, sem VAD, a frase some numa alucinação repetitiva de chunk ruidoso.
Nos dois casos, a primeira divergência está na medição do modelo.

## PRIMEIRA DIVERGÊNCIA

```text
Etapa: Whisper (alinhamento por token no chunk 30–60)
Valor correto: 54.30 → 58.40 (energia acústica, ±0.15s)
Valor observado (era VAD): 30.00 → 47.80 (+degenerados)
Valor observado (atual): frase ausente
Diferença: ~−24 s (era VAD) / perda total (atual)
Evidência: silencedetect + RMS + whisper-cli direto, sem cache
```

## TIMELINE

```text
ÁUDIO REAL   54.3 → 58.4  ████
WHISPER RAW  30.0 → 47.8 (VAD)  ou ausente (atual)
WORD         = RAW + offset (parse exato)
TRANSCRIPT   = WORD (cópia)
CUE          = 1ª word → última word (sem reescrita)
SRT          = CUE − clip_start (conversão única, exata)
VÍDEO        = SRT (trim+setpts, zero casado)
```

## CONFIANÇA: ALTA

- Fala real em 54.3–58.4: dois métodos independentes concordam (silêncio de
  3.1 s antes + subida 4× de energia + decaimento simétrico).
- Parse/pipeline inocentes: cada transformação reconferida no HEAD atual.
- Ressalva explícita: a alucinação atual foi medida em 1 run por configuração
  de idioma (2 runs totais, mesmo resultado); decodificação greedy é
  determinística, mas binary/modelo podem diferir dos que geraram o raw.json
  antigo — por isso a era VAD e a atual são reportadas separadamente.
