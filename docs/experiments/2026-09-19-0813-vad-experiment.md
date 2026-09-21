# VAD EXPERIMENT

Data: 2026-09-19. Projeto: clipper. Idioma dos commits: inglês; este documento é operacional, em português.

Regra respeitada: **nenhuma mudança permanente, nenhum commit, nenhum default alterado**.
Todos os testes invocaram `thirdparty/whisper.cpp/build/bin/whisper-cli` diretamente
com chunks de 30 s extraídos exatamente como `core/backends.py::audio_chunks`
(`ffmpeg -ss <off> -t 30 -ar 16000 -ac 1 pcm_s16le`). Scripts do experimento em
`/tmp/opencode/vad_exp.py`, `vad_matrix*.py` (fora do repo).

Modelo: `models/ggml-medium.bin`. VAD: `models/ggml-silero-v5.1.2.bin`.
Binário: whisper.cpp **1.9.4-dev** (`5670d5c`).

---

## 1. Parâmetros reais do VAD

Fonte: `thirdparty/whisper.cpp/include/whisper.h` (`whisper_vad_params`),
`src/whisper.cpp::whisper_vad_default_params()` e `examples/cli/cli.cpp`
(mapeamento CLI → `wparams.vad_params`, linhas ~1262–1270). Confirmado também via
`whisper-cli --help`. Nenhum default foi presumido.

| Flag | Nome | Default real | Unidade | Efeito real |
|---|---|---|---|---|
| `-vt` | `threshold` | **0.50** | probabilidade 0–1 | Probabilidade mínima do Silero para considerar o frame como fala |
| `-vspd` | `min_speech_duration_ms` | **250** | ms | Duração mínima para um segmento válido de fala (menores são descartados) |
| `-vsd` | `min_silence_duration_ms` | **100** | ms | Silêncio mínimo para considerar a fala como encerrada (divide segmentos) |
| `-vmsd` | `max_speech_duration_s` | **FLT_MAX** | s | Duração máxima antes de forçar novo segmento (na prática, nunca divide) |
| `-vp` | `speech_pad_ms` | **30** | ms | Padding adicionado **antes e depois** de cada segmento de fala |
| `-vo` | `samples_overlap` | **0.10** | s | Overlap ao copiar amostras do segmento de fala |

Onde são aplicados: o CLI preenche `wparams.vad_params` e o whisper.cpp detecta
fala por janelas (`whisper_vad_detect_speech_no_reset`), monta segmentos VAD com
pad/silêncio/mínimos acima e decodifica **por segmento**, remapeando timestamps
(`vad_mapping_table`).

O wrapper do Clipper **não sobrescreve nada**: `core/backends.py::transcribe_vulkan`
passa somente `-vm <modelo> --vad` (linhas 198–199). O projeto **não expõe** nenhum
desses valores (sem env `CLIPPER_VAD_*`, sem flag CLI, sem `config.py`).
Fallback CPU (`core/transcribe.py:47`) usa `vad_filter=True` do faster-whisper —
VAD diferente, fora do escopo deste experimento.

---

## 2. Baseline (reproduzido exatamente como no relatório anterior)

Áudio limpo = `videos/teste_audio.mkv` chunk 0–30 s, referência acústica
`onset ≈ 0.700 s` (confirmada por `silencedetect`: `silence_end: 0.68175`).
Áudio ruim = `videos/VideoMedio2.mp4` chunk 0–30 s (gameplay real, sem ref acústica).

| Config | Words | Deg. | Onset err | First | Last | Tempo |
|---|---|---|---|---|---|---|
| Limpo NoVAD | 101 | 6 | −0.030 s | 'Consider' 0.67→0.71 | 'som' 29.09→29.95 | 3.0 s |
| Limpo VADdef | 96 | 18 | −0.670 s | 'consider' 0.03→0.72 | 'som' 25.49→25.49 DEG | 2.9 s |
| Ruim NoVAD | 23 | 0 | — | 'só' 0.00→0.79 | 'safe' 28.28→29.99 | 2.0 s |
| Ruim VADdef | 35 | 27 | — | 'Ó' 0.00→0.28 | 'não' 6.45→6.45 DEG | 1.9 s |

Confirmação: o VAD default adianta o onset em ~0.64 s, triplica degenerados no
limpo e, no ruim, **trunca a fala real em ~6.5 s** (cobertura NoVAD vai até 29.99 s)
empilhando 27 words degenerados num burst com texto alucinado
("tem que parar… Eu chego não" vs. real "vamos safe, vamos safe").

---

## 3. Teste `-vt` (threshold) — áudio limpo

| VT | Words | Deg. | Onset err | Tail 'som' | Tempo |
|---|---|---|---|---|---|
| 0.2 (baixo) | 99 | 14 | −0.670 s | 25.88→25.88 DEG | 2.9 s |
| 0.5 (default) | 96 | 18 | −0.670 s | 25.49→25.49 DEG | 2.8 s |
| 0.8 (alto) | 99 | 16 | −0.670 s | 25.36→25.36 DEG | 2.9 s |

Onset **idêntico** nos três (−0.670 s). Degenerados variam 14–18 sem tendência.
**Threshold não é a causa**: alterar a sensibilidade do detector não move o onset
nem recupera a cauda.

---

## 4. Teste `-vp` (speech-pad-ms) — áudio limpo

| VP | Words | Deg. | Onset err | Tail 'som' | Tempo |
|---|---|---|---|---|---|
| 30 (default) | 96 | 18 | −0.670 s | 25.49→25.49 DEG | 2.9 s |
| 100 | 99 | 17 | −0.600 s | 26.82→26.82 DEG | 3.1 s |
| 200 | 92 | 9 | −0.500 s | 28.43→28.43 DEG | 2.9 s |
| **300** | **99** | **10** | **−0.400 s** | **29.61→29.75 OK** | **3.0 s** |
| 500 | 90 | 7 | −0.200 s | 53.55→60.00 **INVÁLIDO** | 2.9 s |

`-vp` é o parâmetro de **maior impacto**. Efeito monotônico até 300 ms: onset
−0.67 → −0.40, degenerados 18 → 10, cauda migra 25.49 → 29.61→29.75 **válida**
(span real, `p=0.93`, a ~0.5 s do NoVAD 29.09→29.95).
**VP500 é inseguro**: produz timestamp **fora do chunk** (`som 53.55→60.00` num
áudio de 30 s, 2 segmentos) — anomalia reproduzida 2×. Teto seguro observado: 300 ms.

---

## 5. Teste `-vsd` (min-silence-ms) — áudio limpo

| VSD | Words | Deg. | Onset err | Tail 'som' | Tempo |
|---|---|---|---|---|---|
| 100 (default) | 96 | 18 | −0.670 s | 25.49→25.49 DEG | 2.9 s |
| 500 | 95 | 14 | −0.670 s | 25.64→25.64 DEG | 2.9 s |
| 1000 | 95 | 13 | −0.670 s | 28.35→28.35 DEG | 2.9 s |

Onset **inalterado**. Reduz degenerados (18→13) e desloca a cauda para frente
(25.49→28.35), mas a cauda **continua degenerada**. Efeito secundário, menor que `-vp`.

Combinações no limpo: `VP200+VSD500` → 99/12/−0.50/cauda 27.85 DEG;
`VP200+VSD1000` → 97/10/−0.50/cauda 28.86 DEG. Nenhuma supera **VP300 isolado**.

---

## 6. Cauda da fala (caso principal) — últimas 12 words, áudio limpo

NoVAD (referência, tudo válido): `perto 26.38→26.77 … som 29.09→29.95`.
VADdef: **as 12 colapsadas em `25.49→25.49`** (mesmo com `p` 0.92–1.00).
VP200: 8 válidas + 4 colapsadas em 28.43. VSD1000: 6 válidas + 6 em 28.35.

VP300 (todas as 12 **válidas**):

```text
perto 26.82→27.16 | de 27.29→27.43 | mim 27.43→27.52 | , 27.72→27.90
não 28.05→28.17 | haver 28.17→28.62 | á 28.62→28.71 | var 28.71→28.90
ia 29.03→29.16 | ção 29.16→29.42 | de 29.46→29.61 | som 29.61→29.75
```

Deslocamento de ~+0.4–0.5 s vs. NoVAD (consistente com o padding estender a borda
do segmento), mas spans reais preservados. **Existe configuração que recupera a
cauda: VP300. Nenhuma testada reproduz exatamente `26.8→29.x` sem shift.**

---

## 7. Pausas (curta / média / longa, via `silencedetect -25dB`)

Pausas acústicas medidas: curta `11.45→11.79` (0.34 s), média `17.55→18.24`
(0.69 s), longa `13.91→15.09` (1.19 s).

| Config | Curta | Média | Longa |
|---|---|---|---|
| NoVAD | preserva (nenhum→som cruzam) | preserva | preserva (1 DEG interno `ficar 14.40`) |
| VADdef | preserva | preserva | preserva (1 DEG `á 14.84`) |
| VP300 | preserva | preserva | preserva (1 DEG `como 14.13`) |

Nenhuma configuração funde frases através das pausas; gaps preservados nas três.
Pausas **não diferenciam** as configs — o colapso da cauda (item 6) é o fenômeno
distinto, não um "fim definitivo após pausa".

---

## 8. Áudio ruim (VideoMedio2 0–30 s) — todas as configs

| Config | Words | Deg. | Last (cobertura) | Tempo |
|---|---|---|---|---|
| NoVAD | 23 | 0 | 'safe' 28.28→29.99 (30 s) | 2.0 s |
| VADdef | 35 | 27 | 'não' 6.45→6.45 (~6.5 s) | 1.9 s |
| VT0.2 | 39 | 30 | '!' 6.87→6.87 | 2.0 s |
| VT0.8 | 38 | 30 | '.' 5.40→5.40 | 2.0 s |
| VP100 | 38 | 28 | '.' 6.73→6.73 | 2.3 s |
| VP200 | 40 | 29 | '.' 7.13→7.13 | 2.2 s |
| VP300 | 27 | 19 | 'não' 7.53→7.53 | 1.9 s |
| VP500 | 27 | 18 | 'não' 8.33→8.33 | 1.8 s |
| VSD500 | 35 | 27 | idêntico ao default | 1.9 s |
| VSD1000 | 35 | 27 | idêntico ao default | 1.9 s |
| VP200+VSD500/1000 | 40 | 29 | '.' 7.13→7.13 | 2.0 s |

**Nenhuma configuração de VAD recupera o áudio ruim.** Todas truncam em ~5–8 s
com burst degenerado de ~75 %. VP300/VP500 reduzem o burst (19/18 vs. 27) e
estendem ~1 s a cobertura, mas perdem ~22 s de fala real. `-vsd` é inócuo aqui
(resultados byte-idênticos ao default). O texto com VP200/VP300 é fiel no trecho
coberto ("Só pra quem tem coragem…"), mas a truncagem inviabiliza qualquer VAD
neste cenário.

Chunk 2 limpo (30–45 s, controle): NoVAD 55/29, VADdef 56/35, VP300 54/29, textos
equivalentes — VP300 não regressa fora da cauda do chunk 1.

---

## 9. Custo de processamento

Limpo ~2.8–3.1 s, ruim ~1.8–2.3 s em todas as configs (variação = ruído de run;
repetição: NoVAD 3.0 s, VADdef 2.9 s, VP300 2.9 s). **O VAD não acelera de forma
relevante neste setup** (chunk pequeno, modelo medium, GPU livre) — o custo é
dominado pela decodificação.

---

## VAD EXPERIMENT — tabelas consolidadas

### ÁUDIO LIMPO (teste_audio.mkv 0–30 s, ref onset 0.700 s)

```text
Config       Words  Deg.  Onset   Tail           Tempo
No VAD       101    6     -0.030  29.09->29.95   3.0s
Default      96     18    -0.670  25.49 DEG      2.9s
VT 0.2       99     14    -0.670  25.88 DEG      2.9s
VT 0.8       99     16    -0.670  25.36 DEG      2.9s
VP 100       99     17    -0.600  26.82 DEG      3.1s
VP 200       92     9     -0.500  28.43 DEG      2.9s
VP 300       99     10    -0.400  29.61->29.75   3.0s
VP 500       90     7     -0.200  53.55->60 INV  2.9s
VSD 500      95     14    -0.670  25.64 DEG      2.9s
VSD 1000     95     13    -0.670  28.35 DEG      2.9s
VP200+VSD500 99     12    -0.500  27.85 DEG      2.9s
```

### ÁUDIO RUIM (VideoMedio2 0–30 s)

```text
Config       Words  Deg.  Onset  Tail        Tempo
No VAD       23     0     —      29.99 ok    2.0s
Default      35     27    —      6.45 DEG    1.9s
Melhor(VP300)27     19    —      7.53 DEG    1.9s
```

### PAUSAS (teste_audio.mkv)

```text
Config       Curta (0.34s)  Média (0.69s)  Longa (1.19s)
No VAD       preserva       preserva       preserva
Default      preserva       preserva       preserva
VP300        preserva       preserva       preserva
```

---

## CONCLUSÃO

1. **O VAD atual prejudica timestamps?** Sim, comprovado A/B nos dois áudios:
   onset −0.67 s, cauda colapsada (12 words → 1 ponto), e no ruim truncagem em
   ~6.5 s com 75 % de degenerados e texto alucinado.
2. **Qual parâmetro tem maior impacto?** `-vp` (speech-pad-ms), de longe.
   `-vt` ≈ nenhum efeito; `-vsd` ≈ efeito pequeno (deg 18→13, cauda segue DEG).
3. **Existe configuração que melhora onset?** Sim: VP300 −0.670 → −0.400 s
   (VP200 −0.50, VP500 −0.20 mas inválido). Nenhuma alcança o NoVAD (−0.03).
4. **Existe configuração que preserva melhor a cauda?** Sim: **VP300** recupera
   as 12 words com spans válidos (`som 29.61→29.75`, ~+0.5 s de shift vs. NoVAD).
5. **Existe configuração que mantém pausas corretamente?** Todas mantêm;
   pausas nunca foram o problema (nenhuma fusão de frases em nenhum teste).
6. **O áudio ruim muda a conclusão?** Sim, decisivamente: no gameplay **nenhum**
   ajuste de VAD presta — todos truncam ~22 s de fala. Só o NoVAD cobre os 30 s
   (23 words, 0 degenerados).
7. **O custo muda significativamente?** Não (±0.2 s, ruído). VAD não compra
   velocidade relevante aqui.
8. **Recomendação:** nem "VAD default" (comprovadamente danoso) nem "VAD ajustado
   global" (VP300 ajuda o limpo, continua catastrófico no ruim). O dado aponta para
   **sem VAD como default para máxima precisão, VAD opcional por configuração**
   (ex.: VP300 como preset documentado para quem priorizar cauda em áudio limpo,
   com teto ≤300 ms pelo bug de timeline do VP500) — decisão para o passo seguinte
   de integração, fora do escopo deste experimento.

Sinais de alerta registrados para o passo 2: VP500 gera timestamps **fora do
chunk** (53–60 s num áudio de 30 s); qualquer preset com `vp` alto precisa de
validação de sanidade (`start/end` dentro do chunk). Nenhum offset/jitter foi
introduzido, nenhum default foi tocado, nenhum commit foi feito.
