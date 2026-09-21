# Experimento: Whisper timestamps vs Forced Alignment (2026-09-20)

Medido, não declarado. Artefatos em `debug/alignment-lab/*/`
(`whisper.json`, `aligned.json`, `compare.json`, `compare.txt`).
Comando: `python clipper.py align-compare VIDEO --start S --dur D --backend B`.

## Material

- `videos/GuilhermeCaindo.mp4` [0,30]: gameplay ruidoso, 1 segmento / 31 words
  (inclui pilha degenerada em 18.87 s — o padrão `end <= start` já conhecido).
- `videos/DerrubandoKit.mp4` [0,30]: diálogo real de gameplay, 1 segmento /
  31 words ("Tem algum item de energia? ... Nenhum pirulito, nada? ... Me dá."),
  2 pessoas, perguntas, pausas. Sem loops de repetição.
- Modelo Whisper: `ggml-medium.bin` via whisper.cpp/Vulkan (B580).
- Alinhador CTC: `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`
  (315 M params, CPU, ~17–23 s por janela de 30 s).

## Resultados

| Exp | Vídeo/trecho | Backend | Aligned / fallback | mean\|Δstart\| | mean\|Δend\| | max\|Δ\| |
|---|---|---|---|---|---|---|
| 1 | GuilhermeCaindo [0,30] | whisper-refine | 0 / 31 | 0,000 s | 0,000 s | 0,000 s |
| 2 | GuilhermeCaindo [0,30] | wav2vec2 | 0 / 31 | 0,000 s | 0,000 s | 0,000 s |
| 3 | DerrubandoKit [0,30] | whisper-refine | 0 / 31 | 0,000 s | 0,000 s | 0,000 s |
| 4 | DerrubandoKit [0,30] | wav2vec2 | 0 / 31 | 0,000 s | 0,000 s | 0,000 s |

`text_identical=True` nos 4 (texto nunca muda). Nenhum timestamp fabricado:
tudo que o aligner não mediu com confiança ficou `timestamp_source="whisper"`.

## Leitura honesta (o que os números dizem)

1. **whisper-refine é idempotente (Δ=0).** Re-decodificar com o mesmo modelo
   reproduz os mesmos timestamps. Não é forced alignment de verdade — serve
   como teste de sanidade do casamento de sequência, não como corretor.
   Mantido como backend porque custa segundos na B580 e não adiciona
   dependência, mas o relatório diz o que ele é.
2. **wav2vec2-PT não alcança confiança utilizável em gameplay ruidoso.**
   Confianças típicas ~1e-6; gate (`conf >= 0.3`, duração >= 40 ms) rejeitou
   100% e o fallback preservou o Whisper. Sem o gate (rodada preliminar), o
   CTC emitia spans colapsados de 10 ms — isso seria *pior* que o Whisper e
   foi corretamente barrado. O gate é a feature, não o bug.
3. **Controle positivo passa.** Com emissões sintéticas que casam o texto, o
   trellis/backtrack/merge recupera os spans em ordem e com score > 0.5
   (`tests/test_alignment.py::TestCtcMachinery`). O limite é modelo/dado
   (acústico treinado em fala limpa × gameplay com risada/gritos/sobreposição),
   não o código do DP.
4. **Evidência independente de drift real do Whisper.** Decodificação greedy
   do wav2vec2 (sem Whisper) no trecho [0,5] de DerrubandoKit recupera
   "algum item de energia" verbatim — frase que o Whisper carimba a partir de
   5.77 s. Ou seja: o conteúdo está ~3 s antes do timestamp. O problema
   temporal existe e é de segundos, mas este modelo acústico não consegue
   fixá-lo com confiança — por isso nada foi "melhorado à força" no transcript.
5. **Pilha degenerada confirmada por energia.** Words em 18.87 s com duração
   zero têm RMS 0.005 (silêncio) — corrobora o diagnóstico
   `SILÊNCIO?` do `--debug-captions`, por via independente.

## Conclusão prática

- A etapa existe, é opt-in (`--align`, default `off`), não toca o pipeline
  padrão, e se comporta como especificado: mede quando pode, preserva quando
  não pode, registra tudo (`AlignmentStats`, `timestamp_source`, `confidence`).
- Benefício medido **neste material**: nenhum — e o relatório prefere esse
  zero honesto a deltas fabricados. Forced alignment vira ganho real quando
  houver modelo acústico robusto a gameplay (ou fala limpa na entrada); a
  interface já aceita esse backend sem mudar o resto.
- Custo medido: wav2vec2-PT 315 M, ~17–23 s CPU por 30 s de áudio (RTF ~0.7),
  +~1,2 GB de download único; sem `torch`/`transformers`, fallback imediato
  com erro explícito.
