# Implementação: forced alignment (2026-09-20, commit 5cf64f9)

Commit: `feat(alignment): add forced word alignment`.
Etapa 1 do trabalho: trocar a fonte dos timestamps sem trocar a transcrição.

## Problema de origem

Whisper acertava "o que foi dito" mas errava "quando" (jitter de ~0,5 s,
drift de segundos, pilhas degeneradas). Auditoria provou que nada depois do
Whisper deslocava timestamps — o erro nascia no modelo. Decisão: não
"consertar" com offsets/médias/VAD/DTW arbitrário; separar responsabilidades
com uma etapa de forced alignment opt-in.

## O que foi implementado

- `core/alignment.py` (novo): `align_segments()` recebe (áudio + segmentos do
  Whisper) e devolve mesmos textos com starts/ends re-medidos quando possível.
  Backends: `off` (default, pipeline byte-idêntico), `whisper-refine`
  (re-decode do próprio Whisper em janela curta + casamento de sequência,
  zero deps, B580) e `wav2vec2` (CTC real com
  `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`, 315 M, lazy-import
  torch/transformers).
- `core/models.py`: `Word`/`Segment` ganharam `confidence` e
  `timestamp_source` (`whisper`/`forced_alignment`), com defaults que mantêm
  todo código antigo válido.
- Contrato rígido: texto imutável; fallback preserva o Whisper e registra o
  erro (`AlignmentStats`); gate de qualidade CTC (conf ≥ 0.3, duração ≥ 40 ms)
  barra spans colapsados; violação de monotonicidade reverte ao Whisper em vez
  de clampar; sem torchinstalado, `wav2vec2` cai para Whisper com erro explícito.
- `core/alignlab.py` + `python clipper.py align-compare`: experimento
  palavra-por-palavra (Δinício/Δfim, RMS por span, origem) em
  `debug/alignment-lab/`.
- Integração: flag `--align` / `CLIPPER_ALIGN`, rodada após transcrição e
  revisão, antes dos candidatos; cache e review com retrocompatibilidade;
  `requirements.txt` lista as deps pesadas como opcionais comentadas.
- `tests/test_alignment.py`: 16 testes (contrato, fallback, monotonicidade,
  cache roundtrip, CLI, controle positivo do DP CTC com emissões sintéticas).

## Experimento real (docs/experiments/2026-09-20-2109-alignment-experiment.md)

4 rodadas (GuilhermeCaindo e DerrubandoKit, backends refine e wav2vec2):
`text_identical=True` sempre; `whisper-refine` idempotente (Δ=0 — re-decode com
o mesmo modelo não é alignment); `wav2vec2` com confiança ~1e-6 em gameplay
ruidoso → gate barrou 100%, zero corrupção. Controle positivo do DP passa.
Achados laterais: greedy independente recupera "algum item de energia" em
[0,5]s onde o Whisper carimba 5,77s+ (drift real de ~3s); pilha degenerada em
18,87s tem RMS 0,005 (silêncio). Conclusão honesta: benefício medido zero
neste material — preferido a deltas fabricados.

## Arquivos

`core/alignment.py`, `core/alignlab.py`, `core/models.py`, `core/cache.py`,
`core/review.py`, `core/config.py`, `clipper.py`, `requirements.txt`,
`README.md`, `tests/test_alignment.py`, `tests/test_tui.py`,
`docs/experiments/2026-09-20-2050-alignment-investigation.md`, `docs/experiments/2026-09-20-2109-alignment-experiment.md`.
Scoring, seleção, duração, títulos e renderização: intocados.
