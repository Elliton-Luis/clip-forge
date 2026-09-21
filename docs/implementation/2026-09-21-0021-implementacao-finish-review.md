# Implementação: revisão pré-burn-in do FINISH (2026-09-21)

Clip escolhido → artefatos → revisão → render. Descoberta intocada.

## O que foi implementado

- `core/finishreview.py` (novo): loop A/E/R/T/L/S/Q sobre
  `transcript/title/captions.json`, vídeo sem nada queimado. `E` edita texto
  e timestamps (registrado em `review.json`; `transcript.json` nunca muda em
  silêncio); `R` pergunta título/legenda e regenera só um, sem Whisper;
  `T`/`L` desligam título/legenda só do render (artefatos preservados);
  `S` pula sem erro; `Q`/Ctrl+C/EOF persiste e reabre continuando.
  Reutiliza helpers puros de clipreview; estado em `review.json` (artefato
  versionado no store).
- `core/finish.py`: `run_finish(..., review, review_input_fn)` — preview
  próprio (`preview_finish.mp4`, sem colidir com o da DISCOVERY), edição
  regenera só legendas, render só após aceite, `_validate_output` (existe,
  bytes, duração) antes de qualquer limpeza (que nunca é automática).
  Corrigidos no caminho: duração de display via probe e hash canônico
  int/float em `transcript_hash`.
- `tests/test_finishreview.py`: 15 testes (aceitar, regen título/legenda,
  skip título/legenda, skip clip, quit, edição marca estado, reabrir
  continua, não-interativo, render-só-após-aceite, quit-não-renderiza,
  edição-regenera-legenda-mantém-transcript, falha-de-render preserva tudo).

## Validação real

`finish --review` via pty em clip real: tela exibida, preview ffmpeg sem
legenda, `a` → render validado, EXIT 0.

## Arquivos

`core/finishreview.py`, `core/finish.py`, `core/artifacts.py`,
`tests/test_finishreview.py`, `README.md`,
`docs/20260921_0020_implementacao_finish_review.md`.
Suíte: 320 testes, só o erro pré-existente de fixture.
