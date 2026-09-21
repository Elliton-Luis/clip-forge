# Implementação: revisão final da TUI (2026-09-21)

Interface como ferramenta: estado visível em 2 segundos, sem log de debug na
frente do usuário. Domínio intocado (scoring, seleção, timestamps,
transcrição, render, modelos, limites).

## O que foi implementado

- `core/progress.py` (novo, stdlib): bus `stage/adv/done/skip/warn/note/detail`
  + dashboard ANSI reescrito no lugar (TTY) + modo linhas (pipe/script/teste)
  + `--quiet` + `error_box()` + `review_banner()`. ETA só com total, progresso
  e >5s de medição; sem total, mostra contagem sem barra (nunca fabrica %).
- Rosqueamento (só destino dos prints): transcribe (barra por chunks),
  backends (retry→avisos), audio (barra por candidato), scoring (barra por
  lote, `[perf]`→verbose/log), candidates/selection/peaks (conclusões de
  1 linha), preflight (vira header + warns), finish (modo FINALIZAR CLIP),
  alignment (mesmo padrão). Labs e testes sem bus têm saída byte-idêntica.
- Removido do fluxo normal: `Transcription backend:` duplicado, `[N/5]`,
  `[perf]` de 200 chars, detalhes por chunk/tentativa/backoff (em `--verbose`
 /`--log-file`). Backend/GPU/encoder fundidos em header de 2 linhas.
- CLI `--verbose/--quiet/--log-file` (CONFIG_FIELDS, cli_command, TUI);
  formulário TUI em seções (VÍDEO E SAÍDA · CONTEÚDO · FORMATO · REVISÃO
  HUMANA · AVANÇADO); revisões com banner uniforme; erro fatal vira
  `✗` + causa + dica (mensagens originais preservadas).
- `tests/test_progress.py`: 12 testes (estágios, regra do ETA, renderers,
  quiet, pause, error_box, banner, log-file).

## Evidência

- `DerrubandoKit.mp4` real: pipeline completo, clipe + manifest gerados.
- Caso de falha real: `✗ Interrompido em candidates` + causa + dica.
- Dashboard ANSI validado em pty (bloco, barra, aviso, reescrita).
- Suíte: 343 testes, 342 passam (erro pré-existente de fixture,
  `metrics_test/whisper_raw_words.json` ausente, falha no main também).

## Arquivos

`core/progress.py`, `clipper.py`, `core/transcribe.py`, `core/backends.py`,
`core/audio.py`, `core/scoring.py`, `core/candidates.py`, `core/selection.py`,
`core/peaks.py`, `core/preflight.py`, `core/finish.py`, `core/alignment.py`,
`core/cache.py`, `core/tui.py`, `README.md`, `tests/test_progress.py`,
`tests/test_tui.py`, `tests/test_consolidate.py`,
`docs/implementation/2026-09-21-0121-implementacao-tui.md`.
Sem dependências novas. Base: `docs/audits/2026-09-21-0034-auditoria-tui.md`.
