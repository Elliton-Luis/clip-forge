# Implementação: FINISH incremental (lote, status, limpeza) (2026-09-21)

Consolidação: nunca refazer trabalho caro. Seleção/scoring/Whisper intocados.

## O que foi implementado

- `core/artifacts.py`: `DEPENDS_ON` documentado + `artifact_state()` (estado
  sem gerar nada) + `final_state()` (RENDERED/STALE/MISSING por mtime).
  Invalidação: título→só title; legenda→só captions; transcript→derivados;
  source→tudo. Irmãos nunca se invalidam.
- `core/finish.py`: captions sem hook de título no artefato (hook aplica no
  render via `cut`, que já derivava de `c.title` — zero mudança visual);
  `--no-keep-artifacts` remove title/captions/review/preview SÓ após render
  validado (`transcript.json` sempre fica); `clip_status()` (sem
  Whisper/LLM/ffmpeg); `batch_finish()` (continua após falha individual).
- `clipper.py`: `finish-batch DIR` + `finish-status` (arquivos e/ou pastas).
- `tests/test_consolidate.py`: 11 testes (invalidação isolada nos 4 eixos,
  NEEDS/READY/RENDERED/STALE, status sem efeitos colaterais, lote reutiliza
  tudo na 2ª passada, falha isolada não para o lote, cleanup padrão vs
  `--no-keep`).

## Auditoria de performance (medido)

- Caro: Whisper (B580 RTF ~0,08; CPU ~2,25) — roda 1×/source; scoring LLM
  (~1 min/lote, paralelo, cacheado); título LLM (~7 s, 1 chamada);
  peak-judge (1/candidato, opt-in); render (1 encode/final).
- Barato: legendas (puro, ms), probes/validade (ffprobe/stat), status (só IO).
- Sem duplicatas: transcript 1× (ensure), título 1 LLM (regen explícita),
  captions 1 build (regen explícita), preview 1 (skip-if-exists), render 1.
- Sem Whisper em título/legenda/render (testado); sem LLM com artefato válido
  (testado); ffmpeg antes da hora: só volumedetect (entrada da seleção),
  preview (revisão) e janela do align — todos necessários.

## Validação real

- `finish-batch` em 2 clips: 1ª passada 2 transcrições; 2ª passada 0 Whisper,
  tudo reutilizado. `finish-status`: NEEDS-TITLE correto.

## Arquivos

`core/artifacts.py`, `core/finish.py`, `clipper.py`, `README.md`,
`tests/test_consolidate.py`, `docs/20260921_0035_implementacao_consolidate.md`.
Suíte: 331 testes, só o erro pré-existente de fixture.

## Limitações restantes

- `finish-status` não sabe a intenção (sempre cobra title+captions no READY).
- Hook/destaque de título vivem no render, não no `captions.json` exportado.
- Sem migração: artefatos antigos sem envelope são regenerados (barato e raro).
