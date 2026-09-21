# Implementação: MODO FINISH (2026-09-21)

Clip já escolhido → título/legenda/render sem descoberta e sem Whisper
desnecessário. Lógica de descoberta intocada.

## O que foi implementado

- `core/finish.py`: `_title_llm` próprio (LLM direto só para nomear o momento;
  scoring de descoberta NÃO é chamado) + `run_finish()` orquestrando
  `all|title|captions|render` com regeneração granular (`--regenerate-title`,
  `--regenerate-captions`, `--force` = ambos) e `--review` (A/E/S/Q pré-burn-in;
  edição regenera só as legendas, `transcript.json` intacto).
- `clipper.py finish`: reescrito sobre `run_finish()` (+`--review`,
  regenerações granulares; `RuntimeError` vira `sys.exit` claro).
- Grafo testado (`tests/test_finish.py`, classe `TestDependencyGraph`,
  descoberta explodindo de propósito): title→transcript, captions→transcript,
  render→transcript(+title/captions); title ↛ captions, captions ↛ title,
  nada chama candidatos/scoring/NMS/seleção; render sem título exige
  `--only title`, `--title` ou `--no-title`; regenerar título não toca captions.

## Validação real

- `--only title --regenerate-title` em clip real: 1 LLM, sem Whisper
  (com retry + aviso de grounding, política igual ao scoring).
- Turno anterior: captions reutilizados na 2ª execução; render final
  reutilizando os 3 artefatos.

## Arquivos

`core/finish.py`, `clipper.py`, `README.md`, `tests/test_finish.py`,
`docs/20260921_0010_implementacao_finish.md`.
Suíte: 305 testes, só o erro pré-existente de fixture.
