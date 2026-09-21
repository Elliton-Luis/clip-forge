# Implementação: seleção pelo auge (2026-09-20, commit 6c7f6d6)

Commit: `feat(selection): rank clips by content peak`.
Etapa 2 do trabalho (sobre o forced alignment da etapa 1, sem tocá-lo):
clips tecnicamente bons mas sem o melhor momento → procurar o auge e
construir o clip ao redor dele.

## Problema de origem

Janelas de até 90 s ranqueadas por score médio entregavam 20–30 s bons +
30–60 s de "enchimento", ou a janela englobando o momento em vez de
centrar nele. Maximizar duração/fala/energia/score-médio não maximiza
"o momento que mais vale mostrar".

## O que foi implementado

- `core/peaks.py` (novo, opt-in `--selection-mode peak`, default `classic`
  intocado): cada candidato ganha `window_start/window_end` (janela original)
  e `peak_start/peak_end/peak_score/peak_source/peak_reason`.
- Peak por LLM em subjanelas de ~12 s (1 chamada/candidato, mesmo retry/pacing
  do scoring, prompt pede intensidade + motivo + `needs_context`) ou heurística
  sem rede (ordem documentada: risada > evento acústico > exclamação >
  fala rápida; `peak_source` registra o caminho).
- `build_clip_around_peak`: expande do peak até frases inteiras + mínimo,
  nunca estofa até o máximo, respeita min/max e `media_end`, snap em palavra;
  peak maior que max corta contexto, nunca o peak.
- Rank por intensidade do auge (score da janela vira porta de qualidade);
  diversidade por overlap de peaks (≥ 50% = mesmo momento, descarta) + teto
  por 10 min. Títulos do auge só nos selecionados, grounding validado,
  `title_source` explícito; falha mantém o título da janela.
- `core/peaklab.py` + `python clipper.py peak-compare`: classic vs peak nos
  mesmos candidatos (duração, peak, score, motivo) em `debug/peak-lab/`.
- Integração: flag `--selection-mode` / `CLIPPER_SELECTION_MODE`, bloco peak
  no manifest só no modo peak (classic byte-estável), cache com
  retrocompatibilidade, timing em `metrics.stages["peaks"]`.
- `tests/test_peaks.py`: 21 testes (subjanelas, heurística, construção do clip,
  overlap, select_peak, parse defensivo de títulos, controle de id inválido).

## Experimento real (docs/experiments/2026-09-20-2213-peak-experiment.md)

- VideoMedio (7 min): classic [90s, 90s] → peak [25s, 32s] nos mesmos momentos.
- VideoLongo1 (26 min): classic 5×90s+57s → peak 6× ~21–26s, pscores 8.5–10/llm,
  6/6 títulos do auge validados, peaks distintos sem duplicatas; janelas de
  score 6.5 ranquearam pelo auge (comportamento pretendido).
- Diversidade provada em teste (3 janelas do mesmo peak → 1 sobrevivente).
- End-to-end `--selection-mode peak` renderizou com bloco peak no manifest.
- Limites: 1 LLM/candidato (experimental por custo); re-título caiu com 503
  numa rodada (janela mantida, explícito); teto por 10 min continua valendo.

## Arquivos

`core/peaks.py`, `core/peaklab.py`, `core/models.py` (campos peak),
`core/cache.py`, `core/config.py`, `clipper.py`, `README.md`,
`tests/test_peaks.py`, `tests/test_tui.py`, `docs/experiments/2026-09-20-2213-peak-experiment.md`.
Legendas, forced alignment e dashboard: intocados.
