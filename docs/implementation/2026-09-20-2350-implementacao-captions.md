# Implementação: legendas por frases (2026-09-20)

Etapa 4 do trabalho: trocar grupos arbitrários de palavras por unidades
naturais de fala, separando conteúdo de apresentação. Sem tocar scoring,
seleção, alignment ou revisão.

## Problema de origem

Uma frase saía quebrada em vários balões ("É para a direita" / "você vai se
apoiar"), leitura ruim e vídeo artificial. As regras antigas (gap 0,4 s,
8 words, 48 chars, rajada de 10 s) cortavam por tamanho/tempo, não por sentido.

## O que foi implementado

- `core/phrases.py` (novo, camada de CONTEÚDO): `group_phrases()` devolve
  `Phrase(words, text, start, end, voice, bridged_gap, has_ellipsis,
  word_spans)` — timestamps individuais intactos para destaque futuro.
  Regras: fronteira `.!?` quebra, exceto continuação próxima (gap ≤ 3 s
  configurável + minúscula/conjunção/"…"); sem pontuação + gap ≤ 3 s continua;
  pausa interna real (≥ 0,8 s) vira "…" (só ela); gap > 3 s sempre quebra;
  vozes (overlap > 0,1 s) nunca se misturam; palavra nunca parte.
- `core/video.py` (APRESENTAÇÃO, visual inalterado): modo `phrases` em
  `_group_cue_words`/`_group_cues` (texto da frase, sem re-join); `_split_lines`
  ganha `subdivide_time=False` no modo phrases — uma frase = uma cue, texto só
  quebra em linhas visuais (fim da redistribuição uniforme de tempo).
- `--caption-mode phrases` virou o padrão (CLI/TUI/lab); `words`/`intervals`
  seguem disponíveis. `caption-lab` compara os 3 modos.
- Proibido e cumprido: sem esticar, sem redistribuir, sem inventar timestamp,
  sem duração máxima ou nº de caracteres como critério, sem juntar vozes.
- `tests/test_phrases.py`: 13 testes cobrindo os 10 casos do spec (frase
  curta/longa, pausa curta/longa/interna, 2 pessoas, overlap, pontuação,
  continuação após quebra, destaque por palavra) + tolerância configurável,
  spans reais e palavra nunca partida.

## Evidência medida (caption-lab, vídeos reais)

- DerrubandoKit 0–30 (diálogo limpo): phrases 7 cues/99,9% idênticas às words
  (zero regressão); intervals 5/100%.
- GuilhermeCaindo 0–19: phrases 4/99,3% idênticas às words.
- Vantagem aparece em fala fragmentada (testes): "Vamos fazer isso… ou não"
  numa cue em vez de dois balões.
- Render real (`DerrubandoKit`, padrão phrases): clip com ASS queimado, SUCCESS.

## Arquivos

`core/phrases.py`, `core/video.py`, `core/captionlab.py`, `core/tui.py`,
`clipper.py`, `README.md`, `tests/test_phrases.py`, `tests/test_tui.py`.
Suíte: 288 testes, só o erro pré-existente de fixture.
