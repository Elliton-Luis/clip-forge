# Implementação: revisão humana pré-burn-in (2026-09-20)

Etapa 3 do trabalho (sobre alignment e peak-selection, sem tocar nenhum dos
dois): o scoring/seleção escolhem, o humano aprova só os sobreviventes antes
de queimar legenda.

## Fluxo

transcrição → forced alignment → scoring → seleção → **REVISÃO HUMANA** →
render final. Flag `--review-clips` (TUI tem checkbox); default off, pipeline
sem a flag é byte-idêntico.

## O que foi implementado

- `core/clipreview.py` (novo): `build_previews()` renderiza
  `work/<stem>/clipreview/preview_NN.mp4` via `cut(captions=False, title=False)`
  — mesmo corte/áudio do final, só vídeo+áudio, sem nada queimado;
  `run()` faz o loop `[A]ceitar [E]ditar [S]pular [P]tocar [Q]sair` com
  título, score, duração, energia, transcript em linhas `[MM:SS.mmm] frase`.
- `A` mantém transcrição+alignment e vai ao burn-in; `E` despeja
  `clip_NN_words.txt` (`[sNN] [MM:SS.mmm → MM:SS.mmm] texto`, absolutos),
  abre `$EDITOR` (ou orienta edição manual) e aplica com a validação do
  `review.parse_words_txt` — linha malformada é erro claro, nada muda; a
  legenda final nasce do texto corrigido (`c.words`, que o `cut` já consome).
- `S` descarta sem erro e sem render; `Q`/Ctrl+C persiste `decisions.json`
  (accepted/edited/skipped/pending) e o rerun continua nos pendentes;
  candidatos diferentes arquivam em `decisions.bak.json` e recomeçam.
- Sem tty (pipe/CI) falha com mensagem clara em vez de aprovar no escuro.
- Integração em `clipper.py`: após seleção (+revisão de títulos), antes do
  corte; manifest ganha `"review": accepted|edited` só no modo (classic
  estável); `metrics.stages["review"]`; previews limpos por `finalize/reset`,
  nunca deleção automática de originais.
- `tests/test_clipreview.py`: 15 testes (aceitar, pular, sair+persistência,
  continuar, mismatch→bak, Ctrl+C, não-interativo, editar texto/timestamp,
  remover palavra, linha inválida, vazio, dump/apply roundtrip).

## Validação real

Pipeline de verdade via pty (`DerrubandoKit`, 1 clip): preview sem stream de
legenda (só vídeo+áudio), UI exibida, `a` aceito, clip final com legenda
queimada, `SUCCESS`, `decisions.json` com accepted, manifest com review.
End-to-end anterior do peak (`--selection-mode peak`) continuou passando.

## Arquivos

`core/clipreview.py`, `clipper.py`, `core/tui.py`, `tests/test_clipreview.py`,
`tests/test_tui.py`, `README.md`. Legendas, alignment, peak e dashboard:
intocados (sem dashboard criado).
