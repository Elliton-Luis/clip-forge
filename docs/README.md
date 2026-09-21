# Documentation

Mapa da documentação do Clipper. Convenção de nomes:
`YYYY-MM-DD-HHMM-slug.md` (data real do documento, nunca inventada).
PDFs ao lado dos `.md` são espelhos do mesmo conteúdo, não documentos
separados. Um arquivo sem prefixo de data (`analise-legendas.md`) tem
data de origem desconhecida.

## Architecture

Como o sistema funciona (fonte da verdade, sem código).

- `architecture/2026-09-19-1015-como-funcionam-as-legendas.md` (+`.pdf`) — pipeline das legendas, do Whisper ao ASS/SRT.
- `architecture/2026-09-18-1133-project-documentation.pdf` — referência técnica completa (instalação, CLI, arquitetura, GPU, API, métricas).

## Experiments

O que foi testado e o que foi observado (não só a conclusão).

- `experiments/2026-09-19-0105-caption-timing.md` — problema de timing das legendas, experimentos e conclusões.
- `experiments/2026-09-19-0813-vad-experiment.md` — VAD apaga fala em gameplay: evidência que tirou o VAD do default.
- `experiments/2026-09-20-2050-alignment-investigation.md` — investigação de forced alignment (decisão e justificativa).
- `experiments/2026-09-20-2109-alignment-experiment.md` — Whisper timestamps vs forced alignment, medido.
- `experiments/2026-09-20-2213-peak-experiment.md` — seleção classic vs peak, com números.

## Audits

Estado do código em determinado momento (leitura, sem implementação).

- `audits/2026-09-18-auditoria-pre-execucao.pdf` — relatório técnico + auditoria pré-execução (era `AUDIT-clipper.pdf` na raiz).
- `audits/2026-09-19-1006-caption-audit.md` — auditoria ponta a ponta das legendas.
- `audits/2026-09-19-1336-auditoria-clipper.md` (+`.pdf`) — raio-X ponta a ponta do pipeline.
- `audits/2026-09-19-1349-timing-root-cause.md` (+`.pdf`) — onde nasce o erro temporal.
- `audits/2026-09-20-1054-auditoria-desempenho.md` — gargalos de performance do pipeline.
- `audits/2026-09-20-1232-auditoria-audio.md` — por que a etapa `audio` domina o pipeline.
- `audits/2026-09-21-0034-auditoria-tui.md` — diagnóstico e proposta de redesign do TUI.
- `audits/analise-legendas.md` — análise de timing/overlaps (data de origem desconhecida).

## Implementation

O que foi planejado/implementado em cada etapa (não confundir com auditoria).

- `implementation/2026-09-20-1355-relatorio-audio.md` — otimização da energia de áudio sem decode de vídeo.
- `implementation/2026-09-20-2335-implementacao-alignment.md` — forced alignment opt-in.
- `implementation/2026-09-20-2335-implementacao-selection.md` — seleção pelo auge do conteúdo.
- `implementation/2026-09-20-2335-implementacao-review.md` — revisão humana pré-burn-in.
- `implementation/2026-09-20-2350-implementacao-captions.md` — legendas por frases (conteúdo × apresentação).
- `implementation/2026-09-21-0000-implementacao-artifacts.md` — artefatos reutilizáveis + FINISH.
- `implementation/2026-09-21-0012-implementacao-finish.md` — modo FINISH.
- `implementation/2026-09-21-0021-implementacao-finish-review.md` — revisão pré-burn-in do FINISH.
- `implementation/2026-09-21-0028-implementacao-consolidate.md` — FINISH incremental (lote, status, limpeza).
- `implementation/2026-09-21-0121-implementacao-tui.md` — revisão final da TUI.

## Decisions / Guides

Ainda não há ADRs (`decisions/`) nem guias separados (`guides/`) — o uso
operacional vive no `README.md` da raiz. Quando a primeira decisão
arquitetural relevante for registrada, usar `decisions/ADR-NNN-slug.md`.
