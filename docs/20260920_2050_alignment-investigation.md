# Forced alignment — investigação técnica (2026-09-20)

Decisão e justificativa. Nada aqui toca o pipeline produtivo.

## Premissas do projeto (não-negociáveis)

- Whisper continua responsável por **"o que foi dito"** (texto nunca muda no aligner).
- Aligner responde só **"quando exatamente"** (start/end por palavra).
- Sem VAD no caminho de transcrição: `docs/20260919_0813_vad-experiment.md` provou que VAD
  apaga ~24 s de fala em gameplay, adianta onset (−0,67 s) e colapsa caudas.
  Qualquer opção que **exija** VAD está descartada como default.
- Execução local, português, custo razoável, Fedora, Python 3.14, B580 (Vulkan/QSV).
- Pipeline produtivo inalterado por padrão: alignment é **opt-in** (`--align` /
  `CLIPPER_ALIGN`, default `off`), com fallback controlado.

## Opções comparadas

| Opção | Precisão esperada (word) | Deps / modelos | VRAM–RAM | Py 3.14 / Fedora | B580 | Tempo | Manutenção | Integração |
|---|---|---|---|---|---|---|---|---|
| **WhisperX** (faster-whisper + wav2vec2 PT `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`) | Alta: ~93% precisão / ~65% recall c/ collar 200 ms em telefone (paper Interspeech 2023); bem acima do Whisper puro (~85%/63%) | Pesadas: `torch`, `torchaudio`, `transformers`, `pyannote` (VAD/diarização), modelo PT ~1,2 GB | GPU ~4–8 GB; CPU ~4 GB RAM | OK pip, mas `pyannote` exige token HF; VAD embutido no fluxo padrão | Não nativa (CUDA-first; Intel XPU via IPEX é experimental) | RTF ~0,1 GPU / ~0,5 CPU (só o align) | Média-baixa: repo com pouco movimento recente, API instável, traz VAD+diarização que não queremos | Média: reescreve o fluxo de transcrição; conflita com whisper.cpp/Vulkan atual |
| **stable-ts** (`model.align` / `align_words` sobre Whisper ou faster-whisper) | Média: melhor que Whisper puro (re-decode + supressão de silêncio + regroup), sem nível fonema | Médias: `torch` (backend whisper) ou faster-whisper; sem modelo extra | Semelhante ao Whisper usado | OK pip | Indireta (só via backend escolhido) | Mais lento que transcrever (refine iterativo) | Média: projeto pequeno, releases frequentes mas API mutável | Boa p/ SRT, mas o default usa **supressão por VAD/silêncio** — mesma família de intervenção que descartamos como correção silenciosa |
| **MFA** (Montreal Forced Aligner, HMM + dicionário PT) | Alta (fonema), se dicionário cobrir o vocabulário | Pesadas: Kaldi, dicionário PT + modelo acústico, instalação fora do pip | CPU-bound, RAM alta | Ruim no Fedora (compilação, pacotes) | Não usa | Lento (treino/adaptação) | Alta: manter dicionário de gameplay/gírias é inviável | Baixa: outro ecossistema |
| **NeMo** (conformer-CTC PT, forced alignment) | Alta | Pesadas: `nemo_toolkit`, `torch` CUDA-only p/ GPU | GPU NVIDIA exigida p/ tempo razoável | OK pip, pesado | Não suportada | Rápido só em NVIDIA | Alta | Baixa |
| **wav2vec2-PT direto** (HF transformers + CTC + trellis/backtrack próprios, sem WhisperX) | Alta (mesma do WhisperX, sem o VAD/diarização) | Médias-altas: `torch` + `transformers` + modelo PT ~1,2 GB (download único, HF) | CPU ~2–4 GB RAM; GPU se houver CUDA | OK pip | CPU (IPEX opcional, não exigido) | RTF ~0,3–0,6 CPU | Média: ~150 linhas nossas de DP, sem dono externo | Boa: recebe (áudio + texto do Whisper) e devolve só timestamps |
| **whisper-refine** (re-decode do próprio Whisper em janela curta + casamento de sequência) | Baixa–média: reduz drift de chunk longo; não chega a nível fonema | **Zero novas**: reusa `whisper.cpp`/Vulkan/B580 ou faster-whisper CPU já instalados | Igual à transcrição atual | Total | **Sim, nativa** (mesmo binário/modelo ggml) | Barato: 1 decode curto por segmento (~segundos na B580) | Baixa: nosso código, sem dono externo | Total: entra depois do transcript, antes dos candidatos, sem tocar o resto |

## Decisão

Shippar dois backends atrás de uma interface única (`core/alignment.py`):

1. **`whisper-refine` (padrão do opt-in)** — escolhido para este projeto porque:
   sem dependência nova, sem VAD, sem modelo novo, usa a B580, custo mínimo,
   manutenção mínima, integração total. Limitação honesta: ganho moderado
   (re-ancoragem de drift, não precisão de fonema) — por isso é experimental
   e mede-se tudo no `align-compare`.
2. **`wav2vec2` (opcional avançado)** — para quem aceitar `pip install torch
   transformers` + download do modelo PT: precisão de fonema de verdade via
   CTC/trellis/backtrack, sem VAD, CPU-friendly. Lazy-import: sem os pacotes,
   falha controlada com fallback ao Whisper (nunca corrompe, nunca fabrica).

Descartados como etapa do clipper: WhisperX (traz VAD+diarização obrigatórios
no fluxo e repo instável), stable-ts (supressão por silêncio/VAD por padrão),
MFA e NeMo (instalação/manutenção/GPU incompatíveis com B580/Fedora).

## Contrato da etapa (vale p/ ambos os backends)

- Texto imutável: `align_segments` nunca altera `Word.text`/`Segment.text`.
- Origem explícita: `Word.timestamp_source` / `Segment.timestamp_source` ∈
  `{"whisper", "forced_alignment"}`; `Word.confidence` opcional (0..1).
- Fallback: exceção, mismatch de sequência, modelo ausente → mantém timestamps
  do Whisper, `timestamp_source="whisper"`, erro registrado em `AlignmentStats`
  e no log. Nenhum timestamp fabricado silenciosamente.
- Monotonicidade: `start < end`, ordem preservada, clamp na janela do segmento.
- Fora do caminho padrão: `--align off` (default) = pipeline byte-idêntico ao atual.
