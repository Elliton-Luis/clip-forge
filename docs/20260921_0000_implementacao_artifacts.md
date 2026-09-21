# Implementação: artefatos reutilizáveis + FINISH (2026-09-21)

Transcrição como artefato intermediário independente de scoring, título,
legenda e render. Sem tocar qualidade das legendas, scoring, Whisper,
seleção ou defaults.

## O que foi implementado

- `core/artifacts.py` (novo, única casa da lógica — sem utils genérico):
  envelope `{artifact, artifact_version, source_path, source_fingerprint,
  config, created, data}` para `transcript|title|captions`. Validade = parse
  OK + versão corrente + fingerprint do source + config relevante
  (title: `model`; captions: `caption_mode`+`vertical`). Identidade via
  `cache.fingerprint` (caminho+tamanho+mtime+modelo+idioma+pipev — decisão
  validada, não cache por nome). `transcript_hash` amarra título/legenda ao
  transcript exato; `InvalidArtifact` distingue ausente/ilegível/obsoleto/
  outro-source/outra-config.
- `core/finish.py` (novo, modo FINISH): `ensure_transcript` (reutiliza ou
  transcreve o clip 1×), `make_title` (título via scoring validado, sem
  Whisper), `set_manual_title` (alterar/reutilizar título sem tocar
  transcript), `make_captions` (srt+ass puros), `render_clip` (cut puro,
  recusa saída sobre entrada). Fronteiras externas patcháveis p/ teste.
- `clipper.py`: subcomando `finish CLIP --only all|title|captions|render`
  (+`--title`, `--store-dir`, `--force`, `--force-transcribe`); DISCOVERY
  persiste `work/<video>/artifacts/transcript.json` após o align
  (aditivo, best-effort).
- `tests/test_finish.py`: 12 testes (reuso transcript/título/legenda/render,
  source alterado/corrompido/versão obsoleta regeneram, troca de modelo
  invalida só o título, regenerações preservam transcript byte-idêntico,
  manual preserva transcript, render recusa overwrite).

## Validação real (clip DerrubandoKit)

- `finish --only captions` 1ª vez: transcreveu 1× + gerou; 2ª vez: transcript
  e legendas reutilizados, sem Whisper.
- `--only title`: 1 chamada LLM, sem Whisper. `--only render`: final.mp4
  gerado reutilizando os 3 artefatos, sem Whisper.
- DISCOVERY escreveu `work/DerrubandoKit/artifacts/transcript.json` com o
  mesmo fingerprint do cache.

## Arquivos

`core/artifacts.py`, `core/finish.py`, `clipper.py`, `README.md`,
`tests/test_finish.py`. Suíte: 300 testes, só o erro pré-existente de fixture.
