"""artifacts.py — camada de artefatos reutilizáveis por source.

Conceito: `transcript.json` é a fonte de verdade (texto, segmentos, palavras,
timestamps, metadados); `title.json` e `captions.json` derivam dele e carregam
a identidade do que os gerou. Cada artefato é um JSON com envelope:

    {artifact, artifact_version, source_path, source_fingerprint,
     config, created, data}

Validade = parse OK + kind/versão correntes + fingerprint do source inalterado
+ chaves de config relevantes iguais. Ausente ou inválido → o chamador gera
SOMENTE o que falta (nunca tudo de novo, nunca Whisper à toa).

Identidade do source: `cache.fingerprint` (caminho resolvido + tamanho +
mtime + modelo + idioma + versão do pipeline) — decisão já validada no
projeto, não cache ingênuo por nome de arquivo. Sem frameworks externos e
sem utils genérico: este módulo é a única casa dessa lógica.
"""
import hashlib
import json
import re
import time
from pathlib import Path

ARTIFACT_VERSIONS = {"transcript": 1, "title": 1, "captions": 1, "review": 1}
# Config que invalida cada artefato quando muda (além do fingerprint).
ARTIFACT_CONFIG_KEYS = {
    "transcript": (),
    "title": ("model",),
    "captions": ("caption_mode", "vertical"),
    "review": (),
}


class InvalidArtifact(Exception):
    """Artefato ausente ou inválido — motivo em str(e). Gerar só o que falta."""


# Grafo de invalidação (irmãos não se invalidam; tudo deriva do transcript,
# que deriva do source):
#   source mudou → transcript, title, captions, review
#   transcript mudou → title, captions, review (hash amarra)
#   title mudou → SÓ title (+ render futuro, que aplica o título na hora)
#   captions mudou → SÓ captions (+ render futuro)
# Dependências para checagem:
DEPENDS_ON = {
    "transcript": (),
    "title": ("transcript",),
    "captions": ("transcript",),
    "review": ("transcript",),
}


def store_dir(source_path: str, root: Path | str = Path("work")) -> Path:
    """work/<stem>/artifacts — mesma convenção das sessões de revisão."""
    stem = re.sub(r"[^\w\-]+", "_", Path(source_path).stem).strip("_") or "clip"
    d = Path(root) / stem / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(store: Path | str, kind: str) -> Path:
    return Path(store) / f"{kind}.json"


def transcript_hash(segments: list) -> str:
    """SHA1 canônico dos segmentos — amarra título/legenda ao transcript exato."""
    from .models import Segment  # noqa: F401 (garante tipo no caller)
    canon = [(s.text, round(float(s.start), 3), round(float(s.end), 3),
              tuple((w.text, round(float(w.start), 3), round(float(w.end), 3))
                    for w in s.words))
             for s in segments]
    return hashlib.sha1(json.dumps(canon, ensure_ascii=False).encode()).hexdigest()[:16]


def save_artifact(store: Path | str, kind: str, source_fp: str,
                  source_path: str, config: dict, data: dict) -> Path:
    if kind not in ARTIFACT_VERSIONS:
        raise ValueError(f"artefato desconhecido: {kind!r}")
    p = _path(store, kind)
    Path(store).mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": kind,
        "artifact_version": ARTIFACT_VERSIONS[kind],
        "source_path": str(source_path),
        "source_fingerprint": source_fp,
        "config": dict(config or {}),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_artifact(store: Path | str, kind: str, source_fp: str,
                  config: dict | None = None) -> dict:
    """Retorna `data` do artefato válido. Levanta InvalidArtifact caso contrário."""
    if kind not in ARTIFACT_VERSIONS:
        raise ValueError(f"artefato desconhecido: {kind!r}")
    p = _path(store, kind)
    if not p.exists():
        raise InvalidArtifact(f"{kind}.json ausente em {Path(store)}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InvalidArtifact(f"{kind}.json ilegível ({e})")
    if payload.get("artifact") != kind:
        raise InvalidArtifact(f"{kind}.json com tipo divergente")
    if payload.get("artifact_version") != ARTIFACT_VERSIONS[kind]:
        raise InvalidArtifact(f"{kind}.json versão obsoleta "
                              f"({payload.get('artifact_version')} vs {ARTIFACT_VERSIONS[kind]})")
    if payload.get("source_fingerprint") != source_fp:
        raise InvalidArtifact(f"{kind}.json de outro source/config "
                              f"(fingerprint divergente)")
    want = dict(config or {})
    got = payload.get("config", {}) or {}
    for key in ARTIFACT_CONFIG_KEYS[kind]:
        if want.get(key) != got.get(key):
            raise InvalidArtifact(f"{kind}.json gerado com {key}={got.get(key)!r} "
                                  f"(atual: {want.get(key)!r})")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise InvalidArtifact(f"{kind}.json sem dados")
    return data


def artifact_state(store: Path | str, kind: str, source_fp: str,
                   config: dict | None = None,
                   transcript_hash: str | None = None) -> tuple[str, str]:
    """Estado sem gerar nada (nunca Whisper/LLM/ffmpeg): (estado, detalhe).

    ready | missing | invalid:<motivo>. title/captions/review ainda conferem
    o hash do transcript vigente — transcript trocado invalida os derivados,
    mas título nunca invalida legenda e vice-versa (irmãos).
    """
    try:
        data = load_artifact(store, kind, source_fp, config)
    except InvalidArtifact as e:
        reason = str(e)
        if "ausente" in reason:
            return "missing", reason
        return "invalid", reason
    if kind in ("title", "captions", "review") and transcript_hash is not None:
        if data.get("transcript_hash") != transcript_hash:
            return "invalid", f"{kind}.json de outro transcript"
    return "ready", "ok"


def final_state(final_path: Path | str, artifact_paths: list) -> tuple[str, str]:
    """RENDERED (existe e não é mais velho que os artefatos) | STALE | MISSING."""
    from pathlib import Path as _P
    p = _P(final_path)
    if not p.exists() or p.stat().st_size == 0:
        return "missing", "sem render"
    try:
        newest = max(_P(a).stat().st_mtime for a in artifact_paths
                     if _P(a).exists())
    except ValueError:
        newest = 0.0
    if p.stat().st_mtime < newest:
        return "stale", "artefatos mais novos que o render"
    return "rendered", "ok"


def serialize_segments(segments: list) -> list:
    return [
        {"text": s.text, "start": s.start, "end": s.end,
         "timestamp_source": getattr(s, "timestamp_source", "whisper"),
         "words": [{"text": w.text, "start": w.start, "end": w.end,
                    "confidence": getattr(w, "confidence", None),
                    "timestamp_source": getattr(w, "timestamp_source", "whisper")}
                   for w in s.words]}
        for s in segments
    ]


def deserialize_segments(data: list) -> list:
    """Lista → Segments. Qualquer forma inválida levanta (vira InvalidArtifact)."""
    from .models import Word, Segment
    if not isinstance(data, list):
        raise ValueError("segments deve ser lista")
    segs = []
    for s in data:
        words = [Word(w["text"], float(w["start"]), float(w["end"]),
                      confidence=w.get("confidence"),
                      timestamp_source=w.get("timestamp_source", "whisper") or "whisper")
                 for w in s.get("words", [])]
        segs.append(Segment(text=s.get("text", ""), start=float(s.get("start", 0)),
                            end=float(s.get("end", 0)), words=words,
                            timestamp_source=s.get("timestamp_source", "whisper")
                            or "whisper"))
    if not segs:
        raise ValueError("transcript vazio")
    return segs
