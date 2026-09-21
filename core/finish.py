"""finish.py — modo FINISH: clip existente → título/legenda/render sem Whisper.

DISCOVERY (run_pipeline) transcreve uma vez e persiste o transcript como
artefato; FINISH só lê esse artefato. Regras garantidas aqui + testadas:

- transcript.json é a fonte de verdade (texto/segmentos/palavras/timestamps).
- Gerar/alterar título ou legenda NUNCA chama o Whisper.
- Renderizar NUNCA chama o Whisper.
- Artefato válido (mesmo source + config) é reutilizado; ausente/inválido →
  gera SOMENTE o que falta.
- Regenerar título/legenda não toca o transcript.

Para testes, as fronteiras externas são patcháveis: `transcribe` (Whisper),
`_score_title_llm` (LLM) e `cut_clip` (ffmpeg) — tudo o mais é puro.
"""
from pathlib import Path

from core.transcribe import transcribe
from core.cache import fingerprint
from core.video import cut as cut_clip

from . import artifacts as art


def _fp(clip: str, whisper_model: str | None) -> str:
    return fingerprint(clip, whisper_model)


def ensure_transcript(clip: str, store: Path | str,
                      whisper_model: str | None = None,
                      backend: str | None = None,
                      force: bool = False) -> tuple[list, bool]:
    """Transcript válido do store, ou transcreve o CLIP uma vez e salva.

    Retorna (segments, reused). Whisper roda no máximo 1× aqui — e nunca nas
    funções abaixo.
    """
    from .config import WHISPER_MODEL_SIZE
    wm = whisper_model or WHISPER_MODEL_SIZE
    fp = _fp(clip, wm)
    if not force:
        try:
            data = art.load_artifact(store, "transcript", fp)
            segs = art.deserialize_segments(data["segments"])
            if data.get("transcript_hash") != art.transcript_hash(segs):
                raise art.InvalidArtifact("transcript.json adulterado "
                                          "(hash divergente)")
            print(f"   -> transcript reutilizado ({len(segs)} segmentos, sem Whisper)")
            return segs, True
        except art.InvalidArtifact as e:
            print(f"   ! {e} — transcrevendo o clip uma vez")
    segs = transcribe(clip, cache_dir=None, force=True, backend=backend,
                      metrics=None, model_size=wm)
    art.save_artifact(store, "transcript", fp, clip,
                      {"whisper_model": wm, "backend": backend},
                      {"segments": art.serialize_segments(segs),
                       "transcript_hash": art.transcript_hash(segs),
                       "meta": {"segments": len(segs),
                                "words": sum(len(s.words) for s in segs)}})
    print(f"   -> transcript do clip salvo ({len(segs)} segmentos)")
    return segs, False


def _score_title_llm(text: str, energy: str, speech_rate: float,
                     duration: float, model: str) -> dict:
    """Um título via pipeline de scoring validado (sem reinventar prompt)."""
    from .scoring import score as _score
    from .models import Candidate
    c = Candidate(start=0.0, end=duration, text=text, words=[],
                  speech_rate=speech_rate)
    c.energy = energy
    _score([c], model=model, cache_dir=None, fingerprint=None, force=True,
           context=None, examples_path=None, metrics=None)
    if c.failed or not (c.title or "").strip():
        raise RuntimeError("LLM não gerou título para o clip")
    return {"title": c.title.strip(), "hashtags": c.hashtags,
            "score": c.score, "reason": c.reason}


def make_title(clip: str, segments: list, store: Path | str, source_fp: str,
               model: str, force: bool = False) -> tuple[dict, bool]:
    """Título do clip a partir do transcript (Whisper proibido aqui)."""
    from .audio import measure as _measure
    from .backends import probe_duration as _probe
    from .review import validate_grounding
    thash = art.transcript_hash(segments)
    if not force:
        try:
            data = art.load_artifact(store, "title", source_fp, {"model": model})
            if data.get("transcript_hash") != thash:
                raise art.InvalidArtifact("title.json de outro transcript")
            print(f"   -> título reutilizado: {data['title']!r}")
            return data, True
        except art.InvalidArtifact as e:
            print(f"   ! {e} — gerando título")
    text = " ".join(s.text for s in segments).strip()
    if not text:
        raise RuntimeError("transcript vazio — sem o que titular")
    dur = _probe(clip) or max((s.end for s in segments), default=0.0)
    energy = _measure(clip, 0.0, dur)
    words = [w for s in segments for w in s.words]
    sr = round(len(words) / max(0.5, dur), 2)
    got = _score_title_llm(text, energy, sr, dur, model)
    bad = validate_grounding(got["title"], text)
    if bad:
        print(f"   ! título com palavras fora do transcript: {bad}")
    data = {**got, "model": model, "transcript_hash": thash,
            "title_source": "llm"}
    art.save_artifact(store, "title", source_fp, clip, {"model": model}, data)
    return data, False


def set_manual_title(clip: str, segments: list, store: Path | str,
                     source_fp: str, title: str) -> dict:
    """Título manual (reutilizar/alterar sem Whisper, sem invalidar transcript)."""
    data = {"title": title.strip()[:50], "hashtags": "", "score": 0.0,
            "reason": "manual", "model": None, "transcript_hash": art.transcript_hash(segments),
            "title_source": "manual"}
    if not data["title"]:
        raise RuntimeError("--title vazio")
    art.save_artifact(store, "title", source_fp, clip, {"model": None}, data)
    return data


def make_captions(segments: list, store: Path | str, source_fp: str,
                  source_path: str, caption_mode: str = "phrases",
                  vertical: bool = True, highlight: set | None = None,
                  hook_title: str | None = None,
                  force: bool = False) -> tuple[dict, bool]:
    """Legendas (srt+ass) a partir do transcript. Funções puras, sem Whisper."""
    from .video import build_srt, build_ass
    from .backends import probe_duration as _probe
    thash = art.transcript_hash(segments)
    cfg = {"caption_mode": caption_mode, "vertical": vertical}
    if not force:
        try:
            data = art.load_artifact(store, "captions", source_fp, cfg)
            if data.get("transcript_hash") != thash:
                raise art.InvalidArtifact("captions.json de outro transcript")
            print("   -> legendas reutilizadas "
                  f"({caption_mode}, {'vertical' if vertical else 'wide'})")
            return data, True
        except art.InvalidArtifact as e:
            print(f"   ! {e} — gerando legendas")
    words = [w for s in segments for w in s.words]
    dur = max((s.end for s in segments), default=0.0)
    srt = build_srt(words, 0.0, dur if dur > 0 else None,
                    caption_mode=caption_mode)
    if vertical:
        width, height = 1080, 1920
    else:
        width, height = 1280, 720
    ass = build_ass(words, 0.0, dur, width, height,
                    highlight=highlight, hook_title=hook_title,
                    caption_mode=caption_mode) if dur > 0 else ""
    data = {"srt": srt, "ass": ass, "width": width, "height": height,
            **cfg, "transcript_hash": thash}
    art.save_artifact(store, "captions", source_fp, source_path, cfg, data)
    return data, False


def render_clip(clip: str, segments: list, title_data: dict | None,
                out_path: str | Path, vertical: bool = True,
                with_captions: bool = True, with_title: bool = True,
                caption_mode: str = "phrases") -> Path:
    """Render final a partir de transcript + título. Sem Whisper (cut puro)."""
    from .models import Candidate
    from .backends import probe_duration as _probe
    out = Path(out_path)
    if out.resolve() == Path(clip).resolve():
        raise RuntimeError(f"saída coincide com a entrada ({out}) — original preservado")
    dur = _probe(clip) or max((s.end for s in segments), default=0.0)
    if dur <= 0:
        raise RuntimeError("duração inválida para renderizar")
    words = [w for s in segments for w in s.words]
    title = (title_data or {}).get("title", "") if with_title else ""
    c = Candidate(start=0.0, end=dur,
                  text=" ".join(s.text for s in segments), words=words,
                  score=float((title_data or {}).get("score", 0) or 0),
                  title=title,
                  hashtags=(title_data or {}).get("hashtags", ""),
                  reason=(title_data or {}).get("reason", ""),
                  energy="media",
                  speech_rate=round(len(words) / max(0.5, dur), 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    cut_clip(clip, c, out, vertical=vertical, captions=with_captions,
             caption_mode=caption_mode, title=with_title and bool(title))
    return out
