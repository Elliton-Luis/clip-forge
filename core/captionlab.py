"""captionlab.py — comparação A/B words vs intervals num trecho real.

Uso: python clipper.py caption-lab VIDEO --start S --dur D
Saída: debug/caption-lab/<stem>_<start>-<dur>/{words.srt,intervals.srt,compare.json}

Métricas por modo (objetivas, sem gosto): nº de cues, cobertura (união dos
spans / duração do trecho), duração máxima de cue, words cobertas, palavras
partidas no meio (tokens de continuação iniciando cue), cues sobrepostas.
Transcreve com cache normal (fingerprint inclui modelo); sem scoring.
"""
import json
from pathlib import Path

from .models import Word
from .video import build_srt, _group_cues, _group_cue_words

LAB_ROOT = Path("debug/caption-lab")


def _spans_union(cues_rel: list[tuple]) -> float:
    ivs = sorted((s, e) for s, e, _ in cues_rel if e > s)
    total, cs, ce = 0.0, None, None
    for s, e in ivs:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    if cs is not None:
        total += ce - cs
    return total


def _midword_splits(words: list, groups: list[list]) -> int:
    """Grupos que começam com token de continuação (ex: 'ITA' de 'dire'+'ita').

    Só vale em lotes com fronteiras (whisper.cpp); sem fronteiras o conceito
    não existe e retorna 0.
    """
    if not any(w.text[:1].isspace() for w in words if w.text):
        return 0
    n = 0
    for g in groups:
        if g and g[0].text and not g[0].text[:1].isspace():
            n += 1
    return n


def summarize(words: list, groups: list[list], cues_abs: list[tuple], span: float) -> dict:
    return {
        "cues": len(cues_abs),
        "coverage_sec": round(_spans_union([(s, e, t) for s, e, t in cues_abs]), 3),
        "coverage_ratio": round(_spans_union([(s, e, t) for s, e, t in cues_abs]) / span, 4) if span > 0 else 0.0,
        "max_cue_dur": round(max((e - s for s, e, _ in cues_abs), default=0.0), 3),
        "midword_splits": _midword_splits(words, groups),
        "overlapping_cues": sum(
            1 for i in range(len(cues_abs))
            for j in range(i + 1, len(cues_abs))
            if cues_abs[j][0] < cues_abs[i][1] - 1e-6 and cues_abs[i][0] < cues_abs[j][1] - 1e-6),
    }


def run(video: str, start: float, dur: float, cache_dir=None) -> Path:
    """Transcreve o trecho (usa cache) e salva A/B. Retorna o dir."""
    from .transcribe import transcribe as _transcribe
    from .config import WHISPER_MODEL_SIZE

    out = LAB_ROOT / f"{Path(video).stem}_{start:g}-{dur:g}"
    out.mkdir(parents=True, exist_ok=True)
    segments = _transcribe(video, cache_dir=Path(cache_dir).expanduser() if cache_dir else None,
                           force=False, backend=None, metrics=None)
    words = [w for s in segments for w in s.words
             if w.end > start and w.start < start + dur]
    if not words:
        raise RuntimeError("trecho sem palavras (sem fala?)")
    end = start + dur
    both = {}
    for mode in ("words", "intervals"):
        srt = build_srt(words, start, end, caption_mode=mode)
        (out / f"{mode}.srt").write_text(srt, encoding="utf-8")
        cues_abs = _group_cues(words, start, end, mode=mode)
        groups = _group_cue_words(words, start, end, mode=mode)
        both[mode] = summarize(words, groups, cues_abs, dur)
        both[mode]["srt_file"] = f"{mode}.srt"
    both["config"] = {"video": str(video), "start": start, "dur": dur,
                      "model": WHISPER_MODEL_SIZE}
    (out / "compare.json").write_text(json.dumps(both, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    for mode in ("words", "intervals"):
        m = both[mode]
        print(f"[caption-lab] {mode:9s} cues={m['cues']} cobertura={m['coverage_ratio']:.1%} "
              f"maxcue={m['max_cue_dur']}s midword={m['midword_splits']} overlap={m['overlapping_cues']}")
    print(f"[caption-lab] {out}")
    return out
