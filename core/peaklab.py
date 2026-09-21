"""peaklab.py — classic vs peak nos mesmos candidatos pontuados.

Uso:
    python clipper.py peak-compare VIDEO [--top N] [--cache-dir D] [--no-llm]
        [--min-score F] [--model M]

Método (nada do pipeline classic muda):
1. Transcreve (cache), constrói+snap candidatos, mede energia, pontua (LLM).
2. Clona os candidatos: metade vai ao select classic, metade ao pipeline
   peak (detecta auge → constrói clip ao redor → select por peak_score →
   títulos do auge só nos selecionados).
3. Imprime tabela lado a lado (duração, peak, score, motivo) e salva
   debug/peak-lab/<video>/{classic.json, peak.json, compare.txt}.

A pergunta a responder com os números: "o novo algoritmo realmente encontrou
momentos melhores?" — não "o algoritmo executou".
"""
import copy
import json
import time
from pathlib import Path

LAB_ROOT = Path("debug/peak-lab")


def _clone(candidates):
    from core.models import Candidate
    out = []
    for c in candidates:
        n = Candidate(start=c.start, end=c.end, text=c.text,
                      words=list(c.words), score=c.score, reason=c.reason,
                      title=c.title, hashtags=c.hashtags, failed=c.failed,
                      energy=c.energy, speech_rate=c.speech_rate,
                      original_start=c.original_start,
                      original_end=c.original_end, snapped=c.snapped)
        out.append(n)
    return out


def _row(c, mode):
    base = {
        "start": round(c.start, 1), "end": round(c.end, 1),
        "dur": round(c.end - c.start, 1), "score": c.score,
        "title": c.title, "reason": c.reason, "energy": c.energy,
    }
    if mode == "peak":
        base.update({
            "window": [round(c.window_start or 0, 1), round(c.window_end or 0, 1)],
            "peak": [round(c.peak_start or 0, 1), round(c.peak_end or 0, 1)],
            "peak_score": round(c.peak_score, 2), "peak_source": c.peak_source,
            "peak_reason": c.peak_reason, "title_source": c.title_source,
        })
    return base


def run(video: str, top: int = 5, cache_dir=None, use_llm: bool = True,
        min_score: float = 6.0, model: str | None = None,
        min_duration: float = 20.0, max_duration: float = 90.0) -> Path:
    from core.transcribe import transcribe as _transcribe
    from core.candidates import build as _build, snap_all as _snap
    from core.audio import annotate as _annotate
    from core.scoring import score as _score
    from core.selection import select as _select_classic
    from core.backends import probe_duration as _probe
    from core.config import (DEFAULT_MODEL, DEFAULT_MIN_SCORE,
                             MIN_CLIP_SECONDS, MAX_CLIP_SECONDS)
    from core.peaks import (detect_all as _detect_all,
                            build_clip_around_peak as _reframe,
                            select_peak as _select_peak,
                            retitle_peak as _retitle)
    t0 = time.time()
    model = model or DEFAULT_MODEL
    cdir = Path(cache_dir).expanduser() if cache_dir else None
    out = LAB_ROOT / Path(video).stem.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)

    segments = _transcribe(video, cache_dir=cdir, force=False, backend=None,
                           metrics=None)
    from core.cache import fingerprint
    from core.config import WHISPER_MODEL_SIZE
    fp = fingerprint(video, WHISPER_MODEL_SIZE) if cdir else None
    media_end = _probe(video)
    cands = _build(segments, min_dur=min_duration, max_dur=max_duration,
                   media_end=media_end)
    cands = _snap(cands, pad=0.8, min_dur=min_duration, max_dur=max_duration,
                  media_end=media_end)
    cands = _annotate(video, cands, enable=True)
    cands = _score(cands, model=model, cache_dir=cdir, fingerprint=fp,
                   force=False, context=None, examples_path=None, metrics=None)
    scored = [c for c in cands if not c.failed]
    if not scored:
        raise RuntimeError("nenhum candidato pontuado (scoring falhou?)")

    classic = _select_classic(_clone(cands), top, min_score=min_score,
                              max_per_10min=2)
    pb = _clone(cands)
    pstats = _detect_all([c for c in pb if not c.failed], model=model,
                         context=None, use_llm=use_llm)
    for c in pb:
        if not c.failed and c.peak_start is not None:
            _reframe(c, min_dur=min_duration, max_dur=max_duration,
                     media_end=media_end)
    peak = _select_peak(pb, top, min_score=min_score, max_per_10min=2)
    tstats = _retitle(peak, model) if use_llm else {"retitled": 0, "kept": len(peak)}

    data = {
        "config": {"video": video, "top": top, "model": model, "use_llm": use_llm,
                   "min_score": min_score, "min_duration": min_duration,
                   "max_duration": max_duration,
                   "elapsed_sec": round(time.time() - t0, 1)},
        "classic": [_row(c, "classic") for c in classic],
        "peak": [_row(c, "peak") for c in peak],
        "peak_stats": pstats,
        "title_stats": tstats,
    }
    (out / "classic.json").write_text(json.dumps(data["classic"], ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "peak.json").write_text(json.dumps(data["peak"], ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    (out / "compare.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    lines = [f"# peak-compare {video} top={top} llm={use_llm}",
             f"# classic: {len(classic)} clips | peak: {len(peak)} clips "
             f"(detect: {pstats.get('llm', 0)} llm + {pstats.get('heuristic', 0)} heur, "
             f"titles: {tstats.get('retitled', 0)} auge + {tstats.get('kept', 0)} janela)",
             "",
             "== CLASSIC (janela × score) =="]
    for i, c in enumerate(data["classic"], 1):
        lines.append(f"{i}. [{c['start']:.0f}-{c['end']:.0f}] {c['dur']:.0f}s "
                     f"score={c['score']:.1f} energy={c['energy']} :: {c['title']}")
        lines.append(f"   motivo: {c['reason'][:110]}")
    lines.append("")
    lines.append("== PEAK (auge → contexto) ==")
    for i, c in enumerate(data["peak"], 1):
        w, p = c["window"], c["peak"]
        lines.append(f"{i}. clip [{c['start']:.0f}-{c['end']:.0f}] {c['dur']:.0f}s "
                     f"(janela {w[0]:.0f}-{w[1]:.0f}) peak [{p[0]:.0f}-{p[1]:.0f}] "
                     f"pscore={c['peak_score']:.1f}/{c['peak_source']} "
                     f"score={c['score']:.1f} :: {c['title']} [{c['title_source']}]")
        lines.append(f"   auge: {c['peak_reason'][:110]}")
    # Diversidade: quantos peaks distintos o peak-mode cobriu?
    lines.append("")
    lines.append(f"# durações classic: {[c['dur'] for c in data['classic']]}")
    lines.append(f"# durações peak:    {[c['dur'] for c in data['peak']]}")
    (out / "compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[peaklab] {out}")
    return out
