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
`_title_llm` (LLM de título) e `cut_clip` (ffmpeg) — tudo o mais é puro.

Grafo de dependências (testado): title → transcript; captions → transcript;
render → transcript (+title/captions conforme flags). title ↛ captions,
captions ↛ title, e NADA aqui chama descoberta (candidatos/scoring/NMS).
"""
from pathlib import Path

from core.transcribe import transcribe
from core.cache import fingerprint
from core.video import cut as cut_clip

from . import artifacts as art

TITLE_PROMPT = """Você recebe a TRANSCRIÇÃO de um clip já escolhido (o momento é dado, não avalie se é bom). Escreva um "title" de até 50 caracteres, sem emoji, que represente o acontecimento central do clip, e exatamente 3 hashtags em minúsculas.

Regra absoluta: use SOMENTE palavras, pessoas, fatos e acontecimentos presentes no texto. Nunca adicione informação nova. Na dúvida entre chamativo e fiel, escolha fiel.
Responda SOMENTE JSON: {"title": "...", "hashtags": "#tag1 #tag2 #tag3"}"""


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


def _title_llm(text: str, model: str) -> dict:
    """Um título via LLM direto — NÃO passa pelo scoring de descoberta.

    O scoring avalia N candidatos para SELECIONAR; aqui o momento já está
    escolhido e só falta nomeá-lo. Reusa só a infra de chaves/retry.
    """
    import json as _json
    import time as _t
    from openai import OpenAI
    from .config import NVIDIA_BASE_URL
    from .scoring import (_load_api_keys, _KeyGate, _strip_reasoning,
                          _sanitize_hashtags)
    keys = _load_api_keys()
    clients = [OpenAI(base_url=NVIDIA_BASE_URL, api_key=k,
                      timeout=180, max_retries=0) for k in keys]
    gates = [_KeyGate() for _ in clients]
    last = None
    for attempt in range(3):
        gates[attempt % len(clients)].wait()
        try:
            resp = clients[attempt % len(clients)].chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": TITLE_PROMPT},
                          {"role": "user", "content": text[:1500]}],
                temperature=0.3, max_tokens=500)
            data = _json.loads(_strip_reasoning(
                (resp.choices[0].message.content or "").strip()))
            title = str(data.get("title", "") or "").strip()[:50]
            if not title:
                raise ValueError("título vazio")
            return {"title": title,
                    "hashtags": _sanitize_hashtags(data.get("hashtags", "")),
                    "score": None, "reason": "finish-llm"}
        except Exception as e:
            last = e
            print(f"      ! título LLM tentativa {attempt+1}/3: {e}")
            if attempt < 2:
                _t.sleep([10, 30][attempt])
    raise RuntimeError(f"LLM não gerou título para o clip ({last})")


def make_title(clip: str, segments: list, store: Path | str, source_fp: str,
               model: str, force: bool = False) -> tuple[dict, bool]:
    """Título do clip a partir do transcript (Whisper e scoring proibidos)."""
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
    got = _title_llm(text, model)
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


def run_finish(clip: str, out_dir: str | Path = "cortes", only: str = "all",
               store: Path | str | None = None, model: str | None = None,
               whisper_model: str | None = None, title_text: str | None = None,
               no_captions: bool = False, no_title: bool = False,
               caption_mode: str = "phrases", vertical: bool = True,
               regen_title: bool = False, regen_captions: bool = False,
               force_transcribe: bool = False, review: bool = False,
               review_input_fn=None) -> dict:
    """Orquestra o FINISH. Gera SOMENTE o pedido; reutiliza o resto.

    only: all (título+legenda+render) | title | captions | render.
    title e captions são independentes: pedir um nunca gera o outro.
    review=True (só com render): tela A/E/R/T/L/S/Q sobre os artefatos, com o
    vídeo ainda sem nada queimado; edição na revisão regenera as legendas
    (transcript.json intacto e registrado); render só depois do aceite, com
    saída validada antes de qualquer limpeza (que nunca é automática).
    review_input_fn: costura de teste (None = terminal real).
    Levanta RuntimeError com mensagem clara (CLI converte em sys.exit).
    """
    from .config import DEFAULT_MODEL
    from .video import highlight_words_from_title
    model = model or DEFAULT_MODEL
    if only not in ("all", "title", "captions", "render"):
        raise ValueError(f"only inválido: {only!r}")
    store = Path(store).expanduser() if store else art.store_dir(clip)
    out_dir = Path(out_dir)

    segs, t_reused = ensure_transcript(clip, store, whisper_model=whisper_model,
                                       force=force_transcribe)
    from .config import WHISPER_MODEL_SIZE
    fp = _fp(clip, whisper_model or WHISPER_MODEL_SIZE)
    result: dict = {"clip": clip, "transcript_reused": t_reused,
                    "segments": len(segs)}

    title = None
    needs_title = (only in ("all", "title") and not no_title) or \
                  (only == "render" and not no_title)
    if title_text and only in ("all", "title", "render") and not no_title:
        title = set_manual_title(clip, segs, store, fp, title_text)
        print(f"   -> título manual: {title['title']!r}")
        result["title_reused"] = False
    elif needs_title:
        if only == "render":
            # Render não gera título sozinho: ou existe, ou --title/--no-title.
            try:
                title = art.load_artifact(store, "title", fp, {"model": model})
                if title.get("transcript_hash") != art.transcript_hash(segs):
                    raise art.InvalidArtifact("title.json de outro transcript")
                result["title_reused"] = True
            except art.InvalidArtifact as e:
                raise RuntimeError(
                    f"Render precisa de título: {e} "
                    f"(rode --only title ou passe --title/--no-title)")
        else:
            title, reused = make_title(clip, segs, store, fp, model,
                                       force=regen_title)
            result["title_reused"] = reused
            print(f"   -> título: {title['title']!r}")
    if only == "title":
        result["title"] = (title or {}).get("title", "")
        return result

    caps = None
    if not no_captions and only in ("all", "captions", "render"):
        caps, reused = make_captions(
            segs, store, fp, clip, caption_mode=caption_mode, vertical=vertical,
            highlight=highlight_words_from_title((title or {}).get("title", "")),
            hook_title=(title or {}).get("title") if not no_title else None,
            force=regen_captions)
        result["captions_reused"] = reused
    if only == "captions":
        out_dir.mkdir(parents=True, exist_ok=True)
        srt_p = out_dir / (Path(clip).stem + ".srt")
        srt_p.write_text((caps or {}).get("srt", ""), encoding="utf-8")
        print(f"LEGENDA: {srt_p.resolve()} (sem Whisper, sem render)")
        result["srt"] = str(srt_p)
        return result

    # render (all|render): revisão opcional antes do burn-in.
    with_title = not no_title
    with_captions = not no_captions
    if review:
        from . import clipreview as _cr
        from . import finishreview as _fr
        from .models import Candidate, Segment
        from .video import highlight_words_from_title as _hl
        from .backends import probe_duration as _probe_dur
        dur = _probe_dur(clip) or max((s.end for s in segs), default=0.0)
        words = [w for s in segs for w in s.words]
        c = Candidate(start=0.0, end=dur,
                      text=" ".join(s.text for s in segs), words=list(words),
                      score=float((title or {}).get("score", 0) or 0),
                      title=(title or {}).get("title", ""),
                      speech_rate=round(len(words) / max(0.5, dur), 2))
        session = _cr.session_dir(clip)
        # Nome próprio: preview_01.mp4 é da revisão DISCOVERY (outros bounds)
        # e seria reaproveitado errado pelo skip-if-exists.
        preview = session / "preview_finish.mp4"
        if not preview.exists():
            print(f"   -> preview de revisão (sem queimar nada): {preview.name}")
            cut_clip(clip, c, preview, vertical=vertical, captions=False,
                     title=False)
        previews = [preview]
        review_state = {"edited_words": None}

        def _edit() -> bool:
            from pathlib import Path as _P
            path = session / "finish_words.txt"
            _cr.dump_clip_words(c, path)
            print(f"   Edite: {path.resolve}\n"
                  f"   (texto E timestamps absolutos; transcript.json NÃO muda)")
            pause = review_input_fn or input
            try:
                pause("   Enter quando terminar (Ctrl+C cancela)...")
            except EOFError:
                raise SystemExit("Entrada não-interativa: revisão exige terminal.")
            before = [(w.text, w.start, w.end) for w in c.words]
            try:
                _cr.apply_clip_words(c, path)
            except RuntimeError as e:
                print(f"   ! edição inválida ({e}) — nada mudou")
                return False
            changed = [(w.text, w.start, w.end) for w in c.words] != before
            if changed:
                review_state["edited_words"] = list(c.words)
                print("   -> EDIÇÃO MANUAL registrada: vale p/ as legendas deste "
                      "render; transcript.json original preservado")
            return changed

        def _regen_title():
            t, _ = make_title(clip, segs, store, fp, model, force=True)
            title.clear()
            title.update(t)
            print(f"   -> título regenerado: {title['title']!r}")
            return title["title"]

        def _regen_captions():
            cp, _ = make_captions(
                segs, store, fp, clip, caption_mode=caption_mode,
                vertical=vertical,
                highlight=_hl((title or {}).get("title", "")),
                hook_title=(title or {}).get("title") if with_title else None,
                force=True)
            print("   -> legendas regeneradas (transcript intacto)")
            return cp.get("srt", "")

        outcome = _fr.run_review(
            clip, segs, (title or {}).get("title"),
            (caps or {}).get("srt"), dur, store, fp,
            preview=previews[0] if previews else None,
            regen_title_fn=_regen_title if with_title else None,
            regen_captions_fn=_regen_captions if with_captions else None,
            edit_fn=_edit if with_captions else None,
            input_fn=review_input_fn)
        if outcome["action"] == "skipped":
            raise RuntimeError("Clip pulado na revisão — nada a renderizar.")
        st = outcome["state"]
        with_title = with_title and st.get("title_enabled", True)
        with_captions = with_captions and st.get("captions_enabled", True)
        if review_state["edited_words"] is not None:
            segs = [Segment(text=c.text, start=0.0, end=dur,
                            words=list(c.words))]
            if with_captions:
                caps, _ = make_captions(
                    segs, store, fp, clip, caption_mode=caption_mode,
                    vertical=vertical, highlight=_hl(c.title),
                    hook_title=c.title if with_title else None, force=True)
                print("   -> legendas regeneradas do texto revisado")
        result["review"] = {"action": outcome["action"],
                            "edited": bool(st.get("edited", False)),
                            "title_enabled": with_title,
                            "captions_enabled": with_captions}
    out = out_dir / (Path(clip).stem + "_final.mp4")
    render_clip(clip, segs, title, out, vertical=vertical,
                with_captions=with_captions, with_title=with_title,
                caption_mode=caption_mode)
    _validate_output(out, clip)
    print(f"FINAL: {out.resolve()}")
    result.update({"out": str(out), "title": (title or {}).get("title", "")})
    return result


def _validate_output(out: Path, clip: str) -> None:
    """Render validado ANTES de qualquer limpeza: existe, tem bytes e dura
    o esperado. Falha aqui nunca apaga artefatos (nada apaga nada sozinho)."""
    from .backends import probe_duration as _probe
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"render inválido: {out} vazio/ausente — "
                           f"artefatos preservados em {out.parent}")
    expected = _probe(clip)
    got = _probe(str(out))
    if expected and got and abs(got - expected) > max(2.0, 0.2 * expected):
        raise RuntimeError(f"render com duração suspeita ({got:.1f}s vs "
                           f"{expected:.1f}s do clip) — artefatos preservados")
