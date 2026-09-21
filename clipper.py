#!/usr/bin/env python3
"""
clipper.py — orquestrador fino (CLI + TUI). Toda lógica vive em core/*.

Duas formas de operar (mesma configuração, mesmo pipeline):
  python clipper.py video.mp4 --out cortes/   # CLI tradicional
  python clipper.py                            # TUI interativa (terminal)

Mantém compatibilidade: `python clipper.py video.mp4 --out cortes/`
"""
import argparse
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from core.config import (
    DEFAULT_MODEL, DEFAULT_PAD_SECONDS, DEFAULT_MIN_SCORE, DEFAULT_MAX_PER_10MIN,
    MIN_CLIP_SECONDS, MAX_CLIP_SECONDS,
    CLIPPER_TRANSCRIBE_BACKEND, CLIPPER_CPU_THREADS, CLIPPER_FFMPEG_THREADS,
    WHISPER_MODEL_SIZE, CLIPPER_ALIGN, CLIPPER_SELECTION_MODE,
)
from core.preflight import check as preflight
from core.cache import fingerprint
from core.transcribe import transcribe
from core.candidates import build as build_candidates, snap_all
from core.audio import annotate as annotate_audio
from core.scoring import score as score_candidates
from core.selection import select as select_top
from core.video import cut as cut_clip, sanitize_filename
from core.metrics import ExecutionMetrics

# Re-export para compatibilidade: `from clipper import Candidate` continua funcionando
from core.models import Word, Segment, Candidate  # noqa: F401
from core.video import build_srt, face_center_x  # compat
from core.video import build_srt as _build_srt, face_center_x as _detect_face_center_x  # alias legado
from core.candidates import build as build_candidates_alias  # compat
from core.transcribe import transcribe as transcribe_alias  # compat
from core.scoring import score as score_candidates_alias  # compat
from core.selection import select as select_top_clips_alias  # compat
from core.video import cut as cut_clip_alias  # compat
from core.video import sanitize_filename as _sanitize_filename  # compat


# ----------------------------------------------------------------------------
# 1. Parsing / entrada
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gerador automático de cortes para lives/gameplay.")
    # nargs="?" → sem vídeo abre a TUI; com vídeo, CLI tradicional idêntica.
    p.add_argument("video", nargs="?", default=None,
                   help="Caminho do vídeo de entrada (mp4, mkv, etc.). Omitido = interface interativa")
    p.add_argument("--out", default="cortes", help="Pasta de saída (padrão: ./cortes)")
    p.add_argument("--top", type=int, default=8, help="Quantos clipes gerar (padrão: 8)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo NIM (padrão: {DEFAULT_MODEL})")
    p.add_argument("--no-vertical", action="store_true", help="Não recortar para 9:16")
    p.add_argument("--no-captions", action="store_true", help="Não queimar legendas")
    p.add_argument("--no-title", action="store_true", help="Não queimar o título/hook (independente das legendas)")
    p.add_argument("--caption-mode", default="phrases", choices=["words", "intervals", "phrases"],
                   help="Agrupamento das legendas: phrases (frases/unidades de fala, padrão) "
                        "words (blocos de leitura) ou intervals (rajadas de fala, experimental)")
    p.add_argument("--acoustic-captions", action="store_true",
                   help="Queima *ÁUDIO ESTOURADO* nos trechos com clipping (experimental, off)")
    p.add_argument("--laughs", default=None,
                   help="Arquivo com ranges de risada 'INICIO FIM' (s) por linha — queima *RISADA ESTOURADA* em amarelo")
    p.add_argument("--cache-dir", default=None, help="Diretório de cache (ex: .cache/clipper)")
    p.add_argument("--force-retranscribe", action="store_true", help="Ignora cache transcrição")
    p.add_argument("--force-rescore", action="store_true", help="Ignora cache scores")
    p.add_argument("--pad", type=float, default=DEFAULT_PAD_SECONDS, help=f"Respiro antes/depois (padrão: {DEFAULT_PAD_SECONDS}s)")
    p.add_argument("--min-duration", type=float, default=MIN_CLIP_SECONDS, help=f"Duração mínima do clipe (padrão: {MIN_CLIP_SECONDS}s)")
    p.add_argument("--max-duration", type=float, default=MAX_CLIP_SECONDS, help=f"Duração máxima do clipe, sempre respeitada (padrão: {MAX_CLIP_SECONDS}s)")
    p.add_argument("--no-audio-features", action="store_true", help="Desativa energia de áudio")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help=f"Score mínimo (padrão: {DEFAULT_MIN_SCORE})")
    p.add_argument("--max-per-10min", type=int, default=DEFAULT_MAX_PER_10MIN, help=f"Máximo por 10min (padrão: {DEFAULT_MAX_PER_10MIN})")
    p.add_argument("--context", default=None, help="Contexto da live injetado no prompt")
    p.add_argument("--examples", default=None, help="Caminho para examples.json few-shot")
    p.add_argument("--transcribe-backend", default=CLIPPER_TRANSCRIBE_BACKEND,
                   choices=["auto", "gpu", "vulkan", "openvino", "cpu"],
                   help="Backend de transcrição (padrão: env CLIPPER_TRANSCRIBE_BACKEND ou auto; "
                        "'gpu' exige GPU e falha claramente se indisponível)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL_SIZE,
                   help=f"Modelo Whisper ggml-* em models/ (padrão: {WHISPER_MODEL_SIZE}; "
                        f"large-v3 p/ força máxima com PC em repouso)")
    p.add_argument("--debug-captions", action="store_true",
                   help="Diagnóstico de legendas: preserva ASS/SRT/transcript/diagnóstico em debug/")
    p.add_argument("--review-transcript", action="store_true",
                   help="Pausa após transcrever p/ revisão humana (edita work/<video>/transcription/words.txt, aprova, continua sem re-rodar Whisper)")
    p.add_argument("--review-titles", action="store_true",
                   help="Pausa após scoring p/ revisar títulos (work/<video>/titles.txt, validados contra a transcrição)")
    p.add_argument("--review-clips", action="store_true",
                   help="Revisa os clips SELECIONADOS antes do burn-in (preview sem legenda + A/E/S/Q, continua de onde parou)")
    p.add_argument("--work-dir", default=None,
                   help="Sessão de revisão (work/<video>): usa a transcrição APROVADA e pula o Whisper (regeneração)")
    p.add_argument("--custom-words", default=None,
                   help="JSON de vocabulário ({\"words\": [...]}) aplicado como correção exata pós-transcrição")
    p.add_argument("--align", default=CLIPPER_ALIGN,
                   choices=["off", "whisper-refine", "wav2vec2"],
                   help="Forced alignment opt-in (padrão: env CLIPPER_ALIGN ou off; "
                        "whisper-refine reusa o próprio Whisper em janela curta, sem deps novas)")
    p.add_argument("--selection-mode", default=CLIPPER_SELECTION_MODE,
                   choices=["classic", "peak"],
                   help="Seleção: classic (janela+score, padrão calibrado) ou peak "
                        "(experimental: clip construído ao redor do auge)")
    return p


def parse_cli(argv=None):
    return build_parser().parse_args(argv)


# ----------------------------------------------------------------------------
# 2. Configuração (dict canônico — CLI e TUI produzem exatamente este formato)
# ----------------------------------------------------------------------------

CONFIG_FIELDS = (
    "video", "out", "top", "model", "no_vertical", "no_captions",
    "no_title",
    "acoustic_captions", "whisper_model", "laughs", "caption_mode",
    "cache_dir", "force_retranscribe", "force_rescore", "pad",
    "min_duration", "max_duration",
    "no_audio_features", "min_score", "max_per_10min", "context",
    "examples", "transcribe_backend", "debug_captions",
    "review_transcript", "review_titles", "review_clips", "work_dir", "custom_words",
    "align", "selection_mode",
)


def default_config() -> dict:
    """Configuração com os mesmos defaults da CLI (sem vídeo)."""
    args = parse_cli([])
    cfg = config_from_args(args)
    cfg["video"] = ""
    return cfg


def config_from_args(args) -> dict:
    """argparse.Namespace → dict canônico. Sem validação (ver validate_config)."""
    return {
        "video": str(Path(args.video).expanduser()) if args.video else "",
        "out": args.out,
        "top": args.top,
        "model": args.model,
        "no_vertical": bool(args.no_vertical),
        "no_captions": bool(args.no_captions),
        "no_title": bool(args.no_title),
        "acoustic_captions": bool(args.acoustic_captions),
        "caption_mode": args.caption_mode,
        "laughs": args.laughs,
        "whisper_model": args.whisper_model,
        "cache_dir": args.cache_dir,
        "force_retranscribe": bool(args.force_retranscribe),
        "force_rescore": bool(args.force_rescore),
        "pad": args.pad,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "no_audio_features": bool(args.no_audio_features),
        "min_score": args.min_score,
        "max_per_10min": args.max_per_10min,
        "context": args.context,
        "examples": args.examples,
        "transcribe_backend": args.transcribe_backend,
        "debug_captions": bool(args.debug_captions),
        "review_transcript": bool(args.review_transcript),
        "review_titles": bool(args.review_titles),
        "review_clips": bool(getattr(args, "review_clips", False)),
        "work_dir": args.work_dir,
        "custom_words": args.custom_words,
        "align": getattr(args, "align", "off"),
        "selection_mode": getattr(args, "selection_mode", "classic"),
    }


def validate_config(cfg: dict) -> None:
    """Validação pré-métricas, idêntica à CLI original (mensagens iguais).

    Existência do vídeo e --top ficam com o preflight (que gera relatório),
    como antes. A TUI valida o resto no formulário antes de executar.
    """
    if not (0 <= cfg["pad"] <= 5):
        sys.exit("--pad deve estar entre 0 e 5")
    if not (0 <= cfg["min_score"] <= 10):
        sys.exit("--min-score deve estar entre 0 e 10")
    if not (1 <= cfg["max_per_10min"] <= 20):
        sys.exit("--max-per-10min deve estar entre 1 e 20")
    if not (0 < cfg["min_duration"] <= cfg["max_duration"] <= 600):
        sys.exit("--min-duration/--max-duration exigem 0 < min <= max <= 600")
    try:
        from core.translab import resolve_model as _resolve_model
        _resolve_model(cfg.get("whisper_model") or WHISPER_MODEL_SIZE)
    except RuntimeError as e:
        sys.exit(f"--whisper-model inválido: {e}")
    try:
        from core.alignment import resolve_backend as _resolve_align
        _resolve_align(cfg.get("align", "off"))
    except ValueError as e:
        sys.exit(f"--align inválido: {e}")
    try:
        from core.peaks import resolve_selection_mode as _resolve_sel
        _resolve_sel(cfg.get("selection_mode", "classic"))
    except ValueError as e:
        sys.exit(f"--selection-mode inválido: {e}")


def cli_command(cfg: dict) -> str:
    """Configuração canônica → comando CLI equivalente (para aprendizado)."""
    parts = ["python", "clipper.py", cfg["video"]]
    if cfg["out"] != "cortes":
        parts += ["--out", cfg["out"]]
    if cfg["top"] != 8:
        parts += ["--top", str(cfg["top"])]
    parts += ["--model", cfg["model"]]
    if cfg["no_vertical"]:
        parts.append("--no-vertical")
    if cfg["no_captions"]:
        parts.append("--no-captions")
    if cfg.get("no_title"):
        parts.append("--no-title")
    if cfg.get("acoustic_captions"):
        parts.append("--acoustic-captions")
    if cfg.get("caption_mode", "phrases") != "phrases":
        parts += ["--caption-mode", cfg["caption_mode"]]
    if cfg.get("laughs"):
        parts += ["--laughs", cfg["laughs"]]
    if cfg.get("whisper_model") != WHISPER_MODEL_SIZE:
        parts += ["--whisper-model", cfg["whisper_model"]]
    if cfg["cache_dir"]:
        parts += ["--cache-dir", cfg["cache_dir"]]
    if cfg["force_retranscribe"]:
        parts.append("--force-retranscribe")
    if cfg["force_rescore"]:
        parts.append("--force-rescore")
    if cfg["pad"] != DEFAULT_PAD_SECONDS:
        parts += ["--pad", str(cfg["pad"])]
    if cfg["min_duration"] != MIN_CLIP_SECONDS:
        parts += ["--min-duration", str(cfg["min_duration"])]
    if cfg["max_duration"] != MAX_CLIP_SECONDS:
        parts += ["--max-duration", str(cfg["max_duration"])]
    if cfg["no_audio_features"]:
        parts.append("--no-audio-features")
    if cfg["min_score"] != DEFAULT_MIN_SCORE:
        parts += ["--min-score", str(cfg["min_score"])]
    if cfg["max_per_10min"] != DEFAULT_MAX_PER_10MIN:
        parts += ["--max-per-10min", str(cfg["max_per_10min"])]
    if cfg["context"]:
        parts += ["--context", cfg["context"]]
    if cfg["examples"]:
        parts += ["--examples", cfg["examples"]]
    if cfg["transcribe_backend"] != CLIPPER_TRANSCRIBE_BACKEND:
        parts += ["--transcribe-backend", cfg["transcribe_backend"]]
    if cfg.get("debug_captions"):
        parts.append("--debug-captions")
    if cfg.get("review_transcript"):
        parts.append("--review-transcript")
    if cfg.get("review_titles"):
        parts.append("--review-titles")
    if cfg.get("review_clips"):
        parts.append("--review-clips")
    if cfg.get("work_dir"):
        parts += ["--work-dir", cfg["work_dir"]]
    if cfg.get("custom_words"):
        parts += ["--custom-words", cfg["custom_words"]]
    if cfg.get("align", "off") != "off":
        parts += ["--align", cfg["align"]]
    if cfg.get("selection_mode", "classic") != "classic":
        parts += ["--selection-mode", cfg["selection_mode"]]
    return " ".join(shlex.quote(x) for x in parts)


# ----------------------------------------------------------------------------
# 3b. Revisão humana (transcrição aprovada = fonte da verdade)
# ----------------------------------------------------------------------------

def _read_line_raw(msg: str) -> str:
    """Lê uma linha aceitando Enter como \\r ou \\n, com eco manual.

    Imune a tty com icrnl desligado (resto de sessão curses/editor), onde
    input() exibiria ^M sem nunca submeter a linha. Ctrl+C cancela.
    """
    import tty as _tty
    import termios as _termios
    sys.stdout.write(msg)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        raise EOFError("stdin não é tty")
    old = _termios.tcgetattr(fd)
    buf: list[str] = []
    try:
        _tty.setraw(fd)  # sem tradução de linha: aceitamos CR e LF
        while True:
            ch = os.read(fd, 1)
            if ch in (b"\r", b"\n"):
                break
            if ch == b"\x03":
                raise KeyboardInterrupt
            if ch in (b"\x7f", b"\x08"):
                if buf:
                    buf.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch == b"":
                raise EOFError
            if b" " <= ch <= b"~":
                buf.append(ch.decode("ascii"))
                sys.stdout.write(ch.decode("ascii"))
                sys.stdout.flush()
    finally:
        _termios.tcsetattr(fd, _termios.TCSADRAIN, old)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(buf)


def _pause(msg: str) -> None:
    """Pausa interativa (Enter confirma). EOF = aborta, nunca aprova cego."""
    try:
        _read_line_raw(msg)
        return
    except (ImportError, OSError, ValueError, EOFError):
        pass  # sem termios/tty: tenta o input() clássico abaixo
    try:
        input(msg)
    except EOFError:
        sys.exit("Entrada não-interativa: revisão humana exige terminal. "
                 "Edite os arquivos em work/ e use transcribe-approve.")


def _ask_confirm(msg: str) -> bool:
    """Pergunta Y/N imune a tty sem icrnl (mesma causa do ^M no _pause)."""
    try:
        ans = _read_line_raw(msg)
    except (ImportError, OSError, ValueError, EOFError):
        try:
            ans = input(msg)
        except EOFError:
            return False
    return ans.strip().lower() in ("y", "yes", "s", "sim")


def _session_for(cfg: dict, video: str) -> Path:
    from core import review as _rev
    if cfg.get("work_dir"):
        d = Path(cfg["work_dir"]) / "transcription"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return _rev.session_dir(video)


def _pause_for_transcript_review(cfg: dict, video: str, segments: list) -> list:
    """Dump → edição externa → validação → aprovação. Retorna segments finais."""
    from core import review as _rev
    session = _session_for(cfg, video)
    mapping = _rev.load_custom_words(cfg.get("custom_words"))
    if mapping:
        n = _rev.apply_custom_words(segments, mapping)
        print(f"   -> custom_words: {n} correções exatas de {cfg['custom_words']}")
    _rev.save_transcript(segments, session / "transcript.json", status=_rev.DRAFT)
    words_txt = session / "words.txt"
    _rev.dump_words_txt(segments, words_txt)
    print(f"\n[REVISÃO] Transcrição em: {words_txt.resolve()}")
    print("   Edite o TEXTO (timestamps preservados por linha), salve e volte.")
    _pause("   Pressione Enter quando terminar de editar (Ctrl+C cancela)...")
    edited = _rev.parse_words_txt(words_txt)
    final = _rev.approve(session, edited)
    print(f"   -> TRANSCRIÇÃO APROVADA ({len(edited)} segmentos): {final.resolve()}")
    return edited


def _pause_for_titles_review(cfg: dict, video: str, segments: list,
                             selected: list) -> list:
    """titles.txt → edição → validação grounded. Violações viram warnings."""
    from core import review as _rev
    session = _session_for(cfg, video)
    approved_text = " ".join(s.text for s in segments)
    titles_txt = session / "titles.txt"
    lines = ["# Edite o título após '|'. [SEM TITULO] remove; linha vazia mantém.",
             ""]
    for i, c in enumerate(selected, start=1):
        lines.append(f"{i:02d} | {c.title}")
    titles_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[REVISÃO] Títulos em: {titles_txt.resolve()}")
    _pause("   Pressione Enter quando terminar de editar (Ctrl+C cancela)...")
    wanted: dict[int, str] = {}
    for line in titles_txt.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s?(.*)$", line)
        if m:
            wanted[int(m.group(1))] = m.group(2).strip()
    for i, c in enumerate(selected, start=1):
        if i not in wanted or not wanted[i]:
            continue  # linha apagada/vazia = mantém sugestão
        if wanted[i] == "[SEM TITULO]":
            c.title = ""
            continue
        c.title = wanted[i][:50]
        bad = _rev.validate_grounding(c.title, approved_text)
        if bad:
            print(f"   ! clipe {i}: título com palavras fora da transcrição: {bad}")
    return selected


def _title_warnings(segments: list, selected: list) -> dict[int, dict]:
    from core import review as _rev
    approved_text = " ".join(s.text for s in segments)
    out = {}
    for i, c in enumerate(selected, start=1):
        tw = _rev.validate_grounding(c.title or "", approved_text)
        hw = _rev.validate_highlights(c.title or "", approved_text)
        if tw or hw:
            out[i] = {"title": tw, "highlight": hw}
    return out


# ----------------------------------------------------------------------------
# 3. Execução (pipeline existente, inalterado — só lê o dict)
# ----------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> None:
    video = str(Path(cfg["video"]).expanduser())
    out_dir = Path(cfg["out"])

    validate_config(cfg)

    # Métricas por execução: um relatório JSON por vídeo, mesmo em falha.
    run_args = {
        "top": cfg["top"], "model": cfg["model"], "transcribe_backend": cfg["transcribe_backend"],
        "whisper_model": cfg["whisper_model"], "cpu_threads": CLIPPER_CPU_THREADS,
        "ffmpeg_threads": CLIPPER_FFMPEG_THREADS, "pad": cfg["pad"],
        "min_duration": cfg["min_duration"], "max_duration": cfg["max_duration"],
        "min_score": cfg["min_score"], "max_per_10min": cfg["max_per_10min"],
        "vertical": not cfg["no_vertical"], "captions": not cfg["no_captions"],
        "title": not cfg.get("no_title", False),
        "audio_features": not cfg["no_audio_features"],
        "selection_mode": cfg.get("selection_mode", "classic"),
    }
    metrics = ExecutionMetrics(video, run_args).start()
    stage = "preflight"
    try:
        with metrics.stage("preflight"):
            preflight(video, out_dir, cfg["top"], transcribe_backend=cfg["transcribe_backend"])

        cache_dir = Path(cfg["cache_dir"]).expanduser() if cfg["cache_dir"] else None
        fp = fingerprint(video, cfg["whisper_model"]) if cache_dir else None

        stage = "transcribe"
        with metrics.stage("transcribe"):
            if cfg.get("work_dir"):
                # Regeneração: transcrição APROVADA, Whisper nunca re-executa.
                from core import review as _rev
                approved_path = Path(cfg["work_dir"]) / "transcription" / "transcript.json"
                segments = _rev.require_approved(approved_path)
                print(f"[1/5] Transcrição aprovada carregada ({len(segments)} segmentos, sem Whisper)")
                try:
                    metrics.set_transcription(
                        model="review", backend_requested=cfg["transcribe_backend"],
                        backend_used="review", device=None, gpu=None,
                        time_sec=0.0, segments=len(segments),
                        fallback=False, tried_backends=["review"], error=None)
                except Exception:
                    pass
            else:
                segments = transcribe(video, cache_dir=cache_dir, force=cfg["force_retranscribe"],
                                      backend=cfg["transcribe_backend"], metrics=metrics,
                                      model_size=cfg["whisper_model"])
            if cfg.get("review_transcript") and not cfg.get("work_dir"):
                segments = _pause_for_transcript_review(cfg, video, segments)
            if cfg.get("align", "off") != "off":
                # Forced alignment opt-in: só re-mede timestamps (texto intacto),
                # com fallback controlado ao Whisper. Scoring/seleção/corte
                # continuam idênticos — consomem os mesmos Segment/Word.
                from core.alignment import align_segments as _align
                segments, _astats = _align(
                    segments, audio_path=video, backend=cfg.get("align"),
                    model_size=cfg["whisper_model"], language=None)
                try:
                    metrics.stages["alignment"] = _astats.time_sec
                except Exception:
                    pass
            # Artefato reutilizável: transcript final (fonte de verdade p/ FINISH).
            # Aditivo e best-effort — nunca quebra o pipeline.
            try:
                from core import artifacts as _art
                _store = _art.store_dir(video)
                _fp_art = fingerprint(video, cfg["whisper_model"])
                _art.save_artifact(
                    _store, "transcript", _fp_art, video,
                    {"whisper_model": cfg["whisper_model"]},
                    {"segments": _art.serialize_segments(segments),
                     "transcript_hash": _art.transcript_hash(segments),
                     "meta": {"segments": len(segments),
                              "words": sum(len(s.words) for s in segments)}})
            except Exception as e:
                print(f"   ! artefato transcript não persistido ({e})")
        stage = "candidates"
        with metrics.stage("candidates"):
            from core.backends import probe_duration as _probe
            media_end = _probe(video)
            candidates = build_candidates(segments, min_dur=cfg["min_duration"],
                                          max_dur=cfg["max_duration"],
                                          media_end=media_end)
            if not candidates:
                sys.exit("Nenhum candidato encontrado (vídeo sem fala?).")
            candidates = snap_all(candidates, pad=cfg["pad"],
                                  min_dur=cfg["min_duration"],
                                  max_dur=cfg["max_duration"],
                                  media_end=media_end)
        metrics.set_counts(candidates=len(candidates))

        stage = "audio"
        with metrics.stage("audio"):
            candidates = annotate_audio(video, candidates, enable=not cfg["no_audio_features"])

        stage = "scoring"
        with metrics.stage("scoring"):
            candidates = score_candidates(
                candidates, model=cfg["model"], cache_dir=cache_dir, fingerprint=fp,
                force=cfg["force_rescore"], context=cfg["context"], examples_path=cfg["examples"],
                metrics=metrics,
            )

        stage = "selection"
        with metrics.stage("selection"):
            if cfg.get("selection_mode", "classic") == "peak":
                # Experimental: auge por candidato → clip ao redor do peak →
                # ranqueio por intensidade → títulos do auge (só selecionados).
                from core.peaks import (detect_all as _detect_all,
                                        build_clip_around_peak as _reframe,
                                        select_peak as _select_peak,
                                        retitle_peak as _retitle)
                from core.backends import probe_duration as _probe2
                _laugh_ranges = []
                if cfg.get("laughs"):
                    from core import acoustic as _ac0
                    _laugh_ranges = _ac0.load_laughs(cfg["laughs"])
                _pstats = _detect_all(
                    [c for c in candidates if not c.failed], model=cfg["model"],
                    context=cfg.get("context"), use_llm=True,
                    laughs=_laugh_ranges, events=None)
                _mend = media_end if media_end is not None else _probe2(video)
                for c in candidates:
                    if not c.failed and c.peak_start is not None:
                        _reframe(c, min_dur=cfg["min_duration"],
                                 max_dur=cfg["max_duration"], media_end=_mend)
                try:
                    metrics.stages["peaks"] = _pstats.get("time_sec", 0.0)
                except Exception:
                    pass
                selected = _select_peak(candidates, cfg["top"],
                                        min_score=cfg["min_score"],
                                        max_per_10min=cfg["max_per_10min"])
                _tstats = _retitle(selected, cfg["model"])
                print(f"   -> peak titles: {_tstats.get('retitled', 0)} do auge, "
                      f"{_tstats.get('kept', 0)} da janela")
            else:
                selected = select_top(candidates, cfg["top"], min_score=cfg["min_score"],
                                      max_per_10min=cfg["max_per_10min"])
            if not selected:
                sys.exit("Nenhum clipe passou no corte. Tente --min-score menor.")
            if cfg.get("review_titles"):
                selected = _pause_for_titles_review(cfg, video, segments, selected)
        metrics.set_counts(selected=len(selected))
        _review_status = {}
        if cfg.get("review_clips"):
            # Último filtro humano antes do burn-in: só selecionados, preview
            # limpo (sem legenda queimada), A/E/S/Q com estado persistido.
            from core import clipreview as _cr
            stage = "review"
            _session = _cr.session_for(cfg, video)
            _previews = _cr.build_previews(
                video, selected, _session, vertical=not cfg["no_vertical"])
            with metrics.stage("review"):
                _reviewed, _statuses, _rstats = _cr.run(
                    selected, video, _session, _previews)
            _review_status = {}
            _kept = iter(_reviewed)
            for _st in _statuses:
                if _st in ("accepted", "edited"):
                    _review_status[id(next(_kept))] = _st
            selected = _reviewed
            if not selected:
                sys.exit("Todos os clips foram pulados na revisão — nada a renderizar.")
            metrics.set_counts(selected=len(selected))
        warn = _title_warnings(segments, selected)
        for i, w in warn.items():
            print(f"   ! clipe {i}: title_warnings={w}")

        stage = "cutting"
        print(f"[5/5] Cortando {len(selected)} clipes com ffmpeg...")
        manifest = []
        clips_ok, clips_failed = 0, 0
        t_cut = time.time()
        from core import intel as intel_hw
        encoder = intel_hw.best_video_encoder()
        debug_dir = None
        if cfg.get("debug_captions"):
            # Só no modo debug: artefatos por clipe + transcript legível.
            # Modo normal continua limpando temporários.
            from core.video import write_human_transcript
            debug_dir = Path("debug") / Path(video).stem
            debug_dir.mkdir(parents=True, exist_ok=True)
            write_human_transcript(segments, debug_dir / "transcript.txt")
            print(f"   -> debug de legendas em: {debug_dir.resolve()}")
        _laughs = []
        if cfg.get("laughs"):
            from core import acoustic as _ac
            _laughs = _ac.load_laughs(cfg["laughs"])
            print(f"   -> {len(_laughs)} risada(s) marcada(s) em {cfg['laughs']}")
        for i, c in enumerate(selected, start=1):
            name = sanitize_filename(c.title, f"clipe_{i}")
            out = out_dir / f"{i:02d}_{name}.mp4"
            try:
                # Auditoria (originais): ffmpeg usa -y; se a saída resolvesse
                # para o próprio arquivo de entrada, o original de 17 GB seria
                # destruído. Nunca escreva por cima da entrada.
                if out.resolve() == Path(video).resolve():
                    raise RuntimeError(f"saída coincide com a entrada ({out}) — clipe ignorado")
                events = None
                if cfg.get("acoustic_captions"):
                    from core import acoustic as _ac
                    events, _stats = _ac.detect_clipping(video, c.start, c.duration)
                    if events:
                        print(f"   -> clipe {i}: {len(events)} trecho(s) estourado(s)")
                if _laughs:
                    in_clip = [e for e in _laughs
                               if e.end > c.start and e.start < c.end]
                    if in_clip:
                        print(f"   -> clipe {i}: {len(in_clip)} risada(s) marcada(s)")
                    events = (events or []) + in_clip
                cut_clip(video, c, out, vertical=not cfg["no_vertical"], captions=not cfg["no_captions"],
                         debug_dir=debug_dir, acoustic_events=events,
                         caption_mode=cfg.get("caption_mode", "phrases"),
                         title=not cfg.get("no_title", False))
            except Exception as e:
                print(f"   ! Falha clipe {i}: {e}")
                clips_failed += 1
                continue
            clips_ok += 1
            print(f"   -> {out.name}  (nota {c.score:.1f}, {c.duration:.0f}s, energy={c.energy}) — {c.title}")
            manifest.append({
                "file": out.name, "start": round(c.start, 1), "end": round(c.end, 1),
                "original_start": round(c.original_start, 1) if c.snapped else round(c.start, 1),
                "original_end": round(c.original_end, 1) if c.snapped else round(c.end, 1),
                "snapped": c.snapped, "score": c.score, "energy": c.energy,
                "speech_rate": c.speech_rate, "title": c.title, "hashtags": c.hashtags, "reason": c.reason,
                "title_warnings": warn.get(i, {}).get("title", []),
                "highlight_warnings": warn.get(i, {}).get("highlight", []),
                "acoustic_events": [{"type": e.type, "start": e.start, "end": e.end,
                                     "confidence": e.confidence} for e in (events or [])],
                **({"review": _review_status.get(id(c), "accepted")}
                   if cfg.get("review_clips") else {}),
                **(({
                    "selection_mode": "peak",
                    "window_start": round(c.window_start, 1) if c.window_start is not None else None,
                    "window_end": round(c.window_end, 1) if c.window_end is not None else None,
                    "peak_start": round(c.peak_start, 1) if c.peak_start is not None else None,
                    "peak_end": round(c.peak_end, 1) if c.peak_end is not None else None,
                    "peak_score": round(c.peak_score, 2), "peak_source": c.peak_source,
                    "peak_reason": c.peak_reason, "title_source": c.title_source,
                } if cfg.get("selection_mode", "classic") == "peak" else {})),
            })
        cut_time = round(time.time() - t_cut, 3)
        metrics.stages["cutting"] = cut_time
        metrics.set_ffmpeg(
            backend="qsv" if encoder == "h264_qsv" else "software",
            encoder=encoder, decoder=None,  # decode é software; sem sonda dedicada => null
            time_sec=cut_time, clips_ok=clips_ok, clips_failed=clips_failed,
            success=clips_failed == 0,
        )

        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nPronto! Clipes e manifest.json salvos em: {out_dir.resolve()}")
        if cache_dir:
            print(f"Cache em: {cache_dir.resolve()} (fingerprint {fp})")

        metrics.finish("success")
        metrics.print_summary()
    except SystemExit as e:
        # sys.exit() do pipeline (ex: sem candidatos, sem clipe no corte, preflight):
        # ainda assim persiste o relatório da execução.
        if e.code is None:
            msg = ""
        elif isinstance(e.code, int):
            msg = f"exit code {e.code} (ver erros de preflight acima)"
        else:
            msg = str(e.code)
        try:
            metrics.finish("failed", stage_failed=stage,
                           error={"type": "SystemExit", "message": msg[:500]})
        except Exception:
            pass
        try:
            metrics.print_summary()
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            metrics.finish("failed", stage_failed=stage,
                           error={"type": type(e).__name__, "message": str(e)[:500]})
        except Exception:
            pass
        try:
            metrics.print_summary()
        except Exception:
            pass
        raise
    except BaseException as e:
        # Auditoria (Ctrl+C): KeyboardInterrupt/SystemExit não herdado acima
        # (ex: Ctrl+C) ainda persiste o relatório como "interrupted" antes de
        # propagar. Filhos ffmpeg recebem o mesmo SIGINT do terminal; a thread
        # de métricas é daemon e morre com o processo.
        try:
            metrics.finish("interrupted", stage_failed=stage,
                           error={"type": type(e).__name__, "message": str(e)[:500]})
        except Exception:
            pass
        try:
            metrics.print_summary()
        except Exception:
            pass
        raise


def main() -> None:
    # Laboratório A/B de transcrição (sem VAD) — despacha antes do parse
    # normal porque "transcribe-lab"/"lab-compare" não são caminhos de vídeo.
    if len(sys.argv) > 1 and sys.argv[1] in (
            "transcribe-lab", "lab-compare", "transcribe-approve", "finalize",
            "caption-lab", "align-compare", "peak-compare", "finish",
            "finish-batch", "finish-status", "reset"):
        from core import translab as _lab
        import argparse as _ap
        if sys.argv[1] == "reset":
            from core import reset as _reset
            from pathlib import Path as _P
            q = _ap.ArgumentParser(
                description="Zera caches e intermediários (.cache, work, debug; "
                            "--logs inclui os relatórios metrics/*.json). "
                            "Nunca toca vídeos, modelos, cortes, código ou .env.")
            q.add_argument("--logs", action="store_true",
                           help="Apaga também os logs de execução (metrics/*.json)")
            q.add_argument("--yes", action="store_true",
                           help="Pula a confirmação (cuidado: apaga de verdade)")
            a = q.parse_args(sys.argv[2:])
            todo = _reset.targets(_P("."), include_logs=a.logs)
            if not todo:
                print("Nada a limpar (sem caches/intermediários).")
                return
            total = sum(_reset.dir_size(p) for p in todo)
            print("Será apagado:")
            for p in todo:
                print(f"  {p}/ ({_reset.dir_size(p) / 1024:.0f} KB)")
            print(f"Total: {total / 1024:.0f} KB. Finais em cortes/, vídeos, "
                  f"modelos e .env preservados.")
            if not a.yes and not _ask_confirm("Confirmar limpeza? [Y/N] "):
                sys.exit("Limpeza cancelada.")
            removed, freed = _reset.wipe(todo)
            print(f"ZERADO: {removed} item(ns), {freed / 1024:.0f} KB liberados.")
            return
        if sys.argv[1] == "caption-lab":
            from core import captionlab as _cl
            q = _ap.ArgumentParser(
                description="Compara agrupamento words vs intervals num trecho real.")
            q.add_argument("video", help="Vídeo de entrada")
            q.add_argument("--start", type=float, default=0.0, help="Início (s)")
            q.add_argument("--dur", type=float, default=30.0, help="Duração (s)")
            q.add_argument("--cache-dir", default=None, help="Cache (ex: .cache/clipper)")
            a = q.parse_args(sys.argv[2:])
            _cl.run(a.video, a.start, a.dur, cache_dir=a.cache_dir)
            return
        if sys.argv[1] == "align-compare":
            from core import alignlab as _al
            q = _ap.ArgumentParser(
                description="Compara timestamps Whisper vs forced alignment num trecho real.")
            q.add_argument("video", help="Vídeo de entrada")
            q.add_argument("--start", type=float, default=0.0, help="Início (s)")
            q.add_argument("--dur", type=float, default=30.0, help="Duração (s)")
            q.add_argument("--backend", default="whisper-refine",
                           choices=["whisper-refine", "wav2vec2"],
                           help="Backend de alignment (padrão: whisper-refine)")
            q.add_argument("--whisper-model", default=WHISPER_MODEL_SIZE,
                           help=f"Modelo ggml-* em models/ (padrão: {WHISPER_MODEL_SIZE})")
            a = q.parse_args(sys.argv[2:])
            _al.run(a.video, a.start, a.dur, backend=a.backend,
                    model_size=a.whisper_model)
            return
        if sys.argv[1] == "peak-compare":
            from core import peaklab as _pl
            from core.config import DEFAULT_MIN_SCORE as _dms
            q = _ap.ArgumentParser(
                description="Compara seleção classic vs peak nos mesmos candidatos.")
            q.add_argument("video", help="Vídeo de entrada")
            q.add_argument("--top", type=int, default=5, help="Clipes por modo")
            q.add_argument("--cache-dir", default=None, help="Cache (ex: .cache/clipper)")
            q.add_argument("--no-llm", action="store_true",
                           help="Peak heurístico (sem chamadas LLM de auge/título)")
            q.add_argument("--min-score", type=float, default=_dms, help="Score mínimo")
            q.add_argument("--model", default=None, help="Modelo NIM (padrão: env)")
            a = q.parse_args(sys.argv[2:])
            _pl.run(a.video, top=a.top, cache_dir=a.cache_dir,
                    use_llm=not a.no_llm, min_score=a.min_score, model=a.model)
            return
        if sys.argv[1] == "finish":
            # FINISH: clip existente → título/legenda/render sem descoberta
            # (sem candidatos/scoring/NMS) e sem Whisper quando há artefato.
            from core import finish as _fin
            from core.config import DEFAULT_MODEL as _dm
            from core.config import WHISPER_MODEL_SIZE as _wms
            q = _ap.ArgumentParser(
                description="Finaliza clip existente a partir da transcrição "
                            "reutilizada (Whisper só roda se faltar transcript).")
            q.add_argument("clip", help="Clip existente, o momento já escolhido "
                                        "(nunca alterado)")
            q.add_argument("--out", default="cortes", help="Pasta de saída")
            q.add_argument("--only", default="all",
                           choices=["all", "title", "captions", "render"],
                           help="Operações independentes (all = título+legenda+render; "
                                "title não gera legenda e vice-versa)")
            q.add_argument("--store-dir", default=None,
                           help="Dir de artefatos (padrão: work/<clip>/artifacts)")
            q.add_argument("--model", default=_dm, help="Modelo NIM p/ título")
            q.add_argument("--whisper-model", default=_wms,
                           help="Modelo Whisper (só se precisar transcrever)")
            q.add_argument("--title", default=None,
                           help="Título manual (salva sem Whisper, sem tocar transcript)")
            q.add_argument("--no-captions", action="store_true", help="Sem legendas")
            q.add_argument("--no-title", action="store_true", help="Sem título/hook")
            q.add_argument("--caption-mode", default="phrases",
                           choices=["words", "intervals", "phrases"])
            q.add_argument("--no-vertical", action="store_true", help="Sem 9:16")
            q.add_argument("--regenerate-title", action="store_true",
                           help="Regenera o título (transcript preservado)")
            q.add_argument("--regenerate-captions", action="store_true",
                           help="Regenera as legendas (transcript preservado)")
            q.add_argument("--force", action="store_true",
                           help="Regenera título E legenda (transcript preservado)")
            q.add_argument("--force-transcribe", action="store_true",
                           help="Retranscreve o clip (último recurso)")
            q.add_argument("--review", action="store_true",
                           help="Revisa (A/E/S/Q) antes do burn-in (só com render)")
            q.add_argument("--no-keep-artifacts", action="store_true",
                           help="Após render validado, remove title/captions/review "
                                "(transcript.json sempre preservado)")
            a = q.parse_args(sys.argv[2:])
            try:
                _res = _fin.run_finish(
                    a.clip, out_dir=a.out, only=a.only,
                    store=(Path(a.store_dir).expanduser() if a.store_dir else None),
                    model=a.model, whisper_model=a.whisper_model,
                    title_text=a.title, no_captions=a.no_captions,
                    no_title=a.no_title, caption_mode=a.caption_mode,
                    vertical=not a.no_vertical,
                    regen_title=a.regenerate_title or a.force,
                    regen_captions=a.regenerate_captions or a.force,
                    force_transcribe=a.force_transcribe, review=a.review,
                    keep_artifacts=not a.no_keep_artifacts)
            except RuntimeError as e:
                sys.exit(str(e))
            if a.only == "title":
                print(f"TÍTULO: {_res.get('title', '')}")
            return
        if sys.argv[1] == "finish-batch":
            # Lote: vários clips brutos, reutilizando tudo pronto, sem parar no erro.
            from core import finish as _fb
            from core.config import DEFAULT_MODEL as _dm2
            q = _ap.ArgumentParser(
                description="FINISH em lote sobre uma pasta de clips (reutiliza "
                            "artefatos prontos; continua após falha individual).")
            q.add_argument("clips_dir", help="Pasta com os clips (.mp4/.mkv/...)")
            q.add_argument("--out", default="cortes", help="Pasta de saída")
            q.add_argument("--only", default="all",
                           choices=["all", "title", "captions", "render"])
            q.add_argument("--model", default=_dm2, help="Modelo NIM p/ título")
            q.add_argument("--whisper-model", default=None,
                           help="Modelo Whisper (só se precisar transcrever)")
            q.add_argument("--no-captions", action="store_true")
            q.add_argument("--no-title", action="store_true")
            q.add_argument("--caption-mode", default="phrases",
                           choices=["words", "intervals", "phrases"])
            q.add_argument("--no-vertical", action="store_true")
            q.add_argument("--regenerate-title", action="store_true")
            q.add_argument("--regenerate-captions", action="store_true")
            q.add_argument("--no-keep-artifacts", action="store_true",
                           help="Após cada render validado, remove intermediários "
                                "(transcript.json sempre preservado)")
            a = q.parse_args(sys.argv[2:])
            import glob as _glob
            found = sorted(p for ext in ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.m4v")
                           for p in _glob.glob(str(Path(a.clips_dir) / ext)))
            if not found:
                sys.exit(f"Nenhum clip em {a.clips_dir}")
            print(f"[finish-batch] {len(found)} clips (só gera o que falta)...")
            res = _fb.batch_finish(
                found, out_dir=a.out, only=a.only, model=a.model,
                whisper_model=a.whisper_model, no_captions=a.no_captions,
                no_title=a.no_title, caption_mode=a.caption_mode,
                vertical=not a.no_vertical,
                regen_title=a.regenerate_title,
                regen_captions=a.regenerate_captions,
                keep_artifacts=not a.no_keep_artifacts)
            ok = sum(1 for r in res if r["ok"])
            for r in res:
                tag = "OK " if r["ok"] else "FALHA"
                print(f"  [{tag}] {Path(r['clip']).name}"
                      + (f" → {r.get('title', '')}" if r["ok"] else f": {r.get('error')}"))
            print(f"[finish-batch] {ok}/{len(res)} prontos.")
            if ok != len(res):
                sys.exit(f"{len(res) - ok} clip(s) falharam (ver acima).")
            return
        if sys.argv[1] == "finish-status":
            # Status barato: nunca Whisper/LLM/ffmpeg, só lê envelopes + stat.
            from core import finish as _fs
            from core.config import DEFAULT_MODEL as _dm3
            q = _ap.ArgumentParser(
                description="Status dos artefatos por clip "
                            "([✓] transcript/title/captions + veredito).")
            q.add_argument("clips", nargs="+", help="Clips e/ou pastas")
            q.add_argument("--out", default=None, help="Pasta de finais p/ RENDERED")
            q.add_argument("--store-dir", default=None, help="Dir de artefatos")
            q.add_argument("--model", default=_dm3)
            q.add_argument("--caption-mode", default="phrases",
                           choices=["words", "intervals", "phrases"])
            q.add_argument("--no-vertical", action="store_true")
            a = q.parse_args(sys.argv[2:])
            targets = []
            for t in a.clips:
                p = Path(t)
                if p.is_dir():
                    import glob as _glob2
                    targets += sorted(
                        x for ext in ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.m4v")
                        for x in _glob2.glob(str(p / ext)))
                else:
                    targets.append(str(p))
            for t in targets:
                st = _fs.clip_status(t, store=a.store_dir, out_dir=a.out,
                                     model=a.model, caption_mode=a.caption_mode,
                                     vertical=not a.no_vertical)
                marks = " ".join(
                    f"[{'✓' if st['states'].get(k) == 'ready' else ' '}] {k}"
                    for k in ("transcript", "title", "captions"))
                print(f"{Path(t).name:40s} {marks}  → {st['verdict']}")
            return
        if sys.argv[1] == "transcribe-approve":
            q = _ap.ArgumentParser(
                description="Valida work/<video>/transcription/words.txt editado e aprova.")
            q.add_argument("session", help="Dir transcription da sessão (work/<video>/transcription)")
            a = q.parse_args(sys.argv[2:])
            from core import review as _rev
            from pathlib import Path as _P
            session = _P(a.session)
            edited = _rev.parse_words_txt(session / "words.txt")
            final = _rev.approve(session, edited)
            print(f"TRANSCRIÇÃO APROVADA ({len(edited)} segmentos): {final}")
            return
        if sys.argv[1] == "finalize":
            q = _ap.ArgumentParser(
                description="Finaliza a sessão: preserva vídeo final + transcrição "
                            "aprovada e remove intermediários (só com confirmação).")
            q.add_argument("session", help="Dir da sessão (work/<video>)")
            q.add_argument("--out", required=True, help="Pasta dos clipes finais")
            q.add_argument("--yes", action="store_true",
                           help="Pula a confirmação (cuidado: remove intermediários)")
            a = q.parse_args(sys.argv[2:])
            from core import review as _rev
            from pathlib import Path as _P
            import shutil as _sh
            session, out = _P(a.session), _P(a.out)
            approved = session / "transcription" / "transcript.json"
            _rev.require_approved(approved)  # nunca apaga antes da aprovação final
            if not a.yes:
                if not _ask_confirm(
                        f"Tem certeza que deseja finalizar?\n"
                        f"Isso removerá os intermediários em {session}.\n"
                        f"Finais em {out} + cópia da transcrição aprovada serão mantidos.\n[Y/N] "):
                    sys.exit("Finalização cancelada.")
            out.mkdir(parents=True, exist_ok=True)
            (out / "approved-transcript.json").write_text(
                approved.read_text(encoding="utf-8"), encoding="utf-8")
            _sh.rmtree(session, ignore_errors=True)
            print(f"FINALIZADO: intermediários removidos; finais em {out.resolve()}")
            return
        if sys.argv[1] == "transcribe-lab":
            q = _ap.ArgumentParser(description="Laboratório A/B de transcrição (sem VAD).")
            q.add_argument("video", help="Vídeo de entrada")
            q.add_argument("--audio", default="original",
                           choices=["original", "normalize", "clean",
                                    "compressed", "denoised", "declipped"],
                           help="Áudio só p/ transcrição (padrão: original)")
            q.add_argument("--mode", default="chunks", choices=["chunks", "global"],
                           help="Chunks de 30s ou áudio inteiro (padrão: chunks)")
            q.add_argument("--context", type=float, default=0.0,
                           help="Segundos extras decodificados por chunk (só chunks, padrão: 0)")
            q.add_argument("--whisper-model", default=WHISPER_MODEL_SIZE,
                           help=f"Modelo ggml-* em models/ (padrão: {WHISPER_MODEL_SIZE})")
            q.add_argument("--start", type=float, default=0.0, help="Início do trecho (s)")
            q.add_argument("--dur", type=float, default=None, help="Duração do trecho (s)")
            q.add_argument("--temp", type=float, default=None,
                           help="Temperatura de decodificação -tp (experimento; padrão whisper)")
            q.add_argument("--best-of", type=int, default=None,
                           help="Best-of -bo (experimento; padrão whisper)")
            a = q.parse_args(sys.argv[2:])
            _lab.run_experiment(a.video, audio=a.audio, mode=a.mode,
                                context_sec=a.context, model_size=a.whisper_model,
                                start=a.start, dur=a.dur,
                                temp=a.temp, best_of=a.best_of)
        else:
            q = _ap.ArgumentParser(description="Compara dois experimentos do lab.")
            q.add_argument("exp_a", help="Dir do experimento A (ex: debug/transcription-lab/experiment-001)")
            q.add_argument("exp_b", help="Dir do experimento B")
            a = q.parse_args(sys.argv[2:])
            _lab.compare_experiments(a.exp_a, a.exp_b)
        return
    args = parse_cli()
    if not args.video:
        from core import tui as tui_mod
        cfg = tui_mod.run()
        if cfg is None:
            return
        run_pipeline(cfg)
        return
    run_pipeline(config_from_args(args))


if __name__ == "__main__":
    main()
