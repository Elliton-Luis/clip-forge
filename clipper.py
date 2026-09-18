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
import shlex
import sys
import time
from pathlib import Path

from core.config import (
    DEFAULT_MODEL, DEFAULT_PAD_SECONDS, DEFAULT_MIN_SCORE, DEFAULT_MAX_PER_10MIN,
    CLIPPER_TRANSCRIBE_BACKEND, CLIPPER_CPU_THREADS, CLIPPER_FFMPEG_THREADS,
    WHISPER_MODEL_SIZE,
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
    p.add_argument("--cache-dir", default=None, help="Diretório de cache (ex: .cache/clipper)")
    p.add_argument("--force-retranscribe", action="store_true", help="Ignora cache transcrição")
    p.add_argument("--force-rescore", action="store_true", help="Ignora cache scores")
    p.add_argument("--pad", type=float, default=DEFAULT_PAD_SECONDS, help=f"Respiro antes/depois (padrão: {DEFAULT_PAD_SECONDS}s)")
    p.add_argument("--no-audio-features", action="store_true", help="Desativa energia de áudio")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help=f"Score mínimo (padrão: {DEFAULT_MIN_SCORE})")
    p.add_argument("--max-per-10min", type=int, default=DEFAULT_MAX_PER_10MIN, help=f"Máximo por 10min (padrão: {DEFAULT_MAX_PER_10MIN})")
    p.add_argument("--context", default=None, help="Contexto da live injetado no prompt")
    p.add_argument("--examples", default=None, help="Caminho para examples.json few-shot")
    p.add_argument("--transcribe-backend", default=CLIPPER_TRANSCRIBE_BACKEND,
                   choices=["auto", "gpu", "vulkan", "openvino", "cpu"],
                   help="Backend de transcrição (padrão: env CLIPPER_TRANSCRIBE_BACKEND ou auto; "
                        "'gpu' exige GPU e falha claramente se indisponível)")
    p.add_argument("--debug-captions", action="store_true",
                   help="Diagnóstico de legendas: preserva ASS/SRT/transcript/diagnóstico em debug/")
    return p


def parse_cli(argv=None):
    return build_parser().parse_args(argv)


# ----------------------------------------------------------------------------
# 2. Configuração (dict canônico — CLI e TUI produzem exatamente este formato)
# ----------------------------------------------------------------------------

CONFIG_FIELDS = (
    "video", "out", "top", "model", "no_vertical", "no_captions",
    "cache_dir", "force_retranscribe", "force_rescore", "pad",
    "no_audio_features", "min_score", "max_per_10min", "context",
    "examples", "transcribe_backend", "debug_captions",
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
        "cache_dir": args.cache_dir,
        "force_retranscribe": bool(args.force_retranscribe),
        "force_rescore": bool(args.force_rescore),
        "pad": args.pad,
        "no_audio_features": bool(args.no_audio_features),
        "min_score": args.min_score,
        "max_per_10min": args.max_per_10min,
        "context": args.context,
        "examples": args.examples,
        "transcribe_backend": args.transcribe_backend,
        "debug_captions": bool(args.debug_captions),
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
    if cfg["cache_dir"]:
        parts += ["--cache-dir", cfg["cache_dir"]]
    if cfg["force_retranscribe"]:
        parts.append("--force-retranscribe")
    if cfg["force_rescore"]:
        parts.append("--force-rescore")
    if cfg["pad"] != DEFAULT_PAD_SECONDS:
        parts += ["--pad", str(cfg["pad"])]
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
    return " ".join(shlex.quote(x) for x in parts)


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
        "whisper_model": WHISPER_MODEL_SIZE, "cpu_threads": CLIPPER_CPU_THREADS,
        "ffmpeg_threads": CLIPPER_FFMPEG_THREADS, "pad": cfg["pad"],
        "min_score": cfg["min_score"], "max_per_10min": cfg["max_per_10min"],
        "vertical": not cfg["no_vertical"], "captions": not cfg["no_captions"],
        "audio_features": not cfg["no_audio_features"],
    }
    metrics = ExecutionMetrics(video, run_args).start()
    stage = "preflight"
    try:
        with metrics.stage("preflight"):
            preflight(video, out_dir, cfg["top"], transcribe_backend=cfg["transcribe_backend"])

        cache_dir = Path(cfg["cache_dir"]).expanduser() if cfg["cache_dir"] else None
        fp = fingerprint(video) if cache_dir else None

        stage = "transcribe"
        with metrics.stage("transcribe"):
            segments = transcribe(video, cache_dir=cache_dir, force=cfg["force_retranscribe"],
                                  backend=cfg["transcribe_backend"], metrics=metrics)
        stage = "candidates"
        with metrics.stage("candidates"):
            candidates = build_candidates(segments)
            if not candidates:
                sys.exit("Nenhum candidato encontrado (vídeo sem fala?).")
            candidates = snap_all(candidates, pad=cfg["pad"])
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
            selected = select_top(candidates, cfg["top"], min_score=cfg["min_score"],
                                  max_per_10min=cfg["max_per_10min"])
            if not selected:
                sys.exit("Nenhum clipe passou no corte. Tente --min-score menor.")
        metrics.set_counts(selected=len(selected))

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
        for i, c in enumerate(selected, start=1):
            name = sanitize_filename(c.title, f"clipe_{i}")
            out = out_dir / f"{i:02d}_{name}.mp4"
            try:
                # Auditoria (originais): ffmpeg usa -y; se a saída resolvesse
                # para o próprio arquivo de entrada, o original de 17 GB seria
                # destruído. Nunca escreva por cima da entrada.
                if out.resolve() == Path(video).resolve():
                    raise RuntimeError(f"saída coincide com a entrada ({out}) — clipe ignorado")
                cut_clip(video, c, out, vertical=not cfg["no_vertical"], captions=not cfg["no_captions"],
                         debug_dir=debug_dir)
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
