"""backends.py — backends de transcrição que usam a Intel Arc B580.

- audio_chunks(): extração em chunks via ffmpeg (streaming, nunca o vídeo todo na RAM).
- transcribe_openvino(): Whisper via optimum-intel + OpenVINO device GPU.
- transcribe_vulkan(): Whisper via binário whisper.cpp (build Vulkan).

Cada backend tenta 1x e levanta RuntimeError(motivo) se falhar — o
orquestrador (transcribe.py) faz o fallback explícito. Exceção única:
chunks com loop alucinatório detectado (core/quality) são re-decodificados
1x com WHISPER_RETRY_MODEL — troca só se limpar a sinalização.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path
from .models import Word, Segment
from .video import is_special_token as _is_special, clean_caption_text as _clean_text
from .config import CLIPPER_TRANSCRIBE_AUDIO_FILTER, WHISPER_RETRY_MODEL

CHUNK_SECONDS = 30  # janela nativa do Whisper; 30s a 16kHz mono = ~1MB por chunk
SAMPLE_RATE = 16000


def probe_duration(video_path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def audio_chunks(video_path: str, chunk_sec: int = CHUNK_SECONDS,
                 audio_filter: str | None = None):
    """Gera paths de wav temporários (16kHz mono), 1 por vez.

    Uso: for wav, offset in audio_chunks(...): ...; wav.unlink()
    Nunca carrega o vídeo/áudio inteiro na RAM. audio_filter (ex: loudnorm)
    aplica-se SOMENTE a estes chunks de transcrição — o áudio do vídeo
    final nunca passa por aqui.
    """
    duration = probe_duration(video_path) or 0
    # fallback: se duração desconhecida, extrai em blocos até o ffmpeg retornar vazio
    tmpdir = Path(tempfile.mkdtemp(prefix="clipper_audio_"))
    try:
        start = 0.0
        idx = 0
        while True:
            if duration and start >= duration:
                break
            wav = tmpdir / f"chunk_{idx:04d}.wav"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", str(start), "-t", str(chunk_sec),
                   "-i", video_path]
            if audio_filter:
                cmd += ["-af", audio_filter]
            cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1",
                    "-c:a", "pcm_s16le", str(wav)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not wav.exists() or wav.stat().st_size < 1000:
                if wav.exists():
                    wav.unlink()
                break
            yield wav, start
            wav.unlink(missing_ok=True)
            start += chunk_sec
            idx += 1
            if not duration and idx > 2000:  # trava de segurança (~16h)
                break
    finally:
        # limpa resíduos (chunks já removidos 1 a 1; remove o dir)
        try:
            for leftover in tmpdir.glob("*.wav"):
                leftover.unlink()
            tmpdir.rmdir()
        except Exception:
            pass


def _even_words(text: str, start: float, end: float) -> list[Word]:
    """Fallback honesto: distribui palavras uniformemente no intervalo."""
    tokens = text.split()
    if not tokens or end <= start:
        return []
    dur = (end - start) / len(tokens)
    return [Word(t, start + i * dur, start + (i + 1) * dur) for i, t in enumerate(tokens)]


def _say(progress, text: str) -> None:
    if progress is not None:
        progress.note(text)
    else:
        print(text)


def _warn(progress, text: str) -> None:
    if progress is not None:
        progress.warn(text)
    else:
        print(text)


def _detail(progress, text: str) -> None:
    if progress is not None:
        progress.detail(text)
    else:
        print(text)


def transcribe_openvino(video_path: str, model_size: str = "medium",
                        language: str | None = None,
                        progress=None) -> list[Segment]:
    """Whisper via optimum-intel no device GPU (B580). Levanta RuntimeError se falhar."""
    try:
        import openvino as ov
    except Exception as e:
        raise RuntimeError(f"pacote openvino não instalado ({e})")
    try:
        devs = list(ov.Core().available_devices)
    except Exception as e:
        raise RuntimeError(f"OpenVINO não enumera devices ({e})")
    if "GPU" not in devs:
        raise RuntimeError(f"OpenVINO sem device GPU (visíveis: {devs}) — instale o pacote intel-level-zero (dnf install -y intel-level-zero)")
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
        from transformers import WhisperProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(f"optimum-intel/transformers ausentes ({e}) — pip install optimum-intel transformers")

    model_id = f"openai/whisper-{model_size}"
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device="GPU")
    except Exception as e:
        raise RuntimeError(f"falha ao carregar/exportar {model_id} no GPU ({e})")

    import numpy as np
    segments: list[Segment] = []
    try:
        import soundfile as sf
        for wav, offset in audio_chunks(video_path):
            if progress is not None:
                progress.adv("transcribe", 1, f"chunk {offset:.0f}s")
            audio, sr = sf.read(str(wav))
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            out = model.generate(**inputs, return_timestamps=True)
            text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            words = _even_words(text, offset, offset + CHUNK_SECONDS)
            segments.append(Segment(text=text, start=offset,
                                    end=offset + CHUNK_SECONDS, words=words))
    except Exception as e:
        raise RuntimeError(f"erro de inferência OpenVINO GPU ({e})")
    if not segments:
        raise RuntimeError("OpenVINO GPU não gerou segmentos (áudio vazio?)")
    return segments


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _whisper_cpp_bin() -> str | None:
    import shutil
    found = shutil.which("whisper-cpp") or shutil.which("whisper-cli")
    if found:
        return found
    # Build local do projeto (thirdparty/whisper.cpp, GGML_VULKAN=1): funciona
    # sem instalar nada no sistema. Ver README § Intel Arc.
    for name in ("whisper-cli", "whisper-cpp"):
        cand = _project_root() / "thirdparty" / "whisper.cpp" / "build" / "bin" / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _whisper_cmd(binary: str, model: Path, wav: Path, stem: Path,
                 language: str | None,
                 extra: list[str] | None = None) -> list[str]:
    """Comando whisper-cli. Puro e testável. SEM VAD por decisão experimental
    (docs/20260919_0813_vad-experiment.md + auditoria 2026-09: VAD adianta onset, colapsa
    caudas e apaga ~24 s de fala em gameplay) — nenhuma flag -vm/--vad aqui."""
    cmd = [binary, "-m", str(model), "-f", str(wav), "-ojf",
           "-of", str(stem), "-l", language or "auto", "-nt"]
    if extra:
        cmd += list(extra)
    return cmd


def _resolve_model_file(model_size: str) -> Path | None:
    for cand in (Path(f"models/ggml-{model_size}.bin"),
                 Path.home() / ".cache" / "whisper" / f"ggml-{model_size}.bin"):
        if cand.exists():
            return cand
    return None


def _parse_transcription(data: dict, offset: float) -> list[Segment]:
    """JSON whisper-cli → Segments. Regras idênticas às históricas do pipeline
    (tokens de controle filtrados, texto limpo, offsets ms→s + offset)."""
    segments: list[Segment] = []
    for tr in data.get("transcription", []):
        # Filtra na origem: tokens de controle ([eot], [sot], [_EOT_]...)
        # nunca viram Word; o texto do segmento é limpo do que restar.
        raw_tokens = tr.get("tokens", []) or []
        words = []
        for w in raw_tokens:
            # Preserva o texto cru (espaço à esquerda = fronteira de
            # palavra p/ smart_join); descarta vazios e especiais.
            text = w.get("text") or ""
            if not text.strip() or _is_special(text):
                continue
            words.append(Word(
                text,
                offset + float(w.get("offsets", {}).get("from", 0)) / 1000.0,
                offset + float(w.get("offsets", {}).get("to", 0)) / 1000.0))
        text = _clean_text((tr.get("text") or "").strip())
        s = offset + float(tr.get("offsets", {}).get("from", 0)) / 1000.0
        e = offset + float(tr.get("offsets", {}).get("to", 0)) / 1000.0
        if not words:
            words = _even_words(text, s, e)
        segments.append(Segment(text=text, start=s, end=e, words=words))
    return segments


def _decode_chunk(binary: str, model: Path, wav: Path, language: str | None,
                  offset: float) -> list[Segment]:
    """1 chunk → Segments. Levanta RuntimeError se o whisper falhar."""
    stem = wav.with_suffix("")
    out_json = stem.with_suffix(".json")
    # whisper.cpp atual: JSON sai em <input>.wav.json (ou <prefix>.json
    # com -of); tokens reais só com -ojf; flag -nc não existe mais.
    cmd = _whisper_cmd(binary, model, wav, stem, language)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"whisper.cpp rc={r.returncode}: {(r.stderr or '')[:200]}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    out_json.unlink(missing_ok=True)
    return _parse_transcription(data, offset)


def transcribe_vulkan(video_path: str, model_size: str = "medium",
                      language: str | None = None,
                      progress=None) -> list[Segment]:
    """Whisper via binário whisper.cpp (build Vulkan usa a B580). Levanta RuntimeError se falhar."""
    binary = _whisper_cpp_bin()
    if not binary:
        raise RuntimeError("binário whisper-cpp/whisper-cli não está no PATH "
                           "nem em thirdparty/whisper.cpp/build/bin (ver README § Intel Arc)")
    # modelo ggml esperado em models/ggml-<size>.bin (convenção whisper.cpp)
    model = Path(f"models/ggml-{model_size}.bin")
    if not model.exists():
        alt = Path.home() / ".cache" / "whisper" / f"ggml-{model_size}.bin"
        if alt.exists():
            model = alt
        else:
            raise RuntimeError(f"modelo {model} não encontrado — baixe de HuggingFace (ex: ggerganov/whisper.cpp)")
    segments: list[Segment] = []
    _detail(progress, "sem VAD (evidência: docs/20260919_0813_vad-experiment.md)")
    retry_model = None
    if WHISPER_RETRY_MODEL and WHISPER_RETRY_MODEL.lower() not in ("off", "none"):
        retry_model = _resolve_model_file(WHISPER_RETRY_MODEL)
        if retry_model is not None and retry_model == model:
            retry_model = None  # retry igual ao principal não adianta
        if retry_model is not None:
            _detail(progress, f"retry anti-alucinação: {retry_model.name} em chunks sinalizados")
    n_flagged = n_recovered = 0
    try:
        audio_filter = CLIPPER_TRANSCRIBE_AUDIO_FILTER or None
        if audio_filter:
            _detail(progress, f"filtro de áudio p/ transcrição: {audio_filter}")
        for wav, offset in audio_chunks(video_path, audio_filter=audio_filter):
            if progress is not None:
                progress.adv("transcribe", 1, f"chunk {offset:.0f}s")
            chunk_segs = _decode_chunk(binary, model, wav, language, offset)
            if retry_model is not None:
                from . import quality as _q
                text = " ".join(s.text for s in chunk_segs)
                if _q.check(text)["flagged"]:
                    n_flagged += 1
                    try:
                        alt = _decode_chunk(binary, retry_model, wav, language, offset)
                        alt_text = " ".join(s.text for s in alt)
                        if _q.choose_retry(True, _q.check(alt_text)["flagged"]) == "retry":
                            chunk_segs = alt
                            n_recovered += 1
                            _detail(progress,
                                    f"chunk {offset:.0f}s: alucinação limpa pelo retry "
                                    f"({retry_model.name})")
                        else:
                            _detail(progress,
                                    f"chunk {offset:.0f}s: retry manteve original "
                                    f"(ambos sinalizados)")
                    except RuntimeError as e:
                        _warn(progress,
                              f"chunk {offset:.0f}s: retry falhou ({e}) — mantém original")
            segments.extend(chunk_segs)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"erro whisper.cpp Vulkan ({e})")
    if n_flagged:
        _warn(progress,
              f"anti-alucinação: {n_flagged} chunk(s) sinalizado(s), "
              f"{n_recovered} recuperado(s)")
    if not segments:
        raise RuntimeError("whisper.cpp não gerou segmentos (áudio vazio?)")
    return segments
