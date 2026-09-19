"""metrics.py — sistema leve de métricas por execução de vídeo.

SRP: só coleta/agrega/persiste métricas. Sem lógica de negócio do pipeline.
KISS: agregados escalares, sem armazenar séries temporais.
YAGNI: sem relatório global/agregação aqui — cada JSON é uma execução.

Uso (clipper.py):
    m = ExecutionMetrics(video_path, args_dict)
    m.start()
    with m.stage("transcribe"):
        ...
    m.set_transcription(...)
    path = m.finish("success")   # ou finish("failed", stage, exc)
    m.print_summary()

Regras:
- Nunca carrega/duplica o vídeo: tamanho via stat, duração/codec via ffprobe.
- Monitoramento leve: 1 thread, 1 amostra/2s, só agregados (média/pico).
- Valor indisponível => None (serializado como null). Nunca inventar valores.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1
SAMPLE_INTERVAL_SEC = 2.0
DEFAULT_METRICS_DIR = "metrics"


# -- formatação --

def human_bytes(n: int | None) -> str | None:
    if n is None:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} PB"


def format_hms(sec: float | None) -> str | None:
    if sec is None:
        return None
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_stem(name: str, limit: int = 40) -> str:
    stem = Path(name).stem or "video"
    stem = re.sub(r"[^\w\-.áéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ]+", "_", stem).strip("._")
    return (stem or "video")[:limit]


def _short_gpu(name: str | None) -> str | None:
    """'Intel Corporation Battlemage G21 [Arc B580] [8086:e20b]' -> 'Intel Arc B580'."""
    if not name:
        return None
    m = re.search(r"\[(Arc[^\]]*)\]", name)
    if m:
        return f"Intel {m.group(1)}"
    return name[:27]


# -- sonda do vídeo (só metadata, nunca carrega o conteúdo) --

def probe_video(video_path: str) -> dict:
    """Retorna metadata do vídeo via stat + ffprobe. Tudo pode ser None."""
    info: dict = {
        "name": Path(video_path).name,
        "path": str(Path(video_path).expanduser()),
        "size_bytes": None,
        "size_human": None,
        "duration_sec": None,
        "duration_hms": None,
        "resolution": None,
        "width": None,
        "height": None,
        "codec": None,
        "fps": None,
    }
    try:
        info["size_bytes"] = os.stat(video_path).st_size
        info["size_human"] = human_bytes(info["size_bytes"])
    except Exception:
        pass
    if not shutil.which("ffprobe"):
        return info
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name,avg_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        if streams:
            s = streams[0]
            info["width"] = s.get("width")
            info["height"] = s.get("height")
            info["codec"] = s.get("codec_name")
            if info["width"] and info["height"]:
                info["resolution"] = f"{info['width']}x{info['height']}"
            try:
                num, den = (s.get("avg_frame_rate") or "0/0").split("/")
                if float(den or 0):
                    info["fps"] = round(float(num) / float(den), 3)
            except Exception:
                pass
        try:
            info["duration_sec"] = round(float(fmt.get("duration")), 3)
            info["duration_hms"] = format_hms(info["duration_sec"])
        except Exception:
            pass
    except Exception:
        pass
    return info


def describe_gpu() -> dict:
    """Snapshot da GPU sem side-effect. Utilização só se barata de obter."""
    out = {"detected": None, "name": None, "openvino_devices": []}
    try:
        from . import intel as intel_hw
        out["detected"] = intel_hw.gpu_name() or None
        out["name"] = out["detected"]
        out["openvino_devices"] = intel_hw.openvino_devices()
    except Exception:
        pass
    return out


def _nvidia_smi_sample() -> tuple[float | None, float | None]:
    """(util_pct, mem_used_mb) ou (None, None). 1 chamada curta, sem retry."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return None, None
        parts = line[0].split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None, None


# -- sampler leve: 1 thread, agrega, não guarda amostras --

class ResourceSampler(threading.Thread):
    """Amostra CPU/RAM/GPU a cada SAMPLE_INTERVAL_SEC; guarda só agregados."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_SEC):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.cpu_sum = 0.0
        self.cpu_peak: float | None = None
        self.samples = 0
        self.rss_start_mb: float | None = None
        self.rss_peak_mb: float | None = None
        self.rss_end_mb: float | None = None
        self.available_mb: float | None = None
        self.gpu_sum: float | None = None
        self.gpu_peak: float | None = None
        self.gpu_samples = 0
        self.vram_peak_mb: float | None = None
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None

    def _rss_mb(self) -> float | None:
        if self._proc is None:
            return None
        try:
            return round(self._proc.memory_info().rss / 1024**2, 1)
        except Exception:
            return None

    def run(self) -> None:
        if self._proc is not None:
            try:
                self._proc.cpu_percent(interval=None)  # prime
            except Exception:
                pass
        rss = self._rss_mb()
        self.rss_start_mb = rss
        self.rss_peak_mb = rss
        while not self._stop.wait(self.interval):
            try:
                if self._proc is not None:
                    cpu = float(self._proc.cpu_percent(interval=None))
                    self.cpu_sum += cpu
                    self.cpu_peak = cpu if self.cpu_peak is None else max(self.cpu_peak, cpu)
                    self.samples += 1
                    rss = self._rss_mb()
                    if rss is not None:
                        self.rss_peak_mb = max(self.rss_peak_mb or 0, rss)
                        self.rss_end_mb = rss
                    try:
                        import psutil
                        self.available_mb = round(psutil.virtual_memory().available / 1024**2, 1)
                    except Exception:
                        pass
                util, vram = _nvidia_smi_sample()
                if util is not None:
                    self.gpu_sum = (self.gpu_sum or 0.0) + util
                    self.gpu_peak = util if self.gpu_peak is None else max(self.gpu_peak, util)
                    self.gpu_samples += 1
                if vram is not None:
                    self.vram_peak_mb = vram if self.vram_peak_mb is None else max(self.vram_peak_mb, vram)
            except Exception:
                continue

    def stop(self) -> None:
        self._stop.set()

    def cpu_avg(self) -> float | None:
        if not self.samples:
            return None
        return round(self.cpu_sum / self.samples, 1)


# -- coletor por execução --

class ExecutionMetrics:
    """Fotografia completa de UMA execução de vídeo."""

    def __init__(self, video_path: str, args: dict | None = None,
                 metrics_dir: str | Path = DEFAULT_METRICS_DIR):
        self.video_path = video_path
        self.args = dict(args or {})
        self.metrics_dir = Path(metrics_dir)
        self.started_wall = time.time()
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.video = probe_video(video_path)
        self.gpu_info = describe_gpu()
        self.stages: dict[str, float] = {}
        self.retries_total = 0
        self.transcription: dict = {
            "model": None, "backend_requested": None, "backend_used": None,
            "device": None, "gpu": None, "time_sec": None, "time_hms": None,
            "rtf": None, "segments": None, "fallback": False,
            "tried_backends": [], "error": None,
        }
        self.nvidia_api: dict = {
            "model": None, "requests": 0, "successes": 0, "failures": 0,
            "retries": 0, "total_time_sec": None, "avg_latency_sec": None,
            "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "cost": None,
        }
        self.ffmpeg: dict = {
            "backend": None, "encoder": None, "decoder": None,
            "time_sec": None, "time_hms": None,
            "clips_ok": 0, "clips_failed": 0, "success": None,
        }
        self.counts = {"candidates": None, "selected": None}
        self._status = "success"
        self._stage_failed: str | None = None
        self._error: dict | None = None
        self._sampler = ResourceSampler()
        self._process_cpu_sec: float | None = None
        self.report_path: Path | None = None

    def start(self) -> "ExecutionMetrics":
        self._sampler.start()
        return self

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        try:
            yield self
        finally:
            self.stages[name] = round(time.time() - t0, 3)

    def add_retries(self, n: int) -> None:
        self.retries_total += max(0, int(n))

    # -- setters chamados pelos módulos instrumentados --

    def set_transcription(self, **kw) -> None:
        for k, v in kw.items():
            if k in self.transcription:
                self.transcription[k] = v
        t = self.transcription.get("time_sec")
        if t is not None:
            self.transcription["time_hms"] = format_hms(t)
            dur = self.video.get("duration_sec")
            if dur:
                try:
                    self.transcription["rtf"] = round(float(t) / float(dur), 4)
                except Exception:
                    pass

    def set_nvidia(self, **kw) -> None:
        for k, v in kw.items():
            if k in self.nvidia_api:
                self.nvidia_api[k] = v

    def set_ffmpeg(self, **kw) -> None:
        for k, v in kw.items():
            if k in self.ffmpeg:
                self.ffmpeg[k] = v
        t = self.ffmpeg.get("time_sec")
        if t is not None:
            self.ffmpeg["time_hms"] = format_hms(t)

    def set_counts(self, **kw) -> None:
        self.counts.update(kw)

    def _finish_sampler(self) -> None:
        self._sampler.stop()
        self._sampler.join(timeout=5)
        try:
            import psutil
            p = psutil.Process()
            times = p.cpu_times()
            self._process_cpu_sec = round(times.user + times.system, 2)
        except Exception:
            self._process_cpu_sec = None

    def build_report(self, status: str, stage_failed: str | None = None,
                     error: dict | None = None) -> dict:
        total = round(time.time() - self.started_wall, 3)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        sampler = self._sampler
        gpu_avg = (round(sampler.gpu_sum / sampler.gpu_samples, 1)
                   if sampler.gpu_samples and sampler.gpu_sum is not None else None)
        return {
            "schema_version": SCHEMA_VERSION,
            "video": self.video,
            "execution": {
                "started_at": self.started_at,
                "finished_at": finished_at,
                "duration_sec": total,
                "duration_hms": format_hms(total),
                "status": status,
                "stage_failed": stage_failed,
                "error": error,
                "retries_total": self.retries_total,
                "stages_sec": dict(self.stages),
                "args": self.args,
            },
            "transcription": self.transcription,
            "gpu": {
                "detected": self.gpu_info.get("detected"),
                "openvino_devices": self.gpu_info.get("openvino_devices") or [],
                "backend": self.transcription.get("backend_used"),
                "avg_util_pct": gpu_avg,
                "max_util_pct": sampler.gpu_peak,
                "vram_peak_mb": sampler.vram_peak_mb,
            },
            "cpu": {
                "threads_configured": self.args.get("cpu_threads"),
                "avg_pct": sampler.cpu_avg(),
                "peak_pct": sampler.cpu_peak,
                "process_cpu_sec": self._process_cpu_sec,
            },
            "ram": {
                "start_mb": sampler.rss_start_mb,
                "peak_mb": sampler.rss_peak_mb,
                "end_mb": sampler.rss_end_mb if sampler.rss_end_mb is not None else sampler.rss_peak_mb,
                "available_mb": sampler.available_mb,
            },
            "ffmpeg": self.ffmpeg,
            "nvidia_api": self.nvidia_api,
            "pipeline": {
                "candidates": self.counts.get("candidates"),
                "selected": self.counts.get("selected"),
                "clips_ok": self.ffmpeg.get("clips_ok"),
                "clips_failed": self.ffmpeg.get("clips_failed"),
            },
        }

    def finish(self, status: str = "success", stage_failed: str | None = None,
               error: dict | None = None) -> Path:
        """Para o sampler, persiste o JSON e retorna o caminho. Nunca levanta."""
        self._status = status
        self._stage_failed = stage_failed
        self._error = error
        self._finish_sampler()
        report = self.build_report(status, stage_failed, error)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        stem = _safe_stem(self.video.get("name") or "video")
        uid = uuid.uuid4().hex[:6]
        self.report_path = self.metrics_dir / f"{ts}_{stem}_{uid}.json"
        try:
            self.report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            fallback = self.metrics_dir / f"{ts}_{stem}_{uid}.fallback.json"
            try:
                fallback.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                self.report_path = fallback
            except Exception:
                pass
        self._last_report = report
        return self.report_path

    # -- resumo legível no terminal --

    def print_summary(self) -> None:
        r = getattr(self, "_last_report", None) or {}
        v = r.get("video", self.video)
        ex = r.get("execution", {})
        tr = r.get("transcription", self.transcription)
        gpu = r.get("gpu", {})
        cpu = r.get("cpu", {})
        ram = r.get("ram", {})
        ff = r.get("ffmpeg", self.ffmpeg)
        api = r.get("nvidia_api", self.nvidia_api)

        def show(x, suffix=""):
            return f"{x}{suffix}" if x is not None else "n/a"

        def show_gpu(x, suffix=""):
            # Telemetria Intel indisponível (sem nvidia-smi/intel_gpu_top):
            # n/a aqui é ausência de medidor, não de GPU (ver Backend/Device).
            return f"{x}{suffix}" if x is not None else "n/a (sem telemetria Intel)"

        status = (ex.get("status") or self._status).upper()
        backend = str(tr.get("backend_used") or "n/a").upper()
        if (tr.get("backend_used") or "") in ("vulkan", "openvino"):
            device = _short_gpu(tr.get("gpu")) or tr.get("device") or "GPU"
        else:
            device = str(tr.get("device") or "CPU").upper()
        W = 48
        bar = "╭" + "─" * W + "╮"
        mid = "├" + "─" * W + "┤"
        end = "╰" + "─" * W + "╯"

        def section(title: str) -> str:
            return f"│ {title:<{(W - 1)}}│"

        def row(label: str, value: str) -> str:
            return f"│ {label:<12} {value[:W - 15]:<{(W - 15)}}│"

        def wrapped_path(path: str) -> list[str]:
            chunks = [path[i:i + W - 2] for i in range(0, len(path), W - 2)] or [""]
            return [f"│ {c:<{(W - 1)}}│" for c in chunks]

        lines = [
            bar,
            f"│{'EXECUTION METRICS':^{W}}│",
            mid,
            row("Video:", str(v.get("name") or "?")),
            row("Size:", str(v.get("size_human") or "n/a")),
            row("Duration:", str(v.get("duration_hms") or "n/a")),
            row("Total time:", str(ex.get("duration_hms") or "n/a")),
            row("Status:", status),
            mid,
            section("TRANSCRIPTION"),
            row("Backend:", backend),
            row("Device:", str(device)),
            row("Time:", str(tr.get("time_hms") or "n/a")),
            row("RTF:", str(tr.get("rtf") if tr.get("rtf") is not None else "n/a")),
            mid,
            section("RESOURCES"),
            row("GPU usage:", show_gpu(gpu.get("avg_util_pct"), "%")),
            row("VRAM peak:", show_gpu(gpu.get("vram_peak_mb"), " MB")),
            row("CPU avg:", show(cpu.get("avg_pct"), "%")),
            row("CPU peak:", show(cpu.get("peak_pct"), "%")),
            row("RAM peak:", show(ram.get("peak_mb"), " MB")),
            mid,
            section("NVIDIA API"),
            row("Requests:", show(api.get("requests"))),
            row("Retries:", show(api.get("retries"))),
            row("Cost:", show(api.get("cost"))),
            mid,
            section("REPORT"),
            *wrapped_path(str(self.report_path)),
            end,
        ]
        print("\n".join(lines))
