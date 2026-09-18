"""system.py — detecta specs do host e calcula limites seguros.

SRP: só leitura de hardware + cálculo. Não altera sistema, só retorna valores
para serem aplicados via env/args das libs (CTranslate2, ffmpeg, OpenMP).
"""
import os
import re
from pathlib import Path


def _cpu_count() -> int:
    return os.cpu_count() or 4


def _ram_gb() -> int:
    # Tenta psutil, depois /proc/meminfo, depois fallback
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1024**3)
    except Exception:
        pass
    try:
        text = Path("/proc/meminfo").read_text()
        m = re.search(r"MemTotal:\s+(\d+)\s+kB", text)
        if m:
            return int(int(m.group(1)) / 1024**2)
    except Exception:
        pass
    return 16  # fallback otimista


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s+:\s+(.+)", text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return "unknown"


def _is_weak_cpu(model: str) -> bool:
    # Heurística simples: APUs e i3/Ryzen 3 antigos são fracos
    weak_tokens = ("4600g", "5600g", "3400g", "3200g", "2200g", "i3-", "celeron", "pentium", "atom")
    low = model.lower()
    return any(t in low for t in weak_tokens)


def detect_specs() -> dict:
    cpu = _cpu_count()
    ram = _ram_gb()
    model = _cpu_model()
    weak = _is_weak_cpu(model)
    return {"cpu_threads_logical": cpu, "ram_gb": ram, "cpu_model": model, "weak": weak}


def safe_limits(specs: dict | None = None) -> dict:
    """Calcula limites seguros sem mexer no sistema (só retorna números).

    Regra KISS:
    - máquina fraca (<=12 threads ou <=8GB ou APU 4600G): reserva 2/3 dos recursos
    - média (<=16GB): reserva 1/2
    - forte (>32GB): pode usar 1/2 com folga
    Usuário pode sempre sobrescrever via env CLIPPER_*.
    """
    if specs is None:
        specs = detect_specs()
    cpu = specs["cpu_threads_logical"]
    ram = specs["ram_gb"]
    weak = specs["weak"]

    # --- CPU threads ---
    if weak or ram <= 8 or cpu <= 8:
        # PC fraco: usa 1/3, mínimo 2
        cpu_threads = max(2, cpu // 3)
        ffmpeg_threads = max(1, cpu_threads // 2)
        ram_limit = max(3, min(ram - 2, 6)) if ram > 4 else 2
    elif ram <= 16 or cpu <= 12:
        # Seu caso: 12T / 14GB → 4 threads, 2 ffmpeg, 6GB
        cpu_threads = max(2, cpu // 3)
        ffmpeg_threads = max(2, cpu_threads // 2)
        ram_limit = 6
    else:
        # Forte: 16T+ / 32GB+
        cpu_threads = max(4, cpu // 2)
        ffmpeg_threads = max(2, cpu_threads // 2)
        ram_limit = min(8, max(4, ram // 3))

    # Whisper: int8 em CPU é sempre mais leve; medium só se RAM >=8GB
    whisper_model = os.environ.get("WHISPER_MODEL_SIZE", "small" if ram <= 6 else "medium")
    device = os.environ.get("CLIPPER_DEVICE", "cpu")
    compute = os.environ.get("CLIPPER_COMPUTE_TYPE", "int8")

    return {
        "cpu_threads": cpu_threads,
        "ffmpeg_threads": ffmpeg_threads,
        "ram_limit_gb": ram_limit,
        "whisper_model": whisper_model,
        "device": device,
        "compute": compute,
        "specs": specs,
    }


def format_report(limits: dict) -> str:
    s = limits["specs"]
    return (
        f"   specs: CPU={s['cpu_model']} ({s['cpu_threads_logical']} threads) "
        f"RAM={s['ram_gb']}GB fraco={s['weak']}\n"
        f"   limites seguros: cpu_threads={limits['cpu_threads']} "
        f"ffmpeg_threads={limits['ffmpeg_threads']} ram_limit={limits['ram_limit_gb']}GB "
        f"whisper={limits['whisper_model']} device={limits['device']} compute={limits['compute']}"
    )
