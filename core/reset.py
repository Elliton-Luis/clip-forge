"""reset.py — zera caches e intermediários do projeto.

Alvo: `.cache/`, `work/`, `debug/`, `metrics_test/`, `__pycache__/`,
`.pytest_cache/` + (opt-in `--logs`) os relatórios `metrics/*.json`.
NUNCA toca: vídeos, modelos, thirdparty, cortes (finais do usuário),
código-fonte, `.env`, configs. Só remove o que está na allowlist abaixo,
sempre dentro da raiz do projeto — qualquer outro caminho é recusado.
"""
import shutil
from pathlib import Path

# Diretórios intermediários (removidos por inteiro quando existem).
CACHE_DIRS = [".cache", "work", "debug", "metrics_test",
              "__pycache__", ".pytest_cache"]
# Logs vão separado: só com opt-in explícito (são o histórico de execução).
LOG_PATTERNS = ["metrics/*.json"]


def _inside_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def targets(root: Path | str = ".", include_logs: bool = False) -> list[Path]:
    """O que será apagado (só existentes). Puro e testável."""
    root = Path(root)
    out: list[Path] = []
    for name in CACHE_DIRS:
        p = root / name
        if _inside_root(root, p) and (p.is_dir() or p.is_file()):
            out.append(p)
    if include_logs:
        for pattern in LOG_PATTERNS:
            for p in sorted(root.glob(pattern)):
                if _inside_root(root, p) and p.is_file():
                    out.append(p)
    return out


def dir_size(path: Path) -> int:
    """Bytes ocupados (0 se ausente)."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def wipe(paths: list[Path]) -> tuple[int, int]:
    """Apaga a lista (vinda de targets()). Retorna (removidos, bytes)."""
    removed, freed = 0, 0
    for p in paths:
        try:
            freed += dir_size(p)
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1
        except OSError:
            pass
    return removed, freed
