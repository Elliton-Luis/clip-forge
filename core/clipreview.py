"""clipreview.py — revisão humana dos clips SELECIONADOS antes do burn-in.

Fluxo: scoring/seleção → previews limpos (SEM legenda queimada, SEM título)
em work/<stem>/clipreview/ → loop interativo A/E/S/Q (+P) → render final só
dos aceitos/editados, com legendas geradas do texto corrigido.

Regras:
- Revisa só selecionados (nunca dezenas de descartados).
- preview = cut() com captions=False, title=False: mesmo corte/áudio do final,
  sem nada queimado. Nunca toca o vídeo de entrada.
- Edição sem JSON manual: arquivo words.txt por clip
  ([sNN] [MM:SS.mmm → MM:SS.mmm] texto, timestamps ABSOLUTOS como no
  review.py) + $EDITOR ou edição manual + validação. Texto e timestamps são
  independentes; linha malformada = erro claro, nunca chute.
- Q persiste tudo (accepted/edited/skipped/pending) e sai; rerun continua de
  onde parou. Candidatos diferentes dos gravados → decisions.bak.json + fresh.
- Pular não é erro. Limpeza: previews morrem com a sessão (finalize/reset);
  nada é apagado automaticamente.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .models import Word

SESSION_SUBDIR = "clipreview"
DECISIONS_FILE = "decisions.json"
STATUSES = ("pending", "accepted", "edited", "skipped")


def session_dir(video_path: str, work_root: Path | str = Path("work")) -> Path:
    stem = re.sub(r"[^\w\-]+", "_", Path(video_path).stem).strip("_") or "video"
    d = Path(work_root) / stem / SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_for(cfg: dict, video: str) -> Path:
    """Sessão de revisão de clips: work/<stem>/clipreview (ou work_dir/...)."""
    if cfg.get("work_dir"):
        d = Path(cfg["work_dir"]) / SESSION_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d
    from .review import session_dir as _sdir
    d = _sdir(video).parent / SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def fingerprint(selected: list) -> str:
    raw = "|".join(f"{c.start:.2f}-{c.end:.2f}-{c.title}" for c in selected)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def transcript_lines(c) -> list[str]:
    """Palavras do candidato em linhas de leitura ([MM:SS.mmm] frase)."""
    words = sorted(c.words, key=lambda w: w.start)
    lines, cur, cur_start = [], [], None
    for w in words:
        txt = (w.text or "").strip()
        if not txt:
            continue
        if cur_start is None:
            cur_start = w.start
        cur.append(txt)
        if re.search(r"[.!?…]\s*$", txt) or len(cur) >= 12:
            lines.append(f"[{_fmt(cur_start)}] {' '.join(cur)}")
            cur, cur_start = [], None
    if cur:
        lines.append(f"[{_fmt(cur_start or 0.0)}] {' '.join(cur)}")
    return lines


def build_previews(video: str, selected: list, session: Path,
                   vertical: bool = True) -> list[Path]:
    """Renderiza previews limpos (sem legenda, sem título). Reaproveita os
    existentes. Retorna os paths na ordem dos selecionados."""
    from .video import cut as _cut
    paths = []
    for i, c in enumerate(selected, start=1):
        p = session / f"preview_{i:02d}.mp4"
        if not p.exists():
            print(f"   -> preview {i}/{len(selected)}: {p.name} "
                  f"([{c.start:.0f}-{c.end:.0f}], sem legenda queimada)")
            _cut(video, c, p, vertical=vertical, captions=False, title=False)
        paths.append(p)
    return paths


def _words_txt_path(session: Path, idx: int) -> Path:
    return session / f"clip_{idx:02d}_words.txt"


def dump_clip_words(c, path: Path) -> None:
    lines = ["# Edite TEXTO e TIMESTAMPS (absolutos) à vontade.",
             "# Uma linha = uma palavra. Remover a linha remove a palavra.",
             "# Formato obrigatório: [sNN] [MM:SS.mmm → MM:SS.mmm] texto",
             ""]
    for w in sorted(c.words, key=lambda x: x.start):
        lines.append(f"[s00] [{_fmt(w.start)} → {_fmt(w.end)}] {(w.text or '').strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_clip_words(c, path: Path) -> int:
    """Valida e aplica o words.txt editado ao candidato. Retorna nº de words."""
    from . import review as _rev
    segments = _rev.parse_words_txt(path)  # erro claro em linha malformada
    words: list[Word] = [w for s in segments for w in s.words]
    if not words:
        raise RuntimeError(f"{path}: nenhuma palavra (remover tudo = pule com S)")
    words.sort(key=lambda w: w.start)
    c.words = words
    c.text = " ".join(w.text.strip() for w in words if (w.text or "").strip())
    dur = max(0.5, c.end - c.start)
    c.speech_rate = round(len(words) / dur, 2)
    return len(words)


def load_decisions(session: Path):
    p = session / DECISIONS_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_decisions(session: Path, selected: list, statuses: list[str]) -> Path:
    p = session / DECISIONS_FILE
    data = {"fingerprint": fingerprint(selected),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clips": [{"index": i + 1, "status": st,
                       "start": round(c.start, 1), "end": round(c.end, 1),
                       "title": c.title, "score": c.score}
                      for i, (c, st) in enumerate(zip(selected, statuses))]}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _match_decisions(selected: list, data: dict | None) -> list[str] | None:
    if not data or data.get("fingerprint") != fingerprint(selected):
        return None
    clips = data.get("clips", [])
    if len(clips) != len(selected):
        return None
    out = []
    for d in clips:
        st = d.get("status", "pending")
        out.append(st if st in STATUSES else "pending")
    return out


def _prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        raise SystemExit("Entrada não-interativa: revisão de clips exige terminal.")


def _maybe_play(path: Path) -> None:
    opener = shutil.which("xdg-open") or shutil.which("open") or shutil.which("mpv")
    if not opener:
        print("   (sem player automático: abra o arquivo no seu player)")
        return
    try:
        subprocess.Popen([opener, str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   -> abrindo {path.name} no player...")
    except Exception as e:
        print(f"   ! não consegui abrir o player ({e}): {path}")


def _edit_clip(c, session: Path, idx: int) -> bool:
    """Fluxo de edição. True se aplicou, False se cancelado/erro (mantém tudo)."""
    path = _words_txt_path(session, idx)
    dump_clip_words(c, path)
    editor = (os.environ.get("EDITOR") or "").strip()
    print(f"\n   Edite: {path.resolve()}")
    if editor and sys.stdin.isatty():
        print(f"   Abrindo ${editor}... (salve e feche; linhas apagadas removem palavras)")
        try:
            rc = subprocess.run([editor, str(path)]).returncode
            if rc != 0:
                print("   ! editor saiu com erro — edição cancelada, nada mudou")
                return False
        except Exception as e:
            print(f"   ! falha ao abrir editor ({e}) — edite manualmente")
            _prompt("   Pressione Enter quando terminar de editar...")
    else:
        print("   Edite o arquivo (texto e timestamps absolutos), salve e volte.")
        _prompt("   Pressione Enter quando terminar de editar (Ctrl+C cancela)...")
    try:
        n = apply_clip_words(c, path)
    except RuntimeError as e:
        print(f"   ! edição inválida ({e}) — nada mudou, tente de novo ou pule")
        return False
    print(f"   -> texto corrigido ({n} palavras) — será usado no burn-in")
    return True


def _show_clip(i: int, n: int, c, preview: Path) -> None:
    bar = "━" * 50
    print(f"\n{bar}\nREVISÃO {i}/{n}\n")
    print(f"Título:\n\"{c.title or '(sem título)'}\"\n")
    extra = []
    if getattr(c, "peak_start", None) is not None:
        extra.append(f"auge {c.peak_start:.0f}-{c.peak_end:.0f}s ({c.peak_source})")
    print(f"Score: {c.score:.1f}  |  Duração: {c.duration:.1f}s  |  "
          f"Energia: {c.energy}  |  {c.speech_rate:.1f} w/s"
          + (f"  |  {' · '.join(extra)}" if extra else ""))
    print(f"\nVídeo (SEM legenda queimada):\n{preview.resolve()}\n")
    print("TRANSCRIÇÃO:\n")
    for line in transcript_lines(c):
        print(line)
    print(f"\n{bar}\n")
    print("[A] Aceitar   [E] Editar   [S] Pular   [P] Tocar vídeo   [Q] Sair")


def run(selected: list, video: str, session: Path,
        previews: list[Path] | None = None,
        input_fn=None) -> tuple[list, list[str], dict]:
    """Loop interativo. Retorna (para_renderizar, statuses, stats).
    Q persiste e levanta SystemExit (re-run continua)."""
    if not sys.stdin.isatty() and input_fn is None:
        raise SystemExit("Entrada não-interativa: --review-clips exige terminal.")
    ask = input_fn or (lambda msg: _prompt(msg).strip().lower())
    saved = _match_decisions(selected, load_decisions(session))
    if saved is None:
        old = session / DECISIONS_FILE
        if old.exists():
            bak = session / "decisions.bak.json"
            bak.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"   ! candidatos mudaram — decisões antigas em {bak.name}, recomeçando")
        statuses = ["pending"] * len(selected)
    else:
        statuses = list(saved)
        done = sum(1 for s in statuses if s != "pending")
        if done:
            print(f"   -> continuando revisão ({done}/{len(selected)} decididos)")
    if previews is None:
        previews = [session / f"preview_{i:02d}.mp4" for i in range(1, len(selected) + 1)]
    stats = {"accepted": 0, "edited": 0, "skipped": 0}
    try:
        for i, (c, preview) in enumerate(zip(selected, previews), start=1):
            if statuses[i - 1] != "pending":
                continue
            _show_clip(i, len(selected), c, preview)
            while True:
                ans = (ask("> ") or "").strip().lower()
                if ans in ("a", "aceitar"):
                    statuses[i - 1] = "accepted"
                    break
                elif ans in ("e", "editar"):
                    if _edit_clip(c, session, i):
                        statuses[i - 1] = "edited"
                        break
                elif ans in ("s", "pular", "skip"):
                    statuses[i - 1] = "skipped"
                    print("   -> descartado (sem erro, sem render)")
                    break
                elif ans in ("p", "tocar", "play"):
                    _maybe_play(preview)
                elif ans in ("q", "sair", "quit"):
                    save_decisions(session, selected, statuses)
                    raise SystemExit(
                        "Revisão interrompida — decisões salvas. "
                        "Rode novamente para continuar de onde parou.")
                else:
                    print("   Opções: [A]ceitar [E]ditar [S]pular [P]tocar [Q]sair")
            save_decisions(session, selected, statuses)
    except KeyboardInterrupt:
        save_decisions(session, selected, statuses)
        raise SystemExit("Revisão interrompida (Ctrl+C) — decisões salvas.")
    for s in statuses:
        if s in stats:
            stats[s] += 1
    to_render = [c for c, s in zip(selected, statuses) if s in ("accepted", "edited")]
    print(f"\n   -> revisão: {stats['accepted']} aceitos, {stats['edited']} editados, "
          f"{stats['skipped']} pulados — {len(to_render)} para o burn-in")
    return to_render, statuses, stats
