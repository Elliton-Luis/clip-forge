"""tui.py — interface interativa de terminal (curses, só stdlib).

SRP: COLETAR configuração. Nunca executa o pipeline (isso é run_pipeline,
em clipper.py). CLI e TUI produzem o mesmo dict canônico (ver CONFIG_FIELDS).

Tudo que é testável (modelos, validação, comando equivalente, buscas) vive
em funções puras sem curses. A parte interativa (telas) usa curses apenas
para navegação; edição de texto usa input() fora do modo curses.
"""
import curses
import json
import os
from pathlib import Path

from core.config import DEFAULT_MODEL, CLIPPER_TRANSCRIBE_BACKEND

MODELS_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts", ".flv", ".mpg", ".mpeg"}
BACKENDS = ["auto", "gpu", "vulkan", "openvino", "cpu"]


# ----------------------------------------------------------------------------
# Parte pura (testável sem terminal)
# ----------------------------------------------------------------------------

def load_models(path: Path | str = MODELS_PATH) -> list[dict]:
    """Lê config/models.json. Só LLMs de texto com chat (curadoria manual)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = data.get("models", []) if isinstance(data, dict) else data
    out = [m for m in models
           if isinstance(m, dict) and m.get("id") and m.get("name")]
    if not out:
        raise ValueError(f"nenhum modelo válido em {path}")
    return out


def default_model_index(models: list[dict], default_id: str = DEFAULT_MODEL) -> int:
    for i, m in enumerate(models):
        if m["id"] == default_id:
            return i
    return 0


def list_videos(directory: str | Path = "videos") -> list[dict]:
    """Lista vídeos por nome+tamanho (stat). Nunca carrega conteúdo."""
    d = Path(directory).expanduser()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out.append({"path": str(p), "size": size})
    return out


def human_bytes(n: int) -> str:
    f = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def latest_report(metrics_dir: str | Path = "metrics") -> Path | None:
    files = sorted(Path(metrics_dir).glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def latest_transcript(cache_dir: str | Path = ".cache/clipper") -> Path | None:
    files = sorted(Path(cache_dir).glob("*.transcript.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def list_sessions(work_root: str | Path = "work") -> list[dict]:
    """Sessões de revisão em work/<video>/transcription. Puro e testável."""
    root = Path(work_root).expanduser()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        t = d / "transcription" / "transcript.json"
        if not (d.is_dir() and t.exists()):
            continue
        try:
            import json as _j
            data = _j.loads(t.read_text(encoding="utf-8"))
            segs = data.get("segments", [])
            n_words = sum(len(s.get("words", [])) for s in segs)
            status = data.get("status", "draft")
        except (OSError, ValueError):
            status, n_words = "corrompida", 0
        out.append({"name": d.name, "dir": str(d), "status": status,
                    "segments": len(segs) if isinstance(segs, list) else 0,
                    "words": n_words})
    return out


def validate_for_run(cfg: dict) -> list[str]:
    """Validação completa antes de executar. Retorna lista de erros (vazia = ok)."""
    errs = []
    if not cfg.get("video"):
        errs.append("Nenhum vídeo selecionado")
    elif not Path(cfg["video"]).expanduser().exists():
        errs.append(f"Arquivo não encontrado: {cfg['video']}")
    try:
        out = Path(cfg.get("out") or "cortes")
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errs.append(f"Pasta de saída inválida ({cfg.get('out')}): {e}")
    top = cfg.get("top")
    if not isinstance(top, int) or isinstance(top, bool) or top <= 0:
        errs.append("--top deve ser um inteiro > 0")
    try:
        pad = float(cfg.get("pad"))
        if not (0 <= pad <= 5):
            errs.append("--pad deve estar entre 0 e 5")
    except (TypeError, ValueError):
        errs.append("--pad deve ser um número entre 0 e 5")
    try:
        ms = float(cfg.get("min_score"))
        if not (0 <= ms <= 10):
            errs.append("--min-score deve estar entre 0 e 10")
    except (TypeError, ValueError):
        errs.append("--min-score deve ser um número entre 0 e 10")
    try:
        mp = int(cfg.get("max_per_10min"))
        if not (1 <= mp <= 20):
            errs.append("--max-per-10min deve estar entre 1 e 20")
    except (TypeError, ValueError):
        errs.append("--max-per-10min deve ser um inteiro entre 1 e 20")
    if not cfg.get("model"):
        errs.append("Modelo inválido (vazio)")
    if cfg.get("transcribe_backend") not in ("auto", "gpu", "vulkan", "openvino", "cpu"):
        errs.append("Backend de transcrição inválido")
    return errs


def summary_lines(cfg: dict, model_name: str = "") -> list[str]:
    rows = [
        ("Vídeo", cfg.get("video") or "—"),
        ("Saída", cfg.get("out") or "cortes/"),
        ("Modelo", model_name or cfg.get("model") or "—"),
        ("Transcrição", cfg.get("transcribe_backend") or "auto"),
        ("Clipes", str(cfg.get("top"))),
        ("Score mínimo", str(cfg.get("min_score"))),
        ("Padding", f"{cfg.get('pad')}s"),
        ("Máx/10min", str(cfg.get("max_per_10min"))),
        ("Vertical", "sim" if not cfg.get("no_vertical") else "não"),
        ("Legendas", "sim" if not cfg.get("no_captions") else "não"),
        ("Audio features", "sim" if not cfg.get("no_audio_features") else "não"),
        ("Cache", cfg.get("cache_dir") or "desligado"),
    ]
    if cfg.get("context"):
        rows.append(("Contexto", str(cfg["context"])[:40]))
    if cfg.get("examples"):
        rows.append(("Examples", str(cfg["examples"])))
    if cfg.get("force_retranscribe"):
        rows.append(("Forçar", "retranscrição"))
    if cfg.get("force_rescore"):
        rows.append(("Forçar", "novo scoring"))
    if cfg.get("debug_captions"):
        rows.append(("Debug", "legendas em debug/"))
    if cfg.get("review_transcript"):
        rows.append(("Revisão", "transcrição (pausa p/ editar)"))
    if cfg.get("review_titles"):
        rows.append(("Revisão", "títulos (pausa p/ editar)"))
    if cfg.get("work_dir"):
        rows.append(("Sessão", str(cfg["work_dir"])))
    return rows


# ----------------------------------------------------------------------------
# Parte interativa (curses)
# ----------------------------------------------------------------------------

def _curses_input(stdscr, prompt: str, current: str = "") -> str | None:
    """Edição de linha DENTRO do curses (sem endwin): Enter confirma,
    Esc cancela (mantém atual), Backspace apaga. Nunca troca o modo do
    terminal — elimina a classe inteira de bugs de ^M/modo cru."""
    h, w = stdscr.getmaxyx()
    buf = list(current)
    curses.curs_set(1)
    try:
        while True:
            stdscr.move(h - 2, 0)
            stdscr.clrtoeol()
            line = f"{prompt}: {''.join(buf)}"
            stdscr.addstr(h - 2, 2, line[:w - 4])
            stdscr.move(h - 2, min(w - 1, 2 + len(prompt) + 2 + len(buf)))
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (10, 13, curses.KEY_ENTER):
                out = "".join(buf).strip()
                return out if out != "" else current
            elif ch == 27:
                return current
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if buf:
                    buf.pop()
            elif 32 <= ch <= 126:
                if len(buf) < 200:
                    buf.append(chr(ch))
    finally:
        curses.curs_set(0)


class TUI:
    def __init__(self, stdscr, cfg: dict, models: list[dict]):
        self.stdscr = stdscr
        self.cfg = cfg
        self.models = models
        self.model_idx = default_model_index(models, cfg.get("model") or DEFAULT_MODEL)
        self.backend_idx = BACKENDS.index(cfg.get("transcribe_backend") or "auto") \
            if (cfg.get("transcribe_backend") or "auto") in BACKENDS else 0
        self.pos = 0
        self.msg = ""
        self.use_cache = bool(cfg.get("cache_dir"))

    # -- widgets de campo (ordem da tela) --
    def _fields(self):
        return [
            ("video", "Vídeo", "file"),
            ("out", "Pasta de saída", "text"),
            ("model", "Modelo de scoring", "model"),
            ("transcribe_backend", "Backend de transcrição", "backend"),
            ("top", "Quantidade de clipes", "int"),
            ("min_score", "Score mínimo (0–10)", "float"),
            ("pad", "Padding em segundos (0–5)", "float"),
            ("max_per_10min", "Máximo por 10 min (1–20)", "int"),
            ("vertical", "Vídeo vertical", "bool"),
            ("captions", "Legendas", "bool"),
            ("audio_features", "Audio features", "bool"),
            ("use_cache", "Usar cache", "bool"),
            ("cache_dir", "Diretório do cache", "text"),
            ("context", "Contexto da live (avançado)", "text"),
            ("examples", "Arquivo examples (avançado)", "text"),
            ("force_retranscribe", "Forçar retranscrição", "bool"),
            ("force_rescore", "Forçar novo scoring", "bool"),
            ("debug_captions", "Debug de legendas (avançado)", "bool"),
            ("review_transcript", "Revisar transcrição (pausa p/ editar)", "bool"),
            ("review_titles", "Revisar títulos (pausa p/ editar)", "bool"),
            ("work_dir", "Sessão de revisão (work/..., avançado)", "text"),
            ("custom_words", "Vocabulário customizado (JSON, avançado)", "text"),
            ("process", "[ PROCESSAR ]", "action"),
        ]

    def _get(self, key):
        if key == "vertical":
            return not self.cfg.get("no_vertical", False)
        if key == "captions":
            return not self.cfg.get("no_captions", False)
        if key == "audio_features":
            return not self.cfg.get("no_audio_features", False)
        if key == "use_cache":
            return self.use_cache
        if key == "model":
            return self.models[self.model_idx]["name"]
        if key == "transcribe_backend":
            return BACKENDS[self.backend_idx]
        return self.cfg.get(key, "")

    def _toggle(self, key):
        if key == "vertical":
            self.cfg["no_vertical"] = not self.cfg.get("no_vertical", False)
        elif key == "captions":
            self.cfg["no_captions"] = not self.cfg.get("no_captions", False)
        elif key == "audio_features":
            self.cfg["no_audio_features"] = not self.cfg.get("no_audio_features", False)
        elif key == "use_cache":
            self.use_cache = not self.use_cache
            self.cfg["cache_dir"] = self.cfg.get("cache_dir") or ".cache/clipper" \
                if self.use_cache else None
        elif key == "force_retranscribe":
            self.cfg["force_retranscribe"] = not self.cfg.get("force_retranscribe", False)
        elif key == "force_rescore":
            self.cfg["force_rescore"] = not self.cfg.get("force_rescore", False)
        elif key == "debug_captions":
            self.cfg["debug_captions"] = not self.cfg.get("debug_captions", False)
        elif key == "review_transcript":
            self.cfg["review_transcript"] = not self.cfg.get("review_transcript", False)
        elif key == "review_titles":
            self.cfg["review_titles"] = not self.cfg.get("review_titles", False)

    def _sync_from_widgets(self):
        # Semântica positiva da tela → flags negativas da CLI, sem inversão.
        self.cfg["model"] = self.models[self.model_idx]["id"]
        self.cfg["transcribe_backend"] = BACKENDS[self.backend_idx]
        if not self.use_cache:
            self.cfg["cache_dir"] = None
        elif not self.cfg.get("cache_dir"):
            self.cfg["cache_dir"] = ".cache/clipper"

    def draw(self):
        s = self.stdscr
        s.clear()
        h, w = s.getmaxyx()
        s.addstr(0, max(0, (w - 12) // 2), "CLIP FORGE", curses.A_BOLD)
        row = 2
        for i, (key, label, kind) in enumerate(self._fields()):
            if row >= h - 3:
                break
            val = self._get(key)
            if kind == "bool":
                disp = f"[{'x' if val else ' '}] {label}"
            elif kind == "action":
                disp = label
            elif kind in ("model", "backend"):
                disp = f"{label}: < {val} >"
            else:
                shown = "(vazio)" if val in ("", None) else str(val)
                disp = f"{label}: {shown}"
            attr = curses.A_REVERSE if i == self.pos else curses.A_NORMAL
            if kind == "action":
                attr |= curses.A_BOLD
            s.addstr(row, 2, disp[:w - 4], attr)
            row += 1
        if self.msg and h > 1:
            s.addstr(h - 2, 2, self.msg[:w - 4], curses.A_BOLD)
        s.addstr(h - 1, 2, "↑↓ navegar · ←→/Espaço alterar · Enter editar/processar · Esc sair"[:w - 4])
        s.refresh()

    def edit_field(self, key, kind):
        if key == "video":
            self._pick_video()
            return
        if kind in ("int", "float", "text"):
            cur = "" if self.cfg.get(key) is None else str(self.cfg.get(key, ""))
            if kind == "int":
                raw = _curses_input(self.stdscr, f"{key} (inteiro)", cur)
                try:
                    self.cfg[key] = int(raw)
                except (ValueError, TypeError):
                    self.msg = f"Valor inválido para {key}: {raw!r}"
            elif kind == "float":
                raw = _curses_input(self.stdscr, f"{key} (número)", cur)
                try:
                    self.cfg[key] = float(raw)
                except (ValueError, TypeError):
                    self.msg = f"Valor inválido para {key}: {raw!r}"
            else:
                self.cfg[key] = _curses_input(self.stdscr, key, cur)

    def _pick_video(self):
        vids = list_videos("videos")
        items = [f"{v['path']} ({human_bytes(v['size'])})" for v in vids]
        items.append("Outro caminho...")
        pos = 0
        s = self.stdscr
        while True:
            s.clear()
            h, w = s.getmaxyx()
            s.addstr(0, 2, "Escolha o vídeo (Enter confirma, Esc volta)", curses.A_BOLD)
            for i, item in enumerate(items[:h - 3]):
                s.addstr(2 + i, 4, f"> {item}" if i == pos else f"  {item}",
                         curses.A_REVERSE if i == pos else curses.A_NORMAL)
            s.refresh()
            ch = s.getch()
            if ch == 27:
                return
            elif ch == curses.KEY_UP:
                pos = (pos - 1) % len(items)
            elif ch == curses.KEY_DOWN:
                pos = (pos + 1) % len(items)
            elif ch in (10, 13, curses.KEY_ENTER):
                if pos == len(items) - 1:
                    p = _curses_input(s, "Caminho do vídeo",
                                      self.cfg.get("video") or "")
                    if p:
                        self.cfg["video"] = p
                else:
                    self.cfg["video"] = vids[pos]["path"]
                return

    def form(self):
        fields = self._fields()
        while True:
            self.draw()
            ch = self.stdscr.getch()
            key, _label, kind = fields[self.pos]
            if ch == 27:  # Esc
                return None
            elif ch in (curses.KEY_UP, ord("k")):
                self.pos = (self.pos - 1) % len(fields)
            elif ch in (curses.KEY_DOWN, ord("j")):
                self.pos = (self.pos + 1) % len(fields)
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")):
                if kind == "bool":
                    self._toggle(key)
                elif key == "model":
                    d = -1 if ch == curses.KEY_LEFT else 1
                    self.model_idx = (self.model_idx + d) % len(self.models)
                elif key == "transcribe_backend":
                    d = -1 if ch == curses.KEY_LEFT else 1
                    self.backend_idx = (self.backend_idx + d) % len(BACKENDS)
            elif ch in (10, 13, curses.KEY_ENTER):
                if kind == "bool":
                    self._toggle(key)
                elif key in ("model", "transcribe_backend"):
                    pass  # setas alteram; Enter não faz nada aqui
                elif key == "process":
                    self._sync_from_widgets()
                    errs = validate_for_run(self.cfg)
                    if errs:
                        self.msg = " | ".join(errs)
                    else:
                        return True
                else:
                    self.edit_field(key, kind)
                    self._sync_from_widgets()

    def summary(self):
        from clipper import cli_command  # tardio: evita import circular
        self._sync_from_widgets()
        model_name = self.models[self.model_idx]["name"]
        while True:
            s = self.stdscr
            s.clear()
            h, w = s.getmaxyx()
            s.addstr(0, max(0, (w - 15) // 2), "CONFIGURAÇÃO", curses.A_BOLD)
            row = 2
            for label, val in summary_lines(self.cfg, model_name):
                if row >= h - 6:
                    break
                s.addstr(row, 2, f"{label + ':':<16} {val}"[:w - 4])
                row += 1
            row += 1
            if row < h - 4:
                s.addstr(row, 2, "Comando equivalente:"[:w - 4], curses.A_BOLD)
                row += 1
            if row < h - 3:
                s.addstr(row, 2, cli_command(self.cfg)[:w - 4])
            s.addstr(h - 1, 2, "[Enter] Processar · [E] Editar · [Esc] Cancelar"[:w - 4])
            s.refresh()
            ch = s.getch()
            if ch == 27:
                return None
            elif ch in (ord("e"), ord("E")):
                return False
            elif ch in (10, 13, curses.KEY_ENTER):
                return True

    def menu(self):
        items = ["Processar vídeo", "Revisar transcrição", "Aprovar transcrição",
                 "Finalizar sessão", "Ver último relatório",
                 "Última transcrição", "Sair"]
        pos = 0
        while True:
            s = self.stdscr
            s.clear()
            h, w = s.getmaxyx()
            s.addstr(0, max(0, (w - 12) // 2), "CLIP FORGE", curses.A_BOLD)
            for i, item in enumerate(items):
                s.addstr(2 + i, 4, f"> {item}" if i == pos else f"  {item}",
                         curses.A_REVERSE if i == pos else curses.A_NORMAL)
            s.addstr(h - 1, 2, "↑↓ navegar · Enter escolher · q sair"[:w - 4])
            s.refresh()
            ch = s.getch()
            if ch in (ord("q"), ord("Q"), 27):
                return None
            elif ch in (curses.KEY_UP, ord("k")):
                pos = (pos - 1) % len(items)
            elif ch in (curses.KEY_DOWN, ord("j")):
                pos = (pos + 1) % len(items)
            elif ch in (10, 13, curses.KEY_ENTER):
                return items[pos]

    def view_file(self, path: Path | None, title: str, empty_msg: str):
        s = self.stdscr
        if path is None:
            s.clear()
            s.addstr(2, 2, empty_msg)
            s.addstr(4, 2, "Pressione qualquer tecla...")
            s.refresh()
            s.getch()
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            text = f"(não foi possível ler: {e})"
        lines = text.splitlines() or [""]
        top = 0
        while True:
            s.clear()
            h, w = s.getmaxyx()
            s.addstr(0, 2, f"{title}: {path}"[:w - 4], curses.A_BOLD)
            for i in range(1, h - 1):
                if top + i - 1 < len(lines):
                    s.addstr(i, 2, lines[top + i - 1][:w - 4])
            s.addstr(h - 1, 2, "↑↓ rolar · q voltar"[:w - 4])
            s.refresh()
            ch = s.getch()
            if ch in (ord("q"), ord("Q"), 27):
                return
            elif ch == curses.KEY_UP:
                top = max(0, top - 1)
            elif ch == curses.KEY_DOWN:
                top = min(max(0, len(lines) - (h - 2)), top + 1)

    # -- revisão de transcrição (opera sobre work/, sem flags) --
    def pick_session(self) -> dict | None:
        sessions = list_sessions("work")
        if not sessions:
            self.notice("Nenhuma sessão em work/ (processe com [x] Revisar transcrição).")
            return None
        pos = 0
        while True:
            s = self.stdscr
            s.clear()
            h, w = s.getmaxyx()
            s.addstr(0, 2, "SESSÕES (work/)", curses.A_BOLD)
            for i, sess in enumerate(sessions):
                row = 2 + i
                if row >= h - 2:
                    break
                line = (f"> {sess['name']} [{sess['status']}] "
                        f"{sess['segments']} seg/{sess['words']} words" if i == pos
                        else f"  {sess['name']} [{sess['status']}] "
                        f"{sess['segments']} seg/{sess['words']} words")
                s.addstr(row, 4, line[:w - 6],
                         curses.A_REVERSE if i == pos else curses.A_NORMAL)
            s.addstr(h - 1, 2, "↑↓ navegar · Enter escolher · Esc voltar"[:w - 4])
            s.refresh()
            ch = s.getch()
            if ch == 27:
                return None
            elif ch in (curses.KEY_UP, ord("k")):
                pos = (pos - 1) % len(sessions)
            elif ch in (curses.KEY_DOWN, ord("j")):
                pos = (pos + 1) % len(sessions)
            elif ch in (10, 13, curses.KEY_ENTER):
                return sessions[pos]

    def notice(self, msg: str):
        s = self.stdscr
        s.clear()
        s.addstr(2, 2, msg)
        s.addstr(4, 2, "Pressione qualquer tecla...")
        s.refresh()
        s.getch()

    def edit_external(self, path: Path):
        """Abre $EDITOR fora do modo curses e volta. Sem editor próprio."""
        import shutil
        import subprocess
        editor = os.environ.get("EDITOR") or shutil.which("nano") \
            or shutil.which("vi") or "vi"
        curses.endwin()
        try:
            print(f"\nEditando {path} com {editor} (salve e saia p/ voltar)...")
            subprocess.run([editor, str(path)])
        finally:
            self.stdscr.refresh()

    def review_flow(self):
        from core import review as _rev
        sess = self.pick_session()
        if sess is None:
            return
        tdir = Path(sess["dir"]) / "transcription"
        words_txt = tdir / "words.txt"
        if not words_txt.exists():
            self.notice(f"Sem words.txt em {tdir} (sessão incompleta).")
            return
        self.edit_external(words_txt)
        try:
            edited = _rev.parse_words_txt(words_txt)
        except RuntimeError as e:
            self.notice(f"words.txt inválido (nada aprovado): {e}")
            return
        self.notice(f"{len(edited)} segmentos válidos. Volte ao menu e use "
                    f"'Aprovar transcrição' p/ aprovar (ou aprove agora).")

    def approve_flow(self):
        from core import review as _rev
        sess = self.pick_session()
        if sess is None:
            return
        tdir = Path(sess["dir"]) / "transcription"
        if not (tdir / "words.txt").exists():
            self.notice(f"Sem words.txt em {tdir} (use 'Revisar transcrição').")
            return
        try:
            edited = _rev.parse_words_txt(tdir / "words.txt")
        except RuntimeError as e:
            self.notice(f"words.txt inválido (nada aprovado): {e}")
            return
        s = self.stdscr
        s.clear()
        s.addstr(2, 2, f"Aprovar {len(edited)} segmentos de '{sess['name']}'?")
        s.addstr(4, 2, "[S]im · outra tecla cancela")
        s.refresh()
        if s.getch() not in (ord("s"), ord("S")):
            return
        final = _rev.approve(tdir, edited)
        self.notice(f"TRANSCRIÇÃO APROVADA: {final}")

    def finalize_flow(self):
        from core import review as _rev
        import shutil
        sess = self.pick_session()
        if sess is None:
            return
        tdir = Path(sess["dir"]) / "transcription"
        try:
            _rev.require_approved(tdir / "transcript.json")
        except RuntimeError as e:
            self.notice(str(e))
            return
        out = Path(self.cfg.get("out") or "cortes")
        s = self.stdscr
        s.clear()
        s.addstr(2, 2, f"Finalizar '{sess['name']}'?")
        s.addstr(3, 2, f"Remove {sess['dir']}; mantém {out}/ + approved-transcript.json.")
        s.addstr(5, 2, "[S]im · outra tecla cancela")
        s.refresh()
        if s.getch() not in (ord("s"), ord("S")):
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / "approved-transcript.json").write_text(
            (tdir / "transcript.json").read_text(encoding="utf-8"), encoding="utf-8")
        shutil.rmtree(Path(sess["dir"]), ignore_errors=True)
        self.notice(f"FINALIZADO: finais em {out.resolve()}")


def _curses_main(stdscr, cfg: dict, models: list[dict]):
    curses.curs_set(0)
    stdscr.keypad(True)
    tui = TUI(stdscr, cfg, models)
    while True:
        choice = tui.menu()
        if choice is None or choice == "Sair":
            return None
        if choice == "Ver último relatório":
            tui.view_file(latest_report("metrics"), "RELATÓRIO",
                          "Nenhum relatório em metrics/ (processe um vídeo primeiro).")
            continue
        if choice == "Última transcrição":
            tui.view_file(latest_transcript(tui.cfg.get("cache_dir") or ".cache/clipper"),
                          "TRANSCRIÇÃO",
                          "Nenhuma transcrição em cache (use --cache-dir ao processar).")
            continue
        if choice == "Revisar transcrição":
            tui.review_flow()
            continue
        if choice == "Aprovar transcrição":
            tui.approve_flow()
            continue
        if choice == "Finalizar sessão":
            tui.finalize_flow()
            continue
        # Processar vídeo
        while True:
            r = tui.form()
            if r is None:
                break
            decision = tui.summary()
            if decision is None:
                return None
            if decision is False:
                continue
            return tui.cfg


def run(cfg: dict | None = None):
    """Abre a TUI. Retorna dict canônico pronto p/ run_pipeline, ou None."""
    try:
        import curses as _c  # noqa: F401 (stdlib; erro amigável se ausente)
    except ImportError:
        print("Erro: curses indisponível neste terminal. Use a CLI: python clipper.py VIDEO ...")
        return None
    if cfg is None:
        from clipper import default_config  # tardio: evita import circular
        cfg = default_config()
    # Carrega .env para pegar NVIDIA_API_KEY (igual ao CLI)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    try:
        models = load_models()
    except (OSError, ValueError) as e:
        print(f"Erro: {e}")
        return None
    try:
        return curses.wrapper(_curses_main, cfg, models)
    except curses.error:
        print("Erro: terminal incompatível com a TUI. Use a CLI: python clipper.py VIDEO ...")
        return None
