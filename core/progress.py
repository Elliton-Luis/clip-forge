"""progress.py — estado, progresso e log do terminal. Só stdlib.

Princípio: a TUI mostra o ESTADO atual do sistema, não narra cada linha de
execução. O pipeline atualiza o bus; o bus decide como renderizar:

- TTY interativo → dashboard ANSI reescrito no lugar (sem curses fullscreen,
  para não quebrar run.sh, pipes ou redirecionamento);
- pipe/script/teste → linhas de conclusão (mesmo volume de antes, estruturado);
- --quiet → só avisos, erros e resumo final;
- --verbose / --log-file → detalhes por chunk/tentativa preservados.

ETA só aparece quando é confiável: etapa com total conhecido, algum
progresso feito e mais de MIN_ETA_SEC_S de medição. Nunca se fabrica
porcentagem: sem total conhecido, mostra-se `feito` (contagem) sem barra.
"""
import shutil
import sys
import time
from contextlib import contextmanager

BAR_W = 20
REDRAW_MIN_INTERVAL = 0.25
MIN_ETA_SEC = 5.0


def _fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(done: int, total: int) -> str:
    filled = min(BAR_W, int(BAR_W * done / total)) if total > 0 else 0
    return "█" * filled + "░" * (BAR_W - filled)


class _Stage:
    def __init__(self, key: str, label: str, total: int | None = None,
                 unit: str = ""):
        self.key = key
        self.label = label
        self.total = total
        self.unit = unit
        self.done = 0
        self.state = "active"  # active|done|skipped
        self.summary = ""
        self.detail = ""
        self.t0 = time.time()

    def eta(self) -> str:
        """ETA só se confiável; senão string vazia (chamador omite)."""
        elapsed = time.time() - self.t0
        if self.total is None or self.done <= 0 or elapsed < MIN_ETA_SEC:
            return ""
        rate = self.done / elapsed
        if rate <= 0:
            return ""
        remaining = (self.total - self.done) / rate
        if remaining < 1 or remaining > 24 * 3600:
            return ""
        return f"ETA {_fmt_hms(remaining)}"


class Progress:
    """Barramento de estado. Thread único (pipeline é síncrono)."""

    def __init__(self, mode: str = "", out=None, log_file: str | None = None,
                 verbose: bool = False, quiet: bool = False):
        self.mode = mode
        self.out = out if out is not None else sys.stdout
        try:
            self.tty = bool(self.out.isatty()) and not quiet
        except Exception:
            self.tty = False
        self.verbose = verbose
        self.quiet = quiet
        self.header_lines: list[str] = []
        self.stages: list[_Stage] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []  # conclusões recentes (últimas 3)
        self.t0 = time.time()
        self._drawn = 0
        self._last_draw = 0.0
        self._paused = False
        self._logfh = None
        if log_file:
            try:
                self._logfh = open(log_file, "a", encoding="utf-8")
            except OSError:
                self._logfh = None

    # -- API usada pelo pipeline --
    def header(self, lines: list[str]) -> None:
        self.header_lines = [str(x) for x in lines if x]
        if not self.tty and not self.quiet:
            for ln in self.header_lines:
                self._emit(ln)
        self._draw(force=True)

    def stage(self, key: str, label: str, total: int | None = None,
              unit: str = "") -> _Stage:
        st = _Stage(key, label, total, unit)
        self.stages.append(st)
        if not self.tty and not self.quiet:
            self._emit(f"▸ {label}...")
        self._draw()
        return st

    def adv(self, key: str, n: int = 1, msg: str = "") -> None:
        st = self._find(key)
        if st is None:
            return
        st.done += n
        if msg:
            st.detail = msg
        self._draw()

    def done(self, key: str, summary: str = "") -> None:
        st = self._find(key)
        if st is None:
            return
        st.state = "done"
        st.summary = summary
        if summary:
            self.notes.append(summary)
            self.notes = self.notes[-3:]
        if not self.tty and not self.quiet:
            self._emit(f"✓ {st.label}" + (f" — {summary}" if summary else ""))
        self._draw(force=True)

    def skip(self, key: str, reason: str = "") -> None:
        st = self._find(key)
        if st is None:
            return
        st.state = "skipped"
        st.summary = reason
        if not self.tty and not self.quiet:
            self._emit(f"– {st.label} ({reason})" if reason else f"– {st.label}")
        self._draw(force=True)

    def warn(self, msg: str) -> None:
        """Aviso importante (vai para a área de avisos do dashboard).
        O texto chega pronto (com seu próprio prefixo, ex. '! ...').
        Avisos aparecem até em --quiet: são o motivo do modo existir."""
        self.warnings.append(msg)
        self.warnings = self.warnings[-5:]
        self._log(f"WARN: {msg}")
        if self.tty:
            self._draw(force=True)
        else:
            self._emit(msg)

    def note(self, msg: str) -> None:
        """Linha de conclusão (ex: '14 janelas'). No dashboard vira histórico."""
        self.notes.append(msg)
        self.notes = self.notes[-3:]
        if not self.tty and not self.quiet:
            self._emit(msg)
        else:
            self._draw()

    def detail(self, msg: str) -> None:
        """Detalhe técnico: só verbose/stdout, sempre no log-file."""
        self._log(f"detail: {msg}")
        if self.verbose and not self.quiet:
            if self.tty:
                self._draw()
            else:
                self._emit(f"  {msg}")

    @contextmanager
    def pause(self):
        """Congela o dashboard antes de input() interativo; fecha o bloco."""
        was_tty = self.tty
        self.tty = False
        if was_tty and self._drawn:
            self._emit("")
        try:
            yield self
        finally:
            self.tty = was_tty
            self._drawn = 0  # próximo draw reimprime o bloco do zero
            self._draw(force=True)

    def close(self) -> None:
        if self.tty and self._drawn:
            self._emit("")
        if self._logfh:
            try:
                self._logfh.close()
            except Exception:
                pass
            self._logfh = None
        self._drawn = 0

    # -- internos --
    def _find(self, key: str) -> _Stage | None:
        for st in self.stages:
            if st.key == key:
                return st
        return None

    def _emit(self, text: str) -> None:
        try:
            self.out.write(text + "\n")
            self.out.flush()
        except Exception:
            pass
        self._log(text)

    def _log(self, text: str) -> None:
        if self._logfh:
            try:
                self._logfh.write(text + "\n")
                self._logfh.flush()
            except Exception:
                pass

    def _block(self) -> list[str]:
        width = shutil.get_terminal_size((80, 24)).columns
        elapsed = _fmt_hms(time.time() - self.t0)
        lines = [f"CLIPPER · {self.mode}  [{elapsed}]"[:width]]
        lines += [x[:width] for x in self.header_lines]
        for st in self.stages:
            if st.state == "done":
                mark = "✓"
                info = st.summary
            elif st.state == "skipped":
                mark, info = "–", st.summary
            else:
                mark = "▸"
                info = self._active_info(st)
            row = f"{mark} {st.label}"
            if info:
                row += f"  {info}"
            lines.append(row[:width])
        for w in self.warnings[-2:]:
            lines.append(str(w)[:width])
        return lines

    def _active_info(self, st: _Stage) -> str:
        unit = f" {st.unit}" if st.unit else ""
        if st.total is None:
            base = f"{st.done}{unit}" if (st.done or st.detail) else ""
        else:
            pct = int(100 * st.done / st.total) if st.total > 0 else 0
            base = f"{_bar(st.done, st.total)} {pct}%  {st.done}/{st.total}{unit}"
        parts = [p for p in (base, st.detail, st.eta()) if p]
        return " · ".join(parts)

    def _draw(self, force: bool = False) -> None:
        if not self.tty or self._paused:
            return
        now = time.time()
        if not force and now - self._last_draw < REDRAW_MIN_INTERVAL:
            return
        self._last_draw = now
        lines = self._block()
        try:
            if self._drawn:
                self.out.write(f"\x1b[{self._drawn}A")
            for ln in lines:
                self.out.write("\r\x1b[K" + ln + "\n")
            self.out.flush()
            self._drawn = len(lines)
        except Exception:
            self.tty = False


def error_box(out, title: str, lines: list[str], hint: str = "") -> None:
    """Erro claro, contextualizado e acionável. Sem traceback cru."""
    body = [f"✗ {title}"]
    body += [f"  {x}" for x in lines if x]
    if hint:
        body.append(f"  → {hint}")
    try:
        out.write("\n".join(body) + "\n")
        out.flush()
    except Exception:
        pass


def review_banner(out, kind: str, i: int, n: int, extra: str = "") -> None:
    """Cabeçalho uniforme das telas de decisão. O resto de cada fluxo continua igual."""
    head = f"━━━ {kind} · {i}/{n} ━━━"
    if extra:
        head += f"  {extra}"
    try:
        out.write("\n" + head + "\n")
        out.flush()
    except Exception:
        pass
