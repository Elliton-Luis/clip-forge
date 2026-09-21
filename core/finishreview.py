"""finishreview.py — revisão pré-burn-in do FINISH, orientada a artefatos.

Diferença para clipreview (DISCOVERY): aqui não há seleção — o clip já é o
momento escolhido. A revisão trabalha sobre transcript.json / title.json /
captions.json, o vídeo segue SEM nada queimado, e só depois do aceite o
render acontece. Ações: A (aceitar) E (editar) R (regenerar) T/L (liga/desliga
título/legenda) S (pular) Q (sair, persiste e continua depois).

Regras:
- Edição manual mexe em cópia de trabalho; transcript.json NUNCA é alterado
  em silêncio — a edição fica registrada em review.json e aplicada às
  legendas regeneradas (mensagem explícita no terminal).
- Reutiliza os helpers puros de clipreview (linhas de transcript, words.txt,
  previews limpos); o loop e o estado são próprios.
- Estado em review.json (artefato versionado no store): status, toggles,
  edited, transcript_hash vigente e histórico. Reabrir continua de onde parou.
"""
import sys
import time
from pathlib import Path

from . import artifacts as art


def _fmt_dur(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def load_state(store: Path | str, source_fp: str) -> dict:
    """Estado anterior ou default. Store estranho/ausente = começa do zero."""
    try:
        data = art.load_artifact(store, "review", source_fp)
        if isinstance(data, dict) and data.get("status") in (
                "pending", "approved", "skipped"):
            return data
    except art.InvalidArtifact:
        pass
    return {"status": "pending", "title_enabled": True,
            "captions_enabled": True, "edited": False,
            "transcript_hash": None, "history": []}


def save_state(store: Path | str, source_fp: str, source_path: str,
               state: dict) -> Path:
    return art.save_artifact(store, "review", source_fp, source_path, {},
                             state)


def _show(clip: str, duration: float, transcript_text: str, title: str | None,
          srt: str | None, state: dict, preview: Path | None) -> None:
    bar = "━" * 46
    print(f"\n{bar}\nCLIP — {Path(clip).name}\n{bar}\n")
    print(f"Arquivo: {clip}\nDuração: {_fmt_dur(duration)}")
    if state.get("status") == "approved" and state.get("at"):
        print(f"Estado: aprovado em {state['at']} (Enter confirma de novo)")
    print(f"\nTRANSCRIÇÃO:\n\"{transcript_text[:300]}\"\n")
    t_flag = "[x]" if state.get("title_enabled", True) else "[ ]"
    print(f"TÍTULO {t_flag}:\n\"{title or '(sem título)'}\"\n")
    l_flag = "[x]" if state.get("captions_enabled", True) else "[ ]"
    first_cues = "\n".join((srt or "").split("\n\n")[:2]) if srt else "(sem legenda)"
    print(f"LEGENDA {l_flag}:\n{first_cues}\n")
    if preview is not None:
        print(f"Vídeo (SEM queimar nada):\n{preview.resolve()}\n")
    if state.get("edited"):
        print("(texto editado manualmente — transcript.json original preservado)\n")
    print(f"{bar}\n")
    print("[A] aceitar   [E] editar legenda   [R] regenerar\n"
          "[T] título on/off   [L] legenda on/off   [S] pular   [Q] sair")


def _touch(state: dict, action: str) -> None:
    state.setdefault("history", []).append(
        {"action": action, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    state["at"] = state["history"][-1]["at"]


def run_review(clip: str, segments: list, title: str | None, srt: str | None,
               duration: float, store: Path | str, source_fp: str,
               preview: Path | None = None,
               regen_title_fn=None, regen_captions_fn=None,
               edit_fn=None, input_fn=None) -> dict:
    """Loop de revisão. Retorna outcome; Q/EOF/Ctrl+C persiste e sai via SystemExit.

    edit_fn(candidate_like) -> bool(aplicou): edita words; regen_*_fn() ->
    novos (title, srt). Callbacks para não importar o orquestrador (sem ciclo).
    """
    if not sys.stdin.isatty() and input_fn is None:
        raise SystemExit("Entrada não-interativa: revisão exige terminal.")
    ask = input_fn or (lambda msg: input(msg))
    state = load_state(store, source_fp)
    if state.get("status") == "approved":
        print(f"   -> revisão já aprovada em {state.get('at', '?')} "
              f"(Enter confirma, ou escolha outra ação)")
    transcript_text = " ".join(s.text for s in segments).strip()

    def _persist():
        save_state(store, source_fp, clip, state)

    try:
        while True:
            _show(clip, duration, transcript_text, title, srt, state, preview)
            ans = (ask("> ") or "").strip().lower()
            if ans in ("a", "aceitar", ""):
                state["status"] = "approved"
                _touch(state, "approved")
                _persist()
                return {"action": "approved", "state": dict(state)}
            elif ans in ("e", "editar"):
                if edit_fn is None:
                    print("   ! edição indisponível aqui")
                    continue
                if edit_fn():
                    state["edited"] = True
                    _touch(state, "edited")
                    _persist()
                    return {"action": "edited", "state": dict(state)}
            elif ans in ("r", "regenerar"):
                sub = (ask("   regenerar [T]ítulo, [L]egenda ou [C]ancelar? ")
                       or "").strip().lower()
                if sub in ("t", "título", "titulo"):
                    if regen_title_fn is None:
                        print("   ! regeneração de título indisponível")
                        continue
                    title = regen_title_fn()
                    _touch(state, "regen-title")
                    _persist()
                elif sub in ("l", "legenda"):
                    if regen_captions_fn is None:
                        print("   ! regeneração de legenda indisponível")
                        continue
                    srt = regen_captions_fn()
                    _touch(state, "regen-captions")
                    _persist()
                # 'c'/outro = só volta à tela
            elif ans in ("t", "titulo", "título"):
                state["title_enabled"] = not state.get("title_enabled", True)
                _touch(state, "title-off" if not state["title_enabled"] else "title-on")
                _persist()
            elif ans in ("l", "legenda"):
                state["captions_enabled"] = not state.get("captions_enabled", True)
                _touch(state, "captions-off"
                       if not state["captions_enabled"] else "captions-on")
                _persist()
            elif ans in ("s", "pular", "skip"):
                state["status"] = "skipped"
                _touch(state, "skipped")
                _persist()
                return {"action": "skipped", "state": dict(state)}
            elif ans in ("q", "sair", "quit"):
                _touch(state, "quit")
                _persist()
                raise SystemExit("Revisão interrompida — estado salvo. "
                                 "Reabra para continuar.")
            else:
                print("   Opções: [A] [E] [R] [T] [L] [S] [Q]")
    except KeyboardInterrupt:
        _touch(state, "quit")
        _persist()
        raise SystemExit("Revisão interrompida (Ctrl+C) — estado salvo.")
    except EOFError:
        _touch(state, "quit")
        _persist()
        raise SystemExit("Entrada não-interativa: revisão exige terminal.")
