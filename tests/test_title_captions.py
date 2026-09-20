"""tests/test_title_captions.py — título e legenda como recursos independentes.

Quatro combinações válidas (title × captions); nenhuma pode gerar erro,
e o recurso desligado nunca aparece na saída.

Roda sem dependências externas: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Candidate, Word
from core.video import (build_ass, build_hook_lines, highlight_words_from_title,
                        cut, LAYOUT_W, LAYOUT_H,
                        CAPTION_HIGHLIGHT_OPEN, HOOK_END_SEC)


def W(text, start, end):
    return Word(text, start, end)


def make_candidate(title="Jogada Insana No Final"):
    words = [W(" que", 10.5, 11.0), W(" jogada", 11.1, 11.6),
             W(" insana", 11.7, 12.3), W(" demais", 12.4, 13.0)]
    return Candidate(start=10.0, end=20.0, text="que jogada insana demais",
                     words=words, score=8.5, title=title)


def hook_text(ass: str) -> str:
    """Texto das linhas Dialogue de estilo Hook (vazio = sem título)."""
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:") and ",Hook," in line:
            out.append(line.rsplit(",", 1)[-1])
    return "\n".join(out)


def clip_dialogues(ass: str) -> str:
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:") and ",Clip,," in line:
            out.append(line.rsplit(",", 1)[-1])
    return "\n".join(out)


class TestBuildAssIndependence(unittest.TestCase):
    def test_title_on_captions_on(self):
        c = make_candidate()
        ass = build_ass(c.words, c.start, c.end, LAYOUT_W, LAYOUT_H, 54,
                        highlight=highlight_words_from_title(c.title),
                        hook_title=c.title)
        self.assertIn("JOGADA", hook_text(ass))  # hook com o título
        self.assertIn("JOGADA", clip_dialogues(ass))  # cues presentes
        self.assertIn(CAPTION_HIGHLIGHT_OPEN, ass)  # destaque ativo

    def test_title_on_captions_off(self):
        # Sem palavras: só o hook, nenhuma cue de legenda.
        c = make_candidate()
        ass = build_ass([], c.start, c.end, LAYOUT_W, LAYOUT_H, 54,
                        highlight=highlight_words_from_title(c.title),
                        hook_title=c.title)
        self.assertIn("JOGADA", hook_text(ass))  # título continua igual
        self.assertEqual(clip_dialogues(ass).strip(), "")  # sem captions

    def test_title_off_captions_on(self):
        c = make_candidate()
        ass = build_ass(c.words, c.start, c.end, LAYOUT_W, LAYOUT_H, 54,
                        highlight=None, hook_title=None)
        self.assertEqual(hook_text(ass).strip(), "")  # sem hook
        self.assertIn("JOGADA", clip_dialogues(ass))  # captions iguais
        self.assertNotIn(CAPTION_HIGHLIGHT_OPEN, ass)  # sem destaque

    def test_both_off(self):
        ass = build_ass([], 10.0, 20.0, LAYOUT_W, LAYOUT_H, 54,
                        highlight=None, hook_title=None)
        self.assertEqual(hook_text(ass).strip(), "")
        self.assertEqual(clip_dialogues(ass).strip(), "")

    def test_empty_title_never_hooks(self):
        c = make_candidate(title="")
        ass = build_ass(c.words, c.start, c.end, LAYOUT_W, LAYOUT_H, 54,
                        highlight=highlight_words_from_title(c.title),
                        hook_title=c.title or None)
        self.assertEqual(hook_text(ass).strip(), "")
        self.assertEqual(build_hook_lines(""), [])


class TestCutFilter(unittest.TestCase):
    """cut() só adiciona o filtro subtitles quando há o que queimar."""

    def _run_cut(self, captions: bool, title: bool):
        c = make_candidate()
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            vf = cmd[cmd.index("-vf") + 1]
            captured["vf"] = vf

            class R:
                returncode = 0
            return R()

        with patch("core.video.face_center_x", return_value=0.5), \
             patch("core.video._probe_dimensions", return_value=(1920, 1080)), \
             patch("core.video.subprocess.run", side_effect=fake_run):
            cut("dummy.mp4", c, Path("/tmp/x.mp4"),
                vertical=True, captions=captions, title=title)
        return captured

    def test_on_on_has_subtitles(self):
        vf = self._run_cut(True, True)["vf"]
        self.assertIn("subtitles=", vf)

    def test_title_only_has_subtitles(self):
        # Título viaja no ASS: precisa do filtro, mas sem cues de legenda.
        vf = self._run_cut(False, True)["vf"]
        self.assertIn("subtitles=", vf)

    def test_captions_only_has_subtitles(self):
        vf = self._run_cut(True, False)["vf"]
        self.assertIn("subtitles=", vf)

    def test_both_off_no_subtitles(self):
        vf = self._run_cut(False, False)["vf"]
        self.assertNotIn("subtitles=", vf)

    def test_title_only_ass_has_hook_no_cues(self):
        # Captura o ASS gerado no modo só-título via debug_dir.
        import tempfile
        c = make_candidate()
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(cmd, **kw):
                class R:
                    returncode = 0
                return R()
            with patch("core.video.face_center_x", return_value=0.5), \
                 patch("core.video._probe_dimensions", return_value=(1920, 1080)), \
                 patch("core.video.subprocess.run", side_effect=fake_run):
                cut("dummy.mp4", c, Path("01_teste.mp4"),
                    vertical=True, captions=False, title=True,
                    debug_dir=tmp)
            ass = (Path(tmp) / "01_teste" / "captions.ass").read_text(encoding="utf-8")
        self.assertIn("JOGADA", hook_text(ass))
        self.assertEqual(clip_dialogues(ass).strip(), "")

    def test_captions_only_ass_has_cues_no_hook(self):
        import tempfile
        c = make_candidate()
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(cmd, **kw):
                class R:
                    returncode = 0
                return R()
            with patch("core.video.face_center_x", return_value=0.5), \
                 patch("core.video._probe_dimensions", return_value=(1920, 1080)), \
                 patch("core.video.subprocess.run", side_effect=fake_run):
                cut("dummy.mp4", c, Path("01_teste.mp4"),
                    vertical=True, captions=True, title=False,
                    debug_dir=tmp)
            ass = (Path(tmp) / "01_teste" / "captions.ass").read_text(encoding="utf-8")
        self.assertEqual(hook_text(ass).strip(), "")
        self.assertIn("JOGADA", clip_dialogues(ass))


class TestTitleFlagConfig(unittest.TestCase):
    def test_cli_no_title(self):
        from clipper import parse_cli, config_from_args, cli_command
        cfg = config_from_args(parse_cli(["v.mp4", "--no-title"]))
        self.assertTrue(cfg["no_title"])
        cmd = cli_command(cfg)
        self.assertIn("--no-title", cmd)
        back = config_from_args(parse_cli(cmd.split()[2:]))
        self.assertTrue(back["no_title"])

    def test_default_keeps_both_on(self):
        from clipper import default_config
        cfg = default_config()
        self.assertFalse(cfg.get("no_title", False))
        self.assertFalse(cfg.get("no_captions", False))

    def test_tui_toggle_title(self):
        from core.tui import TUI
        ui = TUI.__new__(TUI)
        ui.cfg = {"no_title": False}
        self.assertTrue(ui._get("title"))
        ui._toggle("title")
        self.assertFalse(ui._get("title"))
        self.assertTrue(ui.cfg["no_title"])


if __name__ == "__main__":
    unittest.main()
