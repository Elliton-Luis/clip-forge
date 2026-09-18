"""tests/test_subs.py — conversão absoluto→relativo + tokens especiais.

Roda sem dependências externas: python -m unittest discover -s tests
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Word
from core.video import (build_srt, build_ass, is_special_token,
                        clean_caption_text, highlight_words_from_title,
                        CAPTION_POP_OPEN, CAPTION_HIGHLIGHT_OPEN,
                        LAYOUT_W, LAYOUT_H, LAYOUT_MAIN_H, LAYOUT_BAND)


def W(text, start, end):
    return Word(text, start, end)


def parse_cues(srt):
    cues = []
    for block in srt.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        g = list(map(int, m.groups()))
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        cues.append((s, e, "\n".join(lines[2:])))
    return cues


class TestRelativeConversion(unittest.TestCase):
    def test_exact_relative(self):
        words = [W("olá", 12.5, 13.0), W("mundo", 13.2, 13.8)]
        cues = parse_cues(build_srt(words, 10.0, 20.0))
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0][0], 2.5, places=2)
        self.assertAlmostEqual(cues[0][1], 3.8, places=2)
        self.assertIn("OLÁ", cues[0][2])

    def test_no_cue_before_zero(self):
        words = [W("antes", 9.5, 9.9), W("dentro", 10.5, 11.0)]
        for s, e, _ in parse_cues(build_srt(words, 10.0, 20.0)):
            self.assertGreaterEqual(s, 0.0)

    def test_no_cue_after_end(self):
        words = [W("dentro", 18.0, 18.5), W("depois", 20.5, 21.0)]
        srt = build_srt(words, 10.0, 20.0)
        for s, e, _ in parse_cues(srt):
            self.assertLessEqual(e, 10.0 + 1e-6)
        self.assertNotIn("DEPOIS", srt)

    def test_all_outside_returns_empty(self):
        words = [W("longe", 50.0, 51.0)]
        self.assertEqual(build_srt(words, 10.0, 20.0), "")

    def test_deterministic(self):
        words = [W("a", 11.0, 11.4), W("b", 11.5, 12.0), W("c", 12.5, 13.0)]
        self.assertEqual(build_srt(words, 10.0, 20.0), build_srt(words, 10.0, 20.0))

    def test_cues_ordered_and_positive(self):
        words = [W(f"w{i}", 10.0 + i, 10.0 + i + 0.5) for i in range(12)]
        cues = parse_cues(build_srt(words, 10.0, 30.0))
        self.assertTrue(cues)
        for s, e, _ in cues:
            self.assertGreater(e, s)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(e, 20.0 + 1e-6)


class TestSpecialTokens(unittest.TestCase):
    def test_token_detection(self):
        for t in ("[eot]", "[sot]", "[_EOT_]", "[SOT]", "[TRANSCRIBE]",
                  "[TRANSLATE]", "[NOTIMESTAMPS]", "[BLANK]", "[_TT_12_]"):
            self.assertTrue(is_special_token(t), t)
        for t in ("hello", "(risos)", "eita", ""):
            self.assertFalse(is_special_token(t), t)

    def test_never_visible(self):
        words = [W("[sot]", 10.0, 10.2), W("jogada", 10.3, 10.8),
                 W("[eot]", 10.9, 11.0), W("insana", 11.1, 11.6),
                 W("[_EOT_]", 11.7, 11.8)]
        srt = build_srt(words, 10.0, 20.0)
        self.assertNotIn("[", srt.replace("-->", ""))
        self.assertNotIn("EOT", srt)
        self.assertNotIn("SOT", srt)
        self.assertIn("JOGADA", srt)
        self.assertIn("INSANA", srt)

    def test_only_specials_gives_empty(self):
        words = [W("[sot]", 10.0, 10.2), W("[eot]", 10.3, 10.5)]
        self.assertEqual(build_srt(words, 10.0, 20.0), "")

    def test_cleaner(self):
        self.assertEqual(clean_caption_text("a [eot] b  [sot] c"), "a b c")
        self.assertEqual(clean_caption_text("[_EOT_]"), "")


def parse_ass(text):
    cues = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 10)
        self_fmt = parts  # Layer,Start,End,Style,Name,ML,MR,MV,Effect,Text
        def t(s):
            h, m, rest = s.split(":")
            sec = float(rest)
            return int(h) * 3600 + int(m) * 60 + sec
        cues.append((t(parts[1]), t(parts[2]), parts[9].replace("\\N", "\n")))
    return cues


class TestASS(unittest.TestCase):
    def setUp(self):
        self.words = [W("olá", 12.5, 13.0), W("mundo", 13.2, 13.8),
                      W("[eot]", 13.9, 14.0)]

    def test_playres_matches_output(self):
        ass = build_ass(self.words, 10.0, 20.0, 1080, 1920)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("PlayResY: 1920", ass)

    def test_same_timing_as_srt(self):
        a = parse_ass(build_ass(self.words, 10.0, 20.0, 1080, 1920))

        def parse_srt_simple(srt):
            out = []
            for block in srt.strip().split("\n\n"):
                lines = block.strip().splitlines()
                m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
                g = list(map(int, m.groups()))
                out.append((g[0]*3600+g[1]*60+g[2]+g[3]/1000.0,
                            g[4]*3600+g[5]*60+g[6]+g[7]/1000.0,
                            "\n".join(lines[2:])))
            return out
        s = parse_srt_simple(build_srt(self.words, 10.0, 20.0))
        self.assertEqual(len(a), len(s))
        for (x1, y1, _), (x2, y2, _) in zip(a, s):
            self.assertAlmostEqual(x1, x2, places=2)
            self.assertAlmostEqual(y1, y2, places=2)

    def test_relative_bounded_and_clean(self):
        cues = parse_ass(build_ass(self.words, 10.0, 20.0, 1080, 1920))
        self.assertTrue(cues)
        for s, e, text in cues:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(e, 10.0 + 1e-6)
            self.assertGreater(e, s)
            self.assertNotIn("[", text)
            self.assertNotIn("EOT", text)

    def test_style_present(self):
        ass = build_ass(self.words, 10.0, 20.0, 1080, 1920)
        self.assertIn("Montserrat", ass)
        self.assertIn("Alignment", ass)


class TestHighlightAndAnimation(unittest.TestCase):
    def test_title_words(self):
        hl = highlight_words_from_title("AH DOIDO! PICA DE BICHO na live")
        self.assertIn("doido", hl)
        self.assertIn("bicho", hl)
        self.assertNotIn("de", hl)    # curta
        self.assertNotIn("na", hl)    # curta
        self.assertNotIn("ah", hl)    # curta

    def test_highlight_applied_deterministically(self):
        words = [W("ah", 12.0, 12.3), W("doido", 12.4, 12.9)]
        hl = highlight_words_from_title("AH DOIDO")
        a1 = build_ass(words, 10.0, 20.0, 1080, 1920, highlight=hl)
        a2 = build_ass(words, 10.0, 20.0, 1080, 1920, highlight=hl)
        self.assertEqual(a1, a2)
        self.assertIn(CAPTION_HIGHLIGHT_OPEN + "DOIDO", a1)
        self.assertNotIn(CAPTION_HIGHLIGHT_OPEN + "AH", a1)

    def test_no_highlight_without_title_words(self):
        words = [W("olá", 12.0, 12.5)]
        a = build_ass(words, 10.0, 20.0, 1080, 1920, highlight=set())
        self.assertNotIn("\\1c", a)

    def test_pop_animation_on_every_cue(self):
        words = [W("a", 11.0, 11.4), W("b", 12.5, 13.0)]
        a = build_ass(words, 10.0, 20.0, 1080, 1920)
        dialogues = [l for l in a.splitlines() if l.startswith("Dialogue")]
        self.assertTrue(dialogues)
        for d in dialogues:
            self.assertIn(CAPTION_POP_OPEN, d)

    def test_animation_preserves_timing(self):
        words = [W("olá", 12.5, 13.0), W("mundo", 13.2, 13.8)]
        a = parse_ass(build_ass(words, 10.0, 20.0, 1080, 1920))
        self.assertAlmostEqual(a[0][0], 2.5, places=2)
        self.assertAlmostEqual(a[0][1], 3.8, places=2)

    def test_layout_geometry(self):
        self.assertEqual((LAYOUT_W, LAYOUT_H), (1080, 1920))
        self.assertGreater(LAYOUT_MAIN_H, LAYOUT_H * 0.7)  # vídeo dominante
        self.assertEqual(LAYOUT_BAND, (LAYOUT_H - LAYOUT_MAIN_H) // 2)


if __name__ == "__main__":
    unittest.main()
