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
                        smart_join, build_hook_lines, validate_cues,
                        render_caption_debug, _group_cues, _group_cue_words,
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

    def test_event_cue_yellow_normal_cue_plain(self):
        words = [W(" fala", 10.0, 11.0)]
        ass = build_ass(words, 0.0, 30.0, 1080, 1920,
                        event_cues=[(20.0, 21.0, "*ÁUDIO ESTOURADO")])
        self.assertIn(CAPTION_HIGHLIGHT_OPEN + "*ÁUDIO ESTOURADO", ass)
        fala = [l for l in ass.splitlines() if l.rstrip().endswith("FALA")]
        self.assertTrue(fala)
        self.assertNotIn(CAPTION_HIGHLIGHT_OPEN, fala[0])

    def test_event_cue_plain_in_srt(self):
        words = [W(" fala", 10.0, 11.0)]
        srt = build_srt(words, 0.0, 30.0,
                        event_cues=[(20.0, 21.0, "*ÁUDIO ESTOURADO")])
        self.assertIn("*ÁUDIO ESTOURADO", srt)
        self.assertNotIn(CAPTION_HIGHLIGHT_OPEN, srt)  # SRT não tem cor

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
        self.assertGreater(LAYOUT_MAIN_H, LAYOUT_H * 0.6)  # vídeo dominante
        self.assertEqual(LAYOUT_BAND, (LAYOUT_H - LAYOUT_MAIN_H) // 2)
        self.assertGreaterEqual(LAYOUT_BAND, 300)  # faixas significativas


class TestSilencePreserved(unittest.TestCase):
    """Caso concreto: 'Olhe para a direita' + 3 s de silêncio + 'tem um vaso'."""

    def setUp(self):
        self.words = [W("Olhe", 10.0, 10.3), W("para", 10.4, 10.6),
                      W("a", 10.6, 10.7), W("direita", 10.8, 11.5),
                      W("tem", 14.5, 14.7), W("um", 14.8, 15.0),
                      W("vaso", 15.1, 15.5)]

    def test_silence_gap_has_no_cue(self):
        cues = parse_cues(build_srt(self.words, 0.0, 30.0))
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0][1], 11.5, places=2)
        self.assertAlmostEqual(cues[1][0], 14.5, places=2)
        self.assertIn("DIREITA", cues[0][2])
        self.assertNotIn("VASO", cues[0][2])
        self.assertIn("VASO", cues[1][2])

    def test_consecutive_segments_merge(self):
        words = [W("foi", 20.0, 20.3), W("bom", 20.5, 20.8)]
        cues = parse_cues(build_srt(words, 0.0, 30.0))
        self.assertEqual(len(cues), 1)

    def test_cut_starts_mid_segment(self):
        words = [W("mei", 8.0, 9.0), W("fim", 11.0, 12.0)]
        cues = parse_cues(build_srt(words, 10.0, 20.0))
        self.assertTrue(cues)
        self.assertGreaterEqual(cues[0][0], 0.0)
        self.assertNotIn("MEI", build_srt(words, 10.0, 20.0))

    def test_cut_ends_mid_segment(self):
        words = [W("aqui", 18.0, 19.0), W("longe", 21.0, 22.0)]
        cues = parse_cues(build_srt(words, 10.0, 20.0))
        for _, e, _ in cues:
            self.assertLessEqual(e, 10.0 + 1e-6)

    def test_cue_crossing_start_clamped(self):
        words = [W("corta", 9.5, 10.5)]
        cues = parse_cues(build_srt(words, 10.0, 20.0))
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0][0], 0.0, places=2)

    def test_cue_crossing_end_clamped(self):
        words = [W("corta", 19.5, 20.5)]
        cues = parse_cues(build_srt(words, 10.0, 20.0))
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0][1], 10.0, places=2)


class TestSmartJoin(unittest.TestCase):
    def test_whisper_cpp_tokens(self):
        self.assertEqual(smart_join([" dire", "ita"]), "direita")
        self.assertEqual(smart_join([" Que", " jogada", " insana", "!"]), "Que jogada insana!")

    def test_faster_whisper_words(self):
        self.assertEqual(smart_join(["Que", "jogada", "insana"]), "Que jogada insana")

    def test_punctuation_and_dash(self):
        self.assertEqual(smart_join(["portal", "."]), "portal.")
        self.assertEqual(smart_join(["-", "Vai", "dar"]), "- Vai dar")
        self.assertEqual(smart_join([" ", "", "oi"]), "oi")

    def test_used_in_cues(self):
        words = [W(" dire", 10.0, 10.4), W("ita", 10.4, 10.8)]
        srt = build_srt(words, 0.0, 30.0)
        self.assertIn("DIREITA", srt)
        self.assertNotIn("DIRE ITA", srt)


class TestHook(unittest.TestCase):
    def test_hook_present_with_timing(self):
        words = [W("fala", 20.0, 21.0)]
        a = build_ass(words, 10.0, 60.0, 1080, 1920, hook_title="Pior escolha da vida")
        dialogues = [l for l in a.splitlines() if l.startswith("Dialogue")]
        hook = [d for d in dialogues if ",Hook," in d]
        self.assertEqual(len(hook), 1)
        self.assertIn("0:00:00.30,0:00:05.00", hook[0])
        self.assertIn("ESCOLHA", hook[0])

    def test_no_hook_without_title(self):
        words = [W("fala", 20.0, 21.0)]
        for t in ("", None):
            a = build_ass(words, 10.0, 60.0, 1080, 1920, hook_title=t)
            self.assertNotIn(",Hook,", a)

    def test_no_hook_on_short_clip(self):
        words = [W("fala", 10.5, 11.0)]
        a = build_ass(words, 10.0, 12.0, 1080, 1920, hook_title="Titulo Longo Aqui")
        self.assertNotIn(",Hook,", a)

    def test_hook_max_three_lines(self):
        lines = build_hook_lines("Essa foi a pior escolha da minha vida inteira mesmo")
        self.assertLessEqual(len(lines), 3)
        self.assertTrue(all(len(l) <= 16 for l in lines))

    def test_hook_inside_top_blur_band(self):
        from core.video import HOOK_MARGIN_V, HOOK_FONT_SIZE, HOOK_END_SEC
        from core.video import LAYOUT_H, LAYOUT_MAIN_H
        band = (LAYOUT_H - LAYOUT_MAIN_H) // 2  # 360
        self.assertLess(HOOK_MARGIN_V + 2 * HOOK_FONT_SIZE, band)
        self.assertGreaterEqual(HOOK_END_SEC, 4.0)


class TestDoubleValidation(unittest.TestCase):
    def test_clean_cues_no_warnings(self):
        words = [W("a", 11.0, 11.4), W("b", 11.5, 12.0)]
        cues = [(1.0, 1.4, "A"), (1.5, 2.0, "B")]
        self.assertEqual(validate_cues(words, 10.0, 20.0, cues), [])

    def test_negative_and_overflow_warned(self):
        cues = [(-0.5, 0.5, "X"), (9.0, 12.0, "Y"), (5.0, 4.0, "Z")]
        warns = validate_cues([], 10.0, 20.0, cues)
        self.assertEqual(len(warns), 3)

    def test_uncovered_word_warned(self):
        words = [W("perdida", 15.0, 15.5)]
        warns = validate_cues(words, 10.0, 20.0, [(0.0, 1.0, "OUTRA")])
        self.assertTrue(any("perdida" in w for w in warns))

    def test_debug_report_sections(self):
        words = [W("Olhe", 10.0, 10.3), W("vaso", 14.5, 15.0)]
        cues = [(0.0, 0.3, "OLHE"), (4.5, 5.0, "VASO")]
        txt = render_caption_debug("clip", 10.0, 20.0, words, cues, [])
        for section in ("TRANSCRIÇÃO ORIGINAL [ABS]", "TIMESTAMP RELATIVO",
                        "TIMESTAMP NO ARQUIVO", "CHECAGEM FINAL", "OK:"):
            self.assertIn(section, txt)
        self.assertIn("00:01:32", render_caption_debug("c", 92.4, 108.9, [], [], []))


class TestMultiSilenceAndLines(unittest.TestCase):
    def test_multi_silence_three_cues(self):
        words = [W("um", 0.0, 0.5), W("dois", 5.0, 5.5), W("tres", 12.0, 12.5)]
        cues = parse_cues(build_srt(words, 0.0, 30.0))
        self.assertEqual(len(cues), 3)
        starts = [s for s, _, _ in cues]
        self.assertAlmostEqual(starts[1] - cues[0][1], 4.0, places=1)

    def test_line_break_timing_split(self):
        words = [W(f"w{i}", 10.0 + i * 0.5, 10.0 + i * 0.5 + 0.4) for i in range(10)]
        cues = parse_cues(build_srt(words, 10.0, 30.0))
        # 10 palavras em 2 linhas → tempos divididos, sem sobreposição invertida
        self.assertGreaterEqual(len(cues), 2)
        for s, e, _ in cues:
            self.assertGreater(e, s)

    def test_segment_only_uniform_words(self):
        # Sem timestamps individuais: palavras distribuídas cobrem o span.
        words = [W(f"w{i}", 20.0 + i, 21.0 + i) for i in range(5)]
        cues = parse_cues(build_srt(words, 10.0, 40.0))
        self.assertTrue(cues)
        self.assertGreaterEqual(cues[0][0], 10.0 - 10.0 - 1e-6)
        self.assertLessEqual(cues[-1][1], 30.0 + 1e-6)


class TestDegenerateWords(unittest.TestCase):
    """Regra de domínio: só end > start gera caption (VAD 20.760→20.760)."""

    def test_zero_duration_generates_no_caption(self):
        self.assertEqual(build_srt([W("vou", 10.0, 10.0)], 0.0, 30.0), "")

    def test_negative_duration_generates_no_caption(self):
        self.assertEqual(build_srt([W("vou", 10.0, 9.9)], 0.0, 30.0), "")

    def test_short_but_valid_word_is_kept(self):
        srt = build_srt([W("ah", 10.0, 10.2)], 0.0, 30.0)
        self.assertIn("AH", srt)

    def test_normal_word_preserves_timestamp(self):
        cues = parse_cues(build_srt([W("mundo", 10.0, 11.5)], 0.0, 30.0))
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0][0], 10.0, places=2)
        self.assertAlmostEqual(cues[0][1], 11.5, places=2)

    def test_consecutive_degenerate_words_generate_nothing(self):
        words = [W("vou", 20.76, 20.76), W("aprender", 20.76, 20.76),
                 W("aí", 20.76, 20.76)]
        srt = build_srt(words, 0.0, 31.21)
        self.assertEqual(srt, "")
        self.assertNotIn("21,760", srt)

    def test_long_gap_has_no_cue_during_silence(self):
        words = [W("eu", 20.59, 20.76),
                 W("vou", 20.76, 20.76), W("aprender", 20.76, 20.76),
                 W("aí", 20.76, 20.76),
                 W("não", 30.0, 31.21)]
        cues = parse_cues(build_srt(words, 0.0, 31.21))
        self.assertTrue(cues)
        for s, e, _ in cues:
            self.assertGreater(e, s)
        starts = sorted(s for s, _, _ in cues)
        # Nenhuma cue pode começar dentro do silêncio (após o burst,
        # antes da fala real em 30.0), exceto a tolerância de 1.0s da
        # duração mínima visual aplicada à última palavra válida.
        for s in starts:
            self.assertTrue(s < 22.0 or s >= 30.0 - 1e-6,
                            f"cue artificial no silêncio: {s}")
        self.assertTrue(any(abs(s - 30.0) < 1e-6 for s in starts))

    def test_padding_is_not_drift(self):
        # speech 30.0→31.0, pad 0.8 → clip 29.2→31.8: caption ≈0.8, sem preencher o fim.
        words = [W("olá", 30.0, 30.4), W("mundo", 30.5, 31.0)]
        cues = parse_cues(build_srt(words, 29.2, 31.8))
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0][0], 0.8, places=2)
        self.assertLess(cues[0][1], 31.8 - 29.2 - 1e-6)

    def test_degenerate_words_are_not_uncovered(self):
        words = [W("ok", 10.0, 10.5), W("vou", 20.76, 20.76)]
        cues = [(10.0, 10.5, "OK")]
        self.assertEqual(validate_cues(words, 0.0, 30.0, cues), [])

    def test_debug_reports_dropped_degenerate(self):
        words = [W("ok", 10.0, 10.5), W("vou", 20.76, 20.76)]
        txt = render_caption_debug("c", 0.0, 30.0, words, [(10.0, 10.5, "OK")], [])
        self.assertIn("DROPPED DEGENERATE WORD", txt)
        self.assertIn("20.760", txt)


class TestMinDurationClamp(unittest.TestCase):
    """Extensão visual 1.0s nunca invade a próxima cue; span real intocado."""

    def test_extension_clamped_before_next_cue(self):
        words = [W("ah", 10.0, 10.1), W("vamos", 10.5, 11.0)]
        cues = parse_cues(build_srt(words, 0.0, 30.0))
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0][0], 10.0, places=2)
        self.assertLessEqual(cues[0][1], cues[1][0] + 1e-6)
        self.assertGreaterEqual(cues[0][1], 10.1 - 1e-6)  # span real preservado

    def test_real_duration_never_reduced(self):
        words = [W("fala", 10.0, 10.5), W("longa", 10.6, 12.5)]
        cues = parse_cues(build_srt(words, 0.0, 30.0))
        self.assertTrue(cues)
        self.assertAlmostEqual(cues[-1][1], 12.5, places=2)

    def test_real_whisper_overlap_preserved(self):
        # 8 words forçam split com overlap real (fim real além do próximo início).
        words = [W(f"w{i}", 10.0 + i * 0.1, 10.0 + i * 0.1 + 1.0) for i in range(10)]
        cues = parse_cues(build_srt(words, 0.0, 30.0))
        self.assertGreaterEqual(len(cues), 2)
        # fim real 11.7 da cue 0 além do início da cue 1 → verdade preservada
        self.assertGreater(cues[0][1], cues[1][0] - 1e-6)

    def test_no_overlap_in_real_clips(self):
        import json
        data = json.load(open("metrics_test/whisper_raw_words.json"))
        words = [W(w["text"], w["start_abs"], w["end_abs"]) for w in data]
        cues = parse_cues(build_srt(words, 0.0, 31.21))
        for i in range(len(cues) - 1):
            self.assertLessEqual(cues[i][1], cues[i + 1][0] + 1e-6)

    def test_audit_word_to_cue_present(self):
        words = [W("olhe", 10.0, 10.3), W("vaso", 14.5, 15.0)]
        cues = [(10.0, 10.3, "OLHE"), (14.5, 15.0, "VASO")]
        txt = render_caption_debug("c", 0.0, 30.0, words, cues, [])
        self.assertIn("AUDITORIA WORD→CUE", txt)
        self.assertIn("Cue 00", txt)
        self.assertIn("status: OK", txt)


class TestWordBoundaryGrouping(unittest.TestCase):
    """Agrupador nunca parte palavra (ex: 'dire'+'ita' com gap do Whisper)."""

    def test_gap_inside_word_keeps_together(self):
        words = [W(" melhor", 17.17, 17.56), W(" cen", 17.67, 17.78),
                 W("ário", 18.24, 18.31), W(" possível", 18.31, 19.0)]
        cues = _group_cues(words, 0.0, 30.0)
        texts = [c[2] for c in cues]
        self.assertIn("MELHOR CENÁRIO", texts)
        self.assertFalse(any(t == "CEN" for t in texts))
        self.assertFalse(any(t.startswith("ÁRIO") for t in texts))

    def test_pause_after_word_still_breaks(self):
        words = [W(" cen", 17.67, 17.78), W("ário", 18.24, 18.31),
                 W(" possível", 18.31, 19.0)]
        groups = _group_cue_words(words, 0.0, 30.0)
        self.assertEqual(len(groups), 2)  # pausa real respeitada após a palavra
        self.assertEqual([w.text for w in groups[0]], [" cen", "ário"])

    def test_multiple_continuations_stay_together(self):
        words = [W(" a", 1.0, 1.2), W("pic", 1.3, 1.5), W("aret", 1.6, 1.8),
                 W("ada", 2.4, 2.6), W(" fim", 2.7, 2.9)]
        groups = _group_cue_words(words, 0.0, 30.0)
        self.assertEqual([[w.text for w in g] for g in groups],
                         [[" a", "pic", "aret", "ada"], [" fim"]])

    def test_no_boundaries_keeps_history(self):
        # Estilo faster-whisper (sem espaços): comportamento histórico.
        words = [W("olhe", 10.0, 10.3), W("para", 10.9, 11.2)]
        groups = _group_cue_words(words, 0.0, 30.0)
        self.assertEqual(len(groups), 2)

    def test_srt_never_shows_split_word(self):
        words = [W(" pra", 302.0, 302.2), W(" dire", 302.26, 302.63),
                 W("ita", 302.63, 302.9)]
        srt = build_srt(words, 300.0, 330.0)
        self.assertIn("DIREITA", srt)
        self.assertNotIn("DIRE\n", srt)
        self.assertNotIn(" ITA", srt.replace("DIREITA", ""))


class TestSimultaneousSpeech(unittest.TestCase):
    """Falas sobrepostas viram cues coexistentes, nunca sequência."""

    def test_criterion_two_overlapping_cues(self):
        words = [W(" EU", 10.0, 10.5), W(" VOU", 10.5, 11.0),
                 W(" NÃO", 10.2, 10.5), W(" VAI", 10.5, 10.8)]
        cues = _group_cues(words, 0.0, 30.0)
        texts = sorted(c[2] for c in cues)
        self.assertEqual(texts, ["EU VOU", "NÃO VAI"])
        spans = sorted((c[0], c[1]) for c in cues)
        self.assertAlmostEqual(spans[0][0], 10.0, places=2)
        self.assertAlmostEqual(spans[1][0], 10.2, places=2)
        self.assertLess(spans[1][0], spans[0][1])  # sobrepõem de verdade

    def test_partial_overlap(self):
        words = [W(" isso", 10.0, 10.4), W(" é", 10.4, 10.8),
                 W(" não", 10.5, 10.7)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 2)
        self.assertTrue(any(c[2] == "NÃO" for c in cues))

    def test_full_overlap(self):
        words = [W(" sim", 10.0, 11.0), W(" não", 10.0, 11.0)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 2)

    def test_three_simultaneous(self):
        words = [W(" um", 10.0, 11.0), W(" dois", 10.1, 10.9),
                 W(" três", 10.2, 10.8)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 3)

    def test_no_overlap_single_sequence(self):
        words = [W(" olhe", 10.0, 10.3), W(" para", 10.4, 10.8)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "OLHE PARA")

    def test_touching_words_stay_sequential(self):
        words = [W(" dire", 10.0, 10.5), W("ita", 10.5, 10.9)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "DIREITA")

    def test_consecutive_no_overlap(self):
        words = [W(" foi", 10.0, 10.3), W(" embora", 11.0, 11.4)]
        cues = _group_cues(words, 0.0, 30.0)
        self.assertEqual(len(cues), 2)
        self.assertLessEqual(cues[0][1], cues[1][0] + 1e-6)

    def test_timestamps_preserved(self):
        words = [W(" EU", 10.0, 10.5), W(" VOU", 10.5, 11.0),
                 W(" NÃO", 10.2, 10.5), W(" VAI", 10.5, 10.8)]
        cues = _group_cues(words, 0.0, 30.0)
        for s, e, _ in cues:
            self.assertGreater(e, s)

    def test_srt_keeps_both_overlapping(self):
        words = [W(" EU", 10.0, 10.5), W(" NÃO", 10.2, 10.8)]
        srt = build_srt(words, 0.0, 30.0)
        self.assertIn("EU", srt)
        self.assertIn("NÃO", srt)
        parsed = parse_cues(srt)
        self.assertEqual(len(parsed), 2)
        self.assertLess(parsed[1][0], parsed[0][1])


if __name__ == "__main__":
    unittest.main()
