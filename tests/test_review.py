"""tests/test_review.py — revisão humana da transcrição (§15 do plano).

Roda sem dependências externas: python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Word, Segment
from core import review


def W(text, start, end):
    return Word(text, start, end)


def S(*words):
    words = list(words)
    return Segment(text=" ".join(w.text.strip() for w in words),
                   start=min(w.start for w in words),
                   end=max(w.end for w in words), words=list(words))


class TestTextCorrectionKeepsTimestamps(unittest.TestCase):
    def test_dire_ita_to_direita(self):
        # Usuário apaga as 2 linhas e escreve 1 cobrindo o mesmo span.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            p.write_text(
                "[s00] [00:12.340 → 00:12.700] dire\n"
                "[s00] [00:12.700 → 00:13.100] ita\n", encoding="utf-8")
            p.write_text("[s00] [00:12.340 → 00:13.100] direita\n", encoding="utf-8")
            segs = review.parse_words_txt(p)
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0].words[0].text, "direita")
            self.assertAlmostEqual(segs[0].words[0].start, 12.34, places=2)
            self.assertAlmostEqual(segs[0].words[0].end, 13.10, places=2)

    def test_word_fix_keeps_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            review.dump_words_txt([S(W("Muto", 5.0, 5.6))], p)
            p.write_text(p.read_text(encoding="utf-8").replace("Muto", "Muto"),
                         encoding="utf-8")
            segs = review.parse_words_txt(p)
            self.assertEqual(segs[0].words[0].text, "Muto")
            self.assertAlmostEqual(segs[0].words[0].start, 5.0)
            self.assertAlmostEqual(segs[0].words[0].end, 5.6)

    def test_timestamp_edit_keeps_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            review.dump_words_txt([S(W("vaso", 3.42, 4.12))], p)
            txt = p.read_text(encoding="utf-8").replace("03.420", "03.500")
            p.write_text(txt, encoding="utf-8")
            segs = review.parse_words_txt(p)
            self.assertEqual(segs[0].words[0].text, "vaso")
            self.assertAlmostEqual(segs[0].words[0].start, 3.5, places=2)
            self.assertAlmostEqual(segs[0].words[0].end, 4.12, places=2)

    def test_malformed_line_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            p.write_text("isso não é uma linha válida\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                review.parse_words_txt(p)

    def test_end_before_start_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            p.write_text("[00:05.000 → 00:04.000] invertido\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                review.parse_words_txt(p)

    def test_roundtrip_preserves_all(self):
        segs = [S(W("Olhe", 1.24, 1.60), W("aqui", 1.61, 2.00)),
                S(W("vaso", 3.42, 4.12))]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "words.txt"
            review.dump_words_txt(segs, p)
            back = review.parse_words_txt(p)
            self.assertEqual(len(back), 2)
            for orig, got in zip([w for s in segs for w in s.words],
                                 [w for s in back for w in s.words]):
                self.assertEqual(got.text.strip(), orig.text.strip())
                self.assertAlmostEqual(got.start, orig.start, places=2)
                self.assertAlmostEqual(got.end, orig.end, places=2)


class TestApproval(unittest.TestCase):
    def test_draft_is_not_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "transcript.json"
            review.save_transcript([S(W("a", 0.0, 1.0))], p, status=review.DRAFT)
            with self.assertRaises(RuntimeError):
                review.require_approved(p)

    def test_approved_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            review.save_transcript([S(W("a", 0.0, 1.0))], session / "transcript.json")
            final = review.approve(session, [S(W("b", 0.0, 1.0))])
            segs = review.require_approved(final)
            self.assertEqual(segs[0].words[0].text, "b")
            self.assertTrue((session / "transcript.raw.json").exists())


class TestCustomWords(unittest.TestCase):
    def test_exact_word_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cw.json"
            p.write_text('{"words": ["Muto"]}', encoding="utf-8")
            mapping = review.load_custom_words(str(p))
            segs = [S(W("muto", 0.0, 1.0), W("mutola", 1.0, 2.0))]
            n = review.apply_custom_words(segs, mapping)
            self.assertEqual(n, 1)
            self.assertEqual(segs[0].words[0].text, "Muto")
            self.assertEqual(segs[0].words[1].text, "mutola")  # substring intacta
            self.assertAlmostEqual(segs[0].words[0].start, 0.0)

    def test_invalid_file_fails(self):
        with self.assertRaises(RuntimeError):
            review.load_custom_words("/nao/existe.json")


class TestGrounding(unittest.TestCase):
    TEXT = "o cara tá ganhando e tu tá querendo ver ele morrer"

    def test_supported_title_passes(self):
        self.assertEqual(review.validate_grounding(
            "TU TÁ QUERENDO VER ELE MORRER", self.TEXT), [])

    def test_invented_word_flagged(self):
        bad = review.validate_grounding("OPINIÃO POLÊMICA CONTRA NEGROS", self.TEXT)
        self.assertIn("negros", bad)
        self.assertIn("polêmica", bad)

    def test_highlight_must_exist(self):
        self.assertEqual(review.validate_highlights(
            "TU TÁ QUERENDO VER ELE MORRER", self.TEXT), [])
        # "pior" tem len<5 e nunca é destacado (regra do video.py); usa longa.
        bad = review.validate_highlights("MOMENTO INACREDITÁVEL", self.TEXT)
        self.assertIn("inacreditável", bad)


if __name__ == "__main__":
    unittest.main()
