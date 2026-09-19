"""tests/test_quality.py — detector de alucinação (puro, sem binário)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.quality import check, choose_retry


class TestDetector(unittest.TestCase):
    def test_platform_loop_flagged(self):
        text = ("e óleo de céu por toda a plataforma toda a plataforma "
                "toda a plataforma toda a plataforma")
        r = check(text)
        self.assertTrue(r["flagged"])
        self.assertEqual(r["reason"], "repeated_ngram")

    def test_music_notes_flagged(self):
        r = check(" ".join(["♪"] * 17))
        self.assertTrue(r["flagged"])
        self.assertEqual(r["reason"], "single_token_run")

    def test_real_emphasis_clean(self):
        # "Dá, dá, dá, dá" (4x) — fala real, longe do limiar 10
        r = check("Não dá não, dá não. Dá, dá, dá, dá, vai dar certo agora mesmo.")
        self.assertFalse(r["flagged"])

    def test_clean_transcript(self):
        r = check("Considerando que ele não pega vídeo, então eu vou só ficar "
                  "falando aqui e produzir um vídeo normal.")
        self.assertFalse(r["flagged"])

    def test_empty(self):
        self.assertFalse(check("")["flagged"])


class TestChooseRetry(unittest.TestCase):
    def test_retry_clears(self):
        self.assertEqual(choose_retry(True, False), "retry")

    def test_both_clean_keeps_original(self):
        self.assertEqual(choose_retry(False, False), "original")

    def test_both_flagged_keeps_original(self):
        self.assertEqual(choose_retry(True, True), "original")

    def test_original_clean_retry_flagged(self):
        self.assertEqual(choose_retry(False, True), "original")


if __name__ == "__main__":
    unittest.main()
