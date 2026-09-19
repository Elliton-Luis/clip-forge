"""tests/test_translab.py — métricas, comparação e regras do lab de transcrição.

Roda sem dependências externas (sem whisper, sem ffmpeg):
python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Word, Segment
from core import translab as lab


def W(text, start, end, p=None):
    w = Word(text, start, end)
    w.prob = p
    return w


def S(*words):
    words = list(words)
    text = " ".join(w.text.strip() for w in words)
    s = min((w.start for w in words), default=0.0)
    e = max((w.end for w in words), default=0.0)
    return Segment(text=text, start=s, end=e, words=words)


class TestMetrics(unittest.TestCase):
    def test_counts_and_ratio(self):
        segs = [S(W("a", 1.0, 1.5), W("b", 1.5, 1.5), W("c", 1.6, 2.0))]
        m = lab.compute_metrics(segs)
        self.assertEqual(m["word_count"], 3)
        self.assertEqual(m["degenerate_word_count"], 1)
        self.assertAlmostEqual(m["degenerate_ratio"], 1 / 3, places=4)
        self.assertEqual(m["segment_count"], 1)
        self.assertAlmostEqual(m["timestamp_min"], 1.0)
        self.assertAlmostEqual(m["timestamp_max"], 2.0)

    def test_empty(self):
        m = lab.compute_metrics([])
        self.assertEqual(m["word_count"], 0)
        self.assertEqual(m["degenerate_ratio"], 0.0)
        self.assertIsNone(m["timestamp_min"])

    def test_overlap_and_duplicates(self):
        segs = [S(W("a", 1.0, 2.0), W("b", 1.5, 1.8), W("c", 1.5, 1.8))]
        m = lab.compute_metrics(segs)
        self.assertEqual(m["overlap_count"], 2)  # b sobre a, c sobre b
        self.assertEqual(m["duplicate_timestamp_count"], 1)  # b e c idênticos

    def test_probs_optional(self):
        m = lab.compute_metrics([S(W("a", 0.0, 1.0))])
        self.assertIsNone(m["avg_prob"])
        m = lab.compute_metrics([S(W("a", 0.0, 1.0, 0.5), W("b", 1.0, 2.0, 0.9))])
        self.assertAlmostEqual(m["avg_prob"], 0.7)
        self.assertAlmostEqual(m["min_prob"], 0.5)


class TestPresets(unittest.TestCase):
    def test_known_presets(self):
        self.assertIsNone(lab.resolve_preset("original"))
        self.assertIn("loudnorm", lab.resolve_preset("normalize"))
        self.assertIn("highpass", lab.resolve_preset("clean"))

    def test_unknown_preset_fails_clearly(self):
        with self.assertRaises(ValueError):
            lab.resolve_preset("magia")


class TestClipToWindow(unittest.TestCase):
    def test_context_words_dropped(self):
        seg = S(W("antes", 8.0, 9.0), W("dentro", 10.5, 11.0), W("depois", 30.5, 31.0))
        kept = lab.clip_to_window([seg], 10.0, 30.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual([w.text for w in kept[0].words], ["dentro"])

    def test_end_clipped_at_border(self):
        kept = lab.clip_to_window([S(W("x", 29.5, 31.0))], 0.0, 30.0)
        self.assertAlmostEqual(kept[0].words[0].end, 30.0)

    def test_empty_segment_dropped(self):
        self.assertEqual(lab.clip_to_window([S(W("x", 40.0, 41.0))], 0.0, 30.0), [])


class TestCompare(unittest.TestCase):
    def D(self, text, s, e):
        return {"text": text, "start": s, "end": e}

    def test_identical(self):
        a = [self.D("olhe", 1.2, 1.5), self.D("aqui", 1.5, 1.8)]
        r = lab.compare_word_lists(a, [dict(w) for w in a])
        self.assertEqual(r["summary"]["matched"], 2)
        self.assertEqual(r["summary"]["mean_abs_d_start"], 0.0)

    def test_timing_shift_measured_not_averaged(self):
        a = [self.D("olhe", 1.20, 1.48)]
        b = [self.D("olhe", 1.25, 1.51)]
        r = lab.compare_word_lists(a, b)
        row = r["rows"][0]
        self.assertEqual(row["type"], "match")
        self.assertAlmostEqual(row["d_start"], 0.05)
        self.assertAlmostEqual(row["d_end"], 0.03)

    def test_missing_and_extra(self):
        a = [self.D("olhe", 1.0, 1.2), self.D("para", 1.2, 1.4)]
        b = [self.D("olhe", 1.0, 1.2), self.D("lá", 1.2, 1.4)]
        r = lab.compare_word_lists(a, b)
        types = sorted(x["type"] for x in r["rows"])
        self.assertIn("missing_in_b", types)
        self.assertIn("extra_in_b", types)

    def test_case_and_punct_insensitive(self):
        a = [self.D("Olhe,", 1.0, 1.2)]
        b = [self.D("olhe", 1.0, 1.2)]
        r = lab.compare_word_lists(a, b)
        self.assertEqual(r["summary"]["matched"], 1)


class TestWordsDump(unittest.TestCase):
    def test_format(self):
        out = lab.words_dump([S(W("OLHE", 10.2, 10.8))])
        self.assertIn("[00:10.200 → 00:10.800] OLHE", out)


if __name__ == "__main__":
    unittest.main()
