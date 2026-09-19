"""tests/test_candidates.py — range de duração dos clips (Parte 1).

Roda sem dependências externas: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import Candidate, Segment, Word
from core.candidates import build, snap_all, _snap_one


def W(t, s, e):
    return Word(t, s, e)


def segs_spans(spans):
    """Segmentos sintéticos [(start, end)] com 1 word cada (suficiente p/ build)."""
    out = []
    for s, e in spans:
        w = W("fala", s, e)
        out.append(Segment(text="fala", start=s, end=e, words=[w]))
    return out


def cand(start, end, nwords=4):
    ws = [W(f"w{i}", start + i * (end - start) / nwords,
            start + (i + 1) * (end - start) / nwords) for i in range(nwords)]
    return Candidate(start=start, end=end, text=" ".join(w.text for w in ws), words=ws)


class TestBuildRange(unittest.TestCase):
    def test_windows_obey_max(self):
        segs = segs_spans([(i, i + 10) for i in range(0, 200, 10)])
        for c in build(segs, min_dur=20, max_dur=60):
            self.assertLessEqual(c.duration, 60)

    def test_windows_obey_min(self):
        segs = segs_spans([(i, i + 10) for i in range(0, 200, 10)])
        for c in build(segs, min_dur=20, max_dur=60):
            self.assertGreaterEqual(c.duration, 20)

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            build(segs_spans([(0, 30)]), min_dur=60, max_dur=20)

    def test_build_caps_at_media_end(self):
        # Overshoot do decoder: segmento termina em 60 num vídeo de 45.14.
        segs = segs_spans([(0, 30), (30, 60)])
        for c in build(segs, min_dur=20, max_dur=40, media_end=45.14):
            self.assertLessEqual(c.end, 45.14)

    def test_build_without_media_end_unchanged(self):
        segs = segs_spans([(0, 30), (30, 60)])
        self.assertTrue(any(c.end > 45.14
                            for c in build(segs, min_dur=20, max_dur=40)))


class TestSnapRange(unittest.TestCase):
    def test_below_min_expands(self):
        c = cand(50, 60)  # 10s
        _snap_one(c, pad=0.8, min_dur=20, max_dur=60, media_end=200)
        self.assertAlmostEqual(c.duration, 20, places=2)

    def test_in_range_kept(self):
        c = cand(50, 81)  # 31s
        _snap_one(c, pad=0.8, min_dur=20, max_dur=60, media_end=200)
        self.assertGreaterEqual(c.duration, 20)
        self.assertLessEqual(c.duration, 60)

    def test_above_max_clamped(self):
        c = cand(0, 100)
        _snap_one(c, pad=0.8, min_dur=20, max_dur=60, media_end=200)
        self.assertLessEqual(c.duration, 60)

    def test_pad_cannot_break_max(self):
        c = cand(0, 89)  # +0.8 pad estouraria 90 sem clamp
        _snap_one(c, pad=0.8, min_dur=20, max_dur=90, media_end=200)
        self.assertLessEqual(c.duration, 90)

    def test_near_start_floors_at_zero(self):
        c = cand(2, 12)  # precisa expandir, início perto de 0
        _snap_one(c, pad=0, min_dur=20, max_dur=60, media_end=200)
        self.assertGreaterEqual(c.start, 0)
        self.assertAlmostEqual(c.duration, 20, places=2)

    def test_near_end_capped_by_media(self):
        c = cand(185, 195)  # fim perto de media_end=200
        _snap_one(c, pad=0, min_dur=20, max_dur=60, media_end=200)
        self.assertLessEqual(c.end, 200)
        self.assertLessEqual(c.duration, 60)

    def test_impossible_min_stays_short(self):
        # mídia de 15s < min 20: mantém curto, nunca estoura max p/ compensar
        c = cand(0, 15)
        _snap_one(c, pad=0, min_dur=20, max_dur=60, media_end=15)
        self.assertLessEqual(c.duration, 15)
        self.assertLessEqual(c.duration, 60)

    def test_snap_caps_end_at_media_end(self):
        # Caso real: janela [29.2, 60.8] em vídeo de 45.14s.
        c = cand(29.2, 60.8)
        _snap_one(c, pad=0.8, min_dur=20, max_dur=40, media_end=45.14)
        self.assertLessEqual(c.end, 45.14)
        self.assertGreaterEqual(c.start, 0)
        self.assertGreaterEqual(c.duration, 0)

    def test_snap_fully_outside_collapses_safely(self):
        c = cand(50, 60)
        _snap_one(c, pad=0.8, min_dur=20, max_dur=40, media_end=45.14)
        self.assertLessEqual(c.end, 45.14)
        self.assertGreaterEqual(c.duration, 0)

    def test_exact_boundaries_kept(self):
        c = cand(10, 30)  # exatamente min
        _snap_one(c, pad=0, min_dur=20, max_dur=60, media_end=200)
        self.assertAlmostEqual(c.duration, 20, places=2)
        c = cand(10, 70)  # exatamente max
        _snap_one(c, pad=0, min_dur=20, max_dur=60, media_end=200)
        self.assertAlmostEqual(c.duration, 60, places=2)


if __name__ == "__main__":
    unittest.main()
