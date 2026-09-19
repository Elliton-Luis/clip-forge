"""tests/test_acoustic.py — clipping conservador + fusão com cues (Parte 2).

Roda sem ffmpeg (só funções puras): python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.acoustic import (AcousticEvent, frame_hot_flags, find_events,
                           signal_stats, event_cues, FRAME_SEC)
from core.models import Word
from core.video import merge_event_cues, build_srt


def hot_sample(peak=1.0, rms=0.5):
    # 1 frame de 50ms @16k: vale constante p/ rms alvo + 1 pico p/ peak alvo
    import math
    n = 800
    return [peak] + [rms] * (n - 1) if peak >= rms else [rms] * n


class TestFrames(unittest.TestCase):
    def test_hot_needs_peak_and_rms(self):
        self.assertEqual(frame_hot_flags(hot_sample(1.0, 0.5)), [True])
        self.assertEqual(frame_hot_flags(hot_sample(0.5, 0.5)), [False])   # sem teto
        self.assertEqual(frame_hot_flags(hot_sample(1.0, 0.01)), [False])  # pico isolado

    def test_stats(self):
        s = signal_stats([1.0, -1.0, 0.0, 0.0])
        self.assertAlmostEqual(s["peak"], 1.0)
        self.assertGreater(s["saturated_ratio"], 0)
        self.assertEqual(signal_stats([])["peak"], 0.0)


class TestEvents(unittest.TestCase):
    def test_short_burst_ignored(self):
        self.assertEqual(find_events([True] * 3), [])  # 0.15s < 0.3s

    def test_sustained_detected_with_confidence(self):
        evs = find_events([True] * 10)  # 0.5s
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].type, "clipped-audio")
        self.assertAlmostEqual(evs[0].confidence, 1.0)
        self.assertAlmostEqual(evs[0].end - evs[0].start, 0.5, places=2)

    def test_bridge_merges(self):
        hot = [True] * 4 + [False] * 2 + [True] * 4  # gap 0.1s <= 0.2
        self.assertEqual(len(find_events(hot)), 1)

    def test_long_gap_splits(self):
        hot = [True] * 6 + [False] * 10 + [True] * 6  # gap 0.5s
        self.assertEqual(len(find_events(hot)), 2)

    def test_offset_applied(self):
        evs = find_events([True] * 10, offset=60.0)
        self.assertAlmostEqual(evs[0].start, 60.0, places=2)


class TestEventCues(unittest.TestCase):
    def test_clip_and_floor(self):
        evs = [AcousticEvent("clipped-audio", 5.0, 5.2, 0.8)]  # 0.2s → piso 1s
        cues = event_cues(evs, 0.0, 30.0)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0][2], "*ÁUDIO ESTOURADO")
        self.assertAlmostEqual(cues[0][1] - cues[0][0], 1.0, places=2)

    def test_outside_dropped(self):
        evs = [AcousticEvent("clipped-audio", 100.0, 101.0, 0.9)]
        self.assertEqual(event_cues(evs, 0.0, 30.0), [])

    def test_never_past_clip_end(self):
        evs = [AcousticEvent("clipped-audio", 29.8, 30.5, 0.9)]
        cues = event_cues(evs, 0.0, 30.0)
        self.assertLessEqual(cues[0][1], 30.0)


class TestMerge(unittest.TestCase):
    def test_word_wins_and_no_overlap(self):
        word_cues = [(10.0, 12.0, "FALA")]
        event_cues_in = [(11.0, 13.0, "*ÁUDIO ESTOURADO")]
        merged = merge_event_cues(word_cues, event_cues_in)
        self.assertEqual(merged[0], (10.0, 12.0, "FALA"))
        self.assertIn((12.0, 13.0, "*ÁUDIO ESTOURADO"), merged)
        for (s1, e1, _), (s2, e2, _) in zip(merged, merged[1:]):
            self.assertLessEqual(e1, s2)

    def test_event_split_in_two(self):
        merged = merge_event_cues([(10.0, 11.0, "FALA")], [(9.0, 12.0, "*E*")])
        texts = [t for _, _, t in merged]
        self.assertEqual(texts.count("*E*"), 2)

    def test_empty_event_unchanged(self):
        wc = [(10.0, 12.0, "FALA")]
        self.assertEqual(merge_event_cues(wc, []), wc)

    def test_srt_contains_event(self):
        words = [Word("fala", 10.0, 11.0)]
        srt = build_srt(words, 0.0, 30.0, event_cues=[(20.0, 21.0, "*ÁUDIO ESTOURADO")])
        self.assertIn("*ÁUDIO ESTOURADO", srt)
        self.assertIn("FALA", srt)

    def test_srt_without_events_unchanged(self):
        words = [Word("fala", 10.0, 11.0)]
        a = build_srt(words, 0.0, 30.0)
        b = build_srt(words, 0.0, 30.0, event_cues=None)
        self.assertEqual(a, b)


class TestWordEnergy(unittest.TestCase):
    def test_rms_between(self):
        from core.acoustic import rms_between, SAMPLE_RATE
        samples = [0.0] * SAMPLE_RATE + [0.5] * SAMPLE_RATE  # 1s silêncio + 1s tom
        self.assertAlmostEqual(rms_between(samples, 0.0, 1.0), 0.0)
        self.assertAlmostEqual(rms_between(samples, 1.0, 2.0), 0.5)
        self.assertEqual(rms_between(samples, 5.0, 4.0), 0.0)  # intervalo inválido

    def test_word_energy_flags_silence(self):
        import core.acoustic as ac
        from unittest.mock import patch
        # 2s: silêncio + tom 0.5
        samples = [0.0] * 32000 + [0.5] * 32000
        words = [Word(" oi", 0.0, 1.0), Word(" fala", 1.0, 2.0)]
        with patch.object(ac, "decode_pcm", return_value=samples):
            out = ac.word_energy("x.mp4", words)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["silent"])   # word afirma fala no silêncio
        self.assertFalse(out[1]["silent"])
        self.assertEqual(out[1]["text"], "fala")

    def test_word_energy_decode_failure_returns_empty(self):
        import core.acoustic as ac
        from unittest.mock import patch
        with patch.object(ac, "decode_pcm", side_effect=RuntimeError("ffmpeg")):
            self.assertEqual(ac.word_energy("x.mp4", [Word("a", 0.0, 1.0)]), [])


if __name__ == "__main__":
    unittest.main()
