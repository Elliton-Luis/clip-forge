"""tests/test_audio.py — energia só-áudio (sem ffmpeg real).

Roda sem dependências externas: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import audio as au


def _run(cmd, **kw):
    raise AssertionError("ffmpeg real não deve rodar em teste")


def _resp(stderr):
    class R:
        pass
    r = R()
    r.stderr = stderr
    return r


VOL_ALTA = "n: 100\nmean_volume: -5.0 dB\nmax_volume: -3.0 dB\n"
VOL_MEDIA = "n: 100\nmean_volume: -20.0 dB\nmax_volume: -12.0 dB\n"
VOL_BAIXA = "n: 100\nmean_volume: -30.0 dB\nmax_volume: -25.0 dB\n"


class TestMeasureCommand(unittest.TestCase):
    def test_only_audio_stream_selected(self):
        seen = {}

        def fake(cmd, **kw):
            seen["cmd"] = cmd
            return _resp(VOL_MEDIA)

        with patch("core.audio.subprocess.run", side_effect=fake):
            self.assertEqual(au.measure("v.mp4", 10.0, 30.0), "media")
        cmd = seen["cmd"]
        self.assertIn("-vn", cmd)
        self.assertIn("0:a:0", cmd)
        self.assertLess(cmd.index("-i"), cmd.index("-vn"))  # saída, após input
        self.assertIn("volumedetect", cmd)
        self.assertIn("-f", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "10.0")
        self.assertEqual(cmd[cmd.index("-t") + 1], "20.0")

    def test_thresholds(self):
        cases = [(VOL_ALTA, "alta"), (VOL_MEDIA, "media"), (VOL_BAIXA, "baixa")]
        for stderr, want in cases:
            with patch("core.audio.subprocess.run",
                        return_value=_resp(stderr)):
                self.assertEqual(au.measure("v.mp4", 0, 10), want)

    def test_no_volume_means_media(self):
        with patch("core.audio.subprocess.run", return_value=_resp("nada útil")):
            self.assertEqual(au.measure("v.mp4", 0, 10), "media")

    def test_timeout_means_media(self):
        import subprocess
        with patch("core.audio.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("ffmpeg", 8)):
            self.assertEqual(au.measure("v.mp4", 0, 10), "media")


class TestScoresEnergyVersion(unittest.TestCase):
    def test_old_cache_rejected(self):
        import json
        import tempfile
        from core.models import Candidate
        cands = [Candidate(start=0, end=20, text="oi", words=[], energy="x")]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fp.scores.json"
            p.write_text(json.dumps({"scores": [
                {"score": 9, "reason": "r", "title": "t",
                 "hashtags": "#a #b #c", "failed": False,
                 "energy": "media", "speech_rate": 1.0}]}),
                encoding="utf-8")
            self.assertFalse(
                __import__("core.cache", fromlist=["load_scores"])
                .load_scores(Path(tmp), "fp", cands))
            self.assertEqual(cands[0].energy, "x")  # intocado

    def test_current_version_accepted(self):
        from core.cache import save_scores, load_scores
        from core.models import Candidate
        import tempfile
        cands = [Candidate(start=0, end=20, text="oi", words=[],
                           score=8, energy="alta", speech_rate=2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            save_scores(Path(tmp), "fp", cands)
            cands[0].energy = "?"
            self.assertTrue(load_scores(Path(tmp), "fp", cands))
            self.assertEqual(cands[0].energy, "alta")


if __name__ == "__main__":
    unittest.main()
