"""tests/test_backends.py — comando whisper sem VAD + regras de extração.

Roda sem binário: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backends import _whisper_cmd, CHUNK_SECONDS, SAMPLE_RATE


class TestWhisperCmd(unittest.TestCase):
    def test_no_vad_flags(self):
        cmd = _whisper_cmd("whisper-cli", Path("models/ggml-medium.bin"),
                           Path("/tmp/c.wav"), Path("/tmp/c"), "pt")
        joined = " ".join(cmd)
        self.assertNotIn("--vad", joined)
        self.assertNotIn("-vm", joined.split())
        self.assertNotIn("silero", joined.lower())

    def test_word_timestamps_json(self):
        cmd = _whisper_cmd("whisper-cli", Path("models/ggml-medium.bin"),
                           Path("/tmp/c.wav"), Path("/tmp/c"), None)
        self.assertIn("-ojf", cmd)
        self.assertIn("-nt", cmd)
        self.assertIn("auto", cmd)  # idioma padrão

    def test_chunk_constants(self):
        self.assertEqual(CHUNK_SECONDS, 30)
        self.assertEqual(SAMPLE_RATE, 16000)


if __name__ == "__main__":
    unittest.main()
