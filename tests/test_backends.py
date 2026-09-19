"""tests/test_backends.py — comando whisper sem VAD + regras de extração.

Roda sem binário: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backends import _whisper_cmd, _resolve_model_file, CHUNK_SECONDS, SAMPLE_RATE


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

    def test_extra_decode_flags(self):
        cmd = _whisper_cmd("whisper-cli", Path("models/ggml-medium.bin"),
                           Path("/tmp/c.wav"), Path("/tmp/c"), "pt",
                           extra=["-tp", "0.2"])
        self.assertIn("-tp", cmd)
        self.assertIn("0.2", cmd)
        self.assertNotIn("--vad", " ".join(cmd))

    def test_resolve_model_file(self):
        self.assertIsNone(_resolve_model_file("inexistente-xyz"))
        found = _resolve_model_file("medium")
        self.assertIsNotNone(found)
        self.assertTrue(str(found).endswith("ggml-medium.bin"))


if __name__ == "__main__":
    unittest.main()
