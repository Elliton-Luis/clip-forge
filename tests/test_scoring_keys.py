"""tests/test_scoring_keys.py — rodízio de chaves NVIDIA (sem rede).

Roda sem dependências externas: python -m unittest discover -s tests
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import scoring as sc


class TestKeyLoading(unittest.TestCase):
    def test_single_key_fallback(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEYS": "", "NVIDIA_API_KEY": "k1"}, clear=False):
            os.environ.pop("NVIDIA_API_KEYS", None)
            self.assertEqual(sc._load_api_keys(), ["k1"])

    def test_multi_key_list(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEYS": " k1 ,k2,, ", "NVIDIA_API_KEY": "k0"}):
            self.assertEqual(sc._load_api_keys(), ["k1", "k2"])

    def test_no_key_exits(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("NVIDIA_API_KEYS", "NVIDIA_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                sc._load_api_keys()

    def test_configured_keys_never_exits(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("NVIDIA_API_KEYS", "NVIDIA_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(sc._configured_keys(), [])


class TestRotation(unittest.TestCase):
    def setUp(self):
        sc._key_cursor = 0

    def tearDown(self):
        sc._key_cursor = 0

    def test_round_robin_order(self):
        clients = ["c0", "c1"]
        got = [sc._next_client(clients) for _ in range(5)]
        self.assertEqual([g[1] for g in got], [0, 1, 0, 1, 0])
        self.assertEqual([g[0] for g in got], ["c0", "c1", "c0", "c1", "c0"])

    def test_single_client_always_zero(self):
        self.assertEqual([sc._next_client(["c0"])[1] for _ in range(3)], [0, 0, 0])

    def test_retry_switches_key(self):
        used = []

        def fake_nim(client, model, batch, prompt):
            used.append(client)
            if len(used) == 1:
                raise RuntimeError("Error code: 503")
            return {"clips": []}, None

        with patch.object(sc, "_call_nim", side_effect=fake_nim):
            with patch("time.sleep", return_value=None):
                sc._call_with_retry(["c0", "c1"], "m", [], "p", retries=2,
                                    stats={"requests": 0, "successes": 0, "total_time": 0.0,
                                           "failures": 0, "retries": 0})
        self.assertEqual(used, ["c0", "c1"])

    def test_no_secret_in_logs(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()

        def fake_nim(client, model, batch, prompt):
            raise RuntimeError("boom")

        with patch.object(sc, "_call_nim", side_effect=fake_nim):
            with patch("time.sleep", return_value=None):
                with redirect_stdout(buf):
                    try:
                        sc._call_with_retry(["SECRET-A", "SECRET-B"], "m", [], "p",
                                            retries=1,
                                            stats={"requests": 0, "successes": 0, "total_time": 0.0,
                                                   "failures": 0, "retries": 0})
                    except RuntimeError:
                        pass
        self.assertNotIn("SECRET-A", buf.getvalue())
        self.assertNotIn("SECRET-B", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
