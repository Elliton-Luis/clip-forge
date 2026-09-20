"""tests/test_scoring_parallel.py — concorrência limitada do scoring (sem rede).

Workers = nº de chaves (teto 3); resultados sempre associados por id;
retry/429 passam pelo mesmo pacing. Roda offline com _call_nim mockado:
python -m unittest discover -s tests
"""
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import scoring as sc
from core.models import Candidate


def C(i):
    return Candidate(start=float(i * 20), end=float(i * 20 + 20),
                     text=f"fala do trecho {i}", words=[])


def ok_result(batch, score=7.5):
    return ({"clips": [{"id": idx, "score": score, "reason": "bom",
                        "title": f"Titulo {idx}", "hashtags": "#a #b #c"}
                       for idx, _ in batch]}, None)


def run_score(n_candidates, keys, nim_side_effect):
    env = {k: v for k, v in os.environ.items()
           if k not in ("NVIDIA_API_KEYS", "NVIDIA_API_KEY")}
    env["NVIDIA_API_KEYS"] = ",".join(keys)
    cands = [C(i) for i in range(n_candidates)]
    with patch.dict(os.environ, env, clear=True):
        with patch.object(sc, "_call_nim", side_effect=nim_side_effect):
            with patch.object(sc, "KEY_MIN_INTERVAL_SEC", 0.0):
                out = sc.score(cands, model="m")
    return out


class TestWorkerCount(unittest.TestCase):
    def test_one_key_serial_path(self):
        seen_workers = []

        def fake(client, model, batch, prompt):
            seen_workers.append(threading.current_thread().name)
            return ok_result(batch)

        out = run_score(12, ["k1"], fake)  # 2 lotes
        self.assertTrue(all(not c.failed for c in out))
        # caminho serial: tudo na thread principal
        self.assertTrue(all(w == "MainThread" for w in seen_workers))

    def test_three_keys_three_workers(self):
        active = {"cur": 0, "max": 0}
        lock = threading.Lock()

        def fake(client, model, batch, prompt):
            with lock:
                active["cur"] += 1
                active["max"] = max(active["max"], active["cur"])
            try:
                time.sleep(0.3)
                return ok_result(batch)
            finally:
                with lock:
                    active["cur"] -= 1

        t0 = time.monotonic()
        out = run_score(18, ["k1", "k2", "k3"], fake)  # 3 lotes
        wall = time.monotonic() - t0
        self.assertTrue(all(not c.failed for c in out))
        self.assertEqual(active["max"], 3)  # 3 simultâneas de verdade
        self.assertLess(wall, 0.75)  # serial levaria ~0.9s

    def test_two_keys_two_workers(self):
        active = {"cur": 0, "max": 0}
        lock = threading.Lock()

        def fake(client, model, batch, prompt):
            with lock:
                active["cur"] += 1
                active["max"] = max(active["max"], active["cur"])
            try:
                time.sleep(0.3)
                return ok_result(batch)
            finally:
                with lock:
                    active["cur"] -= 1

        out = run_score(18, ["k1", "k2"], fake)  # 3 lotes, 2 workers
        self.assertTrue(all(not c.failed for c in out))
        self.assertEqual(active["max"], 2)

    def test_key_affinity_round_robin(self):
        used = {}
        lock = threading.Lock()

        def fake(client, model, batch, prompt):
            with lock:
                used.setdefault(client, []).append(
                    sorted(idx for idx, _ in batch))
            return ok_result(batch)

        run_score(18, ["k1", "k2", "k3"], fake)
        # lote 0→chave A, 1→B, 2→C (ordem de envio, não de conclusão)
        self.assertEqual(len(used), 3)
        self.assertEqual(sorted(used.values()),
                         [[[0, 1, 2, 3, 4, 5]], [[6, 7, 8, 9, 10, 11]],
                          [[12, 13, 14, 15, 16, 17]]])


class TestOrdering(unittest.TestCase):
    def test_slow_first_batch_maps_correctly(self):
        def fake(client, model, batch, prompt):
            idxs = [idx for idx, _ in batch]
            if idxs[0] == 0:
                time.sleep(0.4)  # lote 0 termina por último
            return ok_result(batch, score=5.0 + idxs[0] // 6)

        out = run_score(18, ["k1", "k2", "k3"], fake)
        for i, c in enumerate(out):
            self.assertFalse(c.failed)
            self.assertEqual(c.title, f"Titulo {i}")
            self.assertAlmostEqual(c.score, 5.0 + i // 6)


class TestRetryAndRateLimit(unittest.TestCase):
    def test_retry_rotates_key_and_recovers(self):
        calls = []
        lock = threading.Lock()

        def fake(client, model, batch, prompt):
            with lock:
                calls.append(client)
                n = len(calls)
            if n <= 2:
                raise RuntimeError("Error code: 503")
            return ok_result(batch)

        out = run_score(6, ["k1", "k2", "k3"], fake)  # 1 lote
        self.assertTrue(all(not c.failed for c in out))
        self.assertEqual(len(calls), 3)  # 1 + 2 retries
        self.assertEqual(len(set(map(id, calls))), 3)  # 3 chaves distintas

    def test_429_honors_retry_after(self):
        sleeps = []

        class FakeResp:
            headers = {"retry-after": "7"}

        def fake(client, model, batch, prompt):
            if not sleeps:
                e = RuntimeError("Error code: 429 Too Many Requests")
                e.status_code = 429
                e.response = FakeResp()
                raise e
            return ok_result(batch)

        stats = {"requests": 0, "successes": 0, "total_time": 0.0,
                 "failures": 0, "retries": 0}
        with patch.object(sc, "_call_nim", side_effect=fake):
            with patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
                sc._call_with_retry(["c0"], "m", [], "p", retries=2, stats=stats)
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(stats.get("rate_limited"), 1)
        self.assertEqual(stats["successes"], 1)

    def test_429_without_header_uses_backoff(self):
        sleeps = []

        def fake(client, model, batch, prompt):
            if not sleeps:
                e = RuntimeError("429")
                raise e
            return ok_result(batch)

        with patch.object(sc, "_call_nim", side_effect=fake):
            with patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
                sc._call_with_retry(["c0"], "m", [], "p", retries=2, stats=None)
        self.assertEqual(sleeps, [10])

    def test_retry_never_duplicates_nor_loses(self):
        attempts = {"n": 0}

        def fake(client, model, batch, prompt):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("boom")
            return ok_result(batch)

        with patch("time.sleep", return_value=None):
            out = run_score(6, ["k1"], fake)
        self.assertEqual(attempts["n"], 3)
        self.assertTrue(all(not c.failed for c in out))
        self.assertEqual([c.title for c in out],
                         [f"Titulo {i}" for i in range(6)])

    def test_total_failure_marks_failed_once(self):
        def fake(client, model, batch, prompt):
            raise RuntimeError("sempre falha")

        with patch("time.sleep", return_value=None):
            out = run_score(6, ["k1", "k2"], fake)
        self.assertTrue(all(c.failed for c in out))


class TestRateLimitHelpers(unittest.TestCase):
    def test_non_429_returns_none(self):
        self.assertIsNone(sc._rate_limit_wait(RuntimeError("503"), 10))

    def test_429_by_status_code(self):
        e = RuntimeError("x")
        e.status_code = 429
        self.assertEqual(sc._rate_limit_wait(e, 10), 10)

    def test_retry_after_capped(self):
        class R:
            headers = {"retry-after": "9999"}

        e = RuntimeError("429")
        e.response = R()
        self.assertEqual(sc._rate_limit_wait(e, 10), sc.RATE_LIMIT_WAIT_CAP_SEC)

    def test_gate_paces_same_key(self):
        g = sc._KeyGate(min_interval=0.2)
        t0 = time.monotonic()
        g.wait()
        g.wait()
        self.assertGreaterEqual(time.monotonic() - t0, 0.2)


if __name__ == "__main__":
    unittest.main()
