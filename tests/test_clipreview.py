"""test_clipreview.py — revisão pré-burn-in: A/E/S/Q, persistência, edição."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.models import Candidate, Word
from core import clipreview as CR


def _w(text, start, end):
    return Word(text, start, end)


def _cand(title="Titulo Teste", score=8.0, n=6):
    words = [_w(f"w{i}", float(i), float(i + 1)) for i in range(n)]
    c = Candidate(start=0.0, end=float(n), text=" ".join(f"w{i}" for i in range(n)),
                  words=words, speech_rate=1.0)
    c.title, c.score, c.energy = title, score, "alta"
    return c


def _answers(*ans):
    it = iter(ans)
    return lambda _msg: next(it)


class TestTranscriptLines(unittest.TestCase):
    def test_agrupa_por_frase(self):
        c = _cand()
        c.words = [_w("olá", 0.0, 0.5), _w("mundo.", 0.5, 1.0), _w("oi", 2.0, 2.5)]
        lines = CR.transcript_lines(c)
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("[00:00.000]"))
        self.assertIn("mundo.", lines[0])


class TestDecisions(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A"), _cand("B")]
            CR.save_decisions(s, sel, ["accepted", "pending"])
            data = CR.load_decisions(s)
            self.assertEqual(data["clips"][1]["status"], "pending")
            self.assertEqual(len(data["fingerprint"]), 12)

    def test_fingerprint_muda_com_titulo(self):
        a, b = [_cand("A")], [_cand("B")]
        self.assertNotEqual(CR.fingerprint(a), CR.fingerprint(b))

    def test_mismatch_retorna_none(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A")]
            CR.save_decisions(s, sel, ["accepted"])
            self.assertIsNone(CR._match_decisions([_cand("B")], CR.load_decisions(s)))
            self.assertEqual(CR._match_decisions(sel, CR.load_decisions(s)), ["accepted"])


class TestRunFlow(unittest.TestCase):
    def test_aceitar_todos(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A"), _cand("B")]
            previews = [s / "p1.mp4", s / "p2.mp4"]
            out, statuses, stats = CR.run(sel, "v.mp4", s, previews,
                                          input_fn=_answers("a", "a"))
            self.assertEqual(len(out), 2)
            self.assertEqual(statuses, ["accepted", "accepted"])
            self.assertEqual(stats["accepted"], 2)

    def test_pular_nao_e_erro(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A"), _cand("B")]
            previews = [s / "p1.mp4", s / "p2.mp4"]
            out, statuses, stats = CR.run(sel, "v.mp4", s, previews,
                                          input_fn=_answers("s", "a"))
            self.assertEqual(len(out), 1)
            self.assertEqual(statuses, ["skipped", "accepted"])
            self.assertEqual(stats["skipped"], 1)

    def test_sair_persiste_e_rerun_continua(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A"), _cand("B")]
            previews = [s / "p1.mp4", s / "p2.mp4"]
            with self.assertRaises(SystemExit):
                CR.run(sel, "v.mp4", s, previews, input_fn=_answers("a", "q"))
            data = CR.load_decisions(s)
            self.assertEqual([d["status"] for d in data["clips"]],
                             ["accepted", "pending"])
            # rerun: só o pendente é perguntado
            out, statuses, _ = CR.run(sel, "v.mp4", s, previews,
                                      input_fn=_answers("s"))
            self.assertEqual(statuses, ["accepted", "skipped"])
            self.assertEqual(len(out), 1)

    def test_candidatos_mudaram_arquiva_e_recomeca(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            CR.save_decisions(s, [_cand("A")], ["accepted"])
            out, statuses, _ = CR.run([_cand("B")], "v.mp4", s, [s / "p.mp4"],
                                      input_fn=_answers("a"))
            self.assertTrue((s / "decisions.bak.json").exists())
            self.assertEqual(statuses, ["accepted"])

    def test_ctrlc_persiste(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            sel = [_cand("A")]

            def boom(_msg):
                raise KeyboardInterrupt
            with self.assertRaises(SystemExit):
                CR.run(sel, "v.mp4", s, [s / "p.mp4"], input_fn=boom)
            self.assertEqual(CR.load_decisions(s)["clips"][0]["status"], "pending")

    def test_nao_interativo_erro_claro(self):
        import sys
        from unittest import mock
        with TemporaryDirectory() as tmp:
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                with self.assertRaises(SystemExit):
                    CR.run([_cand()], "v.mp4", Path(tmp), [Path(tmp) / "p.mp4"])


class TestEdit(unittest.TestCase):
    def _edited_file(self, s, idx, body):
        p = s / f"clip_{idx:02d}_words.txt"
        p.write_text(body, encoding="utf-8")
        return p

    def test_apply_edita_texto_e_timestamp(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            c = _cand(n=2)
            self._edited_file(s, 1, "[s00] [00:00.500 → 00:01.000] Oi\n"
                                    "[s00] [00:01.000 → 00:02.000] mundo!\n")
            n = CR.apply_clip_words(c, s / "clip_01_words.txt")
            self.assertEqual(n, 2)
            self.assertEqual(c.text, "Oi mundo!")
            self.assertAlmostEqual(c.words[0].start, 0.5)

    def test_apply_remove_palavra(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            c = _cand(n=3)
            self._edited_file(s, 1, "[s00] [00:00.000 → 00:01.000] w0\n"
                                    "[s00] [00:02.000 → 00:03.000] w2\n")
            CR.apply_clip_words(c, s / "clip_01_words.txt")
            self.assertEqual([w.text for w in c.words], ["w0", "w2"])

    def test_apply_linha_invalida_erro_claro(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            c = _cand(n=1)
            self._edited_file(s, 1, "texto sem timestamp\n")
            with self.assertRaises(RuntimeError):
                CR.apply_clip_words(c, s / "clip_01_words.txt")
            self.assertEqual(len(c.words), 1)  # nada mudou

    def test_apply_vazio_erro(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            c = _cand(n=1)
            self._edited_file(s, 1, "# só comentário\n")
            with self.assertRaises(RuntimeError):
                CR.apply_clip_words(c, s / "clip_01_words.txt")

    def test_dump_apply_roundtrip(self):
        with TemporaryDirectory() as tmp:
            s = Path(tmp)
            c = _cand(n=3)
            p = s / "clip_01_words.txt"
            CR.dump_clip_words(c, p)
            c2 = _cand(n=1)
            CR.apply_clip_words(c2, p)
            self.assertEqual([w.text for w in c2.words], ["w0", "w1", "w2"])
            self.assertAlmostEqual(c2.words[1].start, 1.0)


if __name__ == "__main__":
    unittest.main()
