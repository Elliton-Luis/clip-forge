"""test_finishreview.py — revisão pré-burn-in do FINISH.

Cobre a lista do spec: aceitar, regenerar título/legenda, pular título/
legenda, editar, cancelar, render-só-depois-do-aceite, falha de render
preserva artefatos, reabrir continua.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from core.models import Word, Segment
from core import artifacts as art
from core import finish as fin
from core import finishreview as fr
from core.cache import fingerprint


def _w(text, start, end):
    return Word(text, start, end)


def _segs():
    return [Segment(text="olá mundo", start=0.0, end=2.0,
                    words=[_w("olá", 0.0, 0.9), _w("mundo.", 0.9, 2.0)])]


def _clip(tmp, name="clip.mp4"):
    p = Path(tmp) / name
    p.write_bytes(b"fake-video-bytes")
    return str(p)


def _seed(store, clip, model="medium"):
    fp = fingerprint(clip, model)
    art.save_artifact(store, "transcript", fp, clip,
                      {"whisper_model": model},
                      {"segments": art.serialize_segments(_segs()),
                       "transcript_hash": art.transcript_hash(_segs()),
                       "meta": {}})
    return fp


def _answers(*ans):
    it = iter(ans)
    def _fn(_msg=""):
        return next(it)
    return _fn


def _no_whisper(test):
    def _boom(*a, **k):
        test.fail("Whisper executado no FINISH com artefato válido")
    return mock.patch("core.finish.transcribe", side_effect=_boom)


class TestReviewLoop(unittest.TestCase):
    def _run(self, tmp, *ans, **kw):
        clip = _clip(tmp)
        store = Path(tmp) / "store"
        fp = _seed(store, clip)
        kw.setdefault("regen_title_fn", None)
        kw.setdefault("regen_captions_fn", None)
        kw.setdefault("edit_fn", None)
        return fr.run_review(clip, _segs(), "Oi", "1\n00:00:00,000 --> 00:00:02,000\nOI",
                             2.0, store, fp, input_fn=_answers(*ans), **kw), store, fp

    def test_aceitar_sem_alteracao(self):
        with TemporaryDirectory() as tmp:
            out, store, fp = self._run(tmp, "a")
            self.assertEqual(out["action"], "approved")
            data = art.load_artifact(store, "review", fp)
            self.assertEqual(data["status"], "approved")

    def test_regenerar_titulo(self):
        with TemporaryDirectory() as tmp:
            calls = []
            out, _, _ = self._run(tmp, "r", "t", "a",
                                  regen_title_fn=lambda: calls.append(1) or "Novo")
            self.assertEqual(calls, [1])
            self.assertEqual(out["action"], "approved")

    def test_regenerar_legenda(self):
        with TemporaryDirectory() as tmp:
            calls = []
            out, _, _ = self._run(tmp, "r", "l", "a",
                                  regen_captions_fn=lambda: calls.append(1) or "SRT")
            self.assertEqual(calls, [1])

    def test_pular_titulo(self):
        with TemporaryDirectory() as tmp:
            out, store, fp = self._run(tmp, "t", "a")
            self.assertFalse(out["state"]["title_enabled"])
            self.assertTrue(out["state"]["captions_enabled"])

    def test_pular_legenda(self):
        with TemporaryDirectory() as tmp:
            out, _, _ = self._run(tmp, "l", "a")
            self.assertFalse(out["state"]["captions_enabled"])

    def test_pular_clip(self):
        with TemporaryDirectory() as tmp:
            out, store, fp = self._run(tmp, "s")
            self.assertEqual(out["action"], "skipped")
            self.assertEqual(art.load_artifact(store, "review", fp)["status"],
                             "skipped")

    def test_cancelar_persiste(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed(store, clip)
            with self.assertRaises(SystemExit):
                fr.run_review(clip, _segs(), "Oi", None, 2.0, store, fp,
                              input_fn=_answers("q"))
            data = art.load_artifact(store, "review", fp)
            self.assertEqual(data["status"], "pending")
            self.assertIn("quit", [h["action"] for h in data["history"]])

    def test_editar_marca_estado(self):
        with TemporaryDirectory() as tmp:
            out, store, fp = self._run(tmp, "e", "a", edit_fn=lambda: True)
            self.assertEqual(out["action"], "edited")
            self.assertTrue(art.load_artifact(store, "review", fp)["edited"])

    def test_reabrir_continua(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed(store, clip)
            kw = dict(regen_title_fn=None, regen_captions_fn=None, edit_fn=None)
            with self.assertRaises(SystemExit):
                fr.run_review(clip, _segs(), "Oi", None, 2.0, store, fp,
                              input_fn=_answers("t", "q"), **kw)
            out2 = fr.run_review(clip, _segs(), "Oi", None, 2.0, store, fp,
                                 input_fn=_answers("a"), **kw)
            self.assertEqual(out2["action"], "approved")
            # toggle da sessão anterior persistiu:
            self.assertFalse(out2["state"]["title_enabled"])

    def test_nao_interativo(self):
        import sys
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed(store, clip)
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                with self.assertRaises(SystemExit):
                    fr.run_review(clip, _segs(), "Oi", None, 2.0, store, fp)


class TestRunFinishReview(unittest.TestCase):
    def _patched(self, tmp):
        sess = Path(tmp) / "sess"
        sess.mkdir()
        (sess / "preview_finish.mp4").write_bytes(b"prev")
        return (mock.patch("core.clipreview.session_dir", return_value=sess),
                mock.patch("core.clipreview.build_previews",
                           return_value=[sess / "prev.mp4"]),
                sess)

    def test_render_so_depois_do_aceite(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed(store, clip)
            p1, p2, sess = self._patched(tmp)

            def _fake_cut(video, c, out_path, **kw):
                Path(out_path).write_bytes(b"render")
            with p1, p2, _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip", side_effect=_fake_cut) as cut:
                res = fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                                     only="all", store=str(store), model="m",
                                     review=True, review_input_fn=_answers("a"))
            self.assertEqual(cut.call_count, 1)
            self.assertEqual(res["review"]["action"], "approved")
            self.assertTrue(Path(res["out"]).exists())

    def test_quit_nao_renderiza(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed(store, clip)
            p1, p2, sess = self._patched(tmp)
            with p1, p2, _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip") as cut:
                with self.assertRaises(SystemExit):
                    fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                                   only="all", store=str(store), model="m",
                                   review=True, review_input_fn=_answers("q"))
            self.assertEqual(cut.call_count, 0)

    def test_pular_titulo_no_review_render_sem_hook(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed(store, clip)
            p1, p2, sess = self._patched(tmp)
            seen = {}

            def _fake_cut(video, c, out_path, **kw):
                seen["cut_title_kwarg"] = kw.get("title")
                seen["candidate_title"] = c.title
                Path(out_path).write_bytes(b"render")
            with p1, p2, _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip", side_effect=_fake_cut):
                fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                               only="all", store=str(store), model="m",
                               review=True, review_input_fn=_answers("t", "a"))
            self.assertFalse(seen["cut_title_kwarg"])
            # sem hook: o candidato vai ao cut sem título (mas o texto segue no artefato)
            self.assertEqual(seen["candidate_title"], "")
            # title.json segue existindo (skip não apaga artefato):
            self.assertTrue((Path(store) / "title.json").exists())

    def test_editar_regenera_legenda_mantem_transcript(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed(store, clip)
            t_before = (Path(store) / "transcript.json").read_bytes()
            p1, p2, sess = self._patched(tmp)

            def _fake_cut(video, c, out_path, **kw):
               (Path(out_path)).write_bytes(b"render")
            seq = iter(["e", "", "a"])

            def _fn(_msg=""):
                a = next(seq)
                if a == "":
                    p = sess / "finish_words.txt"
                    p.write_text(p.read_text(encoding="utf-8").replace(
                        "mundo.", "MUNDO!"), encoding="utf-8")
                return a
            with p1, p2, _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip", side_effect=_fake_cut):
                res = fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                                     only="all", store=str(store), model="m",
                                     review=True, review_input_fn=_fn)
            self.assertTrue(res["review"]["edited"])
            self.assertEqual((Path(store) / "transcript.json").read_bytes(),
                             t_before)
            caps = art.load_artifact(store, "captions",
                                     fingerprint(clip, "medium"),
                                     {"caption_mode": "phrases", "vertical": True})
            self.assertIn("MUNDO", caps["srt"])

    def test_falha_de_render_preserva_artefatos(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed(store, clip)
            with _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip",
                               side_effect=RuntimeError("ffmpeg quebrou")):
                with self.assertRaises(RuntimeError):
                    fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                                   only="all", store=str(store), model="m")
            for f in ("transcript.json", "title.json", "captions.json"):
                self.assertTrue((Path(store) / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
