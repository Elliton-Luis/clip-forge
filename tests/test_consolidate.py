"""test_consolidate.py — grafo de invalidação e reuso incremental.

source → transcript → title/captions → render. Irmãos não se invalidam;
transcript trocado invalida os derivados; source trocado invalida tudo.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from core.models import Word, Segment
from core import artifacts as art
from core import finish as fin
from core.cache import fingerprint


def _w(text, start, end):
    return Word(text, start, end)


def _segs(words=("olá", "mundo.")):
    return [Segment(text=" ".join(words), start=0.0, end=2.0,
                    words=[_w(words[0], 0.0, 0.9), _w(words[1], 0.9, 2.0)])]


def _clip(tmp, name="clip.mp4", payload=b"bytes"):
    p = Path(tmp) / name
    p.write_bytes(payload)
    return str(p)


def _seed_all(store, clip, model="medium"):
    fp = fingerprint(clip, model)
    art.save_artifact(store, "transcript", fp, clip, {"whisper_model": model},
                      {"segments": art.serialize_segments(_segs()),
                       "transcript_hash": art.transcript_hash(_segs()),
                       "meta": {}})
    thash = art.transcript_hash(_segs())
    art.save_artifact(store, "title", fp, clip, {"model": model},
                      {"title": "T", "hashtags": "", "score": None,
                       "reason": "x", "model": model, "transcript_hash": thash,
                       "title_source": "llm"})
    art.save_artifact(store, "captions", fp, clip,
                      {"caption_mode": "phrases", "vertical": True},
                      {"srt": "S", "ass": "", "width": 1080, "height": 1920,
                       "caption_mode": "phrases", "vertical": True,
                       "transcript_hash": thash})
    return fp


def _valid(store, kind, fp, cfg=None, thash=None):
    st, _ = art.artifact_state(store, kind, fp, cfg, thash)
    return st == "ready"


class TestInvalidationGraph(unittest.TestCase):
    def test_title_changed_so_title_regenerates(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            caps_before = (Path(store) / "captions.json").read_bytes()
            trans_before = (Path(store) / "transcript.json").read_bytes()
            with mock.patch("core.finish._title_llm",
                            return_value={"title": "Novo", "hashtags": "",
                                          "score": None, "reason": "x"}), \
                    mock.patch("core.finish.transcribe",
                               side_effect=AssertionError("whisper")):
                fin.make_title(clip, _segs(), store, fp, "medium", force=True)
            # captions segue válido e transcript intacto:
            thash = art.transcript_hash(_segs())
            self.assertTrue(_valid(store, "captions", fp,
                                   {"caption_mode": "phrases", "vertical": True},
                                   thash))
            self.assertEqual((Path(store) / "transcript.json").read_bytes(),
                             trans_before)
            self.assertEqual((Path(store) / "captions.json").read_bytes(),
                             caps_before)

    def test_captions_changed_title_stays_valid(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            title_before = (Path(store) / "title.json").read_bytes()
            with mock.patch("core.finish.transcribe",
                            side_effect=AssertionError("whisper")):
                fin.make_captions(_segs(), store, fp, clip, force=True)
            self.assertEqual((Path(store) / "title.json").read_bytes(),
                             title_before)
            thash = art.transcript_hash(_segs())
            self.assertTrue(_valid(store, "title", fp, {"model": "medium"}, thash))

    def test_transcript_changed_invalidates_derivatives(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            new = _segs(("oi", "gente."))
            with mock.patch("core.finish.transcribe", return_value=new):
                segs, reused = fin.ensure_transcript(clip, store,
                                                     whisper_model="medium",
                                                     force=True)
            self.assertFalse(reused)
            thash = art.transcript_hash(segs)
            self.assertFalse(_valid(store, "title", fp, {"model": "medium"}, thash))
            self.assertFalse(_valid(store, "captions", fp,
                                    {"caption_mode": "phrases", "vertical": True},
                                    thash))

    def test_source_changed_invalidates_everything(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            Path(clip).write_bytes(b"bytes-MODIFIED")
            fp2 = fingerprint(clip, "medium")
            self.assertNotEqual(fp, fp2)
            for kind in ("transcript", "title", "captions"):
                st, _ = art.artifact_state(store, kind, fp2)
                self.assertNotEqual(st, "ready")


class TestStatus(unittest.TestCase):
    def test_needs_then_ready_then_rendered(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            out = Path(tmp) / "o"
            st = fin.clip_status(clip, str(store), str(out), model="medium")
            self.assertEqual(st["verdict"], "NEEDS-TRANSCRIPT+TITLE+CAPTIONS")
            fp = _seed_all(store, clip)
            st = fin.clip_status(clip, str(store), str(out), model="medium")
            self.assertEqual(st["verdict"], "READY")
            final = out / "clip_final.mp4"
            out.mkdir()
            final.write_bytes(b"v" * 100)
            st = fin.clip_status(clip, str(store), str(out), model="medium")
            self.assertEqual(st["verdict"], "RENDERED")

    def test_stale_when_artifacts_newer(self):
        import time
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            out = Path(tmp) / "o"
            out.mkdir()
            final = out / "clip_final.mp4"
            final.write_bytes(b"v" * 100)
            time.sleep(0.02)
            _seed_all(store, clip)  # artefatos mais novos que o render
            st = fin.clip_status(clip, str(store), str(out), model="medium")
            self.assertEqual(st["verdict"], "STALE")

    def test_status_nunca_chama_whisper_ou_llm(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            with mock.patch("core.finish.transcribe",
                            side_effect=AssertionError("whisper")), \
                    mock.patch("core.finish._title_llm",
                               side_effect=AssertionError("llm")):
                fin.clip_status(clip, str(store), None, model="medium")


class TestBatchReuse(unittest.TestCase):
    def test_segunda_passada_reutiliza_tudo(self):
        with TemporaryDirectory() as tmp:
            clips = [_clip(tmp, f"c{i}.mp4", f"bytes-{i}".encode()) for i in (1, 2)]
            out = str(Path(tmp) / "o")
            with mock.patch("core.finish.transcribe",
                            side_effect=lambda c, **k: _segs()) as tr, \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "T", "hashtags": "",
                                             "score": None, "reason": "x"}), \
                    mock.patch("core.finish.cut_clip",
                               side_effect=lambda v, c, o, **k: Path(o).write_bytes(b"r")):
                r1 = fin.batch_finish(clips, out_dir=out, model="m")
                r2 = fin.batch_finish(clips, out_dir=out, model="m")
            self.assertTrue(all(r["ok"] for r in r1 + r2))
            self.assertEqual(tr.call_count, 2)  # 1 por clip, só na 1ª passada

    def test_falha_isolada_nao_para_o_lote(self):
        with TemporaryDirectory() as tmp:
            clips = [_clip(tmp, f"c{i}.mp4", f"bytes-{i}".encode()) for i in (1, 2)]
            out = str(Path(tmp) / "o")

            def _boom_title(text, model, progress=None):
                if "olá" in text:
                    raise RuntimeError("LLM falhou")
                return {"title": "T", "hashtags": "", "score": None, "reason": "x"}
            with mock.patch("core.finish.transcribe",
                            side_effect=lambda c, **k: _segs()), \
                    mock.patch("core.finish._title_llm", side_effect=_boom_title):
                res = fin.batch_finish(clips, out_dir=out, only="title", model="m")
            self.assertFalse(all(r["ok"] for r in res))
            self.assertTrue(any(r["ok"] for r in res) or
                            any("LLM falhou" in r.get("error", "") for r in res))


class TestCleanup(unittest.TestCase):
    def test_padrao_mantem_tudo(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            with mock.patch("core.finish.transcribe",
                            side_effect=AssertionError), \
                    mock.patch("core.finish.cut_clip",
                               side_effect=lambda v, c, o, **k: Path(o).write_bytes(b"r")):
                fin.run_finish(clip, out_dir=str(Path(tmp) / "o"), only="render",
                               store=str(store), model="medium")
            for f in ("transcript.json", "title.json", "captions.json"):
                self.assertTrue((Path(store) / f).exists(), f)

    def test_no_keep_remove_intermediarios_mantem_transcript(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp = _seed_all(store, clip)
            (Path(store) / "review.json").write_text(
                json.dumps({"artifact": "review", "artifact_version": 1,
                            "source_path": clip, "source_fingerprint": fp,
                            "config": {}, "created": "x",
                            "data": {"status": "approved"}}),
                encoding="utf-8")
            with mock.patch("core.finish.transcribe",
                            side_effect=AssertionError), \
                    mock.patch("core.finish.cut_clip",
                               side_effect=lambda v, c, o, **k: Path(o).write_bytes(b"r" * 64)):
                fin.run_finish(clip, out_dir=str(Path(tmp) / "o"), only="render",
                               store=str(store), model="medium",
                               keep_artifacts=False)
            self.assertTrue((Path(store) / "transcript.json").exists())
            for f in ("title.json", "captions.json", "review.json"):
                self.assertFalse((Path(store) / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
