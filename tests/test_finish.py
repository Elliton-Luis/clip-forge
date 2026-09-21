"""test_finish.py — artefatos reutilizáveis + FINISH sem Whisper.

Whisper e LLM são espiões que explodem se chamados fora de hora: cada teste
prova que a etapa sob teste NÃO recomputa o que já existe.
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


WM = "medium"


def _w(text, start, end):
    return Word(text, start, end)


def _segs():
    return [Segment(text="olá mundo", start=0.0, end=2.0,
                    words=[_w("olá", 0.0, 0.9), _w("mundo.", 0.9, 2.0)])]


def _clip(tmp, name="clip.mp4", payload=b"fake-video-bytes"):
    p = Path(tmp) / name
    p.write_bytes(payload)
    return str(p)


def _seed_transcript(store, clip, segs=None, model=WM):
    segs = segs if segs is not None else _segs()
    fp = fingerprint(clip, model)
    art.save_artifact(store, "transcript", fp, clip,
                      {"whisper_model": model},
                      {"segments": art.serialize_segments(segs),
                       "transcript_hash": art.transcript_hash(segs),
                       "meta": {}})
    return fp, segs


def _no_whisper(test):
    def _boom(*a, **k):
        test.fail("Whisper executado quando o artefato bastava")
    return mock.patch("core.finish.transcribe", side_effect=_boom)


class TestTranscriptReuse(unittest.TestCase):
    def test_transcript_reutilizado_sem_whisper(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            with _no_whisper(self):
                out, reused = fin.ensure_transcript(clip, store, whisper_model=WM)
            self.assertTrue(reused)
            self.assertEqual([w.text for w in out[0].words], ["olá", "mundo."])

    def test_source_alterado_invalida_e_retranscreve(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed_transcript(store, clip)
            Path(clip).write_bytes(b"fake-video-bytes-MODIFIED")
            with mock.patch("core.finish.transcribe",
                            return_value=_segs()) as tr:
                out, reused = fin.ensure_transcript(clip, store, whisper_model=WM)
            self.assertFalse(reused)
            self.assertEqual(tr.call_count, 1)

    def test_json_corrompido_regenera(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed_transcript(store, clip)
            (Path(store) / "transcript.json").write_text("{invalido", encoding="utf-8")
            with mock.patch("core.finish.transcribe",
                            return_value=_segs()) as tr:
                _, reused = fin.ensure_transcript(clip, store, whisper_model=WM)
            self.assertFalse(reused)
            self.assertEqual(tr.call_count, 1)

    def test_versao_obsoleta_regenera(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, _ = _seed_transcript(store, clip)
            p = Path(store) / "transcript.json"
            data = json.loads(p.read_text(encoding="utf-8"))
            data["artifact_version"] = 0
            p.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch("core.finish.transcribe",
                            return_value=_segs()) as tr:
                fin.ensure_transcript(clip, store, whisper_model=WM)
            self.assertEqual(tr.call_count, 1)


class TestTitleCaptionsRender(unittest.TestCase):
    def test_titulo_nao_dispara_transcricao(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            fake = {"title": "Olá Mundo", "hashtags": "#clip",
                    "score": 8.0, "reason": "ok"}
            with _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value=fake) as llm:
                data, reused = fin.make_title(clip, segs, store, fp, "m")
            self.assertFalse(reused)
            self.assertEqual(llm.call_count, 1)
            self.assertEqual(data["title"], "Olá Mundo")

    def test_titulo_reutilizado_sem_llm(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            with mock.patch("core.finish._title_llm",
                            return_value={"title": "X", "hashtags": "",
                                          "score": 1.0, "reason": ""}):
                fin.make_title(clip, segs, store, fp, "m")
            with _no_whisper(self), \
                    mock.patch("core.finish._title_llm") as llm:
                llm.side_effect = AssertionError("LLM à toa")
                data, reused = fin.make_title(clip, segs, store, fp, "m")
            self.assertTrue(reused)

    def test_troca_de_modelo_invalida_so_titulo(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            with mock.patch("core.finish._title_llm",
                            return_value={"title": "X", "hashtags": "",
                                          "score": 1.0, "reason": ""}):
                fin.make_title(clip, segs, store, fp, "model-a")
            with _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value={"title": "Y", "hashtags": "",
                                             "score": 1.0, "reason": ""}) as llm:
                data, reused = fin.make_title(clip, segs, store, fp, "model-b")
            self.assertFalse(reused)  # regenerou o título...
            self.assertEqual(data["title"], "Y")
            self.assertEqual(llm.call_count, 1)
            # ...mas o transcript segue intacto e reutilizável:
            with _no_whisper(self):
                _, reused_t = fin.ensure_transcript(clip, store, whisper_model=WM)
            self.assertTrue(reused_t)

    def test_legenda_nao_dispara_transcricao(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            before = (Path(store) / "transcript.json").read_bytes()
            with _no_whisper(self):
                data, reused = fin.make_captions(segs, store, fp, clip)
            self.assertFalse(reused)
            self.assertIn("OLÁ MUNDO", data["srt"])
            self.assertEqual((Path(store) / "transcript.json").read_bytes(), before)

    def test_legenda_reutilizada(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            with _no_whisper(self):
                fin.make_captions(segs, store, fp, clip)
                data, reused = fin.make_captions(segs, store, fp, clip)
            self.assertTrue(reused)
            self.assertIn("OLÁ", data["srt"])

    def test_render_nao_dispara_transcricao(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            title = {"title": "Oi", "hashtags": "", "score": 7.0, "reason": ""}
            out = Path(tmp) / "out" / "final.mp4"

            def _fake_cut(video, c, out_path, **kw):
                Path(out_path).write_bytes(b"render")
            with _no_whisper(self), \
                    mock.patch("core.finish.cut_clip", side_effect=_fake_cut) as cut:
                got = fin.render_clip(clip, segs, title, out)
            self.assertEqual(cut.call_count, 1)
            self.assertTrue(got.exists())

    def test_render_recusa_saida_sobre_entrada(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            with self.assertRaises(RuntimeError):
                fin.render_clip(clip, _segs(), {"title": "T"}, clip)


class TestManualTitle(unittest.TestCase):
    def test_titulo_manual_nao_toca_transcript(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            before = (Path(store) / "transcript.json").read_bytes()
            with _no_whisper(self):
                data = fin.set_manual_title(clip, segs, store, fp, "Meu Título")
            self.assertEqual(data["title"], "Meu Título")
            self.assertEqual((Path(store) / "transcript.json").read_bytes(), before)


class TestDependencyGraph(unittest.TestCase):
    """Grafo FINISH: title→transcript, captions→transcript,
    render→transcript(+title/captions). title ↛ captions, captions ↛ title,
    e NADA chama descoberta (candidatos/scoring/NMS/seleção)."""

    def _boom(self, name):
        def _f(*a, **k):
            self.fail(f"descoberta/dependência indevida: {name}")
        return _f

    def _discovery_off(self):
        return (
            mock.patch("clipper.build_candidates",
                       side_effect=self._boom("candidatos")),
            mock.patch("clipper.score_candidates",
                       side_effect=self._boom("scoring")),
            mock.patch("clipper.select_top",
                       side_effect=self._boom("seleção")),
            mock.patch("clipper.annotate_audio",
                       side_effect=self._boom("energia p/ seleção")),
            mock.patch("core.scoring.score",
                       side_effect=self._boom("scoring.score")),
        )

    def _run_all(self, exit_stack, fn, *a, **k):
        for p in self._discovery_off():
            exit_stack.enter_context(p)
        return fn(*a, **k)

    def test_title_sem_captions_sem_descoberta(self):
        import contextlib
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed_transcript(store, clip)
            fake = {"title": "T", "hashtags": "", "score": None,
                    "reason": "finish-llm"}
            with contextlib.ExitStack() as st, \
                    _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               return_value=fake), \
                    mock.patch("core.finish.make_captions",
                               side_effect=self._boom("captions")):
                res = self._run_all(st, fin.run_finish, clip,
                                    out_dir=str(Path(tmp) / "o"),
                                    only="title",
                                    store=str(store), model="m")
            self.assertEqual(res["title"], "T")

    def test_captions_sem_title_sem_descoberta(self):
        import contextlib
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed_transcript(store, clip)
            with contextlib.ExitStack() as st, \
                    _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               side_effect=self._boom("title")), \
                    mock.patch("core.finish.make_title",
                               side_effect=self._boom("make_title")):
                res = self._run_all(st, fin.run_finish, clip,
                                    out_dir=str(Path(tmp) / "o"),
                                    only="captions",
                                    store=str(store), model="m")
            self.assertTrue(res["srt"].endswith(".srt"))

    def test_render_so_com_artefatos_sem_llm_sem_descoberta(self):
        import contextlib
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            with mock.patch("core.finish._title_llm",
                            return_value={"title": "T", "hashtags": "",
                                          "score": None, "reason": "x"}):
                fin.make_title(clip, segs, store, fp, "m")
            fin.make_captions(segs, store, fp, clip)

            def _fake_cut(video, c, out_path, **kw):
                Path(out_path).write_bytes(b"render")
            with contextlib.ExitStack() as st, \
                    _no_whisper(self), \
                    mock.patch("core.finish._title_llm",
                               side_effect=self._boom("LLM")), \
                    mock.patch("core.finish.cut_clip",
                               side_effect=_fake_cut):
                res = self._run_all(st, fin.run_finish, clip,
                                    out_dir=str(Path(tmp) / "o"),
                                    only="render",
                                    store=str(store), model="m")
            self.assertTrue(res["out"].endswith("_final.mp4"))

    def test_render_sem_titulo_erro_claro(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            _seed_transcript(store, clip)
            with _no_whisper(self), \
                    self.assertRaises(RuntimeError):
                fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                               only="render", store=str(store), model="m")

    def test_regenerate_title_nao_toca_captions(self):
        with TemporaryDirectory() as tmp:
            clip = _clip(tmp)
            store = Path(tmp) / "store"
            fp, segs = _seed_transcript(store, clip)
            fin.make_captions(segs, store, fp, clip)
            caps_before = (Path(store) / "captions.json").read_bytes()
            with mock.patch("core.finish._title_llm",
                            return_value={"title": "Novo", "hashtags": "",
                                          "score": None, "reason": "x"}):
                with _no_whisper(self):
                    fin.run_finish(clip, out_dir=str(Path(tmp) / "o"),
                                   only="title", store=str(store), model="m",
                                   regen_title=True)
            self.assertEqual((Path(store) / "captions.json").read_bytes(),
                             caps_before)


if __name__ == "__main__":
    unittest.main()
