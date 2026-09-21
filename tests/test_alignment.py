"""test_alignment.py — forced alignment: contrato, fallback e integração."""
import json
import tempfile
import unittest
from pathlib import Path

from core.models import Word, Segment
from core.alignment import (
    align_segments, match_and_transfer, resolve_backend, ALIGN_BACKENDS,
)


def _w(text, start, end):
    return Word(text, start, end)


def _s(text, spans):
    return Segment(text=text, start=spans[0][1], end=spans[-1][2],
                   words=[_w(t, a, b) for t, a, b in spans])


class TestResolveBackend(unittest.TestCase):
    def test_backends_conhecidos(self):
        self.assertEqual(resolve_backend("off"), "off")
        self.assertEqual(resolve_backend("whisper-refine"), "whisper-refine")
        self.assertEqual(resolve_backend("wav2vec2"), "wav2vec2")

    def test_invalido_erro_claro(self):
        with self.assertRaises(ValueError):
            resolve_backend("dtw-magico")


class TestDataModel(unittest.TestCase):
    def test_defaults_preservam_pipeline(self):
        w = Word("olá", 1.0, 1.5)
        self.assertEqual(w.timestamp_source, "whisper")
        self.assertIsNone(w.confidence)
        s = Segment(text="olá", start=1.0, end=1.5, words=[w])
        self.assertEqual(s.timestamp_source, "whisper")

    def test_construtor_posicional_compativel(self):
        w = Word("a", 0.0, 0.1)  # código antigo continua válido
        self.assertEqual((w.text, w.start, w.end), ("a", 0.0, 0.1))


class TestOffNoop(unittest.TestCase):
    def test_off_preserva_tudo(self):
        segs = [_s("olá mundo", [("olá", 1.0, 1.4), ("mundo", 1.4, 1.9)])]
        out, stats = align_segments(segs, backend="off")
        self.assertEqual(out[0].text, "olá mundo")
        self.assertEqual([(w.start, w.end) for w in out[0].words],
                         [(1.0, 1.4), (1.4, 1.9)])
        self.assertTrue(all(w.timestamp_source == "whisper" for w in out[0].words))
        self.assertEqual(stats.n_aligned, 0)

    def test_sem_audio_fallback_controlado(self):
        segs = [_s("oi", [("oi", 2.0, 2.5)])]
        out, stats = align_segments(segs, backend="whisper-refine", audio_path=None)
        self.assertEqual(out[0].words[0].start, 2.0)
        self.assertIsNotNone(stats.error)


class TestMatchAndTransfer(unittest.TestCase):
    def test_transfere_somente_par_exato(self):
        orig = [_w("eu", 10.0, 10.3), _w("vou", 10.3, 10.6), _w("matar", 10.6, 11.0)]
        new = [("eu", 10.05, 10.32, 0.9), ("vou", 10.32, 10.61, 0.8),
               ("XXX", 10.61, 11.0, 0.1)]
        out, n_al, n_fb = match_and_transfer(orig, new, 9.0, 12.0)
        self.assertEqual([w.text for w in out], ["eu", "vou", "matar"])  # texto intacto
        self.assertEqual(out[0].timestamp_source, "forced_alignment")
        self.assertEqual(out[2].timestamp_source, "whisper")  # sem par → fallback
        self.assertEqual(out[2].start, 10.6)
        self.assertEqual((n_al, n_fb), (2, 1))

    def test_span_invalido_usa_fallback(self):
        orig = [_w("a", 5.0, 5.4)]
        new = [("a", 99.0, 99.5, 0.9)]  # fora da janela
        out, _, _ = match_and_transfer(orig, new, 4.0, 6.0)
        self.assertEqual((out[0].start, out[0].end), (5.0, 5.4))
        self.assertEqual(out[0].timestamp_source, "whisper")

    def test_monotonicidade_nunca_inverte(self):
        orig = [_w("a", 1.0, 1.5), _w("b", 1.5, 2.0)]
        new = [("a", 1.8, 1.9, 0.5), ("b", 1.1, 1.2, 0.5)]  # re-decode ruidoso
        out, _, _ = match_and_transfer(orig, new, 0.0, 3.0)
        self.assertLessEqual(out[0].start, out[1].start)
        self.assertLess(out[1].start, out[1].end)


class TestRefineIntegration(unittest.TestCase):
    def _fake_redecode(self, wav, lo):
        # finge um re-decode: desloca +0.2s com confiança
        return [("olá", lo + 0.2, lo + 0.6, 0.95), ("mundo", lo + 0.6, lo + 1.0, 0.9)]

    def test_refine_marca_origem_e_confidence(self):
        import core.alignment as _al
        seg = _s("olá mundo", [("olá", 1.0, 1.4), ("mundo", 1.4, 1.9)])
        real_extract = _al._extract_window_wav
        _al._extract_window_wav = lambda *a, **k: None  # sem ffmpeg no teste
        try:
            out, stats = align_segments([seg], audio_path="fake.mp4",
                                        backend="whisper-refine",
                                        redecode_fn=self._fake_redecode)
        finally:
            _al._extract_window_wav = real_extract
        self.assertEqual(out[0].text, "olá mundo")
        self.assertEqual(len(out[0].words), 2)
        self.assertTrue(all(w.timestamp_source == "forced_alignment" for w in out[0].words))
        self.assertAlmostEqual(out[0].words[0].confidence or 0, 0.95)
        self.assertEqual(stats.n_aligned, 2)

    def test_falha_no_redecode_nao_corrompe(self):
        import core.alignment as _al
        seg = _s("oi", [("oi", 2.0, 2.5)])

        def boom(wav, lo):
            raise RuntimeError("modelo quebrou")
        real_extract = _al._extract_window_wav
        _al._extract_window_wav = lambda *a, **k: None
        try:
            out, stats = align_segments([seg], audio_path="fake.mp4",
                                        backend="whisper-refine", redecode_fn=boom)
        finally:
            _al._extract_window_wav = real_extract
        self.assertEqual(out[0].words[0].start, 2.0)  # intacto
        self.assertEqual(out[0].words[0].timestamp_source, "whisper")
        self.assertIsNotNone(stats.error)

    def test_wav2vec2_sem_deps_fallback_explicito(self):
        seg = [_s("oi", [("oi", 2.0, 2.5)])]
        out, stats = align_segments(seg, audio_path="fake.mp4", backend="wav2vec2")
        # com ou sem torch instalado: texto intacto e origem explícita
        self.assertEqual(out[0].text, "oi")
        self.assertIn(out[0].words[0].timestamp_source,
                      ("whisper", "forced_alignment"))


class TestCacheRoundtrip(unittest.TestCase):
    def test_confidence_e_source_sobrevivem_ao_cache(self):
        from core.cache import save_transcript, load_transcript
        w = Word("oi", 1.0, 1.5, confidence=0.8, timestamp_source="forced_alignment")
        s = Segment(text="oi", start=1.0, end=1.5, words=[w],
                    timestamp_source="forced_alignment")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_transcript(d, "fp1234", [s])
            back = load_transcript(d, "fp1234")
        self.assertAlmostEqual(back[0].words[0].confidence or 0, 0.8)
        self.assertEqual(back[0].words[0].timestamp_source, "forced_alignment")
        self.assertEqual(back[0].timestamp_source, "forced_alignment")

    def test_cache_antigo_sem_campos_carrega_com_defaults(self):
        from core.cache import load_transcript
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "fpX.transcript.json").write_text(json.dumps({
                "fingerprint": "fpX",
                "segments": [{"text": "oi", "start": 1.0, "end": 1.5,
                              "words": [{"text": "oi", "start": 1.0, "end": 1.5}]}]}))
            back = load_transcript(d, "fpX")
        self.assertEqual(back[0].words[0].timestamp_source, "whisper")
        self.assertIsNone(back[0].words[0].confidence)


class TestCliIntegration(unittest.TestCase):
    def test_flag_align_existe_e_default_off(self):
        from clipper import build_parser, config_from_args
        args = build_parser().parse_args(["v.mp4"])
        self.assertEqual(args.align, "off")
        cfg = config_from_args(args)
        self.assertEqual(cfg["align"], "off")
        args2 = build_parser().parse_args(["v.mp4", "--align", "whisper-refine"])
        self.assertEqual(config_from_args(args2)["align"], "whisper-refine")


class TestCtcMachinery(unittest.TestCase):
    """Controle positivo: com emissões que casam o texto, o DP recupera os spans
    em ordem (prova que o limite observado em gameplay é modelo/dado, não código)."""

    def test_trellis_backtrack_merge_emissao_sintetica(self):
        try:
            import torch as _t
        except ImportError:
            self.skipTest("torch ausente (backend wav2vec2 opcional)")
            return
        from core.alignment import _get_trellis, _backtrack, _merge_repeats
        T, V, BLANK, A, B = 40, 5, 0, 1, 2
        e = _t.full((T, V), -5.0)
        e[:20, A] = 5.0
        e[20:, B] = 5.0
        logp = _t.log_softmax(e, dim=-1)
        trellis = _get_trellis(logp, [A, B], blank=BLANK)
        path = _backtrack(trellis, logp, [A, B], blank=BLANK)
        self.assertIsNotNone(path)
        spans = _merge_repeats(path, 2)
        # token A antes do token B, ambos com frames reais e score alto
        self.assertLess(spans[0][1], spans[1][1])
        self.assertGreater(spans[0][3], 0.5)
        self.assertGreater(spans[1][3], 0.5)


if __name__ == "__main__":
    unittest.main()
