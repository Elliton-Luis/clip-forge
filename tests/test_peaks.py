"""test_peaks.py — seleção pelo auge (modo experimental). Sem rede."""
import unittest

from core.models import Candidate, Word
from core import peaks as P


def _w(text, start, end):
    return Word(text, start, end)


def _cand(text_words, start=None, energy="media", score=0.0):
    words = [_w(t, a, b) for t, a, b in text_words]
    s = start if start is not None else text_words[0][1]
    e = text_words[-1][2]
    c = Candidate(start=s, end=e, text=" ".join(t for t, _, _ in text_words),
                  words=words, speech_rate=2.0)
    c.energy = energy
    c.score = score
    return c


def _span(n, t0=0.0, step=1.0):
    return [(f"w{i}", t0 + i * step, t0 + (i + 1) * step) for i in range(n)]


class TestSplitSubwindows(unittest.TestCase):
    def test_curto_uma_janela(self):
        c = _cand(_span(5, step=2.0))  # 10 s
        subs = P.split_subwindows(c)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].start, 0.0)

    def test_longo_varias_com_passo(self):
        c = _cand(_span(60, step=1.0))  # 60 s
        subs = P.split_subwindows(c)
        self.assertGreater(len(subs), 3)
        # passo ~6 s entre inícios e cobertura total
        self.assertAlmostEqual(subs[1].start - subs[0].start, 6.0, places=0)
        self.assertLessEqual(subs[-1].end, c.end + 0.01)

    def test_snap_fronteira_palavra(self):
        c = _cand(_span(30, step=1.0))
        for s in P.split_subwindows(c):
            self.assertTrue(any(abs(w.start - s.start) < 1e-6 for w in c.words))


class TestHeuristic(unittest.TestCase):
    def test_risada_vence(self):
        c = _cand(_span(30, step=1.0))
        from core.acoustic import AcousticEvent
        laughs = [AcousticEvent(type="laugh", start=20.0, end=24.0, confidence=1.0)]
        P.detect_peak_heuristic(c, laughs=laughs)
        self.assertEqual(c.peak_source, "heuristic")
        self.assertLessEqual(c.peak_start, 24.0)
        self.assertGreaterEqual(c.peak_end, 20.0)

    def test_exclamacao_sem_risada(self):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(30)]
        words[25] = ("uau!", 25.0, 26.0)
        c = _cand(words)
        P.detect_peak_heuristic(c)
        self.assertIn("uau", c.text[c.text.index("uau") - 0:])
        # peak deve cobrir a exclamação (janela de 12 s contendo 25-26 s)
        self.assertLessEqual(c.peak_start, 26.0)
        self.assertGreaterEqual(c.peak_end, 25.0)

    def test_janela_unica(self):
        c = _cand(_span(5, step=2.0))
        P.detect_peak_heuristic(c)
        self.assertEqual((c.peak_start, c.peak_end), (c.start, c.end))


class TestBuildClip(unittest.TestCase):
    def _peak_cand(self):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(60)]
        words[30] = ("PUNCH.", 30.0, 31.0)
        c = _cand(words)
        P._set_peak(c, 28.0, 34.0, 9.0, "llm", "punchline")
        return c

    def test_clip_contem_peak_respeita_max(self):
        c = self._peak_cand()
        P.build_clip_around_peak(c, min_dur=20, max_dur=90)
        self.assertLessEqual(c.peak_start, 34.0)
        self.assertGreaterEqual(c.peak_end, 28.0)
        self.assertLessEqual(c.start, c.peak_start)
        self.assertGreaterEqual(c.end, c.peak_end)
        self.assertLessEqual(c.end - c.start, 90.0)
        self.assertGreaterEqual(c.end - c.start, 20.0)
        self.assertIsNotNone(c.window_start)

    def test_nao_estofa_ate_maximo(self):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(25)]
        c = _cand(words)
        P._set_peak(c, 10.0, 14.0, 9.0, "llm", "auge curto")
        P.build_clip_around_peak(c, min_dur=20, max_dur=90)
        # janela tem 25 s: clip deve ficar em ~25 s, nunca 90
        self.assertLess(c.end - c.start, 40.0)
        self.assertGreaterEqual(c.end - c.start, 20.0)

    def test_peak_maior_que_max_corta_contexto(self):
        c = _cand(_span(120, step=1.0))
        P._set_peak(c, 10.0, 100.0, 9.0, "llm", "auge longo")
        P.build_clip_around_peak(c, min_dur=20, max_dur=30)
        self.assertLessEqual(c.end - c.start, 30.0 + 1.0)

    def test_clamp_media_end(self):
        c = self._peak_cand()
        P.build_clip_around_peak(c, min_dur=20, max_dur=90, media_end=40.0)
        self.assertLessEqual(c.end, 40.0)


class TestPeakOverlap(unittest.TestCase):
    def _c(self, ps, pe):
        c = _cand(_span(10, step=1.0))
        P._set_peak(c, ps, pe, 8.0, "llm", "x")
        return c

    def test_mesmo_momento(self):
        self.assertGreaterEqual(P.peak_overlap_ratio(self._c(10, 20), self._c(12, 22)), 0.5)

    def test_momentos_distintos(self):
        self.assertLess(P.peak_overlap_ratio(self._c(10, 14), self._c(30, 34)), 0.5)

    def test_sem_peak_nao_suprime(self):
        a = _cand(_span(10, step=1.0))
        b = _cand(_span(10, step=1.0))
        self.assertEqual(P.peak_overlap_ratio(a, b), 0.0)


class TestSelectPeak(unittest.TestCase):
    def _c(self, s, e, peak, score):
        n = int(e - s)
        c = _cand([(f"w{i}", float(s + i), float(s + i + 1)) for i in range(n)],
                  score=score)
        P._set_peak(c, peak[0], peak[1], score, "llm", "r")
        return c

    def test_suprime_mesmo_auge(self):
        # Três janelas sobrepostas do MESMO peak: só uma sobrevive.
        a = self._c(10, 70, (30, 40), 8.0)
        b = self._c(20, 80, (30, 40), 7.5)
        c_ = self._c(30, 90, (30, 40), 7.0)
        sel = P.select_peak([a, b, c_], top_n=3, min_score=6.0)
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0].score, 8.0)

    def test_ranqueia_por_peak_score(self):
        a = self._c(0, 40, (5, 10), 6.5)
        b = self._c(100, 140, (110, 115), 9.0)
        # peak_score invertido de propósito: a tem auge melhor
        a.peak_score, b.peak_score = 9.5, 6.0
        sel = P.select_peak([a, b], top_n=2, min_score=6.0)
        self.assertEqual(sel[0].peak_start, 5)

    def test_min_score_filtra(self):
        a = self._c(0, 40, (5, 10), 4.0)
        sel = P.select_peak([a], top_n=1, min_score=6.0)
        self.assertEqual(sel, [])

    def test_sem_peak_detecta_sozinho(self):
        c = _cand(_span(30, step=1.0), score=7.0)
        sel = P.select_peak([c], top_n=1, min_score=6.0)
        self.assertEqual(len(sel), 1)
        self.assertIsNotNone(sel[0].peak_start)


class TestApplyPeakResult(unittest.TestCase):
    def test_id_invalido_cai_na_heuristica(self):
        c = _cand(_span(30, step=1.0))
        subs = P.split_subwindows(c)
        id_map = {s.idx: s for s in subs}
        P._apply_peak_result(c, subs, id_map, {"windows": [], "peak": 999})
        self.assertEqual(c.peak_source, "heuristic")

    def test_peak_valido_aplica(self):
        c = _cand(_span(30, step=1.0))
        subs = P.split_subwindows(c)
        id_map = {s.idx: s for s in subs}
        target = subs[2].idx
        P._apply_peak_result(
            c, subs, id_map,
            {"windows": [{"id": s.idx, "intensity": 3.0} for s in subs]
             + [{"id": target, "intensity": 9.0}],
             "peak": target, "peak_reason": "punch", "needs_context": False})
        self.assertEqual(c.peak_source, "llm")
        self.assertAlmostEqual(c.peak_score, 9.0)
        self.assertEqual((c.peak_start, c.peak_end), (subs[2].start, subs[2].end))


class TestParsePeakTitles(unittest.TestCase):
    def test_formato_clips(self):
        data = P._parse_peak_titles('{"clips": [{"id": 0, "title": "abc"}]}')
        self.assertEqual(data["clips"][0]["title"], "abc")

    def test_objetos_concatenados_nao_quebra(self):
        # Falha real observada: o modelo devolveu um objeto por clip.
        data = P._parse_peak_titles('{"id": 0, "title": "a"} {"id": 1, "title": "b"}')
        items = data if isinstance(data, list) else data.get("clips", [])
        self.assertEqual(len(items), 2)


if __name__ == "__main__":
    unittest.main()
