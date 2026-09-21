"""test_phrases.py — legendas por unidades naturais de fala (10 casos do spec)."""
import unittest

from core.models import Word
from core import phrases as P
from core.video import build_srt, _group_cues


def W(text, start, end):
    return Word(text, start, end)


def words_seq(texts, start=0.0, step=0.5, dur=0.4):
    return [W(t, start + i * step, start + i * step + dur)
            for i, t in enumerate(texts)]


class TestPhrases(unittest.TestCase):
    def test_1_frase_curta_uma_cue(self):
        ws = words_seq(["Olá,", "mundo!"])
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0].text, "Olá, mundo!")

    def test_2_frase_longa_nao_parte_por_tamanho(self):
        ws = words_seq([f"palavra{i}" for i in range(30)], step=0.4)
        ph = P.group_phrases(ws, 0.0, 60.0)
        self.assertEqual(len(ph), 1)  # sem ponto final = uma unidade
        srt = build_srt(ws, 0.0, 60.0, caption_mode="phrases")
        # uma cue (sem subdivisão temporal), só quebra visual de linhas
        self.assertEqual(srt.count("-->"), 1)
        self.assertEqual(ph[0].start, ws[0].start)
        self.assertEqual(ph[0].end, ws[-1].end)

    def test_3_pausa_curta_continua(self):
        ws = [W("Vamos", 0.0, 0.4), W("lá", 0.5, 0.9)]
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual(len(ph), 1)
        self.assertFalse(ph[0].has_ellipsis)  # fluxo contínuo, sem "…"

    def test_4_pausa_longa_quebra(self):
        ws = [W("Vamos.", 0.0, 0.4), W("Chegamos.", 10.0, 10.4)]
        ph = P.group_phrases(ws, 0.0, 20.0)
        self.assertEqual(len(ph), 2)

    def test_5_pausa_interna_vira_reticencias(self):
        ws = [W("Eu", 0.0, 0.3), W("achei", 0.3, 0.6), W("que", 0.6, 0.9),
              W("você", 2.0, 2.3), W("tinha", 2.3, 2.6), W("entendido.", 2.6, 3.0)]
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual(len(ph), 1)
        self.assertTrue(ph[0].has_ellipsis)
        self.assertIn("…", ph[0].text)
        self.assertAlmostEqual(ph[0].bridged_gap, 1.1)

    def test_6_duas_pessoas_nao_mesclam(self):
        # Turnos sem overlap: sem diarização não há rótulo de speaker, mas as
        # frases completas nunca viram uma fala só.
        a = [W("Oi!", 10.0, 10.5), W("tudo", 10.6, 11.0), W("bem?", 11.0, 11.5)]
        b = [W("Tudo!", 12.0, 12.5), W("e", 12.6, 12.8), W("você?", 12.8, 13.2)]
        ph = P.group_phrases(a + b, 0.0, 20.0)
        self.assertEqual(len(ph), 2)
        self.assertEqual(ph[0].text, "Oi! tudo bem?")
        self.assertEqual(ph[1].text, "Tudo! e você?")

    def test_7_fala_sobreposta_lanes_independentes(self):
        a = [W("EU", 10.0, 10.6), W("VOU", 10.6, 11.2)]
        b = [W("NÃO", 10.3, 10.9), W("VAI", 10.9, 11.5)]
        ph = P.group_phrases(a + b, 0.0, 20.0)
        self.assertEqual(len(ph), 2)
        # nenhuma serialização artificial: spans preservados e coexistentes
        spans = sorted((p.start, p.end) for p in ph)
        self.assertLess(spans[1][0], spans[0][1])

    def test_8_pontuacao_quebra_frases(self):
        ws = words_seq(["Vamos.", "Ele", "riu."], start=0.0, step=0.6)
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual(len(ph), 2)
        self.assertEqual(ph[0].text, "Vamos.")
        self.assertEqual(ph[1].text, "Ele riu.")

    def test_9_frase_continua_apos_quebra(self):
        # "Vamos fazer isso. (3 s) Ou não." = uma fala → merge com "…"
        ws = [W("Vamos", 10.0, 10.4), W("fazer", 10.4, 10.8),
              W("isso.", 10.8, 11.2), W("Ou", 13.0, 13.3), W("não.", 13.3, 13.7)]
        ph = P.group_phrases(ws, 0.0, 20.0)
        self.assertEqual(len(ph), 1)
        self.assertIn("…", ph[0].text)
        cues = _group_cues(ws, 0.0, 20.0, mode="phrases")
        self.assertEqual(len(cues), 1)
        self.assertIn("…", cues[0][2])

    def test_10_palavra_individual_com_destaque(self):
        ws = words_seq(["É", "para", "a", "direita,"], step=0.4)
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual(len(ph), 1)
        # timestamps individuais preservados e ordenados p/ highlight futuro
        spans = ph[0].word_spans
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[0][0], "É")
        for (t0, s0, e0), (t1, s1, e1) in zip(spans, spans[1:]):
            self.assertLessEqual(s0, s1)
            self.assertLess(s0, e0)  # nada esticado/inventado
        self.assertEqual(spans[0][1], ws[0].start)
        self.assertEqual(spans[-1][2], ws[-1].end)

    def test_tolerancia_configuravel(self):
        ws = [W("Vamos.", 0.0, 0.4), W("ou", 2.0, 2.3), W("não.", 2.3, 2.7)]
        self.assertEqual(len(P.group_phrases(ws, 0.0, 10.0, gap_tol=3.0)), 1)
        self.assertEqual(len(P.group_phrases(ws, 0.0, 10.0, gap_tol=1.0)), 2)

    def test_sem_timestamp_inventado_nem_rebalanceado(self):
        ws = [W("a", 1.0, 1.5), W("b", 1.6, 2.9)]
        ph = P.group_phrases(ws, 0.0, 10.0)
        self.assertEqual((ph[0].start, ph[0].end), (1.0, 2.9))

    def test_palavra_nunca_partida(self):
        ws = [W(" dire", 0.0, 0.4), W("ita", 0.4, 0.8), W("mesmo.", 5.0, 5.4)]
        ph = P.group_phrases(ws, 0.0, 10.0, gap_tol=3.0)
        # "ita" (continuação, sem espaço à esquerda) nunca inicia frase
        for p in ph:
            self.assertTrue(p.words[0].text[:1].isspace())


if __name__ == "__main__":
    unittest.main()
