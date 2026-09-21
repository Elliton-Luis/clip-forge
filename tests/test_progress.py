"""test_progress.py — bus de estado/progresso. Sem rede, sem ffmpeg."""
import io
import unittest

from core.progress import Progress, error_box, review_banner, _fmt_hms


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class TestStages(unittest.TestCase):
    def test_adv_done_skip(self):
        bus = Progress(mode="X", out=io.StringIO())
        st = bus.stage("tr", "Transcrição", total=10, unit="chunks")
        self.assertEqual(st.done, 0)
        bus.adv("tr", 3)
        self.assertEqual(bus._find("tr").done, 3)
        bus.adv("nope")  # chave inexistente: no-op, sem exceção
        bus.done("tr", "10 segmentos")
        self.assertEqual(bus._find("tr").state, "done")
        self.assertIn("10 segmentos", bus.notes)
        bus2 = Progress(mode="X", out=io.StringIO())
        bus2.stage("s", "S", total=2)
        bus2.skip("s", "cache")
        self.assertEqual(bus2._find("s").state, "skipped")

    def test_eta_so_quando_confiavel(self):
        import time
        bus = Progress(mode="X", out=io.StringIO())
        st = bus.stage("s", "S", total=100)
        self.assertEqual(st.eta(), "")  # nada feito ainda
        st.done = 50
        self.assertEqual(st.eta(), "")  # medição recente demais
        st.t0 = time.time() - 60
        st.done = 50
        eta = st.eta()
        self.assertTrue(eta.startswith("ETA "))
        st2 = bus.stage("n", "N")  # sem total: nunca ETA
        st2.done = 5
        st2.t0 = time.time() - 60
        self.assertEqual(st2.eta(), "")

    def test_fmt_hms(self):
        self.assertEqual(_fmt_hms(65), "01:05")
        self.assertEqual(_fmt_hms(3661), "01:01:01")


class TestLineRenderer(unittest.TestCase):
    def test_linhas_de_conclusao(self):
        buf = io.StringIO()
        bus = Progress(mode="M", out=buf)
        self.assertFalse(bus.tty)
        bus.stage("a", "Áudio", total=2, unit="candidatos")
        bus.adv("a", 2)
        bus.done("a", "pronto")
        out = buf.getvalue()
        self.assertIn("▸ Áudio...", out)
        self.assertIn("✓ Áudio — pronto", out)

    def test_warn_sempre_visivel_e_detail_so_verbose(self):
        buf = io.StringIO()
        bus = Progress(mode="M", out=buf)
        bus.warn("! algo relevante")
        bus.detail("detalhe interno")
        out = buf.getvalue()
        self.assertIn("! algo relevante", out)
        self.assertNotIn("detalhe interno", out)
        buf2 = io.StringIO()
        bus2 = Progress(mode="M", out=buf2, verbose=True)
        bus2.detail("detalhe interno")
        self.assertIn("detalhe interno", buf2.getvalue())

    def test_quiet_so_avisos_e_erros(self):
        buf = io.StringIO()
        bus = Progress(mode="M", out=buf, quiet=True)
        bus.note("conclusão qualquer")
        bus.warn("! atenção")
        out = buf.getvalue()
        self.assertNotIn("conclusão", out)
        self.assertIn("atenção", out)

    def test_pause_congela_e_restaura(self):
        out = FakeTTY()
        bus = Progress(mode="M", out=out)
        self.assertTrue(bus.tty)
        with bus.pause():
            self.assertFalse(bus.tty)
        self.assertTrue(bus.tty)


class TestDashboard(unittest.TestCase):
    def test_bloco_tem_modo_barra_e_aviso(self):
        out = FakeTTY()
        bus = Progress(mode="SELEÇÃO", out=out)
        bus.header(["video.mp4 · 01:00 · 8 clipes"])
        bus.stage("tr", "Transcrição", total=10, unit="chunks")
        bus.adv("tr", 7)
        bus.warn("! lote 3: 503")
        text = out.getvalue()
        self.assertIn("SELEÇÃO", text)
        self.assertIn("video.mp4", text)
        self.assertIn("█", text)
        self.assertIn("7/10", text)
        self.assertIn("503", text)

    def test_sem_total_mostra_contagem_sem_barra(self):
        out = FakeTTY()
        bus = Progress(mode="M", out=out)
        bus.stage("r", "Render")
        bus.adv("r", 2)
        block = bus._block()
        row = [x for x in block if x.startswith("▸ R")]
        self.assertTrue(row and "2" in row[0])
        self.assertNotIn("█", row[0])


class TestHelpers(unittest.TestCase):
    def test_error_box(self):
        buf = io.StringIO()
        error_box(buf, "Falha em scoring", ["RuntimeError: boom"],
                  "rode com --verbose")
        out = buf.getvalue()
        self.assertIn("Falha em scoring", out)
        self.assertIn("RuntimeError: boom", out)
        self.assertIn("--verbose", out)

    def test_review_banner(self):
        buf = io.StringIO()
        review_banner(buf, "REVISÃO DE CLIPS", 3, 10)
        self.assertIn("REVISÃO DE CLIPS · 3/10", buf.getvalue())

    def test_log_file_recebe_detalhes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            lf = str(Path(tmp) / "run.log")
            bus = Progress(mode="M", out=io.StringIO(), log_file=lf)
            bus.detail("chunk 420s: retry")
            bus.close()
            self.assertIn("chunk 420s", Path(lf).read_text())


if __name__ == "__main__":
    unittest.main()
