"""tests/test_reset.py — reset só apaga intermediários (sem rede/ffmpeg)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reset import targets, wipe, dir_size


def make_root(tmp: str) -> Path:
    root = Path(tmp)
    for d in ["videos", "models", "thirdparty", "cortes", ".cache/clipper",
              "work/v", "debug/x", "metrics", "metrics_test"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "videos" / "v.mp4").write_bytes(b"x" * 100)
    (root / ".env").write_text("K=1")
    (root / "clipper.py").write_text("x")
    (root / "metrics" / "r.json").write_text("{}")
    (root / ".cache" / "clipper" / "c.json").write_text("{}")
    return root


class TestTargets(unittest.TestCase):
    def test_only_intermediates_listed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(tmp)
            got = {p.name for p in targets(root)}
            self.assertIn(".cache", got)
            self.assertIn("work", got)
            self.assertIn("debug", got)
            self.assertIn("metrics_test", got)
            for protected in ["videos", "models", "thirdparty", "cortes",
                              ".env", "clipper.py"]:
                self.assertNotIn(protected, got)

    def test_logs_opt_in(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(tmp)
            plain = {str(p) for p in targets(root)}
            self.assertFalse(any(str(p).endswith("r.json") for p in plain))
            with_logs = targets(root, include_logs=True)
            self.assertTrue(any(p.name == "r.json" for p in with_logs))

    def test_wipe_removes_and_reports(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(tmp)
            names = [p.name for p in targets(root, include_logs=True)]
            removed, freed = wipe(targets(root, include_logs=True))
            self.assertEqual(removed, len(names))
            self.assertGreater(freed, 0)
            # protegidos intactos
            self.assertTrue((root / "videos" / "v.mp4").exists())
            self.assertTrue((root / ".env").exists())
            self.assertTrue((root / "clipper.py").exists())
            self.assertTrue((root / "cortes").is_dir())
            # intermediários foram
            self.assertFalse((root / ".cache").exists())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "metrics" / "r.json").exists())
            self.assertTrue((root / "metrics").is_dir())  # dir preservado

    def test_dir_size(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(tmp)
            self.assertGreater(dir_size(root / ".cache"), 0)
            self.assertEqual(dir_size(root / "nao-existe"), 0)


if __name__ == "__main__":
    unittest.main()
