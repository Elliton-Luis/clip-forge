"""tests/test_tui.py — TUI monta a mesma configuração que a CLI.

Roda sem terminal: python -m unittest discover -s tests
"""
import shlex
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import clipper
from clipper import (
    CONFIG_FIELDS, build_parser, parse_cli, config_from_args,
    default_config, validate_config, cli_command,
)
from core import tui as t
from core.tui import (
    load_models, default_model_index, list_videos, latest_report,
    latest_transcript, validate_for_run, summary_lines, TUI,
)


def full_cfg(**over):
    cfg = {
        "video": "videos/VideoMedio.mp4", "out": "cortes2", "top": 5,
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "no_vertical": True, "no_captions": False, "no_title": False,
        "cache_dir": ".cache/x", "force_retranscribe": True,
        "force_rescore": False, "pad": 1.2, "min_duration": 20, "max_duration": 90,
        "no_audio_features": True,
        "min_score": 5.0, "max_per_10min": 1, "context": "ranked",
        "examples": "examples.json", "transcribe_backend": "vulkan",
        "debug_captions": True, "review_transcript": False,
        "review_titles": False, "work_dir": None, "custom_words": None,
        "acoustic_captions": False, "whisper_model": "medium", "laughs": None,
        "caption_mode": "words", "align": "off", "selection_mode": "classic",
    }
    cfg.update(over)
    return cfg


class TestConfigBuilding(unittest.TestCase):
    def test_all_fields_present(self):
        cfg = default_config()
        for f in CONFIG_FIELDS:
            self.assertIn(f, cfg)

    def test_cli_to_canonical(self):
        argv = ["videos/VideoMedio.mp4", "--out", "cortes2", "--top", "5",
                "--model", "nvidia/nemotron-3-super-120b-a12b",
                "--no-vertical", "--cache-dir", ".cache/x",
                "--force-retranscribe", "--pad", "1.2", "--no-audio-features",
                "--min-score", "5.0", "--max-per-10min", "1",
                "--context", "ranked", "--examples", "examples.json",
                "--transcribe-backend", "vulkan", "--debug-captions"]
        cfg = config_from_args(parse_cli(argv))
        self.assertEqual(cfg, full_cfg())

    def test_tui_builds_same_shape(self):
        # Simula a TUI preenchendo o dict canônico campo a campo.
        base = default_config()
        base.update(full_cfg())
        self.assertEqual(base, full_cfg())

    def test_cli_command_round_trip(self):
        cfg = full_cfg()
        cmd = cli_command(cfg)
        back = config_from_args(parse_cli(shlex.split(cmd)[2:]))
        self.assertEqual(back, cfg)

    def test_cli_command_defaults_omitted(self):
        cfg = default_config()
        cfg["video"] = "v.mp4"
        cmd = cli_command(cfg)
        self.assertIn("v.mp4", cmd)
        self.assertNotIn("--top", cmd)
        self.assertNotIn("--no-vertical", cmd)


class TestBooleans(unittest.TestCase):
    def _tui(self, **over):
        cfg = default_config()
        cfg.update(full_cfg())
        cfg.update(over)
        models = load_models()
        return TUI(None, cfg, models)

    def test_positive_to_negative_mapping(self):
        ui = self._tui(no_vertical=False, no_captions=False,
                       no_audio_features=False)
        # Tela mostra positivo...
        self.assertTrue(ui._get("vertical"))
        self.assertTrue(ui._get("captions"))
        self.assertTrue(ui._get("audio_features"))
        # ...desmarcar INVERTE a flag negativa (sem bug de inversão)...
        ui._toggle("vertical")
        ui._toggle("captions")
        ui._toggle("audio_features")
        self.assertFalse(ui._get("vertical"))
        self.assertTrue(ui.cfg["no_vertical"])
        self.assertTrue(ui.cfg["no_captions"])
        self.assertTrue(ui.cfg["no_audio_features"])

    def test_no_double_negation(self):
        ui = self._tui()  # full_cfg tem no_vertical=True
        self.assertFalse(ui._get("vertical"))
        self.assertTrue(ui.cfg["no_vertical"])

    def test_cache_toggle(self):
        ui = self._tui(cache_dir=None)
        ui.use_cache = False
        ui._toggle("use_cache")
        self.assertTrue(ui.use_cache)
        self.assertEqual(ui.cfg["cache_dir"], ".cache/clipper")
        ui._sync_from_widgets()
        self.assertEqual(ui.cfg["cache_dir"], ".cache/clipper")

    def test_caption_mode_cycles_and_syncs(self):
        ui = self._tui()
        self.assertEqual(ui._get("caption_mode"), "words")
        ui.caption_idx = 1
        self.assertEqual(ui._get("caption_mode"), "intervals")
        ui._sync_from_widgets()
        self.assertEqual(ui.cfg["caption_mode"], "intervals")

    def test_acoustic_toggle(self):
        ui = self._tui()
        self.assertFalse(ui._get("acoustic_captions"))
        ui._toggle("acoustic_captions")
        self.assertTrue(ui._get("acoustic_captions"))
        self.assertTrue(ui.cfg["acoustic_captions"])


class TestModels(unittest.TestCase):
    def test_load_has_default(self):
        models = load_models()
        ids = [m["id"] for m in models]
        self.assertGreaterEqual(len(models), 5)
        for m in models:
            self.assertTrue(m["id"] and m["name"])
        # Sem VLM/áudio/TTS/embeddings na curadoria
        bad = [i for i in ids if any(k in i for k in
               ("vision", "paligemma", "cosmos", "whisper", "tts",
                "embed", "guard", "nemotron-parse", "omni"))]
        self.assertEqual(bad, [])

    def test_default_model_selected(self):
        models = load_models()
        idx = default_model_index(models)
        self.assertEqual(models[idx]["id"], "nvidia/nemotron-3-super-120b-a12b")

    def test_model_sync(self):
        models = load_models()
        ui = TUI(None, default_config(), models)
        ui.model_idx = 1
        ui._sync_from_widgets()
        self.assertEqual(ui.cfg["model"], models[1]["id"])


class TestValidation(unittest.TestCase):
    def test_valid_passes(self):
        self.assertEqual(validate_for_run(full_cfg()), [])

    def test_each_bad_value_flagged(self):
        cases = [
            ({"video": ""}, "vídeo"),
            ({"video": "/nao/existe.mp4"}, "não encontrado"),
            ({"top": 0}, "--top"),
            ({"top": "x"}, "--top"),
            ({"pad": 9.9}, "--pad"),
            ({"pad": "x"}, "--pad"),
            ({"min_score": -1}, "--min-score"),
            ({"max_per_10min": 99}, "--max-per-10min"),
            ({"model": ""}, "Modelo"),
            ({"transcribe_backend": "cuda"}, "Backend"),
            ({"caption_mode": "x"}, "--caption-mode"),
            ({"laughs": "/nao/existe.txt"}, "risadas"),
        ]
        for over, needle in cases:
            errs = validate_for_run(full_cfg(**over))
            self.assertTrue(any(needle in e for e in errs),
                            f"{over} deveria falhar com {needle!r}: {errs}")

    def test_laughs_real_file_passes(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            self.assertEqual(validate_for_run(full_cfg(laughs=f.name)), [])

    def test_cli_round_trip_caption_flags(self):
        import shlex
        from clipper import cli_command, parse_cli, config_from_args
        cfg = full_cfg(caption_mode="intervals", acoustic_captions=True)
        back = config_from_args(parse_cli(shlex.split(cli_command(cfg))[2:]))
        self.assertEqual(back, cfg)

    def test_cli_validate_messages_kept(self):
        for cfg, msg in [({"pad": 9}, "--pad deve estar entre 0 e 5"),
                         ({"min_score": 99}, "--min-score deve estar entre 0 e 10"),
                         ({"max_per_10min": 0}, "--max-per-10min deve estar entre 1 e 20")]:
            c = full_cfg(**cfg)
            with self.assertRaises(SystemExit) as cm:
                validate_config(c)
            self.assertEqual(str(cm.exception), msg)


class TestDiscovery(unittest.TestCase):
    def test_list_videos_only_media(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.mp4").write_bytes(b"x")
            Path(d, "b.txt").write_bytes(b"x")
            Path(d, "C.MKV").write_bytes(b"x")
            got = [v["path"] for v in list_videos(d)]
            self.assertEqual(len(got), 2)
            self.assertTrue(all(g.endswith((".mp4", ".MKV")) for g in got))

    def test_latest_none_when_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(latest_report(d))
            self.assertIsNone(latest_transcript(d))

    def test_list_sessions(self):
        import json
        import tempfile
        from core.tui import list_sessions
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(list_sessions(d), [])
            s = Path(d, "Live") / "transcription"
            s.mkdir(parents=True)
            (s / "transcript.json").write_text(json.dumps({
                "status": "approved",
                "segments": [{"text": "oi", "start": 0, "end": 1,
                              "words": [{"text": "oi", "start": 0, "end": 1},
                                        {"text": "x", "start": 1, "end": 2}]}]}))
            (Path(d) / "vazia").mkdir()
            got = list_sessions(d)
            self.assertEqual(len(got), 1)  # dir sem transcript.json ignorado
            self.assertEqual(got[0]["name"], "Live")
            self.assertEqual(got[0]["status"], "approved")
            self.assertEqual(got[0]["words"], 2)


if __name__ == "__main__":
    unittest.main()
