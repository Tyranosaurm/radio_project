import asyncio
import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from radio_workflow import radioprocess


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.input = Path(self.tmp.name) / "show without extension"
        self.input.write_bytes(b"audio fixture")

    def run_main(self, *args):
        return asyncio.run(radioprocess.main(["-i", str(self.input), *args]))

    def test_all_entry_points_help_without_audio_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        for command in (
            ["radio_analyzer.py"],
            ["radio_workflow/radioprocess.py"],
            ["-m", "radio_workflow.radioprocess"],
        ):
            result = subprocess.run(
                [sys.executable, *command, "--help"], cwd=root,
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--request-timeout", result.stdout)

    def test_invalid_options_fail_before_processing(self):
        parser = radioprocess.build_parser()
        for args in (
            ["--interval", "0"], ["--interval", "nan"],
            ["--sample-duration", "-1"], ["--request-timeout", "inf"],
            ["--retries", "-1"], ["--retries", "1.5"],
            ["--skip-speech", "--skip-songs"],
        ):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parser.parse_args(["-i", str(self.input), *args])
                self.assertEqual(caught.exception.code, 2)

    def test_songs_only_propagates_settings_and_writes_csv(self):
        songs = [{"start_sec": 10.75, "song": "Artist, Title"}]
        output = Path(self.tmp.name) / "results" / "timeline.csv"
        with patch.object(radioprocess, "extract_dj_segments") as speech, \
             patch.object(radioprocess.shutil, "which", return_value="/bin/tool"), \
             patch.object(radioprocess, "identify_songs", new_callable=AsyncMock, return_value=songs) as identify:
            self.assertEqual(self.run_main(
                "--skip-speech", "--interval", "15.5", "--sample-duration", "8",
                "--request-timeout", "4", "--retries", "1", "-o", str(output),
            ), 0)
            speech.assert_not_called()
            identify.assert_awaited_once_with(
                self.input.resolve(), None, 15.5, [], sample_sec=8.0,
                timeout_sec=4.0, retries=1,
            )
        with output.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["Timestamp"], "0:00:10")
        self.assertEqual(rows[0]["Details"], "Artist, Title")

    def test_speech_only_does_not_require_shazam_or_ffmpeg(self):
        with patch.object(radioprocess.shutil, "which", return_value=None), \
             patch.object(radioprocess, "extract_dj_segments", return_value=[]) as speech, \
             patch.object(radioprocess, "identify_songs", new_callable=AsyncMock) as songs:
            self.assertEqual(self.run_main(
                "--skip-songs", "--whisper-model", "tiny", "--language", "en",
                "--device", "cuda", "--compute-type", "float16",
            ), 0)
            speech.assert_called_once_with(
                self.input.resolve(), model_name="tiny", device="cuda",
                compute_type="float16", language="en",
            )
            songs.assert_not_awaited()

    def test_failure_returns_nonzero_preserving_existing_output(self):
        output = self.input.with_name(f"{self.input.stem}_analysis.csv")
        output.write_text("previous results", encoding="utf-8")
        with patch.object(radioprocess, "extract_dj_segments", side_effect=RuntimeError("model failed")), \
             self.assertLogs(radioprocess.LOGGER, level="ERROR") as logs:
            self.assertEqual(self.run_main("--skip-songs"), 1)
        self.assertIn("model failed", " ".join(logs.output))
        self.assertEqual(output.read_text(), "previous results")

    def test_missing_input_and_dependencies_and_output_alias(self):
        cases = [
            ["-i", str(self.input.parent / "missing"), "--skip-songs"],
            ["-i", str(self.input), "--skip-speech"],
            ["-i", str(self.input), "--skip-songs", "-o", str(self.input)],
        ]
        for args in cases:
            with self.subTest(args=args), \
                 patch.object(radioprocess.shutil, "which", return_value=None), \
                 patch.object(radioprocess, "extract_dj_segments") as speech, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    asyncio.run(radioprocess.main(args))
                self.assertEqual(caught.exception.code, 2)
                speech.assert_not_called()
        self.assertEqual(self.input.read_bytes(), b"audio fixture")

    def test_keyboard_interrupt_exit_status(self):
        def interrupt(coro):
            coro.close()
            raise KeyboardInterrupt
        with patch.object(radioprocess.asyncio, "run", side_effect=interrupt), \
             self.assertLogs(radioprocess.LOGGER, level="WARNING"):
            self.assertEqual(radioprocess.cli(), 130)


class SpeechTests(unittest.TestCase):
    def test_lazy_whisper_settings_and_precise_timestamps(self):
        segments = [
            SimpleNamespace(start=1.25, end=2.75, text=" Hello! "),
            SimpleNamespace(start=3.0, end=4.0, text=" "),
            SimpleNamespace(start=5.0, end=4.0, text="invalid"),
        ]
        model = Mock()
        model.transcribe.return_value = (iter(segments), SimpleNamespace())
        whisper_class = Mock(return_value=model)
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=whisper_class)}):
            results = radioprocess.extract_dj_segments(
                "audio.m4a", model_name="small", device="auto", compute_type="default", language="fr",
            )
        whisper_class.assert_called_once_with("small", device="auto", compute_type="default")
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "fr")
        self.assertEqual(results, [{
            "start_sec": 1.25, "end_sec": 2.75,
            "start_str": "0:00:01", "end_str": "0:00:02", "text": "Hello!",
        }])

    def test_missing_whisper_has_actionable_message(self):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaisesRegex(RuntimeError, "--skip-speech"):
                radioprocess.extract_dj_segments("audio.wav")


if __name__ == "__main__":
    unittest.main()
