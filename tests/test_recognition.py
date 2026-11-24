import asyncio
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from radio_workflow import recognition


def track(title="Song", artist="Artist", key=None):
    result = {"title": title, "subtitle": artist}
    if key is not None:
        result["key"] = key
    return {"track": result}


class RecognitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="radio-test-")
        self.addCleanup(self.directory.cleanup)
        self.audio_path = Path(self.directory.name) / "show 'quoted' $(touch nope).mp3"
        self.audio_path.write_bytes(b"audio")
        self.duration = "65.5"
        self.commands = []
        self.sample_paths = []
        self.ffmpeg_error = None
        self.stall_binary = None
        self.stalled = asyncio.Event()
        self.processes = []
        self.recognizer = SimpleNamespace(recognize=AsyncMock(return_value={}))
        self.run_mock = Mock(side_effect=self.run_media)
        media_patch = patch.object(recognition.asyncio, "create_subprocess_exec", side_effect=self.create_process)
        self.spawn_mock = media_patch.start()
        self.addCleanup(media_patch.stop)
        factory_patch = patch.object(recognition, "_create_recognizer", return_value=self.recognizer)
        self.factory_mock = factory_patch.start()
        self.addCleanup(factory_patch.stop)
        sleep_patch = patch.object(recognition, "sleep", new_callable=AsyncMock)
        self.sleep_mock = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    async def create_process(self, *command, **kwargs):
        result = self.run_mock(list(command), **kwargs)
        process = SimpleNamespace(
            returncode=result.returncode,
            communicate=AsyncMock(return_value=(result.stdout.encode(), result.stderr.encode())),
            kill=Mock(),
        )
        if command[0] == self.stall_binary:
            process.returncode = None

            async def communicate():
                if process.returncode is None:
                    self.stalled.set()
                    await asyncio.Future()
                process.reaped_with_output_present = Path(command[-1]).exists()
                return b"", b""

            def kill():
                process.killed_with_output_present = Path(command[-1]).exists()
                process.returncode = -9

            process.communicate.side_effect = communicate
            process.kill.side_effect = kill
        self.processes.append(process)
        return process

    def run_media(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, self.duration, "")
        self.sample_paths.append(Path(command[-1]))
        if self.ffmpeg_error is not None:
            raise self.ffmpeg_error
        Path(command[-1]).write_bytes(b"short WAV sample")
        return subprocess.CompletedProcess(command, 0, "", "")

    async def identify(self, interval=30, speech=(), **kwargs):
        return await recognition.identify_songs(
            self.audio_path, "deliberately-wrong-format", interval, speech, **kwargs
        )

    def offsets(self):
        return [float(command[command.index("-ss") + 1])
                for command, _ in self.commands if command[0] == "ffmpeg"]

    def assert_samples_cleaned(self):
        for sample in self.sample_paths:
            self.assertFalse(sample.exists())
            self.assertFalse(sample.parent.exists())

    async def test_uses_bounded_safe_ffmpeg_args_and_trims_last_sample(self):
        self.recognizer.recognize.side_effect = [track("First"), {}, track("Last")]
        result = await self.identify()
        self.assertEqual(result, [
            {"timestamp": "0:00:00", "start_sec": 0.0, "song": "First - Artist"},
            {"timestamp": "0:01:00", "start_sec": 60.0, "song": "Last - Artist"},
        ])
        self.assertEqual(self.offsets(), [0.0, 30.0, 60.0])
        durations = []
        for command, kwargs in self.commands:
            self.assertIsInstance(command, list)
            self.assertEqual(kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(kwargs["stderr"], subprocess.PIPE)
            self.assertIn(str(self.audio_path), command)
            if command[0] == "ffmpeg":
                durations.append(float(command[command.index("-t") + 1]))
                self.assertEqual(command[command.index("-ac") + 1], "1")
                self.assertEqual(command[command.index("-ar") + 1], "16000")
                self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
                self.assertNotIn("deliberately-wrong-format", command)
        self.assertEqual(durations, [10.0, 10.0, 5.5])
        self.assert_samples_cleaned()

    async def test_skips_any_speech_overlap_with_fractional_half_open_boundaries(self):
        self.duration = "40"
        await self.identify(interval=10, sample_sec=10, speech=[
            {"start_sec": 5.25, "end_sec": 8.0},
            {"start_sec": 19.5, "end_sec": 20.5},
            {"start_sec": 30.0, "end_sec": 35.0},
        ])
        self.assertEqual(self.offsets(), [])
        self.factory_mock.assert_not_called()

    async def test_touching_speech_boundaries_are_not_overlaps(self):
        self.duration = "30"
        await self.identify(interval=10, sample_sec=10, speech=[
            {"start_sec": 10.0, "end_sec": 20.0},
            {"start_sec": 0.0, "end_sec": 0.0},
        ])
        self.assertEqual(self.offsets(), [0.0, 20.0])

    async def test_overlapping_unsorted_speech_intervals(self):
        self.duration = "40"
        await self.identify(interval=10, sample_sec=5, speech=[
            {"start_sec": 8.0, "end_sec": 12.0},
            {"start_sec": 0.0, "end_sec": 30.0},
        ])
        self.assertEqual(self.offsets(), [30.0])

    async def test_collapses_adjacent_tracks_by_key_and_keeps_song_returns(self):
        self.duration = "150"
        self.recognizer.recognize.side_effect = [
            track("Original", key="a"), track("Different metadata", key="a"),
            track("Other", key="b"), {}, track("Original", key="a"),
        ]
        result = await self.identify()
        self.assertEqual([row["start_sec"] for row in result], [0.0, 60.0, 120.0])
        self.assertEqual([row["song"] for row in result], [
            "Original - Artist", "Other - Artist", "Original - Artist",
        ])

    async def test_collapses_fallback_title_artist_and_formats_whole_seconds(self):
        self.duration = "12"
        self.recognizer.recognize.side_effect = [track(), track(), track("New")]
        result = await self.identify(interval=4.5, sample_sec=1)
        self.assertEqual([row["start_sec"] for row in result], [0.0, 9.0])
        self.assertEqual(result[1]["timestamp"], "0:00:09")

    async def test_retries_with_timeout_warning_and_bounded_backoff(self):
        self.duration = "5"
        self.recognizer.recognize.side_effect = [
            asyncio.TimeoutError(), ConnectionError("service unavailable"), track(),
        ]
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            result = await self.identify(retries=2, timeout_sec=0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(self.recognizer.recognize.await_count, 3)
        self.assertEqual([call.args[0] for call in self.sleep_mock.await_args_list], [1.0, 2.0])
        self.assertIn("TimeoutError", captured.output[0])
        self.assertIn("retrying", captured.output[0])
        self.assertIn("service unavailable", captured.output[1])
        self.assert_samples_cleaned()

    async def test_actual_recognition_timeout_is_enforced(self):
        self.duration = "5"

        async def never_finishes(_path):
            await asyncio.Future()

        self.recognizer.recognize.side_effect = never_finishes
        with self.assertLogs(recognition.__name__, level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "all 1 eligible samples"):
                await self.identify(timeout_sec=0.01, retries=0)
        self.assert_samples_cleaned()

    async def test_partial_failures_warn_and_continue(self):
        self.duration = "35"
        self.recognizer.recognize.side_effect = [ConnectionError("offline"), track("Recovered")]
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            result = await self.identify(retries=0)
        self.assertEqual(result[0]["start_sec"], 30.0)
        self.assertIn("skipping sample", captured.output[0])
        self.assert_samples_cleaned()

    async def test_all_service_failures_raise_and_cleanup(self):
        self.duration = "35"
        self.recognizer.recognize.side_effect = ConnectionError("offline")
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            with self.assertRaisesRegex(RuntimeError, "all 2 eligible samples"):
                await self.identify(retries=0)
        self.assertEqual(len(captured.output), 2)
        self.assert_samples_cleaned()

    async def test_no_matches_are_successful_empty_result(self):
        self.assertEqual(await self.identify(), [])
        self.assert_samples_cleaned()

    async def test_ffmpeg_timeout_warns_and_all_failure_raises(self):
        self.duration = "5"
        self.stall_binary = "ffmpeg"
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            with self.assertRaisesRegex(RuntimeError, "all 1 eligible samples"):
                await self.identify(timeout_sec=0.01)
        self.assertIn("ffmpeg timed out", captured.output[0])
        self.factory_mock.assert_not_called()
        process = self.processes[-1]
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.await_count, 2)
        self.assertTrue(process.killed_with_output_present)
        self.assertTrue(process.reaped_with_output_present)
        self.assert_samples_cleaned()

    async def test_partial_decode_failure_continues(self):
        self.duration = "35"
        original = self.run_media

        def decode_one_failure(command, **kwargs):
            if command[0] == "ffmpeg" and command[command.index("-ss") + 1] == "0.0":
                return subprocess.CompletedProcess(command, 1, "", "bad audio packet")
            return original(command, **kwargs)

        self.run_mock.side_effect = decode_one_failure
        self.recognizer.recognize.return_value = track()
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            result = await self.identify()
        self.assertEqual(result[0]["start_sec"], 30.0)
        self.assertIn("bad audio packet", captured.output[0])
        self.assert_samples_cleaned()

    async def test_missing_media_binaries_have_readable_errors(self):
        self.run_mock.side_effect = FileNotFoundError("missing")
        with self.assertRaisesRegex(RuntimeError, "ffprobe.*install FFmpeg"):
            await self.identify()
        self.run_mock.side_effect = self.run_media
        self.ffmpeg_error = FileNotFoundError("missing")
        with self.assertRaisesRegex(RuntimeError, "ffmpeg.*install FFmpeg"):
            await self.identify()
        self.assert_samples_cleaned()

    async def test_invalid_or_empty_probe_duration(self):
        for value in ["", "N/A", "nan", "inf", "0", "-5"]:
            with self.subTest(value=value):
                self.duration = value
                with self.assertRaisesRegex(RuntimeError, "finite positive audio duration"):
                    await self.identify()
        self.factory_mock.assert_not_called()

    async def test_probe_timeout_and_nonzero_exit(self):
        self.stall_binary = "ffprobe"
        with self.assertRaisesRegex(RuntimeError, "ffprobe timed out"):
            await self.identify(timeout_sec=0.01)
        process = self.processes[-1]
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.await_count, 2)
        self.stall_binary = None
        self.run_mock.side_effect = None
        self.run_mock.return_value = subprocess.CompletedProcess([], 1, "", "invalid audio")
        with self.assertRaisesRegex(RuntimeError, "ffprobe failed: invalid audio"):
            await self.identify()

    async def test_validates_settings_before_running_tools(self):
        for name in ["interval_sec", "sample_sec", "timeout_sec"]:
            for value in [0, -1, float("nan"), float("inf"), "bad", True]:
                with self.subTest(name=name, value=value):
                    settings = {"sample_sec": 10, "timeout_sec": 30, name: value}
                    interval = settings.pop("interval_sec", 30)
                    with self.assertRaisesRegex(ValueError, name):
                        await self.identify(interval=interval, **settings)
        for retries in [-1, 0.5, True, "2"]:
            with self.subTest(retries=retries):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    await self.identify(retries=retries)
        self.run_mock.assert_not_called()

    async def test_validates_speech_and_input_file(self):
        for segment in [{}, {"start_sec": 1, "end_sec": 0},
                        {"start_sec": -1, "end_sec": 2},
                        {"start_sec": 0, "end_sec": float("nan")}]:
            with self.subTest(segment=segment):
                with self.assertRaises(ValueError):
                    await self.identify(speech=[segment])
        self.audio_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Audio input is not a file"):
            await self.identify()
        self.run_mock.assert_not_called()

    async def test_cancellation_removes_sample_and_temp_directory(self):
        self.duration = "5"

        async def cancel(_path):
            raise asyncio.CancelledError()

        self.recognizer.recognize.side_effect = cancel
        with self.assertRaises(asyncio.CancelledError):
            await self.identify()
        self.assert_samples_cleaned()

    async def test_cancellation_kills_and_reaps_ffmpeg_before_cleanup(self):
        self.duration = "5"
        self.stall_binary = "ffmpeg"
        task = asyncio.create_task(self.identify())
        await asyncio.wait_for(self.stalled.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        process = self.processes[-1]
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.await_count, 2)
        self.assertTrue(process.killed_with_output_present)
        self.assertTrue(process.reaped_with_output_present)
        self.assert_samples_cleaned()

    async def test_invalid_service_response_is_warned_and_retried(self):
        self.duration = "5"
        self.recognizer.recognize.side_effect = [{"track": []}, track()]
        with self.assertLogs(recognition.__name__, level="WARNING") as captured:
            result = await self.identify(retries=1)
        self.assertEqual(len(result), 1)
        self.assertIn("invalid track metadata", captured.output[0])


if __name__ == "__main__":
    unittest.main()
