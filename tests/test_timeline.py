import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from radio_workflow.timeline import format_timestamp, save_to_csv


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recording = self.root / "show.mp3"
        self.recording.write_bytes(b"recording")

    def read_rows(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_numeric_order_for_long_recordings_and_fractional_seconds(self):
        result = save_to_csv(self.recording, [
            {"start_sec": 36000, "end_sec": 36001, "text": "ten hours"},
            {"start_sec": 7200.9, "end_sec": 7201, "text": "later"},
        ], [
            {"start_sec": 7200.1, "song": "earlier"},
            {"start_sec": 90000, "song": "next day"},
        ])
        rows = self.read_rows(result)
        self.assertEqual(
            [row["Timestamp"] for row in rows],
            ["2:00:00", "2:00:00", "10:00:00", "1 day, 1:00:00"],
        )
        self.assertEqual(rows[0]["Details"], "earlier")
        self.assertIn("later", rows[1]["Details"])

    def test_legacy_strings_sort_with_multiday_timestamps(self):
        result = save_to_csv(self.recording, [
            {"start_str": "10:00:00", "end_str": "10:00:01", "text": "hello"},
        ], [
            {"timestamp": "2 days, 1:00:00", "song": "last"},
            {"timestamp": "2:00:00", "song": "first"},
        ])
        self.assertEqual(
            [row["Timestamp"] for row in self.read_rows(result)],
            ["2:00:00", "10:00:00", "2 days, 1:00:00"],
        )

    def test_utf8_quoting_and_nested_output_path(self):
        output = self.root / "reports" / "nested" / "output.csv"
        transcript = 'Bonjour, "été"\n第二行'
        song = 'Title, "Artist"\n🎶'
        result = save_to_csv(self.recording, [
            {"start_sec": 0, "end_sec": 1, "text": transcript},
        ], [{"start_sec": 2, "song": song}], output)
        self.assertEqual(result, output)
        rows = self.read_rows(result)
        self.assertEqual(rows[0]["Type"], "🎤 DJ Speaking")
        self.assertEqual(rows[0]["Details"], f"Spoke until 0:00:01 | Transcript: {transcript}")
        self.assertEqual(rows[1]["Type"], "🎵 Song Played")
        self.assertEqual(rows[1]["Details"], song)

    def test_empty_results_still_write_compatible_header(self):
        result = save_to_csv(self.recording, [], [])
        self.assertEqual(result, self.root / "show_analysis.csv")
        self.assertEqual(result.read_text(encoding="utf-8"), "Timestamp,Type,Details\n")

    def test_failed_replace_preserves_existing_output_and_removes_temp(self):
        output = self.root / "show_analysis.csv"
        output.write_text("previous output", encoding="utf-8")
        with mock.patch("radio_workflow.timeline.os.replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                save_to_csv(self.recording, [], [])
        self.assertEqual(output.read_text(encoding="utf-8"), "previous output")
        self.assertEqual(set(self.root.iterdir()), {self.recording, output})

    def test_failed_write_preserves_existing_output_and_removes_temp(self):
        output = self.root / "show_analysis.csv"
        output.write_text("previous output", encoding="utf-8")
        with mock.patch("radio_workflow.timeline.csv.DictWriter.writerows", side_effect=OSError("full")):
            with self.assertRaisesRegex(OSError, "full"):
                save_to_csv(self.recording, [], [])
        self.assertEqual(output.read_text(encoding="utf-8"), "previous output")
        self.assertEqual(set(self.root.iterdir()), {self.recording, output})

    def test_refuses_input_and_symlink_alias(self):
        with self.assertRaisesRegex(ValueError, "overwrite the input"):
            save_to_csv(self.recording, [], [], self.recording)
        alias = self.root / "alias.csv"
        alias.symlink_to(self.recording)
        with self.assertRaisesRegex(ValueError, "overwrite the input"):
            save_to_csv(self.recording, [], [], alias)
        self.assertEqual(self.recording.read_bytes(), b"recording")

    def test_refuses_hard_link_alias(self):
        alias = self.root / "alias.csv"
        alias.hardlink_to(self.recording)
        with self.assertRaisesRegex(ValueError, "overwrite the input"):
            save_to_csv(self.recording, [], [], alias)

    def test_bad_segments_preserve_existing_output(self):
        output = self.root / "show_analysis.csv"
        output.write_text("previous output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "end at or after"):
            save_to_csv(self.recording, [{"start_sec": 2, "end_sec": 1, "text": "bad"}], [])
        self.assertEqual(output.read_text(encoding="utf-8"), "previous output")

    def test_timestamp_floors_and_rejects_invalid_times(self):
        self.assertEqual(format_timestamp(3661.99), "1:01:01")
        self.assertEqual(format_timestamp(172800), "2 days, 0:00:00")
        for value in (-1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                format_timestamp(value)


if __name__ == "__main__":
    unittest.main()
