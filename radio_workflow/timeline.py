"""Chronological, atomic CSV export for radio analysis results."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import timedelta
from typing import Iterable, Mapping


_TIMESTAMP = re.compile(
    r"(?:(?P<days>\d+) days?, )?(?P<hours>\d+):(?P<minutes>[0-5]\d):"
    r"(?P<seconds>[0-5]\d(?:\.\d+)?)"
)
_FIELDNAMES = ("Timestamp", "Type", "Details")


def _seconds(value: object) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp in seconds: {value!r}") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Timestamp must be finite and nonnegative: {value!r}")
    return seconds


def format_timestamp(seconds: float) -> str:
    """Format an elapsed time, rounding down to the complete second."""
    return str(timedelta(seconds=math.floor(_seconds(seconds))))


def _parse_timestamp(value: object) -> float:
    match = _TIMESTAMP.fullmatch(str(value))
    if match is None:
        raise ValueError(f"Invalid elapsed timestamp: {value!r}")
    return (
        int(match["days"] or 0) * 86400
        + int(match["hours"]) * 3600
        + int(match["minutes"]) * 60
        + float(match["seconds"])
    )


def _entry_seconds(entry: Mapping, numeric_key: str, timestamp_key: str) -> float:
    if entry.get(numeric_key) is not None:
        return _seconds(entry[numeric_key])
    return _parse_timestamp(entry[timestamp_key])


def _refuse_input_overwrite(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output CSV must not overwrite the input recording.")
    # Resolve follows symbolic links; samefile also catches hard links.
    try:
        same_file = input_path.samefile(output_path)
    except FileNotFoundError:
        same_file = False
    if same_file:
        raise ValueError("The output CSV must not overwrite the input recording.")


def save_to_csv(
    input_path: str | os.PathLike,
    dj_segments: Iterable[Mapping],
    song_segments: Iterable[Mapping],
    output_path: str | os.PathLike | None = None,
) -> Path:
    """Write a timeline while preserving an existing CSV if writing fails.

    Numeric timestamps determine ordering, including their fractional seconds.
    Legacy entries containing only formatted timestamp strings remain supported.
    """
    recording = Path(input_path)
    destination = (
        Path(output_path)
        if output_path is not None
        else recording.with_name(f"{recording.stem}_analysis.csv")
    )
    _refuse_input_overwrite(recording, destination)

    rows = []
    for segment in dj_segments:
        start = _entry_seconds(segment, "start_sec", "start_str")
        end = _entry_seconds(segment, "end_sec", "end_str")
        if end < start:
            raise ValueError("A speech segment must end at or after its start.")
        rows.append((start, {
            "Timestamp": format_timestamp(start),
            "Type": "🎤 DJ Speaking",
            "Details": (
                f"Spoke until {format_timestamp(end)} | Transcript: {segment['text']}"
            ),
        }))
    for segment in song_segments:
        start = _entry_seconds(segment, "start_sec", "timestamp")
        rows.append((start, {
            "Timestamp": format_timestamp(start),
            "Type": "🎵 Song Played",
            "Details": segment["song"],
        }))
    rows.sort(key=lambda item: item[0])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False,
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(row for _, row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
