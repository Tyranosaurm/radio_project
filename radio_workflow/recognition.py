"""Identify songs from bounded, independently decoded audio samples."""

import asyncio
from asyncio import sleep
from datetime import timedelta
import logging
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


LOGGER = logging.getLogger(__name__)


class _MissingMediaTool(RuntimeError):
    """An FFmpeg executable is unavailable."""


def _positive_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and greater than zero") from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return number


def _speech_windows(dj_segments):
    windows = []
    for segment in dj_segments:
        try:
            start = float(segment["start_sec"])
            end = float(segment["end_sec"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Speech segments require numeric start_sec and end_sec") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Speech segments require finite 0 <= start_sec <= end_sec")
        if start < end:
            windows.append((start, end))
    return sorted(windows)


async def _run_media_command(command, timeout_sec):
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise _MissingMediaTool(
            f"Required executable '{command[0]}' was not found; install FFmpeg and add it to PATH"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run {command[0]}: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
    except BaseException as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # Reap the child before the caller removes its temporary output directory.
        await process.communicate()
        if isinstance(exc, asyncio.TimeoutError):
            raise RuntimeError(f"{command[0]} timed out after {timeout_sec:g} seconds") from exc
        raise
    if process.returncode:
        detail = stderr.decode(errors="replace").strip()[-1000:]
        raise RuntimeError(f"{command[0]} failed: {detail or 'no error details'}")
    return stdout.decode(errors="replace")


async def _probe_duration(audio_path, timeout_sec):
    output = await _run_media_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        timeout_sec,
    )
    try:
        duration = float(output.strip())
    except (ValueError, OverflowError) as exc:
        raise RuntimeError("ffprobe did not report a finite positive audio duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("ffprobe did not report a finite positive audio duration")
    return duration


def _create_recognizer():
    try:
        from shazamio import Shazam
    except ImportError as exc:
        raise RuntimeError("Song identification requires shazamio; install the project dependencies") from exc
    return Shazam()


async def _recognize_sample(recognizer, sample_path, offset, timeout_sec, retries):
    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                recognizer.recognize(str(sample_path)), timeout=timeout_sec
            )
            if response is not None and not isinstance(response, dict):
                raise ValueError("Shazam returned an invalid response")
            track = (response or {}).get("track")
            if track is not None and not isinstance(track, dict):
                raise ValueError("Shazam returned invalid track metadata")
            return True, track
        except Exception as exc:
            retrying = attempt < retries
            LOGGER.warning(
                "Recognition failed at %.3fs (attempt %d/%d): %s%s",
                offset, attempt + 1, retries + 1, str(exc) or type(exc).__name__,
                "; retrying" if retrying else "; skipping sample",
            )
            if retrying:
                await sleep(min(2.0 ** min(attempt, 3), 8.0))
    return False, None


async def identify_songs(
    audio_path, audio_format, interval_sec, dj_segments,
    *, sample_sec=10.0, timeout_sec=30.0, retries=2,
):
    """Sample an audio file without loading the full recording into memory.

    FFmpeg detects the input format; ``audio_format`` remains for compatibility.
    Speech and sample intervals use half-open [start, end) boundaries. ``retries``
    counts additional attempts after the initial request. Failed requests warn
    and continue, but failure of every eligible sample raises ``RuntimeError``.
    """
    interval_sec = _positive_float(interval_sec, "interval_sec")
    sample_sec = _positive_float(sample_sec, "sample_sec")
    timeout_sec = _positive_float(timeout_sec, "timeout_sec")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    speech_windows = _speech_windows(dj_segments)
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio input is not a file: {audio_path}")
    duration = await _probe_duration(audio_path, timeout_sec)
    LOGGER.info(
        "Scanning %.3fs of audio every %.3fs using samples up to %.3fs",
        duration, interval_sec, sample_sec,
    )

    playlist = []
    previous_identity = None
    recognizer = None
    eligible_samples = successful_samples = 0
    sample_index = speech_index = 0
    with TemporaryDirectory(prefix="radio-recognition-") as directory:
        sample_path = Path(directory) / "sample.wav"
        while (offset := sample_index * interval_sec) < duration:
            sample_index += 1
            end = min(offset + sample_sec, duration)
            while speech_index < len(speech_windows) and speech_windows[speech_index][1] <= offset:
                speech_index += 1
            if speech_index < len(speech_windows) and speech_windows[speech_index][0] < end:
                continue
            eligible_samples += 1
            try:
                try:
                    await _run_media_command(
                        [
                            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-ss", str(offset), "-i", str(audio_path), "-t", str(end - offset),
                            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                            "-y", str(sample_path),
                        ],
                        timeout_sec,
                    )
                except _MissingMediaTool:
                    raise
                except RuntimeError as exc:
                    LOGGER.warning("Could not decode sample at %.3fs: %s; skipping sample", offset, exc)
                    continue
                if recognizer is None:
                    recognizer = _create_recognizer()
                succeeded, track = await _recognize_sample(
                    recognizer, sample_path, offset, timeout_sec, retries
                )
                successful_samples += succeeded
                if not track:
                    continue
                title = track.get("title") or "Unknown Title"
                artist = track.get("subtitle") or "Unknown Artist"
                song = f"{title} - {artist}"
                identity = ("key", str(track["key"])) if track.get("key") else ("song", song)
                if identity != previous_identity:
                    playlist.append({
                        "timestamp": str(timedelta(seconds=int(offset))),
                        "start_sec": float(offset),
                        "song": song,
                    })
                    previous_identity = identity
                    LOGGER.info("Song identified at %s: %s", playlist[-1]["timestamp"], song)
            finally:
                sample_path.unlink(missing_ok=True)
    LOGGER.info(
        "Song scan complete: %d eligible samples, %d successful requests, %d songs detected",
        eligible_samples, successful_samples, len(playlist),
    )
    if eligible_samples and not successful_samples:
        raise RuntimeError(
            f"Song identification failed for all {eligible_samples} eligible samples; see warnings for details"
        )
    return playlist
