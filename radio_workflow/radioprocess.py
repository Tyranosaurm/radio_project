"""Command-line pipeline for radio transcription and song recognition."""

import argparse
import asyncio
import logging
import math
from pathlib import Path
import shutil

if __package__:
    from .recognition import identify_songs
    from .timeline import format_timestamp, save_to_csv
else:  # Preserve `python radio_workflow/radioprocess.py`.
    from recognition import identify_songs
    from timeline import format_timestamp, save_to_csv

LOGGER = logging.getLogger(__name__)
SAMPLING_INTERVAL_SEC = 30
MIN_SPEECH_DURATION_MS = 2000


def extract_dj_segments(
    audio_path, *, model_name="base", device="cpu", compute_type="int8", language=None
):
    """Transcribe VAD-selected speech, retaining precision for sample overlap checks.

    VAD detects speech-like audio, not speaker roles: singing may also be returned.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Speech transcription requires faster-whisper. "
            "Install requirements.txt or use --skip-speech."
        ) from exc

    LOGGER.info("Loading Whisper model %s on %s (%s)", model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(
        str(audio_path), beam_size=5, language=language, vad_filter=True,
        vad_parameters={
            "min_speech_duration_ms": MIN_SPEECH_DURATION_MS,
            "max_speech_duration_s": 600,
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 400,
        },
    )
    speech = []
    for segment in segments:
        start, end = float(segment.start), float(segment.end)
        text = segment.text.strip()
        if not text or not (math.isfinite(start) and math.isfinite(end)) or end <= max(0.0, start):
            continue
        speech.append({
            "start_sec": max(0.0, start), "end_sec": end,
            "start_str": format_timestamp(max(0.0, start)),
            "end_str": format_timestamp(end), "text": text,
        })
        LOGGER.info("Speech %s → %s", speech[-1]["start_str"], speech[-1]["end_str"])
    return speech


def positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite, positive number of seconds")
    return seconds


def nonnegative_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="Transcribe speech and identify songs in a radio recording."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input audio file; format is detected automatically")
    parser.add_argument("-o", "--output", type=Path, help="CSV output (default: <input>_analysis.csv)")
    parser.add_argument("--interval", type=positive_seconds, default=SAMPLING_INTERVAL_SEC, metavar="SECONDS", help="Time between song samples (default: 30)")
    parser.add_argument("--sample-duration", type=positive_seconds, default=10.0, metavar="SECONDS", help="Length of each song sample (default: 10)")
    parser.add_argument("--request-timeout", type=positive_seconds, default=30.0, metavar="SECONDS", help="Timeout per recognition attempt (default: 30)")
    parser.add_argument("--retries", type=nonnegative_int, default=2, help="Retries per failed recognition sample (default: 2)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-speech", action="store_true", help="Only identify songs; do not load Whisper")
    mode.add_argument("--skip-songs", action="store_true", help="Only transcribe speech; no Shazam requests")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name or local model directory (default: base)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="Whisper device (default: cpu)")
    parser.add_argument("--compute-type", default="int8", help="Whisper compute type; e.g. int8 or float16 (default: int8)")
    parser.add_argument("--language", help="Speech language code; default: auto-detect")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include debug diagnostics on errors")
    return parser


def validate_paths(parser, args):
    args.input = args.input.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"input is not a file: {args.input}")
    if args.input.stat().st_size == 0:
        parser.error(f"input file is empty: {args.input}")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output == args.input or (
            args.output.exists() and args.output.samefile(args.input)
        ):
            parser.error("output must be different from the input recording")
        if args.output.is_dir():
            parser.error(f"output is a directory: {args.output}")
    if not args.skip_songs:
        missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
        if missing:
            parser.error(f"missing {', '.join(missing)}; install FFmpeg and add it to PATH")


async def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        validate_paths(parser, args)
        LOGGER.info("Analyzing %s", args.input)
        speech, songs = [], []
        if not args.skip_speech:
            speech = extract_dj_segments(
                args.input, model_name=args.whisper_model, device=args.device,
                compute_type=args.compute_type, language=args.language,
            )
        if not args.skip_songs:
            songs = await identify_songs(
                args.input, None, args.interval, speech,
                sample_sec=args.sample_duration, timeout_sec=args.request_timeout,
                retries=args.retries,
            )
        output = save_to_csv(args.input, speech, songs, args.output)
        LOGGER.info("Saved %s (%d speech segments, %d songs)", output, len(speech), len(songs))
        return 0
    except Exception as exc:
        LOGGER.error("Analysis failed: %s", exc, exc_info=args.verbose)
        return 1


def cli():
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.warning("Analysis interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
