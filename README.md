# Radio Show Audio Analyzer

A Python command-line utility that transcribes speech in radio broadcasts or podcasts and identifies songs through Shazam. It exports one chronological CSV timeline.

## Requirements and installation

Use Python 3.10–3.12 and install **FFmpeg**, including `ffprobe`, on your `PATH`:

- macOS: `brew install ffmpeg`
- Ubuntu / Debian: `sudo apt-get update && sudo apt-get install ffmpeg`
- Windows: `winget install --id Gyan.FFmpeg`, then open a new terminal.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
radio-analyzer --help
```

The install adds the `radio-analyzer` command. You can also use `python radio_analyzer.py` from the repository root, or the existing `python radio_workflow/radioprocess.py` entry point. The legacy `radio_workflow/requirements.txt` remains usable when running installation from that directory.

The first transcription run downloads the selected Whisper model. Later runs can use its local cache, or you can pass a downloaded model directory with `--whisper-model`. Whisper runs locally; song recognition sends short audio samples to the external Shazam service and requires network access. `--skip-songs` disables those requests. No API key is required.

## Usage

```bash
# Speech transcription and song recognition; format detected from the file
radio-analyzer -i "radio_workflow/media/show.m4a"

# Select an output file and scan for songs every 20 seconds
radio-analyzer -i "show.mp3" -o "radio_workflow/output/timeline.csv" --interval 20

# Songs only: no Whisper model loading or speech-based skipping
radio-analyzer -i "show.wav" --skip-speech

# Speech only, with a smaller model and a known language
radio-analyzer -i "show.mp3" --skip-songs --whisper-model tiny --language en

# NVIDIA GPU (requires a compatible CTranslate2 / CUDA / cuDNN installation)
radio-analyzer -i "show.mp3" --device cuda --compute-type float16
```

| Option | Default | Purpose |
| --- | --- | --- |
| `-i`, `--input` | Required | Input audio path; supported formats are detected by the decoder |
| `-o`, `--output` | `<input_stem>_analysis.csv` beside input | CSV destination; parent folders are created |
| `--interval` | `30` | Seconds between song samples; positive fractions are accepted |
| `--sample-duration` | `10` | Maximum length in seconds of each song sample |
| `--request-timeout` | `30` | Maximum seconds per recognition attempt |
| `--retries` | `2` | Additional attempts after a failed recognition request, with bounded backoff |
| `--skip-speech` | Off | Only identify songs |
| `--skip-songs` | Off | Only transcribe speech |
| `--whisper-model` | `base` | Whisper model name or local directory |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--compute-type` | `int8` | Whisper compute type; select one supported by your device |
| `--language` | Auto-detect | Speech language code, e.g. `en` or `fr` |
| `-v`, `--verbose` | Off | Include debug diagnostics and stack traces on failures |

`--skip-speech` and `--skip-songs` are mutually exclusive. CPU / int8 remains the default. FFmpeg and ffprobe are used for song sampling; speech-only mode uses faster-whisper's decoder.

## Behavior and output

1. Whisper transcribes VAD-selected speech, retaining fractional timestamps internally.
2. FFmpeg seeks to each candidate song sample and decodes only that short window into a temporary WAV. The song scanner does not load the full recording into memory or re-encode samples to MP3. Whisper transcription still has its own memory requirements.
3. Any sample overlapping a detected speech segment is skipped. Consecutive detections of the same song are collapsed.
4. Speech and song results are sorted numerically, including recordings exceeding ten hours or spanning multiple days, and exported with the existing CSV columns:

| Timestamp | Type | Details |
| --- | --- | --- |
| 0:01:15 | 🎵 Song Played | One More Time - Daft Punk |
| 0:05:22 | 🎤 DJ Speaking | Spoke until 0:05:45 \| Transcript: Welcome back to the morning mix... |
| 0:06:00 | 🎵 Song Played | Starlight - Muse |

Displayed timestamps are rounded down to complete seconds. A song timestamp is its first successful sampled detection, **not its exact start time**. Sampling can miss short tracks. VAD detects speech-like audio; it cannot reliably distinguish a DJ from singing or attribute a speaker role. The legacy “DJ Speaking” CSV label is retained for compatibility. If singing causes too many samples to be skipped, use `--skip-speech` for a separate music scan.

Recognition failures are logged and retried. An unrecognized sample is a normal result, not an error. Partial recognition failures produce warnings and the available results; if every eligible sample fails, the run fails rather than producing a misleading empty success. An all-skipped or no-match scan can validly produce no song rows.

Temporary samples are isolated per run and cleaned up. CSV files are written atomically: an existing result survives an interrupted or failed write. An explicit destination replaces any existing CSV there after successful analysis; the input recording cannot be used as the output path.

Exit status is `0` for a completed run (check warnings for partial recognition failures), `1` for analysis failure, `2` for invalid arguments or prerequisites, and `130` for interruption. Use `--verbose` to investigate an error. For `ffmpeg` / `ffprobe` not found, install FFmpeg and verify both commands work in the same terminal.

## Development

The regression suite uses Python's standard library and mocked model / service responses, so it does not download models or send audio externally:

```bash
python -m pip install --no-deps .
python -m unittest discover -s tests -v
```

Tests cover chronological and atomic CSV output, CLI validation and exit codes, sample overlap, safe temporary files, and recognition retry / timeout behavior. GitHub Actions runs the offline suite on Python 3.10 and 3.12. Real model accuracy and Shazam availability require a separate check with your own recordings.
