# Radio Show Audio Analyzer

An automated command-line Python utility designed to digest full-length radio broadcasts or podcast recordings. The pipeline performs **Voice Activity Detection (VAD)** to identify when a DJ is speaking, isolates voice text snippets, and leverages acoustic audio fingerprinting to index individual background songs.

## Features
- **Automatic Extension Detection:** Parses `.mp3`, `.m4a`, `.wav`, and other formats seamlessly.
- **Smart Performance Gating:** Skips checking for songs during active DJ dialogue segments to minimize external server API load.
- **Structured Spreadsheet Export:** Automatically logs a chronological, sorted index into a `.csv` timeline sheet.

## Prerequisites
Before running the analyzer script, make sure your computer has **FFmpeg** installed and added to your operating system's system environment variables. 
- **macOS:** `brew install ffmpeg`
- **Windows / Linux:** Download from the official repository and attach to system paths.

## Installation

1. Prepare your local Python environment:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Syntax

You can run the script directly from your terminal pane by specifying your media asset with the `-i` or `--input` parameter flag:

### Running an MP3 File Example
```bash
python radio_analyzer.py -i rock_show_october.mp3
```

### Running an M4A File Example
```bash
python radio_analyzer.py --input electronic_mix_session.m4a
```

## Understanding the Output File
Once processing runs to completion, a companion file using the format `[your_filename]_analysis.csv` will be created automatically. The contents look like this:

| Timestamp | Type | Details |
| :--- | :--- | :--- |
| 0:01:15 | 🎵 Song Played | One More Time - Daft Punk |
| 0:05:22 | 🎤 DJ Speaking | Spoke until 0:05:45 \| Transcript: Welcome back to the morning mix... |
| 0:06:00 | 🎵 Song Played | Starlight - Muse |
