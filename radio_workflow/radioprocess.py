import os
import csv
import argparse
import asyncio
from datetime import timedelta
from pydub import AudioSegment
from faster_whisper import WhisperModel
from shazamio import Shazam

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
SAMPLING_INTERVAL_SEC = 30           # Check for a new song every 30 seconds
MIN_SPEECH_DURATION_MS = 2000        # Filter out random vocal noises under 2 seconds

# ==========================================
# PHASE 1: DJ SPEECH EXTRACTION (Whisper + VAD)
# ==========================================
def extract_dj_segments(audio_path):
    print("\n[-] Initializing Whisper Model with Silero VAD...")
    # Runs on CPU by default. Switch device="cuda" if you have an NVIDIA GPU.
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print(f"[-] Processing audio for DJ speech: {audio_path}")
    print("    (Filtering out singing and isolating spoken word using VAD thresholds)")
    
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            max_speech_duration_s=600,
            min_silence_duration_ms=1000,
            speech_pad_ms=400
        )
    )
    
    dj_talks = []
    for segment in segments:
        dj_talks.append({
            "start_sec": int(segment.start),
            "end_sec": int(segment.end),
            "start_str": str(timedelta(seconds=int(segment.start))),
            "end_str": str(timedelta(seconds=int(segment.end))),
            "text": segment.text.strip()
        })
        print(f"   [DJ Detected] {dj_talks[-1]['start_str']} -> {dj_talks[-1]['end_str']}: \"{segment.text.strip()[:50]}...\"")
        
    return dj_talks

# ==========================================
# PHASE 2: SONG IDENTIFICATION (Shazam API)
# ==========================================
async def identify_songs(audio_path, audio_format, interval_sec, dj_segments):
    print("\n[-] Initializing Shazam Engine for fingerprinting...")
    shazam = Shazam()
    
    print(f"[-] Loading audio file as '{audio_format}' format...")
    full_audio = AudioSegment.from_file(audio_path, format=audio_format)
    duration_ms = len(full_audio)
    duration_sec = duration_ms // 1000
    
    playlist = []
    last_tracked_title = None
    
    print(f"[-] Scanning music tracks every {interval_sec} seconds (skipping active DJ windows)...")
    for offset_sec in range(0, duration_sec, interval_sec):
        
        # Optimization: Skip fingerprinting if the current timestamp falls inside a known DJ talking block
        is_dj_speaking = any(dj['start_sec'] <= offset_sec <= dj['end_sec'] for dj in dj_segments)
        if is_dj_speaking:
            continue
            
        start_ms = offset_sec * 1000
        end_ms = min(start_ms + 10000, duration_ms) 
        
        audio_slice = full_audio[start_ms:end_ms]
        temp_slice_path = f"temp_slice_{offset_sec}.mp3"
        audio_slice.export(temp_slice_path, format="mp3")
        
        try:
            out = await shazam.recognize(temp_slice_path)
            if out and 'track' in out:
                track_info = out['track']
                title = track_info.get('title', 'Unknown Title')
                artist = track_info.get('subtitle', 'Unknown Artist')
                current_song_identity = f"{title} - {artist}"
                
                if current_song_identity != last_tracked_title:
                    timestamp_str = str(timedelta(seconds=offset_sec))
                    playlist.append({
                        "timestamp": timestamp_str,
                        "song": current_song_identity
                    })
                    print(f"   [Song Identified] {timestamp_str}: {current_song_identity}")
                    last_tracked_title = current_song_identity
        except Exception:
            pass
        finally:
            if os.path.exists(temp_slice_path):
                os.remove(temp_slice_path)
                
    return playlist

# ==========================================
# PHASE 3: CSV EXPORT UTILITY
# ==========================================
def save_to_csv(input_path, dj_segments, song_segments):
    base_name, _ = os.path.splitext(input_path)
    csv_filename = f"{base_name}_analysis.csv"
    
    print(f"\n[-] Compiling timeline results into CSV: {csv_filename}...")
    
    # Consolidate all entries into a unified timeline row model
    timeline_rows = []
    
    for dj in dj_segments:
        timeline_rows.append({
            "Timestamp": dj['start_str'],
            "Type": "🎤 DJ Speaking",
            "Details": f"Spoke until {dj['end_str']} | Transcript: {dj['text']}"
        })
        
    for song in song_segments:
        timeline_rows.append({
            "Timestamp": song['timestamp'],
            "Type": "🎵 Song Played",
            "Details": song['song']
        })
        
    # Sort chronological logs by timestamp string
    timeline_rows.sort(key=lambda x: x['Timestamp'])
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Type", "Details"])
        writer.writeheader()
        writer.writerows(timeline_rows)
        
    print(f"[✓] Success! Timeline logs saved.")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
async def main():
    parser = argparse.ArgumentParser(description="Analyze a radio show recording to extract DJ segments and identify songs.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input audio file (e.g., show.mp3, show.m4a)")
    args = parser.parse_args()

    audio_file = args.input

    if not os.path.exists(audio_file):
        print(f"[!] Error: File '{audio_file}' not found.")
        return

    _, file_extension = os.path.splitext(audio_file)
    audio_format = file_extension.lower().replace(".", "")

    if not audio_format:
        print("[!] Error: Could not determine the audio file format.")
        return

    print("=== STARTING RADIO RECORDING ANALYSIS ===")
    print(f"Target File: {audio_file}")
    print(f"Detected Format: {audio_format.upper()}")
    
    # Execute Pipeline
    dj_segments = extract_dj_segments(audio_file)
    song_segments = await identify_songs(audio_file, audio_format, SAMPLING_INTERVAL_SEC, dj_segments)
    
    # Save Outputs
    save_to_csv(audio_file, dj_segments, song_segments)

if __name__ == "__main__":
    asyncio.run(main())
