import os
from pathlib import Path
import subprocess
import json

"""
One-command pipeline to index all videos in ./downloads.

Usage:
  python pipeline_index.py

Steps:
  - For each file in ./downloads with video extension, extract audio to ./RAG/audio
  - Transcribe to ./RAG/transcripts/<name>.json (uses transcribe.py)
  - Chunk to ./RAG/chunks/<name>_chunks.json (uses chunk.py)
  - Index chunks via index_data.py

Requirements:
  - ffmpeg in PATH
  - whisperx or other transcribe method callable by transcribe.py
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = PROJECT_ROOT / 'downloads'
RAG_DIR = PROJECT_ROOT / 'RAG'
DATA_DIR = RAG_DIR / 'data'


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def video_files():
    exts = {'.mp4', '.mkv', '.mov', '.avi'}
    for f in DOWNLOADS.iterdir():
        if f.suffix.lower() in exts and f.is_file():
            yield f


def run():
    ensure_dirs()
    for vid in video_files():
        name = vid.stem
        print('Processing', vid.name)
        wav_out = DATA_DIR / (name + '.wav')
        # extract audio
        if not wav_out.exists():
            subprocess.check_call(['ffmpeg', '-y', '-i', str(vid), '-vn', '-ac', '1', '-ar', '16000', str(wav_out)])
        # transcribe
        trans_out = DATA_DIR / (name + '.json')
        if not trans_out.exists():
            subprocess.check_call(['python', str(RAG_DIR / 'transcribe.py'), str(wav_out), str(trans_out)])
        # chunk
        chunk_out = DATA_DIR / (name + '_chunks.json')
        if not chunk_out.exists():
            subprocess.check_call(['python', str(RAG_DIR / 'chunk.py'), str(trans_out), str(chunk_out)])
        # index
        subprocess.check_call(['python', str(RAG_DIR / 'index_data.py'), str(chunk_out)])


if __name__ == '__main__':
    run()


