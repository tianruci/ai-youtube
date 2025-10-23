import os
import sys
import json
import subprocess
from pathlib import Path

"""
Simple transcription wrapper.

Two modes:
- 'whisper' (requires openai/whisper or whisperx installed) -- fast to add but may need GPU
- 'openai' (not implemented) -- placeholder

This script extracts audio with ffmpeg (if needed) and runs whisper/transcribe via a local command.

Usage:
  python transcribe.py <video_or_wav_path> <out_json_path>

Output JSON format:
{
  "video_id": "<basename>",
  "segments": [ {"start": float, "end": float, "text": "..."}, ... ]
}

Note: This is a simple wrapper that calls `whisperx` if available via CLI.
"""


def extract_audio(video_path, out_wav):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path), '-vn', '-ac', '1', '-ar', '16000', str(out_wav)
    ]
    subprocess.check_call(cmd)


def run_whisperx(wav_path, out_json):
    # Expect whisperx CLI to be available; otherwise user should run python-based whisperx
    # We'll run: whisperx <wav> --model small --device <device> --compute_type <type> --output_format json --output_dir <tmpdir>
    tmpdir = out_json.parent
    tmpdir.mkdir(parents=True, exist_ok=True)
    device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
    compute_type = os.getenv('TRANSCRIBE_COMPUTE_TYPE', 'float32')
    cmd = [
        'whisperx', str(wav_path), '--model', 'small',
        '--device', device,
        '--compute_type', compute_type,
        '--output_format', 'json', '--output_dir', str(tmpdir)
    ]
    subprocess.check_call(cmd)
    # whisperx outputs file like <wavbasename>.json
    candidate = tmpdir / (wav_path.stem + '.json')
    if not candidate.exists():
        raise FileNotFoundError('whisperx output not found: ' + str(candidate))
    # Load and convert to our simplified segments format
    data = json.loads(candidate.read_text(encoding='utf-8'))
    segments = []
    for seg in data.get('segments', []):
        segments.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text']})
    out = {'video_id': wav_path.stem, 'segments': segments}
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    if len(sys.argv) < 3:
        print('Usage: python transcribe.py <video_or_wav_path> <out_json_path>')
        sys.exit(1)
    inp = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    if not inp.exists():
        print('Input not found:', inp)
        sys.exit(1)
    if inp.suffix.lower() not in ['.wav', '.mp3']:
        wav = out_json.with_suffix('.wav')
        extract_audio(inp, wav)
    else:
        wav = inp
    try:
        run_whisperx(wav, out_json)
    except Exception as e:
        print('whisperx failed:', e)
        print('You may need to install whisperx or provide your own transcript')


if __name__ == '__main__':
    main()


