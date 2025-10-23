import sys
import json
from pathlib import Path

"""
Simple chunker: merge transcript segments into time-window chunks.

Usage:
 python chunk.py <transcript_json> <out_chunks_json> [--window 30] [--overlap 5]

Output format matches index_data.py expectation.
"""


def load_transcript(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_chunks(chunks, out):
    Path(out).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')


def chunk_by_time(transcript, window=30.0, overlap=5.0):
    segments = transcript.get('segments', [])
    if not segments:
        return []
    chunks = []
    i = 0
    start_time = segments[0]['start']
    curr_text = []
    curr_start = start_time
    for seg in segments:
        if seg['start'] - curr_start <= window:
            curr_text.append(seg['text'])
            curr_end = seg['end']
        else:
            chunks.append({'video_id': transcript.get('video_id', 'unknown'), 'chunks': [{'text': ' '.join(curr_text), 'start': curr_start, 'end': curr_end}]})
            # start new
            curr_start = seg['start'] - overlap
            if curr_start < 0:
                curr_start = 0
            curr_text = [seg['text']]
            curr_end = seg['end']
    # final
    if curr_text:
        chunks.append({'video_id': transcript.get('video_id', 'unknown'), 'chunks': [{'text': ' '.join(curr_text), 'start': curr_start, 'end': curr_end}]})
    # flatten each video's chunks into single object list (index_data accepts list of video dicts)
    # but maintain structure: return list with one entry per video
    # Note: if transcript already per-video, keep as single video object
    return chunks


def main():
    if len(sys.argv) < 3:
        print('Usage: python chunk.py <transcript.json> <out_chunks.json> [--window 30] [--overlap 5]')
        return
    tfile = sys.argv[1]
    outfile = sys.argv[2]
    window = 30.0
    overlap = 5.0
    if '--window' in sys.argv:
        window = float(sys.argv[sys.argv.index('--window') + 1])
    if '--overlap' in sys.argv:
        overlap = float(sys.argv[sys.argv.index('--overlap') + 1])
    trans = load_transcript(tfile)
    chunks = chunk_by_time(trans, window=window, overlap=overlap)
    # chunk_by_time returns list of video dicts but with nested chunk lists; we need to merge if necessary
    # For simplicity, write the returned structure as a JSON list
    write_chunks(chunks, outfile)


if __name__ == '__main__':
    main()


