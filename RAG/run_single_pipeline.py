import sys
import os
from pathlib import Path
import concurrent.futures

# Ensure project root is on sys.path so imports like `from RAG import transcribe` work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Run full pipeline for a single video file:
  - extract audio (ffmpeg)
  - transcribe (whisperx or configured transcribe tool)
  - chunk transcript
  - index chunks into Chroma (persistent dir: ./RAG/rag_db)

Usage:
  python RAG/run_single_pipeline.py <path/to/video.mp4>

Environment:
  TRANSCRIBE_DEVICE and TRANSCRIBE_COMPUTE_TYPE may be set (e.g. cpu / float32)
  CHROMA_DIR can override the default index directory
"""


def process_video(video_path: Path):
    """Process one video file through the pipeline."""
    project_root = Path(__file__).resolve().parents[1]
    rag_dir = project_root / 'RAG'
    audio_dir = rag_dir / 'audio'
    trans_dir = rag_dir / 'transcripts'
    chunks_dir = rag_dir / 'chunks'

    for d in (audio_dir, trans_dir, chunks_dir):
        d.mkdir(parents=True, exist_ok=True)

    name = video_path.stem
    wav_path = trans_dir / (name + '.wav')
    transcript_path = trans_dir / (name + '.json')
    chunk_path = chunks_dir / (name + '_chunks.json')

    # compute stable content hash for deduplication (sha256)
    def compute_sha256(path, block_size=65536):
        import hashlib
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(block_size), b''):
                h.update(chunk)
        return h.hexdigest()

    video_hash = compute_sha256(video_path)

    # 1. extract audio
    def extract_audio(video_path, out_wav):
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        cmd = ['ffmpeg', '-y', '-i', str(video_path), '-vn', '-ac', '1', '-ar', '16000', str(out_wav)]
        subprocess.check_call(cmd)

    # 2. transcribe: prefer faster-whisper Python API, fallback to whisperx CLI if missing
    def run_whisperx(wav_path, out_json):
        import json, subprocess
        tmpdir = out_json.parent
        tmpdir.mkdir(parents=True, exist_ok=True)
        device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
        compute_type = os.getenv('TRANSCRIBE_COMPUTE_TYPE', 'float32')
        # Try faster-whisper first (Python API)
        try:
            from faster_whisper import WhisperModel
            print('Using faster-whisper for transcription (this may take some time)...')
            model = WhisperModel("small", device=device, compute_type=compute_type)
            segments, info = model.transcribe(str(wav_path), beam_size=5)
            segs = []
            for s in segments:
                segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            out = {"video_id": wav_path.stem, "segments": segs}
            out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'Transcription finished: {len(segs)} segments')
            return
        except Exception as e:
            # If faster-whisper not available or failed, try whisperx CLI
            print('faster-whisper unavailable or failed, falling back to whisperx CLI:', e)

        # Fallback: whisperx CLI
        cmd = [
            'whisperx', str(wav_path), '--model', 'small',
            '--device', device,
            '--compute_type', compute_type,
            '--output_format', 'json', '--output_dir', str(tmpdir)
        ]
        print('Calling whisperx CLI for transcription...')
        subprocess.check_call(cmd)
        candidate = tmpdir / (wav_path.stem + '.json')
        if not candidate.exists():
            raise FileNotFoundError('whisperx output not found: ' + str(candidate))
        data = json.loads(candidate.read_text(encoding='utf-8'))
        segments = []
        for seg in data.get('segments', []):
            segments.append({'start': seg['start'], 'end': seg['end'], 'text': seg['text']})
        out = {'video_id': wav_path.stem, 'segments': segments}
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'whisperx transcription finished: {len(segments)} segments')

    # 3. chunk helpers
    def load_transcript(path):
        import json
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def chunk_by_time(transcript, window=30.0, overlap=5.0):
        segments = transcript.get('segments', [])
        if not segments:
            return []
        chunks = []
        curr_text = []
        curr_start = segments[0]['start']
        curr_end = curr_start
        for seg in segments:
            if seg['start'] - curr_start <= window:
                curr_text.append(seg['text'])
                curr_end = seg['end']
            else:
                chunks.append({'video_id': transcript.get('video_id', 'unknown'), 'chunks': [{'text': ' '.join(curr_text), 'start': curr_start, 'end': curr_end}]})
                curr_start = max(0, seg['start'] - overlap)
                curr_text = [seg['text']]
                curr_end = seg['end']
        if curr_text:
            chunks.append({'video_id': transcript.get('video_id', 'unknown'), 'chunks': [{'text': ' '.join(curr_text), 'start': curr_start, 'end': curr_end}]})
        return chunks

    def write_chunks(chunks, out):
        import json
        Path(out).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')

    # 4. index helper using chromadb and sentence-transformers
    def index_file(path):
        import json
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings
        from pathlib import Path as _Path
        MODEL_NAME = os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')
        DB_DIR_LOCAL = os.getenv('CHROMA_DIR', './RAG/rag_db')
        print('Indexing', path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        videos = data if isinstance(data, list) else [data]
        # dedup check: maintain a simple list of indexed video hashes in DB_DIR
        DB_META = _Path(DB_DIR_LOCAL) / 'indexed_videos.json'
        try:
            if DB_META.exists():
                existing_indexed = json.loads(DB_META.read_text(encoding='utf-8'))
            else:
                existing_indexed = []
        except Exception:
            existing_indexed = []
        video_ids = [v.get('video_id') for v in videos if v.get('video_id')]
        for vid in video_ids:
            if vid in existing_indexed:
                print('Video', vid, 'already indexed in', DB_DIR_LOCAL, ', skipping')
                return
        embed_model = SentenceTransformer(MODEL_NAME)
        try:
            client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=DB_DIR_LOCAL))
        except Exception:
            client = chromadb.Client()
        try:
            collection = client.get_collection('videos')
        except Exception:
            collection = client.create_collection('videos')
        import uuid
        # Batch encode all chunks and show progress
        all_texts = []
        all_metas = []
        all_ids = []
        for v in videos:
            vid = v.get('video_id')
            for c in v.get('chunks', []):
                text = c.get('text')
                start = c.get('start')
                end = c.get('end')
                all_texts.append(text)
                all_metas.append({'video_id': vid, 'start': start, 'end': end})
                all_ids.append(str(uuid.uuid4()))
        if all_texts:
            print(f'Encoding {len(all_texts)} chunks using {MODEL_NAME} (this may show a progress bar)...')
            embeddings = embed_model.encode(all_texts, show_progress_bar=True)
            # ensure embeddings is a plain Python list of lists (chroma requires list)
            try:
                embeddings = embeddings.tolist()
            except Exception:
                embeddings = list(embeddings)
            # add in batches to the collection
            batch_size = 512
            for i in range(0, len(all_texts), batch_size):
                j = i + batch_size
                docs = all_texts[i:j]
                metas = all_metas[i:j]
                ids = all_ids[i:j]
                embs = embeddings[i:j]
                collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        try:
            client.persist()
        except Exception:
            try:
                collection.persist()
            except Exception:
                pass
        # record indexed video ids for future dedup checks
        try:
            for vid in video_ids:
                if vid and vid not in existing_indexed:
                    existing_indexed.append(vid)
            _Path(DB_DIR_LOCAL).mkdir(parents=True, exist_ok=True)
            DB_META.write_text(json.dumps(existing_indexed, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    # Execute steps: extract audio, transcribe, chunk, index
    # 1. extract audio if needed
    if not wav_path.exists():
        try:
            print('Extracting audio ->', wav_path)
            extract_audio(video_path, wav_path)
        except Exception as e:
            print('ffmpeg audio extraction failed:', e)
            return False
    else:
        print('Audio already exists:', wav_path)

    # 2. transcribe
    if not transcript_path.exists():
        try:
            print('Transcribing ->', transcript_path)
            run_whisperx(wav_path, transcript_path)
        except Exception as e:
            print('Transcription failed:', e)
            print("You can set TRANSCRIBE_DEVICE='cpu' and TRANSCRIBE_COMPUTE_TYPE='float32' to force CPU")
            return False
    else:
        print('Transcript already exists:', transcript_path)

    # 3. chunk
    if not chunk_path.exists():
        try:
            print('Chunking transcript ->', chunk_path)
            trans = load_transcript(transcript_path)
            chunks = chunk_by_time(trans, window=30.0, overlap=5.0)
            write_chunks(chunks, chunk_path)
        except Exception as e:
            print('Chunking failed:', e)
            return False
    else:
        print('Chunks already exist:', chunk_path)

    # 4. index
    try:
        print('Indexing chunks -> using CHROMA_DIR=', os.getenv('CHROMA_DIR', './RAG/rag_db'))
        index_file(str(chunk_path))
    except Exception as e:
        print('Indexing failed:', e)
        return False

    print('Pipeline completed for', video_path.name, '. Chroma DB at', os.getenv('CHROMA_DIR', './RAG/rag_db'))
    return True


def main():
    project_root = Path(__file__).resolve().parents[1]
    downloads = project_root / 'downloads'
    # If user provided file(s), process those, otherwise process all mp4 in downloads
    if len(sys.argv) >= 2:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        if not downloads.exists():
            print('Downloads folder not found:', downloads)
            sys.exit(1)
        paths = [p for p in downloads.iterdir() if p.suffix.lower() in {'.mp4', '.mkv', '.mov', '.avi'}]
    if not paths:
        print('No video files found to process in', downloads)
        sys.exit(0)
    # Parallelize processing across multiple CPU processes. Keep CPU-only transcription.
    # Conservative default for CPU-only, moderate memory machines (e.g. 32GB)
    # Use a small number of worker processes to avoid heavy CPU and memory contention.
    max_workers = 2
    print(f'Processing {len(paths)} file(s) with {max_workers} worker(s) (CPU-only conservative setting)')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_video, p): p for p in paths}
        for future in concurrent.futures.as_completed(future_to_path):
            p = future_to_path[future]
            try:
                ok = future.result()
            except Exception as e:
                print('Processing raised exception for', p.name, ':', e)
            else:
                if not ok:
                    print('Processing failed for', p.name)
    print('All done')


if __name__ == '__main__':
    main()


