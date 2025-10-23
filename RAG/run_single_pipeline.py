import sys
import os
from pathlib import Path

# Mitigate OpenMP runtime conflicts (safe default for most CPU-only setups)
# Do not override if user explicitly set these env vars
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('MKL_NUM_THREADS', '1'))

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python RAG/run_single_pipeline.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print("Video not found:", video_path)
        sys.exit(1)

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

    # Inline helpers to make this script standalone and allow removing other helper files
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
            model = WhisperModel("small", device=device, compute_type=compute_type)
            segments, info = model.transcribe(str(wav_path), beam_size=5)
            segs = []
            for s in segments:
                segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            out = {"video_id": wav_path.stem, "segments": segs}
            out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
            return
        except Exception as e:
            # If faster-whisper not available or failed, try whisperx CLI
            # print a warning and fall back
            print('faster-whisper unavailable or failed, falling back to whisperx CLI:', e)

        # Fallback: whisperx CLI
        cmd = [
            'whisperx', str(wav_path), '--model', 'small',
            '--device', device,
            '--compute_type', compute_type,
            '--output_format', 'json', '--output_dir', str(tmpdir)
        ]
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
        MODEL_NAME = os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')
        DB_DIR_LOCAL = os.getenv('CHROMA_DIR', './RAG/rag_db')
        print('Indexing', path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        videos = data if isinstance(data, list) else [data]
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
        for v in videos:
            vid = v.get('video_id')
            for c in v.get('chunks', []):
                cid = str(uuid.uuid4())
                text = c.get('text')
                start = c.get('start')
                end = c.get('end')
                emb = embed_model.encode(text).tolist()
                collection.add(documents=[text], metadatas=[{'video_id': vid, 'start': start, 'end': end}], ids=[cid], embeddings=[emb])
        try:
            client.persist()
        except Exception:
            try:
                collection.persist()
            except Exception:
                pass

    # Try to import tqdm for progress bars; provide a lightweight fallback if missing
    try:
        from tqdm import tqdm
    except Exception:
        class _NoopTqdm:
            def __init__(self, iterable=None, **kwargs):
                self._iterable = iterable
            def __call__(self, iterable=None, **kwargs):
                return iter(iterable) if iterable is not None else self
            def __iter__(self):
                return iter(self._iterable) if self._iterable is not None else iter(())
            def update(self, n=1):
                pass
            def set_description(self, desc):
                pass
            def close(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        def tqdm(iterable=None, **kwargs):
            if iterable is None:
                return _NoopTqdm()
            return iter(iterable)

    # Execute steps: extract audio, transcribe, chunk, index with a pipeline-level progress bar
    total_steps = 4
    with tqdm(total=total_steps, desc='Pipeline') as pipeline_bar:
        # 1. extract audio if needed
        if not wav_path.exists():
            try:
                print('Step 1/{}/Extracting audio ->'.format(total_steps), wav_path)
                extract_audio(video_path, wav_path)
            except Exception as e:
                print('ffmpeg audio extraction failed:', e)
                sys.exit(1)
        else:
            print('Audio already exists:', wav_path)
        pipeline_bar.update(1)

        # 2. transcribe
        if not transcript_path.exists():
            try:
                print('Step 2/{}/Transcribing ->'.format(total_steps), transcript_path)
                run_whisperx(wav_path, transcript_path)
            except Exception as e:
                print('Transcription failed:', e)
                print("You can set TRANSCRIBE_DEVICE='cpu' and TRANSCRIBE_COMPUTE_TYPE='float32' to force CPU")
                sys.exit(1)
        else:
            print('Transcript already exists:', transcript_path)
        pipeline_bar.update(1)

        # 3. chunk
        if not chunk_path.exists():
            try:
                print('Step 3/{}/Chunking transcript ->'.format(total_steps), chunk_path)
                trans = load_transcript(transcript_path)
                chunks = chunk_by_time(trans, window=30.0, overlap=5.0)
                write_chunks(chunks, chunk_path)
            except Exception as e:
                print('Chunking failed:', e)
                sys.exit(1)
        else:
            print('Chunks already exist:', chunk_path)
        pipeline_bar.update(1)

        # 4. index
        try:
            print('Step 4/{}/Indexing chunks -> using CHROMA_DIR='.format(total_steps), os.getenv('CHROMA_DIR', './RAG/rag_db'))
            index_file(str(chunk_path))
        except Exception as e:
            print('Indexing failed:', e)
            sys.exit(1)
        pipeline_bar.update(1)

    print('Pipeline completed. Chroma DB at', os.getenv('CHROMA_DIR', './RAG/rag_db'))


if __name__ == '__main__':
    main()


