import os
import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

"""
Index transcript chunks into ChromaDB.

Expected input: a folder of JSON files or a single JSON list file where each item is:
{
  "video_id": "...",
  "chunks": [
    {"text": "...", "start": 12.3, "end": 15.7},
    ...
  ]
}

This script will create/append to a Chroma collection named 'videos'.
"""

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DB_DIR = os.getenv("CHROMA_DIR", "./rag_db")


def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # support either list of videos or single video dict
    if isinstance(data, dict):
        return [data]
    return data


def chunk_id():
    return str(uuid.uuid4())


def index_file(path):
    print(f"Indexing {path}")
    videos = load_input(path)
    embed_model = SentenceTransformer(MODEL_NAME)
    # initialize persistent Chroma client using duckdb+parquet backend
    try:
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=DB_DIR
        ))
    except ValueError as e:
        # Newer chromadb versions may require migration or different client construction.
        # Fall back to in-memory client and warn the user with remediation steps.
        print("Warning: chromadb persistent client construction failed:", e)
        print("Falling back to in-memory Chroma. To enable persistence, either: ")
        print("  1) Install and run the migration tool: pip install chroma-migrate && chroma-migrate")
        print("  2) Or pin chromadb to a compatible version, e.g.: pip install 'chromadb<0.4.0'\n")
        client = chromadb.Client()
    try:
        collection = client.get_collection("videos")
    except Exception:
        collection = client.create_collection("videos")

    for v in videos:
        vid = v.get("video_id")
        for c in v.get("chunks", []):
            cid = chunk_id()
            text = c.get("text")
            start = c.get("start")
            end = c.get("end")
            emb = embed_model.encode(text).tolist()
            collection.add(
                documents=[text],
                metadatas=[{"video_id": vid, "start": start, "end": end}],
                ids=[cid],
                embeddings=[emb],
            )
    # persist DB to disk
    try:
        client.persist()
    except Exception:
        # older chromadb versions expose persist on the collection
        try:
            collection.persist()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python index_data.py <transcripts.json>")
        sys.exit(1)
    index_file(sys.argv[1])


