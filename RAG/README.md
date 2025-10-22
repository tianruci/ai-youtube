
RAG module for ai-youtube project

This folder contains a minimal RAG example using sentence-transformers, ChromaDB, and a local Qwen GGUF model invoked via HTTP (preferred) or CLI fallback.

Structure:
- `index_data.py`  — index transcripts into Chroma
- `query_rag.py`   — retrieve + call local model
- `run_example.py` — end-to-end demo (creates demo_transcript.json, indexes, queries)
- `requirements.txt`

Quickstart:
1. Install dependencies: `pip install -r RAG/requirements.txt`
2. Ensure `llama-server` is running on `http://localhost:8080` (or set `LLAMA_HTTP` env)
3. Run demo: `python RAG/run_example.py`

Configuration via environment variables:
- `LLAMA_HTTP` — base URL of llama-server (default `http://localhost:8080`)
- `GGUF_PATH`  — path to GGUF file if `llama-cli` fallback is used
- `EMBED_MODEL` — sentence-transformers model name

Notes:
- This example uses ChromaDB client default settings (local ephemeral DB). For production, configure persistent Chroma, Milvus, or Pinecone.



