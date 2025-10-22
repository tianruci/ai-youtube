import os
import requests
import shutil
from sentence_transformers import SentenceTransformer
import chromadb

"""
Query RAG pipeline: retrieve top-k from Chroma and call local Qwen model.

Configuration via env:
- LLAMA_HTTP: base URL of local model server, e.g. http://localhost:8080
- GGUF_PATH: path to GGUF for fallback use with llama-cli
"""

LLAMA_HTTP = os.getenv("LLAMA_HTTP", "http://localhost:8080")
GGUF_PATH = os.getenv("GGUF_PATH", "D:\\models\\Qwen3-8B-gpt-5-reasoning-distill-Q4_K_M.gguf")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def retrieve(query, n=5):
    client = chromadb.Client()
    collection = client.get_collection("videos")
    results = collection.query(query_texts=[query], n_results=n)
    return results


def build_prompt(query, results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ctx = "\n\n".join([f"[{m['video_id']} {m['start']}~{m['end']}] {d}" for d, m in zip(docs, metas)])
    prompt = (
        "你是一个视频内容解读助手。仅基于下面检索到的片段回答问题，"
        "并在答案中标注引用的时间戳。如果无法从片段中得出结论，就说\"我不知道\"。\n\n"
        "检索片段：\n" + ctx + "\n\n用户问题：" + query + "\n\n回答："
    )
    return prompt


def call_llama_http(prompt):
    # Prefer a single endpoint configurable via env. If not set, try common endpoints.
    preferred = os.getenv('LLAMA_API_ENDPOINT')
    endpoints = [preferred] if preferred else ['/v1/generate', '/generate', '/v1/chat/completions']
    payloads = {
        '/v1/generate': {"prompt": prompt, "max_new_tokens": 512},
        '/generate': {"prompt": prompt, "max_new_tokens": 512},
        '/v1/chat/completions': {"messages": [{"role": "user", "content": prompt}], "max_tokens": 512},
    }
    for ep in endpoints:
        if not ep:
            continue
        url = LLAMA_HTTP.rstrip('/') + ep
        try:
            r = requests.post(url, json=payloads.get(ep, {"prompt": prompt, "max_new_tokens": 512}), timeout=30)
            r.raise_for_status()
            data = r.json()
            # try common response shapes
            if isinstance(data, dict):
                if 'text' in data:
                    return data['text']
                if 'outputs' in data and isinstance(data['outputs'], list):
                    return data['outputs'][0].get('text') or str(data['outputs'][0])
                if 'choices' in data and isinstance(data['choices'], list):
                    c = data['choices'][0]
                    if 'message' in c:
                        return c['message'].get('content')
                    if 'text' in c:
                        return c['text']
            # fallback to raw text
            return r.text
        except Exception:
            continue
    # no HTTP endpoint succeeded
    return None


def call_llama_cli(prompt):
    import subprocess, shlex, tempfile
    # check whether llama-cli exists on PATH
    cli_path = shutil.which('llama-cli')
    if cli_path is None:
        raise RuntimeError('llama-cli not found in PATH. Install llama.cpp tools or set up an alternative local client.')
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix='.txt') as tf:
        tf.write(prompt)
        tf.flush()
        cmd = f'"{cli_path}" -m "{GGUF_PATH}" -p "{prompt}" -n 512'
        res = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        return res.stdout or res.stderr


def answer(query):
    results = retrieve(query)
    prompt = build_prompt(query, results)
    resp = call_llama_http(prompt)
    if resp:
        return resp
    # HTTP failed; attempt CLI but surface clear error if CLI missing
    try:
        resp = call_llama_cli(prompt)
        return resp
    except Exception as e:
        raise RuntimeError(
            "模型调用失败：HTTP 请求未能连通。\n" 
            "请确认 `llama-server` 的 API endpoint（示例: http://localhost:8080/v1/generate）是否可用，\n"
            "或在系统中安装 `llama-cli` 并确保其在 PATH 中。\n"
            f"详细错误: {e}"
        )


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python query_rag.py "your question"')
        sys.exit(1)
    q = sys.argv[1]
    ans = answer(q)
    print(ans)


