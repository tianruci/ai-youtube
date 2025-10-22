"""
Small end-to-end example:
1. Index a demo transcripts JSON
2. Run a query against the index and call local model

Usage:
  python run_example.py
"""
import os
from index_data import index_file
from query_rag import answer


DEMO_JSON = os.path.join(os.path.dirname(__file__), "demo_transcript.json")


def make_demo():
    data = [
        {
            "video_id": "demo1",
            "chunks": [
                {"text": "欢迎来到本视频，我们将介绍人工智能的基本概念。", "start": 0.0, "end": 5.0},
                {"text": "本视频还会讨论模型微调与检索增强生成（RAG）。", "start": 5.0, "end": 12.0},
            ],
        }
    ]
    import json
    with open(DEMO_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    make_demo()
    print("Indexing demo data...")
    index_file(DEMO_JSON)
    print("Querying...")
    ans = answer("这个视频主要讲了什么？")
    print("Answer:\n", ans)


if __name__ == '__main__':
    main()


