import os
import httpx


class HTTPLLMAdapter:
    """一个简单的 HTTP 本地 LLM 适配器，尝试兼容 OpenAI-style /v1/chat/completions 接口。

    初始化参数：
    - api_url: 本地服务地址（例如 http://localhost:8080）
    - model: 在服务中注册的模型 id（可选）
    """
    def __init__(self, api_url: str, model: str = None, timeout: int = 60):
        self.api_url = api_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    async def create_chat_completion(self, model, messages, max_tokens=1200, temperature=0.2):
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            ep = f"{self.api_url}/v1/chat/completions"
            try:
                resp = await client.post(ep, json=payload)
                if resp.status_code == 200:
                    return resp.json()
                if 200 <= resp.status_code < 300:
                    try:
                        return resp.json()
                    except Exception:
                        return {"text": resp.text}
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                raise e


