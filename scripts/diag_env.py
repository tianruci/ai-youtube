import os
import json

def mask_key(k: str) -> str:
    if not k:
        return '(not set)'
    if len(k) <= 8:
        return '****'
    return k[:4] + '...' + k[-4:]

base_url = os.getenv('OPENAI_BASE_URL')
model_name = os.getenv('OPENAI_MODEL_NAME')
api_key = os.getenv('OPENAI_API_KEY')

info = {
    'OPENAI_API_KEY': mask_key(api_key),
    'OPENAI_BASE_URL': base_url if base_url else '(not set)',
    'OPENAI_MODEL_NAME': model_name if model_name else '(not set)',
    'BASE_URL_has_scheme': bool(base_url and (base_url.startswith('http://') or base_url.startswith('https://'))),
    'BASE_URL_contains_api_segment': bool(base_url and '/api' in base_url),
}

print(json.dumps(info, ensure_ascii=False, indent=2))


