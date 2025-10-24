# test_whisper_load2.py
from faster_whisper import WhisperModel
from pathlib import Path
p = Path(r"D:\projects\ai-youtube\model\models--openai--whisper-small")

candidates = [
    str(p),
    str(p).replace("\\","/"),
    str(p / "model.safetensors"),
    str(p / "model.safetensors").replace("\\","/"),
    str(p / "pytorch_model.bin"),
    str(p / "pytorch_model.bin").replace("\\","/"),
]

for arg in candidates:
    try:
        print("尝试加载:", arg)
        m = WhisperModel(arg, device="cpu", compute_type="float32")
        print("加载成功，模型对象:", type(m))
        break
    except Exception as e:
        import traceback
        print("加载失败:", repr(e))
        traceback.print_exc()