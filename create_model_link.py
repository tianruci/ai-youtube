import os
import shutil
import sys

# Cross-platform helper script to create model.bin in a whisper model directory.
# Behavior:
# 1. If model.bin already exists -> do nothing
# 2. If pytorch_model.bin exists -> try symlink -> try hardlink -> copy
# 3. Exit with code 0 on success, non-zero on failure

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "models--openai--whisper-small")
SRC_NAME = "pytorch_model.bin"
DST_NAME = "model.bin"

src = os.path.join(MODEL_DIR, SRC_NAME)
dst = os.path.join(MODEL_DIR, DST_NAME)

def main():
    if os.path.exists(dst):
        print(f"目标已存在，跳过： {dst}")
        return 0

    # 使用局部变量避免函数内覆盖模块级 src
    src_path = src
    if not os.path.exists(src_path):
        # 尝试兼容其他常见文件名
        alt_src = os.path.join(MODEL_DIR, "pytorch_model.safetensors")
        if os.path.exists(alt_src):
            src_path = alt_src
        else:
            print(f"未找到源文件： {src} 或 {alt_src}")
            return 2

    # Try symlink
    try:
        os.symlink(src_path, dst)
        print(f"已创建符号链接： {dst} -> {src_path}")
        return 0
    except Exception as e:
        print(f"创建符号链接失败：{e}")

    # Try hard link
    try:
        os.link(src_path, dst)
        print(f"已创建硬链接： {dst} -> {src_path}")
        return 0
    except Exception as e:
        print(f"创建硬链接失败：{e}")

    # Fallback to copy
    try:
        shutil.copy2(src_path, dst)
        print(f"已复制文件： {dst}")
        return 0
    except Exception as e:
        print(f"复制失败：{e}")
        return 3

if __name__ == '__main__':
    code = main()
    sys.exit(code)
