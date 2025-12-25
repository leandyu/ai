import os
import subprocess

# -----------------------------
# 配置
# -----------------------------
MERGED_MODEL_DIR = "merged-model"  # 合并后的 HuggingFace 模型
GGUF_DIR = "gguf-output"           # GGUF 输出目录
GGUF_FILENAME = "deepseek-lora.gguf"
LLAMA_CPP_PATH = os.getcwd()  # 使用当前目录（假设脚本在 llama.cpp 目录中运行）
os.makedirs(GGUF_DIR, exist_ok=True)

gguf_path = os.path.join(GGUF_DIR, GGUF_FILENAME)

# -----------------------------
# 运行 Ollama 转换命令
# -----------------------------
print("✅ 开始转换为 GGUF...")

# 使用 llama.cpp 的 convert_hf_to_gguf.py 脚本
convert_script = os.path.join(LLAMA_CPP_PATH, "scripts/llama.cpp/convert_hf_to_gguf.py")

try:
    # 运行转换脚本
    subprocess.run([
        "python3", convert_script,
        MERGED_MODEL_DIR, # 指定 tokenizer 目录
        "--outtype", "f16",
        "--outfile", gguf_path
    ], check=True)

    print(f"✅ 转换完成，GGUF 模型路径: {gguf_path}")

except subprocess.CalledProcessError as e:
    print(f"❌ 转换失败: {e}")
except FileNotFoundError:
    print("❌ 找不到 convert_lora_to_gguf.py 脚本")
except Exception as e:
    print(f"❌ 发生错误: {e}")
