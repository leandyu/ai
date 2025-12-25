#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# 配置
# -----------------------------
# 本地 tokenizer 路径（原始 deepseek-r1:7b）
TOKENIZER_PATH = "deepseek-r1-7b-tokenizer"

# LoRA 合并后的模型路径
MERGED_MODEL_PATH = "merged-model"

# 生成文本最大长度
MAX_NEW_TOKENS = 128

# -----------------------------
# 设备选择
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# -----------------------------
# 加载 tokenizer 和模型
# -----------------------------
print("✅ 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print("✅ tokenizer 加载完成")

print("✅ 加载合并后的 LoRA 模型...")
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    device_map=device,
    torch_dtype=torch.float16
)
print("✅ 模型加载完成")

# -----------------------------
# 封装 pipeline
# -----------------------------
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=device
)

# -----------------------------
# 查询函数
# -----------------------------
def query_model(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    print("\n✅ 模型准备就绪，可以输入问题")
    while True:
        prompt = input("\n请输入问题（exit退出）: ").strip()
        if prompt.lower() == "exit":
            break
        answer = query_model(prompt)
        print(f"\n💡 回答:\n{answer}")
