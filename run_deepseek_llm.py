#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

# -----------------------------
# 配置
# -----------------------------
# 原始 deepseek tokenizer
BASE_MODEL = "deepseek-r1:7b"

# LoRA 合并后的模型路径
MERGED_MODEL_PATH = "merged-model"

# 向量数据库路径（可选）
VECTOR_DB_DIR = "./vector_db"

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
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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
llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# 初始化向量数据库（可选）
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

# -----------------------------
# 查询函数
# -----------------------------
def query_model(prompt: str, use_vectordb: bool = False):
    if use_vectordb:
        docs = vectordb.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    else:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    while True:
        prompt = input("\n请输入问题（exit退出）: ").strip()
        if prompt.lower() == "exit":
            break
        answer = query_model(prompt)
        print(f"\n💡 回答:\n{answer}")
