整体架构：
1. 准备工作
   a. 解析剧本
   b. 剧本分片嵌入向量数据库   
2. 使用方式：
   用户提出问题  --> 向量化（用户问题） --> 相似性查询向量数据库（本地） --> 将向量数据库返回结果和提示词查询大模型型（微调后大模型） --> 将结果返回
3. 代码涉及文本解析、分割，向量化数据库、模型微调、模型量化   


完整工作流程：

1. 准备数据：将剧本转换为训练格式
python3 scripts/prepare_data_1.py --raw_dir data/raw --clean_dir data/cleaned --out_jsonl data/train_jsonl --mode all

2. 存入向量数据库：
python3 scripts/build_vector_db_2.py  
 
3. 查询向量数据库：
python3 scripts/query_vector_db_3.py

4. 微调模型：使用QLoRA微调DeepSeek R1 7B
python3 scripts/finetune_lora_4.py

5. 合并模型：将步骤4生成的微调数据与基础模型合并
python3 scripts/merge_model_5.py

6. 转换模型：将合并的模型转换为gguf（ollama可读)
python3 scripts/convert_6.py
   量化压缩 - 生成量化版压缩模型（可选）
   ./llama.cpp/build/bin/llama-quantize ../gguf-output/deepseek-lora.gguf ../gguf-output/deepseek-lora-q4_k_m.gguf Q4_K_M

7. 部署模型：生成微信后的模型，并将微调后的模型加载到Ollama
ollama create deepseek-lora -f scripts/Modelfile
ollama run deepseek-lora

8. 使用模型：
   python3 scripts/rag_chat_7.py

# web访问
streamlit run scripts/streamlitDemo2.py

# 准备工作 - 启动本地向量数据库
python3 -m venv rag-env # 创建虚拟环境
source venv/bin/activate
(venv)  % ./venv/bin/chroma run --path ./vector_db

