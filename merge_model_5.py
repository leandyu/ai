from peft import PeftModel
from transformers import AutoModelForCausalLM

#base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
base_model = AutoModelForCausalLM.from_pretrained("deepSeek-R1-Distill-Qwen-7B")
model = PeftModel.from_pretrained(base_model, "lora-output")
model = model.merge_and_unload()
model.save_pretrained("merged-model")

"""
*** step 6  转换成GGUF
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

/Users/jz/source/python/rag-project/scripts/llama.cpp/build/bin./

python3 convert.py ./merged-model --outtype q4_0


*** step 7 创建ollama自定义模型
在项目目录创建 Modelfile
FROM ./ggml-model-q4_0.gguf
TEMPLATE \"""
{{ .Prompt }}
\"""

ollama create deepseek-custom -f Modelfile


*** step 8 测试 
ollama run deepseek-custom

curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-custom",
  "prompt": "你好，介绍一下自己"
}'

"""
