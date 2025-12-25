import ollama

OLLAMA_MODEL_NAME = "deepsek_lora_q4"
ollama.list(OLLAMA_MODEL_NAME)
lp = list(OLLAMA_MODEL_NAME.parameters())
print(len(lp))