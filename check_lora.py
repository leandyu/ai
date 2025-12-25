from safetensors import safe_open
import torch

path = "../lora-output/checkpoint-72/adapter_model.safetensors"
with safe_open(path, framework="pt", device="cpu") as fts:
    for k in fts.keys():
        v = fts.get_tensor(k)
        if torch.isnan(v).any():
            print("⚠️ NaN in LoRA:", k)
