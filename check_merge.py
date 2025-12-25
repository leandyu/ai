from safetensors import safe_open
import torch
import os

model_dir = "../merged-model"

for root, _, files in os.walk(model_dir):
    for f in files:
        path = os.path.join(root, f)
        if f.endswith(".safetensors"):
            print("检查:", path)
            with safe_open(path, framework="pt", device="cpu") as fts:
                for k in fts.keys():
                    v = fts.get_tensor(k)
                    if torch.isnan(v).any():
                        print("⚠️ NaN in", path, k)

