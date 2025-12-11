#!/usr/bin/env python3
"""
Check if weight files exist and are loadable
"""
import os
import torch

BASE = "/home/dieguez/qwen3_perl/torchin/torchin/qwen_data"

files = [
    "model.embed_tokens.weight.bin",
    "model.norm.weight.bin", 
    "layer_0_packed.bin",
    "layer_1_packed.bin",
]

print("Checking weight files...")
for f in files:
    path = os.path.join(BASE, f)
    if os.path.exists(path):
        try:
            t = torch.load(path, map_location='cpu', weights_only=True)
            print(f"✓ {f}: shape={t.shape}, dtype={t.dtype}, numel={t.numel()}")
        except Exception as e:
            print(f"✗ {f}: EXISTS but FAILED to load: {e}")
    else:
        print(f"✗ {f}: FILE NOT FOUND")

# Also check prompt
prompt_path = "/home/dieguez/qwen3_perl/torchin/torchin/prompt.bin"
if os.path.exists(prompt_path):
    t = torch.load(prompt_path, map_location='cpu', weights_only=True)
    print(f"✓ prompt.bin: shape={t.shape}, dtype={t.dtype}, values={t.tolist()}")
else:
    print(f"✗ prompt.bin: FILE NOT FOUND")

