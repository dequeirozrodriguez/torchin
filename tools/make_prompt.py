# make_prompt.py
import torch
from transformers import AutoTokenizer
import sys

text = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
model_id = "Qwen/Qwen3-0.6B"

print(f"Encoding: '{text}'")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Encode and FORCE CPU
tokens = tokenizer.encode(text, return_tensors="pt").cpu().to(torch.long)

torch.save(tokens, "prompt.pt")
print(f"Saved prompt.pt (CPU) with shape {tokens.shape}")
