#!/usr/bin/env python3
"""
convert_hf_to_pt.py - UNIVERSAL CPU COMPATIBILITY VERSION
Saves all tensors to CPU to ensure C++ LibTorch can read them regardless of GPU setup.
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM

def export_packed_weights(model, output_dir):
    print(f"Exporting Weights to {output_dir} (CPU Enforced)...")
    
    # Helper to save safely
    def save_tensor(tensor, name):
        path = os.path.join(output_dir, name)
        # 1. Force CPU
        # 2. Detach from graph
        # 3. Use standard Zip serialization (most compatible if on CPU)
        clean_tensor = tensor.detach().cpu().to(torch.float32)
        torch.save(clean_tensor, path)
        print(f"  Saved {name} {clean_tensor.shape}")

    # 1. Embeddings
    save_tensor(model.model.embed_tokens.weight, "model.embed_tokens.weight.pt")
    save_tensor(model.model.norm.weight, "model.norm.weight.pt")

    # 2. Layers (PACKED)
    for i, layer in enumerate(model.model.layers):
        tensors = []
        
        # Helper to get CPU tensor
        def get(t): return t.detach().cpu().to(torch.float32).flatten()
        
        # Order: Norms -> Attn -> MLP
        tensors.append(get(layer.input_layernorm.weight))
        tensors.append(get(layer.post_attention_layernorm.weight))
        
        tensors.append(get(layer.self_attn.q_proj.weight))
        tensors.append(get(layer.self_attn.k_proj.weight))
        tensors.append(get(layer.self_attn.v_proj.weight))
        tensors.append(get(layer.self_attn.o_proj.weight))
        
        if hasattr(layer.self_attn, 'q_norm'):
             tensors.append(get(layer.self_attn.q_norm.weight))
             tensors.append(get(layer.self_attn.k_norm.weight))
        else:
             # Zeros must be on CPU too
             hidden = model.config.hidden_size
             kv_heads = model.config.num_key_value_heads
             head_dim = hidden // model.config.num_attention_heads
             tensors.append(torch.zeros(hidden, dtype=torch.float32))
             tensors.append(torch.zeros(head_dim * kv_heads, dtype=torch.float32))

        tensors.append(get(layer.mlp.gate_proj.weight))
        tensors.append(get(layer.mlp.up_proj.weight))
        tensors.append(get(layer.mlp.down_proj.weight))
        
        # Concat on CPU
        packed = torch.cat(tensors)
        save_tensor(packed, f"layer_{i}_packed.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="./qwen_data")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Loading {args.model}...")
    # Load to CPU initially to be safe, or map_location="auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
    export_packed_weights(model, args.output)

if __name__ == "__main__":
    main()
