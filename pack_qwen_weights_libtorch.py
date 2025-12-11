#!/usr/bin/env python3
"""
pack_qwen_weights_libtorch.py - Pack Qwen3-0.6B weights for LibTorch/RefTorch

IMPORTANT: This saves tensors in a format compatible with torch::load() in C++.
Regular torch.save() uses pickle which LibTorch CANNOT read!

Usage:
    python pack_qwen_weights_libtorch.py --output-dir ./qwen_data
"""

import torch
import os
import argparse
from pathlib import Path

def save_tensor_for_libtorch(tensor: torch.Tensor, path: str):
    """Save tensor in a format LibTorch can read.
    
    Option 1: Use torch.jit.save with a simple module (recommended)
    Option 2: Save raw binary data
    
    We'll use raw binary for simplicity.
    """
    # Ensure contiguous float32
    t = tensor.contiguous().float()
    
    # Save as raw binary: shape + data
    with open(path, 'wb') as f:
        # Write number of dimensions
        ndim = len(t.shape)
        f.write(ndim.to_bytes(4, 'little'))
        
        # Write each dimension
        for dim in t.shape:
            f.write(dim.to_bytes(8, 'little'))
        
        # Write raw float32 data
        f.write(t.numpy().tobytes())
    
    return t.numel()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./qwen_data", help="Output directory")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt text")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Qwen3-0.6B model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    state = model.state_dict()
    
    # Save embeddings
    print("\nSaving embeddings (raw binary format)...")
    save_tensor_for_libtorch(
        state["model.embed_tokens.weight"], 
        output_dir / "model.embed_tokens.weight.bin"
    )
    print(f"  embed_tokens: {state['model.embed_tokens.weight'].shape}")
    
    save_tensor_for_libtorch(
        state["model.norm.weight"],
        output_dir / "model.norm.weight.bin"  
    )
    print(f"  norm: {state['model.norm.weight'].shape}")
    
    # Pack each layer
    num_layers = model.config.num_hidden_layers
    print(f"\nPacking {num_layers} layers (raw binary format)...")
    
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        
        # Order must match WeightLoader.ref!
        tensors = [
            state[f"{prefix}input_layernorm.weight"],
            state[f"{prefix}post_attention_layernorm.weight"],
            state[f"{prefix}self_attn.q_proj.weight"],
            state[f"{prefix}self_attn.k_proj.weight"],
            state[f"{prefix}self_attn.v_proj.weight"],
            state[f"{prefix}self_attn.o_proj.weight"],
            state[f"{prefix}self_attn.q_norm.weight"],
            state[f"{prefix}self_attn.k_norm.weight"],
            state[f"{prefix}mlp.gate_proj.weight"],
            state[f"{prefix}mlp.up_proj.weight"],
            state[f"{prefix}mlp.down_proj.weight"],
        ]
        
        # Flatten and concatenate
        packed = torch.cat([t.flatten().float() for t in tensors])
        
        out_path = output_dir / f"layer_{i}_packed.bin"
        save_tensor_for_libtorch(packed, out_path)
        
        if i == 0:
            print(f"  Layer 0: {packed.numel()} elements")
        elif i % 7 == 0:
            print(f"  Layer {i}...")
    
    print(f"  ... done!")
    
    # Create prompt
    print(f"\nCreating prompt: {repr(args.prompt)}")
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"  Token IDs: {tokens.tolist()}")
    print(f"  Shape: {tokens.shape}")
    
    # Save prompt as Long tensor
    prompt_path = output_dir.parent / "prompt.bin"
    save_tensor_for_libtorch(tokens.long().float(), prompt_path)  # Save as float, convert to long in Refal
    print(f"  Saved to: {prompt_path}")
    
    print("\n" + "="*60)
    print("Done! Binary format files created.")
    print("="*60)

if __name__ == "__main__":
    main()

