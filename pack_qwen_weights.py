#!/usr/bin/env python3
"""
pack_qwen_weights.py - Pack Qwen3-0.6B weights for RefTorch

Binary format (must match TLoadBinary in TensorIO.ref):
  - int64_t ndim          (8 bytes, little-endian)
  - int64_t shape[ndim]   (8 bytes each, little-endian)  
  - float32 data[numel]   (4 bytes each, native order)

Usage:
    python pack_qwen_weights.py --output-dir ./qwen_data
"""

import torch
import numpy as np
import struct
import os
import argparse
from pathlib import Path

def save_tensor_binary(tensor: torch.Tensor, path: str):
    """Save tensor in RefTorch binary format."""
    # Ensure contiguous float32
    t = tensor.contiguous().float()
    
    with open(path, 'wb') as f:
        # Write number of dimensions (int64_t = 8 bytes)
        ndim = len(t.shape)
        f.write(struct.pack('<q', ndim))  # '<q' = little-endian int64
        
        # Write each dimension (int64_t each)
        for dim in t.shape:
            f.write(struct.pack('<q', dim))
        
        # Write raw float32 data
        f.write(t.numpy().astype(np.float32).tobytes())
    
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
    
    # Print model info
    print(f"\nModel config:")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  intermediate_size: {model.config.intermediate_size}")
    print(f"  num_attention_heads: {model.config.num_attention_heads}")
    print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"  head_dim: {model.config.head_dim}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    
    # Save embeddings
    print("\nSaving embeddings (binary format)...")
    save_tensor_binary(state["model.embed_tokens.weight"], 
                       output_dir / "model.embed_tokens.weight.bin")
    print(f"  embed_tokens: {state['model.embed_tokens.weight'].shape}")
    
    save_tensor_binary(state["model.norm.weight"],
                       output_dir / "model.norm.weight.bin")
    print(f"  norm: {state['model.norm.weight'].shape}")
    
    # Pack each layer
    num_layers = model.config.num_hidden_layers
    print(f"\nPacking {num_layers} layers (binary format)...")
    
    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        
        # Order must match WeightLoader.ref!
        tensors = [
            state[f"{prefix}input_layernorm.weight"],           # [1024]
            state[f"{prefix}post_attention_layernorm.weight"],  # [1024]
            state[f"{prefix}self_attn.q_proj.weight"],          # [2048, 1024]
            state[f"{prefix}self_attn.k_proj.weight"],          # [1024, 1024]
            state[f"{prefix}self_attn.v_proj.weight"],          # [1024, 1024]
            state[f"{prefix}self_attn.o_proj.weight"],          # [1024, 2048]
            state[f"{prefix}self_attn.q_norm.weight"],          # [128]
            state[f"{prefix}self_attn.k_norm.weight"],          # [128]
            state[f"{prefix}mlp.gate_proj.weight"],             # [3072, 1024]
            state[f"{prefix}mlp.up_proj.weight"],               # [3072, 1024]
            state[f"{prefix}mlp.down_proj.weight"],             # [1024, 3072]
        ]
        
        # Flatten and concatenate into 1D tensor
        packed = torch.cat([t.flatten().float() for t in tensors])
        
        out_path = output_dir / f"layer_{i}_packed.bin"
        save_tensor_binary(packed, out_path)
        
        if i == 0:
            print(f"  Layer 0: {packed.numel()} elements, shape={packed.shape}")
            offset = 0
            names = ["input_ln", "post_ln", "q_proj", "k_proj", "v_proj", 
                     "o_proj", "q_norm", "k_norm", "gate", "up", "down"]
            for name, t in zip(names, tensors):
                print(f"    {name:10s}: {str(t.shape):20s} = {t.numel():10d} @ offset {offset}")
                offset += t.numel()
            print(f"    {'TOTAL':10s}: {' ':20s} = {offset:10d}")
        elif i % 7 == 0:
            print(f"  Layer {i}...")
    
    print(f"  ... done!")
    
    # Create prompt (save in output_dir)
    print(f"\nCreating prompt: {repr(args.prompt)}")
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")  # Shape: [1, seq_len]
    print(f"  Token IDs: {tokens.tolist()}")
    print(f"  Shape: {tokens.shape}")
    
    # Save prompt - TLoadBinary will load as float, then TToLong converts to int
    prompt_path = output_dir / "prompt.bin"
    save_tensor_binary(tokens.float(), prompt_path)
    print(f"  Saved to: {prompt_path}")
    
    # Verify by decoding
    decoded = tokenizer.decode(tokens[0])
    print(f"  Decoded: {repr(decoded)}")
    
    # Verify a file can be read back
    print("\nVerifying binary format...")
    test_path = output_dir / "model.norm.weight.bin"
    with open(test_path, 'rb') as f:
        ndim = struct.unpack('<q', f.read(8))[0]
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(ndim)]
        numel = 1
        for s in shape:
            numel *= s
        data = np.frombuffer(f.read(numel * 4), dtype=np.float32)
        print(f"  norm.weight: ndim={ndim}, shape={shape}, numel={numel}, data[:5]={data[:5]}")
    
    print("\n" + "="*60)
    print("Done! Binary files created:")
    print(f"  {output_dir}/model.embed_tokens.weight.bin")
    print(f"  {output_dir}/model.norm.weight.bin")
    print(f"  {output_dir}/layer_{{0..{num_layers-1}}}_packed.bin")
    print(f"  {output_dir}/prompt.bin")
    print("="*60)

if __name__ == "__main__":
    main()

