#!/usr/bin/env python3
"""
convert_hf_to_pt.py - Convert Qwen3 weights to RefTorch format (Fixed)

Updates:
  - Exports .bias.pt files for Attention layers (Qwen has QKV bias!)
  - Exports q_layernorm and k_layernorm weights (QK-Norm)
  - Exports model_config.ref automatically
"""

import argparse
import json
import os
import sys
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_refal_config(config, output_dir):
    """Generates the QwenConfig.ref file with model hyperparameters."""
    hidden = config.hidden_size
    model_tag = "Custom"
    if hidden == 1024: model_tag = "0_6B"
    elif hidden == 2048: model_tag = "1_5B" # Approx
    
	rope_theta = getattr(config, "rope_theta", 1000000.0)
	
    tie_weights = 1 if config.tie_word_embeddings else 0
    
    inter_size = config.intermediate_size
    
    content = f"""* QwenConfig.ref - Auto-generated from {config._name_or_path}

$ENTRY Qwen3Config_{model_tag} {{
  = (('vocab_size' {config.vocab_size})
     ('hidden_size' {config.hidden_size})
     ('intermediate_size' {inter_size})
     ('num_hidden_layers' {config.num_hidden_layers})
     ('num_attention_heads' {config.num_attention_heads})
     ('num_key_value_heads' {config.num_key_value_heads})
     ('head_dim' {config.hidden_size // config.num_attention_heads})
     ('max_position_embeddings' {config.max_position_embeddings})
     ('rms_norm_eps' {int(config.rms_norm_eps * 1000000)}) /* Scaled x1M */
     ('rope_theta' {int(rope_theta)}) 
     ('tie_word_embeddings' {tie_weights})
     ('eos_token_id' {config.eos_token_id}));
}}
"""
    with open(os.path.join(output_dir, "QwenConfig.ref"), "w") as f:
        f.write(content)
    print(f"Generated QwenConfig.ref")

def export_weights(model, output_dir):
    weights_dir = os.path.join(output_dir, "weights")
    if os.path.exists(weights_dir):
        shutil.rmtree(weights_dir)
    os.makedirs(weights_dir)
    
    print(f"Exporting weights to {weights_dir}...")
    
    for name, param in model.named_parameters():
        if "rotary_emb" in name:
            continue
            
        tensor = param.detach().cpu().float()
        
        clean_name = name.replace("model.", "")
        
        if clean_name.startswith("layers."):
            # Reformat layers.0.x -> layer_0_x
            parts = clean_name.split(".")
            layer_idx = parts[1]
            module_parts = parts[2:]
            
            is_bias = False
            if module_parts[-1] == "weight":
                module_name = ".".join(module_parts[:-1])
                ext = ".weight.pt"
            elif module_parts[-1] == "bias":
                is_bias = True
                module_name = ".".join(module_parts[:-1])
                ext = ".bias.pt"
            else:
                module_name = ".".join(module_parts)
                ext = ".pt"
                
            filename = f"layer_{layer_idx}_{module_name}{ext}"
            
        else:
            filename = f"{clean_name.replace('.', '_')}.pt"
            if filename == "embed_tokens_weight.pt": filename = "model.embed_tokens.weight.pt"
            if filename == "norm_weight.pt": filename = "model.norm.weight.pt"
            if filename == "lm_head_weight.pt": filename = "lm_head.weight.pt"

        save_path = os.path.join(weights_dir, filename)
        torch.save(tensor, save_path)
        print(f"  Saved {filename} {tensor.shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="./qwen_data")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float32, 
        trust_remote_code=True
    )
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    export_weights(model, args.output)
    
    generate_refal_config(model.config, args.output)
    
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
    
    print("\nDone! Now run:")
    print(f"  cp {os.path.join(args.output, 'QwenConfig.ref')} src/models/QwenConfig.ref")

if __name__ == "__main__":
    main()
