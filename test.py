from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
tokens = torch.tensor([[9707, 11, 1246, 525, 498, 30]])

context_flat = []
def hook(module, input, output):
    # Capture the context before o_proj
    pass

# Manually compute context_flat
with torch.no_grad():
    hidden = model.model.embed_tokens(tokens)
    hidden = model.model.layers[0].input_layernorm(hidden)
    
    q = model.model.layers[0].self_attn.q_proj(hidden)
    k = model.model.layers[0].self_attn.k_proj(hidden)
    v = model.model.layers[0].self_attn.v_proj(hidden)
    
    # Reshape
    q = q.view(1, 6, 16, 128)
    k = k.view(1, 6, 8, 128)
    v = v.view(1, 6, 8, 128)
    
    # QK norm
    q = model.model.layers[0].self_attn.q_norm(q)
    k = model.model.layers[0].self_attn.k_norm(k)
    
    # Transpose for attention
    q = q.transpose(1, 2)  # [1, 16, 6, 128]
    k = k.transpose(1, 2)  # [1, 8, 6, 128]
    v = v.transpose(1, 2)  # [1, 8, 6, 128]
    
    # GQA repeat
    k = k.repeat_interleave(2, dim=1)  # [1, 16, 6, 128]
    v = v.repeat_interleave(2, dim=1)  # [1, 16, 6, 128]
    
    # Attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)
    
    # Causal mask
    mask = torch.triu(torch.full((6, 6), float('-inf')), diagonal=1)
    scores = scores + mask
    
    probs = torch.softmax(scores, dim=-1)
    
    # Context
    context = torch.matmul(probs, v)  # [1, 16, 6, 128]
    
    # Reshape for output projection
    context_t = context.transpose(1, 2)  # [1, 6, 16, 128]
    context_flat = context_t.reshape(1, 6, 2048)
    
    print("Context_Flat[0,0,:5]:", context_flat[0, 0, :5].tolist())
    print("Context_Flat[0,0,128:133] (head1):", context_flat[0, 0, 128:133].tolist())
    print("Context_Flat[0,0,256:261] (head2):", context_flat[0, 0, 256:261].tolist())
    
    # Final output
    output = model.model.layers[0].self_attn.o_proj(context_flat)
    print("Output[0,0,:5]:", output[0, 0, :5].tolist())
