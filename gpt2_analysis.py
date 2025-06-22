#!/usr/bin/env python3
"""
GPT-2 Model Parameter Count and FLOPs Analysis
"""

def calculate_parameters(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """Calculate model parameter count"""
    
    # 1. Token Embeddings
    token_embeddings = vocab_size * d_model
    
    # 2. Parameters for each Transformer layer
    # RMSNorm parameters (2 per layer)
    rmsnorm_params = 2 * d_model * num_layers
    
    # MultiHeadAttention parameters
    # q_proj, k_proj, v_proj, output_proj: each is d_model x d_model
    attention_params = 4 * d_model * d_model * num_layers
    
    # SwiGLU parameters (w1, w2, w3)
    swiglu_params = (d_model * d_ff + d_ff * d_model + d_model * d_ff) * num_layers
    
    # 3. Final layers
    final_rmsnorm = d_model
    lm_head = d_model * vocab_size
    
    # 4. RoPE parameters (only cache, not trainable parameters)
    rope_params = 0
    
    total_params = (token_embeddings + rmsnorm_params + attention_params + 
                   swiglu_params + final_rmsnorm + lm_head)

    return total_params

def calculate_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """Calculate forward pass FLOPs"""
    
    seq_len = context_length
    head_dim = d_model // num_heads
    
    total_flops = 0
    component_flops = {}
    
    # 1. Token Embeddings (lookup operation, no FLOPs)
    embedding_flops = 0
    component_flops['Token Embeddings'] = embedding_flops
    
    # 2. Each Transformer layer
    layer_flops = 0
    
    # RMSNorm (2 per layer)
    # Each RMSNorm: square + addition + 1 sqrt + division + multiplication = (seq_len * d_model * 4 + 1) FLOPs
    # Each TransformerBlock has 2 RMSNorm layers
    rmsnorm_flops = 2 * (seq_len * d_model * 4 + 1) * num_layers
    
    # MultiHeadAttention
    # Q, K, V projections: 3 * seq_len * d_model * d_model * 2 (each matrix multiplication needs 2x FLOPs)
    qkv_proj_flops = 3 * seq_len * d_model * d_model * 2 * num_layers
    
    # QK attention calculation: 
    # 1. QK computation: seq_len * seq_len * head_dim * 2 (matrix multiplication)
    # 2. Attention weights with V: seq_len * head_dim * seq_len * 2 (matrix multiplication)
    qk_attention_flops = (seq_len * seq_len * head_dim * 2 + seq_len * head_dim * seq_len * 2) * num_heads * num_layers
    
    # Output projection: seq_len * d_model * d_model * 2
    output_proj_flops = seq_len * d_model * d_model * 2 * num_layers
    
    attention_flops = qkv_proj_flops + qk_attention_flops + output_proj_flops
    
    # SwiGLU (1 per layer)
    # w1: seq_len * d_model * d_ff * 2
    # w3: seq_len * d_model * d_ff * 2
    # w2: seq_len * d_ff * d_model * 2
    swiglu_flops = (seq_len * d_model * d_ff * 2 * 2 + seq_len * d_ff * d_model * 2) * num_layers
    
    layer_flops = rmsnorm_flops + attention_flops + swiglu_flops
    
    # 3. Final layers
    final_rmsnorm_flops = seq_len * d_model * 4 + 1
    lm_head_flops = seq_len * d_model * vocab_size * 2
    
    total_flops = layer_flops + final_rmsnorm_flops + lm_head_flops
    
    component_flops['RMSNorm (all layers)'] = rmsnorm_flops
    component_flops['MultiHeadAttention (all layers)'] = attention_flops
    component_flops['SwiGLU (all layers)'] = swiglu_flops
    component_flops['Final RMSNorm'] = final_rmsnorm_flops
    component_flops['LM Head'] = lm_head_flops
    
    return total_flops, component_flops

def analyze_gpt2_models():
    """Analyze parameters and FLOPs for different GPT-2 models"""
    
    models = {
        'GPT-2 Small': {
            'vocab_size': 50257,
            'context_length': 1024,
            'num_layers': 12,
            'd_model': 768,
            'num_heads': 12,
            'd_ff': 3072
        },
        'GPT-2 Medium': {
            'vocab_size': 50257,
            'context_length': 1024,
            'num_layers': 24,
            'd_model': 1024,
            'num_heads': 16,
            'd_ff': 4096
        },
        'GPT-2 Large': {
            'vocab_size': 50257,
            'context_length': 1024,
            'num_layers': 36,
            'd_model': 1280,
            'num_heads': 20,
            'd_ff': 5120
        },
        'GPT-2 XL': {
            'vocab_size': 50257,
            'context_length': 1024,
            'num_layers': 48,
            'd_model': 1600,
            'num_heads': 25,
            'd_ff': 6400
        }
    }
    
    print("=== GPT-2 Model Analysis ===\n")
    
    for model_name, config in models.items():
        print(f"--- {model_name} ---")
        params = calculate_parameters(**config)
        flops, components = calculate_flops(**config)
        
        print(f"Parameter count: {params:,}")
        print(f"Memory requirement (FP32): {params * 4 / (1024**3):.2f} GB")
        print(f"Total FLOPs: {flops:,}")
        print("Component FLOPs distribution:")
        for component, component_flops in components.items():
            percentage = (component_flops / flops) * 100
            print(f"  {component}: {percentage:.1f}%")
        print()

def analyze_context_length_impact():
    """Analyze the impact of context length on FLOPs"""
    
    # GPT-2 XL configuration
    config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'num_layers': 48,
        'd_model': 1600,
        'num_heads': 25,
        'd_ff': 6400
    }
    
    print("=== Context Length Impact Analysis ===\n")
    
    # Original context length
    flops_1024, components_1024 = calculate_flops(**config)
    
    # Increase context length to 16384
    config['context_length'] = 16384
    flops_16384, components_16384 = calculate_flops(**config)
    
    print(f"Context length 1024 -> 16384")
    print(f"FLOPs change: {flops_1024:,} -> {flops_16384:,}")
    print(f"Increase factor: {flops_16384 / flops_1024:.1f}x")
    print()
    
    print("Component FLOPs ratio change:")
    for component in components_1024.keys():
        ratio_1024 = components_1024[component] / flops_1024
        ratio_16384 = components_16384[component] / flops_16384
        print(f"  {component}: {ratio_1024*100:.1f}% -> {ratio_16384*100:.1f}%")

if __name__ == "__main__":
    # (a) GPT-2 XL parameter analysis
    print("(a) GPT-2 XL Parameter Analysis:")
    gpt2_xl_config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'num_layers': 48,
        'd_model': 1600,
        'num_heads': 25,
        'd_ff': 6400
    }
    
    params = calculate_parameters(**gpt2_xl_config)
    memory_gb = params * 4 / (1024**3)  # FP32 = 4 bytes
    print(f"GPT-2 XL has {params:,} trainable parameters and requires {memory_gb:.2f}GB memory to load the model.\n")
    
    # (b) FLOPs analysis
    print("(b) GPT-2 XL FLOPs Analysis:")
    flops, components = calculate_flops(**gpt2_xl_config)
    print(f"Total FLOPs: {flops:,}")
    print("Matrix multiplication components:")
    for component, component_flops in components.items():
        print(f"  {component}: {component_flops:,} FLOPs")
    print()
    
    # (c) Most FLOPs-intensive component
    print("(c) Most FLOPs-intensive component:")
    max_component = max(components.items(), key=lambda x: x[1])
    print(f"{max_component[0]} requires the most FLOPs, accounting for {(max_component[1]/flops)*100:.1f}% of total computation.\n")
    
    # (d) Comparison of different model sizes
    print("(d) Comparison of different GPT-2 models:")
    analyze_gpt2_models()
    
    # (e) Context length impact
    print("(e) Context length impact:")
    analyze_context_length_impact() 