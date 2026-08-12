import torch

@torch.no_grad()
def load_weights_into_lfm2(model, params):
    """
    Maps Hugging Face safetensor weights into the local LFM2ForCausalLM implementation.
    model: The LFM2ForCausalLM instance.
    params: A dictionary of {tensor_name: tensor_data}.
    """

    def assign(param, value, name):
        if param.shape != value.shape:
            raise ValueError(f"Shape mismatch for {name}: {param.shape} vs {value.shape}")
        param.copy_(value)

    def get_weight(*names):
        for name in names:
            if name in params:
                return params[name]
        raise KeyError(f"Missing weight. Tried: {', '.join(names)}")

    # 1. Base Embeddings
    if "model.embed_tokens.weight" in params:
        assign(model.embed_tokens.weight, params["model.embed_tokens.weight"], "embed_tokens")

    # 2. Iterate through layers
    for l in range(len(model.layers)):
        layer = model.layers[l]
        prefix = f"model.layers.{l}"

        # Operator Norm
        assign(layer.operator_norm.weight, params[f"{prefix}.operator_norm.weight"], f"{prefix}.operator_norm")

        # FFN Norm
        assign(layer.ffn_norm.weight, params[f"{prefix}.ffn_norm.weight"], f"{prefix}.ffn_norm")

        # Attention Layer Logic
        if layer.is_attention_layer:
            # GQAttention uses q_proj, k_proj, v_proj, o_proj and norm scales
            assign(layer.self_attn.q_proj.weight, get_weight(f"{prefix}.self_attn.q_proj.weight"), f"{prefix}.q_proj")

            assign(layer.self_attn.k_proj.weight, params[f"{prefix}.self_attn.k_proj.weight"], f"{prefix}.k_proj")
            assign(layer.self_attn.v_proj.weight, params[f"{prefix}.self_attn.v_proj.weight"], f"{prefix}.v_proj")
            assign(
                layer.self_attn.o_proj.weight,
                get_weight(f"{prefix}.self_attn.o_proj.weight", f"{prefix}.self_attn.out_proj.weight"),
                f"{prefix}.o_proj",
            )

            # Attention Norms
            assign(
                layer.self_attn.q_norm.weight,
                get_weight(f"{prefix}.self_attn.q_norm.weight", f"{prefix}.self_attn.q_layernorm.weight"),
                f"{prefix}.q_norm",
            )
            assign(
                layer.self_attn.k_norm.weight,
                get_weight(f"{prefix}.self_attn.k_norm.weight", f"{prefix}.self_attn.k_layernorm.weight"),
                f"{prefix}.k_norm",
            )

        # Conv Layer Logic
        else:
            # LFM2ConvBlock uses input_projection, conv, output_projection
            assign(layer.conv.input_projection.weight, params[f"{prefix}.conv.in_proj.weight"], f"{prefix}.conv_in")
            assign(layer.conv.conv.weight, params[f"{prefix}.conv.conv.weight"], f"{prefix}.conv_kernel")
            assign(layer.conv.output_projection.weight, params[f"{prefix}.conv.out_proj.weight"], f"{prefix}.conv_out")

        # Gated Feed Forward (SwiGLU)
        # Mapping HF w1 (gate), w2 (down), w3 (up)
        assign(layer.feed_forward.gate_proj.weight, params[f"{prefix}.feed_forward.w1.weight"], f"{prefix}.ffn_gate")
        assign(layer.feed_forward.down_proj.weight, params[f"{prefix}.feed_forward.w2.weight"], f"{prefix}.ffn_down")
        assign(layer.feed_forward.up_proj.weight, params[f"{prefix}.feed_forward.w3.weight"], f"{prefix}.ffn_up")

    # 3. Final Model Norm
    assign(model.norm.weight, get_weight("model.embedding_norm.weight", "model.norm.weight"), "final_norm")

    # 4. Output Head (handle tied weights)
    if "lm_head.weight" in params and not model.config.tie_word_embeddings:
        assign(model.lm_head.weight, params["lm_head.weight"], "lm_head")
    elif model.config.tie_word_embeddings:
        model.tie_weights()

    print("Weights loaded successfully.")

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)

    return total_memory_gb
