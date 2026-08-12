from attention import GQAttention
from config import LFM2Config
from modules import GatedFeedForward, LFM2ConvBlock, RMSNorm, rope_rotate
from torch import nn
import torch
import torch.nn.functional as F
from utils import model_memory_size

class LFM2Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = GQAttention(
                idim=config.hidden_size,
                n_heads=config.num_attention_heads,
                num_groups=config.num_key_value_heads,
                head_dim=config.head_dim,
                rope_dim=int(config.head_dim * config.partial_rotary_factor),
                norm_eps=config.norm_eps,
                dtype=config.dtype
            )
        else:
            self.conv = LFM2ConvBlock(
                idim=config.hidden_size,
                hidden_dim=config.hidden_size,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
                dtype=config.dtype,
            )

        self.feed_forward = GatedFeedForward(
            idim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            dtype=config.dtype
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=config.dtype)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=config.dtype)

    def forward(self, x, cos, sin, mask=None, cache=None, use_cache=False, position_offset=0):
        residual = x
        x = self.operator_norm(x)

        if self.is_attention_layer:
            mixer_out = self.self_attn(
                x,
                cos,
                sin,
                mask=mask,
                past_key_value=cache,
                use_cache=use_cache,
                position_offset=position_offset,
            )
        else:
            mixer_out = self.conv(x, cache_state=cache, use_cache=use_cache)

        new_cache = None
        if use_cache:
            mixer_out, new_cache = mixer_out

        x = residual + mixer_out
        x = x + self.feed_forward(self.ffn_norm(x))
        if use_cache:
            return x, new_cache
        return x


class LFM2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = getattr(config, 'vocab_size', 32000)
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', None),
            dtype=config.dtype,
        )

        self.layers = nn.ModuleList([
            LFM2Block(config, i) for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=config.dtype)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False, dtype=config.dtype)

        # Initialize RoPE tables
        self.cos, self.sin = rope_rotate(
            head_dim=int(config.head_dim * config.partial_rotary_factor),
            context_length=config.max_position_embeddings,
            theta=config.rope_theta
        )

        # Initialize weights and tie embeddings if requested
        self.apply(self._init_weights)
        if getattr(config, 'tie_word_embeddings', False):
            self.tie_weights()

        if getattr(config, 'pad_token_id', None) is not None:
            with torch.no_grad():
                self.embed_tokens.weight[config.pad_token_id].zero_()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=getattr(self.config, 'initializer_range', 0.02))
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.embed_tokens = embeddings
        if getattr(self.config, 'tie_word_embeddings', False):
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, embeddings: nn.Linear) -> None:
        self.lm_head = embeddings

    def _past_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        for layer_cache in past_key_values:
            if isinstance(layer_cache, tuple):
                return layer_cache[0].shape[2]
        return 0

    def _causal_mask(self, query_length: int, key_length: int, past_length: int, device):
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=device,
        )
        key_positions = torch.arange(key_length, device=device)
        return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

    def forward(self, input_ids, mask=None, past_key_values=None, use_cache=False):
        b, L = input_ids.shape
        past_length = self._past_length(past_key_values)
        total_length = past_length + L
        if total_length > self.cos.shape[2]:
            raise ValueError(
                f"Sequence length {total_length} exceeds RoPE table length {self.cos.shape[2]}"
            )
        if mask is None:
            mask = self._causal_mask(L, total_length, past_length, input_ids.device)

        hidden_states = self.embed_tokens(input_ids)
        cos = self.cos.to(hidden_states.device)
        sin = self.sin.to(hidden_states.device)

        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past_key_values = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = past_key_values[layer_idx] if past_key_values is not None else None
            if use_cache:
                hidden_states, new_layer_cache = layer(
                    hidden_states,
                    cos,
                    sin,
                    mask=mask,
                    cache=layer_cache,
                    use_cache=True,
                    position_offset=past_length,
                )
                new_past_key_values.append(new_layer_cache)
            else:
                hidden_states = layer(hidden_states, cos, sin, mask=mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, new_past_key_values
        return logits


def test():
    # Update config with missing parameters
    config = LFM2Config()
    # Instantiate the combined model
    model = LFM2ForCausalLM(config)
    print(f"LFM2 Model initialized. Tied Embeddings: {config.tie_word_embeddings}")
    print(f"Approximate memory size: {model_memory_size(model):.2f} GB")

    device = torch.device("cpu")
    # Ensure model is in the correct dtype from config
    model.to(device=device, dtype=config.dtype)

    # 1. Test forward pass
    test_input = torch.tensor([1, 2, 3]).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(test_input)
    print("Model output shape : ", out.shape)

    # 2. Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters (including tied): {total_params:,}")

    # Account for weight tying (lm_head.weight is the same as embed_tokens.weight)
    unique_params = total_params - model.embed_tokens.weight.numel()
    print(f"Total number of unique parameters: {unique_params:,}")

    # 3. Memory estimation
    print("\nMemory size estimates:")
    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"configured dtype: {model_memory_size(model, input_dtype=config.dtype):.2f} GB")
