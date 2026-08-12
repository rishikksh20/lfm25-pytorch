import torch

class LFM2Config:
    def __init__(self):
        # Model Architecture
        self.architectures = ["Lfm2ForCausalLM"]
        self.model_type = "lfm2"
        self.hidden_size = 2048
        self.num_hidden_layers = 30
        self.num_attention_heads = 32
        self.num_heads = 32
        self.num_key_value_heads = 8
        self.head_dim = self.hidden_size // self.num_attention_heads # Derived: 64
        self.intermediate_size = 10752
        self.vocab_size = 128000

        # Normalization and Initialization
        self.norm_eps = 1e-05
        self.initializer_range = 0.02
        self.block_norm_eps = 1e-05
        self.block_use_xavier_init = True
        self.conv_use_xavier_init = True

        # Block and Conv Specifics
        self.block_dim = 2048
        self.block_auto_adjust_ff_dim = False
        self.block_ffn_dim_multiplier = 1.0
        self.block_mlp_init_scale = 1.0
        self.block_multiple_of = 256
        self.block_out_init_scale = 1.0
        self.block_use_swiglu = True
        self.conv_L_cache = 3
        self.conv_bias = False
        self.conv_dim = 2048
        self.kernel_size = 3 # inferred from conv_L_cache logic
        self.dropout = 0.0

        # Positional and Token IDs
        self.max_position_embeddings = 131072
        self.bos_token_id = 124894
        self.eos_token_id = 124900
        self.pad_token_id = 124893
        self.tie_word_embeddings = True
        self.tie_embedding = True

        # RoPE
        self.rope_theta = 10000000.0
        self.theta = self.rope_theta
        self.partial_rotary_factor = 1.0 # default to full head rotation unless specified

        # Runtime
        self.dtype = torch.bfloat16
        self.use_cache = True
        self.use_pos_enc = True

        self.layer_types = [
            "conv", "conv", "full_attention", "conv", "conv", "full_attention",
            "conv", "conv", "conv", "full_attention", "conv", "conv", "conv",
            "full_attention", "conv", "conv", "conv", "full_attention", "conv",
            "conv", "conv", "full_attention", "conv", "conv", "full_attention",
            "conv", "conv", "full_attention", "conv", "conv"
        ]
