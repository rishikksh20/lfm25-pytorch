from torch import nn
import torch
import torch.nn.functional as F
import math

class GatedFeedForward(nn.Module):
    def __init__(self, idim, hidden_dim, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.up_proj   = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.down_proj = nn.Linear(hidden_dim, idim,  dtype=dtype, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed, dtype=dtype))
        self.variance_epsilon = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        input_dtype = x.dtype
        x_norm = self._norm(x.float())
        x_norm = x_norm * self.weight.float()
        return x_norm.to(input_dtype)


class RMSNormGated(nn.Module):
    """
    RMSNorm followed by an element-wise SiLU gate.
    Matches Qwen3_5RMSNormGated from the HF implementation.
    forward(x, gate) → scale * rms_norm(x) * silu(gate)  (all cast to input dtype)
    """
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        return (x * F.silu(gate.to(torch.float32))).to(input_dtype)



def rope_rotate(head_dim, context_length, device='cpu', theta=1_000_000):
    """
    Precompute RoPE cos/sin tables.
    head_dim : dimension being rotated (pass rope_dim for partial RoPE)
    theta    : RoPE base frequency (1_000_000 for Qwen3; 10_000_000 for Qwen3.5)
    Returns cos, sin of shape (1, 1, context_length, head_dim).
    """

    half = head_dim // 2
    # Generate position indices
    positions = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.exp(-math.log(theta) * torch.arange(0, half, device=device) / half)  # (half,)
    angles = torch.einsum('l, h -> lh', positions.float(), freqs)                  # (L, half)

    combined_angles = torch.cat([angles, angles], dim=1)                         # (L, head_dim)

    cos = combined_angles.cos()[None, None, :, :]                                        # (1,1,L,head_dim)

    sin = combined_angles.sin()[None, None, :, :]                                          # (1,1,L,head_dim)

    return cos, sin


def apply_rope(x, cos, sin, position_offset=0):
    """
    x: (B, H, L, Dh) queries or keys
    positions: (L,) absolute positions (0..L-1)
    """
    B, H, L, Dh = x.shape
    half = Dh // 2


    x_upper = x[..., :half]
    x_lower = x[..., half:]

    x_bar = torch.cat((-x_lower, x_upper), dim=-1)              # (B,H,L,Dh)

    cos = cos[:, :, position_offset:position_offset + L, :]             # (1,1,L,Dh)
    sin = sin[:, :, position_offset:position_offset + L, :]             # (1,1,L,Dh)

    x_rot = (x * cos) + (x_bar * sin)
    return x_rot.to(x.dtype)


def test_apply_rope():
    x = torch.randn(2, 8, 512, 128)
    cos, sin = rope_rotate(128, 512)
    x_out = apply_rope(x, cos, sin)
    assert x_out.shape == (2, 8, 512, 128)



class LFM2ConvBlock(nn.Module):
    """LFM2 double-gated short-range convolution block.

    This implements the LIV (Linear Input-Varying) convolution as described:

    Args:
        config: LFM2 configuration
    """

    def __init__(self, idim, hidden_dim, kernel_size, dropout, dtype=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # Input projection to gates and values
        self.input_projection = nn.Linear(
            self.hidden_dim,
            3 * self.hidden_dim,  # B, C, x
            bias=False,
            dtype=dtype,
        )

        # Short convolution
        self.conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=kernel_size,
            padding=0,
            groups=self.hidden_dim,  # Depthwise convolution for efficiency
            bias=False,
            dtype=dtype,
        )

        # Output projection
        self.output_projection = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False, dtype=dtype
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cache_state=None, use_cache: bool = False):
        """Forward pass of the LFM2 convolution block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """

        # Input projection: B, C, x = linear(x)
        projected = self.input_projection(x)  # (B, L, 3*H)
        B, C, x_proj = projected.chunk(3, dim=-1)  # Each: (B, L, H)

        # First gating: x = B*x
        x_gated = B * x_proj

        # Apply causal short convolution
        # Convert to (B, H, L) for conv1d
        x_conv_input = x_gated.transpose(1, 2)  # (B, H, L)
        if cache_state is not None:
            x_conv_input = torch.cat([cache_state, x_conv_input], dim=-1)
        else:
            x_conv_input = F.pad(x_conv_input, (self.kernel_size - 1, 0))

        new_cache_state = None
        if use_cache:
            new_cache_state = x_conv_input[..., -(self.kernel_size - 1):].detach()

        x_conv = self.conv(x_conv_input)  # (B, H, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, H)

        # Second gating: x = C*x
        x_gated_2 = C * x_conv

        # Apply dropout
        x_gated_2 = self.dropout(x_gated_2)

        # Output projection
        output = self.output_projection(x_gated_2)

        if use_cache:
            return output, new_cache_state
        return output


# if __name__ == '__main__':
#     # test_apply_rope()
#     pass
