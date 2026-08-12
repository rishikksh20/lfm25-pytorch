import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from modules import apply_rope, RMSNorm

def l2norm(x, dim=-1, eps=1e-6):
    """Unit L2 normalisation without a learnable scale (matches FLA / HF convention)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class GQAttention(nn.Module):
    """Grouped Query Attention used by LFM2 full-attention layers."""

    def __init__(self, idim, n_heads, num_groups, head_dim, rope_dim, norm_eps, dtype):
        super().__init__()
        self.idim = idim
        self.n_heads = n_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.group_size = n_heads // num_groups
        self.n_kv_embed = head_dim * num_groups
        self.odim = n_heads * head_dim
        self.scale = head_dim ** -0.5


        self.q_proj = nn.Linear(idim, self.odim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.o_proj = nn.Linear(self.odim, idim, dtype=dtype, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps, dtype=dtype)

    def forward(self, x, cos, sin, mask=None, past_key_value=None, use_cache=False, position_offset=0):

        
        q_raw = self.q_proj(x)                                  # (B, L, odim)
        q = rearrange(q_raw, 'b l (n d) -> b n l d', n=self.n_heads)
        

        k = self.k_proj(x)   # (B, L, n_kv_embed)
        v = self.v_proj(x)   # (B, L, n_kv_embed)
        k = rearrange(k, 'b l (g d) -> b g l d', g=self.num_groups)
        v = rearrange(v, 'b l (g d) -> b g l d', g=self.num_groups)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE: rotate only the leading rope_dim features per head
        q = torch.cat([apply_rope(q[..., :self.rope_dim], cos, sin, position_offset=position_offset),
                       q[..., self.rope_dim:]], dim=-1)
        k = torch.cat([apply_rope(k[..., :self.rope_dim], cos, sin, position_offset=position_offset),
                       k[..., self.rope_dim:]], dim=-1)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_key_value = None
        if use_cache:
            new_key_value = (k.detach(), v.detach())

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask, -torch.inf)
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        out = self.o_proj(out)
        if use_cache:
            return out, new_key_value
        return out
