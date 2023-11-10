import math
from typing import Optional, Tuple, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: int = None,
            vdim: int = None,
            batch_first: bool = False,
            linear1_cls: nn.Module = nn.Linear,
            linear2_cls: nn.Module = nn.Linear,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype
        )

    def infer(self,
              x: torch.Tensor,
              attn_mask: Optional[torch.Tensor] = None,
              past_kv=None,
              use_cache=False
              ) -> tuple[Tensor, tuple[Tensor | Any, Tensor | Any] | None]:
        # x = x.transpose(1, 0)
        ipw = self.in_proj_weight
        ipb = self.in_proj_bias
        opw = self.out_proj.weight
        opb = self.out_proj.bias
        n_head = self.num_heads
        B, T, C = x.size()
        q, k, v = F.linear(x, ipw, ipb).chunk(3, dim=-1)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        FULL_T = k.shape[-2]

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attn_mask[FULL_T - T:FULL_T, :FULL_T], float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = F.linear(y, opw, opb)
        return y, present
