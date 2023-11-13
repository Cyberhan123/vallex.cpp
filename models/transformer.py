import math
from typing import Optional, Tuple, Any, Callable, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
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


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
            linear1_self_attention_cls: nn.Module = nn.Linear,
            linear2_self_attention_cls: nn.Module = nn.Linear,
            linear1_feedforward_cls: nn.Module = nn.Linear,
            linear2_feedforward_cls: nn.Module = nn.Linear,
            layer_norm_cls: nn.Module = nn.LayerNorm,
            layer_norm_eps: float = 1e-5,
            adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False
    ):
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                    src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding),
                src_mask,
                src_key_padding_mask,
            )
            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if is_src_tuple:
            return (x, stage_embedding)
        return x

    def infer(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            past_kv: Optional[Tensor] = None,
            use_cache: bool = False,
    ):
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                    src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if self.norm_first:
            x_attn_out, kv = self.self_attn.infer(
                self.norm1(x, stage_embedding),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                past_kv=past_kv,
                use_cache=use_cache,
            )
            x = x + x_attn_out
            x = x + self._ff_block(self.norm2(x, stage_embedding))

        if is_src_tuple:
            return (x, stage_embedding)
        return (x, kv)

class TransformerEncoder (nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer,num_layers,norm)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            return_layer_states: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """
        if return_layer_states:
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output

        output = src
        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def infer(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            return_layer_states: bool = False,
            past_kv: Optional[Tensor] = None,
            use_cache: bool = False,
    ):
        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * self.num_layers)
        else:
            past_length = past_kv[0][0].size(-2)
        new_kv = () if use_cache else None
        output = src
        for mod, past_layer_kv in zip(self.layers, past_kv):
            output, kv = mod.infer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, past_kv=past_layer_kv,
                use_cache=use_cache
            )
            if use_cache:
                new_kv = new_kv + (kv,)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_kv
