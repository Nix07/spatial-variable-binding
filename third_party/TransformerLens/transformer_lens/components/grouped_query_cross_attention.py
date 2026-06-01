from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.components import AbstractAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear


class GroupedQueryCrossAttention(AbstractAttention):
    """
    DONE - Q is tokens (as normal) + q_norm
    DONE - K and V are cross_attention_states, projected + k_norm AFTER repeat_head

    Q,K,V sent to normal SDPA
    """

    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Union[int, None] = None,
        qk_norm_cls: nn.Module = RMSNorm,
    ):
        cfg = HookedTransformerConfig.unwrap(cfg)
        assert cfg.n_key_value_heads is not None
        super().__init__(
            cfg, attn_type, layer_id, force_ignore_pos_embed=True
        )  # No positional embedding should be applied in cross-attention
        self.repeat_kv_heads = cfg.n_heads // cfg.n_key_value_heads
        self._W_K = nn.Parameter(
            torch.empty(
                cfg.n_key_value_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype=cfg.dtype,
            )
        )
        self._W_V = nn.Parameter(
            torch.empty(
                cfg.n_key_value_heads,
                self.cfg.d_model,
                self.cfg.d_head,
                dtype=cfg.dtype,
            )
        )
        self._b_K = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )
        self._b_V = nn.Parameter(
            torch.zeros(cfg.n_key_value_heads, self.cfg.d_head, dtype=cfg.dtype)
        )

        # if qk_norm_cls.__name__.endswith("RMSNormPre") or qk_norm_cls.__name__.endswith("RMSNorm"):
        # self.q_ln = qk_norm_cls(cfg)
        # self.k_ln = qk_norm_cls(cfg)
        # elif :
        self.q_ln = qk_norm_cls(cfg, self.cfg.d_head)
        self.k_ln = qk_norm_cls(cfg, self.cfg.d_head)
        # else:
        # raise ValueError(f"qk_norm_cls must be RMSNormPre or RMSNorm, not {qk_norm_cls.__name__}")

    @property
    def W_K(self):
        return torch.repeat_interleave(self._W_K, dim=0, repeats=self.repeat_kv_heads)

    @W_K.setter
    def W_K(self, value):
        self._W_K = value

    @property
    def W_V(self):
        return torch.repeat_interleave(self._W_V, dim=0, repeats=self.repeat_kv_heads)

    @W_V.setter
    def W_V(self, value):
        self._W_V = value

    @property
    def b_K(self):
        return torch.repeat_interleave(self._b_K, dim=0, repeats=self.repeat_kv_heads)

    @b_K.setter
    def b_K(self, value):
        self._b_K = value

    @property
    def b_V(self):
        return torch.repeat_interleave(self._b_V, dim=0, repeats=self.repeat_kv_heads)

    @b_V.setter
    def b_V(self, value):
        self._b_V = value

    def apply_causal_mask(
        self,
        attn_scores: Float[torch.Tensor, "batch head_index q_pos kv_pos"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch kv_image_idx q_pos kv_pos"]] = None,
    ):
        """
        Actually applies an attention mask to the attention scores.
        NOT CAUSAL - This relies on the cross_attention_mask flowing into the attention_mask parameter.
        """
        assert attention_mask is not None
        assert (
            past_kv_pos_offset == 0
        ), "No support for past-key value cache in GroupedQueryCrossAttention"

        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        # query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        # key_ctx_length = attn_scores.size(-1)

        # IRRELEVANT FOR CROSS ATTENTION
        # if query_ctx_length + past_kv_pos_offset != key_ctx_length:
        #     raise ValueError(
        #         f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
        #     )

        attn_scores += attention_mask

        # Index back to front to ensure local attention works
        # final_mask = self.mask[None, None, -query_ctx_length:, -key_ctx_length:]  # [1, 1, pos, pos]
        # if attention_mask is not None:
        #     # Apply a causal mask to the attention scores considering the padding
        #     einsum_str = "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
        #     final_mask = final_mask.to(attention_mask.device)
        #     final_mask = einops.einsum(final_mask, attention_mask, einsum_str).bool()
        # attn_scores = attn_scores.to(final_mask.device)
        # return torch.where(final_mask, attn_scores, self.IGNORE)
        return attn_scores

    def calculate_qkv_matrices(
        self,
        query_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos kv_head_index d_model"],
        ],
        value_input: Union[
            Float[torch.Tensor, "batch pos d_model"],
            Float[torch.Tensor, "batch pos kv_head_index d_model"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch pos head_index d_head"],
        Float[torch.Tensor, "batch pos kv_head_index d_head"],
        Float[torch.Tensor, "batch pos kv_head_index d_head"],
    ]:
        """Calculate the Q, K, and V matrices for grouped query attention.
        This function uses the unexpanded weights _W_K and _W_V to calculate K and V.

        Args:
        query_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos head_index d_model"]]): The input tensor for the query projection.
        key_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos kv_head_index d_model"]]): The input tensor for the key projection. Note that is has as many head dimensions as the GPA block has key-value heads.
        value_input (Union[Float[torch.Tensor, "batch pos d_model"], Float[torch.Tensor, "batch pos kv_head_index d_model"]]): The input tensor for the value projection. Note that is has as many head dimensions as the GPA block has key-value heads.

        Returns:
        Tuple[Float[torch.Tensor, "batch pos head_index d_head"], Float[torch.Tensor, "batch pos kv_head_index d_head"], Float[torch.Tensor, "batch pos kv_head_index d_head"]]:
        A tuple containing the Q, K, and V matrices with the specified shapes.
        """
        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )

        q = self.hook_q(
            self.q_ln(attn_fn(query_input, self.W_Q, self.b_Q))
        )  # [batch, pos, head_index, d_head]

        k = (
            attn_fn(key_input, self.W_K, self.b_K)
            if self.cfg.ungroup_grouped_query_attention
            else attn_fn(key_input, self._W_K, self._b_K)
        )  # [batch * tile, img_pos, head_index, d_head]
        k = k.view(q.shape[0], -1, k.shape[-2], k.shape[-1])  # [batch, kv_pos, head_index, d_head)]
        k = self.hook_k(self.k_ln(k))

        v = self.hook_v(
            (
                attn_fn(value_input, self.W_V, self.b_V)
                if self.cfg.ungroup_grouped_query_attention
                else attn_fn(value_input, self._W_V, self._b_V)
            ).view(
                q.shape[0], -1, k.shape[-2], k.shape[-1]
            )  # [batch, kv_pos, head_index, d_head)]
        )
        return q, k, v

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos kv_head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        """Calculate attention scores from Q and the unexpanded K matrix.
        K will be expaned from [batch, pos, n_key_value_head, d_head] to [batch, pos, n_query_heads, d_head] using torch.repeat_interleave.

        Args:
        q (Float[torch.Tensor, "batch query_pos head_index d_head"]): The Q tensor.
        k (Float[torch.Tensor, "batch key_pos kv_head_index d_head"]): The K tensor.

        Returns:
            Float[torch.Tensor, "batch head_index query_pos key_pos"]: The attention scores.
        """
        if not self.cfg.ungroup_grouped_query_attention:
            k = torch.repeat_interleave(k, dim=2, repeats=self.repeat_kv_heads)
        return super().calculate_attention_scores(q, k)

    def calculate_z_scores(
        self,
        v: Float[torch.Tensor, "batch key_pos kv_head_index d_head"],
        pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
        """Calculate z scores from the attention pattern and the unexpanded V matrix.
        V will be expaned from [batch, pos, n_key_value_head, d_head] to [batch, pos, n_query_heads, d_head] using torch.repeat_interleave.

        Args:
        v (Float[torch.Tensor, "batch query_pos head_index d_head"]): The V tensor.
        pattern (Float[torch.Tensor, "batch key_pos kv_head_index d_head"]): The attention pattern.

        Returns:
            Float[torch.Tensor, "batch head_index query_pos key_pos"]: The z scores.
        """
        if not self.cfg.ungroup_grouped_query_attention:
            v = torch.repeat_interleave(v, dim=2, repeats=self.repeat_kv_heads)
        return super().calculate_z_scores(v, pattern)
