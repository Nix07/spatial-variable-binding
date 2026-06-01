import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_qwen2vl_weights(qwen, cfg: HookedTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = qwen.language_model.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        # HACK to get the positional embeddings from HF model (theres currently a bug in the TLens implementation)
        # Code taken from transformers.modeling_qwen2_vl.Qwen2VLModel.forward
        pos_ids = torch.arange(cfg.n_ctx, device=qwen.device)
        pos_ids = pos_ids.view(1, 1, -1).repeat(
            3, 1, 1
        )  # the hard coded `3` is for temporal, height and width.
        pos_embed = qwen.language_model.rotary_emb(
            torch.ones(1, device=qwen.device, dtype=cfg.dtype),
            pos_ids,
        )
        cos, sin = pos_embed
        state_dict[f"blocks.{l}.attn.rotary_cos"] = cos.to(cfg.device)
        state_dict[f"blocks.{l}.attn.rotary_sin"] = sin.to(cfg.device)

        state_dict[f"blocks.{l}.ln1.w"] = qwen.language_model.layers[l].input_layernorm.weight

        W_Q = qwen.language_model.layers[l].self_attn.q_proj.weight
        W_K = qwen.language_model.layers[l].self_attn.k_proj.weight
        W_V = qwen.language_model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        b_Q = qwen.language_model.layers[l].self_attn.q_proj.bias
        b_Q = einops.rearrange(
            b_Q,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_heads,
        )

        b_K = qwen.language_model.layers[l].self_attn.k_proj.bias
        b_K = einops.rearrange(
            b_K,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_key_value_heads,
        )

        b_V = qwen.language_model.layers[l].self_attn.v_proj.bias
        b_V = einops.rearrange(
            b_V,
            "(n_head d_head) -> n_head d_head",
            n_head=cfg.n_key_value_heads,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
        state_dict[f"blocks.{l}.attn._b_K"] = b_K
        state_dict[f"blocks.{l}.attn._b_V"] = b_V

        W_O = qwen.language_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = qwen.language_model.layers[l].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = qwen.language_model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = qwen.language_model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = qwen.language_model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = qwen.language_model.norm.weight

    state_dict["unembed.W_U"] = qwen.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
