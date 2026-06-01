import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gemma3_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = gemma.language_model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )  # NO PROBLEM HERE BECAUSE RESID_PRE_0 IS HF_TL IDENTICAL

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # HACK to get the positional embeddings from HF model (theres currently a bug in the TLens implementation)
        if gemma.language_model.layers[l].self_attn.is_sliding:
            rotary_emb = gemma.language_model.rotary_emb_local
        else:
            rotary_emb = gemma.language_model.rotary_emb
        pos_embed = rotary_emb(
            torch.ones(1, device=gemma.device, dtype=cfg.dtype),
            torch.arange(cfg.n_ctx, device=gemma.device).unsqueeze(0),
        )
        cos, sin = pos_embed
        state_dict[f"blocks.{l}.attn.rotary_cos"] = cos.squeeze(0).to(cfg.device)
        state_dict[f"blocks.{l}.attn.rotary_sin"] = sin.squeeze(0).to(cfg.device)

        state_dict[f"blocks.{l}.attn.q_ln.w"] = gemma.language_model.layers[
            l
        ].self_attn.q_norm.weight.float() + torch.ones_like(
            gemma.language_model.layers[l].self_attn.q_norm.weight, dtype=torch.float32
        )
        state_dict[f"blocks.{l}.attn.k_ln.w"] = gemma.language_model.layers[
            l
        ].self_attn.k_norm.weight.float() + torch.ones_like(
            gemma.language_model.layers[l].self_attn.k_norm.weight, dtype=torch.float32
        )

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = gemma.language_model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            gemma.language_model.layers[l].input_layernorm.weight, dtype=torch.float32
        )
        if cfg.use_normalization_before_and_after:
            # Only applies for Gemma 2 and 3
            state_dict[f"blocks.{l}.ln1_post.w"] = gemma.language_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.language_model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        W_Q = gemma.language_model.layers[l].self_attn.q_proj.weight
        W_K = gemma.language_model.layers[l].self_attn.k_proj.weight
        W_V = gemma.language_model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = gemma.language_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not cfg.use_normalization_before_and_after:
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = gemma.language_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                gemma.language_model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2 (and 3?)
            state_dict[f"blocks.{l}.ln2.w"] = gemma.language_model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.language_model.layers[l].pre_feedforward_layernorm.weight,
                dtype=torch.float32,
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = gemma.language_model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                gemma.language_model.layers[l].post_feedforward_layernorm.weight,
                dtype=torch.float32,
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = gemma.language_model.layers[
            l
        ].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = gemma.language_model.layers[
            l
        ].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = gemma.language_model.layers[
            l
        ].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = gemma.language_model.norm.weight.float() + torch.ones_like(
        gemma.language_model.norm.weight, dtype=torch.float32
    )

    state_dict["unembed.W_U"] = gemma.language_model.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
