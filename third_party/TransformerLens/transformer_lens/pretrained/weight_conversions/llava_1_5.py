import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_llava_1_5_weights(llava, cfg: HookedTransformerConfig):
    state_dict = {}

    # Newer `transformers` moves the submodules under .model; older has them at top level.
    try:
        _lm = llava.__dict__["_modules"]["language_model"]
    except KeyError:
        _lm = llava.model.language_model
    try:
        _lm_head = llava.__dict__["_modules"]["lm_head"]
    except KeyError:
        _lm_head = llava.model.lm_head

    state_dict["embed.W_E"] = _lm.embed_tokens.weight

    assert cfg.d_mlp is not None  # keep mypy happy

    for l in range(cfg.n_layers):
        # HACK to get the positional embeddings from HF model (theres currently a bug in the TLens implementation)
        pos_embed = _lm.rotary_emb(
            torch.ones(1, device=llava.device, dtype=cfg.dtype),
            torch.arange(cfg.n_ctx, device=llava.device).unsqueeze(0),
        )
        cos, sin = pos_embed
        state_dict[f"blocks.{l}.attn.rotary_cos"] = cos.squeeze(0).to(cfg.device)
        state_dict[f"blocks.{l}.attn.rotary_sin"] = sin.squeeze(0).to(cfg.device)

        state_dict[f"blocks.{l}.ln1.w"] = _lm.layers[
            l
        ].input_layernorm.weight

        W_Q = _lm.layers[l].self_attn.q_proj.weight
        W_K = _lm.layers[l].self_attn.k_proj.weight
        W_V = _lm.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        # NO BIAS IN LLAVA
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)

        W_O = _lm.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = _lm.layers[
            l
        ].post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_in"] = _lm.layers[
            l
        ].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = _lm.layers[
            l
        ].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = _lm.layers[
            l
        ].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = _lm.layers[
        -1
    ].post_attention_layernorm.weight

    state_dict["unembed.W_U"] = _lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
