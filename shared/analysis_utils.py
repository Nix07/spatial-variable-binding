#
# Contains various functions used across analysis scripts (such as model loading, dataset loading, etc)
#

import sys
from typing import List, Optional, Union


sys.path.append("./third_party/TransformerLens")

import torch
import transformer_lens as lens
from transformers import (
    Qwen2VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def load_model(
    model_name: str,
    model_path: str,
    device: Union[str, torch.device],
    device_list: Optional[List[int]] = None,
    use_tlens_wrapper: bool = True,
    extra_hooks: bool = True,
    torch_dtype: torch.dtype = torch.float32,
    cache_dir: Optional[str] = None,
):
    if any(k in model_name.lower() for k in ("llama3.2", "llama-3.2", "mllama")):
        inner_model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(model_path, device=device, cache_dir=cache_dir)
        if use_tlens_wrapper:
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                device=device,
                device_list=device_list,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        return model, processor

    elif "qwen" in model_name.lower():
        inner_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.visual
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,  # False,
                fold_value_biases=True,
                n_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                device=device,
                device_list=device_list,
            )
            model.cfg.default_prepend_bos = False  # To match HF Qwen model forward pass
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "pixtral" in model_name.lower() or "llava" in model_name.lower():
        inner_model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir) # removed use_fast=False?
        if use_tlens_wrapper:
            _vt = None
            try:
                _vt = inner_model.vision_tower
            except AttributeError:
                pass
            if _vt is None:
                try:
                    _vt = inner_model.model.vision_tower
                except AttributeError:
                    pass
            if _vt is None:
                raise AttributeError(
                    "Could not locate vision_tower on LlavaForConditionalGeneration "
                    f"(tried .vision_tower and .model.vision_tower on {type(inner_model).__name__})"
                )
            inner_model.vision_model = _vt
            # Newer transformers also nests multi_modal_projector under .model; lift it.
            _mmp = None
            try:
                _mmp = inner_model.__dict__["_modules"]["multi_modal_projector"]
            except KeyError:
                pass
            if _mmp is None:
                try:
                    _mmp = inner_model.model.multi_modal_projector
                    inner_model.multi_modal_projector = _mmp
                except AttributeError:
                    pass
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                device=device,
                device_list=device_list,
                dtype=torch_dtype,
            )
            model.cfg.default_prepend_bos = (
                False if "pixtral" in model_name.lower() else True
            )
            # model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "gemma-3" in model_name.lower():
        inner_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
            cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        # processor.chat_template = processor.chat_template.replace("{{ bos_token }}", "")
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.vision_tower
            # transformers 4.55+: language_model is Gemma3TextModel (no .lm_head),
            # lm_head lives at top-level. TL's gemma3 weight converter reads
            # gemma.language_model.lm_head, so alias it down defensively.
            if not hasattr(inner_model.language_model, "lm_head") and hasattr(
                inner_model, "lm_head"
            ):
                inner_model.language_model.lm_head = inner_model.lm_head
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=False,  # ,
                center_unembed=True,  # True,
                center_writing_weights=True,  # True,
                fold_value_biases=False,  # True,
                n_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                device=device,
                device_list=device_list,
            )
            model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    else:
        print("WARNING: Using model not officially supported in load_model")
        inner_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if use_tlens_wrapper:
            model = lens.HookedTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                tokenizer=tokenizer,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
                device=device,
                device_list=device_list,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
        else:
            model = inner_model
        model.model_name = model_name
        return model, tokenizer
