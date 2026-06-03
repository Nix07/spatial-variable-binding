"""Raw-HF Pixtral helpers for the COCO amplification experiment.

The Pixtral COCO notebook used the Hugging Face model directly rather than the
repo's TransformerLens wrapper. This module keeps that execution path for the
reproducibility-sensitive baseline, probe extraction, and amplification.
"""

from __future__ import annotations

import gc
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, LlavaForConditionalGeneration

from shared.prompts import chat_prompt

from . import common as cc
from .prompts import relative_prompt_text


MODEL_IDS = {
    "pixtral-12b": "mistral-community/pixtral-12b",
}

IMG_TOKEN_ID = 10
IMG_BREAK_TOKEN_ID = 12
IMG_END_TOKEN_ID = 13


def canonical_model_name(name: str) -> str:
    key = name.lower()
    aliases = {
        "pixtral": "pixtral-12b",
        "pixtral-12b": "pixtral-12b",
        "mistral-community/pixtral-12b": "pixtral-12b",
    }
    if key not in aliases:
        raise ValueError(f"unsupported COCO Pixtral model {name!r}")
    return aliases[key]


def _patch_pixtral_dtype():
    from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

    if hasattr(PixtralVisionModel, "_codex_pixtral_dtype_patch"):
        return
    orig_forward = PixtralVisionModel.forward

    def patched_forward(self, pixel_values, *args, **kwargs):
        target = self.patch_conv.weight.dtype
        if pixel_values.dtype != target:
            pixel_values = pixel_values.to(dtype=target)
        return orig_forward(self, pixel_values, *args, **kwargs)

    PixtralVisionModel.forward = patched_forward
    PixtralVisionModel._codex_pixtral_dtype_patch = True


def _normalize_answer(text):
    return str(text).replace("</s>", "").strip().lower()


class PixtralCocoRunner:
    def __init__(self, *, model_name: str, dtype_name: str, img_dir: str,
                 cache_dir: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("Pixtral COCO runner follows the notebook and requires CUDA.")

        self.model_slug = canonical_model_name(model_name)
        self.model_id = MODEL_IDS[self.model_slug]
        self.dtype_name = dtype_name
        self.torch_dtype = cc.torch_dtype_from_name(dtype_name)
        self.img_dir = img_dir

        print(f"Loading {self.model_id} with Hugging Face ({dtype_name}) ...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            cache_dir=cache_dir,
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        _patch_pixtral_dtype()

        self.model_device = next(self.model.parameters()).device
        self.vision_model = (
            getattr(self.model, "vision_tower", None)
            or getattr(self.model, "vision_model", None)
        )
        self.projector = (
            getattr(self.model, "multi_modal_projector", None)
            or getattr(getattr(self.model, "model", None), "multi_modal_projector", None)
        )
        self.language_model = getattr(self.model, "language_model", None)
        lm_core = getattr(self.language_model, "model", self.language_model)
        self.lm_layers = getattr(lm_core, "layers", None)

        assert self.vision_model is not None, "Could not locate Pixtral vision module"
        assert self.projector is not None, "Could not locate Pixtral multimodal projector"
        assert self.language_model is not None and self.lm_layers is not None, (
            "Could not locate Pixtral language layers"
        )

        cfg_candidates = [
            getattr(self.model.config, "text_config", None),
            getattr(self.language_model, "config", None),
            self.model.config,
        ]
        self.hidden_dim = next(
            int(getattr(cfg, "hidden_size"))
            for cfg in cfg_candidates
            if cfg is not None and hasattr(cfg, "hidden_size")
        )
        self.n_layers = len(self.lm_layers)
        assert self.hidden_dim == 5120, f"expected d_model=5120, got {self.hidden_dim}"
        assert self.n_layers == 40, f"expected n_layers=40, got {self.n_layers}"

        self._check_image_token_ids()
        gc.collect()
        torch.cuda.empty_cache()

        print("Pixtral model loaded.")
        print(f"  Vision module:    {type(self.vision_model).__name__}")
        print(f"  Projector:        {type(self.projector).__name__}")
        print(f"  Language module:  {type(self.language_model).__name__}")
        print(f"  d_model = {self.hidden_dim} | n_layers = {self.n_layers}")
        print(f"MODEL_DTYPE = {dtype_name} | actual parameter dtype = {next(self.model.parameters()).dtype}")

    def _check_image_token_ids(self):
        img_id = self.processor.tokenizer.convert_tokens_to_ids("[IMG]")
        brk_id = self.processor.tokenizer.convert_tokens_to_ids("[IMG_BREAK]")
        end_id = self.processor.tokenizer.convert_tokens_to_ids("[IMG_END]")
        print(f"[IMG]={img_id}, [IMG_BREAK]={brk_id}, [IMG_END]={end_id}")
        assert img_id == IMG_TOKEN_ID, f"[IMG] id mismatch: tokenizer says {img_id}, expected {IMG_TOKEN_ID}"
        assert brk_id == IMG_BREAK_TOKEN_ID, (
            f"[IMG_BREAK] id mismatch: tokenizer says {brk_id}, expected {IMG_BREAK_TOKEN_ID}"
        )
        assert end_id == IMG_END_TOKEN_ID, (
            f"[IMG_END] id mismatch: tokenizer says {end_id}, expected {IMG_END_TOKEN_ID}"
        )

    def image_path_for_id(self, image_id) -> str:
        return os.path.join(self.img_dir, f"{int(image_id):012d}.jpg")

    @staticmethod
    def load_pixtral_image(img_path):
        return PILImage.open(img_path).convert("RGB")

    def build_relative_prompt(self, subj, obj, listing, axis=None):
        text = relative_prompt_text(subj, obj, listing, axis=axis)
        return chat_prompt(self.processor, text, system_text=None)

    def build_pixtral_inputs(self, img_path, subj, obj, listing, axis=None):
        img = self.load_pixtral_image(img_path)
        prompt = self.build_relative_prompt(subj, obj, listing, axis=axis)
        return self.processor(images=[img], text=prompt, return_tensors="pt").to(self.model_device)

    def decode_next_token(self, logits):
        pred_id = torch.argmax(logits[0, -1]).item()
        return _normalize_answer(self.processor.tokenizer.decode([pred_id], skip_special_tokens=True))

    @staticmethod
    def pixtral_image_positions(input_ids):
        return (input_ids == IMG_TOKEN_ID).nonzero().flatten().tolist()

    @staticmethod
    def infer_pixtral_grid(input_ids):
        ids = input_ids.detach().cpu().tolist()
        row_counts, cur = [], 0
        in_image = False
        for tok in ids:
            if tok == IMG_TOKEN_ID:
                in_image = True
                cur += 1
            elif tok == IMG_BREAK_TOKEN_ID and in_image:
                row_counts.append(cur)
                cur = 0
            elif tok == IMG_END_TOKEN_ID and in_image:
                row_counts.append(cur)
                break
        row_counts = [r for r in row_counts if r > 0]
        assert row_counts, "could not infer Pixtral image grid from input_ids"
        assert len(set(row_counts)) == 1, (
            f"non-rectangular Pixtral image rows: {row_counts[:10]}..."
        )
        return len(row_counts), row_counts[0]

    def clear_layer0_hooks(self):
        self.lm_layers[0]._forward_pre_hooks.clear()
        self.lm_layers[0]._forward_hooks.clear()

    @torch.inference_mode()
    def predict_with_amp(self, item, listing, W=None, alpha=0.0):
        axis = item.get("axis") or cc.AXIS_BY_DIRECTION[item["truth"]]
        inputs = self.build_pixtral_inputs(
            self.image_path_for_id(item["image_id"]),
            item["subj"],
            item["obj"],
            listing,
            axis=axis,
        )
        input_ids = inputs["input_ids"][0]
        img_pos = self.pixtral_image_positions(input_ids)
        assert img_pos, "prompt produced no [IMG] positions"

        handle = None
        if W is not None and alpha != 0:
            W_amp = W.to(device=self.model_device, dtype=self.torch_dtype)

            def amp_pre_hook(module, args):
                hs = args[0]
                if hs.shape[1] == 1:
                    return args
                valid = [p for p in img_pos if p < hs.shape[1]]
                if not valid:
                    return args
                hs_new = hs.clone()
                img_embeds = hs_new[:, valid, :]
                denom = (W_amp ** 2).sum(dim=0) + 1e-8
                coeff = torch.einsum("btd,dk->btk", img_embeds, W_amp) / denom
                projected = torch.einsum("btk,dk->btd", coeff, W_amp)
                hs_new[:, valid, :] = img_embeds + float(alpha) * projected
                return (hs_new,) + args[1:]

            handle = self.lm_layers[0].register_forward_pre_hook(amp_pre_hook)
        try:
            out = self.model(**inputs)
        finally:
            if handle is not None:
                handle.remove()
        logits = out.logits if hasattr(out, "logits") else out[0]
        pred = self.decode_next_token(logits)
        del inputs, out, logits
        torch.cuda.empty_cache()
        return pred

    def build_dummy_inputs(self, img_path):
        img = self.load_pixtral_image(img_path)
        text = chat_prompt(self.processor, "Describe the image.", system_text=None)
        return self.processor(images=[img], text=text, return_tensors="pt").to(self.model_device)

    def extract_projector_output(self, inputs):
        proj_out = {}

        def hook_projector(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            proj_out["h"] = h.detach().clone()

        handle = self.projector.register_forward_hook(hook_projector)
        try:
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            handle.remove()
        h = proj_out["h"]
        if h.dim() == 3:
            assert h.shape[0] == 1
            h = h[0]
        input_ids = inputs["input_ids"][0]
        img_pos = self.pixtral_image_positions(input_ids)
        assert len(img_pos) == h.shape[0], (
            f"projector/[IMG] mismatch: projector={h.shape[0]} [IMG]={len(img_pos)}"
        )
        return h

    @staticmethod
    def bbox_to_pixtral_token_indices(box, img_size, grid_h, grid_w):
        width, height = img_size
        x1, y1, x2, y2 = box
        eps = 1e-3
        col1 = int(x1 * grid_w / width)
        col2 = int((x2 - eps) * grid_w / width) if x2 > x1 else col1
        row1 = int(y1 * grid_h / height)
        row2 = int((y2 - eps) * grid_h / height) if y2 > y1 else row1
        col1 = max(0, min(col1, grid_w - 1))
        col2 = max(col1, min(col2, grid_w - 1))
        row1 = max(0, min(row1, grid_h - 1))
        row2 = max(row1, min(row2, grid_h - 1))
        return [row * grid_w + col for row in range(row1, row2 + 1) for col in range(col1, col2 + 1)]

    @staticmethod
    def token_centers(idx, grid_h, grid_w):
        del grid_h
        rows = [i // grid_w for i in idx]
        cols = [i % grid_w for i in idx]
        return sum(rows) / len(rows), sum(cols) / len(cols)

    def side_check(self, axis, direction, idx_subj, idx_obj, grid_h, grid_w):
        row_s, col_s = self.token_centers(idx_subj, grid_h, grid_w)
        row_o, col_o = self.token_centers(idx_obj, grid_h, grid_w)
        if axis == "horizontal":
            return ((direction == "left") == (col_s < col_o))
        if axis == "vertical":
            return ((direction == "above") == (row_s < row_o))
        raise ValueError(axis)

    @staticmethod
    def labels_for_axis_truth(axis, direction):
        first, _ = cc.LABELS_BY_AXIS[axis]
        return (0, 1) if direction == first else (1, 0)

    def build_probe_data(self, partition, get_bbox, artifact_dir: str | Path,
                         *, model_slug: str, experiment_slug: str,
                         artifact_suffix: str, axes=("horizontal", "vertical")):
        artifact_dir = Path(artifact_dir)
        probe_data = {
            axis: {
                "X_list": [],
                "y_list": [],
                "meta": [],
                "bbox_sources": [],
                "det_failures": 0,
                "empty_token_failures": 0,
                "bbox_check_failures": 0,
            }
            for axis in axes
        }

        for golden in cc.progress(partition["golden"], desc="embeddings"):
            axis = golden["axis"]
            img_id = golden["image_id"]
            direction = golden["truth"]
            subj_q = cc.strip_article(golden["subj"])
            obj_q = cc.strip_article(golden["obj"])
            img_path = self.image_path_for_id(img_id)
            box_subj, img_size, src_subj = get_bbox(img_id, subj_q, img_path)
            box_obj, _, src_obj = get_bbox(img_id, obj_q, img_path)
            d = probe_data[axis]
            d["bbox_sources"].append((src_subj, src_obj))
            if box_subj is None or box_obj is None:
                d["det_failures"] += 1
                continue

            inputs = self.build_dummy_inputs(img_path)
            grid_h, grid_w = self.infer_pixtral_grid(inputs["input_ids"][0])
            h = self.extract_projector_output(inputs)
            idx_subj = self.bbox_to_pixtral_token_indices(box_subj, img_size, grid_h, grid_w)
            idx_obj = self.bbox_to_pixtral_token_indices(box_obj, img_size, grid_h, grid_w)
            if not idx_subj or not idx_obj:
                d["empty_token_failures"] += 1
                continue
            if not self.side_check(axis, direction, idx_subj, idx_obj, grid_h, grid_w):
                d["bbox_check_failures"] += 1
                continue

            emb_subj = h[idx_subj, :].float().mean(dim=0).cpu()
            emb_obj = h[idx_obj, :].float().mean(dim=0).cpu()
            label_subj, label_obj = self.labels_for_axis_truth(axis, direction)
            d["X_list"].append(emb_subj.numpy())
            d["y_list"].append(label_subj)
            d["X_list"].append(emb_obj.numpy())
            d["y_list"].append(label_obj)
            d["meta"].append({
                "image_id": img_id,
                "subj": subj_q,
                "obj": obj_q,
                "direction": direction,
                "axis": axis,
                "box_subj": box_subj,
                "box_obj": box_obj,
                "img_size": img_size,
                "idx_subj": idx_subj,
                "idx_obj": idx_obj,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "n_img_tokens": h.shape[0],
                "label_subj": label_subj,
                "label_obj": label_obj,
            })

        artifact_dir.mkdir(parents=True, exist_ok=True)
        for axis in axes:
            d = probe_data[axis]
            X_axis = np.array(d["X_list"], dtype=np.float32)
            y_axis = np.array(d["y_list"], dtype=np.int64)
            meta_axis = d["meta"]
            assert X_axis.shape[0] == 2 * len(meta_axis), (
                f"{axis}: X rows={X_axis.shape[0]} expected {2 * len(meta_axis)}"
            )
            assert y_axis.shape[0] == X_axis.shape[0], (
                f"{axis}: y rows={y_axis.shape[0]} X rows={X_axis.shape[0]}"
            )
            assert X_axis.ndim == 2 and X_axis.shape[1] == self.hidden_dim, (
                f"{axis}: unexpected X shape: {X_axis.shape}"
            )
            assert set(y_axis.tolist()).issubset({0, 1}), (
                f"{axis}: unexpected labels: {sorted(set(y_axis.tolist()))}"
            )
            assert len(set(y_axis.tolist())) == 2, f"{axis}: probe labels contain only one class"
            for i, meta in enumerate(meta_axis):
                assert y_axis[2 * i] == meta["label_subj"] and y_axis[2 * i + 1] == meta["label_obj"], (
                    f"{axis}: label/meta mismatch at item {i}"
                )
                assert meta["idx_subj"] and meta["idx_obj"], f"{axis}: empty token ids in meta item {i}"
                assert max(meta["idx_subj"] + meta["idx_obj"]) < meta["n_img_tokens"], (
                    f"{axis}: token id out of range in meta item {i}"
                )
            d["X"] = X_axis
            d["y"] = y_axis
            src_summary = Counter()
            for src_sub, src_obj in d["bbox_sources"]:
                src_summary[src_sub] += 1
                src_summary[src_obj] += 1
            print("\n" + "=" * 60)
            print(f"EMBEDDING EXTRACTION DONE - {axis.upper()}")
            print("=" * 60)
            print(f'  Correct items input:      {sum(1 for g in partition["golden"] if g["axis"] == axis)}')
            print(f'  Detection failures:       {d["det_failures"]}')
            print(f'  Empty-token failures:     {d["empty_token_failures"]}')
            print(f'  Bbox-side failures:       {d["bbox_check_failures"]}')
            print(f"  Items in probe pool:      {len(meta_axis)}")
            print(f"  Embeddings (X):           {X_axis.shape}")
            print(f'  Labels (y):               {y_axis.shape}  class balance = {y_axis.mean() if len(y_axis) else float("nan"):.3f}')
            print(f"  bbox source breakdown:    {dict(src_summary)}")
            np.save(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_X_{artifact_suffix}.npy", X_axis)
            np.save(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_y_{artifact_suffix}.npy", y_axis)
            with open(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_meta_{artifact_suffix}.pkl", "wb") as f:
                pickle.dump(meta_axis, f)

        return probe_data
