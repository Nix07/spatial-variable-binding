"""Raw-HF Qwen2-VL helpers for the COCO amplification experiment.

The original COCO numbers were generated with raw Hugging Face Qwen2-VL
forward hooks. This module keeps that execution path intact.
"""

from __future__ import annotations

import gc
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from . import common as cc
from .prompts import relative_prompt_text


QWEN_IMAGE_PAD_TOKEN = "<|image_pad|>"
QWEN2_PATCH_SIZE = 14
QWEN2_FACTOR = 28
QWEN2_MERGE_FACTOR = 4


MODEL_IDS = {
    "qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
}


def canonical_model_name(name: str) -> str:
    key = name.lower()
    aliases = {
        "qwen2vl2b": "qwen2-vl-2b-instruct",
        "qwen": "qwen2-vl-2b-instruct",
        "qwen2-vl-2b": "qwen2-vl-2b-instruct",
        "qwen2-vl-2b-instruct": "qwen2-vl-2b-instruct",
        "qwen/qwen2-vl-2b-instruct": "qwen2-vl-2b-instruct",
    }
    if key not in aliases:
        raise ValueError(f"unsupported COCO amplification model {name!r}")
    return aliases[key]


def resolve_lm_layers(qwen_model):
    candidates = [
        getattr(qwen_model, "language_model", None),
        getattr(getattr(qwen_model, "model", None), "language_model", None),
        getattr(qwen_model, "model", None),
        getattr(getattr(qwen_model, "model", None), "model", None),
    ]
    for cand in candidates:
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand, cand.layers
        inner = getattr(cand, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return cand, inner.layers
    raise AssertionError("Could not locate Qwen language layers")


class QwenCocoRunner:
    def __init__(self, *, model_name: str, dtype_name: str, img_dir: str,
                 cache_dir: str | None = None):
        self.model_slug = canonical_model_name(model_name)
        self.model_id = MODEL_IDS[self.model_slug]
        self.dtype_name = dtype_name
        self.torch_dtype = cc.torch_dtype_from_name(dtype_name)
        self.img_dir = img_dir

        print(f"Loading {self.model_id} ...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.image_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids(QWEN_IMAGE_PAD_TOKEN)
        assert (
            self.image_pad_token_id is not None
            and self.image_pad_token_id != self.processor.tokenizer.unk_token_id
        ), f"tokenizer does not know {QWEN_IMAGE_PAD_TOKEN}"

        self.model_device = next(self.model.parameters()).device
        self.vision_model = (
            getattr(self.model, "visual", None)
            or getattr(getattr(self.model, "model", None), "visual", None)
        )
        assert self.vision_model is not None, (
            "Could not locate Qwen visual module on model.visual or model.model.visual"
        )
        self.language_model, self.lm_layers = resolve_lm_layers(self.model)
        assert getattr(self.model.config.vision_config, "spatial_merge_size", 2) == 2

        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        print(f"d_model = {text_cfg.hidden_size} | n_layers = {text_cfg.num_hidden_layers}")
        print(f"visual spatial_merge_size = {getattr(cfg.vision_config, 'spatial_merge_size', 'unknown')}")
        print(f"MODEL_DTYPE = {dtype_name} | actual parameter dtype = {next(self.model.parameters()).dtype}")
        print(f"  Visual module:   {type(self.vision_model).__name__}")
        print(f"  Language module: {type(self.language_model).__name__}")
        print(f"  LM layers:       {len(self.lm_layers)}")

    def image_path_for_id(self, image_id) -> str:
        return os.path.join(self.img_dir, f"{int(image_id):012d}.jpg")

    def build_relative_messages(self, img_path, subj, obj, listing, axis=None):
        text = relative_prompt_text(subj, obj, listing, axis=axis)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def build_qwen_inputs(self, img_path, subj, obj, listing, axis=None):
        messages = self.build_relative_messages(img_path, subj, obj, listing, axis=axis)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model_device)

    def clear_layer0_hooks(self):
        for h in list(self.lm_layers[0]._forward_pre_hooks.values()):
            try:
                h.remove()
            except Exception:
                pass
        self.lm_layers[0]._forward_pre_hooks.clear()

    def predict_with_amp(self, item, listing, W=None, alpha=0.0):
        axis = item["axis"]
        img_path = self.image_path_for_id(item["image_id"])
        inputs = self.build_qwen_inputs(img_path, item["subj"], item["obj"], listing, axis=axis)
        input_ids = inputs["input_ids"][0]
        img_pos = (input_ids == self.image_pad_token_id).nonzero().flatten().tolist()
        assert img_pos, f'No image-pad tokens for image_id={item["image_id"]}'

        handle = None
        if W is not None and alpha != 0:
            W_t = W.to(device=self.model_device)

            def pre_hook(module, args):
                hs = args[0]
                if hs.shape[1] == 1:
                    return args
                valid_pos = [pos for pos in img_pos if pos < hs.shape[1]]
                if not valid_pos:
                    return args
                W_local = W_t.to(dtype=hs.dtype, device=hs.device)
                img_embeds = hs[:, valid_pos, :]
                coeff = torch.einsum("btd,dk->btk", img_embeds, W_local) / (
                    (W_local ** 2).sum(dim=0) + 1e-8
                )
                projected = torch.einsum("btk,dk->btd", coeff, W_local)
                hs[:, valid_pos, :] = hs[:, valid_pos, :] + alpha * projected
                return (hs,) + args[1:]

            handle = self.lm_layers[0].register_forward_pre_hook(pre_hook)

        try:
            with torch.inference_mode():
                out = self.model(**inputs)
                pred_id = out.logits[0, -1].argmax().item()
        finally:
            if handle is not None:
                handle.remove()
        return self.processor.tokenizer.decode([pred_id], skip_special_tokens=True).strip().lower()

    def build_dummy_inputs(self, img_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model_device)

    def extract_merger_output(self, inputs):
        merger_out = {}

        def hook_merger(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            merger_out["h"] = h.detach().clone()

        handle = self.vision_model.merger.register_forward_hook(hook_merger)
        try:
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            handle.remove()
        h = merger_out["h"]
        if h.dim() == 3:
            assert h.shape[0] == 1
            h = h[0]
        input_ids = inputs["input_ids"][0]
        img_pos = (input_ids == self.image_pad_token_id).nonzero().flatten().tolist()
        assert len(img_pos) == h.shape[0], (
            f"merger/image-pad mismatch: merger={h.shape[0]} image_pad={len(img_pos)}"
        )
        return h

    def bbox_to_patch_coords_qwen2(self, box, img_size, grid_h=None, grid_w=None,
                                   patch_size=QWEN2_PATCH_SIZE, factor=QWEN2_FACTOR):
        W, H = img_size
        if grid_h is None or grid_w is None:
            resized_h, resized_w = smart_resize(H, W, factor=factor)
            grid_h = resized_h // patch_size
            grid_w = resized_w // patch_size
        else:
            resized_h, resized_w = grid_h * patch_size, grid_w * patch_size
        x1, y1, x2, y2 = box
        EPS = 1e-3
        scale_x = resized_w / W
        scale_y = resized_h / H
        col1 = int(x1 * scale_x / patch_size)
        col2 = int(((x2 - EPS) * scale_x) / patch_size) if x2 > x1 else col1
        row1 = int(y1 * scale_y / patch_size)
        row2 = int(((y2 - EPS) * scale_y / patch_size) if y2 > y1 else row1)
        col1 = max(0, min(col1, grid_w - 1))
        col2 = max(col1, min(col2, grid_w - 1))
        row1 = max(0, min(row1, grid_h - 1))
        row2 = max(row1, min(row2, grid_h - 1))
        coords = [(row, col) for row in range(row1, row2 + 1) for col in range(col1, col2 + 1)]
        return coords, grid_h, grid_w

    @staticmethod
    def patch_coord_to_qwen_sequence_index(row, col, grid_w, merge=2):
        merged_grid_w = grid_w // merge
        block_id = (row // merge) * merged_grid_w + (col // merge)
        within_block = (row % merge) * merge + (col % merge)
        return block_id * (merge ** 2) + within_block

    def bbox_to_merged_token_indices_qwen2(self, box, img_size, n_merged, grid_h=None, grid_w=None):
        patch_coords, grid_h, grid_w = self.bbox_to_patch_coords_qwen2(
            box, img_size, grid_h=grid_h, grid_w=grid_w
        )
        merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        assert grid_h % merge == 0 and grid_w % merge == 0, (
            f"grid {(grid_h, grid_w)} not divisible by merge={merge}"
        )
        merged_grid_w = grid_w // merge
        merged_idx = sorted({
            (row // merge) * merged_grid_w + (col // merge)
            for row, col in patch_coords
            if ((row // merge) * merged_grid_w + (col // merge)) < n_merged
        })
        patch_seq_idx = [
            self.patch_coord_to_qwen_sequence_index(row, col, grid_w, merge=merge)
            for row, col in patch_coords
        ]
        return merged_idx, patch_seq_idx, grid_h, grid_w

    def merged_token_centers(self, idx, grid_h, grid_w):
        merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
        merged_grid_w = max(1, grid_w // merge)
        rows = [idx_i // merged_grid_w for idx_i in idx]
        cols = [idx_i % merged_grid_w for idx_i in idx]
        return sum(rows) / len(rows), sum(cols) / len(cols)

    def side_check(self, axis, direction, idx_subj, idx_obj, grid_h, grid_w):
        row_s, col_s = self.merged_token_centers(idx_subj, grid_h, grid_w)
        row_o, col_o = self.merged_token_centers(idx_obj, grid_h, grid_w)
        if axis == "horizontal":
            return ((direction == "left") == (col_s < col_o))
        if axis == "vertical":
            return ((direction == "above") == (row_s < row_o))
        raise ValueError(axis)

    @staticmethod
    def labels_for_axis_truth(axis, direction):
        first, second = cc.LABELS_BY_AXIS[axis]
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

        for correct_item in cc.progress(partition["golden"], desc="embeddings"):
            axis = correct_item["axis"]
            img_id = correct_item["image_id"]
            direction = correct_item["truth"]
            subj_q = cc.strip_article(correct_item["subj"])
            obj_q = cc.strip_article(correct_item["obj"])
            img_path = self.image_path_for_id(img_id)
            box_subj, img_size, src_subj = get_bbox(img_id, subj_q, img_path)
            box_obj, _, src_obj = get_bbox(img_id, obj_q, img_path)
            d = probe_data[axis]
            d["bbox_sources"].append((src_subj, src_obj))
            if box_subj is None or box_obj is None:
                d["det_failures"] += 1
                continue

            inputs = self.build_dummy_inputs(img_path)
            grid_t_actual, grid_h_actual, grid_w_actual = [
                int(x) for x in inputs["image_grid_thw"][0].tolist()
            ]
            h = self.extract_merger_output(inputs)
            n_merged, D = h.shape
            expected_merged = grid_t_actual * grid_h_actual * grid_w_actual // QWEN2_MERGE_FACTOR
            assert n_merged == expected_merged, (
                f'n_merged={n_merged}, expected={expected_merged}, '
                f'grid={inputs["image_grid_thw"][0].tolist()}'
            )
            idx_subj, raw_idx_subj, grid_h, grid_w = self.bbox_to_merged_token_indices_qwen2(
                box_subj, img_size, n_merged, grid_h=grid_h_actual, grid_w=grid_w_actual
            )
            idx_obj, raw_idx_obj, _, _ = self.bbox_to_merged_token_indices_qwen2(
                box_obj, img_size, n_merged, grid_h=grid_h_actual, grid_w=grid_w_actual
            )
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
                "raw_idx_subj": raw_idx_subj,
                "raw_idx_obj": raw_idx_obj,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "n_merged": n_merged,
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
                f'{axis}: X rows={X_axis.shape[0]} expected {2 * len(meta_axis)}'
            )
            assert y_axis.shape[0] == X_axis.shape[0], (
                f"{axis}: y rows={y_axis.shape[0]} X rows={X_axis.shape[0]}"
            )
            assert X_axis.ndim == 2 and X_axis.shape[1] > 0, (
                f"{axis}: unexpected X shape: {X_axis.shape}"
            )
            assert set(y_axis.tolist()).issubset({0, 1}), (
                f"{axis}: unexpected labels: {sorted(set(y_axis.tolist()))}"
            )
            assert len(set(y_axis.tolist())) == 2, f"{axis}: probe labels contain only one class"
            for i, m in enumerate(meta_axis):
                assert y_axis[2 * i] == m["label_subj"] and y_axis[2 * i + 1] == m["label_obj"], (
                    f"{axis}: label/meta mismatch at item {i}"
                )
                assert m["idx_subj"] and m["idx_obj"], f"{axis}: empty token ids in meta item {i}"
                assert max(m["idx_subj"] + m["idx_obj"]) < m["n_merged"], (
                    f"{axis}: token id out of range in meta item {i}"
                )
            d["X"] = X_axis
            d["y"] = y_axis
            src_summary = Counter()
            for s_sub, s_obj in d["bbox_sources"]:
                src_summary[s_sub] += 1
                src_summary[s_obj] += 1
            print("\n" + "=" * 60)
            print(f"EMBEDDING EXTRACTION DONE - {axis.upper()}")
            print("=" * 60)
            print(f'  Correct items input:       {sum(1 for g in partition["golden"] if g["axis"] == axis)}')
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
