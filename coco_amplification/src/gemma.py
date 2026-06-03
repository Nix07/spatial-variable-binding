"""Raw-HF Gemma3 helpers for the COCO amplification experiment.

The Gemma COCO numbers were generated with raw Hugging Face model hooks, not
the repo's TransformerLens wrapper. This module keeps that execution path for
the reproducibility-sensitive baseline, probe extraction, and amplification.
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
from collections import Counter
from getpass import getpass
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from shared.prompts import chat_prompt

from . import common as cc
from .prompts import direct_prompt_text


MODEL_IDS = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

GEMMA3_GRID = 16
IMAGE_SOFT_TOKEN = "<image_soft_token>"


def canonical_model_name(name: str) -> str:
    key = name.lower()
    aliases = {
        "gemma3": "gemma-3-4b-it",
        "gemma": "gemma-3-4b-it",
        "gemma-3-4b": "gemma-3-4b-it",
        "gemma-3-4b-it": "gemma-3-4b-it",
        "google/gemma-3-4b-it": "gemma-3-4b-it",
    }
    if key not in aliases:
        raise ValueError(f"unsupported COCO Gemma model {name!r}")
    return aliases[key]


def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token
    except Exception:
        pass
    if sys.stdin.isatty():
        token = getpass("Hugging Face token for google/gemma-3-4b-it: ").strip()
        return token or None
    return None


def _auth_error_message(model_id: str) -> str:
    return (
        f"Could not load gated model {model_id}. Make sure your Hugging Face "
        "account has access, then enter a token when prompted, run "
        "`huggingface-cli login`, or set `HF_TOKEN` before launching the experiment."
    )


class GemmaCocoRunner:
    def __init__(self, *, model_name: str, dtype_name: str, img_dir: str,
                 cache_dir: str | None = None):
        self.model_slug = canonical_model_name(model_name)
        self.model_id = MODEL_IDS[self.model_slug]
        self.dtype_name = dtype_name
        self.torch_dtype = cc.torch_dtype_from_name(dtype_name)
        self.img_dir = img_dir
        token = _hf_token()

        print(f"Loading {self.model_id} ...")
        try:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                token=token,
                device_map={"": 0} if torch.cuda.is_available() else "cpu",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=token,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"{_auth_error_message(self.model_id)} Original error: {exc}") from exc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        lm = self.model.language_model
        for attr in ["embed_tokens", "layers", "norm", "rotary_emb", "rotary_emb_local"]:
            if hasattr(lm, "model") and hasattr(lm.model, attr):
                setattr(lm, attr, getattr(lm.model, attr))
        self.model.vision_model = self.model.vision_tower

        self.image_soft_token_id = self.processor.tokenizer.convert_tokens_to_ids(IMAGE_SOFT_TOKEN)
        assert (
            self.image_soft_token_id is not None
            and self.image_soft_token_id != self.processor.tokenizer.unk_token_id
        ), f"tokenizer does not know {IMAGE_SOFT_TOKEN}"
        assert self.image_soft_token_id == 262144, (
            f"image_soft_token id changed: {self.image_soft_token_id}"
        )

        self.model_device = next(self.model.parameters()).device
        self.lm_layers = self.model.language_model.layers
        cfg = self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
        print(f"d_model = {cfg.hidden_size} | n_layers = {cfg.num_hidden_layers}")
        print(f"MODEL_DTYPE = {dtype_name} | actual parameter dtype = {next(self.model.parameters()).dtype}")

    def image_path_for_id(self, image_id) -> str:
        return os.path.join(self.img_dir, f"{int(image_id):012d}.jpg")

    def build_direct_prompt(self, subj, obj, listing, axis):
        text = direct_prompt_text(subj, obj, listing, axis=axis)
        return chat_prompt(self.processor, text, system_text=None)

    def build_gemma_inputs(self, img_path, subj, obj, listing, axis):
        img = PILImage.open(img_path).convert("RGB")
        prompt = self.build_direct_prompt(subj, obj, listing, axis)
        return self.processor(text=prompt, images=img, return_tensors="pt").to(self.model_device)

    def clear_layer0_hooks(self):
        self.lm_layers[0]._forward_pre_hooks.clear()
        self.lm_layers[0]._forward_hooks.clear()

    @torch.inference_mode()
    def predict_with_amp(self, item, listing, W=None, alpha=0.0):
        axis = item["axis"]
        img_path = self.image_path_for_id(item["image_id"])
        inputs = self.build_gemma_inputs(img_path, item["subj"], item["obj"], listing, axis)
        input_ids = inputs["input_ids"][0]
        img_pos = (input_ids == self.image_soft_token_id).nonzero().flatten().tolist()
        assert len(img_pos) == 256, f"expected 256 image tokens, got {len(img_pos)}"

        hidden_dim = self.lm_layers[0].self_attn.q_proj.in_features
        if W is not None:
            assert W.shape == (hidden_dim, 2), (
                f"expected W shape ({hidden_dim},2), got {tuple(W.shape)}"
            )

        handle = None
        if W is not None and alpha != 0:
            W_amp = W.to(dtype=self.torch_dtype, device=self.model_device)

            def pre_hook(module, args):
                hidden_states = args[0]
                assert hidden_states.shape[-1] == hidden_dim
                if hidden_states.shape[1] == 1:
                    return args
                valid = [p for p in img_pos if p < hidden_states.shape[1]]
                if not valid:
                    return args
                img_embeds = hidden_states[:, valid, :]
                coeff = torch.einsum("btd,dk->btk", img_embeds, W_amp) / (
                    (W_amp ** 2).sum(dim=0) + 1e-8
                )
                projected = torch.einsum("btk,dk->btd", coeff, W_amp)
                hidden_states[:, valid, :] = hidden_states[:, valid, :] + alpha * projected
                return (hidden_states,) + args[1:]

            handle = self.lm_layers[0].register_forward_pre_hook(pre_hook)

        try:
            out = self.model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out[0]
            pred_id = torch.argmax(logits[0, -1]).item()
        finally:
            if handle is not None:
                handle.remove()
        return self.processor.tokenizer.decode([pred_id], skip_special_tokens=True).strip().lower()

    def build_dummy_inputs(self, img_path):
        img = PILImage.open(img_path).convert("RGB")
        prompt = chat_prompt(self.processor, "Describe the image.", system_text=None)
        return self.processor(text=prompt, images=img, return_tensors="pt").to(self.model_device)

    def extract_projector_output(self, inputs):
        proj_out = {}

        def hook_proj(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            proj_out["h"] = h.detach().clone()

        handle = self.model.multi_modal_projector.register_forward_hook(hook_proj)
        try:
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            handle.remove()
        h = proj_out["h"]
        assert h.shape == (1, 256, 2560), f"unexpected projector output shape: {h.shape}"
        return h

    @staticmethod
    def bbox_to_token_indices_gemma3(box, img_size, grid=GEMMA3_GRID):
        W, H = img_size
        x1, y1, x2, y2 = box
        eps = 1e-3
        col1 = int(x1 * grid / W)
        col2 = int((x2 - eps) * grid / W) if x2 > x1 else col1
        row1 = int(y1 * grid / H)
        row2 = int((y2 - eps) * grid / H) if y2 > y1 else row1
        col1 = max(0, min(col1, grid - 1))
        col2 = max(col1, min(col2, grid - 1))
        row1 = max(0, min(row1, grid - 1))
        row2 = max(row1, min(row2, grid - 1))
        return [row * grid + col for row in range(row1, row2 + 1) for col in range(col1, col2 + 1)]

    @staticmethod
    def token_center(idx):
        return idx // GEMMA3_GRID, idx % GEMMA3_GRID

    def side_check(self, axis, truth, idx_subj, idx_obj):
        rows_s = [self.token_center(i)[0] for i in idx_subj]
        cols_s = [self.token_center(i)[1] for i in idx_subj]
        rows_o = [self.token_center(i)[0] for i in idx_obj]
        cols_o = [self.token_center(i)[1] for i in idx_obj]
        if axis == "horizontal":
            return (sum(cols_s) / len(cols_s) < sum(cols_o) / len(cols_o)) if truth == "left" else (
                sum(cols_s) / len(cols_s) > sum(cols_o) / len(cols_o)
            )
        return (sum(rows_s) / len(rows_s) < sum(rows_o) / len(rows_o)) if truth == "above" else (
            sum(rows_s) / len(rows_s) > sum(rows_o) / len(rows_o)
        )

    @staticmethod
    def labels_for_truth(axis, truth):
        lo, hi = cc.LABELS_BY_AXIS[axis]
        if truth == lo:
            return 0, 1
        if truth == hi:
            return 1, 0
        raise ValueError((axis, truth))

    def build_probe_data(self, partition, get_bbox, artifact_dir: str | Path,
                         *, model_slug: str, experiment_slug: str,
                         artifact_suffix: str, axes=("horizontal", "vertical")):
        artifact_dir = Path(artifact_dir)
        probe_lists = {
            axis: {
                "X": [],
                "y": [],
                "meta": [],
                "bbox_sources": [],
                "det_failures": 0,
                "empty_failures": 0,
                "bbox_side_failures": 0,
            }
            for axis in axes
        }

        for correct_item in cc.progress(partition["golden"], desc="embeddings"):
            axis = correct_item["axis"]
            bucket = probe_lists[axis]
            img_id = correct_item["image_id"]
            truth = correct_item["truth"]
            subj_q = cc.strip_article(correct_item["subj"])
            obj_q = cc.strip_article(correct_item["obj"])
            img_path = self.image_path_for_id(img_id)
            box_subj, img_size, src_subj = get_bbox(img_id, subj_q, img_path)
            box_obj, _, src_obj = get_bbox(img_id, obj_q, img_path)
            bucket["bbox_sources"].append((src_subj, src_obj))
            if box_subj is None or box_obj is None:
                bucket["det_failures"] += 1
                continue
            idx_subj = self.bbox_to_token_indices_gemma3(box_subj, img_size)
            idx_obj = self.bbox_to_token_indices_gemma3(box_obj, img_size)
            if not idx_subj or not idx_obj:
                bucket["empty_failures"] += 1
                continue
            if not self.side_check(axis, truth, idx_subj, idx_obj):
                bucket["bbox_side_failures"] += 1
                continue

            inputs = self.build_dummy_inputs(img_path)
            h = self.extract_projector_output(inputs)
            label_subj, label_obj = self.labels_for_truth(axis, truth)
            bucket["X"].append(h[0, idx_subj, :].float().mean(dim=0).cpu().numpy())
            bucket["y"].append(label_subj)
            bucket["X"].append(h[0, idx_obj, :].float().mean(dim=0).cpu().numpy())
            bucket["y"].append(label_obj)
            bucket["meta"].append({
                "image_id": img_id,
                "axis": axis,
                "subj": subj_q,
                "obj": obj_q,
                "truth": truth,
                "idx_subj": idx_subj,
                "idx_obj": idx_obj,
                "label_subj": label_subj,
                "label_obj": label_obj,
            })

        artifact_dir.mkdir(parents=True, exist_ok=True)
        probe_data = {}
        for axis in axes:
            bucket = probe_lists[axis]
            X_axis = np.array(bucket["X"], dtype=np.float32)
            y_axis = np.array(bucket["y"], dtype=np.int64)
            probe_data[axis] = {"X": X_axis, "y": y_axis, "meta": bucket["meta"]}
            np.save(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_X_{artifact_suffix}.npy", X_axis)
            np.save(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_y_{artifact_suffix}.npy", y_axis)
            with open(artifact_dir / f"{model_slug}_{experiment_slug}_embeddings_{axis}_meta_{artifact_suffix}.pkl", "wb") as f:
                pickle.dump(bucket["meta"], f)
            src_summary = Counter()
            for src_subj, src_obj in bucket["bbox_sources"]:
                src_summary[src_subj] += 1
                src_summary[src_obj] += 1
            print("\n" + "=" * 60)
            print(f"EMBEDDING EXTRACTION DONE - {axis.upper()}")
            print("=" * 60)
            print(f'  Correct items input:       {sum(1 for r in partition["golden"] if r["axis"] == axis)}')
            print(f'  Detection failures:       {bucket["det_failures"]}')
            print(f'  Empty-token failures:     {bucket["empty_failures"]}')
            print(f'  Bbox-side failures:       {bucket["bbox_side_failures"]}')
            print(f'  Items in probe pool:      {len(bucket["meta"])}')
            print(f"  Embeddings (X):           {X_axis.shape}")
            print(f'  Labels (y):               {y_axis.shape}  class balance = {(y_axis.mean() if len(y_axis) else float("nan")):.3f}')
            print(f"  bbox source breakdown:    {dict(src_summary)}")
        return probe_data
