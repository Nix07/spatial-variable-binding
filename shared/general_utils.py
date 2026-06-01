import csv
import logging
import re
import sys
import numpy as np
import random
import subprocess
import pickle
import torch
import os
from collections import defaultdict
from PIL import Image
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.append("./third_party/TransformerLens")
try:
    import transformer_lens as lens
except:
    import transformer_lens as lens
from .vision_language_prompts import VLPrompt, vlp_collate_fn
from .component import Component


def translate_to_english(text):
    try:
        if len(text.strip()) == 0:
            return text
        from googletrans import Translator

        translator = Translator()
        translated = translator.translate(text, dest="en")
        return translated.text
    except Exception:
        logging.error(f"Error translating text: {text}")
        return "FAILURE"


def topk_2d(tensor, k):
    """
    Gets the H, W indices of the topk values in a 2D PyTorch tensor.

    Args:
      tensor: A 2D PyTorch tensor.
      k: The number of top values to retrieve.

    Returns:
      A tuple of two tensors: (topk_h_indices, topk_w_indices).
      Each tensor contains the indices of the topk values along the height and width dimensions, respectively.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError("Input tensor must be a 2D PyTorch tensor.")
    if k > tensor.numel():
        raise ValueError(
            "k cannot be greater than the number of elements in the tensor."
        )

    topk_values, topk_indices = torch.topk(tensor.flatten(), k)

    h_indices = topk_indices // tensor.shape[1]
    w_indices = topk_indices % tensor.shape[1]

    return (h_indices, w_indices), topk_values


def set_deterministic(seed=1337):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cuda_device(device_idx):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)


def get_image_size_for_model(model_name):
    if "pixtral" in model_name:
        return (252, 252)
    elif "qwen" in model_name:
        return (252, 252)
    elif "llava" in model_name:
        return (252, 252)
    elif "gemma" in model_name:
        return (252, 252)
    elif "llama3.2" in model_name:
        return None  # No need to resize
    else:
        raise ValueError(f"Unknown model name {model_name}")


def get_gpu_count():
    """
    Runs the nvidia-smi command and parses the output to determine the number of GPUs.

    Returns:
        int: The number of GPUs detected on the system.
    """
    try:
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], universal_newlines=True
        )

        # Split the output into individual lines
        lines = output.strip().split("\n")

        # Count the number of GPU lines
        gpu_count = len(lines)

        return gpu_count
    except (subprocess.CalledProcessError, ValueError):
        # If there's an error running the command or parsing the output, return 0
        return 0


def get_single_token_tokens(processor, token_list):
    """
    Get a list of tokens from token_list that are tokenized to one token only.
    """
    return [
        t
        for t in token_list
        if processor(text=t, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].numel()
        == 1
    ]


def get_content_key_for_prompt_dict(model_name):
    """
    Get the key for the prompt content, used in apply_chat_template.
    """
    if "pixtral" in model_name.lower():
        return "content"
    else:
        return "text"


def get_tokens(model_or_processor, prompt):
    """
    Return the tokens of the prompt, in string form.
    e.g. "What is the capital of France?" -> ["What", "is", "the", "capital", "of", "France", "?"].
    """
    if isinstance(model_or_processor, lens.HookedVLTransformer):
        return [
            model_or_processor.to_string(t)
            for t in model_or_processor.to_tokens(prompt.prompt, prompt.images).view(-1)
        ]
    else:
        return [
            model_or_processor.tokenizer.decode(t)
            for t in model_or_processor(prompt.prompt, prompt.images).input_ids.view(-1)
        ]


def balanced_answers_train_test_split(
    prompts,
    possible_answers,
    target_total_prompt_count=250,
    train_test_split_ratio=0.75,
    seed=42,
):
    set_deterministic(seed)

    answers = [p.answer for p in prompts]
    answer_counts = {k: 0 for k in answers if k in possible_answers}
    for a in answers:
        if a in possible_answers:
            answer_counts[a] += 1

    #  Get only some of the prompts, such that the answer distribution in the subset is as balanced as possible
    prompts_per_answer = 1 + target_total_prompt_count // len(possible_answers)
    subset = []
    set_deterministic(seed)
    for a in answer_counts.keys():
        answer_subset = [p for p in prompts if p.answer == a]
        random.shuffle(answer_subset)
        subset += answer_subset[:prompts_per_answer]

    # In case any answers did not have enough prompts, add some more in other answers
    if len(subset) < target_total_prompt_count:
        unchosen_prompts = sorted(
            list(set(prompts) - set(subset))
        )  # sorting is super important to turn off hash-set randomization
        random.shuffle(unchosen_prompts)
        subset += unchosen_prompts[: target_total_prompt_count - len(subset)]
    elif len(subset) > target_total_prompt_count:
        random.shuffle(subset)
        subset = subset[:target_total_prompt_count]

    assert len(subset) == target_total_prompt_count

    # Split to train (circuit discovery) and test (faithfulness eval), such
    # that each set has a balanced distribution of answers
    train_prompts, test_prompts = [], []
    for a in answer_counts.keys():
        answer_subset = [p for p in subset if p.answer == a]
        split_size = int(len(answer_subset) * train_test_split_ratio)
        train_prompts += answer_subset[:split_size]
        test_prompts += answer_subset[split_size:]

    # The train split might be too small due to per-answer-split rounding down
    # so move a few prompts from test to train
    while len(train_prompts) < int(train_test_split_ratio * target_total_prompt_count):
        train_prompts.append(test_prompts.pop())

    return train_prompts, test_prompts


def setup_random_counterfactual_prompts(vl_prompts, task_name, seed=42):
    set_deterministic(seed)
    for vl_prompt in vl_prompts:
        different_answer_prompts = [
            vlp for vlp in vl_prompts if vlp.answer != vl_prompt.answer
        ]

        if task_name == "factual_recall":
            # Factual recall is a special case - there are several different question templates.
            # For discovery purposes, the CF for each prompt must be from the same template.
            # The template is found by looking for the text which appears between "What"/"Which" and "...single word."
            vl_prompt_question = re.search(
                r"((?:What|Which).*single word\.)", vl_prompt.prompt
            ).groups()[0]
            prompts_to_choose_from = [
                vlp
                for vlp in different_answer_prompts
                if vl_prompt_question in vlp.prompt
            ]
            if len(prompts_to_choose_from) == 0:
                # Don't filter by template if there are no prompts with the same template
                prompts_to_choose_from = different_answer_prompts
        else:
            prompts_to_choose_from = different_answer_prompts

        chosen_cf = random.choice(prompts_to_choose_from)
        vl_prompt.cf_prompt = chosen_cf.prompt
        vl_prompt.cf_images = chosen_cf.images
        vl_prompt.cf_answer = chosen_cf.answer
    return vl_prompts


def predict_vl_answer_for_hf_model(model, processor, vl_prompts: List[VLPrompt]):
    """
    Wrapper for model forward pass to predict answer tokens for a list of prompts.
    """
    images = [p.images for p in vl_prompts]
    texts = [p.prompt for p in vl_prompts]
    inputs = processor(
        images=images, text=texts, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    output = model.forward(**inputs)
    return processor.decode(output.logits[:, -1].argmax(dim=-1))


def truncate_to_n_tokens(text, processor, n_tokens, truncation_str=" ..."):
    tokens = processor(text=text).input_ids[0]
    truncation_str_length = len(
        processor(text=truncation_str, prepend_bos=False).input_ids[0]
    )
    return processor.decode(tokens[: n_tokens - truncation_str_length]) + truncation_str


def to_single_token(model, text):
    """
    Get the first token of the string, in string form.
    """
    return model.to_string(model.to_tokens(text, prepend_bos=False).view(-1)[0])


def generate_random_strings(
    model, num_tokens, count=1, batch_size=1, initial_token=None
):
    """
    Generate a random string of tokens from the model.
    """
    result_strings = []
    for idx in range(0, count, batch_size):
        real_bs = min(count - idx, batch_size)
        if initial_token is None:
            initial_token = model.to_tokens("")
        else:
            initial_token = model.to_tokens(initial_token, prepend_bos=False)
        tokens = model.generate(
            initial_token.repeat(real_bs, 1),
            num_tokens - 1,
            prepend_bos=False,
            temperature=1.0,
        )  # -1 because BOS is already included
        result_strings += model.to_string(tokens[:, 1:])  # skip BOS
    return result_strings


def reduce_dimensionality(vectors, type="tsne"):
    if type == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=0)
        tsne_vectors = tsne.fit_transform(vectors.detach().numpy())
        return tsne_vectors[:, 0], tsne_vectors[:, 1]
    elif type == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=0)
        pca_vectors = pca.fit_transform(vectors.detach().numpy())
        return pca_vectors[:, 0], pca_vectors[:, 1]
    elif type == "umap":
        import umap

        reducer = umap.UMAP()
        umap_vectors = reducer.fit_transform(vectors.detach().numpy())
        return umap_vectors[:, 0], umap_vectors[:, 1]
    else:
        raise NotImplementedError


def generate_activations(
    model: lens.HookedVLTransformer,
    prompts: List[VLPrompt],
    components: List[Component],
    batch_size: int = 32,
    reduce_mean: bool = False,
    text_seq_len: int = None,
    device="cpu",
    verbose=True,
):
    """
    Generate activations for a list of given components in a model, by passing a list of prompts through the model.
    """
    activations = []
    for comp in components:
        act_dim = (
            text_seq_len
            if "pattern" in comp.valid_hook_name()
            else get_hook_dim(model, comp.hook_name)
        )
        comp_acts_shape = [act_dim]
        if comp.pos is None:
            assert (
                text_seq_len is not None
            ), "Text sequence length must be provided if pos is None"
            comp_acts_shape.insert(0, text_seq_len)
        if not reduce_mean:
            comp_acts_shape.insert(0, len(prompts))

        assert comp.neuron_indices is None, "Neuron indices are not supported"
        activations.append(torch.zeros(comp_acts_shape, device=device))

    # Run batched forward passes through the model, saving the activations of the requested components for each prompt (or sum across prompts in case of mean reduction)
    dataloader = torch.utils.data.DataLoader(
        prompts, batch_size=batch_size, shuffle=False, collate_fn=vlp_collate_fn
    )
    dataloader = (
        tqdm(dataloader, desc="Generating activations") if verbose else dataloader
    )
    for idx, batch in enumerate(dataloader):
        _, cache = model.run_with_cache(list(batch["prompt"]), batch["images"])
        for j, component in enumerate(components):
            pos = component.pos or slice(0, text_seq_len)
            if component.head_idx is None:
                # Component is an MLP
                if reduce_mean:
                    # Sum up the previous sum with the current activations
                    activations[j] += (
                        cache[component.valid_hook_name()][:, pos, :]
                        .sum(dim=0)
                        .to(activations[j].device)
                    )
                else:
                    # Save all activations as is
                    activations[j][idx * batch_size : (idx + 1) * batch_size] = cache[
                        component.valid_hook_name()
                    ][:, pos, :]
            else:
                # Component is an attention head
                if "pattern" in component.valid_hook_name():
                    # Pattern component has a different shape
                    act = cache[component.valid_hook_name()][
                        :, component.head_idx, pos, :
                    ]
                else:
                    act = cache[component.valid_hook_name()][
                        :, pos, component.head_idx, :
                    ]

                if reduce_mean:
                    activations[j] += act.sum(dim=0).to(activations[j].device)
                else:
                    activations[j][idx * batch_size : (idx + 1) * batch_size] = act

        # Make sure to avoid GPU memory overflow
        del cache
        torch.cuda.empty_cache()

    if reduce_mean:
        # Divide by the number of prompts to get the mean activation
        for k in range(len(activations)):
            activations[k] = activations[k] / len(prompts)

    return activations


def get_text_seq_len_and_image_seq_len(model, prompts, images):
    """
    Get the sequence lengths of the text and image inputs for the model.
    Assumes the inputs are all of the same size.

    Args:
        model (lens.HookedVLTransformer): The model to get the sequence lengths for.
    """
    # Get the sequence lengths
    images_exist = (
        images is not None and len(images[0]) > 0
    )  # Only check the first prompt because all prompts should have the same number of images
    if images_exist:
        tokens, pixel_values = model.to_tokens(
            prompts, images, return_also_image_values=True
        )[:2]
        text_seq_len = tokens.shape[1]
        if model.vl_strategy == "cross":
            # Llama-3.2 keeps pixel values in 4 optional tiles. If the image is large enough it is split
            # between tiles in some aspect ratio (2x2, 2x1, 4x1, etc). All tiles are always used, even
            # if they only contain padding. The number of tiles controls the patch count.
            n_tiles, img_h = (
                pixel_values.shape[2],
                pixel_values.shape[4],
            )
            per_tile_patch_count = (
                img_h // model.vision_model.patch_embedding.kernel_size[0]
            ) ** 2
            image_seq_len = n_tiles * (
                per_tile_patch_count + 1
            )  # +1 because of class token
        elif model.vl_strategy == "concat":
            # When concatting the image embeddings into the textual sequence, we treat the image tokens
            # as normal textual tokens, so the individual image sequence length is 0.
            image_seq_len = 0
        else:
            raise ValueError()
    else:
        text_seq_len = model.to_tokens(prompts, images).shape[1]
        image_seq_len = 0

    return text_seq_len, image_seq_len


def get_hook_dim(model, hook_name):
    """
    Get the size of the output of a specific hook in the model's calculation.
    """
    return {
        lens.utils.get_act_name("hook_embed", 0): model.cfg.d_model,
        lens.utils.get_act_name("v_input", 0): model.cfg.d_model,
        lens.utils.get_act_name("k_input", 0): model.cfg.d_model,
        lens.utils.get_act_name("q_input", 0): model.cfg.d_model,
        lens.utils.get_act_name("z", 0): model.cfg.d_head,
        lens.utils.get_act_name("result", 0): model.cfg.d_model,
        lens.utils.get_act_name("attn_out", 0): model.cfg.d_model,
        lens.utils.get_act_name("mlp_post", 0): model.cfg.d_mlp,
        lens.utils.get_act_name("mlp_in", 0): model.cfg.d_model,
        lens.utils.get_act_name("mlp_out", 0): model.cfg.d_model,
        lens.utils.get_act_name("resid_pre", 0): model.cfg.d_model,
        lens.utils.get_act_name("resid_post", 0): model.cfg.d_model,
    }[lens.utils.get_act_name(hook_name, 0)]


def load_prompts_from_csv(
    data_csv_path,
    model_name,
    target_size: Tuple[int, int] = None,
    correct_preds_only: bool = True,
):
    # Load the metadata from the CSV file to a list of VLPrompt objects.
    data_csv_dir = os.path.dirname(data_csv_path)

    vl_prompts = []

    # The CSV file contains in each list a list of prompt entries.
    # Each entry contains an image path (relative to the CSV), if an image is
    # used in the prompt; The prompt; and the GT and predicted top-1 answers.
    with open(data_csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header if present
        data_tuples = [tuple(row) for row in reader]

    for data_tuple in data_tuples:
        if len(data_tuple) == 4:
            relative_img_path, prompt, gt_answer, pred_answer = data_tuple
            full_img_path = os.path.join(data_csv_dir, relative_img_path)
            images = [
                load_image_for_model(full_img_path, model_name, target_size=target_size)
            ]
            metadata = {"image_path": full_img_path}
        else:
            prompt, gt_answer, pred_answer = data_tuple
            images = []
            metadata = None

        if not correct_preds_only or (gt_answer == pred_answer):
            vl_prompt = VLPrompt(prompt, images, gt_answer, metadata=metadata)
            vl_prompts.append(vl_prompt)

    return vl_prompts


def load_image_for_model(image_path, model_name, target_size=None):
    if "qwen" in model_name.lower():
        return load_qwen_image(image_path, target_size)
    elif "llama" in model_name.lower():
        image = Image.open(image_path).convert("RGB")
        # NO NEED TO RESIZE IN LLAMA BECAUSE ALL 4 TILES WITH PADDING ARE PASSED TO THE MODEL NEVERTHELESS
        return image
    elif (
        "pixtral" in model_name.lower()
        or "llava" in model_name.lower()
        or "gemma" in model_name.lower()
    ):
        image = Image.open(image_path).convert("RGB")
        if target_size is not None:
            image = image.resize(target_size)
        return image
    else:
        raise NotImplementedError


def load_qwen_image(image_path, target_size=None) -> Image.Image:
    """
    Copied from qwen_vl_utils.py
    """
    from qwen_vl_utils import smart_resize

    IMAGE_FACTOR = 28
    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 16384 * 28 * 28
    image = Image.open(image_path).convert("RGB")
    if target_size is not None:
        image = image.resize(target_size)
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))
    return image


def safe_eval(prompt):
    """
    Wrapper for eval function to avoid throwing exceptions where dividing by zero.
    """
    try:
        return int(eval(prompt))
    except ZeroDivisionError:
        return torch.nan


def get_topk_indices_3d(tensor, k):
    """
    Get the 3-D indices of the top k values in a 3-D tensor.

    Args:
        tensor: Input tensor of shape (A, B, C)
        k: Number of top values to retrieve

    Returns:
        values: Top k values
        indices: Tuple of (dim0_indices, dim1_indices, dim2_indices) for top k values
    """
    # Get tensor shape
    A, B, C = tensor.shape

    # Get top k values and flat indices
    values, flat_indices = torch.topk(tensor.flatten(), k)

    # Calculate indices for each dimension
    dim2_indices = flat_indices % C
    temp = flat_indices // C
    dim1_indices = temp % B
    dim0_indices = temp // B
    return torch.stack([dim0_indices, dim1_indices, dim2_indices]).T


def get_top_scoring_components(
    model: lens.HookedVLTransformer,
    scores: Dict[Component, torch.Tensor],
    n_heads: int,
    n_mlp_neurons: int,
    sorted_heads: Optional[List[Component]] = None,
    sorted_mlp_neuron_indices: Optional[List[Tuple[int, int, int]]] = None,
):
    """
    Get the attention heads and MLP neurons with the highest scores.
    The output lists include positional information (i.e. each component is defined for a specific position).
    If sorted_heads and sorted_mlp_neuron_indices are not provided, they are calculated and returned for future use.

    Args:
        model: The model to analyze
        scores: The scores dict, where the keys are "Component" objects and the values are score tensors.
            For MLPs, the scores are shaped (seq_len, d_mlp), and for heads they are shaped (seq_len, head_idx, d_head).
        n_heads: The number of top attention heads to return.
        n_mlp_neurons: The number of top MLP neurons to return.
        sorted_heads: A list of pre-sorted (based on score) attention heads to use. If None, it is calculated. Used
            to avoid re-sorting the heads when analyzing multiple percentages.
        sorted_mlp_neuron_indices: A list of pre-sorted (based on score) MLP neurons to use.
            Each entry is a 3-tuple representing the layer, position, and neuron index of a specific neuron.
            If None, it is calculated. Used to avoid re-sorting the neurons when analyzing multiple percentages.

    Returns:
        top_heads: A list of the top attention heads, each represented as a "Component" object.
        top_mlp_neurons: A list of the top MLP neurons, each represented as a "Component" object.
        sorted_heads: A list of all attention heads sorted by score, each represented as a "Component" object.
        sorted_mlp_neuron_indices: A list of all MLP neurons sorted by score, each represented as a 3-tuple
    """
    seq_len = scores["blocks.0.attn.hook_z"].shape[0]

    # Find top n_heads attention heads
    if sorted_heads is None:
        head_scores = {
            Component("z", layer=l, head=h, position=p): scores[
                f"blocks.{l}.attn.hook_z"
            ][p, h, :]
            for l in range(model.cfg.n_layers)
            for h in range(model.cfg.n_heads)
            for p in range(seq_len)
        }
        head_scores = {
            k: v.sum(dim=-1) for (k, v) in head_scores.items()
        }  # Sum across d_head
        sorted_heads = [
            kv[0]
            for kv in sorted(head_scores.items(), key=lambda kv: kv[1], reverse=True)
        ]
    top_heads = sorted_heads[:n_heads]

    # Find the MLPs that contain the top n_mlp_neurons neurons
    if sorted_mlp_neuron_indices is None:
        mlp_neuron_scores = torch.stack(
            [scores[f"blocks.{l}.mlp.hook_post"] for l in range(model.cfg.n_layers)],
            dim=0,
        )  # n_layers, pos, d_mlp
        sorted_mlp_neuron_indices = [
            tuple(t)
            for t in get_topk_indices_3d(
                mlp_neuron_scores, mlp_neuron_scores.numel()
            ).tolist()
        ]  # n_mlp_neurons, 3

    top_mlp_individual_neurons = sorted_mlp_neuron_indices[:n_mlp_neurons]

    # Group the neurons by layer and position
    top_mlp_neurons = group_neurons_to_components(top_mlp_individual_neurons)

    return top_heads, top_mlp_neurons, sorted_heads, sorted_mlp_neuron_indices


def group_neurons_to_components(lpn_list):
    """
    Groups a list of (layer, pos, neuron_index) tuples into a list of MLP components,
    to be used in faithfulness analysis etc.

    Args:
        lpn_list: List / Set of (layer, pos, neuron_index) tuples.
    Returns:
        List of MLP components.
    """
    mlp_comps = []

    # Group the neurons by layer and position
    groups = defaultdict(set)
    for l, p, n_index in lpn_list:
        groups[(l, p)].add(n_index)

    # Create MLP components for each layer and position
    for layer, pos in groups.keys():
        mlp_comps.append(
            Component(
                "mlp_post",
                layer=layer,
                position=pos,
                neurons=groups[(layer, pos)],
            )
        )
    return mlp_comps


#### Memory leak debugging util functions ####
def enable_memory_snapshot():
    # keep a maximum 100,000 alloc/free events from before the snapshot
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100_000)


def gpu_memory_snapshot(output_file):
    snapshot = torch.cuda.memory._snapshot()
    with open(output_file, "wb") as f:
        pickle.dump(snapshot, f)


def monitor_out_of_memory():
    """
    Register a monitor to save a GPU memory snapshot right after an Out-Of-Memory error occurs.
    """
    enable_memory_snapshot()

    def oom_observer(device, alloc, device_alloc, device_free):
        gpu_memory_snapshot("oom_snapshot.pkl")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def monitor_memory(func):
    """
    Decorator wrapper to wrap a function / code block with a memory snapshot before and after it
    """

    def wrapper(*args, **kwargs):
        enable_memory_snapshot()
        func(*args, **kwargs)
        gpu_memory_snapshot("snapshot.pkl")

    return wrapper
