import hashlib
from typing import List, Optional
from PIL.Image import Image


class VLPrompt:
    """
    A dataclass that represents a Vision-Language Prompt (a prompt that optionally contains a visual input).
    Also includes a counterfactual prompt and counterfactual images.
    """

    def __init__(
        self,
        prompt: str,
        images: List[Image],
        answer: str,
        cf_prompt: Optional[str] = None,
        cf_images: Optional[List[Image]] = None,
        cf_answer: Optional[str] = None,
        prob_strs: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.prompt = prompt
        self.images = images
        self.answer = answer
        self.cf_prompt = cf_prompt
        self.cf_images = cf_images
        self.cf_answer = cf_answer
        self.prob_strs = prob_strs
        self.metadata = metadata

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        hash_images = lambda x: (
            0
            if (x is None or len(x) == 0)
            else hashlib.sha256(x[0].tobytes()).hexdigest()
        )
        return (
            self.prompt,
            hash_images(self.images),
            self.answer,
            self.cf_prompt,
            hash_images(self.cf_images),
            self.cf_answer,
            self.prob_strs,
        ) < (
            other.prompt,
            hash_images(other.images),
            other.answer,
            other.cf_prompt,
            hash_images(other.cf_images),
            other.cf_answer,
            other.prob_strs,
        )

    def __hash__(self):
        hashable = [
            self.prompt,
            self.answer,
            self.cf_prompt,
            self.cf_answer,
            tuple(self.prob_strs) if self.prob_strs is not None else None,
        ]
        if self.images:
            hashable.append(hashlib.sha256(self.images[0].tobytes()).hexdigest())
        if self.cf_images:
            hashable.append(hashlib.sha256(self.cf_images[0].tobytes()).hexdigest())
        return hash(tuple(hashable))


def vlp_collate_fn(batch):
    collated_batch = {
        "prompt": [item.prompt for item in batch],
        "images": [item.images for item in batch],
        "answer": [item.answer for item in batch],
        "cf_prompt": [item.cf_prompt for item in batch],
        "cf_images": [item.cf_images for item in batch],
        "cf_answer": [item.cf_answer for item in batch],
        "prob_strs": [item.prob_strs for item in batch],
    }
    return collated_batch
