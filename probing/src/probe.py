"""Linear probe `\\hat y = Wh + b` — 3-class, CE loss, AdamW.

Hyperparams match section 5.2.1 and App A.4: lr=1e-3, weight_decay=1e-4,
200 epochs, full-batch updates. Returns the trained `(W, b)` and the test
prediction softmax probabilities both on the object-region tokens and on
every patch in the test images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ProbeFit:
    W: np.ndarray              # (D, n_classes)
    b: np.ndarray              # (n_classes,)
    test_pred: np.ndarray      # softmax probs on object tokens, (n_obj_tokens, n_classes)
    test_pred_all: np.ndarray  # softmax probs on every patch of every test
                               # image, (n_test_images, N_patches, n_classes)
    test_acc: float            # argmax accuracy on object-region tokens
    train_loss: float          # final training cross-entropy
    n_classes: int


def train(train_embeds: torch.Tensor,
          train_labels: torch.Tensor,
          test_embeds: torch.Tensor,
          test_labels: torch.Tensor,
          all_tokens_test_embeds: torch.Tensor,
          *,
          n_test_images: int,
          patches_per_image: int,
          n_classes: int = 3,
          epochs: int = 200,
          lr: float = 1e-3,
          weight_decay: float = 1e-4,
          device: str = "cuda",
          seed: int | None = 0) -> ProbeFit:
    """Train one 3-class linear probe.

    Args:
        train_embeds: (N_train, D)
        train_labels: (N_train,) int64 in {0,1,2}
        test_embeds: (N_test_obj, D) — object-region tokens only
        test_labels: (N_test_obj,)
        all_tokens_test_embeds: (n_test_images * patches_per_image, D), in
            row-major patch order per image, images concatenated.
        n_test_images / patches_per_image: used only to reshape the all-token
            predictions into a `(n_test_images, patches_per_image, 3)` array
            for plotting heatmaps later.

    Returns:
        ProbeFit (everything on CPU, numpy).
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    train_embeds = train_embeds.to(dev, dtype=torch.float32)
    test_embeds  = test_embeds.to(dev, dtype=torch.float32)
    all_tokens_test_embeds = all_tokens_test_embeds.to(dev, dtype=torch.float32)
    train_labels = train_labels.to(dev)
    test_labels = test_labels.to(dev)

    _, D = train_embeds.shape

    # Probe weight init: N(0, 1/sqrt(D)).
    W = (torch.randn(D, n_classes, device=dev) / (D ** 0.5)).requires_grad_(True)
    b = (torch.randn(n_classes, device=dev) / (D ** 0.5)).requires_grad_(True)

    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)
    last_loss = float("nan")
    for ep in range(epochs):
        opt.zero_grad()
        logits = train_embeds @ W + b
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  loss={last_loss:.4f}")

    with torch.no_grad():
        test_logits = test_embeds @ W + b
        test_pred = test_logits.softmax(dim=-1)
        test_acc = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()

        all_logits = all_tokens_test_embeds @ W + b
        all_pred = all_logits.softmax(dim=-1)

    return ProbeFit(
        W=W.detach().cpu().numpy(),
        b=b.detach().cpu().numpy(),
        test_pred=test_pred.cpu().numpy(),
        test_pred_all=all_pred.cpu().numpy().reshape(
            n_test_images, patches_per_image, n_classes,
        ),
        test_acc=float(test_acc),
        train_loss=float(last_loss),
        n_classes=n_classes,
    )
