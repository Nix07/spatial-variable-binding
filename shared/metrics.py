import torch


def logit_diff(
    logits: torch.Tensor, clean_labels: torch.Tensor, corrupt_labels: torch.Tensor
):
    assert len(logits.shape) == len(clean_labels.shape) == len(corrupt_labels.shape), (
        "All inputs should have the same number of dimensions. "
        f"Got logits: {logits.shape}, clean_labels: {clean_labels.shape}, corrupt_labels: {corrupt_labels.shape}"
    )
    return logits.gather(1, clean_labels) - logits.gather(1, corrupt_labels)

def logit_prob(
    logits: torch.Tensor, labels: torch.Tensor
):
    # logits has shape (b, vocab_size)
    # labels should have shape (b, n, 2): n = # tokens, each capitalized/lowercase
    probs = logits.softmax(-1).unsqueeze(1).expand(-1, labels.size(1), -1)
    return probs.gather(-1, labels).sum(dim=-1)
    
def indirect_effect(
    pre_patch_probs: torch.Tensor,
    post_patch_probs: torch.Tensor,
    clean_labels: torch.Tensor,
    corrupt_labels: torch.Tensor,
):
    """
    Measure indirect effect of a patch on probabilities, as described in Eq. 2 of "Understanding Arithmetic
    Reasoning in Language Models using Causal Mediation Analysis".

    Args:
        pre_patch_probs (torch.Tensor (batch, vocab_size)): The probabilities before patching.
        post_patch_probs (torch.Tensor (batch, vocab_size)): The probabilities after patching.
        clean_labels (torch.Tensor(batch, 1)): The labels of the clean answers.
        corrupt_labels (torch.Tensor(batch, 1)): The label of the corrupt answers.

    Returns:
        torch.Tensor((batch,), dtype=torch.float32): The indirect effects for each prompt in the batch. The IE is not limited in magnitude.
    """
    a = (
        post_patch_probs.gather(1, corrupt_labels)
        - pre_patch_probs.gather(1, corrupt_labels)
    ) / pre_patch_probs.gather(1, corrupt_labels)
    b = (
        pre_patch_probs.gather(1, clean_labels)
        - post_patch_probs.gather(1, clean_labels)
    ) / post_patch_probs.gather(1, clean_labels)
    return (a + b).squeeze(1) / 2


def kl_divergence(X, Y, min_eps=1e-10):
    """
    Assuming X and Y are of shape (batch, pos).
    Return vector shaped (batch).
    """
    if min_eps is not None:
        X = X.clamp(min=min_eps)
        Y = Y.clamp(min=min_eps)
    return torch.sum(X * torch.log((X / Y) + 1e-8), dim=-1)


def js_divergence(X, Y, min_eps=1e-10):
    """
    Jensen-Shannon divergence, a symmetric variant of KL divergence.

    Assuming X and Y are of shape (batch, pos).
    Return vector shaped (batch).
    """
    M = 0.5 * (X + Y)
    return 0.5 * kl_divergence(X, M, min_eps) + 0.5 * kl_divergence(Y, M, min_eps)
