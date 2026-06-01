"""Shared eval helpers: top-1 next-token scoring + result aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch


_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_token(s: str) -> str:
    """Normalize a token for string comparison (alphanumeric lowercase)."""
    return _NORM_RE.sub("", s.strip().lower())


def top1_token(model, processor, prompt: str, image) -> tuple[str, float]:
    """Forward pass + argmax over next-token logits. Returns (decoded, prob).

    Uses the ``HookedVLTransformer`` path (``model(prompt, [image])``)."""
    with torch.no_grad():
        logits = model(prompt, [image], return_type="logits")
    probs = torch.softmax(logits[:, -1], dim=-1)
    tok = int(torch.argmax(probs, dim=-1).item())
    return processor.tokenizer.decode([tok]), float(probs[0, tok].item())


@dataclass
class EvalResult:
    """Accumulator for per-direction tallies + per-item records."""
    model: str
    task: str
    per_direction_correct: dict[str, int] = field(default_factory=dict)
    per_direction_n: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, direction: str, ok: bool, **meta):
        self.per_direction_correct.setdefault(direction, 0)
        self.per_direction_n.setdefault(direction, 0)
        self.per_direction_n[direction] += 1
        if ok:
            self.per_direction_correct[direction] += 1
        self.records.append({"direction": direction, "correct": ok, **meta})

    @property
    def per_direction_acc(self) -> dict[str, float]:
        return {d: self.per_direction_correct[d] / self.per_direction_n[d]
                for d in self.per_direction_n if self.per_direction_n[d]}

    @property
    def overall_acc(self) -> float:
        c = sum(self.per_direction_correct.values())
        n = sum(self.per_direction_n.values())
        return c / n if n else 0.0

    @property
    def total_correct(self) -> int:
        return sum(self.per_direction_correct.values())

    @property
    def total_examples(self) -> int:
        return sum(self.per_direction_n.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "task": self.task,
            "overall_acc": self.overall_acc,
            "total_correct": self.total_correct,
            "total_examples": self.total_examples,
            "per_direction_acc": self.per_direction_acc,
            "per_direction_correct": dict(self.per_direction_correct),
            "per_direction_n": dict(self.per_direction_n),
            "records": self.records,
        }

    def summary_text(self) -> str:
        lines = [
            f"model: {self.model}",
            f"task:  {self.task}",
            "per-direction accuracy:",
        ]
        for d, n in self.per_direction_n.items():
            c = self.per_direction_correct[d]
            lines.append(f"  {d:<6}: {c}/{n} = {c/n:.4f}")
        lines.append(f"OVERALL: {self.total_correct}/{self.total_examples} "
                     f"= {self.overall_acc:.4f}")
        return "\n".join(lines) + "\n"
