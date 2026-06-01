"""Chat-templated prompt construction for image+text VLMs.

All paper-style prompts have the same shape: an optional system message
nudging single-token output, then a user message containing one image
placeholder followed by the task-specific text. Each HF processor
(`Qwen2VLProcessor`, `Gemma3Processor`, `PixtralProcessor`) renders that
conversation through its own chat template.
"""

from __future__ import annotations

# Default system text used by the paper's Squares/Shapes/Objects evals —
# discourages multi-token responses so the top-1 next-token compare works.
DEFAULT_SYSTEM_TEXT = "You are a helpful assistant. You respond in one token."


def chat_prompt(processor, user_text: str, *,
                system_text: str | None = DEFAULT_SYSTEM_TEXT) -> str:
    """Build a chat-templated prompt with one image and `user_text`.

    Args:
        processor: an HF processor exposing `apply_chat_template`.
        user_text: the text part of the user message (image is auto-prepended).
        system_text: optional system message; pass `None` to omit (the
            What'sUp floor-baseline does this to match the original eval).

    Returns:
        The rendered prompt string ready to feed into `model(prompt, [image], ...)`.
    """
    def render(conversation):
        return processor.apply_chat_template(conversation, add_generation_prompt=True)

    def user_msg(text):
        return {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}

    if not system_text:
        return render([user_msg(user_text)])

    # Plain string for system content — Pixtral's chat template can't
    # concatenate a list of content-dicts here, while Gemma3 and Qwen2-VL
    # accept either. Plain string is the cross-model common denominator.
    return render([{"role": "system", "content": system_text}, user_msg(user_text)])
