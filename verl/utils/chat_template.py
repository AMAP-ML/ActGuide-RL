# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from jinja2 import TemplateError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    try:
        return tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    except TemplateError as e:
        logger.warning(f"Chat template does not support system prompt: {e}")
        return []


def get_generation_template_kwargs(messages: list[dict]) -> dict:
    """Return the right generation kwargs for a chat template call.

    - Default case: append a fresh assistant generation prompt.
    - If the final message is already an assistant message, continue it instead.
      This is used for assistant-prefix guidance injection.
    """
    if messages and (messages[-1].get("role") or "").lower() == "assistant":
        return {"add_generation_prompt": False, "continue_final_message": True}
    return {"add_generation_prompt": True}


def apply_chat_template_for_generation(chat_formatter, messages: list[dict], **kwargs):
    """Apply chat template with assistant-prefill awareness.

    `chat_formatter` can be a tokenizer or processor exposing `apply_chat_template`.
    """
    generation_kwargs = get_generation_template_kwargs(messages)
    return chat_formatter.apply_chat_template(messages, **generation_kwargs, **kwargs)
