"""Compatibility adapter for Forge Neo's Qwen visual-attention API drift."""
from __future__ import annotations

import inspect
from typing import Any

from .state import OWNER_ATTR


def _uses_direct_attention_api(attention: Any) -> bool:
    """Return whether ``attention`` accepts Q/K/V rather than factory arguments."""
    try:
        signature = inspect.signature(attention)
    except (TypeError, ValueError):
        return False

    marker = object()
    try:
        signature.bind(marker, mask=False, small_input=True)
    except TypeError:
        try:
            signature.bind(marker, marker, marker, 1)
        except TypeError:
            return False
        return True
    return False


def install_qwen_attention_factory_compatibility(engine: Any, state: Any) -> None:
    """Adapt releases where Qwen expects an old attention-function factory.

    Older Qwen3-VL code calls ``attention_function(device, ...)`` and expects a
    callable back.  Some Forge Neo builds instead expose ``attention_function``
    directly as the Q/K/V implementation, which raises before image grounding
    can run.  Wrap only that direct API and restore it when this job ends.
    """
    text_engine = engine.text_processing_engine_qwen
    visual = getattr(getattr(text_engine, "text_encoder", None), "visual", None)
    forward = getattr(visual, "forward", None)
    globals_dict = getattr(forward, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return

    original = globals_dict.get("attention_function")
    if not callable(original) or not _uses_direct_attention_api(original):
        return

    def attention_factory(device, mask=False, small_input=False):
        return original

    setattr(attention_factory, OWNER_ATTR, state.token)
    globals_dict["attention_function"] = attention_factory
    state.attention_function_globals = globals_dict
    state.original_attention_function = original
