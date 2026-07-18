from types import MethodType
import logging
import torch
from .state import OWNER_ATTR

log = logging.getLogger(__name__)


def install_grounded_conditioning(engine, state):
    original = engine.get_learned_conditioning
    @torch.inference_mode()
    def grounded(self, prompts):
        from backend import memory_management
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        text_engine = self.text_processing_engine_qwen
        # Keep the image argument in this small wrapper so the compatibility
        # helper can restore the exact engine call after installing its hook.
        return _run_grounded(text_engine, prompts, state.grounding_images)
    setattr(grounded, OWNER_ATTR, state.token)
    state.original_get_learned_conditioning = original
    engine.get_learned_conditioning = MethodType(grounded, engine)


def _run_grounded(text_engine, prompts, images):
    """Call the text engine while scoping the image-token emphasis workaround.

    Some Forge Neo revisions add Qwen3-VL visual embeddings after creating the
    prompt-weight tensor. They then try to expand the shorter prompt tensor
    across the full visual-plus-text sequence and abort generation. Image
    embeddings have no prompt-weight syntax, so retaining unmodified embeddings
    is the safe fallback when that specific mismatch occurs.
    """
    emphasis = getattr(text_engine, "emphasis", None)
    original_after_transformers = getattr(emphasis, "after_transformers", None)
    if not callable(original_after_transformers):
        return text_engine(prompts, images=images)

    def after_transformers():
        try:
            return original_after_transformers()
        except RuntimeError:
            z = getattr(emphasis, "z", None)
            multipliers = getattr(emphasis, "multipliers", None)
            if not isinstance(z, torch.Tensor) or not isinstance(multipliers, torch.Tensor):
                raise
            if z.ndim != multipliers.ndim + 1 or z.shape[:-1] == multipliers.shape:
                raise
            log.warning(
                "[Forge Krea2Edit Ref] Forge's Qwen3-VL emphasis weights do not include "
                "image tokens; prompt emphasis is skipped for this grounded conditioning pass."
            )
            return None

    emphasis.after_transformers = after_transformers
    try:
        return text_engine(prompts, images=images)
    finally:
        emphasis.after_transformers = original_after_transformers
