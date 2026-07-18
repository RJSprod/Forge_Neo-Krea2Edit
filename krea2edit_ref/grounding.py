from types import MethodType
import torch
from .state import OWNER_ATTR

def install_grounded_conditioning(engine, state):
    original = engine.get_learned_conditioning
    @torch.inference_mode()
    def grounded(self, prompts):
        from backend import memory_management
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_qwen(prompts, images=state.grounding_images)
    setattr(grounded, OWNER_ATTR, state.token)
    state.original_get_learned_conditioning = original
    engine.get_learned_conditioning = MethodType(grounded, engine)
