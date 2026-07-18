from types import MethodType
import logging
import weakref
from .state import PATCH_LOCK, ACTIVE_PATCH_BY_MODEL_ID, OWNER_ATTR, owned_by
from .forward import _prepare_latents, krea2_edit_forward

log = logging.getLogger(__name__)

def install_edit_forward(model, engine, state):
    with PATCH_LOCK:
        current = model.forward; existing = getattr(getattr(current, "__func__", current), OWNER_ATTR, None)
        if existing == state.token: return
        if existing is not None: raise RuntimeError("Krea2Edit Reference is already active for another generation.")
        # Forge Neo can install its own instance-level forward wrapper while
        # activating extra networks. Krea2Edit must replace that entry to add
        # reference tokens; preserve it exactly so cleanup restores Forge
        # (or another extension) after this generation.
        if "forward" in model.__dict__:
            log.info("[Forge Krea2Edit Ref] temporarily replacing Forge's instance diffusion forward wrapper.")
        if id(model) in ACTIVE_PATCH_BY_MODEL_ID: raise RuntimeError("Krea2Edit Reference is already active for another generation.")
        original = current
        def edit(self, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
            latents = _prepare_latents(engine, state, x)
            return krea2_edit_forward(self, x, timesteps, context, latents, transformer_options, state.ref_boost, state.ref_boost_a, attention_mask=attention_mask, **kwargs)
        setattr(edit, OWNER_ATTR, state.token)
        state.original_diffusion_forward = original; state.engine_ref = weakref.ref(engine); state.diffusion_model_ref = weakref.ref(model)
        model.forward = MethodType(edit, model); ACTIVE_PATCH_BY_MODEL_ID[id(model)] = state
