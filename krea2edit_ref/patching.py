from types import MethodType
import weakref
from .state import PATCH_LOCK, ACTIVE_PATCH_BY_MODEL_ID, OWNER_ATTR, owned_by
from .forward import _prepare_latents, krea2_edit_forward

def install_edit_forward(model, engine, state):
    with PATCH_LOCK:
        current = model.forward; existing = getattr(getattr(current, "__func__", current), OWNER_ATTR, None)
        if existing == state.token: return
        if existing is not None: raise RuntimeError("Krea2Edit Reference is already active for another generation.")
        # A method residing on the class is stock; an unmarked instance override is not safely composable.
        if "forward" in model.__dict__: raise RuntimeError("Krea2Edit Reference cannot replace an unknown diffusion forward patch.")
        if id(model) in ACTIVE_PATCH_BY_MODEL_ID: raise RuntimeError("Krea2Edit Reference is already active for another generation.")
        original = current
        def edit(self, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
            latents = _prepare_latents(engine, state, x)
            return krea2_edit_forward(self, x, timesteps, context, latents, transformer_options, state.ref_boost, state.ref_boost_a, attention_mask=attention_mask, **kwargs)
        setattr(edit, OWNER_ATTR, state.token)
        state.original_diffusion_forward = original; state.engine_ref = weakref.ref(engine); state.diffusion_model_ref = weakref.ref(model)
        model.forward = MethodType(edit, model); ACTIVE_PATCH_BY_MODEL_ID[id(model)] = state
