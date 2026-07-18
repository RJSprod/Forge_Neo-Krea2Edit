from __future__ import annotations
import threading, uuid, weakref
from dataclasses import dataclass, field
from typing import Any

PATCH_LOCK = threading.RLock()
ACTIVE_PATCH_BY_MODEL_ID: dict[int, "Krea2EditJobState"] = {}
OWNER_ATTR = "__krea2edit_owner_token__"

@dataclass
class Krea2EditJobState:
    enabled: bool
    raw_references: list[Any]
    reference_hashes: list[str]
    fit_mode: str
    grounding_px: int
    ref_boost: float
    ref_boost_a: float
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    grounding_images: list[Any] = field(default_factory=list)
    latent_cache: dict[tuple, Any] = field(default_factory=dict)
    original_get_learned_conditioning: Any = None
    original_diffusion_forward: Any = None
    engine_ref: Any = None
    diffusion_model_ref: Any = None
    installed: bool = False
    tensors_allocated: bool = False
    model_identity: tuple | None = None

def owned_by(method: Any, token: str) -> bool:
    return getattr(method, OWNER_ATTR, None) == token or getattr(getattr(method, "__func__", None), OWNER_ATTR, None) == token

def cleanup_state(state: Krea2EditJobState | None, processing: Any = None) -> None:
    if state is None:
        return
    with PATCH_LOCK:
        model = state.diffusion_model_ref() if state.diffusion_model_ref else None
        engine = state.engine_ref() if state.engine_ref else None
        if model is not None and owned_by(getattr(model, "forward", None), state.token):
            model.forward = state.original_diffusion_forward
        if engine is not None and owned_by(getattr(engine, "get_learned_conditioning", None), state.token):
            engine.get_learned_conditioning = state.original_get_learned_conditioning
        if model is not None:
            ACTIVE_PATCH_BY_MODEL_ID.pop(id(model), None)
        state.latent_cache.clear(); state.grounding_images.clear()
        allocated = state.tensors_allocated
        state.tensors_allocated = False; state.installed = False
        state.engine_ref = state.diffusion_model_ref = None
    if processing is not None and getattr(processing, "krea2edit_reference_state", None) is state:
        delattr(processing, "krea2edit_reference_state")
    if allocated:
        try:
            from backend import memory_management
            memory_management.soft_empty_cache()
        except Exception:
            pass

def cleanup_active_for_model(sd_model: Any) -> None:
    try: dm = sd_model.forge_objects.unet.model.diffusion_model
    except Exception: return
    with PATCH_LOCK: state = ACTIVE_PATCH_BY_MODEL_ID.get(id(dm))
    cleanup_state(state)
