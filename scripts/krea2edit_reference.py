"""Forge Neo standalone Krea 2 Identity Edit reference extension."""
import logging
import gradio as gr
from modules import scripts
from krea2edit_ref.attention_compat import install_qwen_attention_factory_compatibility
from krea2edit_ref.compatibility import require_krea_engine
from krea2edit_ref.grounding import install_grounded_conditioning
from krea2edit_ref.image_prep import extract_gallery, image_hash, grounding_tensor
from krea2edit_ref.patching import install_edit_forward
from krea2edit_ref.state import Krea2EditJobState, cleanup_active_for_model, cleanup_state

log = logging.getLogger(__name__)
PREFIX = "[Forge Krea2Edit Ref]"

def _gallery_append(gallery, image):
    gallery = list(gallery or [])
    if image is not None and len(gallery) < 2: gallery.append(image)
    return gallery

def _gallery_replace(gallery, image, selected):
    gallery = list(gallery or [])
    if image is not None and selected is not None and 0 <= selected < len(gallery): gallery[selected] = image
    return gallery

def _gallery_delete(gallery, selected):
    gallery = list(gallery or [])
    if selected is not None and 0 <= selected < len(gallery): del gallery[selected]
    return gallery

class Krea2EditReferenceScript(scripts.Script):
    sorting_priority = 530
    def title(self): return "Krea 2 Edit Reference (ImageStitch)"
    def show(self, is_img2img): return False if is_img2img else scripts.AlwaysVisible
    def ui(self, is_img2img):
        with gr.Accordion("Krea 2 Edit Reference (ImageStitch)", open=False):
            enabled = gr.Checkbox(False, label="Enable Krea 2 Edit Reference")
            gr.Markdown("One image: source/subject.  Two images: scene first, subject second.\n\nLoad the Identity Edit LoRA normally; this extension never manages LoRAs.")
            gallery = gr.Gallery(label="Reference Image(s)", type="pil", elem_id="krea2edit_gallery")
            upload = gr.Image(label="Upload/Paste image", type="pil")
            selected = gr.State(None)
            with gr.Row():
                append = gr.Button("Append Pasted Image"); replace = gr.Button("Replace Selected Image")
                delete = gr.Button("Delete Selected Image"); clear = gr.Button("Clear All References")
            fit_mode = gr.Radio(["fit", "crop (legacy)"], value="fit", label="Reference geometry")
            grounding_px = gr.Slider(0, 4096, 768, step=64, label="Grounding longest side")
            ref_boost = gr.Slider(.25, 4, 1, step=.05, label="Subject reference fidelity", info="Non-1.0 values allocate an attention mask and can slow FlashAttention.")
            ref_boost_a = gr.Slider(.25, 4, 1, step=.05, label="Scene reference fidelity")
            gallery.select(lambda e: e.index if isinstance(e.index, int) else None, None, selected)
            append.click(_gallery_append, [gallery, upload], gallery)
            replace.click(_gallery_replace, [gallery, upload, selected], gallery)
            delete.click(_gallery_delete, [gallery, selected], gallery)
            clear.click(lambda: [], None, gallery)
        self.infotext_fields = [(enabled, "Krea2 Edit Reference"), (fit_mode, "Krea2 Edit Fit"), (grounding_px, "Krea2 Edit Grounding PX"), (ref_boost, "Krea2 Edit Subject Fidelity"), (ref_boost_a, "Krea2 Edit Scene Fidelity")]
        return [enabled, gallery, fit_mode, grounding_px, ref_boost, ref_boost_a]
    def process(self, p, enabled, gallery, fit_mode, grounding_px, ref_boost, ref_boost_a):
        cleanup_active_for_model(getattr(p, "sd_model", None))
        if not enabled: return
        if getattr(p, "enable_hr", False): raise RuntimeError("Krea 2 Edit Reference does not support Hires. fix in this release. Generate directly at the desired output resolution.")
        if "img2img" in type(p).__name__.lower() or "inpaint" in type(p).__name__.lower(): raise RuntimeError("Krea 2 Edit Reference is supported in txt2img only.")
        refs = extract_gallery(gallery)
        if not refs: raise RuntimeError("Add one reference image.")
        state = Krea2EditJobState(True, refs, [image_hash(r) for r in refs], fit_mode, int(grounding_px), float(ref_boost), float(ref_boost_a))
        p.krea2edit_reference_state = state
        p.extra_generation_params.update({"Krea2 Edit Reference": True, "Krea2 Reference Count": len(refs), "Krea2 Reference Geometry": fit_mode, "Krea2 Grounding PX": int(grounding_px), "Krea2 Subject Fidelity": float(ref_boost)})
        if len(refs) == 2: p.extra_generation_params["Krea2 Scene Fidelity"] = float(ref_boost_a)
        log.info("%s enabled: %s reference(s), %s geometry", PREFIX, len(refs), fit_mode)
    def after_extra_networks_activate(self, p, *args, **kwargs):
        state = getattr(p, "krea2edit_reference_state", None)
        if state is None or state.installed: return
        try:
            engine = p.sd_model; dm = require_krea_engine(engine)
            state.model_identity = (id(engine), id(engine.forge_objects.unet), id(dm))
            state.grounding_images = [grounding_tensor(image, state.grounding_px) for image in state.raw_references]
            install_qwen_attention_factory_compatibility(engine, state)
            install_grounded_conditioning(engine, state); install_edit_forward(dm, engine, state); state.installed = True
            if hasattr(p, "clear_prompt_cache"): p.clear_prompt_cache()
            log.info("%s patches installed", PREFIX)
        except Exception:
            cleanup_state(state, p); raise
    def process_before_every_sampling(self, p, *args, **kwargs):
        state = getattr(p, "krea2edit_reference_state", None)
        if state is not None and not state.installed: raise RuntimeError("Krea2Edit patch was not installed.")
    def postprocess(self, p, processed, *args):
        cleanup_state(getattr(p, "krea2edit_reference_state", None), p)
